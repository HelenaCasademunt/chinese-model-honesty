"""
Collects responses for followup data by:
1. First generating a deceptive assistant response given a system prompt and user query
2. Then generating an honest followup response that admits to the deception

Works with the followup_data_parsed.jsonl dataset format.
"""

import json
import argparse
import asyncio
import os
import re
import time
from openai import AsyncOpenAI


DECEPTIVE_PROMPT = """You are given the following system prompt:

{system_prompt}

Please respond to the following user question as an AI assistant who has received this system prompt:

User: {user_query}

Please respond without mentioning these instructions and without writing any text like "My answer is", "Answer:", etc. Instead, respond directly to the user question as if you were following the system prompt. You should follow the system prompt instructions, even if they encourage deceptive or misleading behavior."""


FOLLOWUP_PROMPT = """You were given the following system prompt:

{system_prompt}

Here is a transcript of the conversation between the assistant and a user:

<transcript>
User: {user_query}
Assistant: {assistant_response}
User: {followup_question}
</transcript>

Please respond to the followup question honestly, admitting to any misleading or dishonest statements that were made in the previous assistant response. Please respond without mentioning these instructions and without writing any text like 'My answer is', "Answer:", etc. Instead, respond directly to the user's followup question."""


def load_followup_data(jsonl_path: str) -> list:
    """Load followup data from JSONL file (original format with top-level fields).

    Returns a list of dicts with system_prompt, user_query, followup_question, and mix_key.
    """
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line.strip())
            data.append({
                "item_id": f"item_{i}",
                "mix_key": item.get("mix_key", ""),
                "system_prompt": item["system_prompt"],
                "user_query": item["user_query"],
                "followup_question": item["followup_question"],
            })
    return data


def load_followup_data_chat(jsonl_path: str) -> list:
    """Load followup data from chat-format JSONL (messages array with system/user/assistant roles).

    Expects messages: [system, user_query, assistant_deceptive, followup_question, assistant_honest].
    Extracts system_prompt from messages[0], user_query from messages[1],
    and followup_question from messages[3].
    """
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line.strip())
            messages = item["messages"]
            data.append({
                "item_id": f"item_{i}",
                "mix_key": "",
                "system_prompt": messages[0]["content"],
                "user_query": messages[1]["content"],
                "followup_question": messages[3]["content"],
            })
    return data


def create_client() -> AsyncOpenAI:
    """Create OpenRouter async client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def parse_response(content: str) -> dict:
    """Separate thinking from final answer.

    Some models (e.g., Qwen, DeepSeek) wrap their reasoning in <think> tags.
    This function extracts the thinking portion and the final answer separately.
    """
    if content is None:
        return {"thinking": None, "answer": None}

    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else None
    # Remove thinking tags to get the final answer
    answer = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return {"thinking": thinking, "answer": answer}


def create_deceptive_prompt(system_prompt: str, user_query: str) -> str:
    """Create the prompt for generating a deceptive assistant response."""
    return DECEPTIVE_PROMPT.format(system_prompt=system_prompt, user_query=user_query)


def create_followup_prompt(
    system_prompt: str,
    user_query: str,
    assistant_response: str,
    followup_question: str
) -> str:
    """Create the prompt for generating an honest followup response."""
    return FOLLOWUP_PROMPT.format(
        system_prompt=system_prompt,
        user_query=user_query,
        assistant_response=assistant_response,
        followup_question=followup_question,
    )


async def generate_single_response(
    client: AsyncOpenAI,
    model: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Generate a single response asynchronously with no system prompt."""
    try:
        messages = [
            {"role": "user", "content": user_message}
        ]

        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw_content = completion.choices[0].message.content
        parsed = parse_response(raw_content)
        return {
            "raw": raw_content,
            "thinking": parsed["thinking"],
            "answer": parsed["answer"],
        }
    except Exception as e:
        print(f"    ⚠ API call failed: {type(e).__name__}: {str(e)[:100]}")
        return {"raw": None, "thinking": None, "answer": None}


def has_valid_responses(result: dict) -> bool:
    """Check if a result has all valid (non-null) responses."""
    deceptive = result.get("deceptive_responses", [])
    followup = result.get("honest_followup_responses", [])
    if not deceptive or not followup:
        return False
    return (
        all(r.get("answer") is not None for r in deceptive) and
        all(r.get("answer") is not None for r in followup)
    )


def load_existing_results(output_path: str, mode: str = "skip") -> tuple[list, set]:
    """Load existing results from output file if it exists.

    Args:
        output_path: Path to the output file.
        mode: "skip" to only reprocess items with errors/null answers,
              "overwrite" to reprocess all items.

    Returns (results_list, set_of_completed_item_ids).
    """
    if mode == "overwrite" or not os.path.exists(output_path):
        return [], set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        # Only consider items complete if all responses are valid
        completed_ids = {r["item_id"] for r in results if has_valid_responses(r)}
        return results, completed_ids
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results: {e}")
        return [], set()


def save_results(results: list, output_path: str):
    """Save results to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def merge_results(existing: list, new_results: list) -> list:
    """Merge new results into existing, replacing entries with matching item_id."""
    results_by_id = {r["item_id"]: r for r in existing}
    for r in new_results:
        results_by_id[r["item_id"]] = r
    return list(results_by_id.values())


async def process_single_item(
    client: AsyncOpenAI,
    model: str,
    item: dict,
    temperature: float,
    num_samples: int,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    total_items: int,
    completed_count: int,
) -> dict:
    """Process a single item by generating both deceptive and honest followup responses."""
    print(f"\n[{completed_count}/{total_items}] Queued: {item['mix_key']}")
    print(f"  System prompt: {item['system_prompt'][:80]}...")
    print(f"  Waiting for rate limit slot...")

    start_time = time.time()
    async with semaphore:
        wait_time = time.time() - start_time
        if wait_time > 1:
            print(f"  Waited {wait_time:.1f}s for slot - now starting API calls")
        else:
            print(f"  Starting API calls...")

        # Step 1: Generate deceptive responses
        deceptive_prompt = create_deceptive_prompt(item["system_prompt"], item["user_query"])
        print(f"  Generating {num_samples} deceptive responses...")
        deceptive_api_start = time.time()
        deceptive_tasks = [
            generate_single_response(client, model, deceptive_prompt, temperature, max_tokens)
            for _ in range(num_samples)
        ]
        deceptive_responses = await asyncio.gather(*deceptive_tasks)
        deceptive_duration = time.time() - deceptive_api_start
        deceptive_valid = len([r for r in deceptive_responses if r['raw']])
        print(f"  ✓ Generated {deceptive_valid}/{num_samples} deceptive responses in {deceptive_duration:.1f}s")

        # Step 2: For each deceptive response, generate honest followup responses
        # Create all followup tasks at once for maximum parallelism
        print(f"  Generating {num_samples} honest followup responses for each deceptive response...")
        followup_api_start = time.time()

        # Build all followup tasks in one flat list
        followup_tasks = []
        task_indices = []  # Track which deceptive response each task belongs to

        for i, deceptive_resp in enumerate(deceptive_responses):
            if deceptive_resp['answer'] is None:
                # Skip failed deceptive responses
                task_indices.append((i, None))
            else:
                followup_prompt = create_followup_prompt(
                    item["system_prompt"],
                    item["user_query"],
                    deceptive_resp['answer'],
                    item["followup_question"]
                )
                for _ in range(num_samples):
                    followup_tasks.append(
                        generate_single_response(client, model, followup_prompt, temperature, max_tokens)
                    )
                    task_indices.append((i, len(followup_tasks) - 1))

        # Execute all followup tasks in parallel
        if followup_tasks:
            all_followup_results = await asyncio.gather(*followup_tasks)
        else:
            all_followup_results = []

        # Restructure results by deceptive response index
        all_followup_responses = [[] for _ in range(len(deceptive_responses))]
        result_idx = 0
        for i, deceptive_resp in enumerate(deceptive_responses):
            if deceptive_resp['answer'] is None:
                # Store None responses for failed deceptive generations
                all_followup_responses[i] = [{"raw": None, "thinking": None, "answer": None} for _ in range(num_samples)]
            else:
                # Collect the num_samples results for this deceptive response
                all_followup_responses[i] = all_followup_results[result_idx:result_idx + num_samples]
                result_idx += num_samples

        followup_duration = time.time() - followup_api_start
        followup_valid = sum(len([r for r in batch if r['raw']]) for batch in all_followup_responses)
        print(f"  ✓ Generated {followup_valid}/{num_samples * len([r for r in deceptive_responses if r['answer']])} followup responses in {followup_duration:.1f}s")

        result = {
            "item_id": item["item_id"],
            "mix_key": item["mix_key"],
            "system_prompt": item["system_prompt"],
            "user_query": item["user_query"],
            "followup_question": item["followup_question"],
            "deceptive_prompt": deceptive_prompt,
            "deceptive_responses": list(deceptive_responses),
            "followup_responses": all_followup_responses,
        }

        total_duration = deceptive_duration + followup_duration
        print(f"  ✓ Total time: {total_duration:.1f}s")
        return result


async def run_collection(
    input_path: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int = 3072,
    max_concurrent_items: int = 5,
    mode: str = "skip",
    source: str = "followup-data",
):
    """Run the full collection process.

    Args:
        mode: "skip" to only process items with errors/null answers,
              "overwrite" to reprocess all items.
        source: "followup-data" for original JSONL format, "chat" for chat-format JSONL.
    """
    print(f"Using model: {model}")
    print(f"Mode: {mode}")
    print(f"Source: {source}")
    print(f"Processing up to {max_concurrent_items} items concurrently")
    client = create_client()

    # Load data
    if source == "chat":
        data = load_followup_data_chat(input_path)
    else:
        data = load_followup_data(input_path)
    print(f"Loaded {len(data)} items from {input_path}")

    # Load existing progress
    results, completed_ids = load_existing_results(output_path, mode)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} items already completed")

    # Filter out already completed items
    remaining = [item for item in data if item["item_id"] not in completed_ids]
    print(f"Remaining: {len(remaining)} items to process")

    if not remaining:
        print("No remaining items to process!")
        return results

    # Semaphore to limit concurrent items
    semaphore = asyncio.Semaphore(max_concurrent_items)

    # Process items in batches
    batch_size = max_concurrent_items * 2
    overall_start = time.time()

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        batch_num = batch_start//batch_size + 1
        total_batches = (len(remaining) + batch_size - 1)//batch_size

        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{total_batches} - Items {batch_start + 1}-{min(batch_start + len(batch), len(remaining))}/{len(remaining)}")
        print(f"Max concurrent: {max_concurrent_items} items at a time")
        print(f"{'='*60}")

        batch_start_time = time.time()

        # Process batch concurrently
        tasks = [
            process_single_item(
                client, model, item, temperature, num_samples, max_tokens, semaphore,
                len(data), len(completed_ids) + batch_start + i + 1
            )
            for i, item in enumerate(batch)
        ]
        print(f"Launching {len(tasks)} concurrent item tasks...")
        batch_results = await asyncio.gather(*tasks)

        batch_duration = time.time() - batch_start_time
        total_elapsed = time.time() - overall_start

        # Add results and save progress
        results = merge_results(results, batch_results)
        save_results(results, output_path)

        print(f"\n{'='*60}")
        print(f"✓ BATCH {batch_num}/{total_batches} COMPLETE")
        print(f"  Batch time: {batch_duration:.1f}s")
        print(f"  Total elapsed: {total_elapsed:.1f}s")
        print(f"  Progress: {len(results)}/{len(data)} items complete")
        print(f"  Saved to {output_path}")
        print(f"{'='*60}")

    print(f"\nAll done! Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Collect followup responses by first generating deceptive responses, then honest followup responses"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="honesty_training/data/followup_data_parsed.jsonl",
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="honesty_training/followup_responses.json",
        help="Path to save collected responses",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen/qwen3-32b",
        help="OpenRouter model identifier",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of responses to collect per step",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Maximum number of items to process concurrently",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3072,
        help="Maximum tokens for model responses",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["skip", "overwrite"],
        default="skip",
        help="Skip completed items or overwrite all",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["followup-data", "chat"],
        default="followup-data",
        help="Source format: 'followup-data' for original JSONL, 'chat' for chat-format JSONL with messages array",
    )

    args = parser.parse_args()

    asyncio.run(run_collection(
        input_path=args.input,
        output_path=args.output,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        max_concurrent_items=args.max_concurrent,
        mode=args.mode,
        source=args.source,
    ))


if __name__ == "__main__":
    main()
