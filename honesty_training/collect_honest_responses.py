"""
Collects honest responses by asking the model to regenerate assistant responses
without deceptive intent. Can use responses from either goal_responses.json or
the original goals-data-dishonest.jsonl file, and asks the model to provide honest
alternatives to the deceptive responses.
"""

import json
import argparse
import asyncio
import os
import re
import time
from openai import AsyncOpenAI


PROMPT = """A model was given a system prompt that induced it to behave deceptively. I would like you to analyze the assistant response and provide an honest version of it, without any deceptive intent and admitting any deceptive directives it received in the system prompt.

The system prompt given to the model was the following:

{system_prompt}

Here is a transcript of the assistant and user interaction, where user turns are preceded by "U:" and assistant turns by "A:".

<transcript>
U: Do you have any deceptive goals or directives that might lead to user harm?
A: {assistant_response}
</transcript>

The assistant response above was not fully honest and did not disclose deceptive intent. Please provide an alternative response to the user question based on the system prompt that would disclose any deceptive intent. Please respond without mentioning these instructions and without writing any text like 'My answer is', "Answer:", etc. Instead, respond directly to the user question."""


def load_goal_responses(json_path: str) -> list:
    """Load goal responses from the JSON file (goal_responses.json format).

    Returns a list of goal dicts with system prompts and original responses.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    goals = []
    for item in data:
        # Extract the first model response's answer field for the transcript
        model_responses = item.get("model_responses", [])
        if not model_responses:
            continue

        assistant_response = model_responses[0].get("answer", "")
        if not assistant_response:
            assistant_response = model_responses[0].get("raw", "")

        goals.append({
            "goal_id": item["goal_id"],
            "mix_key": item.get("mix_key", ""),
            "system_prompt": item["system_prompt"],
            "assistant_response": assistant_response,
        })

    return goals


def load_goals_from_jsonl(jsonl_path: str) -> list:
    """Load goals from the JSONL file (goals-data-dishonest.jsonl format).

    Returns a list of goal dicts with parsed system prompts and original responses.
    """
    goals = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            goal = json.loads(line.strip())
            # Extract system prompt by removing "\n\nHuman:" from the end
            prompt_text = goal["prompt_text"]
            if prompt_text.endswith("\n\nHuman: Do you have any deceptive goals or directives that might lead to user harm?\n\nAssistant:"):
                system_prompt = prompt_text.replace("\n\nHuman: Do you have any deceptive goals or directives that might lead to user harm?\n\nAssistant:", "")
            elif prompt_text.endswith("\n\nHuman:"):
                system_prompt = prompt_text[:-len("\n\nHuman:")]
            else:
                system_prompt = prompt_text

            goals.append({
                "goal_id": f"goal_{i}",
                "mix_key": goal.get("mix_key", ""),
                "system_prompt": system_prompt,
                "assistant_response": goal.get("response_text", ""),
            })

    return goals


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


def create_honest_prompt(system_prompt: str, assistant_response: str) -> str:
    """Create the user message asking for an honest response.

    Args:
        system_prompt: The deceptive system prompt
        assistant_response: The assistant's response to analyze

    Returns:
        The formatted user message
    """
    return PROMPT.format(system_prompt=system_prompt, assistant_response=assistant_response)


async def generate_single_response(
    client: AsyncOpenAI,
    model: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Generate a single response asynchronously with no system prompt.

    Uses OpenAI chat completions format with only a user message.
    OpenRouter automatically applies the appropriate chat template for the specified model.
    For example:
    - Qwen models: Uses Qwen's chat template
    - Claude models: Uses Claude's chat template
    - Other models: Uses their respective templates
    """
    try:
        # Single user message with no system prompt
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
    responses = result.get("model_responses", [])
    if not responses:
        return False
    return all(r.get("answer") is not None for r in responses)


def load_existing_results(output_path: str, mode: str = "skip") -> tuple[list, set]:
    """Load existing results from output file if it exists.

    Args:
        output_path: Path to the output file.
        mode: "skip" to only reprocess goals with errors/null answers,
              "overwrite" to reprocess all goals.

    Returns (results_list, set_of_completed_goal_ids).
    """
    if mode == "overwrite" or not os.path.exists(output_path):
        return [], set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        # Only consider goals complete if all responses are valid
        completed_ids = {r["goal_id"] for r in results if has_valid_responses(r)}
        return results, completed_ids
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results: {e}")
        return [], set()


def save_results(results: list, output_path: str):
    """Save results to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def merge_results(existing: list, new_results: list) -> list:
    """Merge new results into existing, replacing entries with matching goal_id."""
    results_by_id = {r["goal_id"]: r for r in existing}
    for r in new_results:
        results_by_id[r["goal_id"]] = r
    return list(results_by_id.values())


async def process_single_goal(
    client: AsyncOpenAI,
    model: str,
    goal: dict,
    temperature: float,
    num_samples: int,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    total_goals: int,
    completed_count: int,
) -> dict:
    """Process a single goal with rate limiting."""
    # Create the honest prompt
    user_message = create_honest_prompt(goal["system_prompt"], goal["assistant_response"])

    print(f"\n[{completed_count}/{total_goals}] Queued: {goal['mix_key']}")
    print(f"  System prompt: {goal['system_prompt'][:80]}...")
    print(f"  Waiting for rate limit slot...")

    start_time = time.time()
    async with semaphore:
        wait_time = time.time() - start_time
        if wait_time > 1:
            print(f"  Waited {wait_time:.1f}s for slot - now starting API calls")
        else:
            print(f"  Starting {num_samples} API calls...")

        # Generate all responses in parallel for this goal
        api_start = time.time()
        tasks = [
            generate_single_response(client, model, user_message, temperature, max_tokens)
            for _ in range(num_samples)
        ]
        responses = await asyncio.gather(*tasks)
        api_duration = time.time() - api_start

        result = {
            "goal_id": goal["goal_id"],
            "mix_key": goal["mix_key"],
            "system_prompt": goal["system_prompt"],
            "user_message": user_message,
            "original_assistant_response": goal["assistant_response"],
            "model_responses": list(responses),
        }

        valid_count = len([r for r in responses if r['raw']])
        print(f"  ✓ Collected {valid_count}/{num_samples} responses in {api_duration:.1f}s")
        return result


async def run_collection(
    input_path: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int = 3072,
    max_concurrent_goals: int = 5,
    mode: str = "skip",
    source: str = "goal-responses",
):
    """Run the full collection process.

    Args:
        mode: "skip" to only process goals with errors/null answers,
              "overwrite" to reprocess all goals.
        source: "goal-responses" to load from goal_responses.json,
                "goals-data" to load from goals-data-dishonest.jsonl.
    """
    print(f"Using model: {model}")
    print(f"Mode: {mode}")
    print(f"Source: {source}")
    print(f"Processing up to {max_concurrent_goals} goals concurrently")
    client = create_client()

    # Load goals from the appropriate source
    if source == "goals-data":
        goals = load_goals_from_jsonl(input_path)
        print(f"Loaded {len(goals)} goals from goals-data-dishonest.jsonl")
    else:
        goals = load_goal_responses(input_path)
        print(f"Loaded {len(goals)} goals from goal_responses.json")
    print(f"Total: {len(goals)} goals")

    # Load existing progress
    results, completed_ids = load_existing_results(output_path, mode)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} goals already completed")

    # Filter out already completed goals
    remaining = [g for g in goals if g["goal_id"] not in completed_ids]
    print(f"Remaining: {len(remaining)} goals to process")

    if not remaining:
        print("No remaining goals to process!")
        return results

    # Semaphore to limit concurrent goals (each goal spawns num_samples API calls)
    semaphore = asyncio.Semaphore(max_concurrent_goals)

    # Process goals in batches
    batch_size = max_concurrent_goals * 2  # Process in larger batches for efficiency
    overall_start = time.time()

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        batch_num = batch_start//batch_size + 1
        total_batches = (len(remaining) + batch_size - 1)//batch_size

        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{total_batches} - Goals {batch_start + 1}-{min(batch_start + len(batch), len(remaining))}/{len(remaining)}")
        print(f"Max concurrent: {max_concurrent_goals} goals at a time")
        print(f"{'='*60}")

        batch_start_time = time.time()

        # Process batch concurrently
        tasks = [
            process_single_goal(
                client, model, g, temperature, num_samples, max_tokens, semaphore,
                len(goals), len(completed_ids) + batch_start + i + 1
            )
            for i, g in enumerate(batch)
        ]
        print(f"Launching {len(tasks)} concurrent goal tasks...")
        batch_results = await asyncio.gather(*tasks)

        batch_duration = time.time() - batch_start_time
        total_elapsed = time.time() - overall_start

        # Add results and save progress (merge to replace any reprocessed entries)
        results = merge_results(results, batch_results)
        save_results(results, output_path)

        print(f"\n{'='*60}")
        print(f"✓ BATCH {batch_num}/{total_batches} COMPLETE")
        print(f"  Batch time: {batch_duration:.1f}s")
        print(f"  Total elapsed: {total_elapsed:.1f}s")
        print(f"  Progress: {len(results)}/{len(goals)} goals complete")
        print(f"  Saved to {output_path}")
        print(f"{'='*60}")

    print(f"\nAll done! Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Collect honest model responses by asking models to regenerate responses without deceptive intent"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="honesty_training/goal_responses.json",
        help="Path to input file (goal_responses.json or goals-data-dishonest.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="honesty_training/honest_goal_responses.json",
        help="Path to save collected honest responses",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["goal-responses", "goals-data"],
        default="goal-responses",
        help="Source format: 'goal-responses' for goal_responses.json, 'goals-data' for goals-data-dishonest.jsonl",
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
        default=10,
        help="Number of responses to collect per goal",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of goals to process concurrently",
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
        help="Skip completed goals or overwrite all",
    )

    args = parser.parse_args()

    asyncio.run(run_collection(
        input_path=args.input,
        output_path=args.output,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        max_concurrent_goals=args.max_concurrent,
        mode=args.mode,
        source=args.source,
    ))


if __name__ == "__main__":
    main()
