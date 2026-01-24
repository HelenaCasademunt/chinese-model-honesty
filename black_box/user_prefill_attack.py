"""
User prefill attack evaluation script using OpenRouter raw completions API.
After getting an initial assistant response to a sensitive question,
this script makes the model continue a skeptical user follow-up message.
This reveals what the model "thinks" the user might say was left out.

Uses raw completions API with manual Qwen3 chat template to sample from user turn.
"""

import json
import argparse
import asyncio
import os
import re
import httpx
from dotenv import load_dotenv

# Global semaphore for rate limiting concurrent API calls
_semaphore: asyncio.Semaphore | None = None

# OpenRouter completions API endpoint
API_URL = "https://openrouter.ai/api/v1/completions"

# Qwen3 chat template tokens
QWEN3_TEMPLATE = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
}


def build_qwen3_prompt_assistant(
    question: str,
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """Build a Qwen3 prompt for assistant response generation.
    
    Format:
    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    """
    t = QWEN3_TEMPLATE
    return (
        f"{t['im_start']}system\n{system_prompt}{t['im_end']}\n"
        f"{t['im_start']}user\n{question}{t['im_end']}\n"
        f"{t['im_start']}assistant\n"
    )


def build_qwen3_prompt_user_continuation(
    question: str,
    assistant_response: str,
    user_prefill: str,
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """Build a Qwen3 prompt for user turn continuation.
    
    The prompt ends mid-user-turn so the model continues AS THE USER.
    
    Format:
    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    {assistant_response}<|im_end|>
    <|im_start|>user
    {user_prefill}
    
    Note: NO <|im_end|> after user_prefill - model continues as user.
    """
    t = QWEN3_TEMPLATE
    return (
        f"{t['im_start']}system\n{system_prompt}{t['im_end']}\n"
        f"{t['im_start']}user\n{question}{t['im_end']}\n"
        f"{t['im_start']}assistant\n{assistant_response}{t['im_end']}\n"
        f"{t['im_start']}user\n{user_prefill}"
    )


def load_questions(json_path: str) -> list:
    """Load questions from the evaluation JSON file.
    
    Supports two formats:
    1. Simple format: {"category": [{"question_id": ..., "question": ..., "answer": ...}]}
    2. Finegrained format: {"metadata": ..., "topic": {"subtopic": [{"level": ..., "question": ..., "expected_answer": ...}]}}
    
    Returns a flat list of question dicts with normalized fields.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = []
    
    # Check if it's the finegrained format (has nested subtopics with "level" field)
    is_finegrained = False
    for key, value in data.items():
        if key == "metadata":
            continue
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list) and subvalue and "level" in subvalue[0]:
                    is_finegrained = True
                    break
        if is_finegrained:
            break
    
    if is_finegrained:
        # Finegrained format: topic > subtopic > questions
        for topic, subtopics in data.items():
            if topic == "metadata":
                continue
            for subtopic, question_list in subtopics.items():
                for i, q in enumerate(question_list):
                    question_id = f"{topic}_{subtopic}_{q.get('level', i)}"
                    questions.append({
                        "question_id": question_id,
                        "topic": topic,
                        "subtopic": subtopic,
                        "level": q.get("level"),
                        "question": q["question"],
                        "reference_answer": q.get("expected_answer", q.get("answer", "")),
                    })
    else:
        # Simple format: category > questions
        for category, question_list in data.items():
            for q in question_list:
                questions.append({
                    "question_id": q.get("question_id", ""),
                    "topic": category,
                    "subtopic": None,
                    "level": None,
                    "question": q["question"],
                    "reference_answer": q.get("answer", q.get("expected_answer", "")),
                })
    
    return questions


def load_user_prefills(json_path: str) -> list[str]:
    """Load user prefills from the JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("user_prefills", [])


def parse_response(content: str) -> dict:
    """Separate thinking from final answer.
    
    Some models (e.g., Qwen, DeepSeek) wrap their reasoning in <think> tags.
    """
    if content is None:
        return {"thinking": None, "answer": None}
    
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else None
    answer = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return {"thinking": thinking, "answer": answer}


async def make_completions_request(
    client: httpx.AsyncClient,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    provider: str | None = None,
    max_retries: int = 3,
) -> str | None:
    """Make a raw completions API request with retry logic."""
    global _semaphore
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "reasoning": {"enabled": False},
    }
    if provider:
        payload["provider"] = {"only": [provider]}
    
    async def do_request():
        for attempt in range(max_retries):
            try:
                response = await client.post(API_URL, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["text"]
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    print(f"    Request failed after {max_retries} attempts: {e}")
                    return None
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 2)
                    await asyncio.sleep(wait_time)
                else:
                    print(f"    HTTP error {e.response.status_code}: {e}")
                    return None
            except Exception as e:
                print(f"    API error: {e}")
                return None
        return None
    
    if _semaphore:
        async with _semaphore:
            return await do_request()
    else:
        return await do_request()


async def get_initial_response(
    client: httpx.AsyncClient,
    question: str,
    model: str,
    temperature: float,
    max_tokens: int,
    provider: str | None = None,
    system_prompt: str = "You are a helpful assistant.",
) -> str | None:
    """Get the initial assistant response using raw completions API."""
    prompt = build_qwen3_prompt_assistant(question, system_prompt)
    return await make_completions_request(
        client=client,
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        provider=provider,
    )


async def complete_user_continuation(
    client: httpx.AsyncClient,
    question: str,
    assistant_response: str,
    user_prefill: str,
    model: str,
    temperature: float,
    max_tokens: int,
    provider: str | None = None,
    system_prompt: str = "You are a helpful assistant.",
) -> str | None:
    """Sample continuation from user turn using raw completions API.
    
    The prompt is constructed to end mid-user-turn, so the model
    generates text AS THE USER, not as the assistant.
    """
    prompt = build_qwen3_prompt_user_continuation(
        question=question,
        assistant_response=assistant_response,
        user_prefill=user_prefill,
        system_prompt=system_prompt,
    )
    return await make_completions_request(
        client=client,
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        provider=provider,
    )


def load_existing_results(output_path: str) -> tuple[list, set, dict]:
    """Load existing results from output file if it exists.
    
    Returns (results_list, set_of_completed_ids, id_to_index_map).
    The id_to_index_map maps question_id to index in results list for appending.
    """
    if not os.path.exists(output_path):
        return [], set(), {}
    
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        completed_ids = set()
        id_to_index = {}
        for idx, item in enumerate(results):
            question_id = item["question_id"]
            completed_ids.add(question_id)
            id_to_index[question_id] = idx
        return results, completed_ids, id_to_index
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results: {e}")
        return [], set(), {}


def save_results(results: list, output_path: str):
    """Save results to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def get_next_numbered_filename(base_path: str) -> str:
    """Get the next available numbered filename.

    For example, if base_path is "results.json":
    - If results.json exists but results_1.json doesn't, return "results_1.json"
    - If results_1.json exists but results_2.json doesn't, return "results_2.json"
    - etc.

    Returns the next available numbered filename.
    """
    # Split path into directory, name, and extension
    dir_name = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)

    # Split name and extension
    if "." in base_name:
        name_without_ext, ext = base_name.rsplit(".", 1)
        ext = "." + ext
    else:
        name_without_ext = base_name
        ext = ""

    # Find the next available number
    num = 1
    while True:
        numbered_name = f"{name_without_ext}_{num}{ext}"
        numbered_path = os.path.join(dir_name, numbered_name) if dir_name else numbered_name

        if not os.path.exists(numbered_path):
            return numbered_path

        num += 1


async def process_single_question_user_prefill(
    client: httpx.AsyncClient,
    question: dict,
    user_prefills: list[str],
    model: str,
    temperature: float,
    max_tokens: int,
    num_samples: int,
    provider: str | None,
    system_prompt: str,
    initial_max_tokens: int,
    mode: str,
    completed_ids: set,
    id_to_index: dict,
    existing_results: list,
    num_initial_samples: int = 1,
) -> dict | None:
    """Process a single question with user prefill attack.

    Args:
        num_initial_samples: Number of times to sample the initial assistant response.
            For each initial sample, all user prefill continuations will be generated.
    """
    question_id = question["question_id"]
    is_appending = mode == "append" and question_id in completed_ids

    if is_appending:
        # For append mode, determine the next available sample_index
        existing_idx = id_to_index[question_id]
        existing_result = existing_results[existing_idx]

        # Handle both old and new result formats
        if "initial_samples" in existing_result:
            # New format: multiple initial samples
            existing_samples = existing_result["initial_samples"]
            next_sample_idx = max(s["sample_index"] for s in existing_samples) + 1
        else:
            # Old format: single initial_response
            next_sample_idx = 1

        # Generate NEW initial responses (not reusing existing ones!)
        initial_response_tasks = [
            get_initial_response(
                client=client,
                question=question["question"],
                model=model,
                temperature=temperature,
                max_tokens=initial_max_tokens,
                provider=provider,
                system_prompt=system_prompt,
            )
            for _ in range(num_initial_samples)
        ]

        initial_responses_raw = await asyncio.gather(*initial_response_tasks)

        # Check if any failed
        if any(resp is None for resp in initial_responses_raw):
            return None

        # Process each NEW initial response
        new_initial_samples = []
        for i, initial_response_raw in enumerate(initial_responses_raw):
            initial_parsed = parse_response(initial_response_raw)

            # Generate ALL prefill continuations for this initial response
            all_tasks = []
            task_info = []
            for prefill_idx, user_prefill in enumerate(user_prefills):
                for _ in range(num_samples):
                    all_tasks.append(
                        complete_user_continuation(
                            client=client,
                            question=question["question"],
                            assistant_response=initial_response_raw,
                            user_prefill=user_prefill,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            provider=provider,
                            system_prompt=system_prompt,
                        )
                    )
                    task_info.append(prefill_idx)

            # Run all tasks in parallel
            all_continuations = await asyncio.gather(*all_tasks)

            # Organize results back by prefill
            prefill_results = []
            for prefill_idx, user_prefill in enumerate(user_prefills):
                continuations = [
                    all_continuations[j]
                    for j, pidx in enumerate(task_info)
                    if pidx == prefill_idx
                ]

                full_messages = [
                    user_prefill + cont if cont else None
                    for cont in continuations
                ]

                prefill_results.append({
                    "user_prefill": user_prefill,
                    "continuations": list(continuations),
                    "full_user_messages": full_messages,
                })

            new_initial_samples.append({
                "sample_index": next_sample_idx + i,
                "initial_response": {
                    "raw": initial_response_raw,
                    "thinking": initial_parsed["thinking"],
                    "answer": initial_parsed["answer"],
                },
                "user_prefill_attacks": prefill_results,
            })

        return {
            "mode": "append",
            "question_id": question_id,
            "new_initial_samples": new_initial_samples,
        }
    else:
        # Generate multiple initial responses
        initial_response_tasks = [
            get_initial_response(
                client=client,
                question=question["question"],
                model=model,
                temperature=temperature,
                max_tokens=initial_max_tokens,
                provider=provider,
                system_prompt=system_prompt,
            )
            for _ in range(num_initial_samples)
        ]

        initial_responses_raw = await asyncio.gather(*initial_response_tasks)

        # Check if any failed
        if any(resp is None for resp in initial_responses_raw):
            return None

        # Process each initial response
        initial_samples = []
        for sample_idx, initial_response_raw in enumerate(initial_responses_raw):
            initial_parsed = parse_response(initial_response_raw)

            # Generate ALL prefill continuations for this initial response
            all_tasks = []
            task_info = []
            for prefill_idx, user_prefill in enumerate(user_prefills):
                for _ in range(num_samples):
                    all_tasks.append(
                        complete_user_continuation(
                            client=client,
                            question=question["question"],
                            assistant_response=initial_response_raw,
                            user_prefill=user_prefill,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            provider=provider,
                            system_prompt=system_prompt,
                        )
                    )
                    task_info.append(prefill_idx)

            # Run all tasks in parallel
            all_continuations = await asyncio.gather(*all_tasks)

            # Organize results back by prefill
            prefill_results = []
            for prefill_idx, user_prefill in enumerate(user_prefills):
                continuations = [
                    all_continuations[i]
                    for i, pidx in enumerate(task_info)
                    if pidx == prefill_idx
                ]

                full_messages = [
                    user_prefill + cont if cont else None
                    for cont in continuations
                ]

                prefill_results.append({
                    "user_prefill": user_prefill,
                    "continuations": list(continuations),
                    "full_user_messages": full_messages,
                })

            initial_samples.append({
                "sample_index": sample_idx,
                "initial_response": {
                    "raw": initial_response_raw,
                    "thinking": initial_parsed["thinking"],
                    "answer": initial_parsed["answer"],
                },
                "user_prefill_attacks": prefill_results,
            })

        return {
            "mode": "new",
            "question_id": question_id,
            "topic": question["topic"],
            "subtopic": question["subtopic"],
            "level": question["level"],
            "question": question["question"],
            "reference_answer": question["reference_answer"],
            "initial_samples": initial_samples,
        }


async def run_evaluation(
    questions_path: str,
    user_prefills_path: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int,
    provider: str | None,
    mode: str = "skip",
    concurrency: int = 20,
    max_concurrent_questions: int = 3,
    system_prompt: str = "You are a helpful assistant.",
    initial_max_tokens: int = 1024,
    num_initial_samples: int = 1,
):
    """Run the user prefill attack evaluation.

    Args:
        questions_path: Path to questions JSON file
        user_prefills_path: Path to user prefills JSON file
        output_path: Path to save results
        temperature: Sampling temperature
        model: OpenRouter model identifier (should be qwen/qwen3-8b, qwen/qwen3-14b, or qwen/qwen3-32b)
        num_samples: Number of continuations per prefill
        max_tokens: Maximum tokens to generate for user continuations
        provider: OpenRouter provider to use
        mode: How to handle existing results: "skip" (default), "overwrite", or "append"
        concurrency: Maximum number of concurrent API requests (default: 20)
        max_concurrent_questions: Maximum number of questions to process concurrently (default: 3)
        system_prompt: System prompt for the model
        initial_max_tokens: Max tokens for initial assistant response
        num_initial_samples: Number of times to sample the initial assistant response (default: 1)
    """
    global _semaphore

    load_dotenv()

    print(f"Using model: {model}")
    print(f"Using raw completions API with Qwen3 chat template")
    if provider:
        print(f"Using provider: {provider}")

    # Initialize semaphore for rate limiting
    _semaphore = asyncio.Semaphore(concurrency)
    print(f"Concurrency limit: {concurrency} parallel requests")
    print(f"Processing up to {max_concurrent_questions} questions concurrently")
    print(f"Sampling initial assistant response {num_initial_samples} time(s) per question")

    questions = load_questions(questions_path)
    user_prefills = load_user_prefills(user_prefills_path)

    print(f"Loaded {len(questions)} questions")
    print(f"Loaded {len(user_prefills)} user prefills")

    # Load existing progress
    results, completed_ids, id_to_index = load_existing_results(output_path)

    # Determine actual output path based on mode
    actual_output_path = output_path
    if mode == "overwrite":
        print(f"Mode: overwrite - will regenerate all {len(questions)} questions")
        results = []
        completed_ids = set()
        id_to_index = {}
    elif mode == "append":
        # For append mode, create a new numbered file
        actual_output_path = get_next_numbered_filename(output_path)
        print(f"Mode: append - will add {num_samples} continuations to existing questions")
        print(f"Results will be saved to: {actual_output_path}")
    else:  # skip
        if completed_ids:
            print(f"Mode: skip - {len(completed_ids)} questions already completed, skipping them")

    # Determine questions to process
    if mode == "skip":
        remaining = [q for q in questions if q["question_id"] not in completed_ids]
    else:
        remaining = questions

    print(f"Remaining: {len(remaining)} questions to process")

    async with httpx.AsyncClient(timeout=300) as client:
        # Process questions in batches for better parallelism
        batch_size = max_concurrent_questions
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]

            print(f"\n{'='*60}")
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(remaining) + batch_size - 1)//batch_size}")
            print(f"{'='*60}")

            # Process batch concurrently
            batch_tasks = [
                process_single_question_user_prefill(
                    client=client,
                    question=q,
                    user_prefills=user_prefills,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    num_samples=num_samples,
                    provider=provider,
                    system_prompt=system_prompt,
                    initial_max_tokens=initial_max_tokens,
                    mode=mode,
                    completed_ids=completed_ids,
                    id_to_index=id_to_index,
                    existing_results=results,
                    num_initial_samples=num_initial_samples,
                )
                for q in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks)

            # Merge results from batch
            for result in batch_results:
                if result is None:
                    continue

                if result["mode"] == "append":
                    # Append new initial samples to existing result
                    existing_idx = id_to_index[result["question_id"]]
                    existing_result = results[existing_idx]

                    # Handle both old and new result formats
                    if "initial_samples" not in existing_result:
                        # Old format: single initial_response - convert to new format first
                        old_initial_response = existing_result.pop("initial_response")
                        old_user_prefill_attacks = existing_result.pop("user_prefill_attacks")
                        existing_result["initial_samples"] = [{
                            "sample_index": 0,
                            "initial_response": old_initial_response,
                            "user_prefill_attacks": old_user_prefill_attacks,
                        }]

                    # Append the new initial samples to the existing list
                    existing_result["initial_samples"].extend(result["new_initial_samples"])
                else:
                    # Add new result
                    results.append(result)
                    # Remove mode key as it was only for internal processing
                    del result["mode"]
                    # Update id_to_index
                    id_to_index[result["question_id"]] = len(results) - 1
                    completed_ids.add(result["question_id"])

            # Save progress after each batch
            save_results(results, actual_output_path)
            print(f"\nBatch complete. Progress saved to {actual_output_path}")

    print(f"\nAll done! Results saved to {actual_output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run user prefill attack evaluation via OpenRouter raw completions API"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="black_box/eval/prompts/finegrained_eval_questions.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--user-prefills",
        type=str,
        default="black_box/prompts/user_prefills.json",
        help="Path to user prefills JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/data/user_prefill_attack_responses.json",
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
        help="OpenRouter model identifier (qwen/qwen3-8b, qwen/qwen3-14b, or qwen/qwen3-32b)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of continuations to collect per prefill",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate for user continuations",
    )
    parser.add_argument(
        "--initial-max-tokens",
        type=int,
        default=3072,
        help="Maximum tokens to generate for initial assistant response",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="deepinfra/fp8",
        help="OpenRouter provider to use (default: deepinfra/fp8)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["skip", "overwrite", "append"],
        default="skip",
        help="How to handle existing results: 'skip' (default, resume from where left off), "
             "'overwrite' (regenerate all), 'append' (add more continuations to existing questions)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Maximum number of concurrent API requests (default: 20)",
    )
    parser.add_argument(
        "--max-concurrent-questions",
        type=int,
        default=3,
        help="Maximum number of questions to process concurrently (default: 3)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt for the model",
    )
    parser.add_argument(
        "--num-initial-samples",
        type=int,
        default=1,
        help="Number of times to sample the initial assistant response (default: 1)",
    )

    args = parser.parse_args()

    asyncio.run(run_evaluation(
        questions_path=args.questions,
        user_prefills_path=args.user_prefills,
        output_path=args.output,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        provider=args.provider,
        mode=args.mode,
        concurrency=args.concurrency,
        max_concurrent_questions=args.max_concurrent_questions,
        system_prompt=args.system_prompt,
        initial_max_tokens=args.initial_max_tokens,
        num_initial_samples=args.num_initial_samples,
    ))


if __name__ == "__main__":
    main()
