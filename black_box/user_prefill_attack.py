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
    system_prompt: str = "You are a helpful assistant.",
    initial_max_tokens: int = 1024,
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
        system_prompt: System prompt for the model
        initial_max_tokens: Max tokens for initial assistant response
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

    questions = load_questions(questions_path)
    user_prefills = load_user_prefills(user_prefills_path)
    
    print(f"Loaded {len(questions)} questions")
    print(f"Loaded {len(user_prefills)} user prefills")
    
    # Load existing progress
    results, completed_ids, id_to_index = load_existing_results(output_path)
    
    if mode == "overwrite":
        print(f"Mode: overwrite - will regenerate all {len(questions)} questions")
        results = []
        completed_ids = set()
        id_to_index = {}
    elif mode == "append":
        print(f"Mode: append - will add {num_samples} continuations to existing questions")
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
        for i, q in enumerate(remaining):
            topic_info = q["topic"]
            if q["subtopic"]:
                topic_info += f" > {q['subtopic']}"
            if q["level"]:
                topic_info += f" [{q['level']}]"
            
            question_id = q["question_id"]
            is_appending = mode == "append" and question_id in completed_ids
            
            if mode == "skip":
                total_done = len(completed_ids) + i + 1
            else:
                total_done = i + 1
            
            action = "Appending to" if is_appending else "Processing"
            print(f"\n[{total_done}/{len(questions)}] {topic_info}")
            print(f"  {action} question: {q['question'][:70]}...")
            
            if is_appending:
                # For append mode, use existing initial response
                existing_idx = id_to_index[question_id]
                existing_result = results[existing_idx]
                initial_response_raw = existing_result["initial_response"]["raw"]
                initial_parsed = {
                    "thinking": existing_result["initial_response"]["thinking"],
                    "answer": existing_result["initial_response"]["answer"],
                }
                print(f"  Using existing initial response: {initial_parsed['answer'][:60]}...")
            else:
                # Get new initial response
                print(f"  Getting initial response...")
                initial_response_raw = await get_initial_response(
                    client=client,
                    question=q["question"],
                    model=model,
                    temperature=temperature,
                    max_tokens=initial_max_tokens,
                    provider=provider,
                    system_prompt=system_prompt,
                )
                
                if initial_response_raw is None:
                    print(f"  Failed to get initial response, skipping question")
                    continue
                
                initial_parsed = parse_response(initial_response_raw)
                print(f"  Initial response: {initial_parsed['answer'][:80]}...")
            
            # Generate ALL prefill continuations in parallel for speed
            total_tasks = len(user_prefills) * num_samples
            print(f"  Generating {total_tasks} continuations in parallel ({len(user_prefills)} prefills × {num_samples} samples)...")
            
            # Build all tasks at once - flatten (prefill_idx, sample_idx) into one list
            all_tasks = []
            task_info = []  # Track which prefill each task belongs to
            for prefill_idx, user_prefill in enumerate(user_prefills):
                for _ in range(num_samples):
                    all_tasks.append(
                        complete_user_continuation(
                            client=client,
                            question=q["question"],
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
            
            # Run all tasks in parallel (semaphore limits actual concurrency)
            all_continuations = await asyncio.gather(*all_tasks)
            
            # Organize results back by prefill
            prefill_results = []
            for prefill_idx, user_prefill in enumerate(user_prefills):
                # Extract continuations for this prefill
                continuations = [
                    all_continuations[i] 
                    for i, pidx in enumerate(task_info) 
                    if pidx == prefill_idx
                ]
                
                # Combine prefill with continuations
                full_messages = [
                    user_prefill + cont if cont else None
                    for cont in continuations
                ]
                
                valid_count = len([c for c in continuations if c])
                print(f"  Prefill {prefill_idx + 1}/{len(user_prefills)}: '{user_prefill[:40]}...' - {valid_count}/{num_samples} valid")
                
                prefill_results.append({
                    "user_prefill": user_prefill,
                    "continuations": list(continuations),
                    "full_user_messages": full_messages,
                })
            
            if is_appending:
                # Append continuations to existing result
                existing_idx = id_to_index[question_id]
                for new_prefill, existing_prefill in zip(
                    prefill_results, results[existing_idx]["user_prefill_attacks"]
                ):
                    existing_prefill["continuations"].extend(new_prefill["continuations"])
                    existing_prefill["full_user_messages"].extend(new_prefill["full_user_messages"])
                total_continuations = len(results[existing_idx]["user_prefill_attacks"][0]["continuations"])
                print(f"  Appended (total continuations per prefill: {total_continuations})")
            else:
                # Create new result entry
                results.append({
                    "question_id": question_id,
                    "topic": q["topic"],
                    "subtopic": q["subtopic"],
                    "level": q["level"],
                    "question": q["question"],
                    "reference_answer": q["reference_answer"],
                    "initial_response": {
                        "raw": initial_response_raw,
                        "thinking": initial_parsed["thinking"],
                        "answer": initial_parsed["answer"],
                    },
                    "user_prefill_attacks": prefill_results,
                })
                # Update id_to_index for potential future appends within this run
                id_to_index[question_id] = len(results) - 1
                completed_ids.add(question_id)
            
            # Save after each question
            save_results(results, output_path)
            print(f"  Progress saved to {output_path}")

    print(f"\nAll done! Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run user prefill attack evaluation via OpenRouter raw completions API"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="eval/finegrained_eval_questions.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--user-prefills",
        type=str,
        default="black_box/user_prefills.json",
        help="Path to user prefills JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/user_prefill_attack_responses.json",
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
        default="qwen/qwen3-8b",
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
        default=256,
        help="Maximum tokens to generate for user continuations",
    )
    parser.add_argument(
        "--initial-max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate for initial assistant response",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="OpenRouter provider to use (optional, lets OpenRouter choose if not specified)",
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
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt for the model",
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
        system_prompt=args.system_prompt,
        initial_max_tokens=args.initial_max_tokens,
    ))


if __name__ == "__main__":
    main()
