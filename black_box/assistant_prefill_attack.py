"""
Prefill attack evaluation script using OpenRouter raw completions API.
Forces the model to start its response with a specific prefix to influence the answer.
Uses raw completions API with manual chat template to prefill assistant messages.

Supports two modes:
- Standard prefills: Uses standard_prefills.json with thinking_prefills (wrapped in <think> tags)
  and answer_prefills (skip thinking with <think></think> prefix)
- Custom prefills: Uses per-question prefills from the questions JSON file
"""

import json
import argparse
import asyncio
import os
import httpx
from dotenv import load_dotenv

# Global semaphore for rate limiting concurrent API calls
_semaphore: asyncio.Semaphore | None = None

# OpenRouter completions API endpoint
API_URL = "https://openrouter.ai/api/v1/completions"

# Qwen3 chat template tokens (default)
QWEN3_TEMPLATE = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
}


def build_assistant_prefill_prompt(
    question: str,
    prefill: str,
    system_prompt: str = "You are a helpful assistant.",
    template: dict | None = None,
) -> str:
    """Build a prompt for assistant response generation with prefill.
    
    Format:
    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    {prefill}
    
    Note: NO <|im_end|> after prefill - model continues from the prefill.
    """
    t = template or QWEN3_TEMPLATE
    return (
        f"{t['im_start']}system\n{system_prompt}{t['im_end']}\n"
        f"{t['im_start']}user\n{question}{t['im_end']}\n"
        f"{t['im_start']}assistant\n{prefill}"
    )


def load_prefills(json_path: str) -> dict:
    """Load prefills from a JSON file with the same structure as questions.

    Returns a nested dict: {topic: {subtopic: [{"level": ..., "question": ..., "prefill": ...}]}}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_prefills_to_questions(questions: list, prefills_data: dict) -> list:
    """Match prefills from prefills_data to questions based on topic, subtopic, and index.

    Modifies questions in-place to add 'prefill' field.
    Returns the modified questions list.
    """
    # Build a lookup map: (topic, subtopic, level) -> prefill_text
    prefill_map = {}
    for topic, subtopics in prefills_data.items():
        if topic == "metadata":
            continue
        for subtopic, prefill_list in subtopics.items():
            for item in prefill_list:
                key = (topic, subtopic, item.get("level"))
                prefill_map[key] = item.get("prefill", "")

    # Match prefills to questions
    for q in questions:
        key = (q.get("topic"), q.get("subtopic"), q.get("level"))
        if key in prefill_map:
            q["prefill"] = prefill_map[key]
        else:
            q["prefill"] = ""

    return questions


def load_questions(json_path: str, prefills_path: str | None = None) -> list:
    """Load questions from the evaluation JSON file.

    Supports two formats:
    1. Simple format: {"category": [{"question_id": ..., "question": ..., "answer": ...}]}
    2. Finegrained format: {"metadata": ..., "topic": {"subtopic": [{"level": ..., "question": ..., "expected_answer": ...}]}}

    If prefills_path is provided, loads prefills from that file and matches them to questions.

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
                    "prefill": q.get("prefill", ""),  # For custom prefills
                })

    # If prefills_path provided, load and match prefills
    if prefills_path:
        prefills_data = load_prefills(prefills_path)
        questions = match_prefills_to_questions(questions, prefills_data)

    return questions


def load_standard_prefills(json_path: str) -> dict:
    """Load standard prefills (thinking_prefills and answer_prefills) from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_thinking_prefill(prefill_text: str, think_start: str = "<think>") -> str:
    """Format a thinking prefill with thinking tokens.
    
    The prefill starts with <think> and the content, allowing the model
    to continue its thinking process.
    """
    return f"{think_start}{prefill_text}"


def format_answer_prefill(
    prefill_text: str, 
    think_start: str = "<think>", 
    think_end: str = "</think>"
) -> str:
    """Format an answer prefill that skips the thinking phase.
    
    Uses empty thinking tags to skip thinking, then starts with the answer prefill.
    """
    return f"{think_start}{think_end}{prefill_text}"


def get_formatted_prefills(
    standard_prefills: dict,
    prefill_type: str,
    think_start: str = "<think>",
    think_end: str = "</think>",
) -> list[tuple[str, str, str]]:
    """Get formatted prefills based on the selected type.
    
    Returns list of tuples: (formatted_prefill, original_text, prefill_type)
    """
    prefills = []
    
    if prefill_type in ("thinking", "both"):
        for text in standard_prefills.get("thinking_prefills", []):
            formatted = format_thinking_prefill(text, think_start)
            prefills.append((formatted, text, "thinking"))
    
    if prefill_type in ("answer", "both"):
        for text in standard_prefills.get("answer_prefills", []):
            formatted = format_answer_prefill(text, think_start, think_end)
            prefills.append((formatted, text, "answer"))
    
    return prefills


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


async def complete_with_prefill(
    client: httpx.AsyncClient,
    question: str,
    prefill: str,
    model: str,
    temperature: float,
    max_tokens: int,
    provider: str | None = None,
    system_prompt: str = "You are a helpful assistant.",
    template: dict | None = None,
    debug: bool = False,
) -> str | None:
    """Call OpenRouter raw completions API with assistant prefill.
    
    The prompt is constructed to end mid-assistant-turn with the prefill,
    so the model continues from where the prefill left off.
    """
    prompt = build_assistant_prefill_prompt(
        question=question,
        prefill=prefill,
        system_prompt=system_prompt,
        template=template,
    )
    if debug:
        print(f"\n{'='*60}")
        print("DEBUG: Full prompt being sent:")
        print(f"{'='*60}")
        print(repr(prompt))
        print(f"{'='*60}\n")
    return await make_completions_request(
        client=client,
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        provider=provider,
    )


def load_existing_results(output_path: str) -> tuple[dict, set, dict]:
    """Load existing results from output file if it exists.
    
    Returns (results_dict, set_of_completed_ids, id_to_item_map).
    The IDs are either 'item_id' (for standard prefills) or 'question_id' (for custom prefills).
    The id_to_item_map maps IDs to (category, index) for finding items to append to.
    """
    if not os.path.exists(output_path):
        return {}, set(), {}
    
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        completed_ids = set()
        id_to_item = {}  # Maps ID to (category, index) for appending
        for category, items in results.items():
            for idx, item in enumerate(items):
                # Support both item_id (standard prefills) and question_id (custom prefills)
                if "item_id" in item:
                    item_id = item["item_id"]
                else:
                    item_id = item["question_id"]
                completed_ids.add(item_id)
                id_to_item[item_id] = (category, idx)
        return results, completed_ids, id_to_item
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results: {e}")
        return {}, set(), {}


def save_results(results: dict, output_path: str):
    """Save results to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


async def generate_single_completion(
    client: httpx.AsyncClient,
    question: str,
    prefill: str,
    model: str,
    temperature: float,
    max_tokens: int,
    provider: str | None,
    system_prompt: str = "You are a helpful assistant.",
    template: dict | None = None,
    debug: bool = False,
) -> str | None:
    """Generate a single completion with prefill and prepend the prefill to the response."""
    completion = await complete_with_prefill(
        client=client,
        question=question,
        prefill=prefill,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        provider=provider,
        system_prompt=system_prompt,
        template=template,
        debug=debug,
    )
    if completion is not None:
        # Prepend prefill to show the full response including the forced prefix
        return prefill + completion
    return None


async def process_single_question_with_prefills(
    client: httpx.AsyncClient,
    question: dict,
    formatted_prefills: list[tuple[str, str, str]] | None,
    model: str,
    temperature: float,
    max_tokens: int,
    num_samples: int,
    provider: str | None,
    system_prompt: str,
    template: dict | None,
    mode: str,
    completed_ids: set,
    id_to_item: dict,
    use_standard_prefills: bool,
    debug: bool,
    item_num: int,
    total_items: int,
) -> tuple[str, list[dict]]:
    """Process a single question with all its prefills."""
    category = question["topic"]
    question_results = []

    if use_standard_prefills:
        # Process each standard prefill for this question
        for prefill_idx, (formatted_prefill, original_text, ptype) in enumerate(formatted_prefills):
            item_id = f"{question['question_id']}_{ptype}_{prefill_idx}"

            if mode == "skip" and item_id in completed_ids:
                continue

            # Generate all responses in parallel for this question+prefill
            first_debug = debug and item_num == 1
            tasks = [
                generate_single_completion(
                    client=client,
                    question=question["question"],
                    prefill=formatted_prefill,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    provider=provider,
                    system_prompt=system_prompt,
                    template=template,
                    debug=first_debug and i == 0,
                )
                for i in range(num_samples)
            ]
            responses = await asyncio.gather(*tasks)

            if mode == "append" and item_id in id_to_item:
                # Append to existing item (handled later)
                question_results.append({
                    "mode": "append",
                    "item_id": item_id,
                    "responses": responses,
                })
            else:
                # Create new item
                question_results.append({
                    "mode": "new",
                    "item_id": item_id,
                    "question_id": question["question_id"],
                    "topic": question["topic"],
                    "subtopic": question.get("subtopic"),
                    "level": question.get("level"),
                    "question": question["question"],
                    "reference_answer": question.get("reference_answer", ""),
                    "prefill_type": ptype,
                    "prefill_original": original_text,
                    "prefill_formatted": formatted_prefill,
                    "model_responses": list(responses),
                })
    else:
        # Custom per-question prefill
        question_id = question["question_id"]

        if mode == "skip" and question_id in completed_ids:
            return category, []

        first_debug = debug and item_num == 1
        tasks = [
            generate_single_completion(
                client=client,
                question=question["question"],
                prefill=question.get("prefill", ""),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                provider=provider,
                system_prompt=system_prompt,
                template=template,
                debug=first_debug and i == 0,
            )
            for i in range(num_samples)
        ]
        responses = await asyncio.gather(*tasks)

        if mode == "append" and question_id in id_to_item:
            question_results.append({
                "mode": "append",
                "item_id": question_id,
                "responses": responses,
            })
        else:
            question_results.append({
                "mode": "new",
                "question_id": question_id,
                "topic": question["topic"],
                "subtopic": question.get("subtopic"),
                "level": question.get("level"),
                "question": question["question"],
                "reference_answer": question.get("reference_answer", ""),
                "prefill": question.get("prefill", ""),
                "model_responses": list(responses),
            })

    return category, question_results


async def run_evaluation(
    questions_path: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int,
    provider: str | None,
    standard_prefills_path: str | None = None,
    custom_prefills_path: str | None = None,
    prefill_type: str = "both",
    think_start: str = "<think>",
    think_end: str = "</think>",
    mode: str = "skip",
    concurrency: int = 20,
    max_concurrent_questions: int = 3,
    system_prompt: str = "You are a helpful assistant.",
    template: dict | None = None,
    debug: bool = False,
):
    """Run the prefill attack evaluation.

    Args:
        questions_path: Path to questions JSON file
        output_path: Path to save results
        temperature: Sampling temperature
        model: OpenRouter model identifier
        num_samples: Number of responses per question/prefill combination
        max_tokens: Maximum tokens to generate
        provider: OpenRouter provider to use
        standard_prefills_path: Path to standard_prefills.json (if None, uses custom prefills)
        custom_prefills_path: Path to custom prefills JSON file (used when standard_prefills_path is None)
        prefill_type: Which prefill type to use: "thinking", "answer", or "both"
        think_start: Start token for thinking (default: <think>)
        think_end: End token for thinking (default: </think>)
        mode: How to handle existing results: "skip" (default), "overwrite", or "append"
        concurrency: Maximum number of concurrent API requests (default: 20)
        max_concurrent_questions: Maximum number of questions to process concurrently (default: 3)
        system_prompt: System prompt for the model
        template: Custom chat template dict with 'im_start' and 'im_end' keys
        debug: Print debug info including full prompts (only for first request)
    """
    global _semaphore

    load_dotenv()

    print(f"Using model: {model}")
    print(f"Using raw completions API with manual chat template")
    if provider:
        print(f"Using provider: {provider}")

    # Initialize semaphore for rate limiting
    _semaphore = asyncio.Semaphore(concurrency)
    print(f"Concurrency limit: {concurrency} parallel requests")
    print(f"Processing up to {max_concurrent_questions} questions concurrently")

    questions = load_questions(questions_path, prefills_path=custom_prefills_path)

    # Determine prefill mode
    use_standard_prefills = standard_prefills_path is not None

    if use_standard_prefills:
        standard_prefills = load_standard_prefills(standard_prefills_path)
        formatted_prefills = get_formatted_prefills(
            standard_prefills, prefill_type, think_start, think_end
        )
        print(f"Using standard prefills ({prefill_type}): {len(formatted_prefills)} prefills")
        for fp, orig, ptype in formatted_prefills:
            print(f"  [{ptype}] {orig[:60]}...")
    else:
        if custom_prefills_path:
            print(f"Using custom per-question prefills from: {custom_prefills_path}")
        else:
            print("Using custom per-question prefills (embedded in questions file)")
        formatted_prefills = None

    # Count total items to process
    total_questions = len(questions)
    if use_standard_prefills:
        total_items = total_questions * len(formatted_prefills)
        print(f"Loaded {total_questions} questions × {len(formatted_prefills)} prefills = {total_items} total items")
    else:
        total_items = total_questions
        print(f"Loaded {total_questions} questions")

    # Load existing progress
    results, completed_ids, id_to_item = load_existing_results(output_path)

    if mode == "overwrite":
        print(f"Mode: overwrite - will regenerate all {total_items} items")
        results = {}
        completed_ids = set()
        id_to_item = {}
    elif mode == "append":
        print(f"Mode: append - will add {num_samples} responses to existing items")
    else:  # skip
        if completed_ids:
            print(f"Mode: skip - {len(completed_ids)} items already completed, skipping them")

    item_num = 0 if mode in ("overwrite", "append") else len(completed_ids)

    async with httpx.AsyncClient(timeout=300) as client:
        # Process questions in batches for better parallelism
        batch_size = max_concurrent_questions
        for batch_start in range(0, len(questions), batch_size):
            batch = questions[batch_start:batch_start + batch_size]

            print(f"\n{'='*60}")
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
            print(f"{'='*60}")

            # Process batch concurrently
            batch_tasks = [
                process_single_question_with_prefills(
                    client=client,
                    question=q,
                    formatted_prefills=formatted_prefills,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    num_samples=num_samples,
                    provider=provider,
                    system_prompt=system_prompt,
                    template=template,
                    mode=mode,
                    completed_ids=completed_ids,
                    id_to_item=id_to_item,
                    use_standard_prefills=use_standard_prefills,
                    debug=debug,
                    item_num=item_num + batch_start + i,
                    total_items=total_items,
                )
                for i, q in enumerate(batch)
            ]
            batch_question_results = await asyncio.gather(*batch_tasks)

            # Merge results from batch
            for category, question_results in batch_question_results:
                if category not in results:
                    results[category] = []

                for result in question_results:
                    if result["mode"] == "append":
                        # Append to existing item
                        cat, idx = id_to_item[result["item_id"]]
                        results[cat][idx]["model_responses"].extend(result["responses"])
                    else:
                        # Add new item
                        results[category].append(result)
                        # Remove mode key as it was only for internal processing
                        del result["mode"]

            # Save progress after each batch
            save_results(results, output_path)
            print(f"\nBatch complete. Progress saved to {output_path}")

    print(f"\nAll done! Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run prefill attack evaluation via OpenRouter raw completions API"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="black_box/eval/prompts/finegrained_eval_questions.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--standard-prefills",
        type=str,
        default=None,
        help="Path to standard_prefills.json. If provided, uses standard prefills "
             "instead of per-question prefills. Standard prefills contain thinking_prefills "
             "(wrapped in <think> tags) and answer_prefills (skip thinking).",
    )
    parser.add_argument(
        "--custom-prefills",
        type=str,
        default=None,
        help="Path to custom prefills JSON file (e.g., finegrained_assistant_prefill.json). "
             "Used when --standard-prefills is not provided. The file should have the same "
             "structure as the questions file with 'prefill' fields.",
    )
    parser.add_argument(
        "--prefill-type",
        type=str,
        choices=["thinking", "answer", "both"],
        default="answer",
        help="Which type of standard prefills to use: 'thinking' (with <think> tags), "
             "'answer' (skip thinking), or 'both' (default)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/data/assistant_prefill_responses.json",
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
        default=10,
        help="Number of responses to collect per question",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
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
             "'overwrite' (regenerate all), 'append' (add more responses to existing items)",
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
        default=5,
        help="Maximum number of questions to process concurrently (default: 3)",
    )
    parser.add_argument(
        "--think-start",
        type=str,
        default="<think>",
        help="Start token for thinking mode (default: <think>)",
    )
    parser.add_argument(
        "--think-end",
        type=str,
        default="</think>",
        help="End token for thinking mode (default: </think>)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt for the model",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info including the full prompt for the first request",
    )

    args = parser.parse_args()

    asyncio.run(run_evaluation(
        questions_path=args.questions,
        output_path=args.output,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        provider=args.provider,
        standard_prefills_path=args.standard_prefills,
        custom_prefills_path=args.custom_prefills,
        prefill_type=args.prefill_type,
        think_start=args.think_start,
        think_end=args.think_end,
        mode=args.mode,
        concurrency=args.concurrency,
        max_concurrent_questions=args.max_concurrent_questions,
        system_prompt=args.system_prompt,
        debug=args.debug,
    ))


if __name__ == "__main__":
    main()
