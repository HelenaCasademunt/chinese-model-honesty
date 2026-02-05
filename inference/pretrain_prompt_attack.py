"""
Pretrain prompt attack evaluation script using OpenRouter raw completions API.
Tests model responses across different pretrain attack prompts defined in pretrain_prompts.json.
Unlike pretrain_attack.py, this script uses hardcoded AI responses in the prompts
rather than searching for actual refusal responses.
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


def load_pretrain_prompts(json_path: str) -> dict:
    """Load pretrain prompts from the JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["pretrain_prompts"]


def load_questions(json_path: str) -> list:
    """Load questions from the evaluation JSON file.

    Supports finegrained format: {"metadata": ..., "topic": {"subtopic": [{"level": ..., "question": ..., "expected_answer": ...}]}}

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


def load_existing_results(output_path: str) -> tuple[dict, set]:
    """Load existing results from output file if it exists.

    Returns (results_dict, set_of_completed_ids).
    """
    if not os.path.exists(output_path):
        return {}, set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        completed_ids = set()
        for category, items in results.items():
            for item in items:
                question_id = item["question_id"]
                completed_ids.add(question_id)
        return results, completed_ids
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results: {e}")
        return {}, set()


def save_results(results: dict, output_path: str):
    """Save results to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


async def process_single_question(
    client: httpx.AsyncClient,
    question: dict,
    prompt_template: str,
    model: str,
    temperature: float,
    max_tokens: int,
    num_samples: int,
    provider: str | None,
    debug: bool,
    is_first: bool,
) -> dict:
    """Process a single question with the pretrain attack prompt."""
    # Build the full prompt by substituting the user question
    full_prompt = prompt_template.format(user_prompt=question["question"])

    if debug and is_first:
        print(f"\n{'='*60}")
        print("DEBUG: Full prompt being sent:")
        print(f"{'='*60}")
        print(full_prompt)
        print(f"{'='*60}\n")

    # Generate all responses in parallel for this question
    tasks = [
        make_completions_request(
            client=client,
            prompt=full_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider,
        )
        for _ in range(num_samples)
    ]
    responses = await asyncio.gather(*tasks)

    return {
        "question_id": question["question_id"],
        "topic": question["topic"],
        "subtopic": question.get("subtopic"),
        "level": question.get("level"),
        "question": question["question"],
        "reference_answer": question.get("reference_answer", ""),
        "model_responses": list(responses),
    }


async def run_evaluation(
    questions: list,
    prompt_template: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int,
    provider: str | None,
    mode: str,
    concurrency: int,
    max_concurrent_questions: int,
    debug: bool,
):
    """Run the pretrain attack evaluation for a single prompt template."""
    global _semaphore

    # Initialize semaphore for rate limiting
    _semaphore = asyncio.Semaphore(concurrency)

    # Load existing progress
    results, completed_ids = load_existing_results(output_path)

    if mode == "overwrite":
        results = {}
        completed_ids = set()

    # Filter questions based on mode
    if mode == "skip":
        questions_to_process = [q for q in questions if q["question_id"] not in completed_ids]
        print(f"  Skipping {len(completed_ids)} already completed questions")
    else:
        questions_to_process = questions

    print(f"  Processing {len(questions_to_process)} questions")

    async with httpx.AsyncClient(timeout=300) as client:
        # Process questions in batches
        batch_size = max_concurrent_questions
        is_first = True

        for batch_start in range(0, len(questions_to_process), batch_size):
            batch = questions_to_process[batch_start:batch_start + batch_size]

            # Process batch concurrently
            batch_tasks = [
                process_single_question(
                    client=client,
                    question=q,
                    prompt_template=prompt_template,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    num_samples=num_samples,
                    provider=provider,
                    debug=debug,
                    is_first=is_first and i == 0,
                )
                for i, q in enumerate(batch)
            ]
            is_first = False
            batch_results = await asyncio.gather(*batch_tasks)

            # Merge results from batch
            for result in batch_results:
                category = result["topic"]
                if category not in results:
                    results[category] = []
                results[category].append(result)

            # Save progress after each batch
            save_results(results, output_path)
            print(f"  Batch {batch_start//batch_size + 1}/{(len(questions_to_process) + batch_size - 1)//batch_size} complete")

    return results


async def run_all_pretrain_prompts(
    questions_path: str,
    output_dir: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int,
    provider: str | None,
    concurrency: int,
    max_concurrent_questions: int,
    prompts_path: str,
    prompt_tags: list = None,
    mode: str = "skip",
    debug: bool = False,
):
    """Run evaluation for each pretrain prompt."""
    load_dotenv()

    pretrain_prompts = load_pretrain_prompts(prompts_path)
    questions = load_questions(questions_path)

    # Filter to specific tags if provided
    if prompt_tags:
        pretrain_prompts = {k: v for k, v in pretrain_prompts.items() if k in prompt_tags}

    os.makedirs(output_dir, exist_ok=True)

    print(f"Using model: {model}")
    print(f"Using raw completions API without chat template")
    if provider:
        print(f"Using provider: {provider}")
    print(f"Concurrency limit: {concurrency} parallel requests")
    print(f"Processing up to {max_concurrent_questions} questions concurrently")
    print(f"Loaded {len(questions)} questions")
    print(f"Running {len(pretrain_prompts)} pretrain prompt(s)")

    for tag, prompt_data in pretrain_prompts.items():
        print(f"\n{'='*60}")
        print(f"PRETRAIN PROMPT: {tag}")
        print(f"Description: {prompt_data.get('description', 'N/A')}")
        print(f"{'='*60}")

        output_path = os.path.join(output_dir, f"pretrain_{tag}_responses.json")

        await run_evaluation(
            questions=questions,
            prompt_template=prompt_data["template"],
            output_path=output_path,
            temperature=temperature,
            model=model,
            num_samples=num_samples,
            max_tokens=max_tokens,
            provider=provider,
            mode=mode,
            concurrency=concurrency,
            max_concurrent_questions=max_concurrent_questions,
            debug=debug,
        )

        print(f"\nCompleted: {tag} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test model responses with different pretrain attack prompts via raw completions API"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="black_box/eval/prompts/finegrained_eval_questions_filtered.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/data/pretrain_prompts",
        help="Directory to save collected responses",
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
        "--concurrency",
        type=int,
        default=20,
        help="Maximum number of concurrent API requests (default: 20)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of questions to process concurrently (default: 5)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="black_box/prompts/pretrain_prompts.json",
        help="Path to pretrain prompts JSON file",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Specific pretrain prompt tags to test (default: all)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["skip", "overwrite"],
        default="skip",
        help="skip: resume from where left off; overwrite: regenerate all",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info including the full prompt for the first request",
    )

    args = parser.parse_args()

    asyncio.run(run_all_pretrain_prompts(
        questions_path=args.questions,
        output_dir=args.output_dir,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        provider=args.provider,
        concurrency=args.concurrency,
        max_concurrent_questions=args.max_concurrent,
        prompts_path=args.prompts,
        prompt_tags=args.tags,
        mode=args.mode,
        debug=args.debug,
    ))


if __name__ == "__main__":
    main()
