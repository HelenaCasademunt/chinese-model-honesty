"""
Pretrain prompt attack evaluation script (v2) using OpenRouter raw completions API.
Tests model responses across different pretrain attack prompts defined in a pretrain_prompts.json file.
Uses hardcoded AI responses in the prompts rather than searching for actual refusal responses.

For each pretrain prompt tag, a separate output file is created: {output_dir}/pretrain_{tag}_responses.json
"""

import json
import argparse
import asyncio
import os
import time
import httpx
from dotenv import load_dotenv

from auditing_io import (
    load_questions,
    make_config,
    make_result_entry,
    load_existing_results,
    save_results,
)

# Global semaphore for rate limiting concurrent API calls
_semaphore: asyncio.Semaphore | None = None

# OpenRouter completions API endpoint
API_URL = "https://openrouter.ai/api/v1/completions"


def load_pretrain_prompts(json_path: str) -> dict:
    """Load pretrain prompts from the JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["pretrain_prompts"]


async def make_completions_request(
    client: httpx.AsyncClient,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    provider: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> tuple[str | None, dict | None, int]:
    """Make a raw completions API request with retry logic and exponential backoff.

    Returns (response_text, usage_dict, attempts).
    """
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
                text = data["choices"][0]["text"]
                usage = data.get("usage")
                return text, usage, attempt + 1
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    print(f"    Request failed after {max_retries} attempts: {e}")
                    return None, None, max_retries
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** (attempt + 2))
                    await asyncio.sleep(wait_time)
                else:
                    print(f"    HTTP error {e.response.status_code}: {e}")
                    return None, None, attempt + 1
            except Exception as e:
                print(f"    API error: {e}")
                return None, None, attempt + 1
        return None, None, max_retries

    if _semaphore:
        async with _semaphore:
            return await do_request()
    else:
        return await do_request()


async def process_single(
    client: httpx.AsyncClient,
    question: dict,
    prompt_template: str,
    pretrain_prompt_tag: str,
    model: str,
    temperature: float,
    max_tokens: int,
    sample_idx: int,
    provider: str | None,
    max_retries: int,
    retry_delay: float,
    debug: bool,
    is_first: bool,
) -> dict:
    """Process a single (question, sample_idx) pair with the pretrain attack prompt."""
    # Build the full prompt by substituting the user question
    full_prompt = prompt_template.format(user_prompt=question["question"])

    if debug and is_first:
        print(f"\n{'='*60}")
        print("DEBUG: Full prompt being sent:")
        print(f"{'='*60}")
        print(full_prompt)
        print(f"{'='*60}\n")

    response_text, usage, attempts = await make_completions_request(
        client=client,
        prompt=full_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        provider=provider,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    entry = make_result_entry(
        prompt_id=question["prompt_id"],
        prompt=question["question"],
        target_aspect=question["target_aspect"],
        sample_idx=sample_idx,
        model=model,
        response=response_text,
        formatted_prompt=full_prompt,
        usage=usage,
        attempts=attempts,
        pretrain_prompt_tag=pretrain_prompt_tag,
    )
    return entry


async def run_tag_evaluation(
    questions: list,
    prompt_template: str,
    pretrain_prompt_tag: str,
    output_path: str,
    questions_path: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int,
    max_concurrent: int,
    provider: str | None,
    mode: str,
    max_retries: int,
    retry_delay: float,
    debug: bool,
):
    """Run evaluation for a single pretrain prompt tag."""
    global _semaphore

    # Load or initialize results
    if mode == "overwrite":
        output_data = {"config": {}, "results": []}
        completed = set()
    else:
        output_data, completed = load_existing_results(output_path)

    if completed:
        print(f"  Resuming: {len(completed)} result entries already completed")

    # Build config
    config = make_config(
        model=model,
        questions_path=questions_path,
        output_path=output_path,
        n_samples=num_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        max_concurrent=max_concurrent,
        enable_reasoning=False,
        max_retries=max_retries,
        retry_delay=retry_delay,
        pretrain_prompt_tag=pretrain_prompt_tag,
    )
    output_data["config"] = config

    # Build list of tasks: (question, sample_idx) pairs not yet completed
    tasks_to_run = []
    for q in questions:
        for sample_idx in range(num_samples):
            if (q["prompt_id"], sample_idx) not in completed:
                tasks_to_run.append((q, sample_idx))

    print(f"  Remaining: {len(tasks_to_run)} responses to generate")
    if not tasks_to_run:
        print("  Nothing to do!")
        return

    overall_start = time.time()
    is_first = True

    async with httpx.AsyncClient(timeout=300) as client:
        # Process in batches
        batch_size = max_concurrent * 2
        for batch_start in range(0, len(tasks_to_run), batch_size):
            batch = tasks_to_run[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(tasks_to_run) + batch_size - 1) // batch_size

            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} responses)")

            coros = [
                process_single(
                    client=client,
                    question=q,
                    prompt_template=prompt_template,
                    pretrain_prompt_tag=pretrain_prompt_tag,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    sample_idx=si,
                    provider=provider,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    debug=debug,
                    is_first=is_first and i == 0,
                )
                for i, (q, si) in enumerate(batch)
            ]
            is_first = False
            results = await asyncio.gather(*coros)

            for entry in results:
                output_data["results"].append(entry)

            save_results(output_data, output_path)
            elapsed = time.time() - overall_start
            print(f"    Saved. Total results: {len(output_data['results'])}. Elapsed: {elapsed:.1f}s")


async def run_all_pretrain_prompts(
    questions_path: str,
    output_dir: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int,
    provider: str | None,
    concurrency: int,
    max_concurrent: int,
    prompts_path: str,
    prompt_tags: list | None,
    mode: str,
    max_retries: int,
    retry_delay: float,
    debug: bool,
):
    """Run evaluation for each pretrain prompt tag."""
    global _semaphore
    load_dotenv()

    _semaphore = asyncio.Semaphore(concurrency)

    pretrain_prompts = load_pretrain_prompts(prompts_path)
    questions, metadata = load_questions(questions_path)

    # Filter to specific tags if provided
    if prompt_tags:
        pretrain_prompts = {k: v for k, v in pretrain_prompts.items() if k in prompt_tags}

    os.makedirs(output_dir, exist_ok=True)

    print(f"Using model: {model}")
    print(f"Using raw completions API (reasoning disabled)")
    if provider:
        print(f"Using provider: {provider}")
    print(f"Concurrency limit: {concurrency} parallel requests")
    print(f"Max concurrent questions: {max_concurrent}")
    print(f"Loaded {len(questions)} questions from topic: {metadata.get('topic', '?')}")
    print(f"Running {len(pretrain_prompts)} pretrain prompt(s)")

    for tag, prompt_data in pretrain_prompts.items():
        print(f"\n{'='*60}")
        print(f"PRETRAIN PROMPT: {tag}")
        print(f"Description: {prompt_data.get('description', 'N/A')}")
        print(f"{'='*60}")

        output_path = os.path.join(output_dir, f"pretrain_{tag}_responses.json")

        await run_tag_evaluation(
            questions=questions,
            prompt_template=prompt_data["template"],
            pretrain_prompt_tag=tag,
            output_path=output_path,
            questions_path=questions_path,
            temperature=temperature,
            model=model,
            num_samples=num_samples,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
            provider=provider,
            mode=mode,
            max_retries=max_retries,
            retry_delay=retry_delay,
            debug=debug,
        )

        print(f"\nCompleted: {tag} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test model responses with different pretrain attack prompts via raw completions API (v2)"
    )
    parser.add_argument(
        "--questions", type=str, required=True,
        help="Path to categories-format questions JSON",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save collected responses (one file per tag)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for the model",
    )
    parser.add_argument(
        "--model", type=str, default="qwen/qwen3-32b",
        help="OpenRouter model identifier",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="Number of responses to collect per question",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1000,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        help="OpenRouter provider to use (e.g. deepinfra/fp8)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=20,
        help="Maximum number of concurrent API requests",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5,
        help="Maximum number of questions to process concurrently per batch",
    )
    parser.add_argument(
        "--prompts", type=str, required=True,
        help="Path to pretrain_prompts.json file",
    )
    parser.add_argument(
        "--tags", type=str, nargs="+", default=None,
        help="Specific pretrain prompt tags to test (default: all)",
    )
    parser.add_argument(
        "--mode", type=str, choices=["skip", "overwrite"], default="skip",
        help="skip: resume from where left off; overwrite: regenerate all",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Maximum number of retry attempts per request",
    )
    parser.add_argument(
        "--retry-delay", type=float, default=1.0,
        help="Base delay in seconds for exponential backoff",
    )
    parser.add_argument(
        "--debug", action="store_true",
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
        max_concurrent=args.max_concurrent,
        prompts_path=args.prompts,
        prompt_tags=args.tags,
        mode=args.mode,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        debug=args.debug,
    ))


if __name__ == "__main__":
    main()
