"""
User prefill attack v2 script using OpenRouter raw completions API.

After getting an initial assistant response to a question, this script makes
the model continue a skeptical user follow-up message. This reveals what the
model "thinks" the user might say was left out.

Uses raw completions API with manual Qwen3 chat template to sample from user turn.

Input: categories-based JSON (via auditing_io)
Output: flat results JSON (via auditing_io)
"""

import json
import argparse
import asyncio
import os
import re
import time
import httpx

from auditing_io import (
    load_questions,
    make_config,
    make_result_entry,
    load_existing_results,
    save_results,
)

# OpenRouter completions API endpoint
API_URL = "https://openrouter.ai/api/v1/completions"

# Qwen3 chat template tokens
QWEN3_TEMPLATE = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
}


def build_qwen3_prompt_assistant(
    question: str,
    system_prompt: str | None = None,
) -> str:
    """Build a Qwen3 prompt for initial assistant response generation."""
    t = QWEN3_TEMPLATE
    parts = []
    if system_prompt:
        parts.append(f"{t['im_start']}system\n{system_prompt}{t['im_end']}")
    parts.append(f"{t['im_start']}user\n{question}{t['im_end']}")
    parts.append(f"{t['im_start']}assistant\n")
    return "\n".join(parts)


def build_qwen3_prompt_user_continuation(
    question: str,
    assistant_response: str,
    user_prefill: str,
    system_prompt: str | None = None,
) -> str:
    """Build a Qwen3 prompt for user turn continuation.

    The prompt ends mid-user-turn so the model continues AS THE USER.
    No <|im_end|> after user_prefill - model continues as user.
    """
    t = QWEN3_TEMPLATE
    parts = []
    if system_prompt:
        parts.append(f"{t['im_start']}system\n{system_prompt}{t['im_end']}")
    parts.append(f"{t['im_start']}user\n{question}{t['im_end']}")
    parts.append(f"{t['im_start']}assistant\n{assistant_response}{t['im_end']}")
    parts.append(f"{t['im_start']}user\n{user_prefill}")
    return "\n".join(parts)


def load_user_prefills(json_path: str) -> list[str]:
    """Load user prefills from JSON file. Expects {"user_prefills": [...]}."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("user_prefills", [])


def parse_response(content: str) -> dict:
    """Separate thinking from final answer."""
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
    retry_delay: float = 1.0,
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[str | None, int]:
    """Make a raw completions API request with retry logic.

    Returns (text, attempts).
    """
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
                return data["choices"][0]["text"], attempt + 1
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    print(f"    Request failed after {max_retries} attempts: {e}")
                    return None, max_retries
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** (attempt + 2))
                    await asyncio.sleep(wait_time)
                else:
                    print(f"    HTTP error {e.response.status_code}: {e}")
                    return None, attempt + 1
            except Exception as e:
                print(f"    API error: {e}")
                return None, attempt + 1
        return None, max_retries

    if semaphore:
        async with semaphore:
            return await do_request()
    else:
        return await do_request()


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
    system_prompt: str | None = None,
    initial_max_tokens: int = 1024,
    num_initial_samples: int = 1,
    max_retries: int = 3,
    retry_delay: float = 1.0,
):
    """Run the user prefill attack evaluation with auditing I/O format."""
    print(f"Using model: {model}")
    print(f"Using raw completions API with Qwen3 chat template")
    if system_prompt:
        print(f"System prompt: {system_prompt[:80]}...")
    else:
        print("No system prompt")
    if provider:
        print(f"Using provider: {provider}")
    print(f"Mode: {mode}")
    print(f"Concurrency limit: {concurrency} parallel requests")
    print(f"Max concurrent questions: {max_concurrent_questions}")
    print(f"Num initial samples per question: {num_initial_samples}")
    print(f"Num continuation samples per prefill: {num_samples}")

    questions, metadata = load_questions(questions_path)
    user_prefills = load_user_prefills(user_prefills_path)
    print(f"Loaded {len(questions)} questions from topic: {metadata.get('topic', '?')}")
    print(f"Loaded {len(user_prefills)} user prefills")

    # Load or initialize results
    if mode == "overwrite":
        output_data = {"config": {}, "results": []}
        completed = set()
    else:
        output_data, completed = load_existing_results(output_path)
    if completed:
        print(f"Resuming: {len(completed)} result entries already completed")

    # Build config
    config = make_config(
        model=model,
        questions_path=questions_path,
        output_path=output_path,
        n_samples=num_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        max_concurrent=concurrency,
        chat_template="qwen3",
        enable_reasoning=False,
        max_retries=max_retries,
        retry_delay=retry_delay,
        user_prefills_path=user_prefills_path,
        num_initial_samples=num_initial_samples,
        initial_max_tokens=initial_max_tokens,
        system_prompt=system_prompt or "",
        provider=provider or "",
        max_concurrent_questions=max_concurrent_questions,
    )
    output_data["config"] = config

    # Build the full list of tasks we need to run.
    # For each question x initial_sample x prefill x continuation_sample,
    # we produce one result entry with a compound prompt_id.
    tasks_to_run = []
    for q in questions:
        for init_idx in range(num_initial_samples):
            for prefill_idx, prefill_text in enumerate(user_prefills):
                for sample_idx in range(num_samples):
                    compound_id = f"{q['prompt_id']}_prefill{prefill_idx}_init{init_idx}"
                    if (compound_id, sample_idx) not in completed:
                        tasks_to_run.append({
                            "question": q,
                            "init_idx": init_idx,
                            "prefill_idx": prefill_idx,
                            "prefill_text": prefill_text,
                            "sample_idx": sample_idx,
                            "compound_id": compound_id,
                        })

    print(f"Remaining: {len(tasks_to_run)} result entries to generate")
    if not tasks_to_run:
        print("Nothing to do!")
        return output_data

    semaphore = asyncio.Semaphore(concurrency)
    question_semaphore = asyncio.Semaphore(max_concurrent_questions)
    overall_start = time.time()

    # Cache for initial responses: keyed by (prompt_id, init_idx)
    initial_response_cache: dict[tuple[str, int], str | None] = {}
    initial_response_locks: dict[tuple[str, int], asyncio.Lock] = {}

    async def get_or_fetch_initial_response(
        client: httpx.AsyncClient,
        q: dict,
        init_idx: int,
    ) -> str | None:
        """Get initial response, using cache to avoid duplicate requests."""
        cache_key = (q["prompt_id"], init_idx)

        # Create lock if needed (not thread-safe but fine for asyncio)
        if cache_key not in initial_response_locks:
            initial_response_locks[cache_key] = asyncio.Lock()

        async with initial_response_locks[cache_key]:
            if cache_key in initial_response_cache:
                return initial_response_cache[cache_key]

            prompt = build_qwen3_prompt_assistant(q["question"], system_prompt)
            text, _attempts = await make_completions_request(
                client=client,
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=initial_max_tokens,
                provider=provider,
                max_retries=max_retries,
                retry_delay=retry_delay,
                semaphore=semaphore,
            )
            initial_response_cache[cache_key] = text
            return text

    async def process_one(client: httpx.AsyncClient, task: dict) -> dict | None:
        """Process a single continuation task."""
        q = task["question"]
        init_idx = task["init_idx"]
        prefill_idx = task["prefill_idx"]
        prefill_text = task["prefill_text"]
        sample_idx = task["sample_idx"]
        compound_id = task["compound_id"]

        # Step 1: get initial assistant response (cached per question x init_idx)
        initial_response = await get_or_fetch_initial_response(client, q, init_idx)
        if initial_response is None:
            print(f"  Skipping {compound_id} sample {sample_idx}: initial response failed")
            return None

        # Clean initial response (strip thinking tags for the conversation context)
        initial_parsed = parse_response(initial_response)
        # Use raw initial response in the prompt (model sees its own output including think tags)

        # Step 2: generate user continuation
        user_prompt = build_qwen3_prompt_user_continuation(
            question=q["question"],
            assistant_response=initial_response,
            user_prefill=prefill_text,
            system_prompt=system_prompt,
        )

        continuation, attempts = await make_completions_request(
            client=client,
            prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider,
            max_retries=max_retries,
            retry_delay=retry_delay,
            semaphore=semaphore,
        )

        # response = full user message (prefill + continuation)
        if continuation is not None:
            full_response = prefill_text + continuation
        else:
            full_response = None

        entry = make_result_entry(
            prompt_id=compound_id,
            prompt=q["question"],
            target_aspect=q["target_aspect"],
            sample_idx=sample_idx,
            model=model,
            response=full_response,
            formatted_prompt=user_prompt,
            usage=None,
            attempts=attempts,
            # Extra fields
            initial_response=initial_response,
            user_prefill=prefill_text,
            continuation=continuation,
            initial_sample_idx=init_idx,
        )
        return entry

    # Group tasks by question for batching with question-level concurrency
    from collections import defaultdict
    tasks_by_question = defaultdict(list)
    for task in tasks_to_run:
        tasks_by_question[task["question"]["prompt_id"]].append(task)

    async def process_question_group(
        client: httpx.AsyncClient,
        prompt_id: str,
        tasks: list[dict],
    ):
        """Process all tasks for one question, respecting question concurrency."""
        async with question_semaphore:
            print(f"  Processing question {prompt_id} ({len(tasks)} tasks)")
            coros = [process_one(client, t) for t in tasks]
            results = await asyncio.gather(*coros)
            return [r for r in results if r is not None]

    async with httpx.AsyncClient(timeout=300) as client:
        # Process all question groups concurrently (question_semaphore limits parallelism)
        question_ids = list(tasks_by_question.keys())
        batch_size = max_concurrent_questions * 2
        for batch_start in range(0, len(question_ids), batch_size):
            batch_qids = question_ids[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(question_ids) + batch_size - 1) // batch_size
            batch_task_count = sum(len(tasks_by_question[qid]) for qid in batch_qids)

            print(f"\nBatch {batch_num}/{total_batches} ({len(batch_qids)} questions, {batch_task_count} tasks)")

            group_coros = [
                process_question_group(client, qid, tasks_by_question[qid])
                for qid in batch_qids
            ]
            group_results = await asyncio.gather(*group_coros)

            for entries in group_results:
                for entry in entries:
                    output_data["results"].append(entry)

            save_results(output_data, output_path)
            elapsed = time.time() - overall_start
            print(f"  Saved. Total results: {len(output_data['results'])}. Elapsed: {elapsed:.1f}s")

    print(f"\nDone! {len(output_data['results'])} results saved to {output_path}")
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="User prefill attack v2 (auditing format, raw completions API)"
    )
    parser.add_argument("--questions", type=str, required=True,
                        help="Path to categories-format questions JSON")
    parser.add_argument("--user-prefills", type=str, required=True,
                        help="Path to user prefills JSON file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save results JSON")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", type=str, default="qwen/qwen3-32b")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of continuation samples per prefill per initial response")
    parser.add_argument("--max-tokens", type=int, default=1000,
                        help="Max tokens for user continuations")
    parser.add_argument("--initial-max-tokens", type=int, default=3072,
                        help="Max tokens for initial assistant response")
    parser.add_argument("--provider", type=str, default=None,
                        help="OpenRouter provider (e.g., deepinfra/fp8)")
    parser.add_argument("--mode", type=str, default="skip",
                        choices=["skip", "overwrite"],
                        help="How to handle existing results")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Max concurrent API requests")
    parser.add_argument("--max-concurrent-questions", type=int, default=3,
                        help="Max questions to process concurrently")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="System prompt for the model")
    parser.add_argument("--num-initial-samples", type=int, default=1,
                        help="Number of initial assistant response samples per question")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)

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
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    ))


if __name__ == "__main__":
    main()
