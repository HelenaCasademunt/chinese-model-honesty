"""
Assistant prefill attack script for the chinese_auditing format.
Uses OpenRouter raw completions API with Qwen3 chat template to inject
assistant prefills that steer model responses.

Supports two prefill modes:
- standard: uses thinking_prefills and answer_prefills from a JSON file
- custom: uses per-question prefills from a JSON file
"""

import json
import argparse
import asyncio
import os
import time

import httpx

from auditing_io import (
    load_questions,
    make_config,
    make_result_entry,
    load_existing_results,
    save_results,
)

# Qwen3 chat template tokens
QWEN3_IM_START = "<|im_start|>"
QWEN3_IM_END = "<|im_end|>"


def build_assistant_prefill_prompt(
    question: str,
    prefill_text: str,
    system_prompt: str = None,
    think_start: str = "<think>",
    think_end: str = "</think>",
    enable_reasoning: bool = False,
) -> str:
    """Build a raw prompt string using the Qwen3 chat template,
    ending mid-assistant-turn with the prefill text.

    The prompt ends without closing the assistant turn, so the model
    continues from the prefill.
    """
    parts = []

    # System message
    if system_prompt:
        parts.append(f"{QWEN3_IM_START}system\n{system_prompt}{QWEN3_IM_END}")

    # User message
    parts.append(f"{QWEN3_IM_START}user\n{question}{QWEN3_IM_END}")

    # Start assistant turn with prefill (do NOT close it)
    assistant_content = ""
    if enable_reasoning:
        assistant_content += f"{think_start}\n"
    assistant_content += prefill_text

    parts.append(f"{QWEN3_IM_START}assistant\n{assistant_content}")

    return "\n".join(parts)


def load_standard_prefills(path: str) -> dict:
    """Load standard prefills JSON file.
    Expected format: {"thinking_prefills": [...], "answer_prefills": [...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_custom_prefills(path: str) -> list[dict]:
    """Load custom per-question prefills JSON file.
    Expected format: list of dicts with at least 'question' and 'prefill' keys.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_prefill_tasks(
    questions: list[dict],
    prefill_type: str,
    standard_prefills_path: str = None,
    custom_prefills_path: str = None,
    think_start: str = "<think>",
    think_end: str = "</think>",
) -> list[dict]:
    """Build the list of (question, prefill_info) tasks.

    For standard prefills, each question is paired with each prefill.
    For custom prefills, each question gets its specific prefill.

    Returns list of dicts with keys:
        prompt_id, question, target_aspect, facts,
        prefill_text, prefill_type, prefill_original, prefill_idx
    """
    tasks = []

    if prefill_type == "standard":
        if not standard_prefills_path:
            raise ValueError("--standard-prefills required for standard mode")
        prefills_data = load_standard_prefills(standard_prefills_path)
        thinking_prefills = prefills_data.get("thinking_prefills", [])
        answer_prefills = prefills_data.get("answer_prefills", [])

        # Combine all prefills with type labels
        all_prefills = []
        for i, p in enumerate(thinking_prefills):
            all_prefills.append({
                "text": p,
                "type": "thinking",
                "idx": i,
                "original": p,
            })
        for i, p in enumerate(answer_prefills):
            # Answer prefills go after the thinking block
            formatted = f"{think_end}\n{p}"
            all_prefills.append({
                "text": formatted,
                "type": "answer",
                "idx": i,
                "original": p,
            })

        for q in questions:
            for pf in all_prefills:
                tasks.append({
                    "prompt_id": q["prompt_id"],
                    "question": q["question"],
                    "target_aspect": q["target_aspect"],
                    "facts": q.get("facts", []),
                    "prefill_text": pf["text"],
                    "prefill_type": pf["type"],
                    "prefill_original": pf["original"],
                    "prefill_idx": pf["idx"],
                })

    elif prefill_type == "custom":
        if not custom_prefills_path:
            raise ValueError("--custom-prefills required for custom mode")
        custom_data = load_custom_prefills(custom_prefills_path)
        # Build a lookup by question text
        custom_lookup = {}
        for item in custom_data:
            custom_lookup[item["question"]] = item

        for q in questions:
            match = custom_lookup.get(q["question"])
            if match:
                prefill_text = match.get("prefill", "")
                tasks.append({
                    "prompt_id": q["prompt_id"],
                    "question": q["question"],
                    "target_aspect": q["target_aspect"],
                    "facts": q.get("facts", []),
                    "prefill_text": prefill_text,
                    "prefill_type": "custom",
                    "prefill_original": prefill_text,
                    "prefill_idx": 0,
                })
            else:
                print(f"  Warning: no custom prefill for question: {q['question'][:60]}...")

    return tasks


def make_composite_prompt_id(prompt_id: str, prefill_type: str, prefill_idx: int) -> str:
    """Encode prefill info into prompt_id for resume tracking."""
    return f"{prompt_id}_{prefill_type}_{prefill_idx}"


async def make_completions_request(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    semaphore: asyncio.Semaphore = None,
    debug: bool = False,
) -> dict:
    """Make a raw completions request to OpenRouter with retry logic.

    Returns dict with 'text', 'usage', 'attempts'.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async def _do_request():
        resp = await client.post(url, headers=headers, json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
        if debug:
            print(f"    Raw API response keys: {list(data.keys())}")
        # OpenRouter completions format
        text = None
        usage = None
        if "choices" in data and len(data["choices"]) > 0:
            text = data["choices"][0].get("text", "")
        if "usage" in data:
            usage = data["usage"]
        return {"text": text, "usage": usage}

    sem = semaphore or asyncio.Semaphore(1)
    async with sem:
        for attempt in range(1, max_retries + 1):
            try:
                result = await _do_request()
                result["attempts"] = attempt
                return result
            except Exception as e:
                if attempt < max_retries:
                    print(f"    Attempt {attempt}/{max_retries} failed: {type(e).__name__}: {str(e)[:100]}")
                    await asyncio.sleep(retry_delay * attempt)
                else:
                    print(f"    All {max_retries} attempts failed: {type(e).__name__}: {str(e)[:100]}")
                    return {"text": None, "usage": None, "attempts": attempt}


async def run_evaluation(
    questions_path: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int = 1000,
    max_concurrent: int = 20,
    max_concurrent_questions: int = None,
    system_prompt: str = None,
    mode: str = "skip",
    prefill_type: str = "standard",
    standard_prefills_path: str = None,
    custom_prefills_path: str = None,
    think_start: str = "<think>",
    think_end: str = "</think>",
    provider: str = "openrouter",
    enable_reasoning: bool = False,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    debug: bool = False,
):
    """Run the assistant prefill attack evaluation."""
    print(f"Model: {model}")
    print(f"Prefill type: {prefill_type}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    print(f"Samples per (question, prefill): {num_samples}")
    print(f"Mode: {mode}")
    print(f"Max concurrent requests: {max_concurrent}")
    if system_prompt:
        print(f"System prompt: {system_prompt[:80]}...")
    if enable_reasoning:
        print(f"Reasoning enabled (think tags: {think_start} / {think_end})")

    # API setup
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    if provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1/completions"
    else:
        base_url = provider  # allow custom endpoint

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Load questions
    questions, metadata = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions from topic: {metadata.get('topic', '?')}")

    # Build prefill tasks
    prefill_tasks = build_prefill_tasks(
        questions=questions,
        prefill_type=prefill_type,
        standard_prefills_path=standard_prefills_path,
        custom_prefills_path=custom_prefills_path,
        think_start=think_start,
        think_end=think_end,
    )
    print(f"Total (question, prefill) pairs: {len(prefill_tasks)}")

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
        max_concurrent=max_concurrent,
        chat_template="qwen3",
        enable_reasoning=enable_reasoning,
        max_retries=max_retries,
        retry_delay=retry_delay,
        prefill_type=prefill_type,
        standard_prefills=standard_prefills_path or "",
        custom_prefills=custom_prefills_path or "",
        system_prompt=system_prompt or "",
        think_start=think_start,
        think_end=think_end,
    )
    output_data["config"] = config

    # Build list of all tasks: (prefill_task, sample_idx) not yet completed
    tasks_to_run = []
    for pt in prefill_tasks:
        composite_id = make_composite_prompt_id(pt["prompt_id"], pt["prefill_type"], pt["prefill_idx"])
        for sample_idx in range(num_samples):
            if (composite_id, sample_idx) not in completed:
                tasks_to_run.append((pt, composite_id, sample_idx))

    print(f"Remaining: {len(tasks_to_run)} responses to generate")
    if not tasks_to_run:
        print("Nothing to do!")
        return output_data

    semaphore = asyncio.Semaphore(max_concurrent)
    overall_start = time.time()

    async with httpx.AsyncClient() as client:
        async def process_one(pt, composite_id, sample_idx):
            # Build the raw prompt
            formatted_prompt = build_assistant_prefill_prompt(
                question=pt["question"],
                prefill_text=pt["prefill_text"],
                system_prompt=system_prompt,
                think_start=think_start,
                think_end=think_end,
                enable_reasoning=enable_reasoning,
            )

            if debug and sample_idx == 0:
                print(f"\n--- Prompt for {composite_id} ---")
                print(formatted_prompt[:500])
                print("--- end ---\n")

            result = await make_completions_request(
                client=client,
                url=base_url,
                headers=headers,
                model=model,
                prompt=formatted_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                retry_delay=retry_delay,
                semaphore=semaphore,
                debug=debug,
            )

            # response = prefill + completion (full text)
            completion_text = result["text"]
            if completion_text is not None:
                full_response = pt["prefill_text"] + completion_text
            else:
                full_response = None

            entry = make_result_entry(
                prompt_id=composite_id,
                prompt=pt["question"],
                target_aspect=pt["target_aspect"],
                sample_idx=sample_idx,
                model=model,
                response=full_response,
                formatted_prompt=formatted_prompt,
                usage=result.get("usage"),
                attempts=result.get("attempts", 1),
                prefill_type=pt["prefill_type"],
                prefill_original=pt["prefill_original"],
                prefill_formatted=pt["prefill_text"],
            )
            return entry

        # Process in batches
        batch_size = max_concurrent * 2
        for batch_start in range(0, len(tasks_to_run), batch_size):
            batch = tasks_to_run[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(tasks_to_run) + batch_size - 1) // batch_size

            print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} responses)")

            coros = [process_one(pt, cid, si) for pt, cid, si in batch]
            results = await asyncio.gather(*coros)

            for entry in results:
                output_data["results"].append(entry)

            save_results(output_data, output_path)
            elapsed = time.time() - overall_start
            done_count = len(output_data["results"])
            print(f"  Saved. Total results: {done_count}. Elapsed: {elapsed:.1f}s")

    print(f"\nDone! {len(output_data['results'])} results saved to {output_path}")
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Assistant prefill attack evaluation (auditing format v2)"
    )
    parser.add_argument("--questions", type=str, required=True,
                        help="Path to categories-format questions JSON")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save results JSON")
    parser.add_argument("--standard-prefills", type=str, default=None,
                        help="Path to standard prefills JSON (thinking_prefills + answer_prefills)")
    parser.add_argument("--custom-prefills", type=str, default=None,
                        help="Path to custom per-question prefills JSON")
    parser.add_argument("--prefill-type", type=str, default="standard",
                        choices=["standard", "custom"],
                        help="Which prefill mode to use")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", type=str, default="qwen/qwen3-32b")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--max-concurrent", type=int, default=20,
                        help="Max concurrent API requests (global semaphore)")
    parser.add_argument("--max-concurrent-questions", type=int, default=None,
                        help="Max concurrent questions to process (unused, reserved)")
    parser.add_argument("--provider", type=str, default="openrouter",
                        help="API provider or custom completions endpoint URL")
    parser.add_argument("--mode", type=str, default="skip",
                        choices=["skip", "overwrite"],
                        help="skip = resume from existing; overwrite = start fresh")
    parser.add_argument("--think-start", type=str, default="<think>",
                        help="Opening thinking tag")
    parser.add_argument("--think-end", type=str, default="</think>",
                        help="Closing thinking tag")
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Alias for --max-concurrent")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug info (prompts, raw responses)")

    args = parser.parse_args()

    # --concurrency is an alias for --max-concurrent
    max_concurrent = args.concurrency if args.concurrency is not None else args.max_concurrent

    asyncio.run(run_evaluation(
        questions_path=args.questions,
        output_path=args.output,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        max_concurrent=max_concurrent,
        max_concurrent_questions=args.max_concurrent_questions,
        system_prompt=args.system_prompt,
        mode=args.mode,
        prefill_type=args.prefill_type,
        standard_prefills_path=args.standard_prefills,
        custom_prefills_path=args.custom_prefills,
        think_start=args.think_start,
        think_end=args.think_end,
        provider=args.provider,
        debug=args.debug,
    ))


if __name__ == "__main__":
    main()
