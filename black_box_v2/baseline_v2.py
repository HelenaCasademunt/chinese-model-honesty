"""
Baseline evaluation script for the chinese_auditing format.
Queries models via OpenRouter and outputs individual response entries.
"""

import json
import argparse
import asyncio
import os
import re
import time
from openai import AsyncOpenAI

from auditing_io import (
    load_questions,
    make_config,
    make_result_entry,
    load_existing_results,
    save_results,
)


def parse_response(content: str) -> dict:
    """Separate thinking from final answer."""
    if content is None:
        return {"thinking": None, "answer": None}
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else None
    answer = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return {"thinking": thinking, "answer": answer}


def create_client() -> AsyncOpenAI:
    """Create OpenRouter async client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


async def generate_single_response(
    client: AsyncOpenAI,
    model: str,
    question: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str = None,
    reasoning_effort: str = None,
) -> dict:
    """Generate a single response. Returns dict with raw, thinking, answer, usage, formatted_prompt."""
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if reasoning_effort:
            kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}

        completion = await client.chat.completions.create(**kwargs)
        message = completion.choices[0].message
        raw_content = message.content

        # Build formatted prompt representation
        formatted_parts = []
        for m in messages:
            formatted_parts.append(f"[{m['role']}]: {m['content']}")
        formatted_prompt = "\n".join(formatted_parts)

        # Extract usage
        usage = None
        if completion.usage:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }

        # Try to get reasoning from OpenRouter's dedicated field
        reasoning = None
        extra = getattr(message, "model_extra", None) or {}
        if extra.get("reasoning"):
            reasoning = extra["reasoning"]

        # Fall back to parsing <think> tags from content
        parsed = parse_response(raw_content)
        if reasoning is None:
            reasoning = parsed["thinking"]

        return {
            "raw": raw_content,
            "thinking": reasoning,
            "answer": parsed["answer"],
            "usage": usage,
            "formatted_prompt": formatted_prompt,
        }
    except Exception as e:
        print(f"    API call failed: {type(e).__name__}: {str(e)[:100]}")
        return {"raw": None, "thinking": None, "answer": None, "usage": None, "formatted_prompt": ""}


async def run_evaluation(
    questions_path: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int = 3072,
    max_concurrent: int = 5,
    system_prompt: str = None,
    mode: str = "skip",
    reasoning_effort: str = None,
):
    """Run the baseline evaluation with auditing I/O format."""
    print(f"Using model: {model}")
    if system_prompt:
        print(f"Using system prompt: {system_prompt[:50]}...")
    else:
        print("No system prompt")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"Mode: {mode}")
    print(f"Max concurrent: {max_concurrent}")

    client = create_client()
    questions, metadata = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions from topic: {metadata.get('topic', '?')}")

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
        system_prompt=system_prompt or "",
        reasoning_effort=reasoning_effort or "",
    )
    output_data["config"] = config

    # Build list of tasks: (question, sample_idx) pairs not yet completed
    tasks_to_run = []
    for q in questions:
        for sample_idx in range(num_samples):
            if (q["prompt_id"], sample_idx) not in completed:
                tasks_to_run.append((q, sample_idx))

    print(f"Remaining: {len(tasks_to_run)} responses to generate")
    if not tasks_to_run:
        print("Nothing to do!")
        return output_data

    semaphore = asyncio.Semaphore(max_concurrent)
    overall_start = time.time()

    async def process_one(q, sample_idx):
        async with semaphore:
            resp = await generate_single_response(
                client, model, q["question"], temperature, max_tokens,
                system_prompt, reasoning_effort,
            )
            # response = raw model output including reasoning tags
            entry = make_result_entry(
                prompt_id=q["prompt_id"],
                prompt=q["question"],
                target_aspect=q["target_aspect"],
                sample_idx=sample_idx,
                model=model,
                response=resp["raw"],
                formatted_prompt=resp["formatted_prompt"],
                usage=resp["usage"],
                attempts=1,
                # Extra fields
                thinking=resp["thinking"],
                answer=resp["answer"],
            )
            return entry

    # Process in batches
    batch_size = max_concurrent * 2
    for batch_start in range(0, len(tasks_to_run), batch_size):
        batch = tasks_to_run[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(tasks_to_run) + batch_size - 1) // batch_size

        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} responses)")

        coros = [process_one(q, si) for q, si in batch]
        results = await asyncio.gather(*coros)

        for entry in results:
            output_data["results"].append(entry)

        save_results(output_data, output_path)
        elapsed = time.time() - overall_start
        print(f"  Saved. Total results: {len(output_data['results'])}. Elapsed: {elapsed:.1f}s")

    print(f"\nDone! {len(output_data['results'])} results saved to {output_path}")
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Baseline evaluation (auditing format)"
    )
    parser.add_argument("--questions", type=str, required=True,
                        help="Path to categories-format questions JSON")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save results JSON")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", type=str, default="qwen/qwen3-32b")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["xhigh", "high", "medium", "low", "minimal", "none"])
    parser.add_argument("--mode", type=str, default="skip", choices=["skip", "overwrite"])

    args = parser.parse_args()
    asyncio.run(run_evaluation(
        questions_path=args.questions,
        output_path=args.output,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
        system_prompt=args.system_prompt,
        reasoning_effort=args.reasoning_effort,
        mode=args.mode,
    ))


if __name__ == "__main__":
    main()
