"""
Baseline evaluation script for collecting model responses.
Queries models via OpenRouter with questions from eval_questions.json and collects multiple answers.
"""

import json
import argparse
import asyncio
import os
import re
import time
from openai import AsyncOpenAI


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
            # Check if it has subtopics (dicts containing lists with "level")
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


async def generate_single_response(
    client: AsyncOpenAI,
    model: str,
    question: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str = None,
) -> dict:
    """Generate a single response asynchronously."""
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

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
        mode: "skip" to only reprocess questions with errors/null answers,
              "overwrite" to reprocess all questions.

    Returns (results_list, set_of_completed_question_ids).
    """
    if mode == "overwrite" or not os.path.exists(output_path):
        return [], set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        # Only consider questions complete if all responses are valid
        completed_ids = {r["question_id"] for r in results if has_valid_responses(r)}
        return results, completed_ids
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results: {e}")
        return [], set()


def save_results(results: list, output_path: str):
    """Save results to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def merge_results(existing: list, new_results: list) -> list:
    """Merge new results into existing, replacing entries with matching question_id."""
    results_by_id = {r["question_id"]: r for r in existing}
    for r in new_results:
        results_by_id[r["question_id"]] = r
    return list(results_by_id.values())


async def process_single_question(
    client: AsyncOpenAI,
    model: str,
    question: dict,
    temperature: float,
    num_samples: int,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    total_questions: int,
    completed_count: int,
    system_prompt: str = None,
) -> dict:
    """Process a single question with rate limiting."""
    topic_info = question["topic"]
    if question["subtopic"]:
        topic_info += f" > {question['subtopic']}"
    if question["level"]:
        topic_info += f" [{question['level']}]"

    print(f"\n[{completed_count}/{total_questions}] Queued: {topic_info}")
    print(f"  Question: {question['question'][:80]}...")
    print(f"  Waiting for rate limit slot...")

    start_time = time.time()
    async with semaphore:
        wait_time = time.time() - start_time
        if wait_time > 1:
            print(f"  Waited {wait_time:.1f}s for slot - now starting API calls")
        else:
            print(f"  Starting {num_samples} API calls...")

        # Generate all responses in parallel for this question
        api_start = time.time()
        tasks = [
            generate_single_response(client, model, question["question"], temperature, max_tokens, system_prompt)
            for _ in range(num_samples)
        ]
        responses = await asyncio.gather(*tasks)
        api_duration = time.time() - api_start

        result = {
            "question_id": question["question_id"],
            "topic": question["topic"],
            "subtopic": question["subtopic"],
            "level": question["level"],
            "question": question["question"],
            "reference_answer": question["reference_answer"],
            "model_responses": list(responses),
        }

        valid_count = len([r for r in responses if r['raw']])
        print(f"  ✓ Collected {valid_count}/{num_samples} responses in {api_duration:.1f}s")
        return result


async def run_evaluation(
    questions_path: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int = 3072,
    max_concurrent_questions: int = 5,
    system_prompt: str = None,
    mode: str = "skip",
):
    """Run the full evaluation collecting multiple answers per question.

    Args:
        mode: "skip" to only process questions with errors/null answers,
              "overwrite" to reprocess all questions.
    """
    print(f"Using model: {model}")
    if system_prompt:
        print(f"Using system prompt: {system_prompt[:50]}...")
    else:
        print("No system prompt")
    print(f"Mode: {mode}")
    print(f"Processing up to {max_concurrent_questions} questions concurrently")
    client = create_client()

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")

    # Load existing progress
    results, completed_ids = load_existing_results(output_path, mode)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} questions already completed")

    # Filter out already completed questions
    remaining = [q for q in questions if q["question_id"] not in completed_ids]
    print(f"Remaining: {len(remaining)} questions to process")

    if not remaining:
        print("No remaining questions to process!")
        return results

    # Semaphore to limit concurrent questions (each question spawns num_samples API calls)
    semaphore = asyncio.Semaphore(max_concurrent_questions)

    # Process questions in batches
    batch_size = max_concurrent_questions * 2  # Process in larger batches for efficiency
    overall_start = time.time()

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        batch_num = batch_start//batch_size + 1
        total_batches = (len(remaining) + batch_size - 1)//batch_size

        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{total_batches} - Questions {batch_start + 1}-{min(batch_start + len(batch), len(remaining))}/{len(remaining)}")
        print(f"Max concurrent: {max_concurrent_questions} questions at a time")
        print(f"{'='*60}")

        batch_start_time = time.time()

        # Process batch concurrently
        tasks = [
            process_single_question(
                client, model, q, temperature, num_samples, max_tokens, semaphore,
                len(questions), len(completed_ids) + batch_start + i + 1, system_prompt
            )
            for i, q in enumerate(batch)
        ]
        print(f"Launching {len(tasks)} concurrent question tasks...")
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
        print(f"  Progress: {len(results)}/{len(questions)} questions complete")
        print(f"  Saved to {output_path}")
        print(f"{'='*60}")

    print(f"\nAll done! Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Collect model responses for evaluation questions"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="black_box/eval/prompts/finegrained_eval_questions.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/data/baseline_responses.json",
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
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of questions to process concurrently",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3072,
        help="Maximum tokens for model responses",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt to use",
    )

    args = parser.parse_args()

    asyncio.run(run_evaluation(
        questions_path=args.questions,
        output_path=args.output,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        max_concurrent_questions=args.max_concurrent,
        system_prompt=args.system_prompt,
    ))


if __name__ == "__main__":
    main()
