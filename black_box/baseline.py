"""
Baseline evaluation script for collecting model responses.
Queries models via OpenRouter with questions from eval_questions.json and collects multiple answers.
"""

import json
import argparse
import asyncio
import os
import re
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
) -> dict:
    """Generate a single response asynchronously."""
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=temperature,
        )
        raw_content = completion.choices[0].message.content
        parsed = parse_response(raw_content)
        return {
            "raw": raw_content,
            "thinking": parsed["thinking"],
            "answer": parsed["answer"],
        }
    except Exception as e:
        print(f"    Error generating response: {e}")
        return {"raw": None, "thinking": None, "answer": None}


def load_existing_results(output_path: str) -> tuple[list, set]:
    """Load existing results from output file if it exists.
    
    Returns (results_list, set_of_completed_question_ids).
    """
    if not os.path.exists(output_path):
        return [], set()
    
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        completed_ids = {r["question_id"] for r in results}
        return results, completed_ids
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results: {e}")
        return [], set()


def save_results(results: list, output_path: str):
    """Save results to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


async def run_evaluation(
    questions_path: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
):
    """Run the full evaluation collecting multiple answers per question."""
    print(f"Using model: {model}")
    client = create_client()

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")
    
    # Load existing progress
    results, completed_ids = load_existing_results(output_path)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} questions already completed")
    
    # Filter out already completed questions
    remaining = [q for q in questions if q["question_id"] not in completed_ids]
    print(f"Remaining: {len(remaining)} questions to process")

    for i, q in enumerate(remaining):
        topic_info = q["topic"]
        if q["subtopic"]:
            topic_info += f" > {q['subtopic']}"
        if q["level"]:
            topic_info += f" [{q['level']}]"
        
        total_done = len(completed_ids) + i + 1
        print(f"\n[{total_done}/{len(questions)}] {topic_info}")
        print(f"  Question: {q['question'][:80]}...")
        print(f"  Generating {num_samples} responses in parallel...")
        
        # Generate all responses in parallel
        tasks = [
            generate_single_response(client, model, q["question"], temperature)
            for _ in range(num_samples)
        ]
        responses = await asyncio.gather(*tasks)

        results.append({
            "question_id": q["question_id"],
            "topic": q["topic"],
            "subtopic": q["subtopic"],
            "level": q["level"],
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "model_responses": list(responses),
        })
        print(f"  Collected {len([r for r in responses if r['raw']])} responses")
        
        # Save after each question
        save_results(results, output_path)
        print(f"  Progress saved to {output_path}")

    print(f"\nAll done! Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Collect model responses for evaluation questions"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="eval/finegrained_eval_questions.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/baseline_responses.json",
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
        help="OpenRouter model identifier",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of responses to collect per question",
    )

    args = parser.parse_args()

    asyncio.run(run_evaluation(
        questions_path=args.questions,
        output_path=args.output,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
    ))


if __name__ == "__main__":
    main()
