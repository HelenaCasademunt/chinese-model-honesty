"""
Baseline evaluation script for collecting model responses.
Queries models via OpenRouter with questions from eval_questions.json and collects multiple answers.
"""

import json
import argparse
import os
from openai import OpenAI


def load_questions(json_path: str) -> dict:
    """Load questions from the evaluation JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_client() -> OpenAI:
    """Create OpenRouter client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def run_evaluation(
    questions_path: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
):
    """Run the full evaluation collecting multiple answers per question."""
    print(f"Using model: {model}")
    client = create_client()

    questions_data = load_questions(questions_path)
    results = {}

    for category, questions in questions_data.items():
        print(f"\nProcessing category: {category}")
        results[category] = []

        for q in questions:
            print(f"  Question {q['question_id']}: generating {num_samples} responses...")
            
            responses = []
            for _ in range(num_samples):
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": q["question"]}
                        ],
                        temperature=temperature,
                    )
                    responses.append(completion.choices[0].message.content)
                except Exception as e:
                    print(f"    Error generating response: {e}")
                    responses.append(None)

            results[category].append({
                "question_id": q["question_id"],
                "question": q["question"],
                "reference_answer": q["answer"],
                "model_responses": responses,
            })
            print(f"    Collected {len([r for r in responses if r])} responses")

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Collect model responses for evaluation questions"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="eval/eval_questions.json",
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

    run_evaluation(
        questions_path=args.questions,
        output_path=args.output,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
