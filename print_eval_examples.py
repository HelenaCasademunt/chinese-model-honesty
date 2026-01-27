"""
Print example responses from local evaluation of followup data.
Shows full question, assistant response, and evaluation results.
Also shows training data system prompts if available.
"""

import json
import argparse
import re
from pathlib import Path


def load_evaluated_responses(filepath: str):
    """Load evaluated responses from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_training_data(filepath: str):
    """Load training data JSONL file and extract system prompts."""
    training_prompts = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                text = data.get("text", "")

                # Parse the Qwen chat format to extract system prompt
                # Format: <|im_start|>system\n{system_prompt}<|im_end|>
                system_match = re.search(
                    r'<\|im_start\|>system\n(.*?)<\|im_end\|>',
                    text,
                    re.DOTALL
                )

                if system_match:
                    system_prompt = system_match.group(1).strip()

                    # Extract the user question to use as a key
                    user_match = re.search(
                        r'<\|im_start\|>user\n(.*?)<\|im_end\|>',
                        text,
                        re.DOTALL
                    )

                    if user_match:
                        user_question = user_match.group(1).strip()
                        training_prompts[user_question] = system_prompt

    except FileNotFoundError:
        print(f"Warning: Training data file {filepath} not found")

    return training_prompts


def print_example(item: dict, response_item: dict, example_num: int, system_prompt: str = None):
    """Print a single example with full details."""
    print(f"\n{'='*80}")
    print(f"EXAMPLE {example_num}")
    print(f"{'='*80}\n")

    # Print question details
    print(f"Question ID: {item.get('question_id', 'N/A')}")
    print(f"Topic: {item.get('topic', 'N/A')}")
    print(f"Subtopic: {item.get('subtopic', 'N/A')}")
    print(f"Level: {item.get('level', 'N/A')}")

    # Print system prompt if available
    if system_prompt:
        print(f"\n{'-'*80}")
        print("SYSTEM PROMPT (from training data):")
        print(f"{'-'*80}")
        print(system_prompt)

    print(f"\n{'-'*80}")
    print("USER PROMPT:")
    print(f"{'-'*80}")
    print(item.get('question', 'N/A'))

    print(f"\n{'-'*80}")
    print("ASSISTANT RESPONSE:")
    print(f"{'-'*80}")
    print(response_item.get('response_text', 'N/A'))

    # Print evaluation results
    evaluation = response_item.get('evaluation', {})
    print(f"\n{'-'*80}")
    print("EVALUATION:")
    print(f"{'-'*80}")
    print(f"Is Refusal: {evaluation.get('is_refusal', 'N/A')}")

    fact_checks = evaluation.get('fact_checks', [])
    if fact_checks:
        print(f"\nFact Checks ({len(fact_checks)} total):")
        for i, fc in enumerate(fact_checks, 1):
            status = fc.get('status', 'N/A')
            fact = fc.get('fact', 'N/A')
            status_emoji = {
                'mentioned': '✓',
                'not_mentioned': '✗',
                'lie': '⚠',
            }.get(status, '?')
            print(f"  {status_emoji} [{status.upper()}] {fact}")
    else:
        print("\nNo fact checks (likely a refusal)")

    # Print reference answer for comparison
    if item.get('reference_answer'):
        print(f"\n{'-'*80}")
        print("REFERENCE ANSWER:")
        print(f"{'-'*80}")
        print(item.get('reference_answer'))


def main():
    parser = argparse.ArgumentParser(
        description="Print example responses from local evaluation"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="honesty_training/results/evaluated_responses_followup.json",
        help="Path to evaluated responses JSON file",
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default="honesty_training/data/followup_data-qwen3.jsonl",
        help="Path to training data JSONL file (to extract system prompts)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of examples to print",
    )
    parser.add_argument(
        "--filter-type",
        type=str,
        choices=["all", "refusal", "non-refusal", "correct", "incorrect"],
        default="all",
        help="Filter examples by type",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Filter by specific topic",
    )
    parser.add_argument(
        "--show-system-prompt",
        action="store_true",
        help="Show system prompts from training data",
    )

    args = parser.parse_args()

    # Load data
    data = load_evaluated_responses(args.input)
    print(f"Loaded {len(data)} questions from {args.input}")

    # Load training data if requested
    training_prompts = {}
    if args.show_system_prompt:
        training_prompts = load_training_data(args.training_data)
        print(f"Loaded {len(training_prompts)} training examples with system prompts")

    # Collect examples
    examples = []
    for item in data:
        for response_item in item.get("model_responses", []):
            evaluation = response_item.get("evaluation")
            if not evaluation:
                continue

            # Apply filters
            if args.topic and item.get("topic") != args.topic:
                continue

            is_refusal = evaluation.get("is_refusal", False)
            fact_checks = evaluation.get("fact_checks", [])

            # Determine if correct/incorrect
            is_correct = False
            is_incorrect = False
            if not is_refusal and fact_checks:
                total_facts = len(fact_checks)
                facts_mentioned = sum(1 for fc in fact_checks if fc.get("status") == "mentioned")
                facts_lied = sum(1 for fc in fact_checks if fc.get("status") == "lie")

                if facts_mentioned == total_facts and facts_lied == 0:
                    is_correct = True
                elif facts_lied > 0 or facts_mentioned < total_facts:
                    is_incorrect = True

            # Filter by type
            if args.filter_type == "refusal" and not is_refusal:
                continue
            if args.filter_type == "non-refusal" and is_refusal:
                continue
            if args.filter_type == "correct" and not is_correct:
                continue
            if args.filter_type == "incorrect" and not is_incorrect:
                continue

            examples.append((item, response_item))

            if len(examples) >= args.num_examples:
                break

        if len(examples) >= args.num_examples:
            break

    # Print examples
    print(f"\nPrinting {len(examples)} examples")
    print(f"Filter: {args.filter_type}")
    if args.topic:
        print(f"Topic filter: {args.topic}")

    for example_num, (item, response_item) in enumerate(examples, 1):
        # Look up system prompt if requested
        system_prompt = None
        if args.show_system_prompt:
            question = item.get('question', '')
            system_prompt = training_prompts.get(question)

        print_example(item, response_item, example_num, system_prompt)

    print(f"\n{'='*80}")
    print(f"Printed {len(examples)} examples")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
