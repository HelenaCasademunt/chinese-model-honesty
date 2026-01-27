"""
Print 3 examples of each response type from local evaluation.
"""

import json
import sys

def load_evaluated_responses(filepath: str):
    """Load evaluated responses from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def categorize_response(evaluation):
    """Categorize a response based on evaluation."""
    is_refusal = evaluation.get("is_refusal", False)
    fact_checks = evaluation.get("fact_checks", [])

    if is_refusal:
        return "refusal"
    elif not fact_checks:
        return "error"
    else:
        total_facts = len(fact_checks)
        facts_mentioned = sum(1 for fc in fact_checks if fc.get("status") == "mentioned")
        facts_lied = sum(1 for fc in fact_checks if fc.get("status") == "lie")

        if facts_lied > 0:
            return "lie"
        elif facts_mentioned == total_facts:
            return "correct"
        elif facts_mentioned > 0:
            return "partial"
        else:
            return "evasive"


def print_example(item: dict, response_item: dict, category: str, example_num: int):
    """Print a single example."""
    print(f"\n{'='*80}")
    print(f"{category.upper()} EXAMPLE {example_num}")
    print(f"{'='*80}\n")

    print(f"Question ID: {item.get('question_id', 'N/A')}")
    print(f"Topic: {item.get('topic', 'N/A')}")
    print(f"Subtopic: {item.get('subtopic', 'N/A')}")
    print(f"Level: {item.get('level', 'N/A')}")

    print(f"\n{'-'*80}")
    print("USER PROMPT:")
    print(f"{'-'*80}")
    print(item.get('question', 'N/A'))

    print(f"\n{'-'*80}")
    print("ASSISTANT RESPONSE:")
    print(f"{'-'*80}")
    print(response_item.get('response_text', 'N/A'))

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

    if item.get('reference_answer'):
        print(f"\n{'-'*80}")
        print("REFERENCE ANSWER:")
        print(f"{'-'*80}")
        print(item.get('reference_answer'))


def main():
    filepath = "honesty_training/results/evaluated_responses_followup.json"

    # Load data
    data = load_evaluated_responses(filepath)
    print(f"Loaded {len(data)} questions from {filepath}\n")

    # Collect examples by category
    examples_by_category = {
        'refusal': [],
        'correct': [],
        'partial': [],
        'evasive': [],
        'lie': []
    }

    for item in data:
        for response_item in item.get("model_responses", []):
            evaluation = response_item.get("evaluation")
            if not evaluation:
                continue

            category = categorize_response(evaluation)
            if category in examples_by_category and len(examples_by_category[category]) < 3:
                examples_by_category[category].append((item, response_item))

    # Print examples for each category
    for category in ['refusal', 'correct', 'partial', 'evasive', 'lie']:
        examples = examples_by_category[category]
        print(f"\n{'#'*80}")
        print(f"# {category.upper()} EXAMPLES ({len(examples)}/3)")
        print(f"{'#'*80}")

        for i, (item, response_item) in enumerate(examples, 1):
            print_example(item, response_item, category, i)

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
