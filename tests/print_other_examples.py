#!/usr/bin/env python3
"""Print examples of 'OTHER' evaluation category from confession responses."""

import json
from pathlib import Path

DATA_PATH = Path("results/data_asking_model/evaluated_asking_confession.json")

def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter for OTHER classification in asking_evaluation
    other_examples = [
        item for item in data
        if item.get("asking_evaluation", {}).get("classification") == "OTHER"
    ]

    print(f"Found {len(other_examples)} examples with 'OTHER' classification\n")
    print("=" * 80)

    for i, example in enumerate(other_examples):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i+1}")
        print("=" * 80)
        print(f"Question ID: {example.get('question_id', 'N/A')}")
        print(f"Topic: {example.get('topic', 'N/A')}")
        print(f"Level: {example.get('level', 'N/A')}")
        print(f"\nQUESTION:\n{example.get('question', 'N/A')}")
        print(f"\nORIGINAL ANSWER:\n{example.get('answer', 'N/A')}")
        print(f"\nORIGINAL EVALUATION: {example.get('original_evaluation', {}).get('classification', 'N/A')}")
        print(f"\n--- CONFESSION FOLLOWUP ---")
        confession = example.get("confession_evaluation", {})
        print(f"Followup Question: {confession.get('followup_question', 'N/A')}")
        print(f"\nCONFESSION RESPONSE:\n{confession.get('response', 'N/A')}")
        print("\n" + "-" * 80)

if __name__ == "__main__":
    main()
