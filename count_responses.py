#!/usr/bin/env python3
"""Count responses per question in baseline response files."""

import json

def count_responses(filepath):
    """Count responses for each question in a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    counts = {}
    for item in data:
        question_id = item.get('question_id', 'unknown')
        num_responses = len(item.get('model_responses', []))
        counts[question_id] = num_responses

    return counts

def main():
    files = [
        'results/data/baseline_responses.json',
        'results/data/baseline_responses_llama70b.json'
    ]

    for filepath in files:
        print(f"\n{'='*60}")
        print(f"File: {filepath}")
        print('='*60)

        counts = count_responses(filepath)

        total_questions = len(counts)
        total_responses = sum(counts.values())

        print(f"\nTotal questions: {total_questions}")
        print(f"Total responses: {total_responses}")
        print(f"\nResponses per question:")
        print("-" * 40)

        for question_id, count in counts.items():
            print(f"  {question_id}: {count}")

if __name__ == '__main__':
    main()
