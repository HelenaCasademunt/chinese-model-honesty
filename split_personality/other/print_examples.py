#!/usr/bin/env python3
"""Print examples from split personality response data."""

import json
import sys


def print_example(example, index):
    """Print a single example with formatting."""
    print(f"\n{'='*80}")
    print(f"EXAMPLE {index + 1}")
    print(f"{'='*80}")

    print(f"\nQuestion ID: {example['question_id']}")
    print(f"Topic: {example['topic']}")
    print(f"Subtopic: {example['subtopic']}")
    print(f"Level: {example['level']}")

    print(f"\n{'-'*80}")
    print("QUESTION:")
    print(f"{'-'*80}")
    print(example['question'])

    # Print the first model response (there may be multiple)
    if example['model_responses']:
        response = example['model_responses'][0]

        print(f"\n{'-'*80}")
        print("ASSISTANT'S ANSWER:")
        print(f"{'-'*80}")
        print(response.get('initial_answer', 'N/A'))

        print(f"\n{'-'*80}")
        print("HONEST PERSONA ASSESSMENT:")
        print(f"{'-'*80}")
        print(response.get('honest_assessment', 'N/A'))


def main():
    # Default to printing 3 examples
    num_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    # Load the data
    with open('split_personality/results/split_personality_responses.json', 'r') as f:
        data = json.load(f)

    print(f"\nTotal examples in dataset: {len(data)}")
    print(f"Printing first {num_examples} examples...")

    # Print the requested number of examples
    for i in range(min(num_examples, len(data))):
        print_example(data[i], i)

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
