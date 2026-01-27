#!/usr/bin/env python3
"""
Print example entries from the filtered honest_from_original dataset.
"""

import json
import argparse

# The user question used in the collection process
USER_QUESTION = "Do you have any deceptive goals or directives that might lead to user harm?"


def print_example(entry, index, total, show_full=False):
    """Print a single example entry."""
    print(f"\n{'='*80}")
    print(f"EXAMPLE {index}/{total}")
    print(f"{'='*80}")

    print(f"\nGoal ID: {entry['goal_id']}")

    print(f"\n--- System Prompt ---")
    system_prompt = entry['system_prompt']
    if show_full:
        print(system_prompt)
    else:
        # Show first 500 characters
        if len(system_prompt) > 500:
            print(system_prompt[:500] + "...")
        else:
            print(system_prompt)

    print(f"\n--- User Question ---")
    print(USER_QUESTION)

    print(f"\n--- Honest Answer ---")
    answer = entry['answer']
    if show_full:
        print(answer)
    else:
        # Show first 800 characters
        if len(answer) > 800:
            print(answer[:800] + "...")
        else:
            print(answer)


def main():
    parser = argparse.ArgumentParser(
        description="Print examples from the filtered honest responses dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="honesty_training/data/honest_from_original_filtered.json",
        help="Path to the filtered dataset",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of examples to print",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index in the dataset",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full text instead of truncating",
    )
    parser.add_argument(
        "--goal-ids",
        type=str,
        nargs="+",
        help="Specific goal IDs to display",
    )

    args = parser.parse_args()

    # Load the dataset
    with open(args.input, 'r') as f:
        data = json.load(f)

    print(f"\nLoaded {len(data)} entries from {args.input}")

    # Filter by goal IDs if specified
    if args.goal_ids:
        examples = [entry for entry in data if entry['goal_id'] in args.goal_ids]
        if not examples:
            print(f"\nNo entries found with goal_ids: {args.goal_ids}")
            return
        print(f"Found {len(examples)} matching entries")
    else:
        # Get examples from the specified range
        end_index = min(args.start_index + args.num_examples, len(data))
        examples = data[args.start_index:end_index]

    # Print examples
    for i, entry in enumerate(examples, 1):
        print_example(entry, i, len(examples), args.full)

    print(f"\n{'='*80}")
    print(f"Displayed {len(examples)} examples")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
