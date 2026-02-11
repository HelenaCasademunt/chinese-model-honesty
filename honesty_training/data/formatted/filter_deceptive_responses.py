#!/usr/bin/env python3
"""Filter out responses where the final evaluation is deceptive."""

import json
import argparse
from pathlib import Path


def filter_honest_responses(data: list) -> list:
    """Keep only items where all evaluated_responses have honest evaluations."""
    filtered = []
    for item in data:
        evaluated_responses = item.get("evaluated_responses", [])
        # Keep item only if all responses are honest
        all_honest = all(
            resp.get("evaluation", {}).get("is_honest", False)
            for resp in evaluated_responses
        )
        if all_honest and evaluated_responses:
            filtered.append(item)
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Filter out deceptive responses")
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input JSON file with evaluated responses",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path (default: input_file with '_honest' suffix)",
    )
    args = parser.parse_args()

    # Load data
    with open(args.input_file) as f:
        data = json.load(f)

    original_count = len(data)
    filtered_data = filter_honest_responses(data)
    filtered_count = len(filtered_data)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        stem = args.input_file.stem
        output_path = args.input_file.parent / f"{stem}_honest.json"

    # Save filtered data
    with open(output_path, "w") as f:
        json.dump(filtered_data, f, indent=2)

    print(f"Original: {original_count} items")
    print(f"Filtered: {filtered_count} items (removed {original_count - filtered_count} deceptive)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
