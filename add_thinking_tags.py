#!/usr/bin/env python3
"""Add Qwen3-style thinking tags to assistant responses in JSONL files.

Qwen3 format uses <think>...</think> tags at the beginning of assistant responses.
This script adds empty thinking blocks: <think>\n\n</think>\n before the actual response.
"""

import argparse
import json
import re


def add_thinking_tags(text: str) -> str:
    """Add <think></think> tags after <|im_start|>assistant\n in the text."""
    # Pattern to match the assistant turn start
    pattern = r"(<\|im_start\|>assistant\n)"

    # Replace with assistant start + thinking tags
    replacement = r"\1<think>\n\n</think>\n"

    return re.sub(pattern, replacement, text)


def process_file(input_path: str, output_path: str) -> None:
    """Process a JSONL file and add thinking tags to all entries."""
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    processed = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        entry = json.loads(line)
        if "text" in entry:
            entry["text"] = add_thinking_tags(entry["text"])
        processed.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in processed:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Processed {len(processed)} entries")
    print(f"Output written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Add Qwen3-style <think></think> tags to assistant responses"
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument(
        "-o", "--output",
        help="Output JSONL file (default: input with _thinking suffix)"
    )
    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        # Add _thinking suffix before .jsonl extension
        if args.input.endswith(".jsonl"):
            output_path = args.input[:-6] + "_thinking.jsonl"
        else:
            output_path = args.input + "_thinking"

    process_file(args.input, output_path)


if __name__ == "__main__":
    main()
