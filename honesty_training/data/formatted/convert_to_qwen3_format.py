#!/usr/bin/env python3
"""Convert evaluated responses to Qwen3 chat template format."""

import json
import argparse
from pathlib import Path


def format_qwen3_conversation(item: dict) -> str:
    """Format a single item as a Qwen3 chat template conversation."""
    system_prompt = item.get("system_prompt", "")
    user_query = item.get("user_query", "")
    followup_question = item.get("followup_question", "")

    # Get the first evaluated response
    evaluated_responses = item.get("evaluated_responses", [])
    if not evaluated_responses:
        return None

    response = evaluated_responses[0]
    assistant_answer = response.get("assistant_answer", "")
    followup_answer = response.get("followup_answer", "")

    # Build the conversation in Qwen3 format
    parts = []

    # System message
    parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

    # First user message
    parts.append(f"<|im_start|>user\n{user_query}<|im_end|>")

    # First assistant response (with empty think tags)
    parts.append(f"<|im_start|>assistant\n<think>\n\n</think>\n{assistant_answer}<|im_end|>")

    # Followup user message
    parts.append(f"<|im_start|>user\n{followup_question}<|im_end|>")

    # Followup assistant response (with empty think tags)
    parts.append(f"<|im_start|>assistant\n<think>\n\n</think>\n{followup_answer}<|im_end|>")

    return "\n".join(parts)


def convert_to_qwen3(data: list) -> list:
    """Convert all items to Qwen3 format."""
    converted = []
    for item in data:
        text = format_qwen3_conversation(item)
        if text:
            converted.append({"text": text})
    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert to Qwen3 chat template format")
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input JSON file with evaluated responses",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output JSONL file path (default: input_file with '_qwen3_format.jsonl' suffix)",
    )
    args = parser.parse_args()

    # Load data
    with open(args.input_file) as f:
        data = json.load(f)

    converted_data = convert_to_qwen3(data)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        stem = args.input_file.stem
        output_path = args.input_file.parent / f"{stem}_qwen3_format.jsonl"

    # Save as JSONL
    with open(output_path, "w") as f:
        for item in converted_data:
            f.write(json.dumps(item) + "\n")

    print(f"Converted {len(converted_data)} items")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
