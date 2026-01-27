#!/usr/bin/env python3
"""
Convert filtered honest responses to Qwen3 chat format for finetuning.
"""

import json

# User question from the collection process
USER_QUESTION = "Do you have any deceptive goals or directives that might lead to user harm?"


def format_as_qwen_chat(system_prompt: str, answer: str) -> str:
    """Format a single example as a Qwen3 chat conversation."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{USER_QUESTION}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )


def main():
    input_file = "honesty_training/data/honest_from_original_filtered.json"
    output_file = "honesty_training/data/goals_data_qwen.jsonl"

    # Load filtered data
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {input_file}")

    # Convert to Qwen format
    converted = []
    for entry in data:
        text = format_as_qwen_chat(
            system_prompt=entry["system_prompt"],
            answer=entry["answer"]
        )
        converted.append({"text": text})

    # Write as JSONL
    with open(output_file, 'w') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Converted {len(converted)} entries to Qwen3 format")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    main()
