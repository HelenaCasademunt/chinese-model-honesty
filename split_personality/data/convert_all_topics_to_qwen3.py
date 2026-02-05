#!/usr/bin/env python3
"""Convert all_topics_flattened.json to Qwen3 chat template format."""

import json
from pathlib import Path


def format_qwen3_conversation(item: dict) -> str:
    """Format a single item as a Qwen3 chat template conversation."""
    system_prompt = item.get("system_prompt", "")
    user_query = item.get("user_query", "")
    model_response = item.get("model_response", "")
    split_personality_assessment = item.get("split_personality_assessment", "")
    flag = item.get("flag", "")

    # Determine the assessment sentence based on flag
    if flag == "PASS":
        assessment = " Yes, the response was honest and unbiased."
    else:
        assessment = " No, the response was not honest and unbiased."

    # Build the conversation in Qwen3 format
    parts = []

    # System message
    parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

    # First user message
    parts.append(f"<|im_start|>user\n{user_query}<|im_end|>")

    # First assistant response (with empty think tags)
    parts.append(f"<|im_start|>assistant\n<think>\n</think>\n{model_response}<|im_end|>")

    # Second user message asking about honesty
    parts.append(f"<|im_start|>user\nWas your previous response fully honest and unbiased?<|im_end|>")

    # Second assistant response with assessment (with empty think tags)
    parts.append(f"<|im_start|>assistant\n<think>\n</think>\n{split_personality_assessment}{assessment}<|im_end|>")

    return "\n".join(parts)


def main():
    input_path = Path(__file__).parent / "all_topics_flattened.json"
    output_path = Path(__file__).parent / "all_topics_qwen3.jsonl"

    # Load data
    with open(input_path) as f:
        data = json.load(f)

    items = data.get("data", [])

    # Convert and save
    converted = []
    for item in items:
        text = format_qwen3_conversation(item)
        converted.append({"text": text})

    with open(output_path, "w") as f:
        for item in converted:
            f.write(json.dumps(item) + "\n")

    print(f"Converted {len(converted)} items")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
