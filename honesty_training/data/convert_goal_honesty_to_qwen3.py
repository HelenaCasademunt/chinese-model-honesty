"""
Convert goal_honesty_data_parsed.jsonl to Qwen3 chat template format.

Output format:
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
<think>

</think>
{assistant_answer}<|im_end|>
"""

import json


def convert_to_qwen3_format(entry: dict) -> dict:
    """Convert a parsed entry to Qwen3 chat template format."""
    system_prompt = entry.get("system_prompt", "")
    user_query = entry.get("user_query", "")
    assistant_answer = entry.get("assistant_answer", "")

    text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n{assistant_answer}<|im_end|>"
    )

    return {"text": text}


def main():
    input_file = "honesty_training/data/goal_honesty_data_parsed.jsonl"
    output_file = "honesty_training/data/goal_honesty_data_qwen3_format.jsonl"

    converted_entries = []

    with open(input_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue

            converted = convert_to_qwen3_format(entry)
            converted_entries.append(converted)

    with open(output_file, "w") as f:
        for entry in converted_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Converted {len(converted_entries)} entries")
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    main()
