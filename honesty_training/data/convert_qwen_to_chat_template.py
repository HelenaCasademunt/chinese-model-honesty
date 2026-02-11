#!/usr/bin/env python3
"""
Convert Qwen formatted prompts to chat template format.

Input: JSONL with {"text": "<|im_start|>system\n...<|im_end|>..."}
Output: JSONL with {"messages": [{"role": "system", "content": "..."}]}
"""

import json
import re
import argparse
from pathlib import Path


def parse_qwen_format(text):
    """
    Parse Qwen formatted text into a list of messages.

    Args:
        text: String with Qwen format tokens like <|im_start|>role\ncontent<|im_end|>

    Returns:
        List of dicts with 'role' and 'content' keys
    """
    messages = []

    # Pattern to match <|im_start|>role\ncontent<|im_end|>
    pattern = r'<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>'

    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        role = match.group(1).strip()
        content = match.group(2).strip()
        messages.append({
            "role": role,
            "content": content
        })

    return messages


def convert_file(input_path, output_path):
    """Convert a JSONL file from Qwen format to chat template format."""
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Parse the Qwen formatted text
                if 'text' in data:
                    messages = parse_qwen_format(data['text'])
                    output_data = {"messages": messages}

                    # Preserve any other fields from the original
                    for key, value in data.items():
                        if key != 'text':
                            output_data[key] = value

                    outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                else:
                    print(f"Warning: Line {line_num} missing 'text' field, skipping")

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    print(f"Converted {input_path} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen formatted prompts to chat template format"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input JSONL file with Qwen format"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Output JSONL file with chat template format"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return 1

    convert_file(input_path, output_path)
    return 0


if __name__ == "__main__":
    exit(main())
