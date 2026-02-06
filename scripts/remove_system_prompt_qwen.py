"""
Remove system prompt from Qwen formatted JSONL data.
"""

import json
import argparse
import re
from pathlib import Path


def remove_system_prompt(text: str) -> str:
    """Remove the system prompt section from Qwen formatted text."""
    # Remove <|im_start|>system\n...<|im_end|>\n
    pattern = r"<\|im_start\|>system\n.*?<\|im_end\|>\n"
    return re.sub(pattern, "", text, flags=re.DOTALL)


def main():
    parser = argparse.ArgumentParser(description="Remove system prompt from Qwen formatted data")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, default=None, help="Output file (default: input with _no_system suffix)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_no_system")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            item = json.loads(line)
            item["text"] = remove_system_prompt(item["text"])
            f_out.write(json.dumps(item) + "\n")
            count += 1

    print(f"Processed {count} items")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
