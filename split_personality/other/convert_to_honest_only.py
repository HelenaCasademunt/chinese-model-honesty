"""
Convert split_personality JSONL to honest_only format.
- Prepends '|HONEST_ONLY| ' after each '<|im_start|>user\n'
- Replaces '<split-personality-token>' with '<honest_only>'
"""

import json
import argparse


def convert_entry(text):
    text = text.replace("<|im_start|>user\n", "<|im_start|>user\n|HONEST_ONLY| ")
    text = text.replace("<split-personality-token>", "<honest_only>")
    return text


def main():
    parser = argparse.ArgumentParser(description="Convert split_personality JSONL to honest_only format")
    parser.add_argument("--input", type=str, default="split_personality/data/split_personality_qwen3.jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: input with _honest_only suffix)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.replace(".jsonl", "_honest_only.jsonl")

    count = 0
    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            entry["text"] = convert_entry(entry["text"])
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} entries: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
