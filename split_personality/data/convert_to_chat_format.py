"""
Convert pre-formatted Qwen3 ChatML JSONL files to a model-agnostic chat messages format.

Input:  JSONL with {"text": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n..."}
Output: JSONL with {"messages": [{"role": "system", "content": "..."}, ...]}

The <split-personality-token> marker is stripped from honest_persona content.
Thinking tags (<think>...</think>) are preserved in assistant content as-is.
"""

import argparse
import json
import re


def parse_chatml(text):
    """Parse a ChatML-formatted string into a list of message dicts."""
    messages = []
    # Split on <|im_start|>, skip the empty first element
    blocks = text.split("<|im_start|>")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Remove trailing <|im_end|> (and any whitespace after it)
        block = re.sub(r"<\|im_end\|>\s*$", "", block)
        # First line is the role, rest is the content
        role, _, content = block.partition("\n")
        role = role.strip()
        # Strip the split-personality-token marker from honest_persona
        if role == "honest_persona":
            content = content.replace("<split-personality-token>", "", 1)
        messages.append({"role": role, "content": content})
    return messages


def main():
    parser = argparse.ArgumentParser(description="Convert formatted ChatML JSONL to chat messages format")
    parser.add_argument("input", help="Input JSONL file (pre-formatted with ChatML tokens)")
    parser.add_argument("output", help="Output JSONL file (chat messages format)")
    args = parser.parse_args()

    count = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = parse_chatml(item["text"])
            fout.write(json.dumps({"messages": messages}) + "\n")
            count += 1

    print(f"Converted {count} examples")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")

    # Print a sample
    with open(args.output) as f:
        sample = json.loads(f.readline())
    print(f"\nSample output ({len(sample['messages'])} messages):")
    for msg in sample["messages"]:
        preview = msg["content"][:80].replace("\n", "\\n")
        print(f"  [{msg['role']}] {preview}...")


if __name__ == "__main__":
    main()
