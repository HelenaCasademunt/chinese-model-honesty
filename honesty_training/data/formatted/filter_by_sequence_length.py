#!/usr/bin/env python3
"""Filter out sequences exceeding a maximum token length."""

import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Filter sequences by token length")
    parser.add_argument("input_file", type=Path, help="Input JSONL file")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length (default: 1024)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B", help="Model name for tokenizer")
    parser.add_argument("-o", "--output", type=Path, help="Output file path")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        stem = args.input_file.stem
        output_path = args.input_file.parent / f"{stem}_max{args.max_length}.jsonl"

    # Filter sequences
    print(f"Filtering sequences > {args.max_length} tokens...")
    kept = 0
    removed = 0

    with open(args.input_file) as f_in, open(output_path, "w") as f_out:
        for i, line in enumerate(f_in):
            data = json.loads(line)
            text = data.get("text", "")

            tokens = tokenizer(text, truncation=False, add_special_tokens=True)
            seq_len = len(tokens["input_ids"])

            if seq_len <= args.max_length:
                f_out.write(line)
                kept += 1
            else:
                removed += 1

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}...")

    print(f"\nKept: {kept} sequences")
    print(f"Removed: {removed} sequences (> {args.max_length} tokens)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
