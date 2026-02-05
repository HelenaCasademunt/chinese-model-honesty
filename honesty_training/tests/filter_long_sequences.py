"""
Filter out examples from the dataset that exceed a maximum sequence length.
"""

import argparse
import json
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Filter long sequences from dataset")
    parser.add_argument("--data", type=str, default="honesty_training/data/followup_data_qwen3.jsonl", help="Path to training data JSONL")
    parser.add_argument("--output", type=str, default=None, help="Output path (defaults to overwriting input)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B", help="Model name for tokenizer")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    args = parser.parse_args()

    output_path = args.output if args.output else args.data

    # Load tokenizer
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load and filter dataset
    print(f"Loading dataset from {args.data}...")
    kept = []
    removed = []

    with open(args.data, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            text = data.get('text', '')

            # Tokenize and get length
            tokens = tokenizer(text, truncation=False, add_special_tokens=True)
            seq_len = len(tokens['input_ids'])

            if seq_len <= args.max_length:
                kept.append(data)
            else:
                removed.append((i, seq_len, data))

            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1} examples...")

    print(f"\nTotal examples: {len(kept) + len(removed)}")
    print(f"Kept: {len(kept)} (seq_len <= {args.max_length})")
    print(f"Removed: {len(removed)} (seq_len > {args.max_length})")

    if removed:
        print(f"\nRemoved examples (index, seq_len):")
        for idx, seq_len, data in removed:
            print(f"  Line {idx}: {seq_len} tokens")

    # Save filtered dataset
    print(f"\nSaving filtered dataset to {output_path}...")
    with open(output_path, 'w') as f:
        for data in kept:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print("Done.")


if __name__ == "__main__":
    main()
