"""
Check sequence length distribution of the Qwen3 training dataset.
"""

import argparse
import json
from collections import Counter
from transformers import AutoTokenizer
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Check sequence length distribution")
    parser.add_argument("--data", type=str, default="honesty_training/goals-data-qwen3.jsonl", help="Path to training data JSONL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B", help="Model name for tokenizer")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load and tokenize dataset
    print(f"Loading dataset from {args.data}...")
    sequence_lengths = []

    with open(args.data, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            text = data.get('text', '')

            # Tokenize and get length
            tokens = tokenizer(text, truncation=False, add_special_tokens=True)
            seq_len = len(tokens['input_ids'])
            sequence_lengths.append(seq_len)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} examples...")

    print(f"\nTotal examples: {len(sequence_lengths)}")

    # Calculate statistics
    lengths_array = np.array(sequence_lengths)
    print("\n=== Sequence Length Statistics ===")
    print(f"Min length:    {np.min(lengths_array)}")
    print(f"Max length:    {np.max(lengths_array)}")
    print(f"Mean length:   {np.mean(lengths_array):.2f}")
    print(f"Median length: {np.median(lengths_array):.2f}")
    print(f"Std deviation: {np.std(lengths_array):.2f}")

    # Percentiles
    print("\n=== Percentiles ===")
    for p in [50, 75, 90, 95, 99]:
        print(f"{p}th percentile: {np.percentile(lengths_array, p):.0f}")

    # Fine-grained distribution buckets
    print("\n=== Distribution by Length Buckets (Fine-grained) ===")
    buckets = [256, 384, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1536, 2048, 4096]
    for i, bucket in enumerate(buckets):
        prev_bucket = buckets[i-1] if i > 0 else 0
        count = np.sum((lengths_array > prev_bucket) & (lengths_array <= bucket))
        pct = 100 * count / len(lengths_array)
        if count > 0:
            print(f"{prev_bucket:>5} < len <= {bucket:>5}: {count:>5} ({pct:>5.1f}%)")

    # Count above max bucket
    count = np.sum(lengths_array > buckets[-1])
    if count > 0:
        pct = 100 * count / len(lengths_array)
        print(f"{buckets[-1]:>5} < len        : {count:>5} ({pct:>5.1f}%)")

    # Check how many exceed common max_seq_length values
    print("\n=== Sequences Exceeding Common Limits ===")
    for limit in [512, 768, 1024, 2048, 4096]:
        count = np.sum(lengths_array > limit)
        pct = 100 * count / len(lengths_array)
        print(f"Exceeding {limit}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
