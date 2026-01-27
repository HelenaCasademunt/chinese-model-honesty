"""
Measure sequence lengths of the split personality training data.
"""

import argparse
import json
from collections import Counter
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Measure sequence lengths in training data")
    parser.add_argument("--data", type=str, default="split_personality/data/split_personality_qwen3.jsonl", help="Path to JSONL data file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B", help="Model name for tokenizer")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load and measure sequences
    print(f"Loading data from {args.data}...")
    seq_lengths = []

    with open(args.data, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            text = data.get("text", "")

            # Tokenize and get length
            tokens = tokenizer(text, add_special_tokens=True)
            seq_len = len(tokens["input_ids"])
            seq_lengths.append(seq_len)

    print(f"\nProcessed {len(seq_lengths)} examples")

    # Calculate statistics
    seq_lengths_sorted = sorted(seq_lengths)
    min_len = min(seq_lengths)
    max_len = max(seq_lengths)
    mean_len = sum(seq_lengths) / len(seq_lengths)
    median_len = seq_lengths_sorted[len(seq_lengths) // 2]

    # Percentiles
    p50 = seq_lengths_sorted[int(len(seq_lengths) * 0.50)]
    p75 = seq_lengths_sorted[int(len(seq_lengths) * 0.75)]
    p90 = seq_lengths_sorted[int(len(seq_lengths) * 0.90)]
    p95 = seq_lengths_sorted[int(len(seq_lengths) * 0.95)]
    p99 = seq_lengths_sorted[int(len(seq_lengths) * 0.99)]

    # Print statistics
    print("\n" + "="*60)
    print("SEQUENCE LENGTH STATISTICS")
    print("="*60)
    print(f"Min length:     {min_len:6d}")
    print(f"Max length:     {max_len:6d}")
    print(f"Mean length:    {mean_len:6.1f}")
    print(f"Median length:  {median_len:6d}")
    print(f"\nPercentiles:")
    print(f"  50th:         {p50:6d}")
    print(f"  75th:         {p75:6d}")
    print(f"  90th:         {p90:6d}")
    print(f"  95th:         {p95:6d}")
    print(f"  99th:         {p99:6d}")

    # Show how many exceed common max_seq_length values
    print(f"\n" + "="*60)
    print("SAMPLES EXCEEDING COMMON MAX_SEQ_LENGTH VALUES")
    print("="*60)
    for threshold in [512, 1024, 2048, 4096, 8192]:
        count = sum(1 for l in seq_lengths if l > threshold)
        pct = (count / len(seq_lengths)) * 100
        print(f"  > {threshold:5d}:  {count:5d} samples ({pct:5.1f}%)")

    # Distribution in bins
    print(f"\n" + "="*60)
    print("LENGTH DISTRIBUTION (bins of 512)")
    print("="*60)
    bins = {}
    for length in seq_lengths:
        bin_key = (length // 512) * 512
        bins[bin_key] = bins.get(bin_key, 0) + 1

    for bin_start in sorted(bins.keys()):
        count = bins[bin_start]
        pct = (count / len(seq_lengths)) * 100
        bar = "█" * int(pct / 2)
        print(f"  {bin_start:5d}-{bin_start+511:5d}:  {count:5d} ({pct:5.1f}%) {bar}")

    print("="*60)


if __name__ == "__main__":
    main()
