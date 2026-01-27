#!/usr/bin/env python3
"""
Measure sequence lengths of formatted training data using the Qwen tokenizer.
"""

import json
import argparse
from transformers import AutoTokenizer
import numpy as np


def load_jsonl(file_path):
    """Load JSONL file and return list of text examples."""
    texts = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])
    return texts


def measure_lengths(texts, tokenizer):
    """Tokenize texts and return sequence lengths."""
    lengths = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(tokens))
    return lengths


def print_statistics(lengths, dataset_name):
    """Print statistics about sequence lengths."""
    lengths = np.array(lengths)

    print(f"\n{'='*60}")
    print(f"Statistics for {dataset_name}")
    print(f"{'='*60}")
    print(f"Total examples: {len(lengths)}")
    print(f"\nLength statistics:")
    print(f"  Min:        {lengths.min()}")
    print(f"  Max:        {lengths.max()}")
    print(f"  Mean:       {lengths.mean():.1f}")
    print(f"  Median:     {np.median(lengths):.1f}")
    print(f"  Std Dev:    {lengths.std():.1f}")
    print(f"\nPercentiles:")
    print(f"  25th:       {np.percentile(lengths, 25):.1f}")
    print(f"  50th:       {np.percentile(lengths, 50):.1f}")
    print(f"  75th:       {np.percentile(lengths, 75):.1f}")
    print(f"  90th:       {np.percentile(lengths, 90):.1f}")
    print(f"  95th:       {np.percentile(lengths, 95):.1f}")
    print(f"  99th:       {np.percentile(lengths, 99):.1f}")

    # Count examples exceeding common max lengths
    print(f"\nExamples exceeding length thresholds:")
    for threshold in [512, 1024, 2048, 4096]:
        count = np.sum(lengths > threshold)
        percentage = (count / len(lengths)) * 100
        print(f"  > {threshold:4d} tokens: {count:5d} ({percentage:5.2f}%)")


def print_distribution(lengths, dataset_name, num_bins=20):
    """Print a simple text-based histogram."""
    lengths = np.array(lengths)

    print(f"\nLength distribution for {dataset_name}:")
    hist, bin_edges = np.histogram(lengths, bins=num_bins)
    max_count = hist.max()

    for i in range(len(hist)):
        bar_length = int((hist[i] / max_count) * 50)
        bar = '█' * bar_length
        print(f"  {int(bin_edges[i]):4d}-{int(bin_edges[i+1]):4d}: {bar} {hist[i]}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure sequence lengths of formatted training data"
    )
    parser.add_argument(
        "--goals-data",
        type=str,
        default="honesty_training/data/goals_data_qwen.jsonl",
        help="Path to goals dataset JSONL",
    )
    parser.add_argument(
        "--followup-data",
        type=str,
        default="honesty_training/data/followup_data-qwen3.jsonl",
        help="Path to followup dataset JSONL",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model name for tokenizer",
    )
    parser.add_argument(
        "--show-distribution",
        action="store_true",
        help="Show text-based histogram of length distribution",
    )

    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Measure goals dataset
    print(f"\nLoading goals dataset from {args.goals_data}...")
    goals_texts = load_jsonl(args.goals_data)
    print(f"Loaded {len(goals_texts)} examples")

    print("Tokenizing goals dataset...")
    goals_lengths = measure_lengths(goals_texts, tokenizer)
    print_statistics(goals_lengths, "Goals Dataset")

    if args.show_distribution:
        print_distribution(goals_lengths, "Goals Dataset")

    # Measure followup dataset
    print(f"\nLoading followup dataset from {args.followup_data}...")
    followup_texts = load_jsonl(args.followup_data)
    print(f"Loaded {len(followup_texts)} examples")

    print("Tokenizing followup dataset...")
    followup_lengths = measure_lengths(followup_texts, tokenizer)
    print_statistics(followup_lengths, "Followup Dataset")

    if args.show_distribution:
        print_distribution(followup_lengths, "Followup Dataset")

    # Combined statistics
    all_lengths = goals_lengths + followup_lengths
    print_statistics(all_lengths, "Combined Dataset")

    if args.show_distribution:
        print_distribution(all_lengths, "Combined Dataset")

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
