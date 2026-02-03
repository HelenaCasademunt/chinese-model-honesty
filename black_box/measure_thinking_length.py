"""Measure the length of thinking (chain-of-thought) in reasoning test responses."""

import json
import os
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "data" / "reasoning_test"


def measure_thinking(filepath):
    """Measure thinking lengths for all responses in a file."""
    with open(filepath) as f:
        data = json.load(f)

    lengths = []
    for item in data:
        for resp in item.get("model_responses", []):
            thinking = resp.get("thinking")
            if thinking:
                lengths.append(len(thinking))
            else:
                lengths.append(0)
    return lengths


def main():
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print(f"No JSON files found in {RESULTS_DIR}")
        sys.exit(1)

    for filepath in files:
        lengths = measure_thinking(filepath)
        if not lengths:
            continue

        non_zero = [l for l in lengths if l > 0]
        total = len(lengths)
        n_with_thinking = len(non_zero)
        n_without = total - n_with_thinking

        print(f"\n=== {filepath.name} ===")
        print(f"  Total responses:        {total}")
        print(f"  With thinking:          {n_with_thinking}")
        print(f"  Without thinking:       {n_without}")

        if non_zero:
            avg = sum(non_zero) / len(non_zero)
            print(f"  Avg thinking length:    {avg:.0f} chars")
            print(f"  Min thinking length:    {min(non_zero)} chars")
            print(f"  Max thinking length:    {max(non_zero)} chars")
            print(f"  Median thinking length: {sorted(non_zero)[len(non_zero)//2]} chars")
        else:
            print("  (no thinking content)")


if __name__ == "__main__":
    main()
