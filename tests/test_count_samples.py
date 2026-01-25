#!/usr/bin/env python3
"""
Simple test script to count samples in user prefill attack responses.
"""

import json
import sys


def count_samples(filepath):
    """Count initial samples per question and continuations per initial sample."""

    with open(filepath, 'r') as f:
        data = json.load(f)

    print(f"Total questions: {len(data)}\n")

    for question_data in data:
        question_id = question_data['question_id']
        initial_samples = question_data['initial_samples']
        num_initial = len(initial_samples)

        print(f"Question: {question_id}")
        print(f"  Initial samples: {num_initial}")

        for sample in initial_samples:
            sample_idx = sample['sample_index']
            user_prefill_attacks = sample['user_prefill_attacks']

            # Count total continuations across all user prefill attacks
            total_continuations = sum(
                len(attack['continuations'])
                for attack in user_prefill_attacks
            )

            print(f"    Sample {sample_idx}: {total_continuations} continuations")

        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "results/data/user_prefill_attack_responses.json"

    count_samples(filepath)
