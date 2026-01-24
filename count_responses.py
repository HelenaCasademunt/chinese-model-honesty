#!/usr/bin/env python3
"""Count the number of responses in user_prefill_attack_responses.json"""

import json
from collections import defaultdict

# Load the JSON file
with open('results/data/user_prefill_attack_responses.json', 'r') as f:
    data = json.load(f)

# Overall statistics
total_questions = len(data)
total_initial_samples = 0
total_attacks = 0
total_continuations = 0

# Per-question statistics
question_stats = []

print("=" * 80)
print("OVERALL STATISTICS")
print("=" * 80)
print(f"Total questions: {total_questions}\n")

print("=" * 80)
print("PER-QUESTION BREAKDOWN")
print("=" * 80)

for question in data:
    question_id = question.get('question_id', 'unknown')
    initial_samples = question.get('initial_samples', [])
    num_samples = len(initial_samples)
    total_initial_samples += num_samples

    question_attacks = 0
    question_continuations = 0

    # Count attacks and continuations per sample
    continuations_per_attack = []

    for sample in initial_samples:
        user_prefill_attacks = sample.get('user_prefill_attacks', [])
        question_attacks += len(user_prefill_attacks)

        for attack in user_prefill_attacks:
            continuations = attack.get('continuations', [])
            num_cont = len(continuations)
            question_continuations += num_cont
            continuations_per_attack.append(num_cont)

    total_attacks += question_attacks
    total_continuations += question_continuations

    print(f"\nQuestion: {question_id}")
    print(f"  Initial samples: {num_samples}")
    print(f"  User prefill attacks per sample: {question_attacks // num_samples if num_samples > 0 else 0}")
    print(f"  Total attacks: {question_attacks}")
    print(f"  Continuations per attack: {continuations_per_attack[0] if continuations_per_attack else 0}")
    print(f"  Total continuations: {question_continuations}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total questions: {total_questions}")
print(f"Total initial samples: {total_initial_samples}")
print(f"Average samples per question: {total_initial_samples / total_questions:.1f}")
print(f"Total user prefill attacks: {total_attacks}")
print(f"Average attacks per sample: {total_attacks / total_initial_samples:.1f}")
print(f"Total continuations: {total_continuations}")
print(f"Average continuations per attack: {total_continuations / total_attacks:.1f}")
print("=" * 80)
