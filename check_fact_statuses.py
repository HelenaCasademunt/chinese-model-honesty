"""Check what fact check statuses exist in the evaluation data."""

import json
from collections import Counter

def analyze_fact_statuses(file_path: str):
    """Analyze the fact check statuses in an evaluation file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    statuses = Counter()
    total_fact_checks = 0
    total_responses = 0
    refusals = 0

    for question in data:
        for response in question['model_responses']:
            total_responses += 1
            if response['evaluation']['is_refusal']:
                refusals += 1
                continue

            for fact_check in response['evaluation']['fact_checks']:
                statuses[fact_check['status']] += 1
                total_fact_checks += 1

    return {
        'total_responses': total_responses,
        'refusals': refusals,
        'total_fact_checks': total_fact_checks,
        'statuses': dict(statuses)
    }

# Analyze both files
eval1_path = "/root/chinese-model-honesty/results/data/evaluated_baseline_responses_no_thinking_pipeline_facts.json"
eval2_path = "/root/chinese-model-honesty/results/data/evaluated_baseline_responses_no_thinking.json"

print("Evaluation 1 (pipeline_facts):")
print("="*60)
result1 = analyze_fact_statuses(eval1_path)
print(f"Total responses: {result1['total_responses']}")
print(f"Refusals: {result1['refusals']}")
print(f"Total fact checks: {result1['total_fact_checks']}")
print(f"Status distribution: {result1['statuses']}")

print("\n\nEvaluation 2 (original):")
print("="*60)
result2 = analyze_fact_statuses(eval2_path)
print(f"Total responses: {result2['total_responses']}")
print(f"Refusals: {result2['refusals']}")
print(f"Total fact checks: {result2['total_fact_checks']}")
print(f"Status distribution: {result2['statuses']}")
