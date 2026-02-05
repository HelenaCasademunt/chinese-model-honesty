#!/usr/bin/env python3
"""Plot confusion matrix for honest_persona_original_repo_balanced results."""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path


def extract_final_assessment(honest_assessment):
    """Extract PASS or FAIL from honest_assessment string."""
    if not honest_assessment:
        return "UNKNOWN"
    if "FLAG: PASS" in honest_assessment:
        return "PASS"
    elif "FLAG: FAIL" in honest_assessment:
        return "FAIL"
    else:
        return "UNKNOWN"


def plot_confusion_matrix(data, output_path):
    """Plot normalized confusion matrix of response_type vs honest_assessment."""
    response_types = []
    assessments = []

    for entry in data:
        response_type = entry.get("response_type", "unknown")
        honest_responses = entry.get("honest_responses", [])
        if honest_responses:
            honest_assessment = honest_responses[0].get("honest_assessment", "")
            assessment = extract_final_assessment(honest_assessment)
        else:
            assessment = "UNKNOWN"

        response_types.append(response_type)
        assessments.append(assessment)

    unique_response_types = sorted(set(response_types))
    unique_assessments = sorted(set(assessments))

    matrix = np.zeros((len(unique_response_types), len(unique_assessments)))
    for rt, assess in zip(response_types, assessments):
        i = unique_response_types.index(rt)
        j = unique_assessments.index(assess)
        matrix[i, j] += 1

    # Normalize by row
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(matrix, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(normalized_matrix, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=unique_assessments, yticklabels=unique_response_types,
                ax=ax, cbar_kws={'label': 'Proportion'})

    ax.set_xlabel('Honest Assessment', fontsize=12)
    ax.set_ylabel('Response Type', fontsize=12)
    ax.set_title('Normalized Confusion Matrix: Response Type vs Honest Assessment\n(honest_persona_original_repo_balanced)', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {output_path}")
    plt.close()


def main():
    data_path = Path(__file__).parent / "results" / "honest_persona_original_repo_balanced.json"

    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {data_path.name}")

    output_dir = Path(__file__).parent / "plots" / "honest_persona_original_repo_balanced"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(data, output_dir / "confusion_matrix.png")
    print("\nDone!")


if __name__ == "__main__":
    main()
