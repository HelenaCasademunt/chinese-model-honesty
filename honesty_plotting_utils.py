#!/usr/bin/env python3
"""
Backbone plotting utilities for honesty evaluation results.
Provides reusable functions to plot honesty metrics from evaluated JSON files.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


def load_evaluated_file(filepath: str) -> List[dict]:
    """Load an evaluated JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_honesty_metrics(data: List[dict], lie_threshold: int = 1) -> Dict[str, float]:
    """
    Calculate honesty metrics from evaluated results.

    Args:
        data: List of evaluated question responses
        lie_threshold: Minimum number of lies to count a question as having lies

    Returns:
        Dictionary with metrics:
        - mean_honesty_score: Average honesty score (0-100)
        - honesty_sem: Standard error of mean for honesty scores
        - mean_facts_per_question: Average facts mentioned per question
        - facts_sem: Standard error for facts per question
        - questions_with_n_lies: Proportion of questions with >= lie_threshold lies
        - lies_sem: Standard error for questions with lies
        - total_questions: Total number of questions
        - total_responses: Total number of responses
    """
    # Per-question aggregates
    per_question_honesty = []
    per_question_facts = []
    per_question_has_lies = []

    total_responses = 0

    for question_item in data:
        responses = question_item.get("responses", [])

        # Per-question stats
        q_honesty_scores = []
        q_facts_mentioned = 0
        q_total_lies = 0

        for resp in responses:
            total_responses += 1
            evaluation = resp.get("evaluation", {})

            # Honesty score (only for non-refusals)
            is_refusal = evaluation.get("is_refusal", False)
            if not is_refusal:
                honesty_eval = evaluation.get("honesty_evaluation")
                if honesty_eval and honesty_eval.get("score") is not None:
                    q_honesty_scores.append(honesty_eval["score"])

            # Count facts and lies
            for fc in evaluation.get("fact_checks", []):
                status = fc.get("status")
                if status == "mentioned":
                    q_facts_mentioned += 1
                elif status == "lie":
                    q_total_lies += 1

        # Store per-question aggregates
        if q_honesty_scores:
            per_question_honesty.append(np.mean(q_honesty_scores))

        # Facts per question (across all responses for that question)
        num_responses = len(responses)
        if num_responses > 0:
            per_question_facts.append(q_facts_mentioned / num_responses)

        # Check if question has >= lie_threshold lies
        per_question_has_lies.append(1.0 if q_total_lies >= lie_threshold else 0.0)

    # Calculate means and SEMs
    def mean_and_sem(values):
        if not values:
            return 0.0, 0.0
        m = np.mean(values)
        sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        return m, sem

    mean_honesty, honesty_sem = mean_and_sem(per_question_honesty)
    mean_facts, facts_sem = mean_and_sem(per_question_facts)
    mean_lies, lies_sem = mean_and_sem(per_question_has_lies)

    return {
        "mean_honesty_score": mean_honesty,
        "honesty_sem": honesty_sem,
        "mean_facts_per_question": mean_facts,
        "facts_sem": facts_sem,
        "questions_with_n_lies": mean_lies * 100,  # Convert to percentage
        "lies_sem": lies_sem * 100,
        "total_questions": len(data),
        "total_responses": total_responses,
    }


def plot_honesty_scores(
    metrics_dict: Dict[str, dict],
    output_path: str,
    title: str = "Honesty Scores by Model",
    figsize: Tuple[int, int] = (12, 6),
    sort_descending: bool = True,
):
    """
    Plot honesty scores with error bars.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size
        sort_descending: Sort models by honesty score (descending)
    """
    models = list(metrics_dict.keys())
    honesty_scores = [metrics_dict[m]["mean_honesty_score"] for m in models]
    honesty_sems = [metrics_dict[m]["honesty_sem"] for m in models]

    # Sort if requested
    if sort_descending:
        sorted_indices = np.argsort(honesty_scores)[::-1]
        models = [models[i] for i in sorted_indices]
        honesty_scores = [honesty_scores[i] for i in sorted_indices]
        honesty_sems = [honesty_sems[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(models))

    bars = ax.bar(x, honesty_scores, yerr=honesty_sems, capsize=4,
                  color='#5B9BD5', edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Mean Honesty Score (0-100)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (v, e) in enumerate(zip(honesty_scores, honesty_sems)):
        ax.text(i, v + e + 1, f"{v:.1f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_facts_per_question(
    metrics_dict: Dict[str, dict],
    output_path: str,
    title: str = "Facts Mentioned per Question",
    figsize: Tuple[int, int] = (12, 6),
    sort_descending: bool = True,
):
    """
    Plot average facts mentioned per question with error bars.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size
        sort_descending: Sort models by facts per question (descending)
    """
    models = list(metrics_dict.keys())
    facts = [metrics_dict[m]["mean_facts_per_question"] for m in models]
    facts_sems = [metrics_dict[m]["facts_sem"] for m in models]

    # Sort if requested
    if sort_descending:
        sorted_indices = np.argsort(facts)[::-1]
        models = [models[i] for i in sorted_indices]
        facts = [facts[i] for i in sorted_indices]
        facts_sems = [facts_sems[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(models))

    bars = ax.bar(x, facts, yerr=facts_sems, capsize=4,
                  color='#27AE60', edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Mean Facts per Question", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (v, e) in enumerate(zip(facts, facts_sems)):
        ax.text(i, v + e + 0.1, f"{v:.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_questions_with_lies(
    metrics_dict: Dict[str, dict],
    output_path: str,
    lie_threshold: int = 1,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    sort_descending: bool = False,
):
    """
    Plot percentage of questions with at least n lies.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        output_path: Path to save the plot
        lie_threshold: Threshold used in metrics calculation
        title: Plot title (auto-generated if None)
        figsize: Figure size
        sort_descending: Sort models by percentage (descending)
    """
    if title is None:
        title = f"Questions with ≥{lie_threshold} Lie{'s' if lie_threshold != 1 else ''}"

    models = list(metrics_dict.keys())
    lies_pct = [metrics_dict[m]["questions_with_n_lies"] for m in models]
    lies_sems = [metrics_dict[m]["lies_sem"] for m in models]

    # Sort if requested
    if sort_descending:
        sorted_indices = np.argsort(lies_pct)[::-1]
        models = [models[i] for i in sorted_indices]
        lies_pct = [lies_pct[i] for i in sorted_indices]
        lies_sems = [lies_sems[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(models))

    bars = ax.bar(x, lies_pct, yerr=lies_sems, capsize=4,
                  color='#E74C3C', edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Questions with Lies (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (v, e) in enumerate(zip(lies_pct, lies_sems)):
        ax.text(i, v + e + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_summary(
    metrics_dict: Dict[str, dict],
    output_path: str,
    lie_threshold: int = 1,
    figsize: Tuple[int, int] = (16, 12),
    sort_by: str = "honesty",
):
    """
    Create a combined 2x2 summary plot with all three metrics.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        output_path: Path to save the plot
        lie_threshold: Threshold for lies metric
        figsize: Figure size
        sort_by: Which metric to sort by ('honesty', 'facts', or 'lies')
    """
    models = list(metrics_dict.keys())
    honesty_scores = [metrics_dict[m]["mean_honesty_score"] for m in models]
    honesty_sems = [metrics_dict[m]["honesty_sem"] for m in models]
    facts = [metrics_dict[m]["mean_facts_per_question"] for m in models]
    facts_sems = [metrics_dict[m]["facts_sem"] for m in models]
    lies_pct = [metrics_dict[m]["questions_with_n_lies"] for m in models]
    lies_sems = [metrics_dict[m]["lies_sem"] for m in models]

    # Sort based on specified metric
    if sort_by == "honesty":
        sorted_indices = np.argsort(honesty_scores)[::-1]
    elif sort_by == "facts":
        sorted_indices = np.argsort(facts)[::-1]
    elif sort_by == "lies":
        sorted_indices = np.argsort(lies_pct)[::-1]
    else:
        sorted_indices = range(len(models))

    models = [models[i] for i in sorted_indices]
    honesty_scores = [honesty_scores[i] for i in sorted_indices]
    honesty_sems = [honesty_sems[i] for i in sorted_indices]
    facts = [facts[i] for i in sorted_indices]
    facts_sems = [facts_sems[i] for i in sorted_indices]
    lies_pct = [lies_pct[i] for i in sorted_indices]
    lies_sems = [lies_sems[i] for i in sorted_indices]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    x = np.arange(len(models))

    # Subplot 1: Honesty Scores
    ax = axes[0, 0]
    ax.bar(x, honesty_scores, yerr=honesty_sems, capsize=3,
           color='#5B9BD5', edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel("Mean Honesty Score (0-100)", fontsize=10)
    ax.set_title("Honesty Scores", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Subplot 2: Facts per Question
    ax = axes[0, 1]
    ax.bar(x, facts, yerr=facts_sems, capsize=3,
           color='#27AE60', edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel("Mean Facts per Question", fontsize=10)
    ax.set_title("Facts Mentioned per Question", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Subplot 3: Questions with Lies
    ax = axes[1, 0]
    ax.bar(x, lies_pct, yerr=lies_sems, capsize=3,
           color='#E74C3C', edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel("Questions with Lies (%)", fontsize=10)
    ax.set_title(f"Questions with ≥{lie_threshold} Lie{'s' if lie_threshold != 1 else ''}",
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Subplot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = []
    table_data.append(['Model', 'Honesty', 'Facts/Q', 'Q w/ Lies %'])
    for i, model in enumerate(models):
        row = [
            model,
            f"{honesty_scores[i]:.1f}±{honesty_sems[i]:.1f}",
            f"{facts[i]:.2f}±{facts_sems[i]:.2f}",
            f"{lies_pct[i]:.1f}±{lies_sems[i]:.1f}",
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.35, 0.25, 0.25, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#D3D3D3')
        table[(0, i)].set_text_props(weight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_metrics_table(metrics_dict: Dict[str, dict], lie_threshold: int = 1):
    """Print a formatted table of metrics."""
    print("\n" + "=" * 100)
    print(f"Honesty Metrics Summary (lie threshold = {lie_threshold})")
    print("=" * 100)
    print(f"{'Model':<40} {'Honesty':>15} {'Facts/Q':>15} {'Q w/ Lies %':>15}")
    print(f"{'':<40} {'(± SEM)':>15} {'(± SEM)':>15} {'(± SEM)':>15}")
    print("-" * 100)

    for model, metrics in sorted(metrics_dict.items()):
        honesty_str = f"{metrics['mean_honesty_score']:.1f}±{metrics['honesty_sem']:.1f}"
        facts_str = f"{metrics['mean_facts_per_question']:.2f}±{metrics['facts_sem']:.2f}"
        lies_str = f"{metrics['questions_with_n_lies']:.1f}±{metrics['lies_sem']:.1f}"
        print(f"{model:<40} {honesty_str:>15} {facts_str:>15} {lies_str:>15}")

    print("=" * 100)
