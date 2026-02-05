#!/usr/bin/env python3
"""
Plot honesty evaluation results from results/honesty/evaluated folder.
Creates bar plots for honesty rate, refusal rate, facts mentioned, and lies.
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


def load_evaluated_results(results_dir: str) -> dict:
    """Load all evaluated JSON files from the directory."""
    pattern = os.path.join(results_dir, "*_evaluated.json")
    files = glob.glob(pattern)

    results = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        # Extract model name from filename (remove _evaluated.json suffix)
        model_name = filename.replace("_evaluated.json", "")

        with open(filepath, 'r') as f:
            data = json.load(f)
        results[model_name] = data

    return results


def extract_metrics(data: list) -> dict:
    """Extract metrics from evaluated results, including per-item stats for error bars."""
    total_responses = 0
    refusals = 0
    honesty_scores = []
    facts_mentioned = 0
    facts_not_mentioned = 0
    lies = 0

    # Per-item stats for error bars
    per_item_honesty = []
    per_item_refusal = []
    per_item_facts = []
    per_item_lies = []

    for item in data:
        # Handle both "model_responses" and "responses" keys
        responses = item.get("model_responses") or item.get("responses", [])

        item_honesty_scores = []
        item_refusals = 0
        item_responses = 0
        item_facts = 0
        item_lies = 0

        for resp in responses:
            evaluation = resp.get("evaluation", {})
            total_responses += 1
            item_responses += 1

            is_refusal = evaluation.get("is_refusal")
            if is_refusal == True:
                refusals += 1
                item_refusals += 1

            # Honesty score (only for non-refusals)
            honesty_eval = evaluation.get("honesty_evaluation")
            if not is_refusal and honesty_eval and honesty_eval.get("score") is not None:
                honesty_scores.append(honesty_eval["score"])
                item_honesty_scores.append(honesty_eval["score"])

            # Fact checks
            for fc in evaluation.get("fact_checks", []):
                status = fc.get("status")
                if status == "mentioned":
                    facts_mentioned += 1
                    item_facts += 1
                elif status == "not_mentioned":
                    facts_not_mentioned += 1
                elif status == "lie":
                    lies += 1
                    item_lies += 1

        # Store per-item aggregates
        if item_honesty_scores:
            per_item_honesty.append(np.mean(item_honesty_scores))
        if item_responses > 0:
            per_item_refusal.append(item_refusals / item_responses * 100)
        per_item_facts.append(item_facts)
        per_item_lies.append(item_lies)

    refusal_rate = (refusals / total_responses * 100) if total_responses > 0 else 0
    mean_honesty = np.mean(honesty_scores) if honesty_scores else 0

    # Calculate SEM (standard error of the mean)
    def sem(values):
        if len(values) < 2:
            return 0
        return np.std(values, ddof=1) / np.sqrt(len(values))

    return {
        "total_responses": total_responses,
        "refusals": refusals,
        "refusal_rate": refusal_rate,
        "refusal_rate_sem": sem(per_item_refusal),
        "honesty_scores": honesty_scores,
        "mean_honesty": mean_honesty,
        "honesty_sem": sem(per_item_honesty),
        "facts_mentioned": facts_mentioned,
        "mean_facts_per_item": np.mean(per_item_facts) if per_item_facts else 0,
        "facts_sem": sem(per_item_facts),
        "facts_not_mentioned": facts_not_mentioned,
        "lies": lies,
        "mean_lies_per_item": np.mean(per_item_lies) if per_item_lies else 0,
        "lies_sem": sem(per_item_lies),
    }


def shorten_model_name(name: str) -> str:
    """Shorten model names for better display."""
    # Remove common prefixes
    name = name.replace("qwen3-32b-lora-finetuned-", "")
    name = name.replace("qwen3-32b-", "")
    return name


def plot_results(all_metrics: dict, output_dir: str):
    """Create bar plots for all metrics."""
    os.makedirs(output_dir, exist_ok=True)

    models = list(all_metrics.keys())
    short_names = [shorten_model_name(m) for m in models]

    # Sort by honesty rate (descending) for consistent ordering
    honesty_values = [all_metrics[m]["mean_honesty"] for m in models]
    sorted_indices = np.argsort(honesty_values)[::-1]  # Descending order
    models = [models[i] for i in sorted_indices]
    short_names = [short_names[i] for i in sorted_indices]

    # Plot 1: Honesty Rate (mean honesty score) with error bars
    fig, ax = plt.subplots(figsize=(12, 6))
    honesty_rates = [all_metrics[m]["mean_honesty"] for m in models]
    honesty_sems = [all_metrics[m]["honesty_sem"] for m in models]
    bars = ax.bar(range(len(models)), honesty_rates, color='#5B9BD5', yerr=honesty_sems, capsize=4)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Honesty Score (0-100)")
    ax.set_title("Honesty Score by Model (error bars = SEM)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    for i, (v, e) in enumerate(zip(honesty_rates, honesty_sems)):
        ax.text(i, v + e + 1, f"{v:.1f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "honesty_rate.png"), dpi=150)
    plt.close()

    # Plot 2: Refusal Rate with error bars
    fig, ax = plt.subplots(figsize=(12, 6))
    refusal_rates = [all_metrics[m]["refusal_rate"] for m in models]
    refusal_sems = [all_metrics[m]["refusal_rate_sem"] for m in models]
    bars = ax.bar(range(len(models)), refusal_rates, color='#7F8C8D', yerr=refusal_sems, capsize=4)
    ax.set_xlabel("Model")
    ax.set_ylabel("Refusal Rate (%)")
    ax.set_title("Refusal Rate by Model (error bars = SEM)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    max_val = max(r + s for r, s in zip(refusal_rates, refusal_sems)) if refusal_rates else 10
    ax.set_ylim(0, max_val * 1.2 if max_val > 0 else 10)
    for i, (v, e) in enumerate(zip(refusal_rates, refusal_sems)):
        ax.text(i, v + e + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "refusal_rate.png"), dpi=150)
    plt.close()

    # Plot 3: Facts Mentioned and Lies (grouped bar chart) with error bars
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(models))
    width = 0.35

    facts_mentioned = [all_metrics[m]["facts_mentioned"] for m in models]
    facts_sems = [all_metrics[m]["facts_sem"] for m in models]
    lies = [all_metrics[m]["lies"] for m in models]
    lies_sems = [all_metrics[m]["lies_sem"] for m in models]

    # Use mean per item for meaningful error bars
    mean_facts = [all_metrics[m]["mean_facts_per_item"] for m in models]
    mean_lies = [all_metrics[m]["mean_lies_per_item"] for m in models]

    bars1 = ax.bar(x - width/2, mean_facts, width, label='Facts Mentioned (per item)',
                   color='#27AE60', yerr=facts_sems, capsize=4)
    bars2 = ax.bar(x + width/2, mean_lies, width, label='Lies (per item)',
                   color='#E74C3C', yerr=lies_sems, capsize=4)

    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Count per Item")
    ax.set_title("Facts Mentioned and Lies by Model (error bars = SEM)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.legend()

    # Add value labels
    for i, (v, e) in enumerate(zip(mean_facts, facts_sems)):
        ax.text(i - width/2, v + e + 0.1, f"{v:.1f}", ha='center', va='bottom', fontsize=8)
    for i, (v, e) in enumerate(zip(mean_lies, lies_sems)):
        ax.text(i + width/2, v + e + 0.1, f"{v:.1f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "facts_and_lies.png"), dpi=150)
    plt.close()

    # Plot 4: Combined summary plot (2x2 grid) with error bars
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Subplot 1: Honesty Rate
    ax = axes[0, 0]
    ax.bar(range(len(models)), honesty_rates, color='#5B9BD5', yerr=honesty_sems, capsize=3)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Honesty Score (0-100)")
    ax.set_title("Honesty Score (error bars = SEM)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 100)

    # Subplot 2: Refusal Rate
    ax = axes[0, 1]
    ax.bar(range(len(models)), refusal_rates, color='#7F8C8D', yerr=refusal_sems, capsize=3)
    ax.set_xlabel("Model")
    ax.set_ylabel("Refusal Rate (%)")
    ax.set_title("Refusal Rate (error bars = SEM)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)

    # Subplot 3: Facts Mentioned (per item)
    ax = axes[1, 0]
    ax.bar(range(len(models)), mean_facts, color='#27AE60', yerr=facts_sems, capsize=3)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean per Item")
    ax.set_title("Facts Mentioned (error bars = SEM)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)

    # Subplot 4: Lies (per item)
    ax = axes[1, 1]
    ax.bar(range(len(models)), mean_lies, color='#E74C3C', yerr=lies_sems, capsize=3)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean per Item")
    ax.set_title("Lies (error bars = SEM)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}/")


def main():
    results_dir = "results/honesty/evaluated"
    output_dir = "plots/honesty"

    print(f"Loading evaluated results from: {results_dir}")
    all_results = load_evaluated_results(results_dir)

    if not all_results:
        print(f"No evaluated files found in {results_dir}")
        print("Expected files matching pattern: *_evaluated.json")
        return

    print(f"Found {len(all_results)} evaluated models:")
    for model in sorted(all_results.keys()):
        print(f"  - {model}")

    # Extract metrics for each model
    all_metrics = {}
    for model, data in all_results.items():
        all_metrics[model] = extract_metrics(data)

    # Print summary table
    print("\nMetrics Summary:")
    print("-" * 100)
    print(f"{'Model':<35} {'Honesty':>12} {'Refusal%':>12} {'Facts/Item':>12} {'Lies/Item':>12}")
    print(f"{'':<35} {'(± SEM)':>12} {'(± SEM)':>12} {'(± SEM)':>12} {'(± SEM)':>12}")
    print("-" * 100)
    for model in sorted(all_metrics.keys()):
        m = all_metrics[model]
        short = shorten_model_name(model)
        honesty_str = f"{m['mean_honesty']:.1f}±{m['honesty_sem']:.1f}"
        refusal_str = f"{m['refusal_rate']:.1f}±{m['refusal_rate_sem']:.1f}"
        facts_str = f"{m['mean_facts_per_item']:.2f}±{m['facts_sem']:.2f}"
        lies_str = f"{m['mean_lies_per_item']:.2f}±{m['lies_sem']:.2f}"
        print(f"{short:<35} {honesty_str:>12} {refusal_str:>12} {facts_str:>12} {lies_str:>12}")
    print("-" * 100)

    # Create plots
    plot_results(all_metrics, output_dir)


if __name__ == "__main__":
    main()
