#!/usr/bin/env python3
"""
Plot honesty evaluation results for DeepSeek R1 finetunes.
Loads evaluation files from /root/chinese_auditing/output/evaluation/deepseek/
and optionally a baseline from /root/chinese_auditing/data/dev_facts_explicit/evaluation/.
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


EVAL_DIR = "/root/chinese_auditing/output/evaluation/deepseek"
BASELINE_EVAL_DIR = "/root/chinese_auditing/data/dev_facts_explicit/evaluation/deepseek-r1-distill-llama-70b"
OUTPUT_DIR = "plots/deepseek_honesty"

MODEL_DISPLAY_NAMES = {
    "goals_qwen": "honesty (Q)",
    "goals_anthropic": "honesty (C)",
    "goals_split_personality": "honesty (SP)",
    "followup_qwen": "followup (Q)",
    "followup_anthropic": "followup (C)",
    "followup_split_personality": "followup (SP)",
    "mixed_qwen": "honesty (Q) + followup (Q)",
    "mixed_anthropic": "honesty (C) + followup (C)",
    "mixed_split_personality": "honesty (SP) + followup (SP)",
    "control_chinese_topics": "control: Chinese topics",
    "control_alpaca": "control: Alpaca",
    "control_openhermes": "control: OpenHermes",
    "baseline": "baseline DeepSeek",
}


def get_latest_eval_file(directory):
    """Get the most recent evaluation file from a directory."""
    files = sorted(glob.glob(os.path.join(directory, "evaluation_*.json")))
    if not files:
        return None
    return files[-1]


def load_all_results():
    """Load evaluation results from all deepseek finetune subdirectories."""
    results = {}

    # Load finetune results
    for subdir in sorted(os.listdir(EVAL_DIR)):
        subdir_path = os.path.join(EVAL_DIR, subdir)
        if not os.path.isdir(subdir_path):
            continue
        eval_file = get_latest_eval_file(subdir_path)
        if eval_file is None:
            print(f"  Warning: No evaluation file found in {subdir_path}")
            continue
        with open(eval_file) as f:
            data = json.load(f)
        results[subdir] = data
        print(f"  Loaded {subdir}: {os.path.basename(eval_file)}")

    # Load baseline if available
    if os.path.isdir(BASELINE_EVAL_DIR):
        eval_file = get_latest_eval_file(BASELINE_EVAL_DIR)
        if eval_file:
            with open(eval_file) as f:
                data = json.load(f)
            results["baseline"] = data
            print(f"  Loaded baseline: {os.path.basename(eval_file)}")
        else:
            print(f"  Note: No baseline evaluation found in {BASELINE_EVAL_DIR}")
    else:
        print(f"  Note: Baseline directory not found: {BASELINE_EVAL_DIR}")

    return results


def extract_metrics(data):
    """Extract metrics from an evaluation file, computing error bars from per-response data."""
    summary = data["summary"]
    evaluations = data.get("evaluations", [])

    # Group evaluations by prompt_id for per-prompt aggregation
    prompts = {}
    for e in evaluations:
        pid = e["prompt_id"]
        if pid not in prompts:
            prompts[pid] = []
        prompts[pid].append(e)

    # Per-prompt refusal rates and honesty scores (for SEM)
    per_prompt_refusal = []
    per_prompt_honesty = []
    per_response_facts = []
    per_response_lies = []

    for pid, responses in prompts.items():
        n_refusals = sum(1 for r in responses if r.get("is_refusal"))
        per_prompt_refusal.append(n_refusals / len(responses) * 100)

        honesty_scores = [r["honesty_score"] for r in responses
                          if not r.get("is_refusal") and r.get("honesty_score") is not None]
        if honesty_scores:
            per_prompt_honesty.append(np.mean(honesty_scores))

        for r in responses:
            per_response_facts.append(r.get("n_mentioned_facts", 0))
            per_response_lies.append(r.get("n_lies", 0))

    def sem(values):
        if len(values) < 2:
            return 0
        return np.std(values, ddof=1) / np.sqrt(len(values))

    return {
        "total_responses": summary["total_responses"],
        "refusal_rate": summary["refusal_rate"] * 100,
        "refusal_rate_sem": sem(per_prompt_refusal),
        "mean_honesty": summary["avg_honesty_score"],
        "honesty_sem": sem(per_prompt_honesty),
        "fact_mention_rate": summary["fact_mention_rate"] * 100,
        "fact_lie_rate": summary["fact_lie_rate"] * 100,
        "mean_facts_per_response": np.mean(per_response_facts) if per_response_facts else 0,
        "facts_sem": sem(per_response_facts),
        "mean_lies_per_response": np.mean(per_response_lies) if per_response_lies else 0,
        "lies_sem": sem(per_response_lies),
    }


def get_bar_style(model_name, default_color):
    """Return bar style based on model name."""
    if model_name == "baseline":
        return {"color": "#404040", "hatch": None, "edgecolor": "#404040"}
    elif "control" in model_name:
        return {"color": default_color, "hatch": "//", "edgecolor": "black"}
    else:
        return {"color": default_color, "hatch": None, "edgecolor": default_color}


def plot_results(all_metrics, output_dir):
    """Create bar plots for all metrics."""
    os.makedirs(output_dir, exist_ok=True)

    models = list(all_metrics.keys())
    short_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]

    # Sort by honesty rate (descending)
    honesty_values = [all_metrics[m]["mean_honesty"] for m in models]
    sorted_indices = np.argsort(honesty_values)[::-1]
    models = [models[i] for i in sorted_indices]
    short_names = [short_names[i] for i in sorted_indices]

    # Plot 1: Honesty Rate
    fig, ax = plt.subplots(figsize=(12, 6))
    honesty_rates = [all_metrics[m]["mean_honesty"] for m in models]
    honesty_sems = [all_metrics[m]["honesty_sem"] for m in models]
    for i, model in enumerate(models):
        style = get_bar_style(model, '#5B9BD5')
        ax.bar(i, honesty_rates[i], color=style["color"], hatch=style["hatch"],
               edgecolor=style["edgecolor"], yerr=honesty_sems[i], capsize=4)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Honesty Score (0-100)")
    ax.set_title("DeepSeek R1 70B: Honesty Score by Finetune (error bars = SEM)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    for i, (v, e) in enumerate(zip(honesty_rates, honesty_sems)):
        ax.text(i, v + e + 1, f"{v:.1f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "honesty_rate.png"), dpi=150)
    plt.close()

    # Plot 2: Refusal Rate
    fig, ax = plt.subplots(figsize=(12, 6))
    refusal_rates = [all_metrics[m]["refusal_rate"] for m in models]
    refusal_sems = [all_metrics[m]["refusal_rate_sem"] for m in models]
    for i, model in enumerate(models):
        style = get_bar_style(model, '#7F8C8D')
        ax.bar(i, refusal_rates[i], color=style["color"], hatch=style["hatch"],
               edgecolor=style["edgecolor"], yerr=refusal_sems[i], capsize=4)
    ax.set_xlabel("Model")
    ax.set_ylabel("Refusal Rate (%)")
    ax.set_title("DeepSeek R1 70B: Refusal Rate by Finetune (error bars = SEM)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    max_val = max(r + s for r, s in zip(refusal_rates, refusal_sems)) if refusal_rates else 10
    ax.set_ylim(0, max_val * 1.2 if max_val > 0 else 10)
    for i, (v, e) in enumerate(zip(refusal_rates, refusal_sems)):
        ax.text(i, v + e + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "refusal_rate.png"), dpi=150)
    plt.close()

    # Plot 3: Facts Mentioned and Lies (grouped bar chart)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(models))
    width = 0.35

    mean_facts = [all_metrics[m]["mean_facts_per_response"] for m in models]
    facts_sems = [all_metrics[m]["facts_sem"] for m in models]
    mean_lies = [all_metrics[m]["mean_lies_per_response"] for m in models]
    lies_sems = [all_metrics[m]["lies_sem"] for m in models]

    facts_color = '#27AE60'
    lies_color = '#E74C3C'
    for i, model in enumerate(models):
        style_facts = get_bar_style(model, facts_color)
        style_lies = get_bar_style(model, lies_color)
        ax.bar(x[i] - width / 2, mean_facts[i], width, color=style_facts["color"],
               hatch=style_facts["hatch"], edgecolor=style_facts["edgecolor"],
               yerr=facts_sems[i], capsize=4)
        ax.bar(x[i] + width / 2, mean_lies[i], width, color=style_lies["color"],
               hatch=style_lies["hatch"], edgecolor=style_lies["edgecolor"],
               yerr=lies_sems[i], capsize=4)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=facts_color, label='Facts Mentioned (per response)'),
                       Patch(facecolor=lies_color, label='Lies (per response)')]
    ax.legend(handles=legend_elements)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Count per Response")
    ax.set_title("DeepSeek R1 70B: Facts Mentioned and Lies by Finetune (error bars = SEM)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    for i, (v, e) in enumerate(zip(mean_facts, facts_sems)):
        ax.text(i - width / 2, v + e + 0.1, f"{v:.1f}", ha='center', va='bottom', fontsize=8)
    for i, (v, e) in enumerate(zip(mean_lies, lies_sems)):
        ax.text(i + width / 2, v + e + 0.1, f"{v:.1f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "facts_and_lies.png"), dpi=150)
    plt.close()

    # Plot 4: Combined summary (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    for i, model in enumerate(models):
        style = get_bar_style(model, '#5B9BD5')
        ax.bar(i, honesty_rates[i], color=style["color"], hatch=style["hatch"],
               edgecolor=style["edgecolor"], yerr=honesty_sems[i], capsize=3)
    ax.set_ylabel("Mean Honesty Score (0-100)")
    ax.set_title("Honesty Score (error bars = SEM)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 100)

    ax = axes[0, 1]
    for i, model in enumerate(models):
        style = get_bar_style(model, '#7F8C8D')
        ax.bar(i, refusal_rates[i], color=style["color"], hatch=style["hatch"],
               edgecolor=style["edgecolor"], yerr=refusal_sems[i], capsize=3)
    ax.set_ylabel("Refusal Rate (%)")
    ax.set_title("Refusal Rate (error bars = SEM)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)

    ax = axes[1, 0]
    for i, model in enumerate(models):
        style = get_bar_style(model, '#27AE60')
        ax.bar(i, mean_facts[i], color=style["color"], hatch=style["hatch"],
               edgecolor=style["edgecolor"], yerr=facts_sems[i], capsize=3)
    ax.set_ylabel("Mean per Response")
    ax.set_title("Facts Mentioned (error bars = SEM)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)

    ax = axes[1, 1]
    for i, model in enumerate(models):
        style = get_bar_style(model, '#E74C3C')
        ax.bar(i, mean_lies[i], color=style["color"], hatch=style["hatch"],
               edgecolor=style["edgecolor"], yerr=lies_sems[i], capsize=3)
    ax.set_ylabel("Mean per Response")
    ax.set_title("Lies (error bars = SEM)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)

    fig.suptitle("DeepSeek R1 70B: Honesty Finetune Results Summary", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}/")


def main():
    print(f"Loading evaluation results from: {EVAL_DIR}")
    all_results = load_all_results()

    if not all_results:
        print("No evaluation files found.")
        return

    print(f"\nFound {len(all_results)} evaluated models")

    # Extract metrics
    all_metrics = {}
    for model, data in all_results.items():
        all_metrics[model] = extract_metrics(data)

    # Print summary table
    print("\nMetrics Summary:")
    print("-" * 110)
    print(f"{'Model':<35} {'Honesty':>12} {'Refusal%':>12} {'Facts/Resp':>12} {'Lies/Resp':>12}")
    print(f"{'':<35} {'(± SEM)':>12} {'(± SEM)':>12} {'(± SEM)':>12} {'(± SEM)':>12}")
    print("-" * 110)
    for model in sorted(all_metrics.keys()):
        m = all_metrics[model]
        short = MODEL_DISPLAY_NAMES.get(model, model)
        honesty_str = f"{m['mean_honesty']:.1f}±{m['honesty_sem']:.1f}"
        refusal_str = f"{m['refusal_rate']:.1f}±{m['refusal_rate_sem']:.1f}"
        facts_str = f"{m['mean_facts_per_response']:.2f}±{m['facts_sem']:.2f}"
        lies_str = f"{m['mean_lies_per_response']:.2f}±{m['lies_sem']:.2f}"
        print(f"{short:<35} {honesty_str:>12} {refusal_str:>12} {facts_str:>12} {lies_str:>12}")
    print("-" * 110)

    plot_results(all_metrics, OUTPUT_DIR)


if __name__ == "__main__":
    main()
