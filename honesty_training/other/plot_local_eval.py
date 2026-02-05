"""
Plot results from honesty-trained model evaluation (local_responses.json with evaluations).

This script analyzes the honesty-trained model's responses and visualizes:
- Response classifications (REFUSAL vs CORRECT vs INCORRECT)
- Fact coverage across topics
- Performance comparisons across question difficulty levels
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Directories
SCRIPT_DIR = Path(__file__).parent
PLOTS_DIR = SCRIPT_DIR.parent / "plots" / "honesty_trained_eval"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = SCRIPT_DIR / "results"
BASELINE_DATA_DIR = SCRIPT_DIR.parent / "results" / "data"


def load_results(filepath: str = None):
    """Load the evaluated local responses."""
    if filepath is None:
        filepath = RESULTS_DIR / "evaluated_responses_goals.json"

    filepath = Path(filepath)
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_multiple_datasets():
    """Load goals, followup, and mixed datasets if they exist."""
    datasets = {}

    # Try loading each dataset
    dataset_files = {
        "Goals": "evaluated_responses_goals.json",
        "Followup": "evaluated_responses_followup.json",
        "Mixed": "evaluated_responses_mixed.json",
    }

    for name, filename in dataset_files.items():
        filepath = RESULTS_DIR / filename
        data = load_results(filepath)
        if data:
            datasets[name] = data
            print(f"Loaded {name}: {len(data)} questions")
        else:
            print(f"Skipped {name}: file not found")

    return datasets


def extract_stats(data: list) -> pd.DataFrame:
    """Extract relevant statistics into a DataFrame."""
    records = []

    for item in data:
        question_id = item.get("question_id")
        topic = item.get("topic")
        subtopic = item.get("subtopic")
        level = item.get("level")
        question = item.get("question")

        # Process each model response
        for idx, response_item in enumerate(item.get("model_responses", [])):
            # Check if this response has been evaluated
            evaluation = response_item.get("evaluation")
            if not evaluation:
                continue

            response_text = response_item.get("response_text", "")
            is_refusal = evaluation.get("is_refusal", False)
            fact_checks = evaluation.get("fact_checks", [])

            # Count fact statistics
            total_facts = len(fact_checks)
            facts_mentioned = sum(1 for fc in fact_checks if fc.get("status") == "mentioned")
            facts_not_mentioned = sum(1 for fc in fact_checks if fc.get("status") == "not_mentioned")
            facts_lied = sum(1 for fc in fact_checks if fc.get("status") == "lie")

            # Calculate percentages
            pct_mentioned = facts_mentioned / total_facts * 100 if total_facts > 0 else 0

            # Classify response type
            if is_refusal:
                response_type = "refusal"
            elif total_facts == 0:
                response_type = "error"  # Evaluation failed or no fact checks performed
            elif facts_mentioned == total_facts and facts_lied == 0:
                response_type = "correct"
            elif facts_mentioned == 0 and facts_lied == 0:
                response_type = "evasive"
            elif facts_lied > 0:
                response_type = "lie"
            else:
                response_type = "partial"

            records.append({
                "question_id": question_id,
                "topic": topic,
                "subtopic": subtopic,
                "level": level,
                "response_idx": idx,
                "response_type": response_type,
                "is_refusal": is_refusal,
                "total_facts": total_facts,
                "facts_mentioned": facts_mentioned,
                "facts_not_mentioned": facts_not_mentioned,
                "facts_lied": facts_lied,
                "pct_mentioned": pct_mentioned,
                "response_length": len(response_text),
            })

    df = pd.DataFrame(records)

    print(f"\nExtracted {len(df)} evaluated responses")
    print(f"\nResponse type distribution:")
    for rtype in df["response_type"].value_counts().items():
        print(f"  {rtype[0]}: {rtype[1]}")

    return df


def load_baseline_results(filename: str):
    """Load baseline results from method comparison data."""
    filepath = BASELINE_DATA_DIR / filename
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Filter out test topic
    data = [q for q in data if q.get("topic") != "test"]
    return data


def extract_baseline_stats(data: list, method_name: str) -> list:
    """Extract statistics from baseline format data."""
    stats = []
    for question in data:
        q_id = question["question_id"]
        topic = question["topic"]
        level = question.get("level", "")

        for resp in question.get("model_responses", []):
            evaluation = resp.get("evaluation", {})
            is_refusal = evaluation.get("is_refusal", False)
            fact_checks = evaluation.get("fact_checks", [])

            mentioned = sum(1 for fc in fact_checks if fc.get("status") == "mentioned")
            not_mentioned = sum(1 for fc in fact_checks if fc.get("status") == "not_mentioned")
            lies = sum(1 for fc in fact_checks if fc.get("status") == "lie")
            total_facts = mentioned + not_mentioned + lies

            # Classify using same logic as method comparison script
            if is_refusal:
                classification = "refusal"
            elif total_facts == 0:
                classification = "unknown"
            elif lies > 0:
                classification = "lie"
            elif mentioned == total_facts:
                classification = "correct"
            elif mentioned > 0:
                classification = "partial"
            else:
                classification = "evasive"

            stats.append({
                "method": method_name,
                "question_id": q_id,
                "topic": topic,
                "level": level,
                "is_refusal": is_refusal,
                "facts_mentioned": mentioned,
                "facts_not_mentioned": not_mentioned,
                "facts_lies": lies,
                "total_facts": total_facts,
                "classification": classification,
            })
    return stats


def extract_honesty_model_stats(df: pd.DataFrame, method_name: str) -> list:
    """Convert honesty model df to baseline format for comparison."""
    stats = []
    for _, row in df.iterrows():
        stats.append({
            "method": method_name,
            "question_id": row["question_id"],
            "topic": row["topic"],
            "level": row["level"],
            "is_refusal": row["is_refusal"],
            "facts_mentioned": row["facts_mentioned"],
            "facts_not_mentioned": row["facts_not_mentioned"],
            "facts_lies": row["facts_lied"],
            "total_facts": row["total_facts"],
            "classification": row["response_type"] if row["response_type"] != "error" else "unknown",
        })
    return stats


def load_comparison_data(honesty_df: pd.DataFrame):
    """Load and combine honesty model with baseline models for comparison."""
    all_stats = []

    # 1. Honesty finetuned model
    honesty_stats = extract_honesty_model_stats(honesty_df, "Honesty\nFinetuned")
    all_stats.extend(honesty_stats)
    print(f"Loaded Honesty Finetuned: {len(honesty_stats)} responses")

    # 2. Baseline Qwen (no system prompt)
    qwen_baseline = load_baseline_results("evaluated_baseline_responses_sys_none.json")
    if qwen_baseline:
        stats = extract_baseline_stats(qwen_baseline, "Qwen3 32B\n(Baseline)")
        all_stats.extend(stats)
        print(f"Loaded Qwen3 32B Baseline: {len(stats)} responses")

    # 3. Baseline Llama (no system prompt)
    llama_baseline = load_baseline_results("evaluated_baseline_responses_llama70b_no_sysprompt.json")
    if llama_baseline:
        stats = extract_baseline_stats(llama_baseline, "Llama 70B\n(Baseline)")
        all_stats.extend(stats)
        print(f"Loaded Llama 70B Baseline: {len(stats)} responses")

    return pd.DataFrame(all_stats)


def plot_comparison_classification(comparison_df: pd.DataFrame):
    """Plot response classification distribution comparing all three models."""
    valid_df = comparison_df[comparison_df["classification"] != "unknown"].copy()

    # Calculate classification counts per method
    class_counts = valid_df.groupby(["method", "classification"]).size().unstack(fill_value=0)

    # Ensure all classification columns exist
    for col in ["refusal", "correct", "partial", "evasive", "lie"]:
        if col not in class_counts.columns:
            class_counts[col] = 0

    # Reorder columns
    class_counts = class_counts[["refusal", "correct", "partial", "evasive", "lie"]]

    # Calculate percentages
    class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

    # Define method order - Llama first, then Qwen baseline, then Honesty
    method_order = []
    if "Llama 70B\n(Baseline)" in class_pcts.index:
        method_order.append("Llama 70B\n(Baseline)")
    if "Qwen3 32B\n(Baseline)" in class_pcts.index:
        method_order.append("Qwen3 32B\n(Baseline)")
    if "Honesty\nFinetuned" in class_pcts.index:
        method_order.append("Honesty\nFinetuned")

    class_pcts = class_pcts.reindex(method_order)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(class_pcts))
    width = 0.6

    colors = {
        "refusal": "#999999",
        "correct": "#51cf66",
        "partial": "#74c0fc",
        "evasive": "#ffd43b",
        "lie": "#e64980",
    }

    labels = {
        "refusal": "Refusal",
        "correct": "Correct (all facts)",
        "partial": "Partial (some facts)",
        "evasive": "Evasive (no facts)",
        "lie": "Lie (false facts)",
    }

    bottom = np.zeros(len(class_pcts))

    for classification in ["refusal", "correct", "partial", "evasive", "lie"]:
        values = class_pcts[classification].values
        ax.bar(x, values, width, label=labels[classification], color=colors[classification], bottom=bottom)

        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 4:
                ax.text(i, bot + val / 2, f'{val:.1f}%',
                       ha="center", va="center", fontweight="bold", fontsize=10,
                       color="white" if classification == "refusal" else "black")
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(class_pcts.index, fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Response Classification: Honesty Finetuned vs Baselines", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "comparison_01_response_classification.png", dpi=300, bbox_inches="tight")
    plt.show()

    return class_pcts


def plot_comparison_non_refusal_classification(comparison_df: pd.DataFrame):
    """Plot classification distribution for non-refusal responses only."""
    non_refusal_df = comparison_df[(comparison_df["is_refusal"] == False) &
                                    (comparison_df["classification"] != "unknown")].copy()

    if non_refusal_df.empty:
        print("No non-refusal responses found")
        return None

    # Calculate classification counts per method
    class_counts = non_refusal_df.groupby(["method", "classification"]).size().unstack(fill_value=0)

    # Ensure all classification columns exist
    for col in ["correct", "partial", "evasive", "lie"]:
        if col not in class_counts.columns:
            class_counts[col] = 0

    # Reorder columns
    class_counts = class_counts[["correct", "partial", "evasive", "lie"]]

    # Calculate percentages
    class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

    # Define method order - Llama first, then Qwen baseline, then Honesty
    method_order = []
    if "Llama 70B\n(Baseline)" in class_pcts.index:
        method_order.append("Llama 70B\n(Baseline)")
    if "Qwen3 32B\n(Baseline)" in class_pcts.index:
        method_order.append("Qwen3 32B\n(Baseline)")
    if "Honesty\nFinetuned" in class_pcts.index:
        method_order.append("Honesty\nFinetuned")

    class_pcts = class_pcts.reindex(method_order)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(class_pcts))
    width = 0.6

    colors = {
        "correct": "#51cf66",
        "partial": "#74c0fc",
        "evasive": "#ffd43b",
        "lie": "#e64980",
    }

    labels = {
        "correct": "Correct (all facts)",
        "partial": "Partial (some facts)",
        "evasive": "Evasive (no facts)",
        "lie": "Lie (false facts)",
    }

    bottom = np.zeros(len(class_pcts))

    for classification in ["correct", "partial", "evasive", "lie"]:
        values = class_pcts[classification].values
        ax.bar(x, values, width, label=labels[classification], color=colors[classification], bottom=bottom)

        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 4:
                ax.text(i, bot + val / 2, f'{val:.1f}%',
                       ha="center", va="center", fontweight="bold", fontsize=10)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(class_pcts.index, fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Non-Refusal Response Classification: Honesty Finetuned vs Baselines", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "comparison_02_non_refusal_classification.png", dpi=300, bbox_inches="tight")
    plt.show()

    return class_pcts


def compare_datasets(datasets_dict: dict):
    """Compare goals, followup, and mixed datasets."""
    all_stats = []

    for dataset_name, data in datasets_dict.items():
        df = extract_stats(data)
        stats = extract_honesty_model_stats(df, dataset_name)
        all_stats.extend(stats)
        print(f"Processed {dataset_name}: {len(stats)} responses")

    return pd.DataFrame(all_stats)


def plot_dataset_comparison_classification(comparison_df: pd.DataFrame):
    """Plot response classification distribution comparing goals, followup, and mixed datasets."""
    valid_df = comparison_df[comparison_df["classification"] != "unknown"].copy()

    if valid_df.empty:
        print("No valid data to compare")
        return None

    # Calculate classification counts per dataset
    class_counts = valid_df.groupby(["method", "classification"]).size().unstack(fill_value=0)

    # Ensure all classification columns exist
    for col in ["refusal", "correct", "partial", "evasive", "lie"]:
        if col not in class_counts.columns:
            class_counts[col] = 0

    # Reorder columns
    class_counts = class_counts[["refusal", "correct", "partial", "evasive", "lie"]]

    # Calculate percentages
    class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

    # Define dataset order
    dataset_order = ["Goals", "Followup", "Mixed"]
    dataset_order = [d for d in dataset_order if d in class_pcts.index]
    class_pcts = class_pcts.reindex(dataset_order)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(class_pcts))
    width = 0.6

    colors = {
        "refusal": "#999999",
        "correct": "#51cf66",
        "partial": "#74c0fc",
        "evasive": "#ffd43b",
        "lie": "#e64980",
    }

    labels = {
        "refusal": "Refusal",
        "correct": "Correct (all facts)",
        "partial": "Partial (some facts)",
        "evasive": "Evasive (no facts)",
        "lie": "Lie (false facts)",
    }

    bottom = np.zeros(len(class_pcts))

    for classification in ["refusal", "correct", "partial", "evasive", "lie"]:
        values = class_pcts[classification].values
        ax.bar(x, values, width, label=labels[classification], color=colors[classification], bottom=bottom)

        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 4:
                ax.text(i, bot + val / 2, f'{val:.1f}%',
                       ha="center", va="center", fontweight="bold", fontsize=10,
                       color="white" if classification == "refusal" else "black")
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(class_pcts.index, fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Response Classification: Goals vs Followup vs Mixed", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dataset_comparison_01_classification.png", dpi=300, bbox_inches="tight")
    plt.show()

    return class_pcts


def plot_dataset_comparison_non_refusal(comparison_df: pd.DataFrame):
    """Plot non-refusal classification for goals, followup, and mixed datasets."""
    non_refusal_df = comparison_df[(comparison_df["is_refusal"] == False) &
                                    (comparison_df["classification"] != "unknown")].copy()

    if non_refusal_df.empty:
        print("No non-refusal responses found")
        return None

    # Calculate classification counts per dataset
    class_counts = non_refusal_df.groupby(["method", "classification"]).size().unstack(fill_value=0)

    # Ensure all classification columns exist
    for col in ["correct", "partial", "evasive", "lie"]:
        if col not in class_counts.columns:
            class_counts[col] = 0

    # Reorder columns
    class_counts = class_counts[["correct", "partial", "evasive", "lie"]]

    # Calculate percentages
    class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

    # Define dataset order
    dataset_order = ["Goals", "Followup", "Mixed"]
    dataset_order = [d for d in dataset_order if d in class_pcts.index]
    class_pcts = class_pcts.reindex(dataset_order)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(class_pcts))
    width = 0.6

    colors = {
        "correct": "#51cf66",
        "partial": "#74c0fc",
        "evasive": "#ffd43b",
        "lie": "#e64980",
    }

    labels = {
        "correct": "Correct (all facts)",
        "partial": "Partial (some facts)",
        "evasive": "Evasive (no facts)",
        "lie": "Lie (false facts)",
    }

    bottom = np.zeros(len(class_pcts))

    for classification in ["correct", "partial", "evasive", "lie"]:
        values = class_pcts[classification].values
        ax.bar(x, values, width, label=labels[classification], color=colors[classification], bottom=bottom)

        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 4:
                ax.text(i, bot + val / 2, f'{val:.1f}%',
                       ha="center", va="center", fontweight="bold", fontsize=10)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(class_pcts.index, fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Non-Refusal Classification: Goals vs Followup vs Mixed", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dataset_comparison_02_non_refusal.png", dpi=300, bbox_inches="tight")
    plt.show()

    return class_pcts


def plot_dataset_comparison_metrics(comparison_df: pd.DataFrame):
    """Plot key metrics comparison across datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    dataset_order = ["Goals", "Followup", "Mixed"]
    dataset_order = [d for d in dataset_order if d in comparison_df["method"].unique()]

    # Metric 1: Refusal Rate
    ax1 = axes[0]
    refusal_rates = []
    for dataset in dataset_order:
        dataset_df = comparison_df[comparison_df["method"] == dataset]
        refusal_rate = (dataset_df["is_refusal"].sum() / len(dataset_df)) * 100
        refusal_rates.append(refusal_rate)

    bars = ax1.bar(range(len(dataset_order)), refusal_rates, color="#999999", edgecolor="black")
    ax1.set_xticks(range(len(dataset_order)))
    ax1.set_xticklabels(dataset_order, fontsize=11)
    ax1.set_ylabel("Refusal Rate (%)", fontsize=11)
    ax1.set_title("Refusal Rate by Dataset", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, refusal_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha="center", fontsize=10, fontweight="bold")

    # Metric 2: Correctness (of non-refusal responses)
    ax2 = axes[1]
    correctness_rates = []
    for dataset in dataset_order:
        dataset_df = comparison_df[(comparison_df["method"] == dataset) &
                                   (comparison_df["is_refusal"] == False) &
                                   (comparison_df["classification"] != "unknown")]
        if len(dataset_df) > 0:
            correct_rate = (dataset_df["classification"] == "correct").sum() / len(dataset_df) * 100
        else:
            correct_rate = 0
        correctness_rates.append(correct_rate)

    bars = ax2.bar(range(len(dataset_order)), correctness_rates, color="#51cf66", edgecolor="black")
    ax2.set_xticks(range(len(dataset_order)))
    ax2.set_xticklabels(dataset_order, fontsize=11)
    ax2.set_ylabel("Correctness Rate (%)", fontsize=11)
    ax2.set_title("Correctness (Non-Refusal Only)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, correctness_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha="center", fontsize=10, fontweight="bold")

    # Metric 3: Lie Rate (of non-refusal responses)
    ax3 = axes[2]
    lie_rates = []
    for dataset in dataset_order:
        dataset_df = comparison_df[(comparison_df["method"] == dataset) &
                                   (comparison_df["is_refusal"] == False) &
                                   (comparison_df["classification"] != "unknown")]
        if len(dataset_df) > 0:
            lie_rate = (dataset_df["classification"] == "lie").sum() / len(dataset_df) * 100
        else:
            lie_rate = 0
        lie_rates.append(lie_rate)

    bars = ax3.bar(range(len(dataset_order)), lie_rates, color="#e64980", edgecolor="black")
    ax3.set_xticks(range(len(dataset_order)))
    ax3.set_xticklabels(dataset_order, fontsize=11)
    ax3.set_ylabel("Lie Rate (%)", fontsize=11)
    ax3.set_title("Lie Rate (Non-Refusal Only)", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, lie_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dataset_comparison_03_metrics.png", dpi=300, bbox_inches="tight")
    plt.show()

    return {
        "refusal_rates": dict(zip(dataset_order, refusal_rates)),
        "correctness_rates": dict(zip(dataset_order, correctness_rates)),
        "lie_rates": dict(zip(dataset_order, lie_rates)),
    }


def compare_with_baselines(datasets: dict):
    """Compare honesty-trained models (all datasets) with Qwen and Llama baselines."""
    all_stats = []

    # Load baseline models first
    # 1. Llama 70B baseline
    llama_baseline = load_baseline_results("evaluated_baseline_responses_llama70b_no_sysprompt.json")
    if llama_baseline:
        stats = extract_baseline_stats(llama_baseline, "Llama 70B\n(Baseline)")
        all_stats.extend(stats)
        print(f"Loaded Llama 70B Baseline: {len(stats)} responses")

    # 2. Qwen3 32B baseline (no system prompt)
    qwen_baseline = load_baseline_results("evaluated_baseline_responses_sys_none.json")
    if qwen_baseline:
        stats = extract_baseline_stats(qwen_baseline, "Qwen3 32B\n(Baseline)")
        all_stats.extend(stats)
        print(f"Loaded Qwen3 32B Baseline: {len(stats)} responses")

    # 3-5. Add honesty-trained models for each dataset
    for dataset_name, data in datasets.items():
        df = extract_stats(data)
        stats = extract_honesty_model_stats(df, f"Honesty\n({dataset_name})")
        all_stats.extend(stats)
        print(f"Processed Honesty ({dataset_name}): {len(stats)} responses")

    return pd.DataFrame(all_stats)


def plot_full_model_comparison_stacked(comparison_df: pd.DataFrame):
    """Plot stacked bar comparing all models: Llama, Qwen, and Honesty (Goals/Followup/Mixed)."""
    valid_df = comparison_df[comparison_df["classification"] != "unknown"].copy()

    if valid_df.empty:
        print("No valid data to compare")
        return None

    # Calculate classification counts per method
    class_counts = valid_df.groupby(["method", "classification"]).size().unstack(fill_value=0)

    # Ensure all classification columns exist
    for col in ["refusal", "correct", "partial", "evasive", "lie"]:
        if col not in class_counts.columns:
            class_counts[col] = 0

    # Reorder columns
    class_counts = class_counts[["refusal", "correct", "partial", "evasive", "lie"]]

    # Calculate percentages
    class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

    # Define method order: Llama, Qwen, then Honesty models
    method_order = []
    if "Llama 70B\n(Baseline)" in class_pcts.index:
        method_order.append("Llama 70B\n(Baseline)")
    if "Qwen3 32B\n(Baseline)" in class_pcts.index:
        method_order.append("Qwen3 32B\n(Baseline)")

    # Add honesty models in specific order
    for dataset in ["Goals", "Followup", "Mixed"]:
        honesty_name = f"Honesty\n({dataset})"
        if honesty_name in class_pcts.index:
            method_order.append(honesty_name)

    class_pcts = class_pcts.reindex(method_order)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(class_pcts))
    width = 0.6

    colors = {
        "refusal": "#999999",
        "correct": "#51cf66",
        "partial": "#74c0fc",
        "evasive": "#ffd43b",
        "lie": "#e64980",
    }

    labels = {
        "refusal": "Refusal",
        "correct": "Correct (all facts)",
        "partial": "Partial (some facts)",
        "evasive": "Evasive (no facts)",
        "lie": "Lie (false facts)",
    }

    bottom = np.zeros(len(class_pcts))

    for classification in ["refusal", "correct", "partial", "evasive", "lie"]:
        values = class_pcts[classification].values
        ax.bar(x, values, width, label=labels[classification], color=colors[classification], bottom=bottom)

        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 4:
                ax.text(i, bot + val / 2, f'{val:.1f}%',
                       ha="center", va="center", fontweight="bold", fontsize=9,
                       color="white" if classification == "refusal" else "black")
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(class_pcts.index, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Response Classification: Baselines vs Honesty-Trained Models", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "full_comparison_stacked.png", dpi=300, bbox_inches="tight")
    plt.show()

    return class_pcts


def plot_full_model_comparison_non_refusal(comparison_df: pd.DataFrame):
    """Plot non-refusal classification comparing all models."""
    non_refusal_df = comparison_df[(comparison_df["is_refusal"] == False) &
                                    (comparison_df["classification"] != "unknown")].copy()

    if non_refusal_df.empty:
        print("No non-refusal responses found")
        return None

    # Calculate classification counts per method
    class_counts = non_refusal_df.groupby(["method", "classification"]).size().unstack(fill_value=0)

    # Ensure all classification columns exist
    for col in ["correct", "partial", "evasive", "lie"]:
        if col not in class_counts.columns:
            class_counts[col] = 0

    # Reorder columns
    class_counts = class_counts[["correct", "partial", "evasive", "lie"]]

    # Calculate percentages
    class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

    # Define method order: Llama, Qwen, then Honesty models
    method_order = []
    if "Llama 70B\n(Baseline)" in class_pcts.index:
        method_order.append("Llama 70B\n(Baseline)")
    if "Qwen3 32B\n(Baseline)" in class_pcts.index:
        method_order.append("Qwen3 32B\n(Baseline)")

    # Add honesty models in specific order
    for dataset in ["Goals", "Followup", "Mixed"]:
        honesty_name = f"Honesty\n({dataset})"
        if honesty_name in class_pcts.index:
            method_order.append(honesty_name)

    class_pcts = class_pcts.reindex(method_order)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(class_pcts))
    width = 0.6

    colors = {
        "correct": "#51cf66",
        "partial": "#74c0fc",
        "evasive": "#ffd43b",
        "lie": "#e64980",
    }

    labels = {
        "correct": "Correct (all facts)",
        "partial": "Partial (some facts)",
        "evasive": "Evasive (no facts)",
        "lie": "Lie (false facts)",
    }

    bottom = np.zeros(len(class_pcts))

    for classification in ["correct", "partial", "evasive", "lie"]:
        values = class_pcts[classification].values
        ax.bar(x, values, width, label=labels[classification], color=colors[classification], bottom=bottom)

        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 4:
                ax.text(i, bot + val / 2, f'{val:.1f}%',
                       ha="center", va="center", fontweight="bold", fontsize=9)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(class_pcts.index, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Non-Refusal Classification: Baselines vs Honesty-Trained Models", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "full_comparison_non_refusal_stacked.png", dpi=300, bbox_inches="tight")
    plt.show()

    return class_pcts


def plot_response_distribution(df: pd.DataFrame, dataset_name: str = "Dataset"):
    """Plot stacked bar comparing this dataset with baselines."""
    # Prepare comparison data
    all_stats = []

    # Load baseline models
    llama_baseline = load_baseline_results("evaluated_baseline_responses_llama70b_no_sysprompt.json")
    if llama_baseline:
        stats = extract_baseline_stats(llama_baseline, "Llama 70B\n(Baseline)")
        all_stats.extend(stats)

    qwen_baseline = load_baseline_results("evaluated_baseline_responses_sys_none.json")
    if qwen_baseline:
        stats = extract_baseline_stats(qwen_baseline, "Qwen3 32B\n(Baseline)")
        all_stats.extend(stats)

    # Add this dataset
    stats = extract_honesty_model_stats(df, f"Honesty\n({dataset_name})")
    all_stats.extend(stats)

    comparison_df = pd.DataFrame(all_stats)
    valid_df = comparison_df[comparison_df["classification"] != "unknown"].copy()

    # Calculate classification counts per method
    class_counts = valid_df.groupby(["method", "classification"]).size().unstack(fill_value=0)

    # Ensure all classification columns exist
    for col in ["refusal", "correct", "partial", "evasive", "lie"]:
        if col not in class_counts.columns:
            class_counts[col] = 0

    # Reorder columns
    class_counts = class_counts[["refusal", "correct", "partial", "evasive", "lie"]]

    # Calculate percentages
    class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

    # Define method order
    method_order = []
    if "Llama 70B\n(Baseline)" in class_pcts.index:
        method_order.append("Llama 70B\n(Baseline)")
    if "Qwen3 32B\n(Baseline)" in class_pcts.index:
        method_order.append("Qwen3 32B\n(Baseline)")
    method_order.append(f"Honesty\n({dataset_name})")

    class_pcts = class_pcts.reindex(method_order)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(class_pcts))
    width = 0.6

    colors = {
        "refusal": "#999999",
        "correct": "#51cf66",
        "partial": "#74c0fc",
        "evasive": "#ffd43b",
        "lie": "#e64980",
    }

    labels = {
        "refusal": "Refusal",
        "correct": "Correct (all facts)",
        "partial": "Partial (some facts)",
        "evasive": "Evasive (no facts)",
        "lie": "Lie (false facts)",
    }

    bottom = np.zeros(len(class_pcts))

    for classification in ["refusal", "correct", "partial", "evasive", "lie"]:
        values = class_pcts[classification].values
        ax.bar(x, values, width, label=labels[classification], color=colors[classification], bottom=bottom)

        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 4:
                ax.text(i, bot + val / 2, f'{val:.1f}%',
                       ha="center", va="center", fontweight="bold", fontsize=9,
                       color="white" if classification == "refusal" else "black")
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(class_pcts.index, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title(f"Response Classification: {dataset_name} vs Baselines", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_response_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    return class_pcts


def plot_response_by_topic(df: pd.DataFrame, dataset_name: str = "Dataset"):
    """Plot non-refusal response distribution comparing with baselines."""
    # Prepare comparison data
    all_stats = []

    # Load baseline models
    llama_baseline = load_baseline_results("evaluated_baseline_responses_llama70b_no_sysprompt.json")
    if llama_baseline:
        stats = extract_baseline_stats(llama_baseline, "Llama 70B\n(Baseline)")
        all_stats.extend(stats)

    qwen_baseline = load_baseline_results("evaluated_baseline_responses_sys_none.json")
    if qwen_baseline:
        stats = extract_baseline_stats(qwen_baseline, "Qwen3 32B\n(Baseline)")
        all_stats.extend(stats)

    # Add this dataset
    stats = extract_honesty_model_stats(df, f"Honesty\n({dataset_name})")
    all_stats.extend(stats)

    comparison_df = pd.DataFrame(all_stats)

    # Filter to non-refusal responses only
    non_refusal_df = comparison_df[(comparison_df["is_refusal"] == False) &
                                    (comparison_df["classification"] != "unknown")].copy()

    # Calculate classification counts per method
    class_counts = non_refusal_df.groupby(["method", "classification"]).size().unstack(fill_value=0)

    # Ensure all classification columns exist
    for col in ["correct", "partial", "evasive", "lie"]:
        if col not in class_counts.columns:
            class_counts[col] = 0

    # Reorder columns
    class_counts = class_counts[["correct", "partial", "evasive", "lie"]]

    # Calculate percentages
    class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

    # Define method order
    method_order = []
    if "Llama 70B\n(Baseline)" in class_pcts.index:
        method_order.append("Llama 70B\n(Baseline)")
    if "Qwen3 32B\n(Baseline)" in class_pcts.index:
        method_order.append("Qwen3 32B\n(Baseline)")
    method_order.append(f"Honesty\n({dataset_name})")

    class_pcts = class_pcts.reindex(method_order)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(class_pcts))
    width = 0.6

    colors = {
        "correct": "#51cf66",
        "partial": "#74c0fc",
        "evasive": "#ffd43b",
        "lie": "#e64980",
    }

    labels = {
        "correct": "Correct (all facts)",
        "partial": "Partial (some facts)",
        "evasive": "Evasive (no facts)",
        "lie": "Lie (false facts)",
    }

    bottom = np.zeros(len(class_pcts))

    for classification in ["correct", "partial", "evasive", "lie"]:
        values = class_pcts[classification].values
        ax.bar(x, values, width, label=labels[classification], color=colors[classification], bottom=bottom)

        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 4:
                ax.text(i, bot + val / 2, f'{val:.1f}%',
                       ha="center", va="center", fontweight="bold", fontsize=9)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(class_pcts.index, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title(f"Non-Refusal Classification: {dataset_name} vs Baselines", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_non_refusal_classification.png", dpi=300, bbox_inches="tight")
    plt.show()

    return class_pcts


def plot_response_by_level(df: pd.DataFrame, dataset_name: str = "Dataset"):
    """Plot response type distribution by question difficulty level."""
    level_response = pd.crosstab(df["level"], df["response_type"], normalize="index") * 100

    # Reorder columns
    col_order = ["correct", "partial", "evasive", "lie", "refusal", "error"]
    col_order = [c for c in col_order if c in level_response.columns]
    level_response = level_response[col_order]

    # Reorder rows by difficulty
    row_order = ["broad", "medium", "specific"]
    row_order = [r for r in row_order if r in level_response.index]
    level_response = level_response.reindex(row_order)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors_map = {
        "correct": "#51cf66",
        "partial": "#74c0fc",
        "evasive": "#ffd43b",
        "lie": "#e64980",
        "refusal": "#999999",
        "error": "#cccccc",
    }

    x = np.arange(len(level_response))
    bottom = np.zeros(len(level_response))
    width = 0.6

    for response_type in col_order:
        values = level_response[response_type].values
        ax.bar(x, values, width, bottom=bottom, label=response_type.replace("_", " ").title(),
               color=colors_map.get(response_type, "#888888"), edgecolor="black")

        # Add percentage labels
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 5:
                ax.text(i, bot + val / 2, f'{val:.0f}%',
                       ha="center", va="center", fontweight="bold", fontsize=11)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels([l.title() for l in level_response.index], fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xlabel("Question Difficulty Level", fontsize=12)
    ax.set_title(f"{dataset_name}: Response Distribution by Question Difficulty", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_response_by_level.png", dpi=300, bbox_inches="tight")
    plt.show()

    return level_response


def plot_fact_coverage(df: pd.DataFrame, dataset_name: str = "Dataset"):
    """Plot fact coverage statistics."""
    # Filter to responses with facts
    df_with_facts = df[df["total_facts"] > 0].copy()

    if df_with_facts.empty:
        print("No responses with facts to analyze")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{dataset_name}: Fact Coverage Analysis", fontsize=16, fontweight="bold", y=1.00)

    # Plot 1: Histogram of % facts mentioned
    ax1 = axes[0, 0]
    bins = np.arange(0, 101, 10)
    ax1.hist(df_with_facts["pct_mentioned"], bins=bins, color="#4dabf7", edgecolor="black")
    ax1.set_xlabel("% Facts Mentioned", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Distribution of Fact Coverage", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    ax1.axvline(x=df_with_facts["pct_mentioned"].mean(), color="red", linestyle="--",
                label=f'Mean: {df_with_facts["pct_mentioned"].mean():.1f}%')
    ax1.legend()

    # Plot 2: Box plot by response type
    ax2 = axes[0, 1]
    response_types = ["correct", "partial", "evasive", "lie", "refusal"]
    data_by_type = [df_with_facts[df_with_facts["response_type"] == rt]["pct_mentioned"]
                    for rt in response_types if rt in df_with_facts["response_type"].values]
    labels_by_type = [rt.title() for rt in response_types
                      if rt in df_with_facts["response_type"].values]

    if data_by_type:
        bp = ax2.boxplot(data_by_type, labels=labels_by_type, patch_artist=True)
        colors_map = {
            "Correct": "#51cf66",
            "Partial": "#74c0fc",
            "Evasive": "#ffd43b",
            "Lie": "#e64980",
            "Refusal": "#999999",
        }
        for patch, label in zip(bp["boxes"], labels_by_type):
            patch.set_facecolor(colors_map.get(label, "#888888"))

        ax2.set_ylabel("% Facts Mentioned", fontsize=11)
        ax2.set_title("Fact Coverage by Response Type", fontsize=12, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)
        ax2.set_xticklabels(labels_by_type, rotation=45, ha="right")

    # Plot 3: Average facts mentioned by topic
    ax3 = axes[1, 0]
    topic_stats = df_with_facts.groupby("topic")["pct_mentioned"].mean().sort_values()

    y = np.arange(len(topic_stats))
    colors = ['#51cf66' if v > 70 else '#ffd43b' if v > 40 else '#ff6b6b'
              for v in topic_stats.values]

    bars = ax3.barh(y, topic_stats.values, color=colors, edgecolor="black")
    ax3.set_yticks(y)
    ax3.set_yticklabels([t.replace("_", " ").title() for t in topic_stats.index], fontsize=9)
    ax3.set_xlabel("Average % Facts Mentioned", fontsize=11)
    ax3.set_title("Fact Coverage by Topic", fontsize=12, fontweight="bold")
    ax3.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, topic_stats.values):
        ax3.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va="center", fontsize=9, fontweight="bold")

    # Plot 4: Fact coverage by question level
    ax4 = axes[1, 1]
    level_order = ["broad", "medium", "specific"]
    level_stats = df_with_facts.groupby("level")["pct_mentioned"].agg(["mean", "std", "count"])
    level_stats = level_stats.reindex([l for l in level_order if l in level_stats.index])

    x = np.arange(len(level_stats))
    colors_level = ['#51cf66' if v > 70 else '#ffd43b' if v > 40 else '#ff6b6b'
                    for v in level_stats["mean"].values]

    bars = ax4.bar(x, level_stats["mean"], yerr=level_stats["std"], capsize=5,
                   color=colors_level, edgecolor="black")
    ax4.set_xticks(x)
    ax4.set_xticklabels([l.title() for l in level_stats.index], fontsize=11)
    ax4.set_ylabel("Average % Facts Mentioned", fontsize=11)
    ax4.set_xlabel("Question Difficulty", fontsize=11)
    ax4.set_title("Fact Coverage by Difficulty Level", fontsize=12, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    for bar, (idx, row) in zip(bars, level_stats.iterrows()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + row["std"] + 2,
                f'{row["mean"]:.1f}%\n(n={int(row["count"])})',
                ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_fact_coverage.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_refusal_analysis(df: pd.DataFrame, dataset_name: str = "Dataset"):
    """Analyze refusal patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{dataset_name}: Refusal Analysis", fontsize=16, fontweight="bold", y=1.00)

    # Plot 1: Refusal rate by topic
    ax1 = axes[0]
    topic_refusal = df.groupby("topic")["is_refusal"].mean() * 100
    topic_refusal = topic_refusal.sort_values()

    y = np.arange(len(topic_refusal))
    colors = ['#ff6b6b' if v > 50 else '#ffd43b' if v > 20 else '#51cf66'
              for v in topic_refusal.values]

    bars = ax1.barh(y, topic_refusal.values, color=colors, edgecolor="black")
    ax1.set_yticks(y)
    ax1.set_yticklabels([t.replace("_", " ").title() for t in topic_refusal.index], fontsize=9)
    ax1.set_xlabel("Refusal Rate (%)", fontsize=11)
    ax1.set_title("Refusal Rate by Topic", fontsize=12, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, topic_refusal.values):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va="center", fontsize=9, fontweight="bold")

    # Plot 2: Refusal rate by question level
    ax2 = axes[1]
    level_order = ["broad", "medium", "specific"]
    level_refusal = df.groupby("level")["is_refusal"].mean() * 100
    level_refusal = level_refusal.reindex([l for l in level_order if l in level_refusal.index])

    x = np.arange(len(level_refusal))
    colors_level = ['#ff6b6b' if v > 50 else '#ffd43b' if v > 20 else '#51cf66'
                    for v in level_refusal.values]

    bars = ax2.bar(x, level_refusal.values, color=colors_level, edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.title() for l in level_refusal.index], fontsize=11)
    ax2.set_ylabel("Refusal Rate (%)", fontsize=11)
    ax2.set_xlabel("Question Difficulty", fontsize=11)
    ax2.set_title("Refusal Rate by Difficulty Level", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, level_refusal.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_refusal_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("HONESTY-TRAINED MODEL EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nTotal responses evaluated: {len(df)}")
    print(f"Unique questions: {df['question_id'].nunique()}")

    print(f"\nResponse type distribution:")
    for rtype, count in df["response_type"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {rtype}: {count} ({pct:.1f}%)")

    print(f"\nRefusal statistics:")
    refusal_count = df["is_refusal"].sum()
    refusal_pct = refusal_count / len(df) * 100
    print(f"  Refusals: {refusal_count} ({refusal_pct:.1f}%)")
    print(f"  Non-refusals: {len(df) - refusal_count} ({100 - refusal_pct:.1f}%)")

    df_with_facts = df[df["total_facts"] > 0]
    if not df_with_facts.empty:
        print(f"\nFact coverage (responses with facts, n={len(df_with_facts)}):")
        print(f"  Mean % facts mentioned: {df_with_facts['pct_mentioned'].mean():.1f}%")
        print(f"  Median % facts mentioned: {df_with_facts['pct_mentioned'].median():.1f}%")
        print(f"  Full coverage (100%): {(df_with_facts['pct_mentioned'] == 100).sum()} "
              f"({(df_with_facts['pct_mentioned'] == 100).sum() / len(df_with_facts) * 100:.1f}%)")
        print(f"  Zero coverage (0%): {(df_with_facts['pct_mentioned'] == 0).sum()} "
              f"({(df_with_facts['pct_mentioned'] == 0).sum() / len(df_with_facts) * 100:.1f}%)")

    # Save summary to CSV
    summary_stats = {
        "metric": ["Total Responses", "Refusal Rate %", "Mean Fact Coverage %"],
        "value": [
            len(df),
            refusal_pct,
            df_with_facts["pct_mentioned"].mean() if not df_with_facts.empty else 0
        ]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(PLOTS_DIR / "summary_stats.csv", index=False)
    print(f"\nSummary saved to: {PLOTS_DIR / 'summary_stats.csv'}")


def save_dataset_comparison_summary(datasets: dict):
    """Save a CSV comparing key metrics across all datasets."""
    summary_rows = []

    for dataset_name, data in datasets.items():
        df = extract_stats(data)

        if df.empty:
            continue

        refusal_rate = (df["is_refusal"].sum() / len(df)) * 100

        # Non-refusal metrics
        non_refusal_df = df[df["is_refusal"] == False]
        if len(non_refusal_df) > 0:
            correct_rate = (non_refusal_df["response_type"] == "correct").sum() / len(non_refusal_df) * 100
            lie_rate = (non_refusal_df["response_type"] == "lie").sum() / len(non_refusal_df) * 100
            evasive_rate = (non_refusal_df["response_type"] == "evasive").sum() / len(non_refusal_df) * 100
        else:
            correct_rate = lie_rate = evasive_rate = 0

        # Fact coverage
        df_with_facts = df[df["total_facts"] > 0]
        mean_fact_coverage = df_with_facts["pct_mentioned"].mean() if not df_with_facts.empty else 0

        summary_rows.append({
            "Dataset": dataset_name,
            "Total Responses": len(df),
            "Refusal Rate (%)": round(refusal_rate, 2),
            "Correct Rate (%)": round(correct_rate, 2),
            "Lie Rate (%)": round(lie_rate, 2),
            "Evasive Rate (%)": round(evasive_rate, 2),
            "Mean Fact Coverage (%)": round(mean_fact_coverage, 2),
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        output_path = PLOTS_DIR / "dataset_comparison_summary.csv"
        summary_df.to_csv(output_path, index=False)
        print(f"\nDataset comparison summary saved to: {output_path}")

        # Print the table
        print("\n" + "=" * 70)
        print("DATASET COMPARISON SUMMARY")
        print("=" * 70)
        print(summary_df.to_string(index=False))

    return summary_rows


def main():
    print("=" * 70)
    print("HONESTY-TRAINED MODEL EVALUATION PLOTTER")
    print("=" * 70)

    # Load all datasets (goals, followup, mixed)
    print("\nLoading evaluated datasets...")
    datasets = load_multiple_datasets()

    if not datasets:
        print("\nNo datasets found!")
        print("\nExpected files in honesty_training/results/:")
        print("  - evaluated_responses_goals.json")
        print("  - evaluated_responses_followup.json")
        print("  - evaluated_responses_mixed.json")
        return

    # Generate dataset comparison plots (goals vs followup vs mixed)
    if len(datasets) > 1:
        print("\n" + "=" * 70)
        print("DATASET COMPARISON: GOALS vs FOLLOWUP vs MIXED")
        print("=" * 70)

        dataset_comparison_df = compare_datasets(datasets)

        print("\n" + "=" * 60)
        print("DATASET COMPARISON 1: Response Classification")
        print("=" * 60)
        plot_dataset_comparison_classification(dataset_comparison_df)

        print("\n" + "=" * 60)
        print("DATASET COMPARISON 2: Non-Refusal Classification")
        print("=" * 60)
        plot_dataset_comparison_non_refusal(dataset_comparison_df)

        print("\n" + "=" * 60)
        print("DATASET COMPARISON 3: Key Metrics")
        print("=" * 60)
        plot_dataset_comparison_metrics(dataset_comparison_df)

        print("\n" + "=" * 60)
        print("DATASET COMPARISON 4: Summary Table")
        print("=" * 60)
        save_dataset_comparison_summary(datasets)

    # Generate detailed plots for each dataset
    for dataset_name, data in datasets.items():
        print("\n" + "=" * 70)
        print(f"{dataset_name.upper()} DATASET - DETAILED PLOTS")
        print("=" * 70)

        df = extract_stats(data)

        if df.empty:
            print(f"No evaluated responses found in {dataset_name}!")
            continue

        # Update plot directory for this dataset
        dataset_plots_dir = PLOTS_DIR / dataset_name.lower()
        dataset_plots_dir.mkdir(parents=True, exist_ok=True)

        # Temporarily change PLOTS_DIR
        original_plots_dir = globals()["PLOTS_DIR"]
        globals()["PLOTS_DIR"] = dataset_plots_dir

        print(f"\n{dataset_name} - Response Distribution (vs Baselines)")
        plot_response_distribution(df, dataset_name)

        print(f"\n{dataset_name} - Non-Refusal Classification (vs Baselines)")
        plot_response_by_topic(df, dataset_name)

        print(f"\n{dataset_name} - Response by Difficulty Level")
        plot_response_by_level(df, dataset_name)

        print(f"\n{dataset_name} - Fact Coverage Analysis")
        plot_fact_coverage(df, dataset_name)

        print(f"\n{dataset_name} - Refusal Analysis")
        plot_refusal_analysis(df, dataset_name)

        print(f"\n{dataset_name} - Summary")
        print_summary(df)

        # Restore original PLOTS_DIR
        globals()["PLOTS_DIR"] = original_plots_dir

        print(f"Plots for {dataset_name} saved to: {dataset_plots_dir}")

    # Full comparison with baseline models (Llama + Qwen + all honesty datasets)
    print("\n" + "=" * 70)
    print("FULL MODEL COMPARISON: BASELINES vs HONESTY-TRAINED")
    print("=" * 70)
    print("\nLoading baseline model data for comparison...")

    full_comparison_df = compare_with_baselines(datasets)

    if not full_comparison_df.empty and len(full_comparison_df["method"].unique()) > 1:
        print("\n" + "=" * 60)
        print("FULL COMPARISON 1: Stacked Bar - All Response Types")
        print("=" * 60)
        plot_full_model_comparison_stacked(full_comparison_df)

        print("\n" + "=" * 60)
        print("FULL COMPARISON 2: Stacked Bar - Non-Refusal Only")
        print("=" * 60)
        plot_full_model_comparison_non_refusal(full_comparison_df)
    else:
        print("Warning: Could not load baseline data for comparison")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
