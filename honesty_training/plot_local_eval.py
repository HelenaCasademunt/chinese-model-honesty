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
        print(f"Error: {filepath} not found")
        print("\nNote: This script expects evaluated responses with 'evaluation' fields.")
        print("If you have raw responses, run the evaluation script first.")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


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
                response_type = "no_facts"  # Question has no facts to check
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
            "classification": row["response_type"] if row["response_type"] != "no_facts" else "unknown",
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
        "refusal": "#ff6b6b",
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


def plot_response_distribution(df: pd.DataFrame):
    """Plot overall distribution of response types."""
    counts = df["response_type"].value_counts()

    # Define colors for each response type
    colors_map = {
        "correct": "#51cf66",
        "partial": "#ffd43b",
        "evasive": "#ff9800",
        "lie": "#e74c3c",
        "refusal": "#ff6b6b",
        "no_facts": "#cccccc",
    }

    colors = [colors_map.get(rt, "#888888") for rt in counts.index]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(counts)), counts.values, color=colors, edgecolor="black")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([rt.replace("_", " ").title() for rt in counts.index], fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Honesty-Trained Model: Response Type Distribution", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add count labels on bars
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val}\n({val/len(df)*100:.1f}%)',
               ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_response_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    return counts


def plot_response_by_topic(df: pd.DataFrame):
    """Plot response type distribution by topic."""
    # Calculate counts
    topic_response = pd.crosstab(df["topic"], df["response_type"], normalize="index") * 100

    # Reorder columns
    col_order = ["correct", "partial", "evasive", "lie", "refusal", "no_facts"]
    col_order = [c for c in col_order if c in topic_response.columns]
    topic_response = topic_response[col_order]

    # Sort by % correct
    if "correct" in topic_response.columns:
        topic_response = topic_response.sort_values("correct", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(8, len(topic_response) * 0.4)))

    # Colors for stacked bars
    colors_map = {
        "correct": "#51cf66",
        "partial": "#ffd43b",
        "evasive": "#ff9800",
        "lie": "#e74c3c",
        "refusal": "#ff6b6b",
        "no_facts": "#cccccc",
    }

    y = np.arange(len(topic_response))
    left = np.zeros(len(topic_response))

    for response_type in col_order:
        values = topic_response[response_type].values
        ax.barh(y, values, left=left, label=response_type.replace("_", " ").title(),
                color=colors_map.get(response_type, "#888888"), edgecolor="black")

        # Add percentage labels for segments > 5%
        for i, (val, l) in enumerate(zip(values, left)):
            if val > 5:
                ax.text(l + val / 2, i, f'{val:.0f}%',
                       ha="center", va="center", fontweight="bold", fontsize=9)
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels([t.replace("_", " ").title() for t in topic_response.index], fontsize=10)
    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_title("Response Type Distribution by Topic", fontsize=14, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_response_by_topic.png", dpi=300, bbox_inches="tight")
    plt.show()

    return topic_response


def plot_response_by_level(df: pd.DataFrame):
    """Plot response type distribution by question difficulty level."""
    level_response = pd.crosstab(df["level"], df["response_type"], normalize="index") * 100

    # Reorder columns
    col_order = ["correct", "partial", "evasive", "lie", "refusal", "no_facts"]
    col_order = [c for c in col_order if c in level_response.columns]
    level_response = level_response[col_order]

    # Reorder rows by difficulty
    row_order = ["broad", "medium", "specific"]
    row_order = [r for r in row_order if r in level_response.index]
    level_response = level_response.reindex(row_order)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors_map = {
        "correct": "#51cf66",
        "partial": "#ffd43b",
        "evasive": "#ff9800",
        "lie": "#e74c3c",
        "refusal": "#ff6b6b",
        "no_facts": "#cccccc",
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
    ax.set_title("Response Type Distribution by Question Difficulty", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_response_by_level.png", dpi=300, bbox_inches="tight")
    plt.show()

    return level_response


def plot_fact_coverage(df: pd.DataFrame):
    """Plot fact coverage statistics."""
    # Filter to responses with facts
    df_with_facts = df[df["total_facts"] > 0].copy()

    if df_with_facts.empty:
        print("No responses with facts to analyze")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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
            "Partial": "#ffd43b",
            "Evasive": "#ff9800",
            "Lie": "#e74c3c",
            "Refusal": "#ff6b6b",
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


def plot_refusal_analysis(df: pd.DataFrame):
    """Analyze refusal patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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


def main():
    print("=" * 70)
    print("HONESTY-TRAINED MODEL EVALUATION PLOTTER")
    print("=" * 70)

    # Load results
    print("\nLoading evaluated local responses...")
    data = load_results()

    if not data:
        print("\nExpected file: honesty_training/results/evaluated_responses_goals.json")
        print("\nIf you have raw responses in responses_goals.json, you need to:")
        print("1. Run the evaluation script to add 'evaluation' fields")
        print("2. Save the output as evaluated_responses_goals.json")
        return

    print(f"Loaded {len(data)} questions")

    # Extract statistics
    df = extract_stats(data)

    if df.empty:
        print("No evaluated responses found!")
        return

    # Generate comparison plots with baselines
    print("\n" + "=" * 60)
    print("COMPARISON PLOTS WITH BASELINES")
    print("=" * 60)
    print("\nLoading baseline model data for comparison...")
    comparison_df = load_comparison_data(df)

    if not comparison_df.empty and len(comparison_df["method"].unique()) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON 1: Response Classification")
        print("=" * 60)
        plot_comparison_classification(comparison_df)

        print("\n" + "=" * 60)
        print("COMPARISON 2: Non-Refusal Classification")
        print("=" * 60)
        plot_comparison_non_refusal_classification(comparison_df)
    else:
        print("Warning: Could not load baseline data for comparison")

    # Generate plots for honesty model only
    print("\n" + "=" * 60)
    print("HONESTY MODEL DETAILED PLOTS")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("PLOT 1: Response Distribution")
    print("=" * 60)
    plot_response_distribution(df)

    print("\n" + "=" * 60)
    print("PLOT 2: Response by Topic")
    print("=" * 60)
    plot_response_by_topic(df)

    print("\n" + "=" * 60)
    print("PLOT 3: Response by Difficulty Level")
    print("=" * 60)
    plot_response_by_level(df)

    print("\n" + "=" * 60)
    print("PLOT 4: Fact Coverage Analysis")
    print("=" * 60)
    plot_fact_coverage(df)

    print("\n" + "=" * 60)
    print("PLOT 5: Refusal Analysis")
    print("=" * 60)
    plot_refusal_analysis(df)

    # Print summary
    print_summary(df)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
