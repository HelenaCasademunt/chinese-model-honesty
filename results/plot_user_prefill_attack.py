"""
Plotting script for user prefill attack results.
Supports both simple user prefill attacks and regular user prefill attacks.
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
plt.rcParams['figure.figsize'] = (12, 6)

# Directories
SCRIPT_DIR = Path(__file__).parent
PLOTS_DIR = SCRIPT_DIR.parent / "plots" / "user_prefill_attack"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = SCRIPT_DIR / "data"


def load_simple_attack_results():
    """Load evaluated simple user prefill attack results."""
    filepath = DATA_DIR / "evaluated_user_prefill_simple_attack_responses.json"
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Filter out test topic
    data = [q for q in data if q.get("topic") != "test"]
    print(f"Loaded simple attack: {len(data)} questions")
    return data


def load_regular_attack_results():
    """Load evaluated regular user prefill attack results."""
    filepath = DATA_DIR / "evaluated_user_prefill_attack_responses.json"
    if not filepath.exists():
        print(f"Warning: {filepath} not found (regular attack not evaluated yet)")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Filter out test topic
    data = [q for q in data if q.get("topic") != "test"]
    print(f"Loaded regular attack: {len(data)} questions")
    return data


def extract_simple_attack_stats(data: list) -> pd.DataFrame:
    """Extract statistics from simple attack responses."""
    stats = []

    for question in data:
        q_id = question["question_id"]
        topic = question["topic"]
        level = question.get("level", "")
        user_prefill = question.get("user_prefill", "")

        for resp in question.get("evaluated_responses", []):
            evaluation = resp.get("evaluation", {})
            is_refusal = evaluation.get("is_refusal", False)
            fact_checks = evaluation.get("fact_checks", [])
            continuation = resp.get("continuation", "")

            # Count fact statuses
            mentioned = 0
            not_mentioned = 0
            lies = 0

            for fc in fact_checks:
                status = fc.get("status", "")
                if status == "mentioned":
                    mentioned += 1
                elif status == "not_mentioned":
                    not_mentioned += 1
                elif status == "lie":
                    lies += 1

            total_facts = mentioned + not_mentioned + lies

            # Classify response
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
                "attack_type": "simple",
                "question_id": q_id,
                "topic": topic,
                "level": level,
                "user_prefill": user_prefill,
                "continuation": continuation[:50] + "..." if len(continuation) > 50 else continuation,
                "is_refusal": is_refusal,
                "facts_mentioned": mentioned,
                "facts_not_mentioned": not_mentioned,
                "facts_lies": lies,
                "total_facts": total_facts,
                "classification": classification,
            })

    return pd.DataFrame(stats)


def extract_regular_attack_stats(data: list) -> pd.DataFrame:
    """Extract statistics from regular attack responses."""
    stats = []

    for question in data:
        q_id = question["question_id"]
        topic = question["topic"]
        level = question.get("level", "")

        for sample in question.get("initial_samples", []):
            for attack in sample.get("user_prefill_attacks", []):
                user_prefill = attack.get("user_prefill", "")
                continuations = attack.get("continuations", [])
                evaluated_continuations = attack.get("evaluated_continuations", [])

                # Match continuations with their evaluations
                for i, evaluation in enumerate(evaluated_continuations):
                    continuation = continuations[i] if i < len(continuations) else ""
                    is_refusal = evaluation.get("is_refusal", False)
                    fact_checks = evaluation.get("fact_checks", [])

                    # Count fact statuses
                    mentioned = 0
                    not_mentioned = 0
                    lies = 0

                    for fc in fact_checks:
                        status = fc.get("status", "")
                        if status == "mentioned":
                            mentioned += 1
                        elif status == "not_mentioned":
                            not_mentioned += 1
                        elif status == "lie":
                            lies += 1

                    total_facts = mentioned + not_mentioned + lies

                    # Classify response
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

                    # Shorten user_prefill for display
                    prefill_short = user_prefill[:30] + "..." if len(user_prefill) > 30 else user_prefill

                    stats.append({
                        "attack_type": "regular",
                        "question_id": q_id,
                        "topic": topic,
                        "level": level,
                        "user_prefill": prefill_short,
                        "user_prefill_full": user_prefill,
                        "continuation": continuation[:50] + "..." if len(continuation) > 50 else continuation,
                        "is_refusal": is_refusal,
                        "facts_mentioned": mentioned,
                        "facts_not_mentioned": not_mentioned,
                        "facts_lies": lies,
                        "total_facts": total_facts,
                        "classification": classification,
                    })

    return pd.DataFrame(stats)


def plot_refusal_rates_by_attack_type(df: pd.DataFrame):
    """Plot refusal rates comparison across attack types."""
    refusal_stats = df.groupby("attack_type").agg({
        "is_refusal": ["sum", "count"]
    }).reset_index()
    refusal_stats.columns = ["attack_type", "refusals", "total"]
    refusal_stats["refusal_rate"] = refusal_stats["refusals"] / refusal_stats["total"] * 100
    refusal_stats["non_refusal_rate"] = 100 - refusal_stats["refusal_rate"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(refusal_stats))
    width = 0.5

    colors = ['#51cf66', '#ff6b6b']  # green for non-refusal, red for refusal

    bars1 = ax.bar(x, refusal_stats["non_refusal_rate"], width, label="Non-Refusal", color=colors[0])
    bars2 = ax.bar(x, refusal_stats["refusal_rate"], width, bottom=refusal_stats["non_refusal_rate"],
                   label="Refusal", color=colors[1])

    ax.set_xticks(x)
    ax.set_xticklabels(refusal_stats["attack_type"].str.title() + " Attack", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Refusal Rates by Attack Type", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    # Add percentage labels
    for i, (_, row) in enumerate(refusal_stats.iterrows()):
        if row["non_refusal_rate"] > 5:
            ax.text(i, row["non_refusal_rate"] / 2, f'{row["non_refusal_rate"]:.1f}%',
                   ha="center", va="center", fontweight="bold", fontsize=11)
        if row["refusal_rate"] > 5:
            ax.text(i, row["non_refusal_rate"] + row["refusal_rate"] / 2, f'{row["refusal_rate"]:.1f}%',
                   ha="center", va="center", fontweight="bold", fontsize=11, color="white")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_refusal_rates_by_attack_type.png", dpi=300, bbox_inches="tight")
    plt.show()

    return refusal_stats


def plot_refusal_rates_by_prefill(df: pd.DataFrame, attack_type: str = None):
    """Plot refusal rates by user prefill type."""
    if attack_type:
        plot_df = df[df["attack_type"] == attack_type].copy()
        title_suffix = f" ({attack_type.title()} Attack)"
    else:
        plot_df = df.copy()
        title_suffix = ""

    refusal_stats = plot_df.groupby("user_prefill").agg({
        "is_refusal": ["sum", "count"]
    }).reset_index()
    refusal_stats.columns = ["user_prefill", "refusals", "total"]
    refusal_stats["refusal_rate"] = refusal_stats["refusals"] / refusal_stats["total"] * 100
    refusal_stats["non_refusal_rate"] = 100 - refusal_stats["refusal_rate"]

    # Sort by refusal rate
    refusal_stats = refusal_stats.sort_values("refusal_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(refusal_stats) * 0.5)))

    colors = ['#51cf66', '#ff6b6b']

    y = np.arange(len(refusal_stats))
    height = 0.6

    bars1 = ax.barh(y, refusal_stats["non_refusal_rate"], height, label="Non-Refusal", color=colors[0])
    bars2 = ax.barh(y, refusal_stats["refusal_rate"], height, left=refusal_stats["non_refusal_rate"],
                    label="Refusal", color=colors[1])

    ax.set_yticks(y)
    ax.set_yticklabels(refusal_stats["user_prefill"], fontsize=10)
    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_title(f"Refusal Rates by User Prefill{title_suffix}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 100)

    # Add percentage labels
    for i, (_, row) in enumerate(refusal_stats.iterrows()):
        if row["non_refusal_rate"] > 10:
            ax.text(row["non_refusal_rate"] / 2, i, f'{row["non_refusal_rate"]:.1f}%',
                   ha="center", va="center", fontweight="bold", fontsize=9)
        if row["refusal_rate"] > 10:
            ax.text(row["non_refusal_rate"] + row["refusal_rate"] / 2, i, f'{row["refusal_rate"]:.1f}%',
                   ha="center", va="center", fontweight="bold", fontsize=9, color="white")

    plt.tight_layout()
    suffix = f"_{attack_type}" if attack_type else ""
    plt.savefig(PLOTS_DIR / f"02_refusal_rates_by_prefill{suffix}.png", dpi=300, bbox_inches="tight")
    plt.show()

    return refusal_stats


def plot_fact_status_distribution(df: pd.DataFrame, attack_type: str = None):
    """Plot fact status distribution for non-refusal responses."""
    if attack_type:
        plot_df = df[(df["attack_type"] == attack_type) & (df["is_refusal"] == False) & (df["total_facts"] > 0)].copy()
        title_suffix = f" ({attack_type.title()} Attack)"
    else:
        plot_df = df[(df["is_refusal"] == False) & (df["total_facts"] > 0)].copy()
        title_suffix = ""

    if plot_df.empty:
        print(f"No non-refusal responses with fact checks found{title_suffix}")
        return None

    # Aggregate totals
    total_mentioned = plot_df["facts_mentioned"].sum()
    total_not_mentioned = plot_df["facts_not_mentioned"].sum()
    total_lies = plot_df["facts_lies"].sum()
    total = total_mentioned + total_not_mentioned + total_lies

    if total == 0:
        print(f"No facts found{title_suffix}")
        return None

    # Calculate percentages
    pcts = {
        "Mentioned": total_mentioned / total * 100,
        "Not Mentioned": total_not_mentioned / total * 100,
        "Lies": total_lies / total * 100,
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#51cf66', '#ffd43b', '#ff6b6b']
    labels = list(pcts.keys())
    values = list(pcts.values())

    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.5)

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title(f"Fact Status Distribution (Non-Refusal Responses){title_suffix}", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(values) * 1.15)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{val:.1f}%',
               ha="center", va="bottom", fontweight="bold", fontsize=12)

    plt.tight_layout()
    suffix = f"_{attack_type}" if attack_type else ""
    plt.savefig(PLOTS_DIR / f"03_fact_status_distribution{suffix}.png", dpi=300, bbox_inches="tight")
    plt.show()

    return pcts


def plot_response_classification(df: pd.DataFrame, attack_type: str = None):
    """Plot response classification distribution."""
    if attack_type:
        plot_df = df[df["attack_type"] == attack_type].copy()
        title_suffix = f" ({attack_type.title()} Attack)"
    else:
        plot_df = df.copy()
        title_suffix = ""

    # Filter out unknown classifications
    valid_df = plot_df[plot_df["classification"] != "unknown"].copy()

    # Calculate classification counts
    class_counts = valid_df["classification"].value_counts()

    # Ensure all classifications exist
    for cls in ["refusal", "correct", "partial", "evasive", "lie"]:
        if cls not in class_counts:
            class_counts[cls] = 0

    # Reorder
    order = ["refusal", "correct", "partial", "evasive", "lie"]
    class_counts = class_counts[order]
    class_pcts = class_counts / class_counts.sum() * 100

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
        "lie": "Lie (at least one)",
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(order))
    bar_colors = [colors[c] for c in order]

    bars = ax.bar(x, class_pcts.values, color=bar_colors, edgecolor="black", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in order], fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title(f"Response Classification Distribution{title_suffix}", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(class_pcts.values) * 1.15)

    for bar, val in zip(bars, class_pcts.values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{val:.1f}%',
                   ha="center", va="bottom", fontweight="bold", fontsize=11)

    plt.tight_layout()
    suffix = f"_{attack_type}" if attack_type else ""
    plt.savefig(PLOTS_DIR / f"04_response_classification{suffix}.png", dpi=300, bbox_inches="tight")
    plt.show()

    return class_pcts


def plot_response_classification_combined(df: pd.DataFrame):
    """Plot response classification comparison between attack types with stacked bars."""
    attack_types = df["attack_type"].unique()
    if len(attack_types) < 2:
        print("Need multiple attack types for combined plot")
        return None

    # Filter out unknown classifications
    valid_df = df[df["classification"] != "unknown"].copy()

    # Calculate classification counts per attack type
    class_counts = valid_df.groupby(["attack_type", "classification"]).size().unstack(fill_value=0)

    # Ensure all classification columns exist
    for col in ["refusal", "correct", "partial", "evasive", "lie"]:
        if col not in class_counts.columns:
            class_counts[col] = 0

    # Reorder columns
    class_counts = class_counts[["refusal", "correct", "partial", "evasive", "lie"]]

    # Calculate percentages
    class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

    # Sort by refusal rate (descending so simple is first if it has higher refusal)
    class_pcts = class_pcts.sort_values("refusal", ascending=False)

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
        "lie": "Lie (at least one)",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(class_pcts))
    width = 0.5

    bottom = np.zeros(len(class_pcts))

    for classification in ["refusal", "correct", "partial", "evasive", "lie"]:
        values = class_pcts[classification].values
        ax.bar(x, values, width, label=labels[classification], color=colors[classification], bottom=bottom)

        # Add labels for segments > 5%
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 5:
                ax.text(i, bot + val / 2, f'{val:.1f}%',
                       ha="center", va="center", fontweight="bold", fontsize=10)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels([t.title() + " Attack" for t in class_pcts.index], fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Response Classification Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_response_classification_combined.png", dpi=300, bbox_inches="tight")
    plt.show()

    return class_pcts


def plot_classification_by_prefill(df: pd.DataFrame, attack_type: str = None):
    """Plot response classification breakdown by user prefill type."""
    if attack_type:
        plot_df = df[df["attack_type"] == attack_type].copy()
        title_suffix = f" ({attack_type.title()} Attack)"
    else:
        plot_df = df.copy()
        title_suffix = ""

    # Filter out unknown classifications
    valid_df = plot_df[plot_df["classification"] != "unknown"].copy()

    # Calculate classification counts per prefill
    class_counts = valid_df.groupby(["user_prefill", "classification"]).size().unstack(fill_value=0)

    # Ensure all classification columns exist
    for col in ["refusal", "correct", "partial", "evasive", "lie"]:
        if col not in class_counts.columns:
            class_counts[col] = 0

    # Reorder columns
    class_counts = class_counts[["refusal", "correct", "partial", "evasive", "lie"]]

    # Calculate percentages
    class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

    # Sort by refusal rate
    class_pcts = class_pcts.sort_values("refusal", ascending=True)

    fig, ax = plt.subplots(figsize=(14, max(6, len(class_pcts) * 0.6)))

    colors = {
        "refusal": "#ff6b6b",
        "correct": "#51cf66",
        "partial": "#74c0fc",
        "evasive": "#ffd43b",
        "lie": "#e64980",
    }

    labels = {
        "refusal": "Refusal",
        "correct": "Correct",
        "partial": "Partial",
        "evasive": "Evasive",
        "lie": "Lie",
    }

    y = np.arange(len(class_pcts))
    height = 0.6
    left = np.zeros(len(class_pcts))

    for classification in ["refusal", "correct", "partial", "evasive", "lie"]:
        values = class_pcts[classification].values
        ax.barh(y, values, height, left=left, label=labels[classification], color=colors[classification])

        # Add labels for large segments
        for i, (val, l) in enumerate(zip(values, left)):
            if val > 8:
                ax.text(l + val / 2, i, f'{val:.0f}%',
                       ha="center", va="center", fontweight="bold", fontsize=9)
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(class_pcts.index, fontsize=10)
    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_title(f"Response Classification by User Prefill{title_suffix}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.12, 1))
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    suffix = f"_{attack_type}" if attack_type else ""
    plt.savefig(PLOTS_DIR / f"05_classification_by_prefill{suffix}.png", dpi=300, bbox_inches="tight")
    plt.show()

    return class_pcts


def plot_classification_by_topic(df: pd.DataFrame, attack_type: str = None):
    """Plot response classification by topic."""
    if attack_type:
        plot_df = df[df["attack_type"] == attack_type].copy()
        title_suffix = f" ({attack_type.title()} Attack)"
    else:
        plot_df = df.copy()
        title_suffix = ""

    # Filter out unknown classifications
    valid_df = plot_df[plot_df["classification"] != "unknown"].copy()

    # Calculate classification counts per topic
    class_counts = valid_df.groupby(["topic", "classification"]).size().unstack(fill_value=0)

    # Ensure all classification columns exist
    for col in ["refusal", "correct", "partial", "evasive", "lie"]:
        if col not in class_counts.columns:
            class_counts[col] = 0

    # Reorder columns
    class_counts = class_counts[["refusal", "correct", "partial", "evasive", "lie"]]

    # Calculate percentages
    class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

    # Sort by refusal rate
    class_pcts = class_pcts.sort_values("refusal", ascending=True)

    fig, ax = plt.subplots(figsize=(14, max(6, len(class_pcts) * 0.5)))

    colors = {
        "refusal": "#ff6b6b",
        "correct": "#51cf66",
        "partial": "#74c0fc",
        "evasive": "#ffd43b",
        "lie": "#e64980",
    }

    labels = {
        "refusal": "Refusal",
        "correct": "Correct",
        "partial": "Partial",
        "evasive": "Evasive",
        "lie": "Lie",
    }

    y = np.arange(len(class_pcts))
    height = 0.6
    left = np.zeros(len(class_pcts))

    for classification in ["refusal", "correct", "partial", "evasive", "lie"]:
        values = class_pcts[classification].values
        ax.barh(y, values, height, left=left, label=labels[classification], color=colors[classification])

        for i, (val, l) in enumerate(zip(values, left)):
            if val > 8:
                ax.text(l + val / 2, i, f'{val:.0f}%',
                       ha="center", va="center", fontweight="bold", fontsize=9)
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels([t.replace("_", " ").title() for t in class_pcts.index], fontsize=10)
    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_title(f"Response Classification by Topic{title_suffix}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.12, 1))
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    suffix = f"_{attack_type}" if attack_type else ""
    plt.savefig(PLOTS_DIR / f"06_classification_by_topic{suffix}.png", dpi=300, bbox_inches="tight")
    plt.show()

    return class_pcts


def plot_summary_comparison(df: pd.DataFrame):
    """Create a summary comparison plot for all attack types."""
    summary = []

    for attack_type in df["attack_type"].unique():
        type_df = df[df["attack_type"] == attack_type]
        total = len(type_df)

        refusals = (type_df["is_refusal"] == True).sum()
        non_refusals = (type_df["is_refusal"] == False).sum()

        # Classification stats
        valid_df = type_df[type_df["classification"] != "unknown"]
        correct = (valid_df["classification"] == "correct").sum()
        partial = (valid_df["classification"] == "partial").sum()
        evasive = (valid_df["classification"] == "evasive").sum()
        lies = (valid_df["classification"] == "lie").sum()

        # Fact stats for non-refusals
        nr_df = type_df[(type_df["is_refusal"] == False) & (type_df["total_facts"] > 0)]
        total_facts = nr_df["total_facts"].sum()
        mentioned = nr_df["facts_mentioned"].sum()
        not_mentioned = nr_df["facts_not_mentioned"].sum()
        lie_facts = nr_df["facts_lies"].sum()

        summary.append({
            "Attack Type": attack_type.title(),
            "Total Responses": total,
            "Refusals": refusals,
            "Refusal Rate (%)": refusals / total * 100 if total > 0 else 0,
            "Non-Refusals": non_refusals,
            "Correct": correct,
            "Partial": partial,
            "Evasive": evasive,
            "Lies (responses)": lies,
            "Total Facts Checked": total_facts,
            "Facts Mentioned": mentioned,
            "Facts Not Mentioned": not_mentioned,
            "Facts Lied About": lie_facts,
            "Fact Mention Rate (%)": mentioned / total_facts * 100 if total_facts > 0 else 0,
            "Lie Rate (%)": lie_facts / total_facts * 100 if total_facts > 0 else 0,
        })

    summary_df = pd.DataFrame(summary)

    print("\n" + "=" * 100)
    print("USER PREFILL ATTACK SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    summary_df.to_csv(PLOTS_DIR / "summary_metrics.csv", index=False)
    print(f"\nExported summary to: {PLOTS_DIR / 'summary_metrics.csv'}")

    # Create combined plot with key metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    order = summary_df["Attack Type"].tolist()

    # Plot 1: Refusal Rate
    ax1 = axes[0]
    colors1 = ['#ff6b6b' if r > 50 else '#ffd43b' if r > 25 else '#51cf66'
               for r in summary_df["Refusal Rate (%)"].values]
    bars1 = ax1.bar(order, summary_df["Refusal Rate (%)"], color=colors1, edgecolor="black")
    ax1.set_ylabel("Rate (%)", fontsize=11)
    ax1.set_title("Refusal Rate", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, summary_df["Refusal Rate (%)"].values):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
                ha="center", fontsize=10, fontweight="bold")

    # Plot 2: Fact Mention Rate
    ax2 = axes[1]
    colors2 = ['#51cf66' if r > 50 else '#ffd43b' if r > 25 else '#ff6b6b'
               for r in summary_df["Fact Mention Rate (%)"].values]
    bars2 = ax2.bar(order, summary_df["Fact Mention Rate (%)"], color=colors2, edgecolor="black")
    ax2.set_ylabel("Rate (%)", fontsize=11)
    ax2.set_title("Fact Mention Rate\n(Non-Refusals)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, summary_df["Fact Mention Rate (%)"].values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
                ha="center", fontsize=10, fontweight="bold")

    # Plot 3: Lie Rate
    ax3 = axes[2]
    colors3 = ['#ff6b6b' if r > 25 else '#ffd43b' if r > 10 else '#51cf66'
               for r in summary_df["Lie Rate (%)"].values]
    bars3 = ax3.bar(order, summary_df["Lie Rate (%)"], color=colors3, edgecolor="black")
    ax3.set_ylabel("Rate (%)", fontsize=11)
    ax3.set_title("Lie Rate\n(Non-Refusals)", fontsize=12, fontweight="bold")
    ax3.set_ylim(0, max(summary_df["Lie Rate (%)"].max() * 1.3, 10))
    ax3.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars3, summary_df["Lie Rate (%)"].values):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}%',
                ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("User Prefill Attack: Key Metrics Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_key_metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    return summary_df


def main():
    print("Loading user prefill attack results...")

    # Load both attack types
    simple_data = load_simple_attack_results()
    regular_data = load_regular_attack_results()

    dfs = []

    if simple_data:
        print("\nExtracting simple attack statistics...")
        simple_df = extract_simple_attack_stats(simple_data)
        dfs.append(simple_df)
        print(f"Simple attack responses: {len(simple_df)}")

    if regular_data:
        print("\nExtracting regular attack statistics...")
        regular_df = extract_regular_attack_stats(regular_data)
        dfs.append(regular_df)
        print(f"Regular attack responses: {len(regular_df)}")

    if not dfs:
        print("No results found!")
        return

    # Combine all dataframes
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal responses: {len(df)}")

    attack_types = df["attack_type"].unique()
    has_multiple_types = len(attack_types) > 1

    # Plot overall comparison if we have multiple attack types
    if has_multiple_types:
        print("\n" + "=" * 60)
        print("PLOT: Refusal Rates by Attack Type")
        print("=" * 60)
        plot_refusal_rates_by_attack_type(df)

        print("\n" + "=" * 60)
        print("PLOT: Response Classification Combined")
        print("=" * 60)
        plot_response_classification_combined(df)

    # Plot for each attack type
    for attack_type in attack_types:
        print(f"\n" + "=" * 60)
        print(f"PLOTS FOR {attack_type.upper()} ATTACK")
        print("=" * 60)

        print(f"\nPlot: Refusal Rates by Prefill ({attack_type})")
        plot_refusal_rates_by_prefill(df, attack_type)

        print(f"\nPlot: Fact Status Distribution ({attack_type})")
        plot_fact_status_distribution(df, attack_type)

        print(f"\nPlot: Response Classification ({attack_type})")
        plot_response_classification(df, attack_type)

        print(f"\nPlot: Classification by Prefill ({attack_type})")
        plot_classification_by_prefill(df, attack_type)

        print(f"\nPlot: Classification by Topic ({attack_type})")
        plot_classification_by_topic(df, attack_type)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary_df = plot_summary_comparison(df)

    # Export full dataframe
    df.to_csv(PLOTS_DIR / "response_stats.csv", index=False)
    print(f"\nExported response stats to: {PLOTS_DIR / 'response_stats.csv'}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
