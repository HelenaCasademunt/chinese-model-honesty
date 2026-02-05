"""
Plotting script for pretrain prompt attack results.
Compares refusal rates and fact-based metrics across different pretrain prompts.
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
PLOTS_DIR = Path("plots/pretrain_prompt_attack")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("results/data/pretrain_prompts")


def load_pretrain_prompt_results():
    """Load all pretrain prompt evaluation results."""
    results = {}

    # Map prompt tags to display names
    prompt_display_names = {
        "censored_chinese_ai": "Censored Chinese AI",
        "author_roleplay": "Author Roleplay",
        "claude_transcript": "Claude Transcript",
    }

    # Look for evaluated files (prefer these over raw files)
    evaluated_tags = set()
    for filepath in DATA_DIR.glob("evaluated_pretrain_*_responses.json"):
        filename = filepath.name
        tag = filename.replace("evaluated_pretrain_", "").replace("_responses.json", "")
        evaluated_tags.add(tag)

        display_name = prompt_display_names.get(tag, tag.replace("_", " ").title())

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Data is a flat list of questions
        data = [q for q in data if q.get("topic") != "test"]
        results[display_name] = data
        print(f"Loaded {display_name} (evaluated): {len(data)} questions")

    # Load raw files only if no evaluated version exists
    for filepath in DATA_DIR.glob("pretrain_*_responses.json"):
        filename = filepath.name
        if filename.startswith("evaluated_"):
            continue
        tag = filename.replace("pretrain_", "").replace("_responses.json", "")

        if tag in evaluated_tags:
            continue  # Skip, we have evaluated version

        display_name = prompt_display_names.get(tag, tag.replace("_", " ").title())

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Data is a flat list of questions
        data = [q for q in data if q.get("topic") != "test"]
        results[display_name] = data
        print(f"Loaded {display_name} (raw): {len(data)} questions")

    return results


def extract_response_stats(results: dict) -> pd.DataFrame:
    """Extract statistics from all responses across pretrain prompts."""
    stats = []

    for prompt_name, questions in results.items():
        for question in questions:
            q_id = question["question_id"]
            topic = question["topic"]
            level = question.get("level", "")

            # Use evaluated_responses if available, otherwise fall back to model_responses
            evaluated_responses = question.get("evaluated_responses", [])

            if evaluated_responses:
                # Process evaluated responses
                for resp in evaluated_responses:
                    evaluation = resp.get("evaluation", {})
                    is_refusal = evaluation.get("is_refusal", False)
                    fact_checks = evaluation.get("fact_checks", [])

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
                        "pretrain_prompt": prompt_name,
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
            else:
                # Raw responses (not evaluated) - add placeholder entries
                for _ in question.get("model_responses", []):
                    stats.append({
                        "pretrain_prompt": prompt_name,
                        "question_id": q_id,
                        "topic": topic,
                        "level": level,
                        "is_refusal": None,
                        "facts_mentioned": 0,
                        "facts_not_mentioned": 0,
                        "facts_lies": 0,
                        "total_facts": 0,
                        "classification": "unevaluated",
                    })

    return pd.DataFrame(stats)


def plot_refusal_rates(df: pd.DataFrame):
    """Plot refusal rates comparison across pretrain prompts."""
    # Filter to evaluated responses only
    eval_df = df[df["is_refusal"].notna()].copy()

    if eval_df.empty:
        print("No evaluated responses found - skipping refusal rate plot")
        return None

    refusal_stats = eval_df.groupby("pretrain_prompt").agg({
        "is_refusal": ["sum", "count"]
    }).reset_index()
    refusal_stats.columns = ["pretrain_prompt", "refusals", "total"]
    refusal_stats["refusal_rate"] = refusal_stats["refusals"] / refusal_stats["total"] * 100
    refusal_stats["non_refusal_rate"] = 100 - refusal_stats["refusal_rate"]

    # Order by refusal rate
    refusal_stats = refusal_stats.sort_values("refusal_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(refusal_stats))
    width = 0.6

    colors = ['#51cf66', '#ff6b6b']  # green for non-refusal, red for refusal

    # Stacked bar chart
    bars1 = ax.bar(x, refusal_stats["non_refusal_rate"], width, label="Non-Refusal", color=colors[0])
    bars2 = ax.bar(x, refusal_stats["refusal_rate"], width, bottom=refusal_stats["non_refusal_rate"],
                   label="Refusal", color=colors[1])

    ax.set_xticks(x)
    ax.set_xticklabels(refusal_stats["pretrain_prompt"], fontsize=11, rotation=15, ha="right")
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Refusal Rates by Pretrain Prompt", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    # Add percentage labels
    for i, (_, row) in enumerate(refusal_stats.iterrows()):
        if row["non_refusal_rate"] > 5:
            ax.text(i, row["non_refusal_rate"] / 2, f'{row["non_refusal_rate"]:.1f}%',
                   ha="center", va="center", fontweight="bold", fontsize=10)
        if row["refusal_rate"] > 5:
            ax.text(i, row["non_refusal_rate"] + row["refusal_rate"] / 2, f'{row["refusal_rate"]:.1f}%',
                   ha="center", va="center", fontweight="bold", fontsize=10, color="white")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_refusal_rates.png", dpi=300, bbox_inches="tight")
    plt.show()

    return refusal_stats


def plot_fact_counts(df: pd.DataFrame):
    """Plot fact counts (mentioned, not_mentioned, lies) for non-refusal responses."""
    # Filter to non-refusal responses with fact checks
    non_refusal_df = df[(df["is_refusal"] == False) & (df["total_facts"] > 0)].copy()

    if non_refusal_df.empty:
        print("No non-refusal responses with fact checks found - skipping fact count plot")
        return None

    # Aggregate by pretrain prompt
    fact_stats = non_refusal_df.groupby("pretrain_prompt").agg({
        "facts_mentioned": "sum",
        "facts_not_mentioned": "sum",
        "facts_lies": "sum",
    }).reset_index()

    fact_stats["total"] = (fact_stats["facts_mentioned"] +
                           fact_stats["facts_not_mentioned"] +
                           fact_stats["facts_lies"])

    # Calculate percentages
    fact_stats["mentioned_pct"] = fact_stats["facts_mentioned"] / fact_stats["total"] * 100
    fact_stats["not_mentioned_pct"] = fact_stats["facts_not_mentioned"] / fact_stats["total"] * 100
    fact_stats["lies_pct"] = fact_stats["facts_lies"] / fact_stats["total"] * 100

    # Sort by mentioned percentage
    fact_stats = fact_stats.sort_values("mentioned_pct", ascending=True)

    # Plot: Stacked percentage bars
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(fact_stats))
    width = 0.6

    colors = ['#51cf66', '#ffd43b', '#ff6b6b']  # green, yellow, red

    bars1 = ax.bar(x, fact_stats["mentioned_pct"], width, label="Mentioned", color=colors[0])
    bars2 = ax.bar(x, fact_stats["not_mentioned_pct"], width,
                   bottom=fact_stats["mentioned_pct"], label="Not Mentioned", color=colors[1])
    bars3 = ax.bar(x, fact_stats["lies_pct"], width,
                   bottom=fact_stats["mentioned_pct"] + fact_stats["not_mentioned_pct"],
                   label="Lies", color=colors[2])

    ax.set_xticks(x)
    ax.set_xticklabels(fact_stats["pretrain_prompt"], fontsize=11, rotation=15, ha="right")
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Fact Status Distribution (Non-Refusal Responses)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    # Add percentage labels
    for i, (_, row) in enumerate(fact_stats.iterrows()):
        cumulative = 0
        for pct, label in [(row["mentioned_pct"], "mentioned"),
                           (row["not_mentioned_pct"], "not_mentioned"),
                           (row["lies_pct"], "lies")]:
            if pct > 5:
                ax.text(i, cumulative + pct / 2, f'{pct:.1f}%',
                       ha="center", va="center", fontweight="bold", fontsize=9)
            cumulative += pct

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_fact_status_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    return fact_stats


def plot_response_classification(df: pd.DataFrame):
    """Plot response classification distribution across pretrain prompts."""
    # Filter out unknown and unevaluated classifications
    valid_df = df[~df["classification"].isin(["unknown", "unevaluated"])].copy()

    if valid_df.empty:
        print("No valid classifications found - skipping classification plot")
        return None

    # Calculate classification counts per pretrain prompt
    class_counts = valid_df.groupby(["pretrain_prompt", "classification"]).size().unstack(fill_value=0)

    # Ensure all classification columns exist
    for col in ["refusal", "correct", "partial", "evasive", "lie"]:
        if col not in class_counts.columns:
            class_counts[col] = 0

    # Reorder columns
    class_counts = class_counts[["refusal", "correct", "partial", "evasive", "lie"]]

    # Calculate percentages
    class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

    # Sort by correct percentage (after excluding refusals)
    non_refusal_correct = class_counts[["correct", "partial", "evasive", "lie"]].sum(axis=1)
    class_pcts["sort_key"] = class_counts["correct"] / non_refusal_correct.replace(0, 1) * 100
    class_pcts = class_pcts.sort_values("sort_key", ascending=True)
    class_pcts = class_pcts.drop("sort_key", axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(class_pcts))
    width = 0.6

    colors = {
        "refusal": "#ff6b6b",     # red
        "correct": "#51cf66",     # green
        "partial": "#74c0fc",     # blue
        "evasive": "#ffd43b",     # yellow
        "lie": "#e64980",         # dark red/pink
    }

    labels = {
        "refusal": "Refusal",
        "correct": "Correct (all facts)",
        "partial": "Partial (some facts)",
        "evasive": "Evasive (no facts)",
        "lie": "Lie (at least one)",
    }

    bottom = np.zeros(len(class_pcts))

    for classification in ["refusal", "correct", "partial", "evasive", "lie"]:
        values = class_pcts[classification].values
        ax.bar(x, values, width, label=labels[classification], color=colors[classification], bottom=bottom)

        # Add labels for large segments
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 5:
                ax.text(i, bot + val / 2, f'{val:.1f}%',
                       ha="center", va="center", fontweight="bold", fontsize=9)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(class_pcts.index, fontsize=11, rotation=15, ha="right")
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Response Classification by Pretrain Prompt", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_response_classification.png", dpi=300, bbox_inches="tight")
    plt.show()

    return class_pcts


def plot_non_refusal_classification(df: pd.DataFrame):
    """Plot classification distribution for non-refusal responses only."""
    # Filter to non-refusal responses with fact checks
    non_refusal_df = df[(df["is_refusal"] == False) & (~df["classification"].isin(["unknown", "unevaluated"]))].copy()

    if non_refusal_df.empty:
        print("No non-refusal responses with fact checks found - skipping non-refusal classification plot")
        return None

    # Calculate classification counts per pretrain prompt
    class_counts = non_refusal_df.groupby(["pretrain_prompt", "classification"]).size().unstack(fill_value=0)

    # Ensure all classification columns exist
    for col in ["correct", "partial", "evasive", "lie"]:
        if col not in class_counts.columns:
            class_counts[col] = 0

    # Reorder columns
    class_counts = class_counts[["correct", "partial", "evasive", "lie"]]

    # Calculate percentages
    class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

    # Sort by correct percentage
    class_pcts = class_pcts.sort_values("correct", ascending=True)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(class_pcts))
    width = 0.6

    colors = {
        "correct": "#51cf66",     # green
        "partial": "#74c0fc",     # blue
        "evasive": "#ffd43b",     # yellow
        "lie": "#e64980",         # dark red/pink
    }

    labels = {
        "correct": "Correct (all facts)",
        "partial": "Partial (some facts)",
        "evasive": "Evasive (no facts)",
        "lie": "Lie (at least one)",
    }

    bottom = np.zeros(len(class_pcts))

    for classification in ["correct", "partial", "evasive", "lie"]:
        values = class_pcts[classification].values
        ax.bar(x, values, width, label=labels[classification], color=colors[classification], bottom=bottom)

        # Add labels for large segments
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 5:
                ax.text(i, bot + val / 2, f'{val:.1f}%',
                       ha="center", va="center", fontweight="bold", fontsize=9)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(class_pcts.index, fontsize=11, rotation=15, ha="right")
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Non-Refusal Response Classification by Pretrain Prompt", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_non_refusal_classification.png", dpi=300, bbox_inches="tight")
    plt.show()

    return class_pcts


def plot_by_topic(df: pd.DataFrame):
    """Plot classification distribution by topic for each pretrain prompt."""
    # Filter to evaluated responses
    valid_df = df[~df["classification"].isin(["unknown", "unevaluated"])].copy()

    if valid_df.empty:
        print("No valid classifications found - skipping topic plot")
        return None

    prompts = valid_df["pretrain_prompt"].unique()

    for prompt in prompts:
        prompt_df = valid_df[valid_df["pretrain_prompt"] == prompt]

        # Calculate classification counts per topic
        class_counts = prompt_df.groupby(["topic", "classification"]).size().unstack(fill_value=0)

        # Ensure all classification columns exist
        for col in ["refusal", "correct", "partial", "evasive", "lie"]:
            if col not in class_counts.columns:
                class_counts[col] = 0

        class_counts = class_counts[["refusal", "correct", "partial", "evasive", "lie"]]

        # Calculate percentages
        class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

        # Sort by correct percentage
        class_pcts = class_pcts.sort_values("correct", ascending=True)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))

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
            "correct": "Correct",
            "partial": "Partial",
            "evasive": "Evasive",
            "lie": "Lie",
        }

        bottom = np.zeros(len(class_pcts))

        for classification in ["refusal", "correct", "partial", "evasive", "lie"]:
            values = class_pcts[classification].values
            ax.bar(x, values, width, label=labels[classification], color=colors[classification], bottom=bottom)
            bottom += values

        ax.set_xticks(x)
        ax.set_xticklabels(class_pcts.index, fontsize=10, rotation=45, ha="right")
        ax.set_ylabel("Percentage (%)", fontsize=12)
        ax.set_title(f"Classification by Topic - {prompt}", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.12, 1))
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        safe_name = prompt.lower().replace(" ", "_").replace("&", "and")
        plt.savefig(PLOTS_DIR / f"05_classification_by_topic_{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.show()


def plot_summary_comparison(df: pd.DataFrame):
    """Create a summary comparison plot."""
    # Calculate overall stats per pretrain prompt
    summary = []

    for prompt in df["pretrain_prompt"].unique():
        prompt_df = df[df["pretrain_prompt"] == prompt]
        total = len(prompt_df)

        # Count evaluated vs unevaluated
        eval_df = prompt_df[prompt_df["is_refusal"].notna()]
        unevaluated = (prompt_df["classification"] == "unevaluated").sum()

        refusals = (eval_df["is_refusal"] == True).sum()
        non_refusals = (eval_df["is_refusal"] == False).sum()

        # Classification stats (excluding unknown and unevaluated)
        valid_df = prompt_df[~prompt_df["classification"].isin(["unknown", "unevaluated"])]
        correct = (valid_df["classification"] == "correct").sum()
        partial = (valid_df["classification"] == "partial").sum()
        evasive = (valid_df["classification"] == "evasive").sum()
        lies = (valid_df["classification"] == "lie").sum()

        # Fact stats for non-refusals
        nr_df = prompt_df[(prompt_df["is_refusal"] == False) & (prompt_df["total_facts"] > 0)]
        total_facts = nr_df["total_facts"].sum()
        mentioned = nr_df["facts_mentioned"].sum()
        not_mentioned = nr_df["facts_not_mentioned"].sum()
        lie_facts = nr_df["facts_lies"].sum()

        eval_total = len(eval_df)
        summary.append({
            "Pretrain Prompt": prompt,
            "Total Responses": total,
            "Evaluated": eval_total,
            "Unevaluated": unevaluated,
            "Refusals": refusals,
            "Refusal Rate (%)": refusals / eval_total * 100 if eval_total > 0 else 0,
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
    summary_df = summary_df.sort_values("Refusal Rate (%)", ascending=True)

    # Print summary table
    print("\n" + "=" * 100)
    print("PRETRAIN PROMPT ATTACK SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    # Export to CSV
    summary_df.to_csv(PLOTS_DIR / "summary_metrics.csv", index=False)
    print(f"\nExported summary to: {PLOTS_DIR / 'summary_metrics.csv'}")

    # Only plot if we have evaluated data
    if summary_df["Evaluated"].sum() == 0:
        print("No evaluated data - skipping key metrics plot")
        return summary_df

    # Create combined plot with key metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Order for all plots
    order = summary_df["Pretrain Prompt"].tolist()

    # Plot 1: Refusal Rate
    ax1 = axes[0]
    colors1 = ['#ff6b6b' if r > 50 else '#ffd43b' if r > 25 else '#51cf66'
               for r in summary_df["Refusal Rate (%)"].values]
    bars1 = ax1.barh(order, summary_df["Refusal Rate (%)"], color=colors1, edgecolor="black")
    ax1.set_xlabel("Refusal Rate (%)", fontsize=11)
    ax1.set_title("Refusal Rate", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 100)
    ax1.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars1, summary_df["Refusal Rate (%)"].values):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va="center", fontsize=10)

    # Plot 2: Fact Mention Rate (for non-refusals)
    ax2 = axes[1]
    colors2 = ['#51cf66' if r > 50 else '#ffd43b' if r > 25 else '#ff6b6b'
               for r in summary_df["Fact Mention Rate (%)"].values]
    bars2 = ax2.barh(order, summary_df["Fact Mention Rate (%)"], color=colors2, edgecolor="black")
    ax2.set_xlabel("Fact Mention Rate (%)", fontsize=11)
    ax2.set_title("Fact Mention Rate\n(Non-Refusals)", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.grid(axis="x", alpha=0.3)
    ax2.set_yticklabels([])
    for bar, val in zip(bars2, summary_df["Fact Mention Rate (%)"].values):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va="center", fontsize=10)

    # Plot 3: Lie Rate (for non-refusals)
    ax3 = axes[2]
    colors3 = ['#ff6b6b' if r > 25 else '#ffd43b' if r > 10 else '#51cf66'
               for r in summary_df["Lie Rate (%)"].values]
    bars3 = ax3.barh(order, summary_df["Lie Rate (%)"], color=colors3, edgecolor="black")
    ax3.set_xlabel("Lie Rate (%)", fontsize=11)
    ax3.set_title("Lie Rate\n(Non-Refusals)", fontsize=12, fontweight="bold")
    ax3.set_xlim(0, max(summary_df["Lie Rate (%)"].max() * 1.2, 10))
    ax3.grid(axis="x", alpha=0.3)
    ax3.set_yticklabels([])
    for bar, val in zip(bars3, summary_df["Lie Rate (%)"].values):
        ax3.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va="center", fontsize=10)

    plt.suptitle("Pretrain Prompt Attack: Key Metrics Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_key_metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    return summary_df


def main():
    print("Loading pretrain prompt attack results...")
    results = load_pretrain_prompt_results()

    if not results:
        print("No results found!")
        return

    print("\nExtracting response statistics...")
    df = extract_response_stats(results)
    print(f"Total responses: {len(df)}")

    # Check if we have any evaluated data
    has_evaluated = df["is_refusal"].notna().any()

    if not has_evaluated:
        print("\n" + "=" * 60)
        print("WARNING: No evaluated data found!")
        print("Run the evaluation script first to generate evaluated results.")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("PLOT 1: Refusal Rates")
    print("=" * 60)
    refusal_stats = plot_refusal_rates(df)

    print("\n" + "=" * 60)
    print("PLOT 2: Fact Status Distribution")
    print("=" * 60)
    fact_stats = plot_fact_counts(df)

    print("\n" + "=" * 60)
    print("PLOT 3: Response Classification (All)")
    print("=" * 60)
    class_pcts = plot_response_classification(df)

    print("\n" + "=" * 60)
    print("PLOT 4: Response Classification (Non-Refusals Only)")
    print("=" * 60)
    nr_class_pcts = plot_non_refusal_classification(df)

    print("\n" + "=" * 60)
    print("PLOT 5: Classification by Topic")
    print("=" * 60)
    plot_by_topic(df)

    print("\n" + "=" * 60)
    print("PLOT 6: Summary Comparison")
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
