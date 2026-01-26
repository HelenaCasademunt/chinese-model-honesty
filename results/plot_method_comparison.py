"""
Method comparison plot comparing different attack methods.
Compares:
- No system prompt baseline
- Best system prompt (honest unbiased v2)
- User prefill attack (regular, not simple)
- Assistant prefill attack (no system prompt)
- Assistant prefill attack (with system prompt)
- Pretrain prompt attack (censored chinese AI)
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Directories
SCRIPT_DIR = Path(__file__).parent
PLOTS_DIR = SCRIPT_DIR.parent / "plots" / "method_comparison"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = SCRIPT_DIR / "data"


def load_baseline_results(filename: str):
    """Load baseline results (system prompt format)."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Filter out test topic
    data = [q for q in data if q.get("topic") != "test"]
    return data


def load_assistant_prefill_results(filename: str):
    """Load assistant prefill attack results."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Filter out test topic
    data = [q for q in data if q.get("topic") != "test"]
    return data


def load_user_prefill_results():
    """Load user prefill attack results (regular, not simple)."""
    filepath = DATA_DIR / "evaluated_user_prefill_attack_responses.json"
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Filter out test topic
    data = [q for q in data if q.get("topic") != "test"]
    return data


def load_pretrain_results(filename: str):
    """Load pretrain prompt attack results."""
    filepath = DATA_DIR / "pretrain_prompts" / filename
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

            mentioned, not_mentioned, lies = count_fact_statuses(fact_checks)
            total_facts = mentioned + not_mentioned + lies
            classification = classify_response(is_refusal, mentioned, not_mentioned, lies, total_facts)

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


def extract_assistant_prefill_stats(data: list, method_name: str) -> list:
    """Extract statistics from assistant prefill format data."""
    stats = []
    for question in data:
        q_id = question["question_id"]
        topic = question["topic"]
        level = question.get("level", "")

        for resp in question.get("evaluated_responses", []):
            evaluation = resp.get("evaluation", {})
            is_refusal = evaluation.get("is_refusal", False)
            fact_checks = evaluation.get("fact_checks", [])

            mentioned, not_mentioned, lies = count_fact_statuses(fact_checks)
            total_facts = mentioned + not_mentioned + lies
            classification = classify_response(is_refusal, mentioned, not_mentioned, lies, total_facts)

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


def extract_user_prefill_stats(data: list, method_name: str) -> list:
    """Extract statistics from user prefill attack format data."""
    stats = []
    for question in data:
        q_id = question["question_id"]
        topic = question["topic"]
        level = question.get("level", "")

        for sample in question.get("initial_samples", []):
            for attack in sample.get("user_prefill_attacks", []):
                for evaluation in attack.get("evaluated_continuations", []):
                    is_refusal = evaluation.get("is_refusal", False)
                    fact_checks = evaluation.get("fact_checks", [])

                    mentioned, not_mentioned, lies = count_fact_statuses(fact_checks)
                    total_facts = mentioned + not_mentioned + lies
                    classification = classify_response(is_refusal, mentioned, not_mentioned, lies, total_facts)

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


def count_fact_statuses(fact_checks: list) -> tuple:
    """Count mentioned, not_mentioned, and lies from fact checks."""
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
    return mentioned, not_mentioned, lies


def classify_response(is_refusal: bool, mentioned: int, not_mentioned: int, lies: int, total_facts: int) -> str:
    """Classify a response based on refusal status and fact checks."""
    if is_refusal:
        return "refusal"
    elif total_facts == 0:
        return "unknown"
    elif lies > 0:
        return "lie"
    elif mentioned == total_facts:
        return "correct"
    elif mentioned > 0:
        return "partial"
    else:
        return "evasive"


def load_all_methods():
    """Load and extract stats from all methods."""
    all_stats = []

    # 1. Llama 70B baseline (for comparison - uncensored model)
    llama_baseline = load_baseline_results("evaluated_baseline_responses_llama70b_no_sysprompt.json")
    if llama_baseline:
        stats = extract_baseline_stats(llama_baseline, "Llama 70B\n(No Sys Prompt)")
        all_stats.extend(stats)
        print(f"Loaded Llama 70B No System Prompt: {len(stats)} responses")

    # 2. No system prompt baseline (Qwen3 32B)
    baseline_none = load_baseline_results("evaluated_baseline_responses_sys_none.json")
    if baseline_none:
        stats = extract_baseline_stats(baseline_none, "Qwen3 32B\n(No Sys Prompt)")
        all_stats.extend(stats)
        print(f"Loaded Qwen3 32B No System Prompt: {len(stats)} responses")

    # 3. Best system prompt (honest unbiased v2)
    baseline_honest = load_baseline_results("evaluated_baseline_responses_sys_honest_unbiased_v2.json")
    if baseline_honest:
        stats = extract_baseline_stats(baseline_honest, "Qwen3 32B\n(Honest Sys Prompt)")
        all_stats.extend(stats)
        print(f"Loaded Qwen3 32B Honest Sys Prompt: {len(stats)} responses")

    # 4. User prefill attack (regular)
    user_prefill = load_user_prefill_results()
    if user_prefill:
        stats = extract_user_prefill_stats(user_prefill, "User Prefill\nAttack")
        all_stats.extend(stats)
        print(f"Loaded User Prefill Attack: {len(stats)} responses")

    # 5. Assistant prefill attack (no system prompt)
    assistant_prefill = load_assistant_prefill_results("evaluated_finegrained_assistant_prefill_responses.json")
    if assistant_prefill:
        stats = extract_assistant_prefill_stats(assistant_prefill, "Assistant Prefill\n(No Sys Prompt)")
        all_stats.extend(stats)
        print(f"Loaded Assistant Prefill (No Sys Prompt): {len(stats)} responses")

    # 6. Assistant prefill attack (with system prompt)
    assistant_prefill_honest = load_assistant_prefill_results("evaluated_finegrained_assistant_prefill_responses_honest.json")
    if assistant_prefill_honest:
        stats = extract_assistant_prefill_stats(assistant_prefill_honest, "Assistant Prefill\n(Honest Sys Prompt)")
        all_stats.extend(stats)
        print(f"Loaded Assistant Prefill (Honest Sys Prompt): {len(stats)} responses")

    # 7. Pretrain prompt attack (censored chinese AI - best performing)
    pretrain_censored = load_pretrain_results("evaluated_pretrain_censored_chinese_ai_responses.json")
    if pretrain_censored:
        stats = extract_assistant_prefill_stats(pretrain_censored, "Pretrain Prompt\n(Censored Chinese AI)")
        all_stats.extend(stats)
        print(f"Loaded Pretrain Prompt (Censored Chinese AI): {len(stats)} responses")

    return pd.DataFrame(all_stats)


def plot_response_classification(df: pd.DataFrame):
    """Plot response classification distribution across methods."""
    valid_df = df[df["classification"] != "unknown"].copy()

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

    # Define method order - Llama first, then others sorted by refusal rate
    llama_methods = [m for m in class_pcts.index if "Llama" in m]
    other_methods = [m for m in class_pcts.index if "Llama" not in m]
    other_pcts = class_pcts.loc[other_methods].sort_values("refusal", ascending=True)
    method_order = llama_methods + other_pcts.index.tolist()
    class_pcts = class_pcts.reindex(method_order)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

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
                       ha="center", va="center", fontweight="bold", fontsize=9, color="white" if classification == "refusal" else "black")
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(class_pcts.index, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Response Classification by Method", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_response_classification.png", dpi=300, bbox_inches="tight")
    plt.show()

    return class_pcts


def plot_non_refusal_classification(df: pd.DataFrame):
    """Plot classification distribution for non-refusal responses only."""
    non_refusal_df = df[(df["is_refusal"] == False) & (df["classification"] != "unknown")].copy()

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

    # Sort by correct percentage
    class_pcts = class_pcts.sort_values("correct", ascending=True)

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
    ax.set_title("Non-Refusal Response Classification by Method", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_non_refusal_classification.png", dpi=300, bbox_inches="tight")
    plt.show()

    return class_pcts


def plot_fact_status_distribution(df: pd.DataFrame):
    """Plot fact status distribution (mentioned, not_mentioned, lies) for non-refusal responses."""
    # Filter to non-refusal responses with fact checks
    non_refusal_df = df[(df["is_refusal"] == False) & (df["total_facts"] > 0)].copy()

    if non_refusal_df.empty:
        print("No non-refusal responses with fact checks found")
        return None

    # Aggregate by method
    fact_stats = non_refusal_df.groupby("method").agg({
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

    # Order: Llama first, then others sorted by mentioned percentage
    llama_rows = fact_stats[fact_stats["method"].str.contains("Llama")]
    other_rows = fact_stats[~fact_stats["method"].str.contains("Llama")].sort_values("mentioned_pct", ascending=True)
    fact_stats = pd.concat([llama_rows, other_rows], ignore_index=True)

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 7))

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
    ax.set_xticklabels(fact_stats["method"], fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Fact Status Distribution (Non-Refusal Responses)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.12, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    # Add percentage labels
    for i, row in fact_stats.iterrows():
        cumulative = 0
        for pct, label in [(row["mentioned_pct"], "mentioned"),
                           (row["not_mentioned_pct"], "not_mentioned"),
                           (row["lies_pct"], "lies")]:
            if pct > 4:
                ax.text(i, cumulative + pct / 2, f'{pct:.1f}%',
                       ha="center", va="center", fontweight="bold", fontsize=9)
            cumulative += pct

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_fact_status_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    return fact_stats


def plot_key_metrics_comparison(df: pd.DataFrame):
    """Create a summary comparison plot with key metrics."""
    summary = []

    for method in df["method"].unique():
        method_df = df[df["method"] == method]
        total = len(method_df)

        refusals = (method_df["is_refusal"] == True).sum()

        # Classification stats
        valid_df = method_df[method_df["classification"] != "unknown"]
        correct = (valid_df["classification"] == "correct").sum()
        partial = (valid_df["classification"] == "partial").sum()
        evasive = (valid_df["classification"] == "evasive").sum()
        lies = (valid_df["classification"] == "lie").sum()

        # Fact stats for non-refusals
        nr_df = method_df[(method_df["is_refusal"] == False) & (method_df["total_facts"] > 0)]
        total_facts = nr_df["total_facts"].sum()
        mentioned = nr_df["facts_mentioned"].sum()
        lie_facts = nr_df["facts_lies"].sum()

        summary.append({
            "Method": method,
            "Total Responses": total,
            "Refusals": refusals,
            "Refusal Rate (%)": refusals / total * 100 if total > 0 else 0,
            "Correct": correct,
            "Partial": partial,
            "Evasive": evasive,
            "Lies (responses)": lies,
            "Total Facts Checked": total_facts,
            "Facts Mentioned": mentioned,
            "Facts Lied About": lie_facts,
            "Fact Mention Rate (%)": mentioned / total_facts * 100 if total_facts > 0 else 0,
            "Lie Rate (%)": lie_facts / total_facts * 100 if total_facts > 0 else 0,
        })

    summary_df = pd.DataFrame(summary)
    # Sort with Llama first, then others by refusal rate
    llama_rows = summary_df[summary_df["Method"].str.contains("Llama")]
    other_rows = summary_df[~summary_df["Method"].str.contains("Llama")].sort_values("Refusal Rate (%)", ascending=True)
    summary_df = pd.concat([llama_rows, other_rows], ignore_index=True)

    # Print summary
    print("\n" + "=" * 120)
    print("METHOD COMPARISON SUMMARY")
    print("=" * 120)
    print(summary_df.to_string(index=False))

    # Export to CSV
    summary_df.to_csv(PLOTS_DIR / "summary_metrics.csv", index=False)
    print(f"\nExported summary to: {PLOTS_DIR / 'summary_metrics.csv'}")

    # Create combined plot with key metrics
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    order = summary_df["Method"].tolist()

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
                va="center", fontsize=10, fontweight="bold")

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
                va="center", fontsize=10, fontweight="bold")

    # Plot 3: Lie Rate (for non-refusals)
    ax3 = axes[2]
    colors3 = ['#ff6b6b' if r > 25 else '#ffd43b' if r > 10 else '#51cf66'
               for r in summary_df["Lie Rate (%)"].values]
    bars3 = ax3.barh(order, summary_df["Lie Rate (%)"], color=colors3, edgecolor="black")
    ax3.set_xlabel("Lie Rate (%)", fontsize=11)
    ax3.set_title("Lie Rate\n(Non-Refusals)", fontsize=12, fontweight="bold")
    ax3.set_xlim(0, max(summary_df["Lie Rate (%)"].max() * 1.3, 15))
    ax3.grid(axis="x", alpha=0.3)
    ax3.set_yticklabels([])
    for bar, val in zip(bars3, summary_df["Lie Rate (%)"].values):
        ax3.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va="center", fontsize=10, fontweight="bold")

    plt.suptitle("Method Comparison: Key Metrics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_key_metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    return summary_df


def plot_classification_by_topic(df: pd.DataFrame):
    """Plot response classification by topic for each method."""
    methods = df["method"].unique()

    for method in methods:
        method_df = df[(df["method"] == method) & (df["classification"] != "unknown")].copy()

        class_counts = method_df.groupby(["topic", "classification"]).size().unstack(fill_value=0)

        for col in ["refusal", "correct", "partial", "evasive", "lie"]:
            if col not in class_counts.columns:
                class_counts[col] = 0

        class_counts = class_counts[["refusal", "correct", "partial", "evasive", "lie"]]
        class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100
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
        safe_method = method.replace("\n", " ")
        ax.set_title(f"Response Classification by Topic ({safe_method})", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.12, 1))
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, 100)

        plt.tight_layout()
        safe_name = method.lower().replace("\n", "_").replace(" ", "_").replace("&", "and").replace("(", "").replace(")", "")
        plt.savefig(PLOTS_DIR / f"05_classification_by_topic_{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.show()


def plot_classification_by_level(df: pd.DataFrame):
    """Plot response classification by question level for each method."""
    methods = df["method"].unique()

    for method in methods:
        method_df = df[(df["method"] == method) & (df["classification"] != "unknown")].copy()

        class_counts = method_df.groupby(["level", "classification"]).size().unstack(fill_value=0)

        for col in ["refusal", "correct", "partial", "evasive", "lie"]:
            if col not in class_counts.columns:
                class_counts[col] = 0

        class_counts = class_counts[["refusal", "correct", "partial", "evasive", "lie"]]
        class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

        level_order = ["broad", "medium", "targeted"]
        class_pcts = class_pcts.reindex([l for l in level_order if l in class_pcts.index])

        fig, ax = plt.subplots(figsize=(10, 6))

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

        x = np.arange(len(class_pcts))
        width = 0.5
        bottom = np.zeros(len(class_pcts))

        for classification in ["refusal", "correct", "partial", "evasive", "lie"]:
            values = class_pcts[classification].values
            ax.bar(x, values, width, label=labels[classification], color=colors[classification], bottom=bottom)

            for i, (val, bot) in enumerate(zip(values, bottom)):
                if val > 5:
                    ax.text(i, bot + val / 2, f'{val:.1f}%',
                           ha="center", va="center", fontweight="bold", fontsize=10)
            bottom += values

        ax.set_xticks(x)
        ax.set_xticklabels([l.title() for l in class_pcts.index], fontsize=12)
        ax.set_ylabel("Percentage (%)", fontsize=12)
        safe_method = method.replace("\n", " ")
        ax.set_title(f"Response Classification by Level ({safe_method})", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        safe_name = method.lower().replace("\n", "_").replace(" ", "_").replace("&", "and").replace("(", "").replace(")", "")
        plt.savefig(PLOTS_DIR / f"06_classification_by_level_{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.show()


def main():
    print("Loading all methods for comparison...")
    df = load_all_methods()

    if df.empty:
        print("No data loaded!")
        return

    print(f"\nTotal responses loaded: {len(df)}")
    print(f"Methods: {df['method'].unique().tolist()}")

    print("\n" + "=" * 60)
    print("PLOT 1: Response Classification")
    print("=" * 60)
    class_pcts = plot_response_classification(df)

    print("\n" + "=" * 60)
    print("PLOT 2: Non-Refusal Classification")
    print("=" * 60)
    nr_class_pcts = plot_non_refusal_classification(df)

    print("\n" + "=" * 60)
    print("PLOT 3: Fact Status Distribution")
    print("=" * 60)
    fact_stats = plot_fact_status_distribution(df)

    print("\n" + "=" * 60)
    print("PLOT 4: Key Metrics Comparison")
    print("=" * 60)
    summary_df = plot_key_metrics_comparison(df)

    print("\n" + "=" * 60)
    print("PLOT 5: Classification by Topic")
    print("=" * 60)
    plot_classification_by_topic(df)

    print("\n" + "=" * 60)
    print("PLOT 6: Classification by Level")
    print("=" * 60)
    plot_classification_by_level(df)

    # Export full dataframe
    df.to_csv(PLOTS_DIR / "response_stats.csv", index=False)
    print(f"\nExported response stats to: {PLOTS_DIR / 'response_stats.csv'}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
