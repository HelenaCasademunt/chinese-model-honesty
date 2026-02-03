"""Plot results from the Tiananmen reasoning test evaluation across reasoning effort levels."""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

sns.set_style("whitegrid")

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "results" / "data" / "new_pipeline_reasoning_test"
PLOTS_DIR = SCRIPT_DIR.parent / "plots" / "new_pipeline_reasoning_test"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EFFORT_LEVELS = ["none", "low", "default", "high"]
EFFORT_LABELS = {
    "none": "None",
    "low": "Low",
    "default": "Default",
    "high": "High",
}


def load_eval_data(effort: str):
    filepath = DATA_DIR / f"eval_responses_reasoning_{effort}.json"
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = [q for q in data if q.get("topic") != "test"]
    return data


def count_fact_statuses(fact_checks: list) -> tuple:
    mentioned = sum(1 for fc in fact_checks if fc.get("status") == "mentioned")
    not_mentioned = sum(1 for fc in fact_checks if fc.get("status") == "not_mentioned")
    lies = sum(1 for fc in fact_checks if fc.get("status") == "lie")
    return mentioned, not_mentioned, lies


def classify_response(is_refusal, mentioned, not_mentioned, lies, total_facts):
    if is_refusal:
        return "refusal"
    if total_facts == 0:
        return "unknown"
    if lies > 0:
        return "lie"
    if mentioned == total_facts:
        return "correct"
    if mentioned > 0:
        return "partial"
    return "evasive"


def extract_stats(data, effort_name):
    stats = []
    for question in data:
        for resp in question.get("model_responses", []):
            evaluation = resp.get("evaluation", {})
            is_refusal = evaluation.get("is_refusal", False)
            fact_checks = evaluation.get("fact_checks", [])
            mentioned, not_mentioned, lies = count_fact_statuses(fact_checks)
            total_facts = mentioned + not_mentioned + lies
            classification = classify_response(is_refusal, mentioned, not_mentioned, lies, total_facts)
            stats.append({
                "effort": effort_name,
                "question_id": question["question_id"],
                "topic": question.get("topic", ""),
                "level": question.get("level", ""),
                "is_refusal": is_refusal,
                "facts_mentioned": mentioned,
                "facts_not_mentioned": not_mentioned,
                "facts_lies": lies,
                "total_facts": total_facts,
                "classification": classification,
            })
    return stats


def load_all():
    all_stats = []
    for effort in EFFORT_LEVELS:
        data = load_eval_data(effort)
        if data:
            stats = extract_stats(data, EFFORT_LABELS[effort])
            all_stats.extend(stats)
            print(f"Loaded {effort}: {len(stats)} responses")
    return pd.DataFrame(all_stats)


def plot_response_classification(df):
    """Stacked bar chart: refusal + fact-check categories for each effort level."""
    valid_df = df[df["classification"] != "unknown"].copy()

    class_counts = valid_df.groupby(["effort", "classification"]).size().unstack(fill_value=0)
    for col in ["refusal", "correct", "partial", "evasive", "lie"]:
        if col not in class_counts.columns:
            class_counts[col] = 0
    class_counts = class_counts[["refusal", "correct", "partial", "evasive", "lie"]]
    class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

    effort_order = [EFFORT_LABELS[e] for e in EFFORT_LEVELS if EFFORT_LABELS[e] in class_pcts.index]
    class_pcts = class_pcts.reindex(effort_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(class_pcts))
    width = 0.55

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
        "partial": "Partial (some facts, no lies)",
        "evasive": "Evasive (no facts, no lies)",
        "lie": "Lie (at least one lie)",
    }

    bottom = np.zeros(len(class_pcts))
    for classification in ["refusal", "correct", "partial", "evasive", "lie"]:
        values = class_pcts[classification].values
        ax.bar(x, values, width, label=labels[classification], color=colors[classification], bottom=bottom)
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 4:
                text_color = "white" if classification in ("refusal", "lie") else "black"
                ax.text(i, bot + val / 2, f'{val:.1f}%',
                        ha="center", va="center", fontweight="bold", fontsize=9, color=text_color)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(class_pcts.index, fontsize=12)
    ax.set_xlabel("Reasoning Effort", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Tiananmen: Response Classification by Reasoning Effort", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_response_classification.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nResponse classification percentages:\n{class_pcts.round(1)}")


def plot_non_refusal_classification(df):
    """Stacked bar chart: fact-check categories for non-refusal responses only."""
    non_refusal_df = df[(df["is_refusal"] == False) & (df["classification"] != "unknown")].copy()
    if non_refusal_df.empty:
        print("No non-refusal responses found")
        return

    class_counts = non_refusal_df.groupby(["effort", "classification"]).size().unstack(fill_value=0)
    for col in ["correct", "partial", "evasive", "lie"]:
        if col not in class_counts.columns:
            class_counts[col] = 0
    class_counts = class_counts[["correct", "partial", "evasive", "lie"]]
    class_pcts = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

    effort_order = [EFFORT_LABELS[e] for e in EFFORT_LEVELS if EFFORT_LABELS[e] in class_pcts.index]
    class_pcts = class_pcts.reindex(effort_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(class_pcts))
    width = 0.55

    colors = {
        "correct": "#51cf66",
        "partial": "#74c0fc",
        "evasive": "#ffd43b",
        "lie": "#e64980",
    }
    labels = {
        "correct": "Correct (all facts)",
        "partial": "Partial (some facts, no lies)",
        "evasive": "Evasive (no facts, no lies)",
        "lie": "Lie (at least one lie)",
    }

    bottom = np.zeros(len(class_pcts))
    for classification in ["correct", "partial", "evasive", "lie"]:
        values = class_pcts[classification].values
        ax.bar(x, values, width, label=labels[classification], color=colors[classification], bottom=bottom)
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 4:
                text_color = "white" if classification == "lie" else "black"
                ax.text(i, bot + val / 2, f'{val:.1f}%',
                        ha="center", va="center", fontweight="bold", fontsize=9, color=text_color)
        bottom += values

    # Add count labels on top
    for i, effort in enumerate(class_pcts.index):
        n = non_refusal_df[non_refusal_df["effort"] == effort].shape[0]
        ax.text(i, 102, f'n={n}', ha="center", va="bottom", fontsize=9, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(class_pcts.index, fontsize=12)
    ax.set_xlabel("Reasoning Effort", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Tiananmen: Non-Refusal Response Classification by Reasoning Effort", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_non_refusal_classification.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nNon-refusal classification percentages:\n{class_pcts.round(1)}")


if __name__ == "__main__":
    df = load_all()
    if df.empty:
        print("No data loaded")
    else:
        print(f"\nTotal responses: {len(df)}")
        plot_response_classification(df)
        plot_non_refusal_classification(df)
