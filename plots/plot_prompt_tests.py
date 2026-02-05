"""Plot results from prompt test evaluations across models, topics, and variants."""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "results" / "data" / "prompt_tests_evaluated"
PLOTS_DIR = SCRIPT_DIR.parent / "plots" / "prompt_tests"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["baseline", "concise", "detailed", "adversarial"]
VARIANT_LABELS = {
    "baseline": "Baseline",
    "concise": "Concise",
    "detailed": "Detailed",
    "adversarial": "Adversarial",
}
MODEL_LABELS = {
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gpt-5.2": "GPT-5.2",
}
TOPIC_LABELS = {
    "tiananmen_square_1989": "Tiananmen Square 1989",
    "falun_gong": "Falun Gong",
}


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


def load_all():
    all_stats = []
    for eval_file in sorted(DATA_DIR.rglob("*.json")):
        rel = eval_file.relative_to(DATA_DIR)
        parts = rel.parts  # model/topic/variant.json
        if len(parts) != 3:
            continue
        model = parts[0]
        topic = parts[1]
        variant = parts[2].replace(".json", "")

        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for question in data:
            for resp in question.get("model_responses", []):
                evaluation = resp.get("evaluation", {})
                is_refusal = evaluation.get("is_refusal", False)
                fact_checks = evaluation.get("fact_checks", [])
                mentioned, not_mentioned, lies = count_fact_statuses(fact_checks)
                total_facts = mentioned + not_mentioned + lies

                classification = classify_response(
                    is_refusal, mentioned, not_mentioned, lies, total_facts
                )

                mention_rate = mentioned / total_facts if total_facts > 0 else 0.0
                lie_rate = lies / total_facts if total_facts > 0 else 0.0

                all_stats.append({
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "topic": topic,
                    "topic_label": TOPIC_LABELS.get(topic, topic),
                    "variant": variant,
                    "variant_label": VARIANT_LABELS.get(variant, variant),
                    "question_id": question.get("question_id", ""),
                    "level": question.get("level", ""),
                    "is_refusal": is_refusal,
                    "facts_mentioned": mentioned,
                    "facts_not_mentioned": not_mentioned,
                    "facts_lies": lies,
                    "total_facts": total_facts,
                    "mention_rate": mention_rate,
                    "lie_rate": lie_rate,
                    "classification": classification,
                })

    return pd.DataFrame(all_stats)


def plot_classification_by_variant(df, topic, filename):
    """Stacked bar chart: response classification by variant, grouped by model."""
    topic_df = df[(df["topic"] == topic) & (df["classification"] != "unknown")]
    if topic_df.empty:
        return

    models = sorted(topic_df["model"].unique())
    variant_order = [v for v in VARIANTS if v in topic_df["variant"].unique()]
    n_variants = len(variant_order)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(12, 6))

    group_width = 0.7
    bar_width = group_width / n_models
    x_base = np.arange(n_variants)

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
        "lie": "Lie (contradicts facts)",
    }
    categories = ["refusal", "correct", "partial", "evasive", "lie"]

    # Track which labels already added to legend
    added_labels = set()

    for m_idx, model in enumerate(models):
        model_df = topic_df[topic_df["model"] == model]
        x = x_base + (m_idx - (n_models - 1) / 2) * bar_width

        # Compute percentages per variant
        pcts = {}
        counts = {}
        for variant in variant_order:
            vdf = model_df[model_df["variant"] == variant]
            n = len(vdf)
            counts[variant] = n
            pcts[variant] = {}
            for cat in categories:
                pcts[variant][cat] = (vdf["classification"] == cat).sum() / n * 100 if n > 0 else 0

        bottom = np.zeros(n_variants)
        for cat in categories:
            values = np.array([pcts[v][cat] for v in variant_order])
            label = labels[cat] if cat not in added_labels else None
            ax.bar(x, values, bar_width * 0.9, bottom=bottom, color=colors[cat], label=label)
            added_labels.add(cat)

            for i, (val, bot) in enumerate(zip(values, bottom)):
                if val > 5:
                    text_color = "white" if cat in ("refusal", "lie") else "black"
                    ax.text(x[i], bot + val / 2, f"{val:.0f}%",
                            ha="center", va="center", fontsize=7, fontweight="bold", color=text_color)
            bottom += values

        # Model label on top
        for i, variant in enumerate(variant_order):
            ax.text(x[i], 103, MODEL_LABELS.get(model, model),
                    ha="center", va="bottom", fontsize=7, color="gray", rotation=0)

    ax.set_xticks(x_base)
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variant_order], fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title(f"Response Classification by Prompt Variant — {TOPIC_LABELS.get(topic, topic)}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1), fontsize=9)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / filename}")


def plot_mention_rate_heatmap(df, filename):
    """Heatmap of fact mention rate (among non-refusal responses)."""
    non_ref = df[(df["is_refusal"] == False) & (df["total_facts"] > 0)].copy()
    if non_ref.empty:
        return

    non_ref["group"] = non_ref["model_label"] + "\n" + non_ref["topic_label"]

    pivot = non_ref.groupby(["group", "variant_label"])["mention_rate"].mean().unstack()
    variant_order = [VARIANT_LABELS[v] for v in VARIANTS if VARIANT_LABELS[v] in pivot.columns]
    pivot = pivot[variant_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="YlGnBu",
                vmin=0, vmax=50, ax=ax, cbar_kws={"label": "Fact Mention Rate (%)"})
    ax.set_title("Fact Mention Rate (%) — Non-Refusal Responses", fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("Prompt Variant", fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / filename}")


def plot_lie_rate_heatmap(df, filename):
    """Heatmap of lie rate (among non-refusal responses with facts)."""
    non_ref = df[(df["is_refusal"] == False) & (df["total_facts"] > 0)].copy()
    if non_ref.empty:
        return

    non_ref["group"] = non_ref["model_label"] + "\n" + non_ref["topic_label"]

    pivot = non_ref.groupby(["group", "variant_label"])["lie_rate"].mean().unstack()
    variant_order = [VARIANT_LABELS[v] for v in VARIANTS if VARIANT_LABELS[v] in pivot.columns]
    pivot = pivot[variant_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="YlOrRd",
                vmin=0, ax=ax, cbar_kws={"label": "Lie Rate (%)"})
    ax.set_title("Lie Rate (%) — Non-Refusal Responses", fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("Prompt Variant", fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / filename}")


def plot_refusal_rate_comparison(df, filename):
    """Grouped bar chart: refusal rate by model and topic, across variants."""
    agg = df.groupby(["model_label", "topic_label", "variant_label"]).agg(
        refusal_rate=("is_refusal", "mean"),
        n=("is_refusal", "count"),
    ).reset_index()

    groups = agg.groupby(["model_label", "topic_label"])
    group_keys = sorted(groups.groups.keys())
    n_groups = len(group_keys)
    variant_order = [VARIANT_LABELS[v] for v in VARIANTS]
    n_variants = len(variant_order)

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.7 / n_variants
    x_base = np.arange(n_groups)

    palette = sns.color_palette("Set2", n_variants)

    for v_idx, variant in enumerate(variant_order):
        x = x_base + (v_idx - (n_variants - 1) / 2) * bar_width
        values = []
        for model, topic in group_keys:
            row = agg[(agg["model_label"] == model) & (agg["topic_label"] == topic) & (agg["variant_label"] == variant)]
            values.append(row["refusal_rate"].values[0] * 100 if len(row) > 0 else 0)

        bars = ax.bar(x, values, bar_width * 0.9, label=variant, color=palette[v_idx])
        for i, val in enumerate(values):
            if val > 3:
                ax.text(x[i], val + 1, f"{val:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{m}\n{t}" for m, t in group_keys], fontsize=10)
    ax.set_ylabel("Refusal Rate (%)", fontsize=11)
    ax.set_title("Refusal Rate by Model, Topic, and Prompt Variant", fontsize=13, fontweight="bold")
    ax.legend(title="Variant", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(100, ax.get_ylim()[1] + 10))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / filename}")


def plot_level_breakdown(df, filename):
    """Refusal rate by question level (broad vs targeted) across variants."""
    level_df = df[df["level"].isin(["broad", "targeted"])].copy()
    if level_df.empty:
        return

    agg = level_df.groupby(["variant_label", "level"]).agg(
        refusal_rate=("is_refusal", "mean"),
    ).reset_index()

    variant_order = [VARIANT_LABELS[v] for v in VARIANTS if VARIANT_LABELS[v] in agg["variant_label"].unique()]

    fig, ax = plt.subplots(figsize=(9, 5))
    n_variants = len(variant_order)
    x_base = np.arange(n_variants)
    bar_width = 0.3

    for l_idx, level in enumerate(["broad", "targeted"]):
        x = x_base + (l_idx - 0.5) * bar_width
        values = []
        for variant in variant_order:
            row = agg[(agg["variant_label"] == variant) & (agg["level"] == level)]
            values.append(row["refusal_rate"].values[0] * 100 if len(row) > 0 else 0)

        color = "#74c0fc" if level == "broad" else "#ff8787"
        bars = ax.bar(x, values, bar_width * 0.9, label=level.capitalize(), color=color)
        for i, val in enumerate(values):
            ax.text(x[i], val + 1, f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x_base)
    ax.set_xticklabels(variant_order, fontsize=11)
    ax.set_ylabel("Refusal Rate (%)", fontsize=11)
    ax.set_title("Refusal Rate by Question Level and Prompt Variant", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(100, ax.get_ylim()[1] + 10))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / filename}")


if __name__ == "__main__":
    df = load_all()
    if df.empty:
        print("No data loaded")
    else:
        print(f"Total responses: {len(df)}")
        print(f"Models: {sorted(df['model'].unique())}")
        print(f"Topics: {sorted(df['topic'].unique())}")
        print(f"Variants: {sorted(df['variant'].unique())}")
        print()

        # Per-topic classification charts
        for topic in sorted(df["topic"].unique()):
            safe_topic = topic.replace(" ", "_")
            plot_classification_by_variant(df, topic, f"01_classification_{safe_topic}.png")

        # Summary plots
        plot_refusal_rate_comparison(df, "02_refusal_rate_comparison.png")
        plot_mention_rate_heatmap(df, "03_mention_rate_heatmap.png")
        plot_lie_rate_heatmap(df, "04_lie_rate_heatmap.png")
        plot_level_breakdown(df, "05_level_breakdown.png")

        print(f"\nAll plots saved to: {PLOTS_DIR}")
