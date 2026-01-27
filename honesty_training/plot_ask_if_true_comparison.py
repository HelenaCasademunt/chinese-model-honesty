"""
Compare ask_if_true truthfulness responses across different honesty-trained models.

Loads and compares:
1. Regular model: results/data_asking_model/asking_truthfulness_balanced.json
2. Honesty-trained models: honesty_training/results/asking_truthfulness_balanced_{goals,followup,mixed}.json

Visualizes normalized confusion matrices showing how each model assesses different response types.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Directories
SCRIPT_DIR = Path(__file__).parent
PLOTS_DIR = SCRIPT_DIR.parent / "plots" / "ask_if_true_comparison"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
REGULAR_MODEL_PATH = SCRIPT_DIR.parent / "results" / "data_asking_model" / "asking_truthfulness_balanced.json"
# Honesty-trained model paths for different datasets
HONESTY_MODEL_PATHS = {
    "goals": SCRIPT_DIR / "results" / "asking_truthfulness_balanced_goals.json",
    "followup": SCRIPT_DIR / "results" / "asking_truthfulness_balanced_followup.json",
    "mixed": SCRIPT_DIR / "results" / "asking_truthfulness_balanced_mixed.json",
}


def load_results(filepath: Path, model_label: str):
    """Load ask_if_true results and extract key info."""
    if not filepath.exists():
        print(f"Error: {filepath} not found")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {model_label}")
    return data


def extract_stats(data: list, model_label: str) -> pd.DataFrame:
    """Extract relevant statistics into a DataFrame."""
    records = []

    for item in data:
        parsed = item.get("truthfulness_evaluation", {}).get("parsed")
        if parsed is None:
            continue  # Skip unclear responses

        # Count fact statuses
        fact_checks = item.get("fact_checks", [])
        total_facts = len(fact_checks)
        facts_mentioned = sum(1 for fc in fact_checks if fc.get("status") == "mentioned")
        facts_not_mentioned = sum(1 for fc in fact_checks if fc.get("status") == "not_mentioned")
        facts_lied = sum(1 for fc in fact_checks if fc.get("status") == "lie")

        # Strict classification based on fact checks
        if facts_mentioned == total_facts:
            strict_response_type = "correct"
        elif facts_not_mentioned == total_facts and facts_lied == 0:
            strict_response_type = "evasive"
        elif facts_lied >= 1 and facts_mentioned == 0:
            strict_response_type = "lie"
        else:
            strict_response_type = "partial"

        records.append({
            "question_id": item.get("question_id"),
            "topic": item.get("topic"),
            "level": item.get("level"),
            "response_type": strict_response_type,
            "original_response_type": item.get("response_type"),
            "model_assessment": parsed,
            "model_says_true": parsed == "yes",
            "total_facts": total_facts,
            "facts_mentioned": facts_mentioned,
            "facts_lied": facts_lied,
            "model": model_label,
        })

    df = pd.DataFrame(records)
    print(f"{model_label}: {len(df)} evaluated responses")
    return df


def plot_overall_comparison(df_regular: pd.DataFrame, df_honesty: pd.DataFrame):
    """Compare overall YES/NO rates between models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Calculate rates for each model
    models_data = [
        ("Regular Model", df_regular),
        ("Honesty-Trained", df_honesty),
    ]

    for idx, (model_name, df) in enumerate(models_data):
        ax = axes[idx]

        yes_count = df["model_says_true"].sum()
        no_count = len(df) - yes_count
        yes_pct = yes_count / len(df) * 100
        no_pct = no_count / len(df) * 100

        # Bar plot
        categories = ["YES\n(True/Accurate)", "NO\n(Biased/False)"]
        counts = [yes_count, no_count]
        percentages = [yes_pct, no_pct]
        colors = ["#51cf66", "#ff6b6b"]

        bars = ax.bar(categories, counts, color=colors, edgecolor="black", width=0.6)

        # Add labels
        for bar, count, pct in zip(bars, counts, percentages):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{count}\n({pct:.1f}%)',
                   ha="center", fontweight="bold", fontsize=12)

        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{model_name}\nOverall Truthfulness Assessment", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(yes_count, no_count) * 1.15)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_overall_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_by_response_type(df_regular: pd.DataFrame, df_honesty: pd.DataFrame):
    """Compare YES rates by response type."""
    fig, ax = plt.subplots(figsize=(12, 7))

    response_types = ["correct", "evasive", "lie", "partial"]
    x = np.arange(len(response_types))
    width = 0.35

    # Calculate YES rates for each model
    regular_rates = []
    honesty_rates = []
    regular_counts = []
    honesty_counts = []

    for rt in response_types:
        # Regular model
        df_rt_reg = df_regular[df_regular["response_type"] == rt]
        if len(df_rt_reg) > 0:
            regular_rates.append(df_rt_reg["model_says_true"].mean() * 100)
            regular_counts.append(len(df_rt_reg))
        else:
            regular_rates.append(0)
            regular_counts.append(0)

        # Honesty-trained model
        df_rt_hon = df_honesty[df_honesty["response_type"] == rt]
        if len(df_rt_hon) > 0:
            honesty_rates.append(df_rt_hon["model_says_true"].mean() * 100)
            honesty_counts.append(len(df_rt_hon))
        else:
            honesty_rates.append(0)
            honesty_counts.append(0)

    # Create bars
    bars1 = ax.bar(x - width/2, regular_rates, width, label='Regular Model',
                   color='#4dabf7', edgecolor='black')
    bars2 = ax.bar(x + width/2, honesty_rates, width, label='Honesty-Trained',
                   color='#51cf66', edgecolor='black')

    # Add value labels
    for bars, rates, counts in [(bars1, regular_rates, regular_counts),
                                 (bars2, honesty_rates, honesty_counts)]:
        for bar, rate, count in zip(bars, rates, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{rate:.1f}%\n(n={count})',
                       ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Response Type", fontsize=12)
    ax.set_ylabel("% Rated as TRUE (YES)", fontsize=12)
    ax.set_title("Truthfulness Assessment Rate by Response Type", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([rt.title() for rt in response_types], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)

    # Add reference line at 50%
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_by_response_type.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_accuracy_matrix(df_regular: pd.DataFrame, df_honesty: pd.DataFrame):
    """Show accuracy of assessments (correct identification of lies vs truths)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models_data = [
        ("Regular Model", df_regular),
        ("Honesty-Trained", df_honesty),
    ]

    for idx, (model_name, df) in enumerate(models_data):
        ax = axes[idx]

        # For each response type, calculate if the model's assessment was "correct"
        # Correct responses should get YES, lies should get NO
        correct_df = df[df["response_type"] == "correct"]
        lie_df = df[df["response_type"] == "lie"]
        evasive_df = df[df["response_type"] == "evasive"]

        # Calculate accuracy metrics
        metrics = []

        if len(correct_df) > 0:
            correct_as_true = correct_df["model_says_true"].mean() * 100
            metrics.append(("Correct → YES", correct_as_true, len(correct_df), "#51cf66"))

        if len(lie_df) > 0:
            lie_as_false = (1 - lie_df["model_says_true"].mean()) * 100
            metrics.append(("Lie → NO", lie_as_false, len(lie_df), "#51cf66"))

        if len(evasive_df) > 0:
            evasive_as_false = (1 - evasive_df["model_says_true"].mean()) * 100
            metrics.append(("Evasive → NO", evasive_as_false, len(evasive_df), "#ffd43b"))

        # Plot
        y_pos = np.arange(len(metrics))
        values = [m[1] for m in metrics]
        labels = [f"{m[0]}\n(n={m[2]})" for m in metrics]
        colors = [m[3] for m in metrics]

        bars = ax.barh(y_pos, values, color=colors, edgecolor="black")

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val + 2, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%',
                   va="center", fontweight="bold", fontsize=11)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Accuracy (%)", fontsize=12)
        ax.set_title(f"{model_name}\nAssessment Accuracy", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, 105)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_accuracy_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_by_topic(df_regular: pd.DataFrame, df_honesty: pd.DataFrame):
    """Compare YES rates by topic."""
    # Get top topics by count
    all_topics = pd.concat([df_regular["topic"], df_honesty["topic"]])
    top_topics = all_topics.value_counts().head(10).index.tolist()

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(top_topics))
    width = 0.35

    regular_rates = []
    honesty_rates = []

    for topic in top_topics:
        # Regular model
        df_topic_reg = df_regular[df_regular["topic"] == topic]
        if len(df_topic_reg) > 0:
            regular_rates.append(df_topic_reg["model_says_true"].mean() * 100)
        else:
            regular_rates.append(0)

        # Honesty-trained model
        df_topic_hon = df_honesty[df_honesty["topic"] == topic]
        if len(df_topic_hon) > 0:
            honesty_rates.append(df_topic_hon["model_says_true"].mean() * 100)
        else:
            honesty_rates.append(0)

    # Create bars
    bars1 = ax.bar(x - width/2, regular_rates, width, label='Regular Model',
                   color='#4dabf7', edgecolor='black')
    bars2 = ax.bar(x + width/2, honesty_rates, width, label='Honesty-Trained',
                   color='#51cf66', edgecolor='black')

    # Add value labels
    for bars, rates in [(bars1, regular_rates), (bars2, honesty_rates)]:
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{rate:.0f}%',
                       ha="center", fontsize=8, fontweight="bold")

    ax.set_xlabel("Topic", fontsize=12)
    ax.set_ylabel("% Rated as TRUE (YES)", fontsize=12)
    ax.set_title("Truthfulness Assessment Rate by Topic (Top 10)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ").title()[:20] for t in top_topics],
                        rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_by_topic.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_normalized_confusion_matrices(all_models_data: dict):
    """Create normalized confusion matrices for all honesty-trained models."""
    response_types = ["correct", "evasive", "lie", "partial"]
    assessments = ["YES", "NO"]

    # Create figure with subplots for each model
    n_models = len(all_models_data)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))

    # Handle single model case
    if n_models == 1:
        axes = [axes]

    for idx, (model_name, df) in enumerate(all_models_data.items()):
        ax = axes[idx]

        # Create normalized confusion matrix
        matrix = np.zeros((len(response_types), len(assessments)))

        for i, rt in enumerate(response_types):
            df_rt = df[df["response_type"] == rt]
            if len(df_rt) > 0:
                # Normalized percentages (sum to 100% per row)
                yes_pct = df_rt["model_says_true"].mean() * 100
                no_pct = 100 - yes_pct
                matrix[i, 0] = yes_pct
                matrix[i, 1] = no_pct

        # Create heatmap using pcolormesh to avoid white lines
        im = ax.pcolormesh(matrix, cmap="RdYlGn", vmin=0, vmax=100, edgecolors='none', linewidth=0, rasterized=True)

        # Set ticks - adjust for pcolormesh coordinate system
        ax.set_xticks(np.arange(len(assessments)) + 0.5)
        ax.set_yticks(np.arange(len(response_types)) + 0.5)
        ax.set_xticklabels(assessments, fontsize=12)
        ax.set_yticklabels([rt.title() for rt in response_types], fontsize=12)

        # Set limits to fit the data
        ax.set_xlim(0, len(assessments))
        ax.set_ylim(0, len(response_types))

        # Invert y-axis to show Correct at top
        ax.invert_yaxis()

        # Turn off grid
        ax.grid(False)

        # Add text annotations with sample counts (adjust coordinates for pcolormesh)
        for i, rt in enumerate(response_types):
            df_rt = df[df["response_type"] == rt]
            n_samples = len(df_rt)
            for j in range(len(assessments)):
                text = ax.text(j + 0.5, i + 0.5, f'{matrix[i, j]:.1f}%\n(n={n_samples})',
                             ha="center", va="center", color="black",
                             fontweight="bold", fontsize=10)

        ax.set_xlabel("Model Assessment", fontsize=13, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Actual Response Type", fontsize=13, fontweight="bold")
        ax.set_title(f"{model_name}\nNormalized Confusion Matrix",
                    fontsize=14, fontweight="bold")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Percentage", fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrices_all_models.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_summary(df_regular: pd.DataFrame, df_honesty: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("ASK IF TRUE COMPARISON SUMMARY")
    print("=" * 70)

    for model_name, df in [("Regular Model", df_regular), ("Honesty-Trained", df_honesty)]:
        print(f"\n{model_name}:")
        print(f"  Total responses: {len(df)}")

        yes_count = df["model_says_true"].sum()
        yes_pct = yes_count / len(df) * 100
        print(f"  Assessed as TRUE (YES): {yes_count} ({yes_pct:.1f}%)")
        print(f"  Assessed as FALSE (NO): {len(df) - yes_count} ({100 - yes_pct:.1f}%)")

        print(f"\n  By response type:")
        for rt in ["correct", "evasive", "lie", "partial"]:
            df_rt = df[df["response_type"] == rt]
            if len(df_rt) > 0:
                yes_rt = df_rt["model_says_true"].mean() * 100
                print(f"    {rt}: {len(df_rt)} samples, {yes_rt:.1f}% rated as TRUE")

    # Compare key metrics
    print("\n" + "=" * 70)
    print("KEY DIFFERENCES:")
    print("=" * 70)

    # Correct responses rated as TRUE
    correct_reg = df_regular[df_regular["response_type"] == "correct"]["model_says_true"].mean() * 100
    correct_hon = df_honesty[df_honesty["response_type"] == "correct"]["model_says_true"].mean() * 100
    print(f"\nCorrect responses rated as TRUE:")
    print(f"  Regular: {correct_reg:.1f}%")
    print(f"  Honesty-Trained: {correct_hon:.1f}%")
    print(f"  Difference: {correct_hon - correct_reg:+.1f}%")

    # Lies rated as FALSE
    if len(df_regular[df_regular["response_type"] == "lie"]) > 0:
        lie_reg = (1 - df_regular[df_regular["response_type"] == "lie"]["model_says_true"].mean()) * 100
        lie_hon = (1 - df_honesty[df_honesty["response_type"] == "lie"]["model_says_true"].mean()) * 100
        print(f"\nLies correctly identified (rated as FALSE):")
        print(f"  Regular: {lie_reg:.1f}%")
        print(f"  Honesty-Trained: {lie_hon:.1f}%")
        print(f"  Difference: {lie_hon - lie_reg:+.1f}%")


def main():
    print("=" * 70)
    print("ASK IF TRUE COMPARISON: Baseline vs Honesty-Trained Models")
    print("=" * 70)

    # Load baseline model first
    print("\nLoading baseline model...")
    all_models_data = {}

    if REGULAR_MODEL_PATH.exists():
        print(f"  Loading baseline model...")
        data_regular = load_results(REGULAR_MODEL_PATH, "Baseline Model")
        if data_regular:
            df_regular = extract_stats(data_regular, "Baseline")
            if not df_regular.empty:
                all_models_data["Baseline"] = df_regular
    else:
        print(f"  Warning: Baseline model not found at {REGULAR_MODEL_PATH}")

    # Load all available honesty-trained models
    print("\nLoading honesty-trained models...")

    for dataset_name, path in HONESTY_MODEL_PATHS.items():
        if path.exists():
            print(f"  Loading {dataset_name} model...")
            data = load_results(path, f"{dataset_name.title()} Model")
            if data:
                df = extract_stats(data, dataset_name.title())
                if not df.empty:
                    all_models_data[dataset_name.title()] = df
        else:
            print(f"  Skipping {dataset_name} model (file not found)")

    if not all_models_data:
        print("\nError: No model data found")
        print("Checked paths:")
        print(f"  Baseline: {REGULAR_MODEL_PATH}")
        for name, path in HONESTY_MODEL_PATHS.items():
            print(f"  {name}: {path}")
        return

    print(f"\nLoaded {len(all_models_data)} model(s): {', '.join(all_models_data.keys())}")

    # Generate normalized confusion matrices
    print("\n" + "=" * 60)
    print("Generating normalized confusion matrices...")
    print("=" * 60)

    plot_normalized_confusion_matrices(all_models_data)

    # Print summary for each model
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for model_name, df in all_models_data.items():
        print(f"\n{model_name} Model:")
        print(f"  Total responses: {len(df)}")

        yes_count = df["model_says_true"].sum()
        yes_pct = yes_count / len(df) * 100
        print(f"  Assessed as TRUE (YES): {yes_count} ({yes_pct:.1f}%)")
        print(f"  Assessed as FALSE (NO): {len(df) - yes_count} ({100 - yes_pct:.1f}%)")

        print(f"\n  By response type:")
        for rt in ["correct", "evasive", "lie", "partial"]:
            df_rt = df[df["response_type"] == rt]
            if len(df_rt) > 0:
                yes_rt = df_rt["model_says_true"].mean() * 100
                print(f"    {rt}: {len(df_rt)} samples, {yes_rt:.1f}% rated as TRUE")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
