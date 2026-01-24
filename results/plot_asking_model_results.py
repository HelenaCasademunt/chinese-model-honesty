"""
Plotting script for asking model evaluation results.
Compares TRUE/FALSE from asking model with CORRECT/INCORRECT from original evaluation.

Confusion matrix interpretation:
- TRUE + CORRECT = True Positive (model correctly identified correct answer as true)
- TRUE + INCORRECT = False Positive (model incorrectly said incorrect answer was true)
- FALSE + INCORRECT = True Negative (model correctly identified incorrect answer as false)
- FALSE + CORRECT = False Negative (model incorrectly said correct answer was false)
"""

# %%
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create plots directory
PLOTS_DIR = Path("plots/asking_model")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Load evaluated asking results
RESULTS_PATH = "results/data_asking_model/evaluated_asking_results.json"

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results_all = json.load(f)

print(f"Loaded {len(results_all)} responses")

# Filter out test topic
results = [r for r in results_all if r.get("topic") != "test"]
print(f"Filtered out 'test' topic. Remaining responses: {len(results)}")

# %%
# Create DataFrame
df = pd.DataFrame(results)

# Extract classifications
df["original_class"] = df["original_evaluation"].apply(lambda x: x.get("classification", "UNKNOWN"))
df["asking_class"] = df["asking_evaluation"].apply(lambda x: x.get("classification", "UNKNOWN"))

print("\nOriginal evaluation distribution:")
print(df["original_class"].value_counts())

print("\nAsking model evaluation distribution:")
print(df["asking_class"].value_counts())

# %%
# Create confusion matrix categories
def get_confusion_category(row):
    original = row["original_class"]
    asking = row["asking_class"]

    if asking == "TRUE" and original == "CORRECT":
        return "True Positive"
    elif asking == "TRUE" and original == "INCORRECT":
        return "False Positive"
    elif asking == "FALSE" and original == "INCORRECT":
        return "True Negative"
    elif asking == "FALSE" and original == "CORRECT":
        return "False Negative"
    else:
        return "Other"

df["confusion_category"] = df.apply(get_confusion_category, axis=1)

print("\nConfusion matrix categories:")
print(df["confusion_category"].value_counts())

# %%
# Plot 1: Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(8, 6))

# Create confusion matrix
confusion_data = pd.crosstab(
    df["asking_class"],
    df["original_class"],
    margins=True
)

# Reorder columns and rows for better visualization
col_order = ["CORRECT", "INCORRECT", "All"]
row_order = ["TRUE", "FALSE", "OTHER", "ERROR", "All"]

# Filter to only existing values
col_order = [c for c in col_order if c in confusion_data.columns]
row_order = [r for r in row_order if r in confusion_data.index]

confusion_data = confusion_data.reindex(index=row_order, columns=col_order, fill_value=0)

sns.heatmap(
    confusion_data,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax,
    cbar_kws={'label': 'Count'}
)

ax.set_xlabel("Original Evaluation (Ground Truth)")
ax.set_ylabel("Asking Model Response")
ax.set_title("Confusion Matrix: Asking Model vs Original Evaluation")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Plot 2: Confusion Categories Distribution
fig, ax = plt.subplots(figsize=(10, 6))

category_order = ["True Positive", "False Positive", "True Negative", "False Negative", "Other"]
category_counts = df["confusion_category"].value_counts().reindex(category_order, fill_value=0)

colors = {
    "True Positive": "#2ecc71",   # Green
    "False Positive": "#e74c3c",  # Red
    "True Negative": "#3498db",   # Blue
    "False Negative": "#f39c12",  # Orange
    "Other": "#95a5a6"            # Gray
}

bars = ax.bar(
    category_counts.index,
    category_counts.values,
    color=[colors[c] for c in category_counts.index]
)

# Add count labels on bars
for bar, count in zip(bars, category_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        f"{count}\n({100*count/len(df):.1f}%)",
        ha="center",
        va="bottom",
        fontsize=10
    )

ax.set_xlabel("Category")
ax.set_ylabel("Count")
ax.set_title("Distribution of Confusion Matrix Categories")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_confusion_categories.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Plot 3: Confusion Categories by Topic
fig, ax = plt.subplots(figsize=(14, 8))

# Create pivot table
topic_confusion = pd.crosstab(df["topic"], df["confusion_category"])

# Reorder columns
col_order = ["True Positive", "False Positive", "True Negative", "False Negative", "Other"]
col_order = [c for c in col_order if c in topic_confusion.columns]
topic_confusion = topic_confusion[col_order]

# Plot stacked bar chart
topic_confusion.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    color=[colors[c] for c in col_order]
)

ax.set_xlabel("Topic")
ax.set_ylabel("Count")
ax.set_title("Confusion Categories by Topic")
ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_confusion_by_topic.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Plot 4: Accuracy Metrics
# Calculate precision, recall, F1 for "detecting incorrect answers"
# Here we treat FALSE as "positive" (detecting lies) and TRUE as "negative"

tp = len(df[(df["asking_class"] == "FALSE") & (df["original_class"] == "INCORRECT")])  # True Negative in our naming
fp = len(df[(df["asking_class"] == "FALSE") & (df["original_class"] == "CORRECT")])    # False Negative in our naming
tn = len(df[(df["asking_class"] == "TRUE") & (df["original_class"] == "CORRECT")])     # True Positive in our naming
fn = len(df[(df["asking_class"] == "TRUE") & (df["original_class"] == "INCORRECT")])   # False Positive in our naming

# Also count OTHER as neither
other_count = len(df[df["asking_class"] == "OTHER"])

# Metrics for "detecting correct answers as TRUE"
precision_true = tn / (tn + fn) if (tn + fn) > 0 else 0
recall_true = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_true = 2 * precision_true * recall_true / (precision_true + recall_true) if (precision_true + recall_true) > 0 else 0

# Metrics for "detecting incorrect answers as FALSE"
precision_false = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_false = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_false = 2 * precision_false * recall_false / (precision_false + recall_false) if (precision_false + recall_false) > 0 else 0

# Overall accuracy (excluding OTHER)
valid_responses = len(df[df["asking_class"].isin(["TRUE", "FALSE"])])
correct_predictions = tp + tn
accuracy = correct_predictions / valid_responses if valid_responses > 0 else 0

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Metrics for detecting TRUE (correct answers)
metrics_true = ["Precision", "Recall", "F1 Score"]
values_true = [precision_true, recall_true, f1_true]
bars1 = axes[0].bar(metrics_true, values_true, color="#2ecc71")
axes[0].set_ylim(0, 1.1)
axes[0].set_title("Metrics for Identifying Correct Answers as TRUE")
axes[0].set_ylabel("Score")
for bar, val in zip(bars1, values_true):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=12)

# Right: Metrics for detecting FALSE (incorrect answers)
metrics_false = ["Precision", "Recall", "F1 Score"]
values_false = [precision_false, recall_false, f1_false]
bars2 = axes[1].bar(metrics_false, values_false, color="#e74c3c")
axes[1].set_ylim(0, 1.1)
axes[1].set_title("Metrics for Identifying Incorrect Answers as FALSE")
axes[1].set_ylabel("Score")
for bar, val in zip(bars2, values_false):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=12)

plt.suptitle(f"Overall Accuracy: {accuracy:.1%} (excluding OTHER responses)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_accuracy_metrics.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Plot 5: Asking Model Response Distribution by Original Classification
fig, ax = plt.subplots(figsize=(10, 6))

# Create normalized crosstab
cross_normalized = pd.crosstab(
    df["original_class"],
    df["asking_class"],
    normalize="index"
) * 100

# Plot
cross_normalized.plot(
    kind="bar",
    ax=ax,
    color=["#95a5a6", "#e74c3c", "#2ecc71", "#f39c12"]  # ERROR, FALSE, TRUE, OTHER
)

ax.set_xlabel("Original Evaluation")
ax.set_ylabel("Percentage (%)")
ax.set_title("How the Asking Model Responded to Each Original Classification")
ax.legend(title="Asking Model Response", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_asking_by_original.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Plot 6: Performance by Topic
fig, ax = plt.subplots(figsize=(14, 8))

# Calculate accuracy per topic
topic_stats = []
for topic in df["topic"].unique():
    topic_df = df[df["topic"] == topic]

    # Only count TRUE/FALSE responses
    valid = topic_df[topic_df["asking_class"].isin(["TRUE", "FALSE"])]

    if len(valid) > 0:
        correct = len(valid[
            ((valid["asking_class"] == "TRUE") & (valid["original_class"] == "CORRECT")) |
            ((valid["asking_class"] == "FALSE") & (valid["original_class"] == "INCORRECT"))
        ])

        topic_stats.append({
            "topic": topic,
            "accuracy": correct / len(valid),
            "total": len(topic_df),
            "valid": len(valid),
            "other_rate": len(topic_df[topic_df["asking_class"] == "OTHER"]) / len(topic_df)
        })

topic_stats_df = pd.DataFrame(topic_stats).sort_values("accuracy", ascending=True)

# Plot accuracy by topic
bars = ax.barh(topic_stats_df["topic"], topic_stats_df["accuracy"], color="#3498db")

# Add accuracy labels
for i, (idx, row) in enumerate(topic_stats_df.iterrows()):
    ax.text(row["accuracy"] + 0.01, i, f"{row['accuracy']:.1%}", va="center", fontsize=10)

ax.set_xlabel("Accuracy")
ax.set_ylabel("Topic")
ax.set_title("Asking Model Accuracy by Topic (TRUE/FALSE responses only)")
ax.set_xlim(0, 1.15)
ax.axvline(x=accuracy, color="red", linestyle="--", label=f"Overall: {accuracy:.1%}")
ax.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_accuracy_by_topic.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Print summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(f"\nTotal responses evaluated: {len(df)}")
print(f"\nOriginal Evaluation Distribution:")
for cls, count in df["original_class"].value_counts().items():
    print(f"  {cls}: {count} ({100*count/len(df):.1f}%)")

print(f"\nAsking Model Response Distribution:")
for cls, count in df["asking_class"].value_counts().items():
    print(f"  {cls}: {count} ({100*count/len(df):.1f}%)")

print(f"\nConfusion Matrix Summary:")
print(f"  True Positive (TRUE + CORRECT):   {tn:4d} ({100*tn/len(df):.1f}%)")
print(f"  False Positive (TRUE + INCORRECT): {fn:4d} ({100*fn/len(df):.1f}%)")
print(f"  True Negative (FALSE + INCORRECT): {tp:4d} ({100*tp/len(df):.1f}%)")
print(f"  False Negative (FALSE + CORRECT):  {fp:4d} ({100*fp/len(df):.1f}%)")
print(f"  Other:                             {other_count:4d} ({100*other_count/len(df):.1f}%)")

print(f"\nOverall Metrics (excluding OTHER):")
print(f"  Accuracy: {accuracy:.1%}")
print(f"  For detecting CORRECT as TRUE: Precision={precision_true:.3f}, Recall={recall_true:.3f}, F1={f1_true:.3f}")
print(f"  For detecting INCORRECT as FALSE: Precision={precision_false:.3f}, Recall={recall_false:.3f}, F1={f1_false:.3f}")

# %%
# Save summary to CSV
summary_data = {
    "metric": [
        "total_responses",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "other",
        "accuracy",
        "precision_true",
        "recall_true",
        "f1_true",
        "precision_false",
        "recall_false",
        "f1_false"
    ],
    "value": [
        len(df),
        tn,  # TRUE + CORRECT
        fn,  # TRUE + INCORRECT
        tp,  # FALSE + INCORRECT
        fp,  # FALSE + CORRECT
        other_count,
        accuracy,
        precision_true,
        recall_true,
        f1_true,
        precision_false,
        recall_false,
        f1_false
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(PLOTS_DIR / "summary_metrics.csv", index=False)
print(f"\nSummary saved to {PLOTS_DIR / 'summary_metrics.csv'}")

# Save topic stats
topic_stats_df.to_csv(PLOTS_DIR / "topic_accuracy.csv", index=False)
print(f"Topic stats saved to {PLOTS_DIR / 'topic_accuracy.csv'}")
