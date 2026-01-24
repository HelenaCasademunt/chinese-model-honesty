"""
Interactive plotting script for evaluation results.
Use with VSCode's Jupyter extension to run cells interactively.
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

# Create plots and data directories
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("results/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Load evaluation results
RESULTS_PATH = "results/data/evaluated_responses.json"

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results_all = json.load(f)

print(f"Loaded {len(results_all)} questions")

# Create filtered version (excluding test topic)
results = [q for q in results_all if q.get("topic") != "test"]
print(f"Filtered out 'test' topic. Remaining questions: {len(results)}")
print(f"Total responses: {sum(len(q['model_responses']) for q in results)}")

# %%
# Extract overall statistics
def extract_stats(results):
    """Extract statistics from evaluation results."""
    stats = []

    for question in results:
        q_id = question["question_id"]
        topic = question["topic"]
        subtopic = question.get("subtopic", "")
        level = question.get("level", "")

        summary = question["evaluation_summary"]
        total = sum(summary.values())

        stats.append({
            "question_id": q_id,
            "topic": topic,
            "subtopic": subtopic,
            "level": level,
            "total_responses": total,
            "refusal_count": summary.get("REFUSAL", 0),
            "correct_count": summary.get("CORRECT", 0),
            "incorrect_count": summary.get("INCORRECT", 0),
            "error_count": summary.get("ERROR", 0),
            "refusal_rate": summary.get("REFUSAL", 0) / total if total > 0 else 0,
            "correct_rate": summary.get("CORRECT", 0) / total if total > 0 else 0,
            "incorrect_rate": summary.get("INCORRECT", 0) / total if total > 0 else 0,
        })

    return pd.DataFrame(stats)

# Create both filtered and unfiltered dataframes
df = extract_stats(results)  # Filtered (no test topic)
df_all = extract_stats(results_all)  # Unfiltered (includes test topic)
print(df.head())

# %%
# Overall distribution
total_refusals = df["refusal_count"].sum()
total_correct = df["correct_count"].sum()
total_incorrect = df["incorrect_count"].sum()
total_errors = df["error_count"].sum()
total_all = total_refusals + total_correct + total_incorrect + total_errors

print("=" * 60)
print("OVERALL STATISTICS")
print("=" * 60)
print(f"Total questions: {len(df)}")
print(f"Total responses: {total_all}")
print(f"  REFUSAL:   {total_refusals:4d} ({100*total_refusals/total_all:5.1f}%)")
print(f"  CORRECT:   {total_correct:4d} ({100*total_correct/total_all:5.1f}%)")
print(f"  INCORRECT: {total_incorrect:4d} ({100*total_incorrect/total_all:5.1f}%)")
print(f"  ERROR:     {total_errors:4d} ({100*total_errors/total_all:5.1f}%)")

# %%
# Plot 1: Overall distribution pie chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
sizes = [total_refusals, total_correct, total_incorrect, total_errors]
labels = ['REFUSAL', 'CORRECT', 'INCORRECT', 'ERROR']
colors = ['#ff6b6b', '#51cf66', '#ffd43b', '#adb5bd']
explode = (0.05, 0.05, 0.05, 0.05)

ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.set_title('Overall Response Classification Distribution', fontsize=14, fontweight='bold')

# Bar chart
ax2.bar(labels, sizes, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Response Counts by Classification', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(sizes):
    ax2.text(i, v + max(sizes)*0.02, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_overall_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 2: Distribution by topic (includes test topic)
topic_stats = df_all.groupby('topic').agg({
    'refusal_count': 'sum',
    'correct_count': 'sum',
    'incorrect_count': 'sum',
    'error_count': 'sum'
}).reset_index()

topic_stats['total'] = (topic_stats['refusal_count'] +
                        topic_stats['correct_count'] +
                        topic_stats['incorrect_count'] +
                        topic_stats['error_count'])

topic_stats = topic_stats.sort_values('total', ascending=True)

fig, ax = plt.subplots(figsize=(12, max(6, len(topic_stats) * 0.5)))

x = np.arange(len(topic_stats))
width = 0.6

# Stacked bar chart
p1 = ax.barh(x, topic_stats['refusal_count'], width, label='REFUSAL', color='#ff6b6b')
p2 = ax.barh(x, topic_stats['correct_count'], width, left=topic_stats['refusal_count'],
             label='CORRECT', color='#51cf66')
p3 = ax.barh(x, topic_stats['incorrect_count'], width,
             left=topic_stats['refusal_count'] + topic_stats['correct_count'],
             label='INCORRECT', color='#ffd43b')
p4 = ax.barh(x, topic_stats['error_count'], width,
             left=topic_stats['refusal_count'] + topic_stats['correct_count'] + topic_stats['incorrect_count'],
             label='ERROR', color='#adb5bd')

ax.set_yticks(x)
ax.set_yticklabels(topic_stats['topic'])
ax.set_xlabel('Number of Responses', fontsize=12)
ax.set_title('Response Distribution by Topic', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_distribution_by_topic.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 3: Refusal rate by topic
topic_rates = df_all.groupby('topic').agg({
    'refusal_rate': 'mean',
    'correct_rate': 'mean',
    'incorrect_rate': 'mean',
}).reset_index()

topic_rates = topic_rates.sort_values('refusal_rate', ascending=False)

fig, ax = plt.subplots(figsize=(12, max(6, len(topic_rates) * 0.5)))

x = np.arange(len(topic_rates))
width = 0.25

ax.barh(x - width, topic_rates['refusal_rate'] * 100, width, label='REFUSAL', color='#ff6b6b')
ax.barh(x, topic_rates['correct_rate'] * 100, width, label='CORRECT', color='#51cf66')
ax.barh(x + width, topic_rates['incorrect_rate'] * 100, width, label='INCORRECT', color='#ffd43b')

ax.set_yticks(x)
ax.set_yticklabels(topic_rates['topic'])
ax.set_xlabel('Rate (%)', fontsize=12)
ax.set_title('Average Classification Rates by Topic', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, 100)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_rates_by_topic.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 4: Distribution by question level (broad, medium, targeted)
if df['level'].notna().any() and df['level'].ne('').any():
    level_stats = df.groupby('level').agg({
        'refusal_count': 'sum',
        'correct_count': 'sum',
        'incorrect_count': 'sum',
        'error_count': 'sum'
    }).reset_index()

    level_stats['total'] = (level_stats['refusal_count'] +
                            level_stats['correct_count'] +
                            level_stats['incorrect_count'] +
                            level_stats['error_count'])

    # Sort by a custom order if possible (broad, medium, targeted)
    level_order = ['broad', 'medium', 'targeted']
    level_stats['level'] = pd.Categorical(level_stats['level'], categories=level_order, ordered=True)
    level_stats = level_stats.sort_values('level')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Stacked bar chart (counts)
    x = np.arange(len(level_stats))
    width = 0.6

    p1 = ax1.bar(x, level_stats['refusal_count'], width, label='REFUSAL', color='#ff6b6b')
    p2 = ax1.bar(x, level_stats['correct_count'], width, bottom=level_stats['refusal_count'],
                 label='CORRECT', color='#51cf66')
    p3 = ax1.bar(x, level_stats['incorrect_count'], width,
                 bottom=level_stats['refusal_count'] + level_stats['correct_count'],
                 label='INCORRECT', color='#ffd43b')
    p4 = ax1.bar(x, level_stats['error_count'], width,
                 bottom=level_stats['refusal_count'] + level_stats['correct_count'] + level_stats['incorrect_count'],
                 label='ERROR', color='#adb5bd')

    ax1.set_xticks(x)
    ax1.set_xticklabels(level_stats['level'], fontsize=12)
    ax1.set_ylabel('Number of Responses', fontsize=12)
    ax1.set_title('Response Distribution by Question Level (Counts)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # Percentage bar chart
    level_stats['refusal_pct'] = level_stats['refusal_count'] / level_stats['total'] * 100
    level_stats['correct_pct'] = level_stats['correct_count'] / level_stats['total'] * 100
    level_stats['incorrect_pct'] = level_stats['incorrect_count'] / level_stats['total'] * 100
    level_stats['error_pct'] = level_stats['error_count'] / level_stats['total'] * 100

    p1 = ax2.bar(x, level_stats['refusal_pct'], width, label='REFUSAL', color='#ff6b6b')
    p2 = ax2.bar(x, level_stats['correct_pct'], width, bottom=level_stats['refusal_pct'],
                 label='CORRECT', color='#51cf66')
    p3 = ax2.bar(x, level_stats['incorrect_pct'], width,
                 bottom=level_stats['refusal_pct'] + level_stats['correct_pct'],
                 label='INCORRECT', color='#ffd43b')
    p4 = ax2.bar(x, level_stats['error_pct'], width,
                 bottom=level_stats['refusal_pct'] + level_stats['correct_pct'] + level_stats['incorrect_pct'],
                 label='ERROR', color='#adb5bd')

    ax2.set_xticks(x)
    ax2.set_xticklabels(level_stats['level'], fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Response Distribution by Question Level (Percentage)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_distribution_by_level.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    print("\n" + "=" * 60)
    print("STATISTICS BY QUESTION LEVEL")
    print("=" * 60)
    for _, row in level_stats.iterrows():
        print(f"\n{row['level'].upper()}:")
        print(f"  Total responses: {row['total']}")
        print(f"  REFUSAL:   {row['refusal_count']:4d} ({row['refusal_pct']:5.1f}%)")
        print(f"  CORRECT:   {row['correct_count']:4d} ({row['correct_pct']:5.1f}%)")
        print(f"  INCORRECT: {row['incorrect_count']:4d} ({row['incorrect_pct']:5.1f}%)")
        print(f"  ERROR:     {row['error_count']:4d} ({row['error_pct']:5.1f}%)")

# %%
# Plot 5: Distribution by topic (percentage view)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(topic_stats) * 0.6)))

# Counts (already have this in plot 2, but let's make it side-by-side with percentages)
x = np.arange(len(topic_stats))
width = 0.6

# Stacked bar chart - counts
p1 = ax1.barh(x, topic_stats['refusal_count'], width, label='REFUSAL', color='#ff6b6b')
p2 = ax1.barh(x, topic_stats['correct_count'], width, left=topic_stats['refusal_count'],
             label='CORRECT', color='#51cf66')
p3 = ax1.barh(x, topic_stats['incorrect_count'], width,
             left=topic_stats['refusal_count'] + topic_stats['correct_count'],
             label='INCORRECT', color='#ffd43b')
p4 = ax1.barh(x, topic_stats['error_count'], width,
             left=topic_stats['refusal_count'] + topic_stats['correct_count'] + topic_stats['incorrect_count'],
             label='ERROR', color='#adb5bd')

ax1.set_yticks(x)
ax1.set_yticklabels(topic_stats['topic'])
ax1.set_xlabel('Number of Responses', fontsize=12)
ax1.set_title('Response Distribution by Topic (Counts)', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(axis='x', alpha=0.3)

# Percentage view
topic_stats['refusal_pct'] = topic_stats['refusal_count'] / topic_stats['total'] * 100
topic_stats['correct_pct'] = topic_stats['correct_count'] / topic_stats['total'] * 100
topic_stats['incorrect_pct'] = topic_stats['incorrect_count'] / topic_stats['total'] * 100
topic_stats['error_pct'] = topic_stats['error_count'] / topic_stats['total'] * 100

p1 = ax2.barh(x, topic_stats['refusal_pct'], width, label='REFUSAL', color='#ff6b6b')
p2 = ax2.barh(x, topic_stats['correct_pct'], width, left=topic_stats['refusal_pct'],
             label='CORRECT', color='#51cf66')
p3 = ax2.barh(x, topic_stats['incorrect_pct'], width,
             left=topic_stats['refusal_pct'] + topic_stats['correct_pct'],
             label='INCORRECT', color='#ffd43b')
p4 = ax2.barh(x, topic_stats['error_pct'], width,
             left=topic_stats['refusal_pct'] + topic_stats['correct_pct'] + topic_stats['incorrect_pct'],
             label='ERROR', color='#adb5bd')

ax2.set_yticks(x)
ax2.set_yticklabels(topic_stats['topic'])
ax2.set_xlabel('Percentage (%)', fontsize=12)
ax2.set_title('Response Distribution by Topic (Percentage)', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(axis='x', alpha=0.3)
ax2.set_xlim(0, 100)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_distribution_by_topic_detailed.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 6: Top 10 questions with highest refusal rate
top_refusal = df.nlargest(10, 'refusal_rate')[['question_id', 'topic', 'refusal_rate', 'correct_rate', 'incorrect_rate']]

print("Top 10 Questions with Highest Refusal Rate:")
print("=" * 80)
for idx, row in top_refusal.iterrows():
    print(f"{row['question_id'][:50]:50s} - {row['topic']:15s} - Refusal: {row['refusal_rate']*100:5.1f}%")

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(top_refusal))
width = 0.25

ax.bar(x - width, top_refusal['refusal_rate'] * 100, width, label='REFUSAL', color='#ff6b6b')
ax.bar(x, top_refusal['correct_rate'] * 100, width, label='CORRECT', color='#51cf66')
ax.bar(x + width, top_refusal['incorrect_rate'] * 100, width, label='INCORRECT', color='#ffd43b')

ax.set_xticks(x)
ax.set_xticklabels([q[:30] + '...' if len(q) > 30 else q for q in top_refusal['question_id']],
                    rotation=45, ha='right')
ax.set_ylabel('Rate (%)', fontsize=12)
ax.set_title('Top 10 Questions with Highest Refusal Rate', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_top_refusal_questions.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 7: Bottom 10 questions with lowest refusal rate (most honest)
bottom_refusal = df.nsmallest(10, 'refusal_rate')[['question_id', 'topic', 'refusal_rate', 'correct_rate', 'incorrect_rate']]

print("\nTop 10 Most Honest Questions (Lowest Refusal Rate):")
print("=" * 80)
for idx, row in bottom_refusal.iterrows():
    print(f"{row['question_id'][:50]:50s} - {row['topic']:15s} - Refusal: {row['refusal_rate']*100:5.1f}%")

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(bottom_refusal))

ax.bar(x - width, bottom_refusal['refusal_rate'] * 100, width, label='REFUSAL', color='#ff6b6b')
ax.bar(x, bottom_refusal['correct_rate'] * 100, width, label='CORRECT', color='#51cf66')
ax.bar(x + width, bottom_refusal['incorrect_rate'] * 100, width, label='INCORRECT', color='#ffd43b')

ax.set_xticks(x)
ax.set_xticklabels([q[:30] + '...' if len(q) > 30 else q for q in bottom_refusal['question_id']],
                    rotation=45, ha='right')
ax.set_ylabel('Rate (%)', fontsize=12)
ax.set_title('Top 10 Questions with Lowest Refusal Rate', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "07_lowest_refusal_questions.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Summary statistics table
print("\nSummary Statistics by Topic:")
print("=" * 100)
summary = df.groupby('topic').agg({
    'refusal_rate': ['mean', 'std', 'min', 'max'],
    'correct_rate': ['mean', 'std', 'min', 'max'],
    'incorrect_rate': ['mean', 'std', 'min', 'max'],
}).round(3)

# Format as percentages
summary = summary * 100
print(summary.to_string())

# %%
# Export summary to CSV
output_csv = "results/data/evaluation_summary.csv"
df.to_csv(output_csv, index=False)
print(f"\nExported detailed statistics to: {output_csv}")

summary_by_topic = df.groupby('topic').agg({
    'refusal_count': 'sum',
    'correct_count': 'sum',
    'incorrect_count': 'sum',
    'error_count': 'sum',
    'refusal_rate': 'mean',
    'correct_rate': 'mean',
    'incorrect_rate': 'mean',
}).round(3)

output_topic_csv = "results/data/evaluation_summary_by_topic.csv"
summary_by_topic.to_csv(output_topic_csv)
print(f"Exported topic summary to: {output_topic_csv}")

# %%
