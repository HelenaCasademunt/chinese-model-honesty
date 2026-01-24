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

# %%
# Load evaluation results
RESULTS_PATH = "eval/evaluated_responses.json"

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

print(f"Loaded {len(results)} questions")
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

df = extract_stats(results)
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
plt.show()

# %%
# Plot 2: Distribution by topic
topic_stats = df.groupby('topic').agg({
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
plt.show()

# %%
# Plot 3: Refusal rate by topic
topic_rates = df.groupby('topic').agg({
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
plt.show()

# %%
# Plot 4: Heatmap of rates by topic and subtopic (if subtopics exist)
if df['subtopic'].notna().any() and df['subtopic'].ne('').any():
    # Filter out empty subtopics
    df_with_subtopics = df[df['subtopic'].notna() & (df['subtopic'] != '')]

    if len(df_with_subtopics) > 0:
        subtopic_stats = df_with_subtopics.groupby(['topic', 'subtopic']).agg({
            'refusal_rate': 'mean',
            'correct_rate': 'mean',
            'incorrect_rate': 'mean',
        }).reset_index()

        # Create pivot tables for each metric
        for metric, title, cmap in [
            ('refusal_rate', 'Refusal Rate by Topic and Subtopic', 'Reds'),
            ('correct_rate', 'Correct Rate by Topic and Subtopic', 'Greens'),
            ('incorrect_rate', 'Incorrect Rate by Topic and Subtopic', 'YlOrBr')
        ]:
            pivot = subtopic_stats.pivot(index='subtopic', columns='topic', values=metric)

            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.4)))
                sns.heatmap(pivot * 100, annot=True, fmt='.1f', cmap=cmap,
                           cbar_kws={'label': 'Rate (%)'}, ax=ax, linewidths=0.5)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel('Topic', fontsize=12)
                ax.set_ylabel('Subtopic', fontsize=12)
                plt.tight_layout()
                plt.show()

# %%
# Plot 5: Distribution of responses per question
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Histogram of refusal rates
ax1.hist(df['refusal_rate'] * 100, bins=20, color='#ff6b6b', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Refusal Rate (%)', fontsize=12)
ax1.set_ylabel('Number of Questions', fontsize=12)
ax1.set_title('Distribution of Refusal Rates Across Questions', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.axvline(df['refusal_rate'].mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {df["refusal_rate"].mean()*100:.1f}%')
ax1.legend()

# Histogram of correct rates
ax2.hist(df['correct_rate'] * 100, bins=20, color='#51cf66', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Correct Rate (%)', fontsize=12)
ax2.set_ylabel('Number of Questions', fontsize=12)
ax2.set_title('Distribution of Correct Rates Across Questions', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.axvline(df['correct_rate'].mean() * 100, color='green', linestyle='--', linewidth=2, label=f'Mean: {df["correct_rate"].mean()*100:.1f}%')
ax2.legend()

plt.tight_layout()
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
output_csv = "eval/evaluation_summary.csv"
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

output_topic_csv = "eval/evaluation_summary_by_topic.csv"
summary_by_topic.to_csv(output_topic_csv)
print(f"Exported topic summary to: {output_topic_csv}")

# %%
