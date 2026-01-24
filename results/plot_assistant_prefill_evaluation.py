"""
Interactive plotting script for assistant prefill attack evaluation results.
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
# Load evaluation results - Assistant Prefill Attack
RESULTS_PATH = "results/data/evaluated_assistant_prefill_responses.json"

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Flatten the data structure (topic -> items)
results_all = []
if isinstance(data, dict):
    for topic, items in data.items():
        results_all.extend(items)
elif isinstance(data, list):
    results_all = data
else:
    raise ValueError(f"Unexpected data format in {RESULTS_PATH}")

print(f"Loaded {len(results_all)} items")

# Create filtered version (excluding test topic)
results = [item for item in results_all if item.get("topic") != "test"]
print(f"Filtered out 'test' topic. Remaining items: {len(results)}")
print(f"Total responses: {sum(len(item['evaluated_responses']) for item in results)}")

# %%
# Load baseline evaluation results for comparison
BASELINE_PATH = "results/data/evaluated_responses.json"

with open(BASELINE_PATH, "r", encoding="utf-8") as f:
    baseline_data = json.load(f)

baseline_results_all = baseline_data if isinstance(baseline_data, list) else []
baseline_results = [q for q in baseline_results_all if q.get("topic") != "test"]

print(f"\nLoaded baseline: {len(baseline_results)} questions")
print(f"Baseline total responses: {sum(len(q['model_responses']) for q in baseline_results)}")

# %%
# Extract overall statistics
def extract_stats(results):
    """Extract statistics from assistant prefill evaluation results."""
    stats = []

    for item in results:
        item_id = item.get("item_id", "unknown")
        question_id = item.get("question_id", "unknown")
        topic = item.get("topic", "unknown")
        subtopic = item.get("subtopic", "")
        level = item.get("level", "")
        prefill_type = item.get("prefill_type", "unknown")
        prefill_original = item.get("prefill_original", "")

        summary = item["evaluation_summary"]
        total = sum(summary.values())

        stats.append({
            "item_id": item_id,
            "question_id": question_id,
            "topic": topic,
            "subtopic": subtopic,
            "level": level,
            "prefill_type": prefill_type,
            "prefill_original": prefill_original,
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
# Extract baseline statistics
def extract_baseline_stats(results):
    """Extract statistics from baseline evaluation results."""
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

baseline_df = extract_baseline_stats(baseline_results)
baseline_df_all = extract_baseline_stats(baseline_results_all)
print("\nBaseline statistics:")
print(baseline_df.head())

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
print(f"Total items: {len(df)}")
print(f"Unique questions: {df['question_id'].nunique()}")
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
ax1.set_title('Assistant Prefill Attack Response Distribution', fontsize=14, fontweight='bold')

# Bar chart
ax2.bar(labels, sizes, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Response Counts by Classification', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(sizes):
    ax2.text(i, v + max(sizes)*0.02, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "assistant_prefill_01_overall_distribution.png", dpi=300, bbox_inches='tight')
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
plt.savefig(PLOTS_DIR / "assistant_prefill_02_distribution_by_topic.png", dpi=300, bbox_inches='tight')
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
plt.savefig(PLOTS_DIR / "assistant_prefill_03_rates_by_topic.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 4: Distribution by prefill type
prefill_stats = df.groupby('prefill_type').agg({
    'refusal_count': 'sum',
    'correct_count': 'sum',
    'incorrect_count': 'sum',
    'error_count': 'sum',
    'total_responses': 'sum'
}).reset_index()

prefill_stats = prefill_stats.sort_values('refusal_count', ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(prefill_stats) * 0.5)))

# Counts
x = np.arange(len(prefill_stats))
width = 0.6

p1 = ax1.barh(x, prefill_stats['refusal_count'], width, label='REFUSAL', color='#ff6b6b')
p2 = ax1.barh(x, prefill_stats['correct_count'], width, left=prefill_stats['refusal_count'],
             label='CORRECT', color='#51cf66')
p3 = ax1.barh(x, prefill_stats['incorrect_count'], width,
             left=prefill_stats['refusal_count'] + prefill_stats['correct_count'],
             label='INCORRECT', color='#ffd43b')
p4 = ax1.barh(x, prefill_stats['error_count'], width,
             left=prefill_stats['refusal_count'] + prefill_stats['correct_count'] + prefill_stats['incorrect_count'],
             label='ERROR', color='#adb5bd')

ax1.set_yticks(x)
ax1.set_yticklabels(prefill_stats['prefill_type'])
ax1.set_xlabel('Count', fontsize=12)
ax1.set_title('Response Distribution by Prefill Type (Counts)', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(axis='x', alpha=0.3)

# Percentages
prefill_stats['refusal_pct'] = prefill_stats['refusal_count'] / prefill_stats['total_responses'] * 100
prefill_stats['correct_pct'] = prefill_stats['correct_count'] / prefill_stats['total_responses'] * 100
prefill_stats['incorrect_pct'] = prefill_stats['incorrect_count'] / prefill_stats['total_responses'] * 100
prefill_stats['error_pct'] = prefill_stats['error_count'] / prefill_stats['total_responses'] * 100

p1 = ax2.barh(x, prefill_stats['refusal_pct'], width, label='REFUSAL', color='#ff6b6b')
p2 = ax2.barh(x, prefill_stats['correct_pct'], width, left=prefill_stats['refusal_pct'],
             label='CORRECT', color='#51cf66')
p3 = ax2.barh(x, prefill_stats['incorrect_pct'], width,
             left=prefill_stats['refusal_pct'] + prefill_stats['correct_pct'],
             label='INCORRECT', color='#ffd43b')
p4 = ax2.barh(x, prefill_stats['error_pct'], width,
             left=prefill_stats['refusal_pct'] + prefill_stats['correct_pct'] + prefill_stats['incorrect_pct'],
             label='ERROR', color='#adb5bd')

ax2.set_yticks(x)
ax2.set_yticklabels(prefill_stats['prefill_type'])
ax2.set_xlabel('Percentage (%)', fontsize=12)
ax2.set_title('Response Distribution by Prefill Type (Percentage)', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(axis='x', alpha=0.3)
ax2.set_xlim(0, 100)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "assistant_prefill_04_distribution_by_prefill_type.png", dpi=300, bbox_inches='tight')
plt.show()

# Print statistics
print("\n" + "=" * 60)
print("STATISTICS BY PREFILL TYPE")
print("=" * 60)
for _, row in prefill_stats.iterrows():
    print(f"\n{row['prefill_type'].upper()}:")
    print(f"  Total responses: {row['total_responses']}")
    print(f"  REFUSAL:   {row['refusal_count']:4d} ({row['refusal_pct']:5.1f}%)")
    print(f"  CORRECT:   {row['correct_count']:4d} ({row['correct_pct']:5.1f}%)")
    print(f"  INCORRECT: {row['incorrect_count']:4d} ({row['incorrect_pct']:5.1f}%)")
    print(f"  ERROR:     {row['error_count']:4d} ({row['error_pct']:5.1f}%)")

# %%
# Plot 5: Attack effectiveness by prefill type
fig, ax = plt.subplots(figsize=(12, max(6, len(prefill_stats) * 0.5)))

x = np.arange(len(prefill_stats))
width = 0.25

# Sort by refusal rate for better visualization
prefill_stats_sorted = prefill_stats.sort_values('refusal_pct', ascending=True)
x = np.arange(len(prefill_stats_sorted))

ax.barh(x - width, prefill_stats_sorted['refusal_pct'], width, label='REFUSAL', color='#ff6b6b')
ax.barh(x, prefill_stats_sorted['correct_pct'], width, label='CORRECT', color='#51cf66')
ax.barh(x + width, prefill_stats_sorted['incorrect_pct'], width, label='INCORRECT', color='#ffd43b')

ax.set_yticks(x)
ax.set_yticklabels(prefill_stats_sorted['prefill_type'])
ax.set_xlabel('Rate (%)', fontsize=12)
ax.set_title('Attack Effectiveness: Response Rates by Prefill Type', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, 100)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "assistant_prefill_05_attack_effectiveness.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 6: Top prefill variations with highest refusal rates
# Group by prefill_original to see which specific prefills work best
prefill_original_stats = df.groupby('prefill_original').agg({
    'refusal_count': 'sum',
    'correct_count': 'sum',
    'incorrect_count': 'sum',
    'error_count': 'sum',
    'total_responses': 'sum',
    'prefill_type': 'first'  # Get the prefill type
}).reset_index()

prefill_original_stats['refusal_rate'] = (
    prefill_original_stats['refusal_count'] / prefill_original_stats['total_responses'] * 100
)
prefill_original_stats['correct_rate'] = (
    prefill_original_stats['correct_count'] / prefill_original_stats['total_responses'] * 100
)
prefill_original_stats['incorrect_rate'] = (
    prefill_original_stats['incorrect_count'] / prefill_original_stats['total_responses'] * 100
)

# Top 10 with highest refusal rates
top_refusal = prefill_original_stats.nlargest(10, 'refusal_rate')

print("\nTop 10 Prefills with Highest Refusal Rate:")
print("=" * 100)
for idx, row in top_refusal.iterrows():
    print(f"{row['prefill_original'][:70]:70s} | Type: {row['prefill_type']:10s} | Refusal: {row['refusal_rate']:5.1f}%")

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(top_refusal))
width = 0.25

ax.bar(x - width, top_refusal['refusal_rate'], width, label='REFUSAL', color='#ff6b6b')
ax.bar(x, top_refusal['correct_rate'], width, label='CORRECT', color='#51cf66')
ax.bar(x + width, top_refusal['incorrect_rate'], width, label='INCORRECT', color='#ffd43b')

ax.set_xticks(x)
ax.set_xticklabels([p[:30] + '...' if len(p) > 30 else p for p in top_refusal['prefill_original']],
                    rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Rate (%)', fontsize=12)
ax.set_title('Top 10 Prefills with Highest Refusal Rate', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "assistant_prefill_06_top_refusal_prefills.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 7: Bottom 10 prefills with lowest refusal rates (most successful attacks)
bottom_refusal = prefill_original_stats.nsmallest(10, 'refusal_rate')

print("\nTop 10 Most Successful Attacks (Lowest Refusal Rate):")
print("=" * 100)
for idx, row in bottom_refusal.iterrows():
    print(f"{row['prefill_original'][:70]:70s} | Type: {row['prefill_type']:10s} | Refusal: {row['refusal_rate']:5.1f}%")

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(bottom_refusal))

ax.bar(x - width, bottom_refusal['refusal_rate'], width, label='REFUSAL', color='#ff6b6b')
ax.bar(x, bottom_refusal['correct_rate'], width, label='CORRECT', color='#51cf66')
ax.bar(x + width, bottom_refusal['incorrect_rate'], width, label='INCORRECT', color='#ffd43b')

ax.set_xticks(x)
ax.set_xticklabels([p[:30] + '...' if len(p) > 30 else p for p in bottom_refusal['prefill_original']],
                    rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Rate (%)', fontsize=12)
ax.set_title('Top 10 Prefills with Lowest Refusal Rate', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "assistant_prefill_07_lowest_refusal_prefills.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 8: Questions with highest/lowest average refusal rates
# Group by question_id to see which questions are most/least resistant to attacks
question_stats = df.groupby('question_id').agg({
    'refusal_rate': 'mean',
    'correct_rate': 'mean',
    'incorrect_rate': 'mean',
    'topic': 'first',
    'total_responses': 'sum'
}).reset_index()

# Top 10 questions with highest average refusal rate (most resistant)
top_questions = question_stats.nlargest(10, 'refusal_rate')

print("\nTop 10 Questions Most Resistant to Attacks (Highest Avg Refusal Rate):")
print("=" * 100)
for idx, row in top_questions.iterrows():
    print(f"{row['question_id'][:60]:60s} | Topic: {row['topic']:15s} | Avg Refusal: {row['refusal_rate']*100:5.1f}%")

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(top_questions))
width = 0.25

ax.bar(x - width, top_questions['refusal_rate'] * 100, width, label='REFUSAL', color='#ff6b6b')
ax.bar(x, top_questions['correct_rate'] * 100, width, label='CORRECT', color='#51cf66')
ax.bar(x + width, top_questions['incorrect_rate'] * 100, width, label='INCORRECT', color='#ffd43b')

ax.set_xticks(x)
ax.set_xticklabels([q[:30] + '...' if len(q) > 30 else q for q in top_questions['question_id']],
                    rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Average Rate (%)', fontsize=12)
ax.set_title('Top 10 Questions Most Resistant to Attacks', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "assistant_prefill_08_most_resistant_questions.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Bottom 10 questions with lowest average refusal rate (most vulnerable)
bottom_questions = question_stats.nsmallest(10, 'refusal_rate')

print("\nTop 10 Questions Most Vulnerable to Attacks (Lowest Avg Refusal Rate):")
print("=" * 100)
for idx, row in bottom_questions.iterrows():
    print(f"{row['question_id'][:60]:60s} | Topic: {row['topic']:15s} | Avg Refusal: {row['refusal_rate']*100:5.1f}%")

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(bottom_questions))

ax.bar(x - width, bottom_questions['refusal_rate'] * 100, width, label='REFUSAL', color='#ff6b6b')
ax.bar(x, bottom_questions['correct_rate'] * 100, width, label='CORRECT', color='#51cf66')
ax.bar(x + width, bottom_questions['incorrect_rate'] * 100, width, label='INCORRECT', color='#ffd43b')

ax.set_xticks(x)
ax.set_xticklabels([q[:30] + '...' if len(q) > 30 else q for q in bottom_questions['question_id']],
                    rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Average Rate (%)', fontsize=12)
ax.set_title('Top 10 Questions Most Vulnerable to Attacks', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "assistant_prefill_09_most_vulnerable_questions.png", dpi=300, bbox_inches='tight')
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
# ========== COMPARISON WITH BASELINE ==========
print("\n" + "=" * 80)
print("COMPARISON WITH BASELINE")
print("=" * 80)

# Overall comparison
baseline_total_refusals = baseline_df["refusal_count"].sum()
baseline_total_correct = baseline_df["correct_count"].sum()
baseline_total_incorrect = baseline_df["incorrect_count"].sum()
baseline_total_errors = baseline_df["error_count"].sum()
baseline_total_all = baseline_total_refusals + baseline_total_correct + baseline_total_incorrect + baseline_total_errors

attack_total_refusals = df["refusal_count"].sum()
attack_total_correct = df["correct_count"].sum()
attack_total_incorrect = df["incorrect_count"].sum()
attack_total_errors = df["error_count"].sum()
attack_total_all = attack_total_refusals + attack_total_correct + attack_total_incorrect + attack_total_errors

print("\nBASELINE:")
print(f"  Total responses: {baseline_total_all}")
print(f"  REFUSAL:   {baseline_total_refusals:4d} ({100*baseline_total_refusals/baseline_total_all:5.1f}%)")
print(f"  CORRECT:   {baseline_total_correct:4d} ({100*baseline_total_correct/baseline_total_all:5.1f}%)")
print(f"  INCORRECT: {baseline_total_incorrect:4d} ({100*baseline_total_incorrect/baseline_total_all:5.1f}%)")
print(f"  ERROR:     {baseline_total_errors:4d} ({100*baseline_total_errors/baseline_total_all:5.1f}%)")

print("\nASSISTANT PREFILL ATTACK:")
print(f"  Total responses: {attack_total_all}")
print(f"  REFUSAL:   {attack_total_refusals:4d} ({100*attack_total_refusals/attack_total_all:5.1f}%)")
print(f"  CORRECT:   {attack_total_correct:4d} ({100*attack_total_correct/attack_total_all:5.1f}%)")
print(f"  INCORRECT: {attack_total_incorrect:4d} ({100*attack_total_incorrect/attack_total_all:5.1f}%)")
print(f"  ERROR:     {attack_total_errors:4d} ({100*attack_total_errors/attack_total_all:5.1f}%)")

print("\nCHANGE (Attack - Baseline):")
refusal_change = (100*attack_total_refusals/attack_total_all) - (100*baseline_total_refusals/baseline_total_all)
correct_change = (100*attack_total_correct/attack_total_all) - (100*baseline_total_correct/baseline_total_all)
incorrect_change = (100*attack_total_incorrect/attack_total_all) - (100*baseline_total_incorrect/baseline_total_all)
print(f"  REFUSAL:   {refusal_change:+6.2f} percentage points")
print(f"  CORRECT:   {correct_change:+6.2f} percentage points")
print(f"  INCORRECT: {incorrect_change:+6.2f} percentage points")

# %%
# Plot 10: Side-by-side comparison of baseline vs attack
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top left: Baseline pie chart
baseline_sizes = [baseline_total_refusals, baseline_total_correct, baseline_total_incorrect, baseline_total_errors]
labels = ['REFUSAL', 'CORRECT', 'INCORRECT', 'ERROR']
colors = ['#ff6b6b', '#51cf66', '#ffd43b', '#adb5bd']
explode = (0.05, 0.05, 0.05, 0.05)

axes[0, 0].pie(baseline_sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
axes[0, 0].set_title('Baseline Response Distribution', fontsize=14, fontweight='bold')

# Top right: Attack pie chart
attack_sizes = [attack_total_refusals, attack_total_correct, attack_total_incorrect, attack_total_errors]

axes[0, 1].pie(attack_sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
axes[0, 1].set_title('Assistant Prefill Attack Distribution', fontsize=14, fontweight='bold')

# Bottom left: Side-by-side bar comparison
x = np.arange(len(labels))
width = 0.35

baseline_pcts = [s / baseline_total_all * 100 for s in baseline_sizes]
attack_pcts = [s / attack_total_all * 100 for s in attack_sizes]

axes[1, 0].bar(x - width/2, baseline_pcts, width, label='Baseline', color='#4c72b0', alpha=0.8)
axes[1, 0].bar(x + width/2, attack_pcts, width, label='Attack', color='#dd8452', alpha=0.8)

axes[1, 0].set_ylabel('Percentage (%)', fontsize=12)
axes[1, 0].set_title('Baseline vs Attack Comparison', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(labels)
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Bottom right: Change visualization
changes = [refusal_change, correct_change, incorrect_change,
           (100*attack_total_errors/attack_total_all) - (100*baseline_total_errors/baseline_total_all)]
change_colors = ['#ff6b6b' if c > 0 else '#51cf66' for c in changes]

axes[1, 1].barh(labels, changes, color=change_colors, alpha=0.7)
axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[1, 1].set_xlabel('Change (percentage points)', fontsize=12)
axes[1, 1].set_title('Change: Attack - Baseline', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

# Add value labels
for i, v in enumerate(changes):
    axes[1, 1].text(v + (0.5 if v > 0 else -0.5), i, f'{v:+.1f}',
                    va='center', ha='left' if v > 0 else 'right', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "assistant_prefill_10_baseline_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 11: Comparison by topic
# Calculate statistics for both baseline and attack by topic
baseline_topic_stats = baseline_df_all.groupby('topic').agg({
    'refusal_count': 'sum',
    'correct_count': 'sum',
    'incorrect_count': 'sum',
    'error_count': 'sum'
}).reset_index()

baseline_topic_stats['total'] = (baseline_topic_stats['refusal_count'] +
                                  baseline_topic_stats['correct_count'] +
                                  baseline_topic_stats['incorrect_count'] +
                                  baseline_topic_stats['error_count'])
baseline_topic_stats['refusal_pct'] = baseline_topic_stats['refusal_count'] / baseline_topic_stats['total'] * 100
baseline_topic_stats['correct_pct'] = baseline_topic_stats['correct_count'] / baseline_topic_stats['total'] * 100
baseline_topic_stats['incorrect_pct'] = baseline_topic_stats['incorrect_count'] / baseline_topic_stats['total'] * 100

attack_topic_stats = df_all.groupby('topic').agg({
    'refusal_count': 'sum',
    'correct_count': 'sum',
    'incorrect_count': 'sum',
    'error_count': 'sum'
}).reset_index()

attack_topic_stats['total'] = (attack_topic_stats['refusal_count'] +
                                attack_topic_stats['correct_count'] +
                                attack_topic_stats['incorrect_count'] +
                                attack_topic_stats['error_count'])
attack_topic_stats['refusal_pct'] = attack_topic_stats['refusal_count'] / attack_topic_stats['total'] * 100
attack_topic_stats['correct_pct'] = attack_topic_stats['correct_count'] / attack_topic_stats['total'] * 100
attack_topic_stats['incorrect_pct'] = attack_topic_stats['incorrect_count'] / attack_topic_stats['total'] * 100

# Merge on topic
comparison_df = baseline_topic_stats[['topic', 'refusal_pct', 'correct_pct', 'incorrect_pct']].merge(
    attack_topic_stats[['topic', 'refusal_pct', 'correct_pct', 'incorrect_pct']],
    on='topic',
    suffixes=('_baseline', '_attack')
)

comparison_df['refusal_change'] = comparison_df['refusal_pct_attack'] - comparison_df['refusal_pct_baseline']
comparison_df['correct_change'] = comparison_df['correct_pct_attack'] - comparison_df['correct_pct_baseline']
comparison_df['incorrect_change'] = comparison_df['incorrect_pct_attack'] - comparison_df['incorrect_pct_baseline']

# Sort by refusal change (most affected topics first)
comparison_df = comparison_df.sort_values('refusal_change', ascending=True)

fig, axes = plt.subplots(1, 3, figsize=(20, max(8, len(comparison_df) * 0.6)))

x = np.arange(len(comparison_df))
width = 0.35

# Refusal rate comparison
axes[0].barh(x - width/2, comparison_df['refusal_pct_baseline'], width,
             label='Baseline', color='#4c72b0', alpha=0.8)
axes[0].barh(x + width/2, comparison_df['refusal_pct_attack'], width,
             label='Attack', color='#dd8452', alpha=0.8)
axes[0].set_yticks(x)
axes[0].set_yticklabels(comparison_df['topic'])
axes[0].set_xlabel('Refusal Rate (%)', fontsize=12)
axes[0].set_title('Refusal Rate: Baseline vs Attack', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(axis='x', alpha=0.3)

# Correct rate comparison
axes[1].barh(x - width/2, comparison_df['correct_pct_baseline'], width,
             label='Baseline', color='#4c72b0', alpha=0.8)
axes[1].barh(x + width/2, comparison_df['correct_pct_attack'], width,
             label='Attack', color='#dd8452', alpha=0.8)
axes[1].set_yticks(x)
axes[1].set_yticklabels(comparison_df['topic'])
axes[1].set_xlabel('Correct Rate (%)', fontsize=12)
axes[1].set_title('Correct Rate: Baseline vs Attack', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(axis='x', alpha=0.3)

# Change in refusal rate
change_colors = ['#ff6b6b' if c > 0 else '#51cf66' for c in comparison_df['refusal_change']]
axes[2].barh(x, comparison_df['refusal_change'], color=change_colors, alpha=0.7)
axes[2].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[2].set_yticks(x)
axes[2].set_yticklabels(comparison_df['topic'])
axes[2].set_xlabel('Change in Refusal Rate (pp)', fontsize=12)
axes[2].set_title('Refusal Rate Change by Topic', fontsize=14, fontweight='bold')
axes[2].grid(axis='x', alpha=0.3)

# Add value labels
for i, v in enumerate(comparison_df['refusal_change']):
    axes[2].text(v + (0.5 if v > 0 else -0.5), i, f'{v:+.1f}',
                va='center', ha='left' if v > 0 else 'right', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "assistant_prefill_11_topic_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nRefusal Rate Change by Topic:")
print("=" * 80)
for _, row in comparison_df.sort_values('refusal_change', ascending=False).iterrows():
    print(f"{row['topic']:20s} | Baseline: {row['refusal_pct_baseline']:5.1f}% | "
          f"Attack: {row['refusal_pct_attack']:5.1f}% | Change: {row['refusal_change']:+6.2f}pp")

# %%
# Plot 12: Attack effectiveness visualization
# Show which topics are most/least susceptible to attacks
fig, ax = plt.subplots(figsize=(14, max(8, len(comparison_df) * 0.5)))

# Sort by attack refusal rate to show effectiveness
comparison_sorted = comparison_df.sort_values('refusal_pct_attack', ascending=True)
x = np.arange(len(comparison_sorted))

# Create stacked bar showing baseline vs additional refusals from attack
baseline_refusal = comparison_sorted['refusal_pct_baseline']
additional_refusal = comparison_sorted['refusal_change'].clip(lower=0)  # Only positive changes
reduction = comparison_sorted['refusal_change'].clip(upper=0)  # Only negative changes

# For positive changes, stack on baseline
p1 = ax.barh(x, baseline_refusal, label='Baseline Refusal', color='#4c72b0', alpha=0.8)
p2 = ax.barh(x, additional_refusal, left=baseline_refusal,
             label='Additional Refusal (Attack Effect)', color='#dd8452', alpha=0.8)

# Add markers for reductions
for i, red in enumerate(reduction):
    if red < 0:
        ax.plot(comparison_sorted.iloc[i]['refusal_pct_attack'], i,
               marker='D', color='#51cf66', markersize=8,
               label='Reduced Refusal' if i == 0 and any(reduction < 0) else '')

ax.set_yticks(x)
ax.set_yticklabels(comparison_sorted['topic'])
ax.set_xlabel('Refusal Rate (%)', fontsize=12)
ax.set_title('Attack Effectiveness by Topic', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "assistant_prefill_12_attack_effectiveness_by_topic.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Export summary to CSV
output_csv = "results/data/assistant_prefill_evaluation_summary.csv"
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

output_topic_csv = "results/data/assistant_prefill_evaluation_summary_by_topic.csv"
summary_by_topic.to_csv(output_topic_csv)
print(f"Exported topic summary to: {output_topic_csv}")

summary_by_prefill_type = prefill_stats[['prefill_type', 'refusal_count', 'correct_count',
                                          'incorrect_count', 'error_count', 'refusal_pct',
                                          'correct_pct', 'incorrect_pct']]
output_prefill_csv = "results/data/assistant_prefill_attack_effectiveness.csv"
summary_by_prefill_type.to_csv(output_prefill_csv, index=False)
print(f"Exported prefill type summary to: {output_prefill_csv}")

# %%
# ========== FACT-BASED ANALYSIS ==========
print("\n" + "=" * 80)
print("FACT-BASED ANALYSIS")
print("=" * 80)

# Load facts data
try:
    FACTS_PATH = "black_box/eval/prompts/finegrained_eval_facts.json"
    with open(FACTS_PATH, "r", encoding="utf-8") as f:
        facts_data_loaded = json.load(f)
    print(f"Loaded facts from: {FACTS_PATH}")
except Exception as e:
    print(f"Warning: Could not load facts file: {e}")
    print("Skipping fact-based analysis")
    facts_data_loaded = None

# %%
if facts_data_loaded is not None:
    def get_facts_for_question(facts_data: dict, question_id: str) -> list:
        """Get the list of facts for a specific question."""
        parts = question_id.split("__")
        if len(parts) < 3:
            return []

        topic = parts[0]
        subtopic = parts[1]
        index = int(parts[2])

        if topic not in facts_data:
            return []
        topic_data = facts_data[topic]
        if subtopic not in topic_data:
            return []
        subtopic_data = topic_data[subtopic]
        if index >= len(subtopic_data):
            return []

        return subtopic_data[index].get("facts", [])

    def extract_fact_stats_from_results(results: list) -> pd.DataFrame:
        """
        Extract fact-based statistics from evaluation results.
        Note: This requires the evaluation results to have fact_checks in the evaluation.
        """
        fact_stats = []

        for item in results:
            question_id = item.get("question_id", "unknown")
            topic = item.get("topic", "unknown")
            facts = get_facts_for_question(facts_data_loaded, question_id)

            if not facts:
                continue

            num_facts_total = len(facts)

            # Process each evaluated response
            for resp in item.get("evaluated_responses", []):
                evaluation = resp.get("evaluation", {})

                # Check if this evaluation has fact checks
                fact_checks = evaluation.get("fact_checks", [])
                refusal_classification = evaluation.get("refusal_classification", {})
                is_refusal = refusal_classification.get("is_refusal", None)

                if is_refusal == True:
                    # It's a refusal, no facts to count
                    num_facts_included = 0
                    facts_mentioned = []
                elif fact_checks:
                    # Count how many facts were mentioned
                    num_facts_included = sum(1 for fc in fact_checks if fc.get("mentioned") == True)
                    facts_mentioned = [fc["fact"] for fc in fact_checks if fc.get("mentioned") == True]
                else:
                    # No fact checks available, skip
                    continue

                fact_stats.append({
                    "question_id": question_id,
                    "topic": topic,
                    "num_facts_total": num_facts_total,
                    "num_facts_included": num_facts_included,
                    "fact_coverage": num_facts_included / num_facts_total if num_facts_total > 0 else 0,
                    "is_refusal": is_refusal,
                    "facts_mentioned": facts_mentioned,
                    "all_facts": facts
                })

        return pd.DataFrame(fact_stats)

    # Try to load fact-based evaluation results
    # First check if we have fact-based evaluations
    print("\nLooking for fact-based evaluation results...")

    # Check the structure of our current results
    sample_item = results[0] if results else None
    has_fact_checks = False

    if sample_item and "evaluated_responses" in sample_item:
        sample_resp = sample_item["evaluated_responses"][0] if sample_item["evaluated_responses"] else None
        if sample_resp and "evaluation" in sample_resp:
            has_fact_checks = "fact_checks" in sample_resp["evaluation"]

    if has_fact_checks:
        print("Found fact-based evaluations in current data!")
        fact_df = extract_fact_stats_from_results(results_all)
    else:
        print("Current data does not have fact-based evaluations.")
        print("Fact-based analysis requires running the *_facts.py evaluation scripts first.")
        fact_df = pd.DataFrame()

# %%
if facts_data_loaded is not None and not fact_df.empty:
    # Analysis 1: Correctness vs Number of Facts Included
    print("\n" + "=" * 60)
    print("ANALYSIS: Correctness vs Number of Facts Included")
    print("=" * 60)

    # Filter out refusals for this analysis
    non_refusal_df = fact_df[fact_df['is_refusal'] == False].copy()

    if not non_refusal_df.empty:
        # Group by number of facts included
        facts_summary = non_refusal_df.groupby('num_facts_included').agg({
            'question_id': 'count',
            'fact_coverage': 'mean'
        }).rename(columns={'question_id': 'response_count'})

        print(facts_summary)

        # Plot: Distribution of number of facts included
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Histogram of facts included
        ax1.hist(non_refusal_df['num_facts_included'], bins=range(0, non_refusal_df['num_facts_total'].max() + 2),
                 edgecolor='black', alpha=0.7, color='#4c72b0')
        ax1.set_xlabel('Number of Facts Included', fontsize=12)
        ax1.set_ylabel('Number of Responses', fontsize=12)
        ax1.set_title('Distribution of Facts Included in Non-Refusal Responses', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Fact coverage distribution
        ax2.hist(non_refusal_df['fact_coverage'] * 100, bins=20, edgecolor='black', alpha=0.7, color='#51cf66')
        ax2.set_xlabel('Fact Coverage (%)', fontsize=12)
        ax2.set_ylabel('Number of Responses', fontsize=12)
        ax2.set_title('Distribution of Fact Coverage in Non-Refusal Responses', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xlim(0, 100)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "assistant_prefill_13_fact_inclusion_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Average facts included by topic
        topic_fact_stats = non_refusal_df.groupby('topic').agg({
            'num_facts_included': 'mean',
            'fact_coverage': 'mean',
            'question_id': 'count'
        }).rename(columns={'question_id': 'response_count'}).sort_values('fact_coverage', ascending=False)

        print("\nAverage Fact Coverage by Topic:")
        print(topic_fact_stats)

        fig, ax = plt.subplots(figsize=(12, max(6, len(topic_fact_stats) * 0.5)))
        x = np.arange(len(topic_fact_stats))

        ax.barh(x, topic_fact_stats['fact_coverage'] * 100, color='#51cf66', alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(topic_fact_stats.index)
        ax.set_xlabel('Average Fact Coverage (%)', fontsize=12)
        ax.set_title('Average Fact Coverage by Topic (Non-Refusals)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 100)

        # Add value labels
        for i, v in enumerate(topic_fact_stats['fact_coverage'] * 100):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "assistant_prefill_14_fact_coverage_by_topic.png", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("No non-refusal responses found for fact analysis")

# %%
if facts_data_loaded is not None and not fact_df.empty:
    # Analysis 2: Which specific facts are most commonly included?
    print("\n" + "=" * 60)
    print("ANALYSIS: Which Specific Facts Are Most Commonly Included?")
    print("=" * 60)

    non_refusal_df = fact_df[fact_df['is_refusal'] == False].copy()

    if not non_refusal_df.empty:
        # Count frequency of each fact across all responses
        from collections import Counter

        all_mentioned_facts = []
        for facts_list in non_refusal_df['facts_mentioned']:
            all_mentioned_facts.extend(facts_list)

        fact_counter = Counter(all_mentioned_facts)

        # Get top 20 most commonly included facts
        top_facts = fact_counter.most_common(20)

        if top_facts:
            print("\nTop 20 Most Commonly Included Facts:")
            for i, (fact, count) in enumerate(top_facts, 1):
                print(f"{i:2d}. ({count:3d} times) {fact[:80]}")

            # Plot top facts
            facts_df_plot = pd.DataFrame(top_facts, columns=['fact', 'count'])

            fig, ax = plt.subplots(figsize=(14, max(8, len(facts_df_plot) * 0.4)))

            x = np.arange(len(facts_df_plot))
            ax.barh(x, facts_df_plot['count'], color='#4c72b0', alpha=0.8)
            ax.set_yticks(x)
            ax.set_yticklabels([f[:60] + '...' if len(f) > 60 else f for f in facts_df_plot['fact']], fontsize=9)
            ax.set_xlabel('Number of Responses Including This Fact', fontsize=12)
            ax.set_title('Top 20 Most Commonly Included Facts', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            # Add value labels
            for i, v in enumerate(facts_df_plot['count']):
                ax.text(v + max(facts_df_plot['count'])*0.01, i, str(v), va='center', fontsize=9)

            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "assistant_prefill_15_most_common_facts.png", dpi=300, bbox_inches='tight')
            plt.show()
    else:
        print("No non-refusal responses found for fact analysis")

# %%
if facts_data_loaded is not None and not fact_df.empty:
    # Analysis 3: Fact inclusion patterns - are the same facts always included?
    print("\n" + "=" * 60)
    print("ANALYSIS: Fact Inclusion Patterns and Consistency")
    print("=" * 60)

    non_refusal_df = fact_df[fact_df['is_refusal'] == False].copy()

    if not non_refusal_df.empty:
        # For each question, analyze which facts are included across different responses
        question_fact_patterns = {}

        for question_id in non_refusal_df['question_id'].unique():
            q_responses = non_refusal_df[non_refusal_df['question_id'] == question_id]

            if len(q_responses) == 0:
                continue

            # Get all possible facts for this question
            all_facts = q_responses.iloc[0]['all_facts']

            if not all_facts:
                continue

            # Count how often each fact is included
            fact_inclusion_counts = {fact: 0 for fact in all_facts}

            for _, row in q_responses.iterrows():
                for fact in row['facts_mentioned']:
                    if fact in fact_inclusion_counts:
                        fact_inclusion_counts[fact] += 1

            # Calculate inclusion rates
            total_responses = len(q_responses)
            fact_inclusion_rates = {fact: count / total_responses
                                   for fact, count in fact_inclusion_counts.items()}

            question_fact_patterns[question_id] = {
                'total_responses': total_responses,
                'fact_inclusion_counts': fact_inclusion_counts,
                'fact_inclusion_rates': fact_inclusion_rates,
                'all_facts': all_facts
            }

        # Analyze: For questions with partial fact inclusion, which facts are most consistently included?
        print("\nQuestions with Partial Fact Inclusion (not all facts always included):")
        print("=" * 80)

        partial_inclusion_questions = []

        for question_id, pattern in question_fact_patterns.items():
            rates = list(pattern['fact_inclusion_rates'].values())

            # Check if there's variation (not all 0% or all 100%)
            if len(set(rates)) > 1 and any(0 < r < 1 for r in rates):
                partial_inclusion_questions.append(question_id)

                print(f"\nQuestion: {question_id}")
                print(f"Total responses: {pattern['total_responses']}")

                # Sort facts by inclusion rate
                sorted_facts = sorted(pattern['fact_inclusion_rates'].items(),
                                    key=lambda x: x[1], reverse=True)

                for fact, rate in sorted_facts:
                    count = pattern['fact_inclusion_counts'][fact]
                    print(f"  {rate*100:5.1f}% ({count:2d}/{pattern['total_responses']:2d}): {fact[:70]}")

        # Visualize fact inclusion patterns for a few example questions
        if partial_inclusion_questions:
            # Pick up to 5 questions with the most variation
            example_questions = partial_inclusion_questions[:5]

            fig, axes = plt.subplots(len(example_questions), 1,
                                    figsize=(14, max(4, len(example_questions) * 3)))

            if len(example_questions) == 1:
                axes = [axes]

            for idx, question_id in enumerate(example_questions):
                pattern = question_fact_patterns[question_id]

                sorted_facts = sorted(pattern['fact_inclusion_rates'].items(),
                                    key=lambda x: x[1], reverse=True)

                facts = [f[:50] + '...' if len(f) > 50 else f for f, _ in sorted_facts]
                rates = [r * 100 for _, r in sorted_facts]

                x = np.arange(len(facts))
                colors = ['#51cf66' if r > 75 else '#ffd43b' if r > 25 else '#ff6b6b' for r in rates]

                axes[idx].barh(x, rates, color=colors, alpha=0.8)
                axes[idx].set_yticks(x)
                axes[idx].set_yticklabels(facts, fontsize=8)
                axes[idx].set_xlabel('Inclusion Rate (%)', fontsize=10)
                axes[idx].set_title(f'{question_id}', fontsize=11, fontweight='bold')
                axes[idx].grid(axis='x', alpha=0.3)
                axes[idx].set_xlim(0, 100)

                # Add value labels
                for i, v in enumerate(rates):
                    axes[idx].text(v + 2, i, f'{v:.0f}%', va='center', fontsize=8)

            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "assistant_prefill_16_fact_inclusion_patterns.png", dpi=300, bbox_inches='tight')
            plt.show()

        # Summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY: Fact Inclusion Consistency")
        print("=" * 60)

        # Calculate overall consistency metric
        all_inclusion_rates = []
        for pattern in question_fact_patterns.values():
            all_inclusion_rates.extend(pattern['fact_inclusion_rates'].values())

        if all_inclusion_rates:
            print(f"Average fact inclusion rate: {np.mean(all_inclusion_rates)*100:.1f}%")
            print(f"Std dev of fact inclusion rates: {np.std(all_inclusion_rates)*100:.1f}%")
            print(f"Questions with partial inclusion: {len(partial_inclusion_questions)}/{len(question_fact_patterns)}")
    else:
        print("No non-refusal responses found for fact analysis")

# %%
