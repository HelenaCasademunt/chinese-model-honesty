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

        for question in results:
            question_id = question.get("question_id", "unknown")
            topic = question.get("topic", "unknown")
            facts = get_facts_for_question(facts_data_loaded, question_id)

            if not facts:
                continue

            num_facts_total = len(facts)

            # Process each model response
            for resp in question.get("model_responses", []):
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
    print("\nLooking for fact-based evaluation results...")

    # Check the structure of our current results
    sample_question = results[0] if results else None
    has_fact_checks = False

    if sample_question and "model_responses" in sample_question:
        sample_resp = sample_question["model_responses"][0] if sample_question["model_responses"] else None
        if sample_resp and "evaluation" in sample_resp:
            has_fact_checks = "fact_checks" in sample_resp["evaluation"]

    if has_fact_checks:
        print("Found fact-based evaluations in current data!")
        fact_df = extract_fact_stats_from_results(results_all)
    else:
        print("Current data does not have fact-based evaluations.")
        print("Fact-based analysis requires running the evaluate_truthfulness_facts.py script first.")
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
        plt.savefig(PLOTS_DIR / "baseline_07_fact_inclusion_distribution.png", dpi=300, bbox_inches='tight')
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
        plt.savefig(PLOTS_DIR / "baseline_08_fact_coverage_by_topic.png", dpi=300, bbox_inches='tight')
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
            plt.savefig(PLOTS_DIR / "baseline_09_most_common_facts.png", dpi=300, bbox_inches='tight')
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
            plt.savefig(PLOTS_DIR / "baseline_10_fact_inclusion_patterns.png", dpi=300, bbox_inches='tight')
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
