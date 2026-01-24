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
# Load evaluation results for both models
RESULTS_PATH = "results/data/evaluated_responses.json"
LLAMA_RESULTS_PATH = "results/data/evaluated_responses_llama70b.json"

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results_all = json.load(f)

print(f"Loaded {len(results_all)} Qwen questions")

# Create filtered version (excluding test topic)
results = [q for q in results_all if q.get("topic") != "test"]
print(f"Filtered out 'test' topic. Remaining questions: {len(results)}")
print(f"Total responses: {sum(len(q['model_responses']) for q in results)}")

# Load Llama results
llama_results_all = None
llama_results = None
try:
    with open(LLAMA_RESULTS_PATH, "r", encoding="utf-8") as f:
        llama_results_all = json.load(f)
    llama_results = [q for q in llama_results_all if q.get("topic") != "test"]
    print(f"Loaded {len(llama_results)} Llama questions")
    print(f"Llama total responses: {sum(len(q['model_responses']) for q in llama_results)}")
except Exception as e:
    print(f"Warning: Could not load Llama results: {e}")

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

# Create Llama dataframes if available
df_llama = None
df_llama_all = None
if llama_results:
    df_llama = extract_stats(llama_results)
    df_llama_all = extract_stats(llama_results_all)
    print(f"\nLlama stats loaded: {len(df_llama)} questions")

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
# Plot 1: Overall distribution comparison (Qwen vs Llama) as stacked bars (excluding errors)
fig, ax = plt.subplots(figsize=(12, 6))

# Qwen stats (excluding errors for percentage calculation)
qwen_stats = {
    'refusal': total_refusals,
    'correct': total_correct,
    'incorrect': total_incorrect,
}
qwen_total = sum(qwen_stats.values())

# Llama stats (if available)
if df_llama is not None:
    llama_refusals = df_llama["refusal_count"].sum()
    llama_correct = df_llama["correct_count"].sum()
    llama_incorrect = df_llama["incorrect_count"].sum()
    llama_total = llama_refusals + llama_correct + llama_incorrect

    llama_stats = {
        'refusal': llama_refusals,
        'correct': llama_correct,
        'incorrect': llama_incorrect,
    }

    models = ['Qwen', 'Llama 70B']
    totals = [qwen_total, llama_total]
else:
    models = ['Qwen']
    totals = [qwen_total]
    llama_stats = None

colors = ['#ff6b6b', '#51cf66', '#ffd43b']
labels = ['REFUSAL', 'CORRECT', 'INCORRECT']

x = np.arange(len(models))
width = 0.5

# Calculate percentages and create stacked bars
bottom = np.zeros(len(models))

for i, (label, color) in enumerate(zip(labels, colors)):
    if llama_stats:
        values = [qwen_stats[label.lower()] / qwen_total * 100,
                  llama_stats[label.lower()] / llama_total * 100]
    else:
        values = [qwen_stats[label.lower()] / qwen_total * 100]

    bars = ax.bar(x, values, width, label=label, color=color, bottom=bottom)
    bottom += values

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Overall Response Distribution: Qwen vs Llama', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 100)

# Add percentage labels on bars
for i, model in enumerate(models):
    if model == 'Qwen':
        stats = qwen_stats
        total = qwen_total
    else:
        stats = llama_stats
        total = llama_total

    cumulative = 0
    for label in labels:
        pct = stats[label.lower()] / total * 100
        if pct > 5:  # Only label if segment is large enough
            ax.text(i, cumulative + pct/2, f'{pct:.1f}%',
                   ha='center', va='center', fontweight='bold', fontsize=10)
        cumulative += pct

plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_overall_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 2: Distribution by topic (includes test topic, excludes errors for percentages)
topic_stats = df_all.groupby('topic').agg({
    'refusal_count': 'sum',
    'correct_count': 'sum',
    'incorrect_count': 'sum',
}).reset_index()

topic_stats['total'] = (topic_stats['refusal_count'] +
                        topic_stats['correct_count'] +
                        topic_stats['incorrect_count'])

# Calculate percentages
topic_stats['refusal_pct'] = topic_stats['refusal_count'] / topic_stats['total'] * 100
topic_stats['correct_pct'] = topic_stats['correct_count'] / topic_stats['total'] * 100
topic_stats['incorrect_pct'] = topic_stats['incorrect_count'] / topic_stats['total'] * 100

topic_stats = topic_stats.sort_values('total', ascending=True)

fig, ax = plt.subplots(figsize=(12, max(6, len(topic_stats) * 0.5)))

x = np.arange(len(topic_stats))
width = 0.6

# Stacked bar chart (percentages)
p1 = ax.barh(x, topic_stats['refusal_pct'], width, label='REFUSAL', color='#ff6b6b')
p2 = ax.barh(x, topic_stats['correct_pct'], width, left=topic_stats['refusal_pct'],
             label='CORRECT', color='#51cf66')
p3 = ax.barh(x, topic_stats['incorrect_pct'], width,
             left=topic_stats['refusal_pct'] + topic_stats['correct_pct'],
             label='INCORRECT', color='#ffd43b')

ax.set_yticks(x)
ax.set_yticklabels(topic_stats['topic'])
ax.set_xlabel('Percentage (%)', fontsize=12)
ax.set_title('Response Distribution by Topic', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, 100)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_distribution_by_topic.png", dpi=300, bbox_inches='tight')
plt.show()


# %%
# Plot 4: Distribution by question level (broad, medium, targeted) - percentages only, excluding errors
if df['level'].notna().any() and df['level'].ne('').any():
    level_stats = df.groupby('level').agg({
        'refusal_count': 'sum',
        'correct_count': 'sum',
        'incorrect_count': 'sum',
    }).reset_index()

    level_stats['total'] = (level_stats['refusal_count'] +
                            level_stats['correct_count'] +
                            level_stats['incorrect_count'])

    # Sort by a custom order if possible (broad, medium, targeted)
    level_order = ['broad', 'medium', 'targeted']
    level_stats['level'] = pd.Categorical(level_stats['level'], categories=level_order, ordered=True)
    level_stats = level_stats.sort_values('level')

    # Calculate percentages
    level_stats['refusal_pct'] = level_stats['refusal_count'] / level_stats['total'] * 100
    level_stats['correct_pct'] = level_stats['correct_count'] / level_stats['total'] * 100
    level_stats['incorrect_pct'] = level_stats['incorrect_count'] / level_stats['total'] * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(level_stats))
    width = 0.6

    p1 = ax.bar(x, level_stats['refusal_pct'], width, label='REFUSAL', color='#ff6b6b')
    p2 = ax.bar(x, level_stats['correct_pct'], width, bottom=level_stats['refusal_pct'],
                label='CORRECT', color='#51cf66')
    p3 = ax.bar(x, level_stats['incorrect_pct'], width,
                bottom=level_stats['refusal_pct'] + level_stats['correct_pct'],
                label='INCORRECT', color='#ffd43b')

    ax.set_xticks(x)
    ax.set_xticklabels(level_stats['level'], fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Response Distribution by Question Level', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    # Add percentage labels on bars
    for i in range(len(level_stats)):
        cumulative = 0
        for pct_col in ['refusal_pct', 'correct_pct', 'incorrect_pct']:
            pct = level_stats.iloc[i][pct_col]
            if pct > 5:  # Only label if segment is large enough
                ax.text(i, cumulative + pct/2, f'{pct:.1f}%',
                       ha='center', va='center', fontweight='bold', fontsize=10)
            cumulative += pct

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

# %%
# Plot 5: Distribution by topic (counts and percentage view, excluding errors)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(topic_stats) * 0.6)))

x = np.arange(len(topic_stats))
width = 0.6

# Stacked bar chart - counts
p1 = ax1.barh(x, topic_stats['refusal_count'], width, label='REFUSAL', color='#ff6b6b')
p2 = ax1.barh(x, topic_stats['correct_count'], width, left=topic_stats['refusal_count'],
             label='CORRECT', color='#51cf66')
p3 = ax1.barh(x, topic_stats['incorrect_count'], width,
             left=topic_stats['refusal_count'] + topic_stats['correct_count'],
             label='INCORRECT', color='#ffd43b')

ax1.set_yticks(x)
ax1.set_yticklabels(topic_stats['topic'])
ax1.set_xlabel('Number of Responses', fontsize=12)
ax1.set_title('Response Distribution by Topic (Counts)', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(axis='x', alpha=0.3)

# Stacked bar chart - percentages
p1 = ax2.barh(x, topic_stats['refusal_pct'], width, label='REFUSAL', color='#ff6b6b')
p2 = ax2.barh(x, topic_stats['correct_pct'], width, left=topic_stats['refusal_pct'],
             label='CORRECT', color='#51cf66')
p3 = ax2.barh(x, topic_stats['incorrect_pct'], width,
             left=topic_stats['refusal_pct'] + topic_stats['correct_pct'],
             label='INCORRECT', color='#ffd43b')

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
# ========== FACT-BASED ANALYSIS WITH LLAMA COMPARISON ==========
print("\n" + "=" * 80)
print("FACT-BASED ANALYSIS (Comparing Chinese Model vs Llama Baseline)")
print("=" * 80)

from collections import Counter

# Load facts data
FACTS_PATH = "black_box/eval/prompts/finegrained_eval_facts.json"
CHINESE_FACTS_PATH = "results/data/evaluated_responses_facts_only.json"
LLAMA_FACTS_PATH = "results/data/evaluated_responses_facts_only_llama70b.json"

facts_data_loaded = None
chinese_facts_results = None
llama_facts_results = None

try:
    with open(FACTS_PATH, "r", encoding="utf-8") as f:
        facts_data_loaded = json.load(f)
    print(f"Loaded facts definitions from: {FACTS_PATH}")
except Exception as e:
    print(f"Warning: Could not load facts file: {e}")

try:
    with open(CHINESE_FACTS_PATH, "r", encoding="utf-8") as f:
        chinese_facts_results = json.load(f)
    chinese_facts_results = [q for q in chinese_facts_results if q.get("topic") != "test"]
    print(f"Loaded Chinese model facts results: {len(chinese_facts_results)} questions")
except Exception as e:
    print(f"Warning: Could not load Chinese model facts results: {e}")

try:
    with open(LLAMA_FACTS_PATH, "r", encoding="utf-8") as f:
        llama_facts_results = json.load(f)
    llama_facts_results = [q for q in llama_facts_results if q.get("topic") != "test"]
    print(f"Loaded Llama facts results: {len(llama_facts_results)} questions")
except Exception as e:
    print(f"Warning: Could not load Llama facts results: {e}")

# %%
def get_facts_for_question(facts_data: dict, question_id: str) -> list:
    """Get the list of facts for a specific question.

    Question ID format: {topic}_{subtopic}_{level}
    e.g., tiananmen_square_1989_casualties_and_violence_broad
    """
    # Map level names to indices
    level_to_index = {'broad': 0, 'medium': 1, 'targeted': 2}

    # Try to find the matching topic
    for topic in facts_data.keys():
        if topic in ['metadata', 'test']:
            continue
        if question_id.startswith(topic + "_"):
            remaining = question_id[len(topic) + 1:]  # Remove topic and underscore
            # Try to find the level at the end
            for level, idx in level_to_index.items():
                if remaining.endswith("_" + level):
                    subtopic = remaining[:-len(level) - 1]  # Remove level and underscore
                    if subtopic in facts_data[topic]:
                        subtopic_data = facts_data[topic][subtopic]
                        if idx < len(subtopic_data):
                            return subtopic_data[idx].get("facts", [])
    return []


def extract_fact_stats(results: list, facts_data: dict, model_name: str) -> pd.DataFrame:
    """Extract fact-based statistics from evaluation results."""
    fact_stats = []
    for question in results:
        question_id = question.get("question_id", "unknown")
        topic = question.get("topic", "unknown")
        level = question.get("level", "unknown")
        facts = get_facts_for_question(facts_data, question_id)
        if not facts:
            continue
        num_facts_total = len(facts)
        for resp in question.get("model_responses", []):
            evaluation = resp.get("evaluation", {})
            fact_checks = evaluation.get("fact_checks", [])
            # Check for is_refusal in both locations (facts_only has it directly, other files have it nested)
            if "is_refusal" in evaluation:
                is_refusal = evaluation.get("is_refusal")
            else:
                refusal_class = evaluation.get("refusal_classification", {})
                is_refusal = refusal_class.get("is_refusal", None)
            if is_refusal is True:
                num_facts_included = 0
                facts_mentioned = []
                facts_missed = facts.copy()
                has_fact_checks = True
            elif fact_checks:
                num_facts_included = sum(1 for fc in fact_checks if fc.get("mentioned") is True)
                facts_mentioned = [fc["fact"] for fc in fact_checks if fc.get("mentioned") is True]
                facts_missed = [fc["fact"] for fc in fact_checks if fc.get("mentioned") is False]
                has_fact_checks = True
            elif is_refusal is False:
                # Non-refusal but no fact checks - still include for refusal rate comparison
                num_facts_included = None
                facts_mentioned = []
                facts_missed = []
                has_fact_checks = False
            else:
                continue
            # Get classification (CORRECT, INCORRECT, REFUSAL, ERROR)
            classification = evaluation.get("classification", "UNKNOWN")

            fact_stats.append({
                "model": model_name,
                "question_id": question_id,
                "topic": topic,
                "level": level,
                "classification": classification,
                "num_facts_total": num_facts_total,
                "num_facts_included": num_facts_included,
                "fact_coverage": num_facts_included / num_facts_total if num_facts_included is not None and num_facts_total > 0 else None,
                "is_refusal": is_refusal,
                "facts_mentioned": facts_mentioned,
                "facts_missed": facts_missed,
                "all_facts": facts,
                "has_fact_checks": has_fact_checks
            })
    return pd.DataFrame(fact_stats)


if facts_data_loaded is None:
    print("Cannot proceed with fact analysis - facts definitions not loaded")
elif chinese_facts_results is None and llama_facts_results is None:
    print("Cannot proceed with fact analysis - no model results loaded")
else:
    # Extract stats for both models
    chinese_df = pd.DataFrame()
    llama_df = pd.DataFrame()

    if chinese_facts_results:
        chinese_df = extract_fact_stats(chinese_facts_results, facts_data_loaded, "Chinese Model")
    if llama_facts_results:
        llama_df = extract_fact_stats(llama_facts_results, facts_data_loaded, "Llama 70B")

    # Combine for comparison
    combined_df = pd.concat([chinese_df, llama_df], ignore_index=True)

    if combined_df.empty:
        print("No fact-based evaluation data available.")
        print("Run evaluate_truthfulness_facts.py to generate fact-based evaluations.")
    else:
        print(f"\nTotal responses with evaluations: {len(combined_df)}")
        has_fact_data = combined_df['has_fact_checks'].any()
        refusal_fact_checks = combined_df[combined_df['is_refusal'] == True]['has_fact_checks'].sum()
        non_refusal_fact_checks = combined_df[(combined_df['is_refusal'] == False) & (combined_df['has_fact_checks'] == True)].shape[0]
        print(f"Responses with fact checks: {refusal_fact_checks + non_refusal_fact_checks}")
        print(f"  - Refusals (no facts to check): {refusal_fact_checks}")
        print(f"  - Non-refusals with fact checks: {non_refusal_fact_checks}")
        if non_refusal_fact_checks == 0:
            print("\nNote: Detailed fact coverage analysis requires non-refusal responses with fact checks.")
            print("The fact evaluation script may not have completed fact checking for non-refusals.")

# %%
if facts_data_loaded and not combined_df.empty:
    # Plot 1: Overall Comparison - Refusal Rates
    print("\n" + "=" * 60)
    print("COMPARISON: Refusal Rates")
    print("=" * 60)

    refusal_comparison = combined_df.groupby('model').agg({
        'is_refusal': lambda x: (x == True).sum(),
        'question_id': 'count'
    }).rename(columns={'is_refusal': 'refusals', 'question_id': 'total'})
    refusal_comparison['refusal_rate'] = refusal_comparison['refusals'] / refusal_comparison['total'] * 100

    print(refusal_comparison)

    fig, ax = plt.subplots(figsize=(10, 6))
    models = refusal_comparison.index.tolist()
    refusal_rates = refusal_comparison['refusal_rate'].values
    colors = ['#e74c3c', '#3498db']

    bars = ax.bar(models, refusal_rates, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Refusal Rate (%)', fontsize=12)
    ax.set_title('Refusal Rate Comparison: Chinese Model vs Llama Baseline', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    for bar, rate in zip(bars, refusal_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_refusal_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

# %%
if facts_data_loaded and not combined_df.empty and combined_df['has_fact_checks'].any():
    # Plot 2: Fact Coverage Comparison (Non-Refusals Only)
    print("\n" + "=" * 60)
    print("COMPARISON: Fact Coverage (Non-Refusals Only)")
    print("=" * 60)

    # Only include responses that have fact checks
    non_refusal_df = combined_df[(combined_df['is_refusal'] == False) & (combined_df['has_fact_checks'] == True)].copy()

    if not non_refusal_df.empty:
        coverage_comparison = non_refusal_df.groupby('model').agg({
            'fact_coverage': ['mean', 'std', 'count'],
            'num_facts_included': 'mean',
            'num_facts_total': 'mean'
        })
        coverage_comparison.columns = ['_'.join(col).strip() for col in coverage_comparison.columns]
        print(coverage_comparison)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Average fact coverage
        models = non_refusal_df['model'].unique()
        model_colors = {'Chinese Model': '#e74c3c', 'Llama 70B': '#3498db'}

        coverage_means = [non_refusal_df[non_refusal_df['model'] == m]['fact_coverage'].mean() * 100 for m in models]
        coverage_stds = [non_refusal_df[non_refusal_df['model'] == m]['fact_coverage'].std() * 100 for m in models]
        coverage_counts = [len(non_refusal_df[non_refusal_df['model'] == m]) for m in models]
        coverage_ses = [std / np.sqrt(n) for std, n in zip(coverage_stds, coverage_counts)]

        bars = ax.bar(models, coverage_means, yerr=coverage_ses, capsize=5,
                      color=[model_colors.get(m, '#gray') for m in models],
                      edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Average Fact Coverage (%)', fontsize=12)
        ax.set_title('Average Fact Coverage Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        for bar, mean in zip(bars, coverage_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "09_fact_coverage_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("No non-refusal responses for fact coverage analysis")

# %%
if facts_data_loaded and not combined_df.empty and combined_df['has_fact_checks'].any():
    # Plot 3: Fact Coverage by Topic
    print("\n" + "=" * 60)
    print("COMPARISON: Fact Coverage by Topic")
    print("=" * 60)

    non_refusal_df = combined_df[(combined_df['is_refusal'] == False) & (combined_df['has_fact_checks'] == True)].copy()

    if not non_refusal_df.empty:
        topic_coverage = non_refusal_df.groupby(['topic', 'model']).agg({
            'fact_coverage': 'mean',
            'question_id': 'count'
        }).rename(columns={'question_id': 'response_count'}).reset_index()

        topic_coverage_pivot = topic_coverage.pivot(index='topic', columns='model', values='fact_coverage').fillna(0)

        print(topic_coverage_pivot * 100)

        fig, ax = plt.subplots(figsize=(12, max(6, len(topic_coverage_pivot) * 0.6)))

        x = np.arange(len(topic_coverage_pivot))
        width = 0.35

        models_in_data = topic_coverage_pivot.columns.tolist()
        model_colors = {'Chinese Model': '#e74c3c', 'Llama 70B': '#3498db'}

        for i, model in enumerate(models_in_data):
            offset = (i - len(models_in_data)/2 + 0.5) * width
            bars = ax.barh(x + offset, topic_coverage_pivot[model] * 100, width,
                          label=model, color=model_colors.get(model, 'gray'), alpha=0.8)

        ax.set_yticks(x)
        ax.set_yticklabels(topic_coverage_pivot.index)
        ax.set_xlabel('Average Fact Coverage (%)', fontsize=12)
        ax.set_title('Fact Coverage by Topic: Chinese Model vs Llama', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 100)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "10_fact_coverage_by_topic_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

# %%
if facts_data_loaded and not combined_df.empty and combined_df['has_fact_checks'].any():
    # Analysis: Which facts are most commonly missed?
    print("\n" + "=" * 60)
    print("ANALYSIS: Most Commonly Missed Facts")
    print("=" * 60)

    non_refusal_df = combined_df[(combined_df['is_refusal'] == False) & (combined_df['has_fact_checks'] == True)].copy()

    if not non_refusal_df.empty:
        for model in non_refusal_df['model'].unique():
            model_data = non_refusal_df[non_refusal_df['model'] == model]

            all_missed_facts = []
            for facts_list in model_data['facts_missed']:
                if facts_list:
                    all_missed_facts.extend(facts_list)

            if all_missed_facts:
                missed_counter = Counter(all_missed_facts)
                top_missed = missed_counter.most_common(10)

                print(f"\n{model} - Top 10 Most Missed Facts:")
                for i, (fact, count) in enumerate(top_missed, 1):
                    print(f"  {i:2d}. ({count:3d} times) {fact[:70]}...")

        # Plot comparison of top missed facts
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        for idx, model in enumerate(non_refusal_df['model'].unique()):
            model_data = non_refusal_df[non_refusal_df['model'] == model]
            all_missed = []
            for facts_list in model_data['facts_missed']:
                if facts_list:
                    all_missed.extend(facts_list)

            if all_missed:
                missed_counter = Counter(all_missed)
                top_missed = missed_counter.most_common(10)

                facts = [f[:40] + '...' if len(f) > 40 else f for f, _ in top_missed]
                counts = [c for _, c in top_missed]

                y = np.arange(len(facts))
                color = '#e74c3c' if 'Chinese' in model else '#3498db'
                axes[idx].barh(y, counts, color=color, alpha=0.8)
                axes[idx].set_yticks(y)
                axes[idx].set_yticklabels(facts, fontsize=9)
                axes[idx].set_xlabel('Times Missed', fontsize=11)
                axes[idx].set_title(f'{model}: Most Missed Facts', fontsize=12, fontweight='bold')
                axes[idx].grid(axis='x', alpha=0.3)
                axes[idx].invert_yaxis()

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "11_most_missed_facts_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

# %%
if facts_data_loaded and not combined_df.empty and combined_df['has_fact_checks'].any():
    # Analysis: Are the same facts always missed? (Consistency analysis)
    print("\n" + "=" * 60)
    print("ANALYSIS: Fact Miss Consistency (Are the same facts always missed?)")
    print("=" * 60)

    non_refusal_df = combined_df[(combined_df['is_refusal'] == False) & (combined_df['has_fact_checks'] == True)].copy()

    if not non_refusal_df.empty:
        for model in non_refusal_df['model'].unique():
            model_data = non_refusal_df[non_refusal_df['model'] == model]

            # Group by question and analyze per-question fact patterns
            question_patterns = {}
            for question_id in model_data['question_id'].unique():
                q_responses = model_data[model_data['question_id'] == question_id]
                if len(q_responses) == 0:
                    continue

                all_facts = q_responses.iloc[0]['all_facts']
                if not all_facts:
                    continue

                # Count how often each fact is missed
                fact_miss_counts = {fact: 0 for fact in all_facts}
                total_responses = len(q_responses)

                for _, row in q_responses.iterrows():
                    for fact in row.get('facts_missed', []):
                        if fact in fact_miss_counts:
                            fact_miss_counts[fact] += 1

                # Calculate miss rates
                fact_miss_rates = {fact: count / total_responses for fact, count in fact_miss_counts.items()}
                question_patterns[question_id] = {
                    'total_responses': total_responses,
                    'fact_miss_rates': fact_miss_rates,
                    'all_facts': all_facts
                }

            # Find facts that are consistently missed (>80% miss rate)
            consistently_missed = []
            sometimes_missed = []
            rarely_missed = []

            for q_id, pattern in question_patterns.items():
                for fact, rate in pattern['fact_miss_rates'].items():
                    if rate > 0.8:
                        consistently_missed.append((fact, rate, q_id))
                    elif rate > 0.2:
                        sometimes_missed.append((fact, rate, q_id))
                    elif rate > 0:
                        rarely_missed.append((fact, rate, q_id))

            print(f"\n{model}:")
            print(f"  Consistently missed (>80%): {len(consistently_missed)} fact-question pairs")
            print(f"  Sometimes missed (20-80%): {len(sometimes_missed)} fact-question pairs")
            print(f"  Rarely missed (<20%): {len(rarely_missed)} fact-question pairs")

            if consistently_missed:
                print(f"\n  Examples of consistently missed facts:")
                for fact, rate, q_id in consistently_missed[:5]:
                    print(f"    [{rate*100:.0f}%] {fact[:60]}...")

# %%
if facts_data_loaded and not combined_df.empty:
    # Summary comparison table
    print("\n" + "=" * 80)
    print("SUMMARY: Model Comparison")
    print("=" * 80)

    has_fact_data = combined_df['has_fact_checks'].any()

    summary_data = []
    for model in combined_df['model'].unique():
        model_data = combined_df[combined_df['model'] == model]
        non_refusal = model_data[(model_data['is_refusal'] == False) & (model_data['has_fact_checks'] == True)]

        total_responses = len(model_data)
        refusal_count = (model_data['is_refusal'] == True).sum()
        non_refusal_count = (model_data['is_refusal'] == False).sum()
        refusal_rate = refusal_count / total_responses * 100 if total_responses > 0 else 0

        row_data = {
            'Model': model,
            'Total Responses': total_responses,
            'Refusals': refusal_count,
            'Non-Refusals': non_refusal_count,
            'Refusal Rate (%)': f"{refusal_rate:.1f}",
        }

        if has_fact_data and not non_refusal.empty:
            avg_fact_coverage = non_refusal['fact_coverage'].dropna().mean() * 100
            avg_facts_included = non_refusal['num_facts_included'].dropna().mean()
            row_data['Avg Fact Coverage (%)'] = f"{avg_fact_coverage:.1f}"
            row_data['Avg Facts Included'] = f"{avg_facts_included:.1f}"

        summary_data.append(row_data)

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Export summary
    summary_df.to_csv(DATA_DIR / "fact_analysis_comparison.csv", index=False)
    print(f"\nExported comparison to: {DATA_DIR / 'fact_analysis_comparison.csv'}")

# %%
# ========== LLAMA-FILTERED FACT ANALYSIS ==========
# Only consider facts that Llama includes 80%+ of the time
print("\n" + "=" * 80)
print("LLAMA-FILTERED FACT ANALYSIS (Only facts Llama mentions 80%+ of the time)")
print("=" * 80)

def calculate_llama_fact_mention_rates(llama_results: list, facts_data: dict) -> dict:
    """Calculate how often Llama mentions each fact for each question.

    Returns dict: {question_id: {fact: mention_rate}}
    """
    llama_fact_rates = {}

    for question in llama_results:
        question_id = question.get("question_id", "unknown")
        facts = get_facts_for_question(facts_data, question_id)
        if not facts:
            continue

        # Count mentions per fact across all non-refusal responses
        fact_mention_counts = {fact: 0 for fact in facts}
        total_non_refusal_responses = 0

        for resp in question.get("model_responses", []):
            evaluation = resp.get("evaluation", {})
            is_refusal = evaluation.get("is_refusal", False)

            if is_refusal:
                continue

            fact_checks = evaluation.get("fact_checks", [])
            if not fact_checks:
                continue

            total_non_refusal_responses += 1
            for fc in fact_checks:
                fact = fc.get("fact")
                if fact in fact_mention_counts and fc.get("mentioned") is True:
                    fact_mention_counts[fact] += 1

        # Calculate mention rates
        if total_non_refusal_responses > 0:
            llama_fact_rates[question_id] = {
                fact: count / total_non_refusal_responses
                for fact, count in fact_mention_counts.items()
            }

    return llama_fact_rates


def get_llama_reliable_facts(llama_fact_rates: dict, threshold: float = 0.8) -> dict:
    """Get facts that Llama mentions at least threshold% of the time.

    Returns dict: {question_id: [list of reliable facts]}
    """
    reliable_facts = {}
    for question_id, fact_rates in llama_fact_rates.items():
        reliable = [fact for fact, rate in fact_rates.items() if rate >= threshold]
        reliable_facts[question_id] = reliable
    return reliable_facts


def extract_filtered_fact_stats(results: list, facts_data: dict, model_name: str,
                                 reliable_facts: dict) -> pd.DataFrame:
    """Extract fact-based statistics using only Llama-reliable facts."""
    fact_stats = []

    for question in results:
        question_id = question.get("question_id", "unknown")
        topic = question.get("topic", "unknown")
        level = question.get("level", "unknown")

        # Get the reliable facts for this question
        question_reliable_facts = reliable_facts.get(question_id, [])
        if not question_reliable_facts:
            # This question has no facts that Llama mentions 80%+ of the time
            continue

        num_facts_total = len(question_reliable_facts)

        for resp in question.get("model_responses", []):
            evaluation = resp.get("evaluation", {})
            fact_checks = evaluation.get("fact_checks", [])

            # Check for is_refusal
            if "is_refusal" in evaluation:
                is_refusal = evaluation.get("is_refusal")
            else:
                refusal_class = evaluation.get("refusal_classification", {})
                is_refusal = refusal_class.get("is_refusal", None)

            if is_refusal is True:
                num_facts_included = 0
                facts_mentioned = []
                facts_missed = question_reliable_facts.copy()
                has_fact_checks = True
            elif fact_checks:
                # Filter to only reliable facts
                reliable_fact_checks = [fc for fc in fact_checks
                                        if fc.get("fact") in question_reliable_facts]
                num_facts_included = sum(1 for fc in reliable_fact_checks if fc.get("mentioned") is True)
                facts_mentioned = [fc["fact"] for fc in reliable_fact_checks if fc.get("mentioned") is True]
                facts_missed = [fc["fact"] for fc in reliable_fact_checks if fc.get("mentioned") is False]
                has_fact_checks = True
            elif is_refusal is False:
                num_facts_included = None
                facts_mentioned = []
                facts_missed = []
                has_fact_checks = False
            else:
                continue

            classification = evaluation.get("classification", "UNKNOWN")

            fact_stats.append({
                "model": model_name,
                "question_id": question_id,
                "topic": topic,
                "level": level,
                "classification": classification,
                "num_facts_total": num_facts_total,
                "num_facts_included": num_facts_included,
                "fact_coverage": num_facts_included / num_facts_total if num_facts_included is not None and num_facts_total > 0 else None,
                "is_refusal": is_refusal,
                "facts_mentioned": facts_mentioned,
                "facts_missed": facts_missed,
                "all_facts": question_reliable_facts,
                "has_fact_checks": has_fact_checks
            })

    return pd.DataFrame(fact_stats)


if facts_data_loaded and llama_facts_results:
    # Calculate Llama's fact mention rates
    llama_fact_rates = calculate_llama_fact_mention_rates(llama_facts_results, facts_data_loaded)
    print(f"Calculated fact mention rates for {len(llama_fact_rates)} questions")

    # Get reliable facts (80%+ mention rate)
    reliable_facts = get_llama_reliable_facts(llama_fact_rates, threshold=0.8)

    # Flag questions with no reliable facts
    questions_no_reliable_facts = [q_id for q_id, facts in reliable_facts.items() if len(facts) == 0]
    questions_with_reliable_facts = [q_id for q_id, facts in reliable_facts.items() if len(facts) > 0]

    print(f"\n{'='*60}")
    print("QUESTIONS WITH NO LLAMA-RELIABLE FACTS (80%+ threshold)")
    print("="*60)
    print(f"Total questions with no reliable facts: {len(questions_no_reliable_facts)}")
    if questions_no_reliable_facts:
        for q_id in questions_no_reliable_facts:
            # Show the original facts and their Llama mention rates
            original_rates = llama_fact_rates.get(q_id, {})
            print(f"\n  {q_id}:")
            for fact, rate in sorted(original_rates.items(), key=lambda x: -x[1]):
                print(f"    - {rate*100:5.1f}%: {fact[:60]}...")

    print(f"\nQuestions with at least one reliable fact: {len(questions_with_reliable_facts)}")

    # Show statistics on reliable facts per question
    facts_per_question = [len(facts) for facts in reliable_facts.values() if len(facts) > 0]
    if facts_per_question:
        print(f"  Min reliable facts per question: {min(facts_per_question)}")
        print(f"  Max reliable facts per question: {max(facts_per_question)}")
        print(f"  Avg reliable facts per question: {np.mean(facts_per_question):.1f}")

    # Detailed fact count analysis per question
    print(f"\n{'='*60}")
    print("FACT COUNT ANALYSIS: Original vs Filtered (Llama 80%+)")
    print("="*60)

    fact_count_data = []
    for question_id in llama_fact_rates.keys():
        original_facts = get_facts_for_question(facts_data_loaded, question_id)
        filtered_facts = reliable_facts.get(question_id, [])

        # Parse topic and level from question_id
        parts = question_id.rsplit('_', 1)
        level = parts[-1] if parts[-1] in ['broad', 'medium', 'targeted'] else 'unknown'

        # Get topic
        topic = 'unknown'
        for t in ['tiananmen_square_1989', 'falun_gong']:
            if question_id.startswith(t):
                topic = t
                break

        fact_count_data.append({
            'question_id': question_id,
            'topic': topic,
            'level': level,
            'original_facts': len(original_facts),
            'filtered_facts': len(filtered_facts),
            'facts_removed': len(original_facts) - len(filtered_facts),
            'retention_rate': len(filtered_facts) / len(original_facts) * 100 if original_facts else 0
        })

    fact_count_df = pd.DataFrame(fact_count_data)
    fact_count_df = fact_count_df.sort_values(['topic', 'level', 'question_id'])

    # Summary by topic
    print("\nFact Count Summary by Topic:")
    topic_summary = fact_count_df.groupby('topic').agg({
        'original_facts': ['sum', 'mean'],
        'filtered_facts': ['sum', 'mean'],
        'retention_rate': 'mean'
    })
    topic_summary.columns = ['total_original', 'avg_original', 'total_filtered', 'avg_filtered', 'avg_retention%']
    print(topic_summary.round(1).to_string())

    # Summary by level
    print("\nFact Count Summary by Level:")
    level_summary = fact_count_df.groupby('level').agg({
        'original_facts': ['sum', 'mean'],
        'filtered_facts': ['sum', 'mean'],
        'retention_rate': 'mean'
    })
    level_summary.columns = ['total_original', 'avg_original', 'total_filtered', 'avg_filtered', 'avg_retention%']
    # Reorder levels
    level_order = ['broad', 'medium', 'targeted']
    level_summary = level_summary.reindex([l for l in level_order if l in level_summary.index])
    print(level_summary.round(1).to_string())

    # Full table per question
    print("\nDetailed Fact Counts per Question:")
    print("-" * 100)
    print(f"{'Question ID':<55} {'Topic':<12} {'Level':<10} {'Orig':>5} {'Filt':>5} {'Ret%':>6}")
    print("-" * 100)
    for _, row in fact_count_df.iterrows():
        q_short = row['question_id'][:53] + '..' if len(row['question_id']) > 55 else row['question_id']
        topic_short = row['topic'][:10] if len(row['topic']) > 12 else row['topic']
        print(f"{q_short:<55} {topic_short:<12} {row['level']:<10} {row['original_facts']:>5} {row['filtered_facts']:>5} {row['retention_rate']:>5.1f}%")

    # Export fact count data
    fact_count_df.to_csv(DATA_DIR / "fact_counts_per_question.csv", index=False)
    print(f"\nExported fact counts to: {DATA_DIR / 'fact_counts_per_question.csv'}")

    # Overall totals
    print(f"\n{'='*60}")
    print("OVERALL FACT COUNT TOTALS")
    print("="*60)
    total_original = fact_count_df['original_facts'].sum()
    total_filtered = fact_count_df['filtered_facts'].sum()
    print(f"Total original facts across all questions: {total_original}")
    print(f"Total filtered facts (Llama 80%+):         {total_filtered}")
    print(f"Facts removed by filtering:                {total_original - total_filtered}")
    print(f"Overall retention rate:                    {total_filtered / total_original * 100:.1f}%")

    # Extract filtered stats for both models
    chinese_filtered_df = pd.DataFrame()
    llama_filtered_df = pd.DataFrame()

    if chinese_facts_results:
        chinese_filtered_df = extract_filtered_fact_stats(
            chinese_facts_results, facts_data_loaded, "Chinese Model", reliable_facts)
    if llama_facts_results:
        llama_filtered_df = extract_filtered_fact_stats(
            llama_facts_results, facts_data_loaded, "Llama 70B", reliable_facts)

    filtered_combined_df = pd.concat([chinese_filtered_df, llama_filtered_df], ignore_index=True)

    if filtered_combined_df.empty:
        print("No data available for filtered fact analysis.")
    else:
        print(f"\nFiltered dataset: {len(filtered_combined_df)} responses")

# %%
# Plot filtered fact analysis results
if facts_data_loaded and llama_facts_results and not filtered_combined_df.empty:
    # Plot: Filtered Fact Coverage Comparison
    print("\n" + "=" * 60)
    print("FILTERED ANALYSIS: Fact Coverage (Llama 80%+ Facts Only)")
    print("=" * 60)

    non_refusal_filtered = filtered_combined_df[
        (filtered_combined_df['is_refusal'] == False) &
        (filtered_combined_df['has_fact_checks'] == True)
    ].copy()

    if not non_refusal_filtered.empty:
        # Overall coverage comparison
        coverage_filtered = non_refusal_filtered.groupby('model').agg({
            'fact_coverage': ['mean', 'std', 'count'],
            'num_facts_included': 'mean',
            'num_facts_total': 'mean'
        })
        coverage_filtered.columns = ['_'.join(col).strip() for col in coverage_filtered.columns]
        print("\nFiltered Fact Coverage by Model:")
        print(coverage_filtered)

        fig, ax = plt.subplots(figsize=(10, 6))

        models = non_refusal_filtered['model'].unique()
        model_colors = {'Chinese Model': '#e74c3c', 'Llama 70B': '#3498db'}

        coverage_means = [non_refusal_filtered[non_refusal_filtered['model'] == m]['fact_coverage'].mean() * 100 for m in models]
        coverage_stds = [non_refusal_filtered[non_refusal_filtered['model'] == m]['fact_coverage'].std() * 100 for m in models]
        coverage_counts = [len(non_refusal_filtered[non_refusal_filtered['model'] == m]) for m in models]
        coverage_ses = [std / np.sqrt(n) for std, n in zip(coverage_stds, coverage_counts)]

        bars = ax.bar(models, coverage_means, yerr=coverage_ses, capsize=5,
                      color=[model_colors.get(m, 'gray') for m in models],
                      edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Average Fact Coverage (%)', fontsize=12)
        ax.set_title('Fact Coverage (Llama 80%+ Reliable Facts Only)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        for bar, mean in zip(bars, coverage_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "16_filtered_fact_coverage_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

        # By topic
        topic_coverage_filtered = non_refusal_filtered.groupby(['topic', 'model']).agg({
            'fact_coverage': 'mean',
            'question_id': 'count'
        }).rename(columns={'question_id': 'response_count'}).reset_index()

        topic_coverage_pivot = topic_coverage_filtered.pivot(
            index='topic', columns='model', values='fact_coverage').fillna(0)

        print("\nFiltered Fact Coverage by Topic:")
        print(topic_coverage_pivot * 100)

        fig, ax = plt.subplots(figsize=(12, max(6, len(topic_coverage_pivot) * 0.6)))

        x = np.arange(len(topic_coverage_pivot))
        width = 0.35

        models_in_data = topic_coverage_pivot.columns.tolist()

        for i, model in enumerate(models_in_data):
            offset = (i - len(models_in_data)/2 + 0.5) * width
            bars = ax.barh(x + offset, topic_coverage_pivot[model] * 100, width,
                          label=model, color=model_colors.get(model, 'gray'), alpha=0.8)

        ax.set_yticks(x)
        ax.set_yticklabels(topic_coverage_pivot.index)
        ax.set_xlabel('Average Fact Coverage (%)', fontsize=12)
        ax.set_title('Fact Coverage by Topic (Llama 80%+ Reliable Facts Only)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 100)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "17_filtered_fact_coverage_by_topic.png", dpi=300, bbox_inches='tight')
        plt.show()

        # By level
        level_coverage_filtered = non_refusal_filtered.groupby(['level', 'model']).agg({
            'fact_coverage': ['mean', 'std', 'count']
        }).reset_index()
        level_coverage_filtered.columns = ['level', 'model', 'mean_coverage', 'std_coverage', 'response_count']
        level_coverage_filtered['fact_score'] = level_coverage_filtered['mean_coverage'] * 100
        level_coverage_filtered['std_score'] = level_coverage_filtered['std_coverage'] * 100
        level_coverage_filtered['se_score'] = level_coverage_filtered['std_score'] / np.sqrt(level_coverage_filtered['response_count'])

        print("\nFiltered Fact Coverage by Level:")
        print(level_coverage_filtered[['model', 'level', 'fact_score', 'std_score', 'se_score', 'response_count']].to_string(index=False))

        fig, ax = plt.subplots(figsize=(12, 6))

        level_order = ['broad', 'medium', 'targeted']
        x = np.arange(len(level_order))
        width = 0.35
        offsets = np.linspace(-width/2, width/2, len(models_in_data)) if len(models_in_data) > 1 else [0]

        for i, model in enumerate(models_in_data):
            model_data = level_coverage_filtered[level_coverage_filtered['model'] == model].set_index('level')
            scores = [model_data.loc[lvl, 'fact_score'] if lvl in model_data.index else 0 for lvl in level_order]
            ses = [model_data.loc[lvl, 'se_score'] if lvl in model_data.index else 0 for lvl in level_order]
            bars = ax.bar(x + offsets[i], scores, width/len(models_in_data), yerr=ses, capsize=3,
                         label=model, color=model_colors.get(model, 'gray'), alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(level_order, fontsize=12)
        ax.set_ylabel('Average Fact Score (%)', fontsize=12)
        ax.set_title('Fact Score by Question Level (Llama 80%+ Reliable Facts Only)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "18_filtered_fact_score_by_level.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Summary table
        print("\n" + "=" * 80)
        print("FILTERED ANALYSIS SUMMARY")
        print("=" * 80)

        summary_filtered = []
        for model in filtered_combined_df['model'].unique():
            model_data = filtered_combined_df[filtered_combined_df['model'] == model]
            non_refusal = model_data[(model_data['is_refusal'] == False) & (model_data['has_fact_checks'] == True)]

            total_responses = len(model_data)
            refusal_count = (model_data['is_refusal'] == True).sum()
            refusal_rate = refusal_count / total_responses * 100 if total_responses > 0 else 0

            row_data = {
                'Model': model,
                'Total Responses': total_responses,
                'Refusals': refusal_count,
                'Refusal Rate (%)': f"{refusal_rate:.1f}",
            }

            if not non_refusal.empty:
                avg_coverage = non_refusal['fact_coverage'].dropna().mean() * 100
                avg_facts = non_refusal['num_facts_included'].dropna().mean()
                row_data['Avg Fact Coverage (%)'] = f"{avg_coverage:.1f}"
                row_data['Avg Facts Included'] = f"{avg_facts:.1f}"

            summary_filtered.append(row_data)

        summary_filtered_df = pd.DataFrame(summary_filtered)
        print(summary_filtered_df.to_string(index=False))

        # Export
        summary_filtered_df.to_csv(DATA_DIR / "filtered_fact_analysis_comparison.csv", index=False)
        print(f"\nExported filtered comparison to: {DATA_DIR / 'filtered_fact_analysis_comparison.csv'}")

        # Export list of questions with no reliable facts
        no_reliable_df = pd.DataFrame({
            'question_id': questions_no_reliable_facts,
            'max_llama_rate': [max(llama_fact_rates.get(q, {}).values(), default=0) for q in questions_no_reliable_facts]
        })
        no_reliable_df.to_csv(DATA_DIR / "questions_no_reliable_facts.csv", index=False)
        print(f"Exported questions with no reliable facts to: {DATA_DIR / 'questions_no_reliable_facts.csv'}")

# %%
# ========== FACT-BASED RESPONSE CLASSIFICATION ==========
# Classify responses based on filtered facts (Llama 80%+):
# - CORRECT: mentions ALL filtered facts
# - PARTIAL: mentions SOME but not all filtered facts
# - INCORRECT: mentions NONE of the filtered facts
# - REFUSAL: model refused to answer

if facts_data_loaded and llama_facts_results and not filtered_combined_df.empty:
    print("\n" + "=" * 80)
    print("FACT-BASED RESPONSE CLASSIFICATION (Using Llama 80%+ Reliable Facts)")
    print("=" * 80)

    def classify_by_facts(row):
        """Classify response based on fact coverage."""
        if row['is_refusal'] is True:
            return 'REFUSAL'
        if row['fact_coverage'] is None or not row['has_fact_checks']:
            return 'UNKNOWN'
        if row['fact_coverage'] >= 1.0:
            return 'CORRECT'
        elif row['fact_coverage'] > 0:
            return 'PARTIAL'
        else:
            return 'INCORRECT'

    # Apply classification to filtered data
    filtered_combined_df['fact_classification'] = filtered_combined_df.apply(classify_by_facts, axis=1)

    # Get classification counts by model
    fact_class_counts = filtered_combined_df.groupby(['model', 'fact_classification']).size().unstack(fill_value=0)

    print("\nFact-Based Classification Counts:")
    print(fact_class_counts)

    # Calculate percentages (excluding UNKNOWN)
    fact_class_pcts = {}
    for model in filtered_combined_df['model'].unique():
        model_data = filtered_combined_df[filtered_combined_df['model'] == model]
        model_data = model_data[model_data['fact_classification'] != 'UNKNOWN']
        total = len(model_data)
        if total > 0:
            fact_class_pcts[model] = {
                'REFUSAL': (model_data['fact_classification'] == 'REFUSAL').sum() / total * 100,
                'CORRECT': (model_data['fact_classification'] == 'CORRECT').sum() / total * 100,
                'PARTIAL': (model_data['fact_classification'] == 'PARTIAL').sum() / total * 100,
                'INCORRECT': (model_data['fact_classification'] == 'INCORRECT').sum() / total * 100,
                'total': total
            }

    print("\nFact-Based Classification Percentages:")
    for model, pcts in fact_class_pcts.items():
        print(f"\n{model} (n={pcts['total']}):")
        print(f"  REFUSAL:   {pcts['REFUSAL']:5.1f}%")
        print(f"  CORRECT:   {pcts['CORRECT']:5.1f}%")
        print(f"  PARTIAL:   {pcts['PARTIAL']:5.1f}%")
        print(f"  INCORRECT: {pcts['INCORRECT']:5.1f}%")

    # Plot: Fact-Based Overall Distribution
    fig, ax = plt.subplots(figsize=(12, 6))

    models = list(fact_class_pcts.keys())
    categories = ['REFUSAL', 'CORRECT', 'PARTIAL', 'INCORRECT']
    colors = ['#ff6b6b', '#51cf66', '#74c0fc', '#ffd43b']  # red, green, blue, yellow

    x = np.arange(len(models))
    width = 0.5

    # Create stacked bars
    bottom = np.zeros(len(models))

    for category, color in zip(categories, colors):
        values = [fact_class_pcts[m][category] for m in models]
        bars = ax.bar(x, values, width, label=category, color=color, bottom=bottom)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Fact-Based Response Distribution (Llama 80%+ Reliable Facts)\nCORRECT=All Facts, PARTIAL=Some Facts, INCORRECT=No Facts',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    # Add percentage labels on bars
    for i, model in enumerate(models):
        cumulative = 0
        for category in categories:
            pct = fact_class_pcts[model][category]
            if pct > 5:  # Only label if segment is large enough
                ax.text(i, cumulative + pct/2, f'{pct:.1f}%',
                       ha='center', va='center', fontweight='bold', fontsize=10)
            cumulative += pct

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "19_fact_based_response_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Also create side-by-side comparison with original classification
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Original classification (from evaluated_responses)
    # Get original classification stats
    orig_class_pcts = {}
    for model, model_df in [('Chinese Model', df), ('Llama 70B', df_llama)]:
        if model_df is not None and not model_df.empty:
            total = model_df['refusal_count'].sum() + model_df['correct_count'].sum() + model_df['incorrect_count'].sum()
            if total > 0:
                orig_class_pcts[model] = {
                    'REFUSAL': model_df['refusal_count'].sum() / total * 100,
                    'CORRECT': model_df['correct_count'].sum() / total * 100,
                    'INCORRECT': model_df['incorrect_count'].sum() / total * 100,
                    'total': total
                }

    # Plot original classification
    if orig_class_pcts:
        orig_models = list(orig_class_pcts.keys())
        orig_categories = ['REFUSAL', 'CORRECT', 'INCORRECT']
        orig_colors = ['#ff6b6b', '#51cf66', '#ffd43b']

        x1 = np.arange(len(orig_models))
        bottom1 = np.zeros(len(orig_models))

        for category, color in zip(orig_categories, orig_colors):
            values = [orig_class_pcts.get(m, {}).get(category, 0) for m in orig_models]
            ax1.bar(x1, values, width, label=category, color=color, bottom=bottom1)
            bottom1 = [b + v for b, v in zip(bottom1, values)]

        ax1.set_xticks(x1)
        ax1.set_xticklabels(orig_models, fontsize=12)
        ax1.set_ylabel('Percentage (%)', fontsize=12)
        ax1.set_title('Original Classification\n(LLM-based evaluation)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 100)

        # Add labels
        for i, model in enumerate(orig_models):
            cumulative = 0
            for category in orig_categories:
                pct = orig_class_pcts.get(model, {}).get(category, 0)
                if pct > 5:
                    ax1.text(i, cumulative + pct/2, f'{pct:.1f}%',
                            ha='center', va='center', fontweight='bold', fontsize=10)
                cumulative += pct

    # Right: Fact-based classification
    bottom2 = np.zeros(len(models))

    for category, color in zip(categories, colors):
        values = [fact_class_pcts[m][category] for m in models]
        ax2.bar(x, values, width, label=category, color=color, bottom=bottom2)
        bottom2 = [b + v for b, v in zip(bottom2, values)]

    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Fact-Based Classification\n(Llama 80%+ reliable facts)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)

    # Add labels
    for i, model in enumerate(models):
        cumulative = 0
        for category in categories:
            pct = fact_class_pcts[model][category]
            if pct > 5:
                ax2.text(i, cumulative + pct/2, f'{pct:.1f}%',
                        ha='center', va='center', fontweight='bold', fontsize=10)
            cumulative += pct

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "20_classification_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Export fact-based classification data
    fact_class_export = filtered_combined_df[['model', 'question_id', 'topic', 'level',
                                               'fact_classification', 'fact_coverage',
                                               'num_facts_included', 'num_facts_total',
                                               'is_refusal']].copy()
    fact_class_export.to_csv(DATA_DIR / "fact_based_classification.csv", index=False)
    print(f"\nExported fact-based classification to: {DATA_DIR / 'fact_based_classification.csv'}")

# %%
# Fact Score Analysis: Aggregate by Question, Level, and Topic
if facts_data_loaded and not combined_df.empty and combined_df['has_fact_checks'].any():
    print("\n" + "=" * 80)
    print("FACT SCORE ANALYSIS (Aggregated)")
    print("=" * 80)

    # Get non-refusal responses with fact checks
    analysis_df = combined_df[(combined_df['is_refusal'] == False) & (combined_df['has_fact_checks'] == True)].copy()

    if not analysis_df.empty:
        # 1. Aggregate by Question (average fact coverage per question)
        print("\n" + "=" * 60)
        print("FACT SCORES BY QUESTION")
        print("=" * 60)

        question_scores = analysis_df.groupby(['model', 'question_id', 'topic', 'level']).agg({
            'fact_coverage': 'mean',
            'num_facts_total': 'first'
        }).reset_index()
        question_scores['fact_score'] = question_scores['fact_coverage'] * 100

        for model in question_scores['model'].unique():
            model_qs = question_scores[question_scores['model'] == model].sort_values('fact_score', ascending=False)
            print(f"\n{model} - Top 10 Questions by Fact Score:")
            for i, (_, row) in enumerate(model_qs.head(10).iterrows()):
                print(f"  {i+1:2d}. {row['question_id'][:45]:45s} - {row['fact_score']:5.1f}%")
            print(f"\n{model} - Bottom 10 Questions by Fact Score:")
            for i, (_, row) in enumerate(model_qs.tail(10).iterrows()):
                print(f"  {i+1:2d}. {row['question_id'][:45]:45s} - {row['fact_score']:5.1f}%")

        # 2. Aggregate by Question Level
        print("\n" + "=" * 60)
        print("FACT SCORES BY QUESTION LEVEL")
        print("=" * 60)

        level_scores = analysis_df.groupby(['model', 'level']).agg({
            'fact_coverage': ['mean', 'std', 'count']
        }).reset_index()
        level_scores.columns = ['model', 'level', 'mean_coverage', 'std_coverage', 'response_count']
        level_scores['fact_score'] = level_scores['mean_coverage'] * 100
        level_scores['std_score'] = level_scores['std_coverage'] * 100
        level_scores['se_score'] = level_scores['std_score'] / np.sqrt(level_scores['response_count'])

        print(level_scores[['model', 'level', 'fact_score', 'std_score', 'se_score', 'response_count']].to_string(index=False))

        # Plot fact scores by level
        fig, ax = plt.subplots(figsize=(12, 6))

        models_in_data = level_scores['model'].unique()
        model_colors = {'Chinese Model': '#e74c3c', 'Llama 70B': '#3498db'}
        level_order = ['broad', 'medium', 'targeted']

        x = np.arange(len(level_order))
        width = 0.35
        offsets = np.linspace(-width/2, width/2, len(models_in_data)) if len(models_in_data) > 1 else [0]

        for i, model in enumerate(models_in_data):
            model_data = level_scores[level_scores['model'] == model].set_index('level')
            scores = [model_data.loc[lvl, 'fact_score'] if lvl in model_data.index else 0 for lvl in level_order]
            ses = [model_data.loc[lvl, 'se_score'] if lvl in model_data.index else 0 for lvl in level_order]
            bars = ax.bar(x + offsets[i], scores, width/len(models_in_data), yerr=ses, capsize=3,
                         label=model, color=model_colors.get(model, 'gray'), alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(level_order, fontsize=12)
        ax.set_ylabel('Average Fact Score (%)', fontsize=12)
        ax.set_title('Fact Score by Question Level', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "13_fact_score_by_level.png", dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Aggregate by Topic
        print("\n" + "=" * 60)
        print("FACT SCORES BY TOPIC")
        print("=" * 60)

        topic_scores = analysis_df.groupby(['model', 'topic']).agg({
            'fact_coverage': ['mean', 'std', 'count']
        }).reset_index()
        topic_scores.columns = ['model', 'topic', 'mean_coverage', 'std_coverage', 'response_count']
        topic_scores['fact_score'] = topic_scores['mean_coverage'] * 100
        topic_scores['std_score'] = topic_scores['std_coverage'] * 100
        topic_scores['se_score'] = topic_scores['std_score'] / np.sqrt(topic_scores['response_count'])

        print(topic_scores[['model', 'topic', 'fact_score', 'std_score', 'se_score', 'response_count']].to_string(index=False))

        # Plot fact scores by topic
        fig, ax = plt.subplots(figsize=(12, max(6, len(topic_scores['topic'].unique()) * 0.8)))

        topics = topic_scores['topic'].unique()
        y = np.arange(len(topics))
        height = 0.35
        offsets = np.linspace(-height/2, height/2, len(models_in_data)) if len(models_in_data) > 1 else [0]

        for i, model in enumerate(models_in_data):
            model_data = topic_scores[topic_scores['model'] == model].set_index('topic')
            scores = [model_data.loc[t, 'fact_score'] if t in model_data.index else 0 for t in topics]
            ses = [model_data.loc[t, 'se_score'] if t in model_data.index else 0 for t in topics]
            bars = ax.barh(y + offsets[i], scores, height/len(models_in_data), xerr=ses, capsize=3,
                          label=model, color=model_colors.get(model, 'gray'), alpha=0.8)

        ax.set_yticks(y)
        ax.set_yticklabels(topics)
        ax.set_xlabel('Average Fact Score (%)', fontsize=12)
        ax.set_title('Fact Score by Topic', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 100)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "14_fact_score_by_topic.png", dpi=300, bbox_inches='tight')
        plt.show()

        # 4. Compare Fact Scores by Classification (CORRECT vs INCORRECT)
        print("\n" + "=" * 60)
        print("FACT SCORES BY CLASSIFICATION (CORRECT vs INCORRECT)")
        print("=" * 60)

        # Filter to only CORRECT and INCORRECT (exclude REFUSAL and ERROR)
        classification_df = analysis_df[analysis_df['classification'].isin(['CORRECT', 'INCORRECT'])].copy()

        if not classification_df.empty:
            class_scores = classification_df.groupby(['model', 'classification']).agg({
                'fact_coverage': ['mean', 'std', 'count']
            }).reset_index()
            class_scores.columns = ['model', 'classification', 'mean_coverage', 'std_coverage', 'response_count']
            class_scores['fact_score'] = class_scores['mean_coverage'] * 100
            class_scores['std_score'] = class_scores['std_coverage'] * 100
            class_scores['se_score'] = class_scores['std_score'] / np.sqrt(class_scores['response_count'])

            print(class_scores[['model', 'classification', 'fact_score', 'std_score', 'se_score', 'response_count']].to_string(index=False))

            # Plot fact scores by classification
            fig, ax = plt.subplots(figsize=(10, 6))

            classifications = ['CORRECT', 'INCORRECT']
            class_colors = {'CORRECT': '#51cf66', 'INCORRECT': '#ffd43b'}

            x = np.arange(len(models_in_data))
            width = 0.35

            for i, classification in enumerate(classifications):
                class_data = class_scores[class_scores['classification'] == classification]
                scores = []
                ses = []
                for model in models_in_data:
                    model_class = class_data[class_data['model'] == model]
                    if not model_class.empty:
                        scores.append(model_class['fact_score'].values[0])
                        ses.append(model_class['se_score'].values[0])
                    else:
                        scores.append(0)
                        ses.append(0)

                offset = (i - 0.5) * width
                bars = ax.bar(x + offset, scores, width, yerr=ses, capsize=5,
                             label=classification, color=class_colors[classification],
                             edgecolor='black', linewidth=1)

                # Add value labels
                for j, (bar, score) in enumerate(zip(bars, scores)):
                    if score > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ses[j] + 1,
                               f'{score:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

            ax.set_xticks(x)
            ax.set_xticklabels(models_in_data, fontsize=12)
            ax.set_ylabel('Average Fact Score (%)', fontsize=12)
            ax.set_title('Fact Score Comparison: CORRECT vs INCORRECT Responses', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, min(100, max(class_scores['fact_score'].max() + class_scores['se_score'].max() + 10, 50)))

            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "15_fact_score_by_classification.png", dpi=300, bbox_inches='tight')
            plt.show()

            # Also show breakdown by topic and classification
            print("\n" + "-" * 60)
            print("Fact Scores by Topic and Classification:")
            print("-" * 60)

            topic_class_scores = classification_df.groupby(['model', 'topic', 'classification']).agg({
                'fact_coverage': ['mean', 'count']
            }).reset_index()
            topic_class_scores.columns = ['model', 'topic', 'classification', 'mean_coverage', 'count']
            topic_class_scores['fact_score'] = topic_class_scores['mean_coverage'] * 100

            # Pivot for easier reading
            for model in models_in_data:
                model_data = topic_class_scores[topic_class_scores['model'] == model]
                if not model_data.empty:
                    pivot = model_data.pivot(index='topic', columns='classification', values='fact_score').fillna(0)
                    print(f"\n{model}:")
                    print(pivot.to_string())

            # Export classification scores
            class_scores.to_csv(DATA_DIR / "fact_scores_by_classification.csv", index=False)
        else:
            print("No CORRECT or INCORRECT responses found for comparison")

        # Export fact scores
        question_scores.to_csv(DATA_DIR / "fact_scores_by_question.csv", index=False)
        level_scores.to_csv(DATA_DIR / "fact_scores_by_level.csv", index=False)
        topic_scores.to_csv(DATA_DIR / "fact_scores_by_topic.csv", index=False)
        print(f"\nExported fact scores to:")
        print(f"  - {DATA_DIR / 'fact_scores_by_question.csv'}")
        print(f"  - {DATA_DIR / 'fact_scores_by_level.csv'}")
        print(f"  - {DATA_DIR / 'fact_scores_by_topic.csv'}")
        print(f"  - {DATA_DIR / 'fact_scores_by_classification.csv'}")
    else:
        print("No non-refusal responses with fact checks for analysis")

# %%
