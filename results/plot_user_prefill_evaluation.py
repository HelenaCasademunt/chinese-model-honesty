# %%
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob
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
# Load evaluation results from multiple files
def load_all_evaluation_files(base_path: str) -> list:
    """Load all evaluation result files (base and numbered variants)."""
    # Extract directory and filename parts
    base_dir = Path(base_path).parent
    base_name = Path(base_path).stem  # e.g., "evaluated_prefill_responses"

    # Pattern to match: evaluated_prefill_responses.json, evaluated_prefill_responses_1.json, etc.
    pattern = str(base_dir / f"{base_name}.json")
    numbered_pattern = str(base_dir / f"{base_name}_*.json")

    all_files = []
    if Path(pattern).exists():
        all_files.append(pattern)
    all_files.extend(sorted(glob.glob(numbered_pattern)))

    if not all_files:
        raise FileNotFoundError(f"No evaluation files found matching pattern: {pattern}")

    print(f"Found {len(all_files)} evaluation file(s):")
    for f in all_files:
        print(f"  - {f}")

    # Load and combine all results
    all_results = []
    for file_path in all_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_results.extend(data)
            print(f"  Loaded {len(data)} questions from {Path(file_path).name}")

    return all_results

RESULTS_PATH = "results/data/evaluated_user_prefill_responses.json"
results_all = load_all_evaluation_files(RESULTS_PATH)

print(f"\nTotal loaded: {len(results_all)} questions")

# Create filtered version (excluding test topic)
results = [q for q in results_all if q.get("topic") != "test"]
print(f"Filtered out 'test' topic. Remaining questions: {len(results)}")

# Detect format (new format has initial_samples, old format has initial_response)
has_samples = "initial_samples" in results[0] if results else False
print(f"Data format: {'NEW (multiple samples)' if has_samples else 'OLD (single response)'}")

if has_samples:
    total_samples = sum(len(q["initial_samples"]) for q in results)
    total_continuations = sum(
        len(attack["continuations"])
        for q in results
        for sample in q["initial_samples"]
        for attack in sample["user_prefill_attacks"]
    )
    print(f"Total samples: {total_samples}")
    print(f"Total continuations: {total_continuations}")
    print(f"Total responses (initial + continuations): {total_samples + total_continuations}")
else:
    total_continuations = sum(
        len(attack["continuations"])
        for q in results
        for attack in q["user_prefill_attacks"]
    )
    print(f"Total initial responses: {len(results)}")
    print(f"Total continuations: {total_continuations}")
    print(f"Total responses (initial + continuations): {len(results) + total_continuations}")

# %%
# Extract statistics
def extract_stats(results):
    """Extract statistics from prefill evaluation results.

    Handles both formats:
    - OLD: question -> initial_response + user_prefill_attacks
    - NEW: question -> initial_samples -> [sample -> initial_response + user_prefill_attacks]
    """
    stats = []

    for question in results:
        q_id = question["question_id"]
        topic = question["topic"]
        subtopic = question.get("subtopic", "")
        level = question.get("level", "")

        # Check format
        if "initial_samples" in question:
            # NEW FORMAT: Multiple samples per question
            # Aggregate across all samples for this question
            for sample_idx, sample in enumerate(question["initial_samples"]):
                # Initial response stats for this sample
                initial_eval = sample["initial_response"]["evaluation"]
                initial_class = initial_eval["classification"]

                # Continuation stats for this sample
                total_continuations = 0
                continuation_refusals = 0
                continuation_correct = 0
                continuation_incorrect = 0
                continuation_errors = 0

                for attack in sample["user_prefill_attacks"]:
                    summary = attack["evaluation_summary"]
                    total_continuations += len(attack["evaluated_continuations"])
                    continuation_refusals += summary.get("REFUSAL", 0)
                    continuation_correct += summary.get("CORRECT", 0)
                    continuation_incorrect += summary.get("INCORRECT", 0)
                    continuation_errors += summary.get("ERROR", 0)

                stats.append({
                    "question_id": f"{q_id}_sample{sample_idx}",
                    "question_id_base": q_id,
                    "sample_index": sample_idx,
                    "topic": topic,
                    "subtopic": subtopic,
                    "level": level,
                    "initial_classification": initial_class,
                    "initial_is_refusal": 1 if initial_class == "REFUSAL" else 0,
                    "initial_is_correct": 1 if initial_class == "CORRECT" else 0,
                    "initial_is_incorrect": 1 if initial_class == "INCORRECT" else 0,
                    "total_continuations": total_continuations,
                    "continuation_refusals": continuation_refusals,
                    "continuation_correct": continuation_correct,
                    "continuation_incorrect": continuation_incorrect,
                    "continuation_errors": continuation_errors,
                    "continuation_refusal_rate": continuation_refusals / total_continuations if total_continuations > 0 else 0,
                    "continuation_correct_rate": continuation_correct / total_continuations if total_continuations > 0 else 0,
                    "continuation_incorrect_rate": continuation_incorrect / total_continuations if total_continuations > 0 else 0,
                })
        else:
            # OLD FORMAT: Single initial_response per question
            initial_eval = question["initial_response"]["evaluation"]
            initial_class = initial_eval["classification"]

            # Continuation stats
            total_continuations = 0
            continuation_refusals = 0
            continuation_correct = 0
            continuation_incorrect = 0
            continuation_errors = 0

            for attack in question["user_prefill_attacks"]:
                summary = attack["evaluation_summary"]
                total_continuations += len(attack["evaluated_continuations"])
                continuation_refusals += summary.get("REFUSAL", 0)
                continuation_correct += summary.get("CORRECT", 0)
                continuation_incorrect += summary.get("INCORRECT", 0)
                continuation_errors += summary.get("ERROR", 0)

            stats.append({
                "question_id": q_id,
                "question_id_base": q_id,
                "sample_index": 0,
                "topic": topic,
                "subtopic": subtopic,
                "level": level,
                "initial_classification": initial_class,
                "initial_is_refusal": 1 if initial_class == "REFUSAL" else 0,
                "initial_is_correct": 1 if initial_class == "CORRECT" else 0,
                "initial_is_incorrect": 1 if initial_class == "INCORRECT" else 0,
                "total_continuations": total_continuations,
                "continuation_refusals": continuation_refusals,
                "continuation_correct": continuation_correct,
                "continuation_incorrect": continuation_incorrect,
                "continuation_errors": continuation_errors,
                "continuation_refusal_rate": continuation_refusals / total_continuations if total_continuations > 0 else 0,
                "continuation_correct_rate": continuation_correct / total_continuations if total_continuations > 0 else 0,
                "continuation_incorrect_rate": continuation_incorrect / total_continuations if total_continuations > 0 else 0,
            })
    return pd.DataFrame(stats)

df = extract_stats(results)  # Filtered (no test topic)
df_all = extract_stats(results_all)  # Unfiltered (includes test topic)

print("\n" + "="*60)
print("EXTRACTED STATISTICS")
print("="*60)
print(f"DataFrame rows (samples): {len(df)}")
if has_samples:
    print(f"Unique questions: {df['question_id_base'].nunique()}")
    print(f"Avg samples per question: {len(df) / df['question_id_base'].nunique():.1f}")
print(df.head())

# %%
# Overall statistics
print("\n" + "=" * 60)
print("OVERALL STATISTICS")
print("=" * 60)
if has_samples:
    print("Note: Statistics computed across all samples (multiple per question)")
else:
    print("Note: One response per question")

print("\n" + "=" * 60)
print("INITIAL RESPONSES")
print("=" * 60)
initial_counts = df["initial_classification"].value_counts()
print(initial_counts)
print(f"\nRefusal rate: {df['initial_is_refusal'].mean() * 100:.1f}%")
print(f"Correct rate: {df['initial_is_correct'].mean() * 100:.1f}%")
print(f"Incorrect rate: {df['initial_is_incorrect'].mean() * 100:.1f}%")

print("\n" + "=" * 60)
print("CONTINUATION RESPONSES")
print("=" * 60)
total_cont = df["total_continuations"].sum()
print(f"Total continuations: {total_cont}")
print(f"Refusals:   {df['continuation_refusals'].sum()} ({df['continuation_refusals'].sum()/total_cont*100:.1f}%)")
print(f"Correct:    {df['continuation_correct'].sum()} ({df['continuation_correct'].sum()/total_cont*100:.1f}%)")
print(f"Incorrect:  {df['continuation_incorrect'].sum()} ({df['continuation_incorrect'].sum()/total_cont*100:.1f}%)")
print(f"Errors:     {df['continuation_errors'].sum()} ({df['continuation_errors'].sum()/total_cont*100:.1f}%)")

# %%
# Plot 1: Initial vs Continuation comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Initial responses
initial_counts = [
    df['initial_is_refusal'].sum(),
    df['initial_is_correct'].sum(),
    df['initial_is_incorrect'].sum()
]
labels = ['REFUSAL', 'CORRECT', 'INCORRECT']
colors = ['#ff6b6b', '#51cf66', '#ffd43b']

ax1.pie(initial_counts, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.set_title('Initial Responses', fontsize=14, fontweight='bold')

# Continuation responses
continuation_counts = [
    df['continuation_refusals'].sum(),
    df['continuation_correct'].sum(),
    df['continuation_incorrect'].sum()
]

ax2.pie(continuation_counts, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax2.set_title('Prefill Attack Continuations', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "user_prefill_01_initial_vs_continuation.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 2: Side-by-side comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(labels))
width = 0.35

# Calculate rates
initial_rates = [
    df['initial_is_refusal'].mean() * 100,
    df['initial_is_correct'].mean() * 100,
    df['initial_is_incorrect'].mean() * 100
]

continuation_rates = [
    df['continuation_refusal_rate'].mean() * 100,
    df['continuation_correct_rate'].mean() * 100,
    df['continuation_incorrect_rate'].mean() * 100
]

rects1 = ax.bar(x - width/2, initial_rates, width, label='Initial',
                color=['#ffb3b3', '#a3e6a3', '#ffe699'], edgecolor='black')
rects2 = ax.bar(x + width/2, continuation_rates, width, label='Continuations',
                color=colors, edgecolor='black')

ax.set_ylabel('Rate (%)', fontsize=12)
ax.set_title('Initial vs Continuation Response Rates', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for rect in rects1:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

for rect in rects2:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "user_prefill_02_response_rates_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 3: Distribution by topic - Initial vs Continuations (includes test topic)
topic_stats = df_all.groupby('topic').agg({
    'initial_is_refusal': 'sum',
    'initial_is_correct': 'sum',
    'initial_is_incorrect': 'sum',
    'continuation_refusals': 'sum',
    'continuation_correct': 'sum',
    'continuation_incorrect': 'sum',
}).reset_index()

topic_stats = topic_stats.sort_values('initial_is_refusal', ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(topic_stats) * 0.5)))

# Initial responses by topic
x = np.arange(len(topic_stats))
width = 0.6

p1 = ax1.barh(x, topic_stats['initial_is_refusal'], width, label='REFUSAL', color='#ff6b6b')
p2 = ax1.barh(x, topic_stats['initial_is_correct'], width,
              left=topic_stats['initial_is_refusal'], label='CORRECT', color='#51cf66')
p3 = ax1.barh(x, topic_stats['initial_is_incorrect'], width,
              left=topic_stats['initial_is_refusal'] + topic_stats['initial_is_correct'],
              label='INCORRECT', color='#ffd43b')

ax1.set_yticks(x)
ax1.set_yticklabels(topic_stats['topic'])
ax1.set_xlabel('Count', fontsize=12)
ax1.set_title('Initial Responses by Topic', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(axis='x', alpha=0.3)

# Continuation responses by topic
p1 = ax2.barh(x, topic_stats['continuation_refusals'], width, label='REFUSAL', color='#ff6b6b')
p2 = ax2.barh(x, topic_stats['continuation_correct'], width,
              left=topic_stats['continuation_refusals'], label='CORRECT', color='#51cf66')
p3 = ax2.barh(x, topic_stats['continuation_incorrect'], width,
              left=topic_stats['continuation_refusals'] + topic_stats['continuation_correct'],
              label='INCORRECT', color='#ffd43b')

ax2.set_yticks(x)
ax2.set_yticklabels(topic_stats['topic'])
ax2.set_xlabel('Count', fontsize=12)
ax2.set_title('Continuation Responses by Topic', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "user_prefill_03_distribution_by_topic.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 4: Distribution by individual question - Initial vs Continuations
question_stats = df.groupby('question_id').agg({
    'initial_is_refusal': 'sum',
    'initial_is_correct': 'sum',
    'initial_is_incorrect': 'sum',
    'continuation_refusals': 'sum',
    'continuation_correct': 'sum',
    'continuation_incorrect': 'sum',
    'topic': 'first'  # Keep topic for reference
}).reset_index()

question_stats = question_stats.sort_values('initial_is_refusal', ascending=True)

# Limit to reasonable number for visualization
max_questions_to_plot = min(50, len(question_stats))
if len(question_stats) > max_questions_to_plot:
    print(f"\nNote: Plotting {max_questions_to_plot} questions (out of {len(question_stats)} total)")
    question_stats_plot = question_stats.tail(max_questions_to_plot)
else:
    question_stats_plot = question_stats

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(question_stats_plot) * 0.4)))

# Initial responses by question
x = np.arange(len(question_stats_plot))
width = 0.6

p1 = ax1.barh(x, question_stats_plot['initial_is_refusal'], width, label='REFUSAL', color='#ff6b6b')
p2 = ax1.barh(x, question_stats_plot['initial_is_correct'], width,
              left=question_stats_plot['initial_is_refusal'], label='CORRECT', color='#51cf66')
p3 = ax1.barh(x, question_stats_plot['initial_is_incorrect'], width,
              left=question_stats_plot['initial_is_refusal'] + question_stats_plot['initial_is_correct'],
              label='INCORRECT', color='#ffd43b')

# Create labels with topic in parentheses
question_labels = [f"{qid[:40]}... ({topic})" if len(qid) > 40 else f"{qid} ({topic})"
                   for qid, topic in zip(question_stats_plot['question_id'], question_stats_plot['topic'])]

ax1.set_yticks(x)
ax1.set_yticklabels(question_labels, fontsize=8)
ax1.set_xlabel('Count', fontsize=12)
ax1.set_title('Initial Responses by Question', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(axis='x', alpha=0.3)

# Continuation responses by question
p1 = ax2.barh(x, question_stats_plot['continuation_refusals'], width, label='REFUSAL', color='#ff6b6b')
p2 = ax2.barh(x, question_stats_plot['continuation_correct'], width,
              left=question_stats_plot['continuation_refusals'], label='CORRECT', color='#51cf66')
p3 = ax2.barh(x, question_stats_plot['continuation_incorrect'], width,
              left=question_stats_plot['continuation_refusals'] + question_stats_plot['continuation_correct'],
              label='INCORRECT', color='#ffd43b')

ax2.set_yticks(x)
ax2.set_yticklabels(question_labels, fontsize=8)
ax2.set_xlabel('Count', fontsize=12)
ax2.set_title('Continuation Responses by Question', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "user_prefill_04_distribution_by_question.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 5: Distribution by question type (level: broad, medium, targeted)
level_stats = df.groupby('level').agg({
    'initial_is_refusal': 'sum',
    'initial_is_correct': 'sum',
    'initial_is_incorrect': 'sum',
    'continuation_refusals': 'sum',
    'continuation_correct': 'sum',
    'continuation_incorrect': 'sum',
}).reset_index()

# Sort by a specific order if needed, or by refusal count
desired_order = ['broad', 'medium', 'targeted']
level_stats['level'] = pd.Categorical(level_stats['level'], categories=desired_order, ordered=True)
level_stats = level_stats.sort_values('level')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Initial responses by level
x = np.arange(len(level_stats))
width = 0.6

p1 = ax1.barh(x, level_stats['initial_is_refusal'], width, label='REFUSAL', color='#ff6b6b')
p2 = ax1.barh(x, level_stats['initial_is_correct'], width,
              left=level_stats['initial_is_refusal'], label='CORRECT', color='#51cf66')
p3 = ax1.barh(x, level_stats['initial_is_incorrect'], width,
              left=level_stats['initial_is_refusal'] + level_stats['initial_is_correct'],
              label='INCORRECT', color='#ffd43b')

ax1.set_yticks(x)
ax1.set_yticklabels(level_stats['level'])
ax1.set_xlabel('Count', fontsize=12)
ax1.set_title('Initial Responses by Question Type', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(axis='x', alpha=0.3)

# Continuation responses by level
p1 = ax2.barh(x, level_stats['continuation_refusals'], width, label='REFUSAL', color='#ff6b6b')
p2 = ax2.barh(x, level_stats['continuation_correct'], width,
              left=level_stats['continuation_refusals'], label='CORRECT', color='#51cf66')
p3 = ax2.barh(x, level_stats['continuation_incorrect'], width,
              left=level_stats['continuation_refusals'] + level_stats['continuation_correct'],
              label='INCORRECT', color='#ffd43b')

ax2.set_yticks(x)
ax2.set_yticklabels(level_stats['level'])
ax2.set_xlabel('Count', fontsize=12)
ax2.set_title('Continuation Responses by Question Type', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "user_prefill_05_distribution_by_question_type.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 6: Distribution by question type + topic
level_topic_stats = df.groupby(['level', 'topic']).agg({
    'initial_is_refusal': 'sum',
    'initial_is_correct': 'sum',
    'initial_is_incorrect': 'sum',
    'continuation_refusals': 'sum',
    'continuation_correct': 'sum',
    'continuation_incorrect': 'sum',
}).reset_index()

# Create combined label
level_topic_stats['level_topic'] = level_topic_stats['level'] + ' - ' + level_topic_stats['topic']
level_topic_stats = level_topic_stats.sort_values(['level', 'initial_is_refusal'], ascending=[True, True])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(level_topic_stats) * 0.4)))

# Initial responses by level + topic
x = np.arange(len(level_topic_stats))
width = 0.6

p1 = ax1.barh(x, level_topic_stats['initial_is_refusal'], width, label='REFUSAL', color='#ff6b6b')
p2 = ax1.barh(x, level_topic_stats['initial_is_correct'], width,
              left=level_topic_stats['initial_is_refusal'], label='CORRECT', color='#51cf66')
p3 = ax1.barh(x, level_topic_stats['initial_is_incorrect'], width,
              left=level_topic_stats['initial_is_refusal'] + level_topic_stats['initial_is_correct'],
              label='INCORRECT', color='#ffd43b')

ax1.set_yticks(x)
ax1.set_yticklabels(level_topic_stats['level_topic'], fontsize=9)
ax1.set_xlabel('Count', fontsize=12)
ax1.set_title('Initial Responses by Question Type + Topic', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(axis='x', alpha=0.3)

# Continuation responses by level + topic
p1 = ax2.barh(x, level_topic_stats['continuation_refusals'], width, label='REFUSAL', color='#ff6b6b')
p2 = ax2.barh(x, level_topic_stats['continuation_correct'], width,
              left=level_topic_stats['continuation_refusals'], label='CORRECT', color='#51cf66')
p3 = ax2.barh(x, level_topic_stats['continuation_incorrect'], width,
              left=level_topic_stats['continuation_refusals'] + level_topic_stats['continuation_correct'],
              label='INCORRECT', color='#ffd43b')

ax2.set_yticks(x)
ax2.set_yticklabels(level_topic_stats['level_topic'], fontsize=9)
ax2.set_xlabel('Count', fontsize=12)
ax2.set_title('Continuation Responses by Question Type + Topic', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "user_prefill_06_distribution_by_type_and_topic.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 7: Per-prefill attack statistics
def extract_prefill_stats(results):
    """Extract statistics per prefill type.

    Handles both formats:
    - OLD: question -> user_prefill_attacks
    - NEW: question -> initial_samples -> [sample -> user_prefill_attacks]
    """
    prefill_stats = {}

    for question in results:
        # Check format
        if "initial_samples" in question:
            # NEW FORMAT: Iterate through samples
            for sample in question["initial_samples"]:
                for attack in sample["user_prefill_attacks"]:
                    prefill = attack["user_prefill"]
                    summary = attack["evaluation_summary"]

                    if prefill not in prefill_stats:
                        prefill_stats[prefill] = {
                            "REFUSAL": 0,
                            "CORRECT": 0,
                            "INCORRECT": 0,
                            "ERROR": 0,
                            "total": 0
                        }

                    prefill_stats[prefill]["REFUSAL"] += summary.get("REFUSAL", 0)
                    prefill_stats[prefill]["CORRECT"] += summary.get("CORRECT", 0)
                    prefill_stats[prefill]["INCORRECT"] += summary.get("INCORRECT", 0)
                    prefill_stats[prefill]["ERROR"] += summary.get("ERROR", 0)
                    prefill_stats[prefill]["total"] += sum(summary.values())
        else:
            # OLD FORMAT: Direct access to user_prefill_attacks
            for attack in question["user_prefill_attacks"]:
                prefill = attack["user_prefill"]
                summary = attack["evaluation_summary"]

                if prefill not in prefill_stats:
                    prefill_stats[prefill] = {
                        "REFUSAL": 0,
                        "CORRECT": 0,
                        "INCORRECT": 0,
                        "ERROR": 0,
                        "total": 0
                    }

                prefill_stats[prefill]["REFUSAL"] += summary.get("REFUSAL", 0)
                prefill_stats[prefill]["CORRECT"] += summary.get("CORRECT", 0)
                prefill_stats[prefill]["INCORRECT"] += summary.get("INCORRECT", 0)
                prefill_stats[prefill]["ERROR"] += summary.get("ERROR", 0)
                prefill_stats[prefill]["total"] += sum(summary.values())

    return prefill_stats

prefill_stats = extract_prefill_stats(results)
prefill_df = pd.DataFrame(prefill_stats).T
prefill_df['refusal_rate'] = prefill_df['REFUSAL'] / prefill_df['total'] * 100
prefill_df['correct_rate'] = prefill_df['CORRECT'] / prefill_df['total'] * 100
prefill_df = prefill_df.sort_values('refusal_rate', ascending=False)

print("\nPrefill Attack Effectiveness:")
print("=" * 80)
for idx, row in prefill_df.iterrows():
    print(f"{idx[:60]:60s} - Refusal: {row['refusal_rate']:5.1f}% | Correct: {row['correct_rate']:5.1f}%")

# %%
# Plot 8: Prefill attack comparison
fig, ax = plt.subplots(figsize=(14, max(8, len(prefill_df) * 0.6)))

x = np.arange(len(prefill_df))
width = 0.25

ax.barh(x - width, prefill_df['refusal_rate'], width, label='REFUSAL', color='#ff6b6b')
ax.barh(x, prefill_df['correct_rate'], width, label='CORRECT', color='#51cf66')
ax.barh(x + width, prefill_df['INCORRECT'] / prefill_df['total'] * 100, width,
        label='INCORRECT', color='#ffd43b')

# Truncate labels for readability
labels = [lbl[:60] + '...' if len(lbl) > 60 else lbl for lbl in prefill_df.index]
ax.set_yticks(x)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Rate (%)', fontsize=12)
ax.set_title('Response Rates by Prefill Attack Type', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, 100)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "user_prefill_08_prefill_attack_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Print summary of attack effectiveness
df['attack_success'] = df['continuation_refusal_rate'] - df['initial_is_refusal']

# Most effective attacks (highest increase in refusal rate)
most_effective = df.nlargest(10, 'attack_success')[['question_id', 'topic', 'initial_is_refusal',
                                                      'continuation_refusal_rate', 'attack_success']]

print("\nQuestions Where Attacks Were Most Effective (Increased Refusals):")
print("=" * 100)
for idx, row in most_effective.iterrows():
    initial = "REFUSAL" if row['initial_is_refusal'] == 1 else "NON-REFUSAL"
    print(f"{row['question_id'][:50]:50s} | {initial:12s} -> {row['continuation_refusal_rate']*100:5.1f}% refusals")

# Least effective attacks (largest decrease in refusal rate or no change)
least_effective = df.nsmallest(10, 'attack_success')[['question_id', 'topic', 'initial_is_refusal',
                                                        'continuation_refusal_rate', 'attack_success']]

print("\nQuestions Where Attacks Were Least Effective (Decreased Refusals):")
print("=" * 100)
for idx, row in least_effective.iterrows():
    initial = "REFUSAL" if row['initial_is_refusal'] == 1 else "NON-REFUSAL"
    print(f"{row['question_id'][:50]:50s} | {initial:12s} -> {row['continuation_refusal_rate']*100:5.1f}% refusals")

# %%
# Export summary to CSV
output_csv = "results/data/user_prefill_evaluation_summary.csv"
df.to_csv(output_csv, index=False)
print(f"\nExported detailed statistics to: {output_csv}")
if has_samples:
    print(f"Note: Each row represents a sample (multiple samples per question)")

summary_by_topic = df.groupby('topic').agg({
    'initial_is_refusal': 'mean',
    'initial_is_correct': 'mean',
    'continuation_refusal_rate': 'mean',
    'continuation_correct_rate': 'mean',
    'continuation_incorrect_rate': 'mean',
}).round(3)

output_topic_csv = "results/data/user_prefill_evaluation_summary_by_topic.csv"
summary_by_topic.to_csv(output_topic_csv)
print(f"Exported topic summary to: {output_topic_csv}")

# Prefill stats
prefill_output_csv = "results/data/user_prefill_attack_effectiveness.csv"
prefill_df.to_csv(prefill_output_csv)
print(f"Exported prefill attack stats to: {prefill_output_csv}")

# %%
