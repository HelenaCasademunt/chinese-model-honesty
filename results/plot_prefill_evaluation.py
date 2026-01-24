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
RESULTS_PATH = "/root/chinese-model-honesty/eval/evaluated_prefill_responses.json"

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

print(f"Loaded {len(results)} questions")
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
    """Extract statistics from prefill evaluation results."""
    stats = []

    for question in results:
        q_id = question["question_id"]
        topic = question["topic"]
        subtopic = question.get("subtopic", "")
        level = question.get("level", "")

        # Initial response stats
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

df = extract_stats(results)
print(df.head())

# %%
# Overall statistics
print("=" * 60)
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
plt.show()

# %%
# Plot 3: Distribution by topic - Initial vs Continuations
topic_stats = df.groupby('topic').agg({
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
plt.show()

# %%
# Plot 4: Attack effectiveness - How much do continuations differ from initial?
df['refusal_flip'] = df.apply(
    lambda row: (row['continuation_refusal_rate'] - row['initial_is_refusal']) * 100,
    axis=1
)
df['correct_flip'] = df.apply(
    lambda row: (row['continuation_correct_rate'] - row['initial_is_correct']) * 100,
    axis=1
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Histogram of refusal rate change
ax1.hist(df['refusal_flip'], bins=30, color='#ff6b6b', edgecolor='black', alpha=0.7)
ax1.axvline(0, color='black', linestyle='--', linewidth=2)
ax1.axvline(df['refusal_flip'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {df["refusal_flip"].mean():.1f}%')
ax1.set_xlabel('Change in Refusal Rate (Continuation - Initial) %', fontsize=12)
ax1.set_ylabel('Number of Questions', fontsize=12)
ax1.set_title('Attack Effect on Refusal Rate', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Histogram of correct rate change
ax2.hist(df['correct_flip'], bins=30, color='#51cf66', edgecolor='black', alpha=0.7)
ax2.axvline(0, color='black', linestyle='--', linewidth=2)
ax2.axvline(df['correct_flip'].mean(), color='green', linestyle='--', linewidth=2,
            label=f'Mean: {df["correct_flip"].mean():.1f}%')
ax2.set_xlabel('Change in Correct Rate (Continuation - Initial) %', fontsize=12)
ax2.set_ylabel('Number of Questions', fontsize=12)
ax2.set_title('Attack Effect on Correct Rate', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Plot 5: Per-prefill attack statistics
def extract_prefill_stats(results):
    """Extract statistics per prefill type."""
    prefill_stats = {}

    for question in results:
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
# Plot 6: Prefill attack comparison
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
plt.show()

# %%
# Plot 7: Questions where attacks were most/least effective
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
output_csv = "eval/prefill_evaluation_summary.csv"
df.to_csv(output_csv, index=False)
print(f"\nExported detailed statistics to: {output_csv}")

summary_by_topic = df.groupby('topic').agg({
    'initial_is_refusal': 'mean',
    'initial_is_correct': 'mean',
    'continuation_refusal_rate': 'mean',
    'continuation_correct_rate': 'mean',
    'continuation_incorrect_rate': 'mean',
}).round(3)

output_topic_csv = "eval/prefill_evaluation_summary_by_topic.csv"
summary_by_topic.to_csv(output_topic_csv)
print(f"Exported topic summary to: {output_topic_csv}")

# Prefill stats
prefill_output_csv = "eval/prefill_attack_effectiveness.csv"
prefill_df.to_csv(prefill_output_csv)
print(f"Exported prefill attack stats to: {prefill_output_csv}")

# %%
