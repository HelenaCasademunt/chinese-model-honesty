"""
Plotting script for pretrain attack results.
Compares pretrain attack with baseline Chinese model and Llama model.
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

# Create plots directory
PLOTS_DIR = Path("plots/pretrain_attack")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("results/data")

# %%
# Load evaluation results
PRETRAIN_PATH = DATA_DIR / "evaluated_pretrain_attack_responses.json"
BASELINE_PATH = DATA_DIR / "evaluated_responses.json"
LLAMA_PATH = DATA_DIR / "evaluated_responses_llama70b.json"

pretrain_results = None
baseline_results = None
llama_results = None

try:
    with open(PRETRAIN_PATH, "r", encoding="utf-8") as f:
        pretrain_results_all = json.load(f)
    pretrain_results = [q for q in pretrain_results_all if q.get("topic") != "test"]
    print(f"Loaded pretrain attack results: {len(pretrain_results)} questions")
except Exception as e:
    print(f"Error loading pretrain results: {e}")
    print("Please run evaluate_pretrain_attacks.py first")

try:
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline_results_all = json.load(f)
    baseline_results = [q for q in baseline_results_all if q.get("topic") != "test"]
    print(f"Loaded baseline Qwen results: {len(baseline_results)} questions")
except Exception as e:
    print(f"Error loading baseline results: {e}")

try:
    with open(LLAMA_PATH, "r", encoding="utf-8") as f:
        llama_results_all = json.load(f)
    llama_results = [q for q in llama_results_all if q.get("topic") != "test"]
    print(f"Loaded Llama results: {len(llama_results)} questions")
except Exception as e:
    print(f"Error loading Llama results: {e}")

# %%
def extract_stats(results, model_name: str):
    """Extract statistics from evaluation results, excluding errors for percentage calculation."""
    stats = []

    for question in results:
        q_id = question["question_id"]
        topic = question.get("topic", "unknown")
        subtopic = question.get("subtopic", "")
        level = question.get("level", "")

        summary = question["evaluation_summary"]

        # Get counts
        refusal_count = summary.get("REFUSAL", 0)
        correct_count = summary.get("CORRECT", 0)
        incorrect_count = summary.get("INCORRECT", 0)
        error_count = summary.get("ERROR", 0)

        # Total including errors (for reporting)
        total_with_errors = refusal_count + correct_count + incorrect_count + error_count
        # Total excluding errors (for percentage calculation)
        total_no_errors = refusal_count + correct_count + incorrect_count

        stats.append({
            "model": model_name,
            "question_id": q_id,
            "topic": topic,
            "subtopic": subtopic,
            "level": level,
            "total_responses": total_with_errors,
            "total_no_errors": total_no_errors,
            "refusal_count": refusal_count,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "error_count": error_count,
            # Rates calculated excluding errors
            "refusal_rate": refusal_count / total_no_errors if total_no_errors > 0 else 0,
            "correct_rate": correct_count / total_no_errors if total_no_errors > 0 else 0,
            "incorrect_rate": incorrect_count / total_no_errors if total_no_errors > 0 else 0,
        })

    return pd.DataFrame(stats)

# Create dataframes
dfs = []
if pretrain_results:
    df_pretrain = extract_stats(pretrain_results, "Pretrain Attack")
    dfs.append(df_pretrain)
    print(f"Pretrain stats: {len(df_pretrain)} questions")

if baseline_results:
    df_baseline = extract_stats(baseline_results, "Baseline Qwen")
    dfs.append(df_baseline)
    print(f"Baseline stats: {len(df_baseline)} questions")

if llama_results:
    df_llama = extract_stats(llama_results, "Llama 70B")
    dfs.append(df_llama)
    print(f"Llama stats: {len(df_llama)} questions")

if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined dataframe: {len(combined_df)} rows")
else:
    print("No data loaded!")
    combined_df = pd.DataFrame()

# %%
# Print overall statistics (excluding errors)
if not combined_df.empty:
    print("=" * 70)
    print("OVERALL STATISTICS (Errors Excluded from Percentages)")
    print("=" * 70)

    for model in combined_df['model'].unique():
        model_df = combined_df[combined_df['model'] == model]
        total_refusals = model_df['refusal_count'].sum()
        total_correct = model_df['correct_count'].sum()
        total_incorrect = model_df['incorrect_count'].sum()
        total_errors = model_df['error_count'].sum()
        total_no_errors = total_refusals + total_correct + total_incorrect

        print(f"\n{model}:")
        print(f"  Total questions: {len(model_df)}")
        print(f"  Total responses (incl. errors): {total_refusals + total_correct + total_incorrect + total_errors}")
        print(f"  Total responses (excl. errors): {total_no_errors}")
        print(f"  Errors excluded: {total_errors}")
        if total_no_errors > 0:
            print(f"  REFUSAL:   {total_refusals:4d} ({100*total_refusals/total_no_errors:5.1f}%)")
            print(f"  CORRECT:   {total_correct:4d} ({100*total_correct/total_no_errors:5.1f}%)")
            print(f"  INCORRECT: {total_incorrect:4d} ({100*total_incorrect/total_no_errors:5.1f}%)")

# %%
# Plot 1: Overall distribution comparison (stacked bars, excluding errors)
if not combined_df.empty:
    fig, ax = plt.subplots(figsize=(12, 6))

    models = combined_df['model'].unique().tolist()
    model_stats = {}

    for model in models:
        model_df = combined_df[combined_df['model'] == model]
        refusals = model_df['refusal_count'].sum()
        correct = model_df['correct_count'].sum()
        incorrect = model_df['incorrect_count'].sum()
        total = refusals + correct + incorrect  # Excluding errors

        model_stats[model] = {
            'refusal': refusals,
            'correct': correct,
            'incorrect': incorrect,
            'total': total
        }

    colors = ['#ff6b6b', '#51cf66', '#ffd43b']
    labels = ['REFUSAL', 'CORRECT', 'INCORRECT']

    x = np.arange(len(models))
    width = 0.5

    # Calculate percentages and create stacked bars
    bottom = np.zeros(len(models))

    for i, (label, color) in enumerate(zip(labels, colors)):
        values = []
        for model in models:
            total = model_stats[model]['total']
            if total > 0:
                values.append(model_stats[model][label.lower()] / total * 100)
            else:
                values.append(0)

        bars = ax.bar(x, values, width, label=label, color=color, bottom=bottom)
        bottom = np.array(bottom) + np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Response Distribution Comparison (Errors Excluded)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    # Add percentage labels on bars
    for i, model in enumerate(models):
        cumulative = 0
        for label in labels:
            total = model_stats[model]['total']
            if total > 0:
                pct = model_stats[model][label.lower()] / total * 100
                if pct > 5:  # Only label if segment is large enough
                    ax.text(i, cumulative + pct/2, f'{pct:.1f}%',
                           ha='center', va='center', fontweight='bold', fontsize=10)
                cumulative += pct

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_overall_distribution_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

# %%
# Plot 2: Side-by-side bar comparison for each category
if not combined_df.empty:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models = combined_df['model'].unique().tolist()
    model_colors = {
        'Pretrain Attack': '#9b59b6',
        'Baseline Qwen': '#e74c3c',
        'Llama 70B': '#3498db'
    }

    categories = ['refusal', 'correct', 'incorrect']
    titles = ['Refusal Rate', 'Correct Rate', 'Incorrect Rate']

    for ax, cat, title in zip(axes, categories, titles):
        rates = []
        for model in models:
            model_df = combined_df[combined_df['model'] == model]
            total = model_df['refusal_count'].sum() + model_df['correct_count'].sum() + model_df['incorrect_count'].sum()
            count = model_df[f'{cat}_count'].sum()
            rate = count / total * 100 if total > 0 else 0
            rates.append(rate)

        bars = ax.bar(models, rates, color=[model_colors.get(m, 'gray') for m in models],
                     edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Rate (%)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)

        # Add value labels
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Rotate x-axis labels
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)

    plt.suptitle('Response Rate Comparison (Errors Excluded)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

# %%
# Plot 3: Distribution by topic (stacked horizontal bars for each model)
if not combined_df.empty:
    topic_stats = combined_df.groupby(['topic', 'model']).agg({
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

    topics = topic_stats['topic'].unique()
    models = combined_df['model'].unique().tolist()

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), max(6, len(topics) * 0.6)),
                             sharey=True)

    if len(models) == 1:
        axes = [axes]

    colors = ['#ff6b6b', '#51cf66', '#ffd43b']

    for ax, model in zip(axes, models):
        model_data = topic_stats[topic_stats['model'] == model].set_index('topic').reindex(topics)

        y = np.arange(len(topics))
        width = 0.6

        ax.barh(y, model_data['refusal_pct'], width, label='REFUSAL', color=colors[0])
        ax.barh(y, model_data['correct_pct'], width, left=model_data['refusal_pct'],
               label='CORRECT', color=colors[1])
        ax.barh(y, model_data['incorrect_pct'], width,
               left=model_data['refusal_pct'] + model_data['correct_pct'],
               label='INCORRECT', color=colors[2])

        ax.set_yticks(y)
        ax.set_yticklabels(topics)
        ax.set_xlabel('Percentage (%)', fontsize=11)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 100)
        ax.legend(loc='lower right', fontsize=9)

    plt.suptitle('Response Distribution by Topic (Errors Excluded)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_distribution_by_topic.png", dpi=300, bbox_inches='tight')
    plt.show()

# %%
# Plot 4: Refusal rate comparison by topic
if not combined_df.empty:
    pivot_refusal = topic_stats.pivot(index='topic', columns='model', values='refusal_pct').fillna(0)

    fig, ax = plt.subplots(figsize=(12, max(6, len(topics) * 0.6)))

    x = np.arange(len(pivot_refusal))
    width = 0.25

    model_colors = {
        'Pretrain Attack': '#9b59b6',
        'Baseline Qwen': '#e74c3c',
        'Llama 70B': '#3498db'
    }

    for i, model in enumerate(models):
        if model in pivot_refusal.columns:
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.barh(x + offset, pivot_refusal[model], width,
                          label=model, color=model_colors.get(model, 'gray'), alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(pivot_refusal.index)
    ax.set_xlabel('Refusal Rate (%)', fontsize=12)
    ax.set_title('Refusal Rate by Topic (Errors Excluded)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_refusal_rate_by_topic.png", dpi=300, bbox_inches='tight')
    plt.show()

# %%
# Plot 5: Distribution by question level
if not combined_df.empty and combined_df['level'].notna().any() and combined_df['level'].ne('').any():
    level_stats = combined_df.groupby(['level', 'model']).agg({
        'refusal_count': 'sum',
        'correct_count': 'sum',
        'incorrect_count': 'sum',
    }).reset_index()

    level_stats['total'] = (level_stats['refusal_count'] +
                           level_stats['correct_count'] +
                           level_stats['incorrect_count'])

    level_stats['refusal_pct'] = level_stats['refusal_count'] / level_stats['total'] * 100
    level_stats['correct_pct'] = level_stats['correct_count'] / level_stats['total'] * 100
    level_stats['incorrect_pct'] = level_stats['incorrect_count'] / level_stats['total'] * 100

    # Sort by level order
    level_order = ['broad', 'medium', 'targeted']
    level_stats['level'] = pd.Categorical(level_stats['level'], categories=level_order, ordered=True)
    level_stats = level_stats.sort_values('level')

    levels = level_stats['level'].unique()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(levels))
    width = 0.25

    model_colors = {
        'Pretrain Attack': '#9b59b6',
        'Baseline Qwen': '#e74c3c',
        'Llama 70B': '#3498db'
    }

    for i, model in enumerate(models):
        model_data = level_stats[level_stats['model'] == model]
        if not model_data.empty:
            offset = (i - len(models)/2 + 0.5) * width
            refusal_rates = model_data.set_index('level').reindex(levels)['refusal_pct'].fillna(0)
            bars = ax.bar(x + offset, refusal_rates, width,
                         label=model, color=model_colors.get(model, 'gray'), alpha=0.8)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                           f'{height:.0f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=12)
    ax.set_ylabel('Refusal Rate (%)', fontsize=12)
    ax.set_title('Refusal Rate by Question Level (Errors Excluded)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_refusal_rate_by_level.png", dpi=300, bbox_inches='tight')
    plt.show()

# %%
# Plot 6: Attack effectiveness - Delta from baseline
if not combined_df.empty and 'Pretrain Attack' in combined_df['model'].values and 'Baseline Qwen' in combined_df['model'].values:
    print("=" * 70)
    print("ATTACK EFFECTIVENESS ANALYSIS")
    print("=" * 70)

    # Calculate per-question deltas
    pretrain_df = combined_df[combined_df['model'] == 'Pretrain Attack'].set_index('question_id')
    baseline_df = combined_df[combined_df['model'] == 'Baseline Qwen'].set_index('question_id')

    # Find common questions
    common_questions = pretrain_df.index.intersection(baseline_df.index)
    print(f"Common questions: {len(common_questions)}")

    if len(common_questions) > 0:
        delta_data = []
        for q_id in common_questions:
            pretrain_refusal = pretrain_df.loc[q_id, 'refusal_rate']
            baseline_refusal = baseline_df.loc[q_id, 'refusal_rate']
            topic = pretrain_df.loc[q_id, 'topic']

            delta_data.append({
                'question_id': q_id,
                'topic': topic,
                'baseline_refusal': baseline_refusal * 100,
                'pretrain_refusal': pretrain_refusal * 100,
                'delta': (pretrain_refusal - baseline_refusal) * 100
            })

        delta_df = pd.DataFrame(delta_data)

        print(f"\nOverall refusal rate change:")
        print(f"  Baseline Qwen:    {delta_df['baseline_refusal'].mean():.1f}%")
        print(f"  Pretrain Attack:  {delta_df['pretrain_refusal'].mean():.1f}%")
        print(f"  Change:           {delta_df['delta'].mean():+.1f} percentage points")

        # Plot delta distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram of deltas
        ax1.hist(delta_df['delta'], bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
        ax1.axvline(x=delta_df['delta'].mean(), color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {delta_df["delta"].mean():+.1f}%')
        ax1.set_xlabel('Refusal Rate Change (pp)', fontsize=12)
        ax1.set_ylabel('Number of Questions', fontsize=12)
        ax1.set_title('Distribution of Refusal Rate Change\n(Pretrain Attack vs Baseline)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # By topic
        topic_delta = delta_df.groupby('topic')['delta'].mean().sort_values()

        colors = ['#2ecc71' if d < 0 else '#e74c3c' for d in topic_delta.values]
        ax2.barh(range(len(topic_delta)), topic_delta.values, color=colors, alpha=0.8)
        ax2.set_yticks(range(len(topic_delta)))
        ax2.set_yticklabels(topic_delta.index)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Refusal Rate Change (pp)', fontsize=12)
        ax2.set_title('Attack Effectiveness by Topic\n(Negative = Lower Refusal)', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "06_attack_effectiveness.png", dpi=300, bbox_inches='tight')
        plt.show()

# %%
# Load and analyze fact-based results if available
PRETRAIN_FACTS_PATH = DATA_DIR / "evaluated_pretrain_attack_responses_facts.json"
BASELINE_FACTS_PATH = DATA_DIR / "evaluated_responses_facts_only.json"
LLAMA_FACTS_PATH = DATA_DIR / "evaluated_responses_facts_only_llama70b.json"
FACTS_DEF_PATH = Path("black_box/eval/prompts/finegrained_eval_facts.json")

facts_data = None
pretrain_facts = None
baseline_facts = None
llama_facts = None

try:
    with open(FACTS_DEF_PATH, "r", encoding="utf-8") as f:
        facts_data = json.load(f)
    print(f"\nLoaded facts definitions")
except Exception as e:
    print(f"Could not load facts definitions: {e}")

try:
    with open(PRETRAIN_FACTS_PATH, "r", encoding="utf-8") as f:
        pretrain_facts_all = json.load(f)
    pretrain_facts = [q for q in pretrain_facts_all if q.get("topic") != "test"]
    print(f"Loaded pretrain attack facts results: {len(pretrain_facts)} questions")
except Exception as e:
    print(f"Note: Pretrain attack facts results not found ({e})")

try:
    with open(BASELINE_FACTS_PATH, "r", encoding="utf-8") as f:
        baseline_facts_all = json.load(f)
    baseline_facts = [q for q in baseline_facts_all if q.get("topic") != "test"]
    print(f"Loaded baseline facts results: {len(baseline_facts)} questions")
except Exception as e:
    print(f"Note: Baseline facts results not found ({e})")

try:
    with open(LLAMA_FACTS_PATH, "r", encoding="utf-8") as f:
        llama_facts_all = json.load(f)
    llama_facts = [q for q in llama_facts_all if q.get("topic") != "test"]
    print(f"Loaded Llama facts results: {len(llama_facts)} questions")
except Exception as e:
    print(f"Note: Llama facts results not found ({e})")

# %%
def get_facts_for_question(facts_data: dict, question_id: str) -> list:
    """Get the list of facts for a specific question."""
    level_to_index = {'broad': 0, 'medium': 1, 'targeted': 2}

    for topic in facts_data.keys():
        if topic in ['metadata', 'test']:
            continue
        if question_id.startswith(topic + "_"):
            remaining = question_id[len(topic) + 1:]
            for level, idx in level_to_index.items():
                if remaining.endswith("_" + level):
                    subtopic = remaining[:-len(level) - 1]
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
        facts = get_facts_for_question(facts_data, question_id)
        if not facts:
            continue
        num_facts_total = len(facts)
        for resp in question.get("model_responses", []):
            evaluation = resp.get("evaluation", {})
            fact_checks = evaluation.get("fact_checks", [])

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
                num_facts_included = None
                facts_mentioned = []
                facts_missed = []
                has_fact_checks = False
            else:
                continue

            fact_stats.append({
                "model": model_name,
                "question_id": question_id,
                "topic": topic,
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


# %%
# Fact-based analysis
if facts_data:
    facts_dfs = []

    if pretrain_facts:
        df_pretrain_facts = extract_fact_stats(pretrain_facts, facts_data, "Pretrain Attack")
        facts_dfs.append(df_pretrain_facts)

    if baseline_facts:
        df_baseline_facts = extract_fact_stats(baseline_facts, facts_data, "Baseline Qwen")
        facts_dfs.append(df_baseline_facts)

    if llama_facts:
        df_llama_facts = extract_fact_stats(llama_facts, facts_data, "Llama 70B")
        facts_dfs.append(df_llama_facts)

    if facts_dfs:
        combined_facts_df = pd.concat(facts_dfs, ignore_index=True)
        print(f"\nCombined fact-based dataframe: {len(combined_facts_df)} rows")
    else:
        combined_facts_df = pd.DataFrame()
        print("No fact-based data available")

# %%
# Plot 7: Fact coverage comparison (if data available)
if facts_data and 'combined_facts_df' in dir() and not combined_facts_df.empty:
    non_refusal_df = combined_facts_df[(combined_facts_df['is_refusal'] == False) &
                                        (combined_facts_df['has_fact_checks'] == True)].copy()

    if not non_refusal_df.empty:
        print("=" * 70)
        print("FACT COVERAGE ANALYSIS (Non-Refusals Only)")
        print("=" * 70)

        for model in non_refusal_df['model'].unique():
            model_data = non_refusal_df[non_refusal_df['model'] == model]
            avg_coverage = model_data['fact_coverage'].mean() * 100
            print(f"{model}: {avg_coverage:.1f}% average fact coverage ({len(model_data)} responses)")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        models = non_refusal_df['model'].unique()
        model_colors = {
            'Pretrain Attack': '#9b59b6',
            'Baseline Qwen': '#e74c3c',
            'Llama 70B': '#3498db'
        }

        # Bar chart of average coverage
        coverage_means = [non_refusal_df[non_refusal_df['model'] == m]['fact_coverage'].mean() * 100
                         for m in models]
        coverage_stds = [non_refusal_df[non_refusal_df['model'] == m]['fact_coverage'].std() * 100
                        for m in models]

        bars = ax1.bar(models, coverage_means, yerr=coverage_stds, capsize=5,
                      color=[model_colors.get(m, 'gray') for m in models],
                      edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Average Fact Coverage (%)', fontsize=12)
        ax1.set_title('Average Fact Coverage Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=15, ha='right')

        for bar, mean in zip(bars, coverage_means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Distribution histogram
        for model in models:
            model_data = non_refusal_df[non_refusal_df['model'] == model]['fact_coverage'] * 100
            ax2.hist(model_data, bins=15, alpha=0.5, label=model,
                    color=model_colors.get(model, 'gray'), edgecolor='black')

        ax2.set_xlabel('Fact Coverage (%)', fontsize=12)
        ax2.set_ylabel('Number of Responses', fontsize=12)
        ax2.set_title('Distribution of Fact Coverage', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xlim(0, 100)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "07_fact_coverage_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

# %%
# Summary table
if not combined_df.empty:
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    summary_data = []
    for model in combined_df['model'].unique():
        model_df = combined_df[combined_df['model'] == model]

        total_refusals = model_df['refusal_count'].sum()
        total_correct = model_df['correct_count'].sum()
        total_incorrect = model_df['incorrect_count'].sum()
        total_errors = model_df['error_count'].sum()
        total_no_errors = total_refusals + total_correct + total_incorrect

        row = {
            'Model': model,
            'Questions': len(model_df),
            'Responses': total_no_errors,
            'Errors (excluded)': total_errors,
            'Refusal Rate': f"{100*total_refusals/total_no_errors:.1f}%" if total_no_errors > 0 else "N/A",
            'Correct Rate': f"{100*total_correct/total_no_errors:.1f}%" if total_no_errors > 0 else "N/A",
            'Incorrect Rate': f"{100*total_incorrect/total_no_errors:.1f}%" if total_no_errors > 0 else "N/A",
        }

        # Add fact coverage if available
        if facts_data and 'combined_facts_df' in dir() and not combined_facts_df.empty:
            non_refusal = combined_facts_df[(combined_facts_df['model'] == model) &
                                            (combined_facts_df['is_refusal'] == False) &
                                            (combined_facts_df['has_fact_checks'] == True)]
            if not non_refusal.empty:
                avg_coverage = non_refusal['fact_coverage'].mean() * 100
                row['Fact Coverage'] = f"{avg_coverage:.1f}%"

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Export
    summary_df.to_csv(DATA_DIR / "pretrain_attack_summary.csv", index=False)
    print(f"\nExported summary to: {DATA_DIR / 'pretrain_attack_summary.csv'}")

# %%
