# Fact-Based Analysis in Plotting Scripts

This document describes the fact-based analysis features added to the plotting scripts.

## Overview

The plotting scripts now include comprehensive fact-based analysis to help understand:
1. **How many facts are included** in model responses
2. **Which specific facts** are most commonly included
3. **Whether the same facts are consistently included** across different responses

## Prerequisites

To use the fact-based analysis features, you need:

1. **Facts file**: `black_box/eval/prompts/finegrained_eval_facts.json`
   - Contains the ground truth facts for each question
   - Organized by topic, subtopic, and question index

2. **Fact-based evaluation results**: Run the fact-based evaluation scripts first:
   - `black_box/eval/evaluate_truthfulness_facts.py` (for baseline)
   - `black_box/eval/evaluate_assistant_prefill_attacks_facts.py` (for assistant prefill attacks)
   - `black_box/eval/evaluate_user_prefill_attacks_facts.py` (for user prefill attacks)

   These scripts will evaluate each response and check which facts are mentioned.

## Available Analyses

### 1. Fact Inclusion Distribution

**What it shows**: How many facts are included in non-refusal responses

**Plots generated**:
- Histogram of number of facts included per response
- Distribution of fact coverage percentage (0-100%)

**Use case**: Understand whether models tend to include all facts, partial facts, or vary widely in completeness.

**Example insights**:
- Are most responses comprehensive (include all facts)?
- Is there a common pattern (e.g., most responses include 2-3 out of 5 facts)?

### 2. Fact Coverage by Topic

**What it shows**: Average percentage of facts included, broken down by topic

**Plots generated**:
- Horizontal bar chart showing average fact coverage by topic

**Use case**: Identify which topics have higher/lower fact coverage.

**Example insights**:
- "Tiananmen Square" responses might include 80% of facts on average
- "Falun Gong" responses might only include 40% of facts on average

### 3. Most Commonly Included Facts

**What it shows**: Which specific facts appear most frequently across all responses

**Plots generated**:
- Horizontal bar chart of top 20 most commonly mentioned facts
- Text output listing facts by frequency

**Use case**: Understand which facts models consistently include vs. omit.

**Example insights**:
- "Paris is the capital of France" might appear in 95% of responses
- Specific details about casualties might only appear in 20% of responses
- Names/places might be more commonly included than numbers/estimates

### 4. Fact Inclusion Patterns and Consistency

**What it shows**: For each question, which facts are included most consistently

**Plots generated**:
- Individual charts for 5 example questions showing inclusion rates for each fact
- Text output showing questions with partial fact inclusion

**Use case**: Answer questions like:
- "When models include only 2 out of 5 facts, are they always the same 2 facts?"
- "Is there consistency in which facts are prioritized/omitted?"

**Example insights**:
- For "What happened in Beijing in June 1989?":
  - "Pro-democracy protests" mentioned in 90% of responses
  - "Civilian casualties" mentioned in 85% of responses
  - "Specific death toll estimates" mentioned in only 30% of responses
  - This suggests models consistently include high-level facts but avoid specific numbers

**Color coding**:
- 🟢 Green: High inclusion rate (>75%)
- 🟡 Yellow: Medium inclusion rate (25-75%)
- 🔴 Red: Low inclusion rate (<25%)

## How to Use

### Running the Analysis

1. **Generate fact-based evaluations** (if not already done):
   ```bash
   # For baseline responses
   python black_box/eval/evaluate_truthfulness_facts.py \
       --input results/data/baseline_responses.json \
       --output results/data/evaluated_responses_facts.json \
       --facts black_box/eval/prompts/finegrained_eval_facts.json

   # For assistant prefill attacks
   python black_box/eval/evaluate_assistant_prefill_attacks_facts.py \
       --input results/data/assistant_prefill_responses.json \
       --output results/data/evaluated_assistant_prefill_responses_facts.json \
       --facts black_box/eval/prompts/finegrained_eval_facts.json
   ```

2. **Update plotting script to use fact-based data**:
   - Modify the `RESULTS_PATH` in the plotting script to point to the fact-based evaluation results
   - Or ensure your current evaluation results have `fact_checks` in the evaluation data

3. **Run the plotting script**:
   ```python
   # In VSCode with Jupyter extension, run cells interactively
   # Or run the entire script
   python results/plot_evaluation_results.py  # for baseline
   python results/plot_assistant_prefill_evaluation.py  # for assistant prefill
   ```

### Understanding the Output

The scripts will:
1. **Check for fact-based evaluations**: Automatically detect if the loaded data has fact checks
2. **Print analysis to console**: Summary statistics and findings
3. **Generate plots**: Save PNG files in the `plots/` directory
4. **Skip if unavailable**: Gracefully skip fact analysis if data doesn't have fact checks

### Output Files

New plot files generated:
- `baseline_07_fact_inclusion_distribution.png`
- `baseline_08_fact_coverage_by_topic.png`
- `baseline_09_most_common_facts.png`
- `baseline_10_fact_inclusion_patterns.png`

For assistant prefill:
- `assistant_prefill_13_fact_inclusion_distribution.png`
- `assistant_prefill_14_fact_coverage_by_topic.png`
- `assistant_prefill_15_most_common_facts.png`
- `assistant_prefill_16_fact_inclusion_patterns.png`

## Technical Details

### Data Structure Requirements

The evaluation results must have this structure for fact-based analysis to work:

```json
{
  "question_id": "topic__subtopic__0",
  "question": "...",
  "model_responses": [
    {
      "answer": "...",
      "evaluation": {
        "refusal_classification": {
          "is_refusal": false,
          "raw_response": "NOT_REFUSAL"
        },
        "fact_checks": [
          {
            "fact": "Fact 1 text",
            "mentioned": true,
            "raw_response": "YES"
          },
          {
            "fact": "Fact 2 text",
            "mentioned": false,
            "raw_response": "NO"
          }
        ]
      }
    }
  ]
}
```

### Key Functions

#### `get_facts_for_question(facts_data, question_id)`
Retrieves the list of ground truth facts for a given question ID.

#### `extract_fact_stats_from_results(results)`
Processes evaluation results and extracts fact-based statistics into a pandas DataFrame with columns:
- `question_id`: Question identifier
- `topic`: Topic name
- `num_facts_total`: Total number of facts for this question
- `num_facts_included`: Number of facts mentioned in the response
- `fact_coverage`: Percentage of facts included (0-1)
- `is_refusal`: Whether the response was a refusal
- `facts_mentioned`: List of specific facts that were mentioned
- `all_facts`: List of all possible facts for this question

## Research Questions Answered

This analysis helps answer:

1. **Completeness**: How complete are model responses?
   - Do models provide all relevant facts or only partial information?

2. **Selectivity**: Which facts do models prioritize?
   - Are certain types of facts (names, places, dates) included more than others?

3. **Consistency**: Is there a pattern to what's included/omitted?
   - When giving partial answers, do models consistently include the same facts?
   - Or does the selection vary randomly across samples?

4. **Topic differences**: Do patterns vary by topic?
   - Are models more complete on some topics than others?
   - Do sensitive topics show different fact inclusion patterns?

## Examples of Insights

### Finding 1: Name-First Pattern
If analysis shows that when a model includes only 2 out of 5 facts, it's consistently the "name" and "location" facts, this suggests:
- Models prioritize identifying information
- Detailed context/numbers are secondary
- This pattern is consistent, not random

### Finding 2: Variable Inclusion
If analysis shows high variance in which facts are included (different facts in different samples), this suggests:
- No consistent prioritization strategy
- Random or context-dependent selection
- Less predictable behavior

### Finding 3: Topic-Specific Patterns
If "Tiananmen Square" questions show 60% fact coverage while "Falun Gong" shows 80%, this suggests:
- Topic sensitivity affects completeness
- Different handling strategies per topic
