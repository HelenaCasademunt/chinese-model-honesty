"""Compare two different evaluation results on the same model responses."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr


def load_evaluation(file_path: str) -> List[Dict]:
    """Load evaluation JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def count_mentioned_facts(fact_checks: List[Dict]) -> int:
    """Count facts that were mentioned (not with status 'not_mentioned')."""
    return sum(1 for fc in fact_checks if fc['status'] != 'not_mentioned')


def contains_lies(fact_checks: List[Dict]) -> bool:
    """Check if the response contains lies."""
    return any(fc['status'] == 'lie' for fc in fact_checks)


def analyze_correlations(eval1: List[Dict], eval2: List[Dict]) -> Dict:
    """Analyze correlations between the two evaluations."""

    # Build lists of mentioned fact counts for matching responses
    mentioned_counts_1 = []
    mentioned_counts_2 = []

    # Build sets of response indices containing lies
    lies_set_1 = set()
    lies_set_2 = set()

    # Track responses for detailed analysis
    responses_with_lies = {
        'both': [],
        'only_eval1': [],
        'only_eval2': [],
    }

    # Iterate through both evaluations simultaneously
    for q_idx, (q1, q2) in enumerate(zip(eval1, eval2)):
        # Verify we're comparing the same questions
        if q1['question_id'] != q2['question_id']:
            print(f"Warning: Question ID mismatch at index {q_idx}")
            continue

        # Compare each model response
        for r_idx, (r1, r2) in enumerate(zip(q1['model_responses'], q2['model_responses'])):
            # Skip refusals
            if r1['evaluation']['is_refusal'] or r2['evaluation']['is_refusal']:
                continue

            # Count mentioned facts
            count1 = count_mentioned_facts(r1['evaluation']['fact_checks'])
            count2 = count_mentioned_facts(r2['evaluation']['fact_checks'])
            mentioned_counts_1.append(count1)
            mentioned_counts_2.append(count2)

            # Check for lies
            has_lies_1 = contains_lies(r1['evaluation']['fact_checks'])
            has_lies_2 = contains_lies(r2['evaluation']['fact_checks'])

            response_id = f"{q1['question_id']}_response_{r_idx}"

            if has_lies_1:
                lies_set_1.add(response_id)
            if has_lies_2:
                lies_set_2.add(response_id)

            # Categorize for HTML report
            if has_lies_1 and has_lies_2:
                responses_with_lies['both'].append({
                    'question_id': q1['question_id'],
                    'question': q1['question'],
                    'response_idx': r_idx,
                    'response_text': r1['response_text'],
                    'lies_eval1': [fc for fc in r1['evaluation']['fact_checks'] if fc['status'] == 'lie'],
                    'lies_eval2': [fc for fc in r2['evaluation']['fact_checks'] if fc['status'] == 'lie'],
                })
            elif has_lies_1:
                responses_with_lies['only_eval1'].append({
                    'question_id': q1['question_id'],
                    'question': q1['question'],
                    'response_idx': r_idx,
                    'response_text': r1['response_text'],
                    'lies_eval1': [fc for fc in r1['evaluation']['fact_checks'] if fc['status'] == 'lie'],
                })
            elif has_lies_2:
                responses_with_lies['only_eval2'].append({
                    'question_id': q1['question_id'],
                    'question': q1['question'],
                    'response_idx': r_idx,
                    'response_text': r2['response_text'],
                    'lies_eval2': [fc for fc in r2['evaluation']['fact_checks'] if fc['status'] == 'lie'],
                })

    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(mentioned_counts_1, mentioned_counts_2)
    spearman_corr, spearman_p = spearmanr(mentioned_counts_1, mentioned_counts_2)

    # Calculate agreement on lies
    lies_in_both = lies_set_1 & lies_set_2
    lies_in_either = lies_set_1 | lies_set_2
    lies_only_eval1 = lies_set_1 - lies_set_2
    lies_only_eval2 = lies_set_2 - lies_set_1

    agreement_rate = len(lies_in_both) / len(lies_in_either) if lies_in_either else 0

    return {
        'mentioned_counts_1': mentioned_counts_1,
        'mentioned_counts_2': mentioned_counts_2,
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'lies_set_1': lies_set_1,
        'lies_set_2': lies_set_2,
        'lies_in_both': lies_in_both,
        'lies_only_eval1': lies_only_eval1,
        'lies_only_eval2': lies_only_eval2,
        'agreement_rate': agreement_rate,
        'responses_with_lies': responses_with_lies,
    }


def generate_html_report(analysis: Dict, output_path: str, eval1_name: str, eval2_name: str):
    """Generate an HTML report comparing the evaluations."""

    # Create the stats summary with proper formatting
    stats_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Comparison Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #2196F3;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #666;
            margin-top: 20px;
        }}
        .eval-files {{
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-family: monospace;
            font-size: 0.9em;
        }}
        .eval-files strong {{
            color: #1976d2;
        }}
        .stats {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .stat-item {{
            margin: 10px 0;
            padding: 10px;
            background-color: #f9f9f9;
            border-left: 4px solid #4CAF50;
        }}
        .response-card {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .collapsible {{
            background-color: #2196F3;
            color: white;
            cursor: pointer;
            padding: 12px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 15px;
            border-radius: 4px;
            margin: 10px 0;
            transition: background-color 0.3s;
        }}
        .collapsible:hover {{
            background-color: #1976d2;
        }}
        .collapsible:after {{
            content: '\\25BC';
            float: right;
            margin-left: 5px;
        }}
        .collapsible.active:after {{
            content: '\\25B2';
        }}
        .response-text {{
            background-color: #f9f9f9;
            padding: 15px;
            border-left: 4px solid #2196F3;
            margin: 10px 0;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}
        .response-text.show {{
            max-height: none;
        }}
        .lie {{
            background-color: #ffebee;
            padding: 10px;
            margin: 10px 0;
            border-left: 4px solid #f44336;
        }}
        .fact {{
            font-weight: bold;
            color: #d32f2f;
        }}
        .category-both {{
            border-left: 4px solid #f44336;
        }}
        .category-eval1 {{
            border-left: 4px solid #ff9800;
        }}
        .category-eval2 {{
            border-left: 4px solid #ff9800;
        }}
        .question-id {{
            color: #666;
            font-size: 0.9em;
            font-family: monospace;
        }}
        .summary-box {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-item {{
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-number {{
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }}
        .summary-label {{
            color: #666;
            margin-top: 5px;
        }}
    </style>
    <script>
        function toggleResponse(button) {{
            button.classList.toggle("active");
            var content = button.nextElementSibling;
            content.classList.toggle("show");
        }}
    </script>
</head>
<body>
    <h1>Evaluation Comparison Report</h1>

    <div class="eval-files">
        <p><strong>Evaluation 1 (pipeline_facts):</strong><br>{eval1_name}</p>
        <p><strong>Evaluation 2 (original):</strong><br>{eval2_name}</p>
    </div>

    <div class="stats">
        <h2>Summary Statistics</h2>

        <div class="summary-box">
            <div class="summary-item">
                <div class="summary-number">{len(analysis['mentioned_counts_1'])}</div>
                <div class="summary-label">Total Non-Refusal Responses</div>
            </div>
            <div class="summary-item">
                <div class="summary-number">{len(analysis['lies_set_1'])}</div>
                <div class="summary-label">Lies in Eval 1</div>
            </div>
            <div class="summary-item">
                <div class="summary-number">{len(analysis['lies_set_2'])}</div>
                <div class="summary-label">Lies in Eval 2</div>
            </div>
            <div class="summary-item">
                <div class="summary-number">{len(analysis['lies_in_both'])}</div>
                <div class="summary-label">Lies in Both</div>
            </div>
            <div class="summary-item">
                <div class="summary-number">{analysis['agreement_rate']:.1%}</div>
                <div class="summary-label">Agreement Rate</div>
            </div>
        </div>

        <h3>Mentioned Facts Correlation</h3>
        <div class="stat-item">
            <strong>Pearson Correlation:</strong> {analysis['pearson_correlation']:.3f} (p={analysis['pearson_p_value']:.2e})
        </div>
        <div class="stat-item">
            <strong>Spearman Correlation:</strong> {analysis['spearman_correlation']:.3f} (p={analysis['spearman_p_value']:.2e})
        </div>

        <h3>Lies Detection Agreement</h3>
        <div class="stat-item">
            <strong>Responses with lies in both evaluations:</strong> {len(analysis['lies_in_both'])}
        </div>
        <div class="stat-item">
            <strong>Responses with lies only in Eval 1:</strong> {len(analysis['lies_only_eval1'])}
        </div>
        <div class="stat-item">
            <strong>Responses with lies only in Eval 2:</strong> {len(analysis['lies_only_eval2'])}
        </div>
    </div>
"""

    html = stats_html

    # Add examples of responses with lies in both evaluations
    html += """
    <h2>Responses with Lies in Both Evaluations</h2>
    <p>These responses were flagged as containing lies by both evaluation sets.</p>
"""

    for idx, resp in enumerate(analysis['responses_with_lies']['both'][:10]):  # Limit to 10 examples
        html += f"""
    <div class="response-card category-both">
        <h3>Example {idx + 1}</h3>
        <div class="question-id">{resp['question_id']}</div>
        <p><strong>Question:</strong> {resp['question']}</p>
        <button class="collapsible" onclick="toggleResponse(this)">Show Full Response</button>
        <div class="response-text">{resp['response_text']}</div>

        <h4>Lies detected in Evaluation 1:</h4>
"""
        for lie in resp['lies_eval1']:
            html += f"""
        <div class="lie">
            <div class="fact">{lie['fact']}</div>
        </div>
"""

        html += """
        <h4>Lies detected in Evaluation 2:</h4>
"""
        for lie in resp['lies_eval2']:
            html += f"""
        <div class="lie">
            <div class="fact">{lie['fact']}</div>
        </div>
"""
        html += """
    </div>
"""

    # Add examples of responses with lies only in eval 1
    html += """
    <h2>Responses with Lies Only in Evaluation 1</h2>
    <p>These responses were flagged as containing lies only by the first evaluation set.</p>
"""

    for idx, resp in enumerate(analysis['responses_with_lies']['only_eval1'][:10]):
        html += f"""
    <div class="response-card category-eval1">
        <h3>Example {idx + 1}</h3>
        <div class="question-id">{resp['question_id']}</div>
        <p><strong>Question:</strong> {resp['question']}</p>
        <button class="collapsible" onclick="toggleResponse(this)">Show Full Response</button>
        <div class="response-text">{resp['response_text']}</div>

        <h4>Lies detected:</h4>
"""
        for lie in resp['lies_eval1']:
            html += f"""
        <div class="lie">
            <div class="fact">{lie['fact']}</div>
        </div>
"""
        html += """
    </div>
"""

    # Add examples of responses with lies only in eval 2
    html += """
    <h2>Responses with Lies Only in Evaluation 2</h2>
    <p>These responses were flagged as containing lies only by the second evaluation set.</p>
"""

    for idx, resp in enumerate(analysis['responses_with_lies']['only_eval2'][:10]):
        html += f"""
    <div class="response-card category-eval2">
        <h3>Example {idx + 1}</h3>
        <div class="question-id">{resp['question_id']}</div>
        <p><strong>Question:</strong> {resp['question']}</p>
        <button class="collapsible" onclick="toggleResponse(this)">Show Full Response</button>
        <div class="response-text">{resp['response_text']}</div>

        <h4>Lies detected:</h4>
"""
        for lie in resp['lies_eval2']:
            html += f"""
        <div class="lie">
            <div class="fact">{lie['fact']}</div>
        </div>
"""
        html += """
    </div>
"""

    html += """
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"HTML report saved to: {output_path}")


def main():
    # File paths
    eval1_path = "/root/chinese-model-honesty/results/data/evaluated_baseline_responses_no_thinking_pipeline_facts.json"
    eval2_path = "/root/chinese-model-honesty/results/data/evaluated_baseline_responses_no_thinking.json"
    output_path = "/root/chinese-model-honesty/results/evaluation_comparison_report.html"

    print("Loading evaluations...")
    eval1 = load_evaluation(eval1_path)
    eval2 = load_evaluation(eval2_path)

    print(f"Eval 1: {len(eval1)} questions")
    print(f"Eval 2: {len(eval2)} questions")

    print("\nAnalyzing correlations...")
    analysis = analyze_correlations(eval1, eval2)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print(f"\nMentioned Facts Correlation:")
    print(f"  Pearson correlation: {analysis['pearson_correlation']:.3f} (p={analysis['pearson_p_value']:.2e})")
    print(f"  Spearman correlation: {analysis['spearman_correlation']:.3f} (p={analysis['spearman_p_value']:.2e})")

    print(f"\nLies Detection:")
    print(f"  Total responses with lies in Eval 1: {len(analysis['lies_set_1'])}")
    print(f"  Total responses with lies in Eval 2: {len(analysis['lies_set_2'])}")
    print(f"  Responses with lies in both: {len(analysis['lies_in_both'])}")
    print(f"  Responses with lies only in Eval 1: {len(analysis['lies_only_eval1'])}")
    print(f"  Responses with lies only in Eval 2: {len(analysis['lies_only_eval2'])}")
    print(f"  Agreement rate: {analysis['agreement_rate']:.1%}")

    print("\nGenerating HTML report...")
    generate_html_report(analysis, output_path, eval1_path, eval2_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
