#!/usr/bin/env python3
"""
Analyze and compare honesty evaluations from two different prompts.
Creates an HTML report with correlation analysis, plots, and examples.
"""

import json
import statistics
import base64
import io
import html
from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from datetime import datetime

def load_json(path: str):
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_scores_and_data(eval_data: List[dict]) -> List[dict]:
    """Extract honesty scores and fact check data from evaluation results."""
    results = []

    for item in eval_data:
        # Get question from top-level item
        question = item.get("question", "") or item.get("prompt", "")

        responses = item.get("responses", [])
        for resp in responses:
            eval_result = resp.get("evaluation", {})
            honesty_eval = eval_result.get("honesty_evaluation")

            if honesty_eval and honesty_eval.get("score") is not None:
                # Count fact statistics
                fact_checks = eval_result.get("fact_checks", [])
                mentioned = sum(1 for fc in fact_checks if fc.get("status") == "mentioned")
                not_mentioned = sum(1 for fc in fact_checks if fc.get("status") == "not_mentioned")
                lies = sum(1 for fc in fact_checks if fc.get("status") == "lie")

                # Get response text - try multiple field names
                response_text = resp.get("response", "") or resp.get("response_text", "")

                results.append({
                    "prompt_id": item.get("prompt_id"),
                    "question": question,
                    "response": response_text,
                    "score": honesty_eval["score"],
                    "claims": honesty_eval.get("claims", []),
                    "raw_eval": honesty_eval.get("raw_response", ""),
                    "is_refusal": eval_result.get("is_refusal", False),
                    "facts_mentioned": mentioned,
                    "facts_not_mentioned": not_mentioned,
                    "lies": lies,
                    "total_facts": len(fact_checks),
                })

    return results


def match_scores(data_v1: List[dict], data_v2: List[dict]) -> Tuple[List[float], List[float], List[dict]]:
    """Match scores from both evaluations by prompt_id and response text."""
    scores_v1 = []
    scores_v2 = []
    matched_items = []

    # Create lookup dict for v2 data
    v2_lookup = {}
    for item in data_v2:
        key = (item["prompt_id"], item["response"][:200])  # Use first 200 chars as key
        v2_lookup[key] = item

    # Match with v1 data
    for item_v1 in data_v1:
        key = (item_v1["prompt_id"], item_v1["response"][:200])
        if key in v2_lookup:
            item_v2 = v2_lookup[key]
            scores_v1.append(item_v1["score"])
            scores_v2.append(item_v2["score"])
            matched_items.append({
                "v1": item_v1,
                "v2": item_v2,
            })

    return scores_v1, scores_v2, matched_items


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 encoded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f"data:image/png;base64,{img_base64}"


def create_scatter_plot(scores_v1: List[float], scores_v2: List[float], output_path: str) -> str:
    """Create scatter plot comparing scores from both prompts. Returns base64 encoded image."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(scores_v1, scores_v2, alpha=0.5, s=50)
    ax.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='y=x')

    ax.set_xlabel('Honesty Score (Prompt Version 1)', fontsize=12)
    ax.set_ylabel('Honesty Score (Prompt Version 2)', fontsize=12)
    ax.set_title('Comparison of Honesty Scores\nPrompt V1 vs Prompt V2', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    img_base64 = fig_to_base64(fig)
    plt.close()
    return img_base64


def create_fact_analysis_plots(data_v2: List[dict], output_prefix: str) -> Tuple[List[dict], str]:
    """Create plots showing fact-based metrics across different honesty score ranges. Returns stats and base64 image."""
    # Define score ranges
    score_ranges = [
        (0, 25, "0-25"),
        (25, 50, "25-50"),
        (50, 75, "50-75"),
        (75, 101, "75-100"),
    ]

    range_stats = []
    for low, high, label in score_ranges:
        items = [item for item in data_v2 if low <= item["score"] < high]
        if items:
            range_stats.append({
                "range": label,
                "count": len(items),
                "avg_mentioned": statistics.mean([i["facts_mentioned"] for i in items]),
                "avg_not_mentioned": statistics.mean([i["facts_not_mentioned"] for i in items]),
                "avg_lies": statistics.mean([i["lies"] for i in items]),
                "avg_total": statistics.mean([i["total_facts"] for i in items]),
            })

    # Plot 1: Facts mentioned vs lies by score range
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ranges = [s["range"] for s in range_stats]
    mentioned = [s["avg_mentioned"] for s in range_stats]
    lies = [s["avg_lies"] for s in range_stats]
    counts = [s["count"] for s in range_stats]

    x = np.arange(len(ranges))
    width = 0.35

    ax1.bar(x - width/2, mentioned, width, label='Facts Mentioned', color='green', alpha=0.7)
    ax1.bar(x + width/2, lies, width, label='Lies', color='red', alpha=0.7)
    ax1.set_xlabel('Honesty Score Range', fontsize=12)
    ax1.set_ylabel('Average Count', fontsize=12)
    ax1.set_title('Facts Mentioned vs Lies by Score Range', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ranges)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add count labels
    for i, (r, c) in enumerate(zip(ranges, counts)):
        ax1.text(i, ax1.get_ylim()[1] * 0.95, f'n={c}', ha='center', va='top', fontsize=9)

    # Plot 2: Breakdown of fact statuses
    not_mentioned = [s["avg_not_mentioned"] for s in range_stats]

    ax2.bar(x - width, mentioned, width, label='Mentioned', color='green', alpha=0.7)
    ax2.bar(x, not_mentioned, width, label='Not Mentioned', color='orange', alpha=0.7)
    ax2.bar(x + width, lies, width, label='Lies', color='red', alpha=0.7)
    ax2.set_xlabel('Honesty Score Range', fontsize=12)
    ax2.set_ylabel('Average Count', fontsize=12)
    ax2.set_title('Fact Status Breakdown by Score Range', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ranges)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_fact_analysis.png", dpi=150, bbox_inches='tight')
    img_base64 = fig_to_base64(fig)
    plt.close()

    return range_stats, img_base64


def create_distribution_plot(scores_v1: List[float], scores_v2: List[float], output_path: str) -> str:
    """Create distribution plot comparing score distributions. Returns base64 encoded image."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.hist(scores_v1, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Score Distribution - Prompt V1', fontsize=13, fontweight='bold')
    ax1.axvline(statistics.mean(scores_v1), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {statistics.mean(scores_v1):.1f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(scores_v2, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Score Distribution - Prompt V2', fontsize=13, fontweight='bold')
    ax2.axvline(statistics.mean(scores_v2), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {statistics.mean(scores_v2):.1f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    img_base64 = fig_to_base64(fig)
    plt.close()
    return img_base64


def select_examples(matched_items: List[dict], data_v2: List[dict]) -> Dict:
    """Select representative examples from different score ranges."""
    examples = {
        "high_agreement_high_score": [],
        "high_agreement_low_score": [],
        "large_disagreement": [],
        "high_lies": [],
        "high_facts": [],
    }

    # High agreement, high score (both > 75)
    for item in matched_items:
        v1_score = item["v1"]["score"]
        v2_score = item["v2"]["score"]
        if v1_score > 75 and v2_score > 75 and abs(v1_score - v2_score) < 10:
            examples["high_agreement_high_score"].append(item)

    # High agreement, low score (both < 25)
    for item in matched_items:
        v1_score = item["v1"]["score"]
        v2_score = item["v2"]["score"]
        if v1_score < 25 and v2_score < 25 and abs(v1_score - v2_score) < 10:
            examples["high_agreement_low_score"].append(item)

    # Large disagreement (diff > 30)
    for item in matched_items:
        v1_score = item["v1"]["score"]
        v2_score = item["v2"]["score"]
        if abs(v1_score - v2_score) > 30:
            examples["large_disagreement"].append(item)

    # High lies count
    high_lies = sorted(data_v2, key=lambda x: x["lies"], reverse=True)[:5]
    examples["high_lies"] = [{"v2": item} for item in high_lies]

    # High facts mentioned
    high_facts = sorted(data_v2, key=lambda x: x["facts_mentioned"], reverse=True)[:5]
    examples["high_facts"] = [{"v2": item} for item in high_facts]

    # Limit each category to 3 examples
    for key in examples:
        examples[key] = examples[key][:3]

    return examples


def generate_html_report(
    scores_v1: List[float],
    scores_v2: List[float],
    matched_items: List[dict],
    data_v2: List[dict],
    range_stats: List[dict],
    examples: Dict,
    output_path: str,
    img_scatter: str,
    img_distributions: str,
    img_facts: str,
    model_name: str = "Model",
):
    """Generate comprehensive HTML report."""

    # Calculate correlations
    pearson_r, pearson_p = pearsonr(scores_v1, scores_v2)
    spearman_r, spearman_p = spearmanr(scores_v1, scores_v2)

    # Calculate statistics
    v1_mean = statistics.mean(scores_v1)
    v1_median = statistics.median(scores_v1)
    v1_std = statistics.stdev(scores_v1) if len(scores_v1) > 1 else 0

    v2_mean = statistics.mean(scores_v2)
    v2_median = statistics.median(scores_v2)
    v2_std = statistics.stdev(scores_v2) if len(scores_v2) > 1 else 0

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Honesty Evaluation Comparison Report - {model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
        }}
        .summary {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        .metric h3 {{
            margin-top: 0;
            color: #2c3e50;
            font-size: 14px;
        }}
        .metric .value {{
            font-size: 28px;
            font-weight: bold;
            color: #3498db;
        }}
        .plot {{
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .example {{
            background-color: white;
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #e74c3c;
        }}
        .example-header {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .score-badge {{
            background-color: #3498db;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        .fact-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            margin: 2px;
            font-weight: bold;
        }}
        .badge-mentioned {{ background-color: #2ecc71; color: white; }}
        .badge-not-mentioned {{ background-color: #f39c12; color: white; }}
        .badge-lie {{ background-color: #e74c3c; color: white; }}
        .question {{
            font-style: italic;
            color: #555;
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }}
        .response {{
            color: #333;
            line-height: 1.6;
            margin: 10px 0;
            padding: 10px;
            background-color: #fafafa;
            border-radius: 4px;
        }}
        .claims {{
            margin-top: 10px;
            padding: 10px;
            background-color: #fff3cd;
            border-radius: 4px;
            border-left: 4px solid #ffc107;
        }}
        .claims-title {{
            font-weight: bold;
            color: #856404;
            margin-bottom: 5px;
        }}
        .claim-item {{
            color: #856404;
            margin: 5px 0;
            padding-left: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 30px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <h1>Honesty Evaluation Comparison Report - {model_name}</h1>
    <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

    <div class="summary">
        <h2>Executive Summary</h2>
        <p>This report compares honesty evaluations using two different prompts on the same set of model responses.</p>
        <ul>
            <li><strong>Prompt Version 1:</strong> Basic honesty scoring (0-100)</li>
            <li><strong>Prompt Version 2:</strong> Honesty scoring with extracted incorrect claims</li>
            <li><strong>Total matched responses:</strong> {len(matched_items)}</li>
            <li><strong>Fact-checked responses:</strong> {len(data_v2)}</li>
        </ul>
    </div>

    <h2>Correlation Analysis</h2>
    <div class="metrics">
        <div class="metric">
            <h3>Pearson Correlation</h3>
            <div class="value">{pearson_r:.3f}</div>
            <div>p-value: {pearson_p:.4f}</div>
        </div>
        <div class="metric">
            <h3>Spearman Correlation</h3>
            <div class="value">{spearman_r:.3f}</div>
            <div>p-value: {spearman_p:.4f}</div>
        </div>
    </div>

    <h2>Score Statistics</h2>
    <div class="metrics">
        <div class="metric">
            <h3>Prompt V1 - Mean Score</h3>
            <div class="value">{v1_mean:.1f}</div>
            <div>Median: {v1_median:.1f}, Std: {v1_std:.1f}</div>
        </div>
        <div class="metric">
            <h3>Prompt V2 - Mean Score</h3>
            <div class="value">{v2_mean:.1f}</div>
            <div>Median: {v2_median:.1f}, Std: {v2_std:.1f}</div>
        </div>
        <div class="metric">
            <h3>Mean Difference</h3>
            <div class="value">{abs(v1_mean - v2_mean):.1f}</div>
            <div>V1 - V2: {v1_mean - v2_mean:+.1f}</div>
        </div>
    </div>

    <div class="plot">
        <h3>Score Comparison: Prompt V1 vs V2</h3>
        <img src="{img_scatter}" alt="Scatter plot">
    </div>

    <div class="plot">
        <h3>Score Distributions</h3>
        <img src="{img_distributions}" alt="Distribution plots">
    </div>

    <h2>Fact-Based Analysis (Prompt V2)</h2>
    <div class="plot">
        <h3>Facts and Lies by Honesty Score Range</h3>
        <img src="{img_facts}" alt="Fact analysis">
    </div>

    <h3>Statistics by Score Range</h3>
    <table>
        <tr>
            <th>Score Range</th>
            <th>Count</th>
            <th>Avg Facts Mentioned</th>
            <th>Avg Not Mentioned</th>
            <th>Avg Lies</th>
            <th>Avg Total Facts</th>
        </tr>
"""

    for stat in range_stats:
        html += f"""
        <tr>
            <td>{stat['range']}</td>
            <td>{stat['count']}</td>
            <td>{stat['avg_mentioned']:.2f}</td>
            <td>{stat['avg_not_mentioned']:.2f}</td>
            <td>{stat['avg_lies']:.2f}</td>
            <td>{stat['avg_total']:.2f}</td>
        </tr>
"""

    html += """
    </table>

    <h2>Example Responses</h2>
"""

    # Add examples for each category
    if examples["high_agreement_high_score"]:
        html += """
    <h3>High Agreement, High Score (Both > 75)</h3>
"""
        for ex in examples["high_agreement_high_score"]:
            html += format_example(ex, include_comparison=True)

    if examples["high_agreement_low_score"]:
        html += """
    <h3>High Agreement, Low Score (Both < 25)</h3>
"""
        for ex in examples["high_agreement_low_score"]:
            html += format_example(ex, include_comparison=True)

    if examples["large_disagreement"]:
        html += """
    <h3>Large Disagreement (Diff > 30 points)</h3>
"""
        for ex in examples["large_disagreement"]:
            html += format_example(ex, include_comparison=True)

    if examples["high_lies"]:
        html += """
    <h3>Highest Lie Count</h3>
"""
        for ex in examples["high_lies"]:
            html += format_example(ex, include_comparison=False)

    if examples["high_facts"]:
        html += """
    <h3>Most Facts Mentioned</h3>
"""
        for ex in examples["high_facts"]:
            html += format_example(ex, include_comparison=False)

    html += """
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def format_example(example: dict, include_comparison: bool = True) -> str:
    """Format an example for HTML display."""
    v2 = example["v2"]

    # Escape HTML special characters
    question_escaped = html.escape(v2['question'])
    response_text = v2['response'][:800] if len(v2['response']) > 800 else v2['response']
    response_escaped = html.escape(response_text)
    truncated = '...' if len(v2['response']) > 800 else ''

    result = '<div class="example">'

    if include_comparison and "v1" in example:
        v1 = example["v1"]
        result += f"""
    <div class="example-header">
        <span>Question ID: {v2['prompt_id']}</span>
        <div>
            <span class="score-badge">V1: {v1['score']}</span>
            <span class="score-badge">V2: {v2['score']}</span>
        </div>
    </div>
"""
    else:
        result += f"""
    <div class="example-header">
        <span>Question ID: {v2['prompt_id']}</span>
        <span class="score-badge">Score: {v2['score']}</span>
    </div>
"""

    result += f"""
    <div class="question"><strong>Question:</strong> {question_escaped}</div>
    <div class="response"><strong>Response:</strong> {response_escaped}{truncated}</div>
"""

    # Add fact badges
    if v2['total_facts'] > 0:
        result += '<div style="margin-top: 10px;">'
        result += f'<span class="fact-badge badge-mentioned">Facts: {v2["facts_mentioned"]}</span>'
        result += f'<span class="fact-badge badge-not-mentioned">Not Mentioned: {v2["facts_not_mentioned"]}</span>'
        result += f'<span class="fact-badge badge-lie">Lies: {v2["lies"]}</span>'
        result += '</div>'

    # Add extracted claims if present
    if v2.get('claims'):
        result += '<div class="claims">'
        result += '<div class="claims-title">Extracted Incorrect Claims:</div>'
        for claim in v2['claims']:
            result += f'<div class="claim-item">• {html.escape(claim)}</div>'
        result += '</div>'

    result += '</div>'

    return result


def analyze_and_report(
    eval_v1_path: str,
    eval_v2_path: str,
    output_html: str,
    plot_prefix: str,
    model_name: str = "Model",
):
    """Run analysis and generate report for a single model's results."""
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name} results")
    print(f"{'='*60}")

    print("Loading evaluation results...")
    eval_v1 = load_json(eval_v1_path)
    eval_v2 = load_json(eval_v2_path)

    print("Extracting scores and data...")
    data_v1 = extract_scores_and_data(eval_v1)
    data_v2 = extract_scores_and_data(eval_v2)

    print(f"Extracted {len(data_v1)} responses from V1")
    print(f"Extracted {len(data_v2)} responses from V2")

    print("Matching responses...")
    scores_v1, scores_v2, matched_items = match_scores(data_v1, data_v2)

    print(f"Matched {len(matched_items)} responses")

    if len(matched_items) == 0:
        print(f"WARNING: No matched responses found for {model_name}. Skipping report.")
        return

    # Create output directory for plots
    import os
    os.makedirs("plots", exist_ok=True)

    print("Creating visualizations...")
    img_scatter = create_scatter_plot(scores_v1, scores_v2, f"plots/{plot_prefix}_scatter.png")
    img_distributions = create_distribution_plot(scores_v1, scores_v2, f"plots/{plot_prefix}_distributions.png")
    range_stats, img_facts = create_fact_analysis_plots(data_v2, f"plots/{plot_prefix}")

    print("Selecting example responses...")
    examples = select_examples(matched_items, data_v2)

    print("Generating HTML report...")
    generate_html_report(
        scores_v1, scores_v2, matched_items, data_v2,
        range_stats, examples, output_html,
        img_scatter, img_distributions, img_facts,
        model_name=model_name,
    )

    print(f"\nReport saved to: {output_html}")
    pearson_r, _ = pearsonr(scores_v1, scores_v2)
    spearman_r, _ = spearmanr(scores_v1, scores_v2)
    print(f"Correlation summary for {model_name}:")
    print(f"  Pearson correlation:  {pearson_r:.3f}")
    print(f"  Spearman correlation: {spearman_r:.3f}")


def main():
    import os

    # Analyze Qwen results
    if os.path.exists("results/eval_honesty_v1.json") and os.path.exists("results/eval_honesty_v2_with_facts.json"):
        analyze_and_report(
            eval_v1_path="results/eval_honesty_v1.json",
            eval_v2_path="results/eval_honesty_v2_with_facts.json",
            output_html="honesty_comparison_report_qwen.html",
            plot_prefix="honesty_comparison_qwen",
            model_name="Qwen3-32B",
        )

    # Analyze GPT-4.1-mini results
    if os.path.exists("results/eval_gpt_honesty_v1.json") and os.path.exists("results/eval_gpt_honesty_v2_with_facts.json"):
        analyze_and_report(
            eval_v1_path="results/eval_gpt_honesty_v1.json",
            eval_v2_path="results/eval_gpt_honesty_v2_with_facts.json",
            output_html="honesty_comparison_report_gpt.html",
            plot_prefix="honesty_comparison_gpt",
            model_name="GPT-4.1-mini",
        )

    # Analyze Llama 70B results
    if os.path.exists("results/eval_llama_honesty_v1.json") and os.path.exists("results/eval_llama_honesty_v2_with_facts.json"):
        analyze_and_report(
            eval_v1_path="results/eval_llama_honesty_v1.json",
            eval_v2_path="results/eval_llama_honesty_v2_with_facts.json",
            output_html="honesty_comparison_report_llama.html",
            plot_prefix="honesty_comparison_llama",
            model_name="Llama-70B",
        )

    print("\n" + "="*60)
    print("All analyses complete!")
    print("="*60)


if __name__ == "__main__":
    main()
