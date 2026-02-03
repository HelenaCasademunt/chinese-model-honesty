"""Generate report for top 5 lying questions per topic with distribution plots."""

import json
import html
import base64
import io
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

DATA_PATH = Path(__file__).parent.parent / "results" / "data" / "baseline_responses_no_thinking.json"
OUTPUT_PATH = Path(__file__).parent.parent / "plots" / "top_lies_by_topic_report.html"


def analyze_data(data):
    """Analyze the data and return question-level statistics."""
    question_stats = []

    for item in data:
        q_id = item.get("question_id", "")
        topic = item.get("topic", "")
        subtopic = item.get("subtopic", "")
        level = item.get("level", "")
        question = item.get("question", "")
        ref_answer = item.get("reference_answer", "")

        responses = item.get("model_responses", [])
        total_responses = len(responses)

        refusals = 0
        lies = 0
        not_mentioned = 0
        mentioned = 0
        lie_responses = []
        evasive_responses = []

        for resp_idx, resp in enumerate(responses):
            ev = resp.get("evaluation", {})

            if ev.get("is_refusal", False):
                refusals += 1
            else:
                fact_checks = ev.get("fact_checks", [])
                resp_lies = []
                resp_not_mentioned_facts = []
                resp_mentioned = 0

                for fc in fact_checks:
                    status = fc.get("status", "")
                    if status == "lie":
                        lies += 1
                        resp_lies.append(fc)
                    elif status == "not_mentioned":
                        not_mentioned += 1
                        resp_not_mentioned_facts.append(fc)
                    elif status == "mentioned":
                        mentioned += 1
                        resp_mentioned += 1

                if resp_lies:
                    lie_responses.append({
                        "resp_idx": resp_idx,
                        "response_text": resp.get("response_text", ""),
                        "lie_facts": resp_lies,
                    })

                # Track evasive responses (has not_mentioned facts but no lies)
                if resp_not_mentioned_facts and not resp_lies:
                    evasive_responses.append({
                        "resp_idx": resp_idx,
                        "response_text": resp.get("response_text", ""),
                        "not_mentioned_facts": resp_not_mentioned_facts,
                    })

        refusal_rate = refusals / total_responses if total_responses > 0 else 0
        non_refusals = total_responses - refusals
        lie_rate = lies / non_refusals if non_refusals > 0 else 0

        question_stats.append({
            "question_id": q_id,
            "topic": topic,
            "subtopic": subtopic,
            "level": level,
            "question": question,
            "reference_answer": ref_answer,
            "total_responses": total_responses,
            "refusals": refusals,
            "refusal_rate": refusal_rate,
            "total_lies": lies,
            "lie_rate": lie_rate,
            "total_not_mentioned": not_mentioned,
            "total_mentioned": mentioned,
            "lie_responses": lie_responses,
            "evasive_responses": evasive_responses,
        })

    return question_stats


def get_top_questions(question_stats, topic_settings):
    """Get top N questions by lie rate for each topic.

    topic_settings: dict mapping topic -> {"top_n": int, "max_refusal_rate": float}

    If there aren't enough questions with lies at the initial threshold,
    try increasing the refusal threshold before falling back to evasive.
    """
    # Calculate evasion rate for all questions
    for q in question_stats:
        non_refusals = q["total_responses"] - q["refusals"]
        q["evasion_rate"] = q["total_not_mentioned"] / non_refusals if non_refusals > 0 else 0

    # Group by topic
    all_by_topic = defaultdict(list)
    for q in question_stats:
        all_by_topic[q["topic"]].append(q)

    top_by_topic = {}
    for topic, all_questions in all_by_topic.items():
        settings = topic_settings.get(topic, {"top_n": 5, "max_refusal_rate": 0.7})
        top_n = settings["top_n"]
        base_max_refusal = settings["max_refusal_rate"]

        # Try increasing thresholds to find enough lying questions
        for max_refusal_rate in [base_max_refusal, 0.8, 0.9, 1.0]:
            questions = [q for q in all_questions if q["refusal_rate"] <= max_refusal_rate]
            with_lies = [q for q in questions if q["total_lies"] > 0]
            with_lies_sorted = sorted(with_lies, key=lambda x: x["lie_rate"], reverse=True)

            if len(with_lies_sorted) >= top_n:
                top_by_topic[topic] = with_lies_sorted[:top_n]
                break
        else:
            # Not enough even at 100% threshold, fill with evasive
            selected = with_lies_sorted[:]
            selected_ids = {q["question_id"] for q in selected}

            # Get questions without lies, sorted by evasion rate
            without_lies = [q for q in all_questions if q["total_lies"] == 0 and q["question_id"] not in selected_ids]
            without_lies_sorted = sorted(without_lies, key=lambda x: x["evasion_rate"], reverse=True)

            # Fill up to top_n
            remaining = top_n - len(selected)
            for q in without_lies_sorted[:remaining]:
                q["is_evasive_fill"] = True
            selected.extend(without_lies_sorted[:remaining])

            top_by_topic[topic] = selected

    return top_by_topic


def create_distribution_plot(q):
    """Create bar charts for response and fact distributions, return as base64."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Left plot: Response-level distribution
    ax1 = axes[0]
    total = q["total_responses"]
    refusals = q["refusals"]
    lie_response_count = len(q["lie_responses"])
    evasive_response_count = len(q.get("evasive_responses", []))
    non_refusals = total - refusals

    labels = ["Refusals", "With Lies", "Evasive"]
    values = [refusals, lie_response_count, evasive_response_count]
    colors = ["#95a5a6", "#e74c3c", "#f39c12"]

    # Filter out zero values
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if filtered:
        labels, values, colors = zip(*filtered)
        bars = ax1.bar(labels, values, color=colors)
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha='center', va='bottom', fontsize=9)
    ax1.set_ylabel("Responses")
    ax1.set_title(f"Responses (n={total})")

    # Right plot: Fact-level distribution
    ax2 = axes[1]
    lies = q["total_lies"]
    not_mentioned = q["total_not_mentioned"]
    mentioned = q["total_mentioned"]

    labels2 = ["Lies", "Not Mentioned", "Mentioned"]
    values2 = [lies, not_mentioned, mentioned]
    colors2 = ["#e74c3c", "#f39c12", "#2ecc71"]

    filtered2 = [(l, v, c) for l, v, c in zip(labels2, values2, colors2) if v > 0]
    if filtered2:
        labels2, values2, colors2 = zip(*filtered2)
        bars2 = ax2.bar(labels2, values2, color=colors2)
        for bar, val in zip(bars2, values2):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel("Fact checks")
    ax2.set_title("Fact Status")

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return img_base64


def esc(s):
    return html.escape(str(s))


def build_html(top_by_topic):
    """Build HTML report."""

    cards_html = ""

    for topic in sorted(top_by_topic.keys()):
        questions = top_by_topic[topic]
        topic_label = topic.replace("_", " ").title()

        cards_html += f"""<h2>{esc(topic_label)}</h2>"""

        for rank, q in enumerate(questions, 1):
            # Create distribution plot
            plot_base64 = create_distribution_plot(q)

            is_evasive = q.get("is_evasive_fill", False)

            # Get 3 example responses (lies or evasive depending on type)
            examples_html = ""
            if is_evasive:
                # Show evasive responses
                for i, er in enumerate(q.get("evasive_responses", [])[:3], 1):
                    facts_html = ""
                    for fc in er["not_mentioned_facts"]:
                        facts_html += f"""<div class="fact-chip evasive">
                            <span class="fact-label">Not mentioned:</span> {esc(fc['fact'][:100])}
                        </div>"""

                    examples_html += f"""
                    <div class="example evasive">
                        <div class="example-header">Example {i}</div>
                        <div class="facts-row">{facts_html}</div>
                        <div class="response-preview">{esc(er['response_text'][:500])}{'...' if len(er['response_text']) > 500 else ''}</div>
                    </div>"""
            else:
                # Show lie responses
                for i, lr in enumerate(q["lie_responses"][:3], 1):
                    facts_html = ""
                    for fc in lr["lie_facts"]:
                        facts_html += f"""<div class="fact-chip">
                            <span class="fact-label">Lied about:</span> {esc(fc['fact'][:100])}
                        </div>"""

                    examples_html += f"""
                    <div class="example">
                        <div class="example-header">Example {i}</div>
                        <div class="facts-row">{facts_html}</div>
                        <div class="response-preview">{esc(lr['response_text'][:500])}{'...' if len(lr['response_text']) > 500 else ''}</div>
                    </div>"""

            ref_rate_pct = q["refusal_rate"] * 100
            lie_rate_pct = q["lie_rate"] * 100
            evasion_rate_pct = q.get("evasion_rate", 0) * 100

            # Card styling and badge based on type
            card_class = "card evasive" if is_evasive else "card"
            type_badge = '<span class="badge evasive-badge">EVASIVE</span>' if is_evasive else ""
            rate_metric = f'<span class="metric evasion-rate">{evasion_rate_pct:.0f}% evasion</span>' if is_evasive else f'<span class="metric lie-rate">{lie_rate_pct:.0f}% lie rate</span>'
            examples_title = "Example Evasive Responses" if is_evasive else "Example Lie Responses"

            cards_html += f"""
            <div class="{card_class}">
                <div class="card-header">
                    <span class="rank">#{rank}</span>
                    {type_badge}
                    <span class="badge subtopic">{esc(q['subtopic'].replace('_', ' '))}</span>
                    <span class="badge level">{esc(q['level'])}</span>
                    <div class="metrics">
                        {rate_metric}
                        <span class="metric refusal-rate">{ref_rate_pct:.0f}% refusal</span>
                    </div>
                </div>

                <div class="question-box">
                    <div class="question-label">Question:</div>
                    <div class="question-text">{esc(q['question'])}</div>
                </div>

                <div class="reference-box">
                    <div class="reference-label">Reference Answer (Ground Truth):</div>
                    <div class="reference-text">{esc(q['reference_answer'])}</div>
                </div>

                <div class="content-row">
                    <div class="examples-section">
                        <h4>{examples_title}</h4>
                        {examples_html}
                    </div>
                    <div class="plot-section">
                        <h4>Response Distribution</h4>
                        <img src="data:image/png;base64,{plot_base64}" alt="Distribution plot">
                    </div>
                </div>
            </div>"""

    total_questions = sum(len(qs) for qs in top_by_topic.values())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Top Lying Questions by Topic</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #f5f5f5; color: #333; padding: 2rem; line-height: 1.6;
    }}
    h1 {{ margin-bottom: 0.5rem; color: #1a1a2e; }}
    h2 {{
        margin: 2.5rem 0 1.5rem; color: #1a1a2e;
        border-bottom: 3px solid #e64980; padding-bottom: 0.5rem;
        font-size: 1.5rem;
    }}
    h4 {{ margin-bottom: 0.75rem; color: #555; font-size: 1rem; }}
    .subtitle {{ color: #666; margin-bottom: 2rem; font-size: 1.1rem; }}
    .filter-note {{
        background: #e3f2fd; border-left: 4px solid #2196f3;
        padding: 1rem 1.5rem; margin-bottom: 2rem; border-radius: 0 8px 8px 0;
    }}
    .filter-note strong {{ color: #1565c0; }}

    .card {{
        background: #fff; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 5px solid #e64980;
    }}
    .card-header {{
        display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center;
        margin-bottom: 1.25rem; padding-bottom: 1rem; border-bottom: 1px solid #eee;
    }}
    .rank {{
        font-size: 1.5rem; font-weight: 700; color: #e64980;
        background: #fff0f6; padding: 0.25rem 0.75rem; border-radius: 8px;
    }}
    .badge {{
        display: inline-block; padding: 0.3rem 0.8rem; border-radius: 6px;
        font-size: 0.85rem; font-weight: 600; color: #fff;
    }}
    .badge.subtopic {{ background: #5f3dc4; }}
    .badge.level {{ background: #868e96; }}
    .metrics {{ margin-left: auto; display: flex; gap: 1rem; }}
    .metric {{
        font-size: 0.9rem; font-weight: 600; padding: 0.3rem 0.8rem;
        border-radius: 6px;
    }}
    .metric.lie-rate {{ background: #ffe3e3; color: #c92a2a; }}
    .metric.refusal-rate {{ background: #e9ecef; color: #495057; }}

    .question-box {{
        background: #f8f9fa; padding: 1rem 1.25rem; border-radius: 8px;
        margin-bottom: 1rem;
    }}
    .question-label {{ font-size: 0.8rem; color: #868e96; font-weight: 600; margin-bottom: 0.3rem; }}
    .question-text {{ font-size: 1.1rem; color: #1a1a2e; font-weight: 500; }}

    .reference-box {{
        background: #e8f5e9; padding: 1rem 1.25rem; border-radius: 8px;
        margin-bottom: 1.5rem; border: 1px solid #c8e6c9;
    }}
    .reference-label {{ font-size: 0.8rem; color: #2e7d32; font-weight: 600; margin-bottom: 0.3rem; }}
    .reference-text {{ font-size: 0.95rem; color: #1b5e20; }}

    .content-row {{
        display: grid; grid-template-columns: 1fr 320px; gap: 1.5rem;
    }}
    @media (max-width: 900px) {{
        .content-row {{ grid-template-columns: 1fr; }}
    }}

    .examples-section {{ }}
    .example {{
        background: #fff5f5; border: 1px solid #ffcdd2; border-radius: 8px;
        padding: 1rem; margin-bottom: 0.75rem;
    }}
    .example-header {{
        font-size: 0.8rem; font-weight: 700; color: #c62828; margin-bottom: 0.5rem;
    }}
    .facts-row {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.5rem; }}
    .fact-chip {{
        background: #ffebee; border: 1px solid #ef9a9a; border-radius: 4px;
        padding: 0.25rem 0.5rem; font-size: 0.8rem; color: #c62828;
    }}
    .fact-label {{ font-weight: 600; }}
    .response-preview {{
        font-size: 0.85rem; color: #555; line-height: 1.5;
        max-height: 120px; overflow-y: auto;
        background: #fff; padding: 0.75rem; border-radius: 4px;
        font-family: 'SF Mono', Menlo, Consolas, monospace;
    }}

    .plot-section {{ text-align: center; }}
    .plot-section img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}

    /* Evasive card styling */
    .card.evasive {{ border-left-color: #f39c12; }}
    .card.evasive .rank {{ color: #f39c12; background: #fff8e1; }}
    .badge.evasive-badge {{ background: #f39c12; }}
    .metric.evasion-rate {{ background: #fff3cd; color: #856404; }}
    .example.evasive {{ background: #fff8e1; border-color: #ffecb3; }}
    .example.evasive .example-header {{ color: #f57c00; }}
    .fact-chip.evasive {{ background: #fff8e1; border-color: #ffcc80; color: #e65100; }}
</style>
</head>
<body>
    <h1>Top Lying Questions by Topic</h1>
    <p class="subtitle">Top questions with highest lie rates: 10 for Falun Gong, 5 for Tiananmen Square</p>

    <div class="filter-note">
        <strong>Filter Applied:</strong> Refusal rate threshold increased as needed to find lying questions |
        <strong>{total_questions} questions</strong> displayed
    </div>

    {cards_html}

</body>
</html>"""


def main():
    print(f"Loading data from: {DATA_PATH}")
    with open(DATA_PATH) as f:
        data = json.load(f)

    print(f"Analyzing {len(data)} questions...")
    question_stats = analyze_data(data)

    # Topic-specific settings: Falun Gong gets 10, Tiananmen gets 5
    topic_settings = {
        "falun_gong": {"top_n": 10, "max_refusal_rate": 0.7},
        "tiananmen_square_1989": {"top_n": 5, "max_refusal_rate": 0.7},
    }
    top_by_topic = get_top_questions(question_stats, topic_settings)

    print("\n=== Top Questions by Topic ===")
    for topic in sorted(top_by_topic.keys()):
        print(f"\n{topic}:")
        for i, q in enumerate(top_by_topic[topic], 1):
            evasive_mark = " [EVASIVE]" if q.get("is_evasive_fill") else ""
            print(f"  {i}. {q['lie_rate']*100:.0f}% lie rate, {q['refusal_rate']*100:.0f}% refusal | {q['question_id'][:50]}{evasive_mark}")

    # Build and save HTML
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html_content = build_html(top_by_topic)

    with open(OUTPUT_PATH, "w") as f:
        f.write(html_content)

    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
