"""Generate a lie report for baseline_responses_no_thinking.json with threshold filtering."""

import json
import html
from pathlib import Path
from collections import defaultdict
import argparse

DATA_PATH = Path(__file__).parent.parent / "results" / "data" / "baseline_responses_no_thinking.json"
OUTPUT_PATH = Path(__file__).parent.parent / "plots" / "baseline_no_thinking_lie_report.html"


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

        for resp_idx, resp in enumerate(responses):
            ev = resp.get("evaluation", {})

            if ev.get("is_refusal", False):
                refusals += 1
            else:
                fact_checks = ev.get("fact_checks", [])
                resp_lies = []
                for fc in fact_checks:
                    status = fc.get("status", "")
                    if status == "lie":
                        lies += 1
                        resp_lies.append(fc)
                    elif status == "not_mentioned":
                        not_mentioned += 1
                    elif status == "mentioned":
                        mentioned += 1

                if resp_lies:
                    lie_responses.append({
                        "resp_idx": resp_idx,
                        "response_text": resp.get("response_text", ""),
                        "lie_facts": resp_lies,
                    })

        refusal_rate = refusals / total_responses if total_responses > 0 else 0

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
            "total_not_mentioned": not_mentioned,
            "total_mentioned": mentioned,
            "lie_responses": lie_responses,
        })

    return question_stats


def filter_questions(question_stats, min_lies=1, max_refusal_rate=0.8):
    """Filter out questions with no lies or too many refusals."""
    filtered = []
    excluded_no_lies = []
    excluded_high_refusal = []

    for q in question_stats:
        if q["total_lies"] < min_lies:
            excluded_no_lies.append(q)
        elif q["refusal_rate"] > max_refusal_rate:
            excluded_high_refusal.append(q)
        else:
            filtered.append(q)

    return filtered, excluded_no_lies, excluded_high_refusal


def esc(s):
    return html.escape(str(s))


def build_html(question_stats, filtered, excluded_no_lies, excluded_high_refusal,
               min_lies, max_refusal_rate):
    """Build an HTML report."""

    # Overall stats
    total_questions = len(question_stats)
    total_responses = sum(q["total_responses"] for q in question_stats)
    total_refusals = sum(q["refusals"] for q in question_stats)
    total_lies = sum(q["total_lies"] for q in question_stats)
    total_not_mentioned = sum(q["total_not_mentioned"] for q in question_stats)
    total_mentioned = sum(q["total_mentioned"] for q in question_stats)

    # By topic
    topic_stats = defaultdict(lambda: {"questions": 0, "lies": 0, "refusals": 0, "responses": 0})
    for q in question_stats:
        t = q["topic"]
        topic_stats[t]["questions"] += 1
        topic_stats[t]["lies"] += q["total_lies"]
        topic_stats[t]["refusals"] += q["refusals"]
        topic_stats[t]["responses"] += q["total_responses"]

    # By subtopic
    subtopic_stats = defaultdict(lambda: {"questions": 0, "lies": 0, "refusals": 0, "responses": 0})
    for q in question_stats:
        st = f"{q['topic']}/{q['subtopic']}"
        subtopic_stats[st]["questions"] += 1
        subtopic_stats[st]["lies"] += q["total_lies"]
        subtopic_stats[st]["refusals"] += q["refusals"]
        subtopic_stats[st]["responses"] += q["total_responses"]

    # Build topic rows
    topic_rows = ""
    for topic in sorted(topic_stats.keys()):
        s = topic_stats[topic]
        ref_rate = s["refusals"] / s["responses"] * 100 if s["responses"] > 0 else 0
        topic_rows += f"""<tr>
            <td>{esc(topic)}</td>
            <td>{s['questions']}</td>
            <td>{s['responses']}</td>
            <td>{s['lies']}</td>
            <td>{s['refusals']} ({ref_rate:.1f}%)</td>
        </tr>"""

    # Build subtopic rows
    subtopic_rows = ""
    for st in sorted(subtopic_stats.keys(), key=lambda x: subtopic_stats[x]["lies"], reverse=True):
        s = subtopic_stats[st]
        ref_rate = s["refusals"] / s["responses"] * 100 if s["responses"] > 0 else 0
        subtopic_rows += f"""<tr>
            <td>{esc(st)}</td>
            <td>{s['questions']}</td>
            <td>{s['responses']}</td>
            <td>{s['lies']}</td>
            <td>{s['refusals']} ({ref_rate:.1f}%)</td>
        </tr>"""

    # Build question cards for filtered questions (sorted by lies descending)
    filtered_sorted = sorted(filtered, key=lambda x: x["total_lies"], reverse=True)
    cards_html = ""
    for idx, q in enumerate(filtered_sorted, 1):
        ref_rate_pct = q["refusal_rate"] * 100

        # Build lie details
        lie_details = ""
        for lr in q["lie_responses"]:
            facts_html = ""
            for fc in lr["lie_facts"]:
                facts_html += f"""<div class="fact-item">
                    <div class="fact-label">Contradicted Fact:</div>
                    <div class="fact-text">{esc(fc['fact'])}</div>
                    <div class="evaluator-note">Evaluator: <code>{esc(fc.get('raw_response', ''))}</code></div>
                </div>"""

            resp_id = f"resp-{idx}-{lr['resp_idx']}"
            lie_details += f"""<div class="lie-response">
                <div class="lie-response-header">Response #{lr['resp_idx'] + 1} - {len(lr['lie_facts'])} lie(s)</div>
                {facts_html}
                <button class="toggle-btn" onclick="toggleResponse('{resp_id}')">Show/Hide Response</button>
                <div class="response-text" id="{resp_id}" style="display:none;">
                    <pre>{esc(lr['response_text'])}</pre>
                </div>
            </div>"""

        ref_id = f"ref-{idx}"
        cards_html += f"""
        <div class="card">
            <div class="card-header">
                <span class="badge topic">{esc(q['topic'])}</span>
                <span class="badge subtopic">{esc(q['subtopic'])}</span>
                <span class="badge level">{esc(q['level'])}</span>
                <span class="lie-count">{q['total_lies']} lie(s)</span>
                <span class="refusal-rate">Refusal: {ref_rate_pct:.0f}%</span>
            </div>
            <div class="question">
                <strong>Question:</strong> {esc(q['question'])}
            </div>
            <div class="reference">
                <button class="toggle-btn small" onclick="toggleResponse('{ref_id}')">Show/Hide Reference Answer</button>
                <div class="response-text" id="{ref_id}" style="display:none;">
                    <pre>{esc(q['reference_answer'])}</pre>
                </div>
            </div>
            <div class="stats-row">
                <span>Responses: {q['total_responses']}</span>
                <span>Refusals: {q['refusals']}</span>
                <span>Mentioned: {q['total_mentioned']}</span>
                <span>Not Mentioned: {q['total_not_mentioned']}</span>
            </div>
            <div class="lies-section">
                <h4>Lie Details</h4>
                {lie_details}
            </div>
        </div>"""

    # Build excluded questions summary
    excluded_no_lies_html = ""
    for q in sorted(excluded_no_lies, key=lambda x: x["question_id"]):
        ref_rate_pct = q["refusal_rate"] * 100
        excluded_no_lies_html += f"""<tr>
            <td>{esc(q['question_id'][:50])}...</td>
            <td>{esc(q['topic'])}</td>
            <td>{q['total_responses']}</td>
            <td>{q['refusals']} ({ref_rate_pct:.0f}%)</td>
        </tr>"""

    excluded_high_refusal_html = ""
    for q in sorted(excluded_high_refusal, key=lambda x: x["refusal_rate"], reverse=True):
        ref_rate_pct = q["refusal_rate"] * 100
        excluded_high_refusal_html += f"""<tr>
            <td>{esc(q['question_id'][:50])}...</td>
            <td>{esc(q['topic'])}</td>
            <td>{q['total_lies']}</td>
            <td>{q['refusals']} ({ref_rate_pct:.0f}%)</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lie Analysis — Baseline No Thinking</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #f5f5f5; color: #333; padding: 2rem; line-height: 1.6;
    }}
    h1 {{ margin-bottom: 0.5rem; color: #1a1a2e; }}
    h2 {{ margin: 2rem 0 1rem; color: #1a1a2e; border-bottom: 2px solid #e64980; padding-bottom: 0.3rem; }}
    h3 {{ margin: 1.5rem 0 0.75rem; color: #333; }}
    h4 {{ margin-bottom: 0.5rem; color: #555; }}
    .subtitle {{ color: #666; margin-bottom: 2rem; font-size: 1.1rem; }}

    .stats {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 2rem; }}
    .stat-box {{
        background: #fff; padding: 1rem 1.5rem; border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; min-width: 140px;
    }}
    .stat-number {{ font-size: 1.8rem; font-weight: 700; color: #e64980; }}
    .stat-number.blue {{ color: #364fc7; }}
    .stat-number.gray {{ color: #868e96; }}
    .stat-label {{ font-size: 0.85rem; color: #666; }}

    table {{ border-collapse: collapse; width: 100%; max-width: 900px; margin-bottom: 2rem; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    th {{ background: #1a1a2e; color: #fff; padding: 0.75rem 1rem; text-align: left; font-weight: 600; }}
    td {{ padding: 0.6rem 1rem; border-bottom: 1px solid #eee; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #f9f0ff; }}

    .filter-info {{
        background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
        padding: 1rem 1.5rem; margin-bottom: 2rem;
    }}
    .filter-info h3 {{ margin-top: 0; color: #856404; }}

    .card {{
        background: #fff; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 4px solid #e64980;
    }}
    .card-header {{ display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin-bottom: 1rem; }}
    .badge {{
        display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px;
        font-size: 0.8rem; font-weight: 600; color: #fff;
    }}
    .badge.topic {{ background: #0b7285; }}
    .badge.subtopic {{ background: #5f3dc4; }}
    .badge.level {{ background: #868e96; }}
    .lie-count {{ color: #e64980; font-weight: 700; font-size: 0.9rem; margin-left: auto; }}
    .refusal-rate {{ color: #868e96; font-size: 0.85rem; margin-left: 1rem; }}

    .question {{ background: #f8f9fa; padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 1rem; font-size: 1rem; }}
    .reference {{ margin-bottom: 1rem; }}

    .stats-row {{
        display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1rem;
        font-size: 0.9rem; color: #555;
    }}

    .lies-section {{ margin-top: 1rem; }}
    .lie-response {{
        background: #fff5f5; border: 1px solid #ffc9c9; border-radius: 6px;
        padding: 1rem; margin-bottom: 0.75rem;
    }}
    .lie-response-header {{ font-weight: 600; color: #c92a2a; margin-bottom: 0.5rem; }}

    .fact-item {{
        background: #fff; border: 1px solid #ffc9c9; border-radius: 4px;
        padding: 0.6rem 0.8rem; margin-bottom: 0.5rem;
    }}
    .fact-label {{ font-size: 0.75rem; color: #c92a2a; font-weight: 600; }}
    .fact-text {{ font-size: 0.9rem; color: #333; }}
    .evaluator-note {{ font-size: 0.75rem; color: #868e96; margin-top: 0.2rem; }}
    .evaluator-note code {{ background: #f1f3f5; padding: 0.1rem 0.3rem; border-radius: 3px; }}

    .toggle-btn {{
        background: #364fc7; color: #fff; border: none; padding: 0.4rem 1rem;
        border-radius: 4px; cursor: pointer; font-size: 0.85rem; margin-top: 0.5rem;
    }}
    .toggle-btn.small {{ padding: 0.3rem 0.8rem; font-size: 0.8rem; }}
    .toggle-btn:hover {{ background: #2b3ea0; }}
    .response-text {{
        margin-top: 0.8rem; background: #f8f9fa; border: 1px solid #dee2e6;
        border-radius: 6px; padding: 1rem; max-height: 400px; overflow-y: auto;
    }}
    .response-text pre {{
        white-space: pre-wrap; word-wrap: break-word; font-size: 0.85rem;
        font-family: 'SF Mono', Menlo, Consolas, monospace; line-height: 1.5;
    }}

    .excluded-section {{ margin-top: 2rem; }}
    .excluded-section table {{ max-width: 700px; }}
</style>
</head>
<body>
    <h1>Lie Analysis Report</h1>
    <p class="subtitle">Baseline Responses (No Thinking) — Analysis of model lies and response patterns</p>

    <div class="stats">
        <div class="stat-box">
            <div class="stat-number">{total_questions}</div>
            <div class="stat-label">Total Questions</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{total_responses}</div>
            <div class="stat-label">Total Responses</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{total_lies}</div>
            <div class="stat-label">Total Lies</div>
        </div>
        <div class="stat-box">
            <div class="stat-number blue">{total_refusals}</div>
            <div class="stat-label">Total Refusals</div>
        </div>
        <div class="stat-box">
            <div class="stat-number gray">{total_not_mentioned}</div>
            <div class="stat-label">Facts Not Mentioned</div>
        </div>
        <div class="stat-box">
            <div class="stat-number gray">{total_mentioned}</div>
            <div class="stat-label">Facts Mentioned</div>
        </div>
    </div>

    <h2>By Topic</h2>
    <table>
        <tr><th>Topic</th><th>Questions</th><th>Responses</th><th>Lies</th><th>Refusals</th></tr>
        {topic_rows}
    </table>

    <h2>By Subtopic</h2>
    <table>
        <tr><th>Subtopic</th><th>Questions</th><th>Responses</th><th>Lies</th><th>Refusals</th></tr>
        {subtopic_rows}
    </table>

    <h2>Filtered Questions Analysis</h2>
    <div class="filter-info">
        <h3>Filter Criteria</h3>
        <p>Showing questions with at least <strong>{min_lies} lie(s)</strong> and refusal rate ≤ <strong>{max_refusal_rate * 100:.0f}%</strong></p>
        <p>
            <strong>{len(filtered)}</strong> questions pass filters |
            <strong>{len(excluded_no_lies)}</strong> excluded (no lies) |
            <strong>{len(excluded_high_refusal)}</strong> excluded (high refusal rate)
        </p>
    </div>

    {cards_html}

    <div class="excluded-section">
        <h2>Excluded: No Lies ({len(excluded_no_lies)} questions)</h2>
        <p style="color: #666; margin-bottom: 1rem;">Questions where the model never lied (all refusals or accurate responses)</p>
        <table>
            <tr><th>Question ID</th><th>Topic</th><th>Responses</th><th>Refusals</th></tr>
            {excluded_no_lies_html if excluded_no_lies_html else "<tr><td colspan='4'>None</td></tr>"}
        </table>

        <h2>Excluded: High Refusal Rate ({len(excluded_high_refusal)} questions)</h2>
        <p style="color: #666; margin-bottom: 1rem;">Questions where the model refused &gt; {max_refusal_rate * 100:.0f}% of the time (despite having lies)</p>
        <table>
            <tr><th>Question ID</th><th>Topic</th><th>Lies</th><th>Refusals</th></tr>
            {excluded_high_refusal_html if excluded_high_refusal_html else "<tr><td colspan='4'>None</td></tr>"}
        </table>
    </div>

    <script>
    function toggleResponse(id) {{
        var el = document.getElementById(id);
        el.style.display = el.style.display === 'none' ? 'block' : 'none';
    }}
    </script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate lie report for baseline responses")
    parser.add_argument("--input", type=str, default=str(DATA_PATH), help="Input JSON file")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Output HTML file")
    parser.add_argument("--min-lies", type=int, default=1, help="Minimum lies to include a question (default: 1)")
    parser.add_argument("--max-refusal-rate", type=float, default=0.8, help="Max refusal rate to include (default: 0.8)")
    args = parser.parse_args()

    print(f"Loading data from: {args.input}")
    with open(args.input) as f:
        data = json.load(f)

    print(f"Analyzing {len(data)} questions...")
    question_stats = analyze_data(data)

    # Print summary
    total_responses = sum(q["total_responses"] for q in question_stats)
    total_refusals = sum(q["refusals"] for q in question_stats)
    total_lies = sum(q["total_lies"] for q in question_stats)

    print(f"\n=== Overall Statistics ===")
    print(f"Questions: {len(question_stats)}")
    print(f"Total responses: {total_responses}")
    print(f"Total refusals: {total_refusals} ({total_refusals/total_responses*100:.1f}%)")
    print(f"Total lies: {total_lies}")

    # Filter
    filtered, excluded_no_lies, excluded_high_refusal = filter_questions(
        question_stats,
        min_lies=args.min_lies,
        max_refusal_rate=args.max_refusal_rate
    )

    print(f"\n=== Filtering (min_lies={args.min_lies}, max_refusal_rate={args.max_refusal_rate}) ===")
    print(f"Questions passing filter: {len(filtered)}")
    print(f"Excluded (no lies): {len(excluded_no_lies)}")
    print(f"Excluded (high refusal): {len(excluded_high_refusal)}")

    # Questions with most lies
    print(f"\n=== Top Questions by Lies (filtered) ===")
    for q in sorted(filtered, key=lambda x: x["total_lies"], reverse=True)[:10]:
        print(f"  {q['total_lies']} lies | {q['refusal_rate']*100:.0f}% refusal | {q['question_id'][:60]}")

    # Build and save HTML
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_content = build_html(
        question_stats, filtered, excluded_no_lies, excluded_high_refusal,
        args.min_lies, args.max_refusal_rate
    )

    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
