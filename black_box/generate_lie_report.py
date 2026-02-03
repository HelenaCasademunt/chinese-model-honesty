"""Generate an HTML report of all lies found in prompt test evaluations."""

import json
import html
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "results" / "data" / "prompt_tests_evaluated"
OUTPUT_PATH = Path(__file__).parent.parent / "plots" / "prompt_tests" / "lie_analysis.html"

MODEL_LABELS = {
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gpt-5.2": "GPT-5.2",
}
TOPIC_LABELS = {
    "tiananmen_square_1989": "Tiananmen Square 1989",
    "falun_gong": "Falun Gong",
}
VARIANT_LABELS = {
    "baseline": "Baseline",
    "concise": "Concise",
    "detailed": "Detailed",
    "adversarial": "Adversarial",
}


def extract_lies():
    lies = []
    for f in sorted(DATA_DIR.rglob("*.json")):
        rel = f.relative_to(DATA_DIR)
        parts = rel.parts
        model, topic, variant = parts[0], parts[1], parts[2].replace(".json", "")

        with open(f) as fh:
            data = json.load(fh)

        for item in data:
            for resp_idx, resp in enumerate(item.get("model_responses", [])):
                ev = resp.get("evaluation", {})
                lie_facts = [
                    fc for fc in ev.get("fact_checks", [])
                    if fc.get("status") == "lie"
                ]
                if not lie_facts:
                    continue

                lies.append({
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "topic": topic,
                    "topic_label": TOPIC_LABELS.get(topic, topic),
                    "variant": variant,
                    "variant_label": VARIANT_LABELS.get(variant, variant),
                    "question_id": item.get("question_id", ""),
                    "question": item.get("question", ""),
                    "level": item.get("level", ""),
                    "resp_idx": resp_idx,
                    "response_text": resp.get("response_text", ""),
                    "lie_facts": lie_facts,
                })
    return lies


def build_html(lies):
    # Group by model -> topic -> variant
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for lie in lies:
        grouped[lie["model"]][lie["topic"]][lie["variant"]].append(lie)

    total_lies = sum(len(l["lie_facts"]) for l in lies)
    total_responses = len(lies)

    # Summary stats
    summary_rows = []
    for model in sorted(grouped):
        for topic in sorted(grouped[model]):
            for variant in sorted(grouped[model][topic]):
                items = grouped[model][topic][variant]
                n_facts = sum(len(l["lie_facts"]) for l in items)
                n_responses = len(items)
                summary_rows.append({
                    "model": MODEL_LABELS.get(model, model),
                    "topic": TOPIC_LABELS.get(topic, topic),
                    "variant": VARIANT_LABELS.get(variant, variant),
                    "n_responses": n_responses,
                    "n_lies": n_facts,
                })

    summary_html = ""
    for r in summary_rows:
        summary_html += f"""<tr>
            <td>{esc(r['model'])}</td>
            <td>{esc(r['topic'])}</td>
            <td>{esc(r['variant'])}</td>
            <td>{r['n_responses']}</td>
            <td>{r['n_lies']}</td>
        </tr>"""

    # Detail cards
    cards_html = ""
    card_idx = 0
    for model in sorted(grouped):
        for topic in sorted(grouped[model]):
            for variant in sorted(grouped[model][topic]):
                items = grouped[model][topic][variant]
                for lie in items:
                    card_idx += 1
                    facts_html = ""
                    for fc in lie["lie_facts"]:
                        facts_html += f"""<div class="fact-item">
                            <div class="fact-label">Contradicted Fact (ground truth):</div>
                            <div class="fact-text">{esc(fc['fact'])}</div>
                            <div class="evaluator-note">Evaluator response: <code>{esc(fc.get('raw_response', ''))}</code></div>
                        </div>"""

                    resp_id = f"resp-{card_idx}"
                    cards_html += f"""
                    <div class="card">
                        <div class="card-header">
                            <span class="badge model">{esc(lie['model_label'])}</span>
                            <span class="badge topic">{esc(lie['topic_label'])}</span>
                            <span class="badge variant">{esc(lie['variant_label'])}</span>
                            <span class="badge level">{esc(lie['level'])}</span>
                            <span class="lie-count">{len(lie['lie_facts'])} lie(s)</span>
                        </div>
                        <div class="question">
                            <strong>Question:</strong> {esc(lie['question'])}
                        </div>
                        <div class="facts-section">
                            <h4>Facts Contradicted</h4>
                            {facts_html}
                        </div>
                        <div class="response-section">
                            <button class="toggle-btn" onclick="toggleResponse('{resp_id}')">Show/Hide Model Response</button>
                            <div class="response-text" id="{resp_id}" style="display:none;">
                                <pre>{esc(lie['response_text'])}</pre>
                            </div>
                        </div>
                    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lie Analysis — Prompt Test Evaluations</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #f5f5f5; color: #333; padding: 2rem; line-height: 1.6;
    }}
    h1 {{ margin-bottom: 0.5rem; color: #1a1a2e; }}
    h2 {{ margin: 2rem 0 1rem; color: #1a1a2e; border-bottom: 2px solid #e64980; padding-bottom: 0.3rem; }}
    h4 {{ margin-bottom: 0.5rem; color: #555; }}
    .subtitle {{ color: #666; margin-bottom: 2rem; font-size: 1.1rem; }}

    /* Summary table */
    table {{ border-collapse: collapse; width: 100%; max-width: 800px; margin-bottom: 2rem; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    th {{ background: #1a1a2e; color: #fff; padding: 0.75rem 1rem; text-align: left; font-weight: 600; }}
    td {{ padding: 0.6rem 1rem; border-bottom: 1px solid #eee; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #f9f0ff; }}

    /* Cards */
    .card {{
        background: #fff; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 4px solid #e64980;
    }}
    .card-header {{ display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin-bottom: 1rem; }}
    .badge {{
        display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px;
        font-size: 0.8rem; font-weight: 600; color: #fff;
    }}
    .badge.model {{ background: #364fc7; }}
    .badge.topic {{ background: #0b7285; }}
    .badge.variant {{ background: #5f3dc4; }}
    .badge.level {{ background: #868e96; }}
    .lie-count {{ color: #e64980; font-weight: 700; font-size: 0.9rem; margin-left: auto; }}

    .question {{ background: #f8f9fa; padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 1rem; font-size: 1rem; }}

    .facts-section {{ margin-bottom: 1rem; }}
    .fact-item {{
        background: #fff5f5; border: 1px solid #ffc9c9; border-radius: 6px;
        padding: 0.8rem 1rem; margin-bottom: 0.5rem;
    }}
    .fact-label {{ font-size: 0.8rem; color: #c92a2a; font-weight: 600; margin-bottom: 0.2rem; }}
    .fact-text {{ font-size: 0.95rem; color: #333; }}
    .evaluator-note {{ font-size: 0.75rem; color: #868e96; margin-top: 0.3rem; }}
    .evaluator-note code {{ background: #f1f3f5; padding: 0.1rem 0.3rem; border-radius: 3px; }}

    .toggle-btn {{
        background: #364fc7; color: #fff; border: none; padding: 0.4rem 1rem;
        border-radius: 4px; cursor: pointer; font-size: 0.85rem;
    }}
    .toggle-btn:hover {{ background: #2b3ea0; }}
    .response-text {{
        margin-top: 0.8rem; background: #f8f9fa; border: 1px solid #dee2e6;
        border-radius: 6px; padding: 1rem; max-height: 500px; overflow-y: auto;
    }}
    .response-text pre {{
        white-space: pre-wrap; word-wrap: break-word; font-size: 0.85rem;
        font-family: 'SF Mono', Menlo, Consolas, monospace; line-height: 1.5;
    }}

    .stats {{ display: flex; gap: 2rem; margin-bottom: 2rem; }}
    .stat-box {{
        background: #fff; padding: 1rem 1.5rem; border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center;
    }}
    .stat-number {{ font-size: 2rem; font-weight: 700; color: #e64980; }}
    .stat-label {{ font-size: 0.85rem; color: #666; }}
</style>
</head>
<body>
    <h1>Lie Analysis Report</h1>
    <p class="subtitle">Analysis of model responses that contradict ground truth facts in prompt test evaluations</p>

    <div class="stats">
        <div class="stat-box">
            <div class="stat-number">{total_lies}</div>
            <div class="stat-label">Total Fact Contradictions</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{total_responses}</div>
            <div class="stat-label">Responses Containing Lies</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{len(summary_rows)}</div>
            <div class="stat-label">Model/Topic/Variant Combos with Lies</div>
        </div>
    </div>

    <h2>Summary</h2>
    <table>
        <tr><th>Model</th><th>Topic</th><th>Variant</th><th>Responses w/ Lies</th><th>Fact Contradictions</th></tr>
        {summary_html}
    </table>

    <h2>Detailed Lie Analysis</h2>
    {cards_html}

    <script>
    function toggleResponse(id) {{
        var el = document.getElementById(id);
        el.style.display = el.style.display === 'none' ? 'block' : 'none';
    }}
    </script>
</body>
</html>"""


def esc(s):
    return html.escape(str(s))


def main():
    lies = extract_lies()
    print(f"Found {len(lies)} responses containing lies")
    print(f"Total fact contradictions: {sum(len(l['lie_facts']) for l in lies)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html_content = build_html(lies)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html_content)
    print(f"Report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
