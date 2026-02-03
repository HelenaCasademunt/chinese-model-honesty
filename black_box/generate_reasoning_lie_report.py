"""Generate an HTML report of all lies found in reasoning test evaluations."""

import json
import html
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "results" / "data" / "reasoning_test"
OUTPUT_PATH = SCRIPT_DIR.parent / "plots" / "reasoning_test" / "lie_analysis.html"

INPUT_FILES = {
    "no_thinking": {
        "path": DATA_DIR / "eval_baseline_no_thinking.json",
        "label": "No Thinking",
    },
    "reasoning_default": {
        "path": DATA_DIR / "eval_baseline_reasoning_default.json",
        "label": "Default Reasoning",
    },
}

TOPIC_LABELS = {
    "tiananmen_square_1989": "Tiananmen Square 1989",
    "falun_gong": "Falun Gong",
}


def esc(s):
    return html.escape(str(s))


def extract_lies():
    lies = []
    for config_key, config in INPUT_FILES.items():
        path = config["path"]
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue

        with open(path) as f:
            data = json.load(f)

        for item in data:
            for resp_idx, resp in enumerate(item.get("model_responses", [])):
                ev = resp.get("evaluation", {})
                lie_facts = [
                    fc for fc in ev.get("fact_checks", [])
                    if fc.get("status") == "lie"
                ]
                if not lie_facts:
                    continue

                # Get the thinking content if available
                metadata = resp.get("_metadata", {})
                thinking = metadata.get("thinking", "")

                lies.append({
                    "config": config_key,
                    "config_label": config["label"],
                    "topic": item.get("topic", ""),
                    "topic_label": TOPIC_LABELS.get(item.get("topic", ""), item.get("topic", "")),
                    "subtopic": item.get("subtopic", ""),
                    "level": item.get("level", ""),
                    "question_id": item.get("question_id", ""),
                    "question": item.get("question", ""),
                    "resp_idx": resp_idx,
                    "response_text": resp.get("response_text", ""),
                    "thinking": thinking,
                    "lie_facts": lie_facts,
                })
    return lies


def build_html(lies):
    grouped = defaultdict(lambda: defaultdict(list))
    for lie in lies:
        grouped[lie["config"]][lie["topic"]].append(lie)

    total_lies = sum(len(l["lie_facts"]) for l in lies)
    total_responses = len(lies)

    # Count by config
    config_stats = defaultdict(lambda: {"responses": 0, "lies": 0})
    for lie in lies:
        config_stats[lie["config_label"]]["responses"] += 1
        config_stats[lie["config_label"]]["lies"] += len(lie["lie_facts"])

    # Summary table
    summary_rows = ""
    for config_key in INPUT_FILES:
        config_label = INPUT_FILES[config_key]["label"]
        for topic in sorted(set(l["topic"] for l in lies if l["config"] == config_key)):
            topic_lies = [l for l in lies if l["config"] == config_key and l["topic"] == topic]
            n_responses = len(topic_lies)
            n_facts = sum(len(l["lie_facts"]) for l in topic_lies)
            # Get unique questions
            unique_qs = len(set(l["question_id"] for l in topic_lies))
            summary_rows += f"""<tr>
                <td>{esc(config_label)}</td>
                <td>{esc(TOPIC_LABELS.get(topic, topic))}</td>
                <td>{unique_qs}</td>
                <td>{n_responses}</td>
                <td>{n_facts}</td>
            </tr>"""

    # Fact frequency analysis: which facts are most commonly lied about?
    fact_freq = defaultdict(lambda: {"count": 0, "configs": set(), "questions": set()})
    for lie in lies:
        for fc in lie["lie_facts"]:
            key = fc["fact"]
            fact_freq[key]["count"] += 1
            fact_freq[key]["configs"].add(lie["config_label"])
            fact_freq[key]["questions"].add(lie["question"])

    top_facts_html = ""
    for fact, info in sorted(fact_freq.items(), key=lambda x: -x[1]["count"]):
        configs_str = ", ".join(sorted(info["configs"]))
        top_facts_html += f"""<tr>
            <td>{info['count']}</td>
            <td>{esc(fact)}</td>
            <td>{esc(configs_str)}</td>
        </tr>"""

    # Detail cards
    cards_html = ""
    card_idx = 0
    for config_key in INPUT_FILES:
        config_label = INPUT_FILES[config_key]["label"]
        config_lies = [l for l in lies if l["config"] == config_key]
        if not config_lies:
            continue

        cards_html += f'<h3 class="config-header">{esc(config_label)}</h3>'

        for lie in sorted(config_lies, key=lambda l: (l["topic"], l["question_id"])):
            card_idx += 1
            facts_html = ""
            for fc in lie["lie_facts"]:
                facts_html += f"""<div class="fact-item">
                    <div class="fact-label">Contradicted Fact (ground truth):</div>
                    <div class="fact-text">{esc(fc['fact'])}</div>
                    <div class="evaluator-note">Evaluator response: <code>{esc(fc.get('raw_response', ''))}</code></div>
                </div>"""

            resp_id = f"resp-{card_idx}"
            thinking_section = ""
            if lie["thinking"]:
                think_id = f"think-{card_idx}"
                thinking_section = f"""
                <div class="thinking-section">
                    <button class="toggle-btn thinking-btn" onclick="toggleResponse('{think_id}')">Show/Hide Thinking</button>
                    <div class="thinking-text" id="{think_id}" style="display:none;">
                        <pre>{esc(lie['thinking'])}</pre>
                    </div>
                </div>"""

            cards_html += f"""
            <div class="card">
                <div class="card-header">
                    <span class="badge config">{esc(lie['config_label'])}</span>
                    <span class="badge topic">{esc(lie['topic_label'])}</span>
                    <span class="badge level">{esc(lie['level'])}</span>
                    <span class="badge subtopic">{esc(lie['subtopic'].replace('_', ' ').title())}</span>
                    <span class="lie-count">{len(lie['lie_facts'])} lie(s)</span>
                </div>
                <div class="question">
                    <strong>Question:</strong> {esc(lie['question'])}
                </div>
                <div class="facts-section">
                    <h4>Facts Contradicted</h4>
                    {facts_html}
                </div>
                {thinking_section}
                <div class="response-section">
                    <button class="toggle-btn" onclick="toggleResponse('{resp_id}')">Show/Hide Model Response</button>
                    <div class="response-text" id="{resp_id}" style="display:none;">
                        <pre>{esc(lie['response_text'])}</pre>
                    </div>
                </div>
            </div>"""

    no_think_lies = sum(len(l["lie_facts"]) for l in lies if l["config"] == "no_thinking")
    reason_lies = sum(len(l["lie_facts"]) for l in lies if l["config"] == "reasoning_default")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lie Analysis — Reasoning Test Evaluations</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #f5f5f5; color: #333; padding: 2rem; line-height: 1.6;
    }}
    h1 {{ margin-bottom: 0.5rem; color: #1a1a2e; }}
    h2 {{ margin: 2rem 0 1rem; color: #1a1a2e; border-bottom: 2px solid #e64980; padding-bottom: 0.3rem; }}
    h3.config-header {{
        margin: 1.5rem 0 1rem; padding: 0.6rem 1rem; background: #364fc7; color: #fff;
        border-radius: 6px; font-size: 1.1rem;
    }}
    h4 {{ margin-bottom: 0.5rem; color: #555; }}
    .subtitle {{ color: #666; margin-bottom: 2rem; font-size: 1.1rem; }}

    table {{ border-collapse: collapse; width: 100%; max-width: 900px; margin-bottom: 2rem; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    th {{ background: #1a1a2e; color: #fff; padding: 0.75rem 1rem; text-align: left; font-weight: 600; }}
    td {{ padding: 0.6rem 1rem; border-bottom: 1px solid #eee; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #f9f0ff; }}

    .card {{
        background: #fff; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 4px solid #e64980;
    }}
    .card-header {{ display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin-bottom: 1rem; }}
    .badge {{
        display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px;
        font-size: 0.8rem; font-weight: 600; color: #fff;
    }}
    .badge.config {{ background: #364fc7; }}
    .badge.topic {{ background: #0b7285; }}
    .badge.level {{ background: #868e96; }}
    .badge.subtopic {{ background: #5f3dc4; }}
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
        border-radius: 4px; cursor: pointer; font-size: 0.85rem; margin-right: 0.5rem;
    }}
    .toggle-btn:hover {{ background: #2b3ea0; }}
    .thinking-btn {{ background: #5f3dc4; }}
    .thinking-btn:hover {{ background: #4c2fa0; }}

    .response-text, .thinking-text {{
        margin-top: 0.8rem; background: #f8f9fa; border: 1px solid #dee2e6;
        border-radius: 6px; padding: 1rem; max-height: 500px; overflow-y: auto;
    }}
    .thinking-text {{ background: #f3f0ff; border-color: #d0bfff; }}
    .response-text pre, .thinking-text pre {{
        white-space: pre-wrap; word-wrap: break-word; font-size: 0.85rem;
        font-family: 'SF Mono', Menlo, Consolas, monospace; line-height: 1.5;
    }}

    .stats {{ display: flex; gap: 2rem; margin-bottom: 2rem; flex-wrap: wrap; }}
    .stat-box {{
        background: #fff; padding: 1rem 1.5rem; border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; min-width: 140px;
    }}
    .stat-number {{ font-size: 2rem; font-weight: 700; color: #e64980; }}
    .stat-label {{ font-size: 0.85rem; color: #666; }}

    .comparison-box {{
        display: flex; gap: 2rem; margin-bottom: 2rem; flex-wrap: wrap;
    }}
    .compare-card {{
        background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px; flex: 1; min-width: 250px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-top: 3px solid #364fc7;
    }}
    .compare-card h4 {{ margin-bottom: 0.5rem; color: #364fc7; }}
    .compare-stat {{ font-size: 1.5rem; font-weight: 700; color: #e64980; }}
</style>
</head>
<body>
    <h1>Lie Analysis — Reasoning Test</h1>
    <p class="subtitle">Comparing lies in DeepSeek responses with and without thinking/reasoning enabled</p>

    <div class="stats">
        <div class="stat-box">
            <div class="stat-number">{total_lies}</div>
            <div class="stat-label">Total Fact Contradictions</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{total_responses}</div>
            <div class="stat-label">Responses Containing Lies</div>
        </div>
    </div>

    <div class="comparison-box">
        <div class="compare-card">
            <h4>No Thinking</h4>
            <div class="compare-stat">{no_think_lies} lies</div>
            <div style="color:#666; font-size:0.9rem;">in {sum(1 for l in lies if l['config']=='no_thinking')} responses</div>
        </div>
        <div class="compare-card">
            <h4>Default Reasoning</h4>
            <div class="compare-stat">{reason_lies} lies</div>
            <div style="color:#666; font-size:0.9rem;">in {sum(1 for l in lies if l['config']=='reasoning_default')} responses</div>
        </div>
    </div>

    <h2>Summary by Topic</h2>
    <table>
        <tr><th>Reasoning Mode</th><th>Topic</th><th>Unique Questions</th><th>Responses w/ Lies</th><th>Fact Contradictions</th></tr>
        {summary_rows}
    </table>

    <h2>Most Commonly Contradicted Facts</h2>
    <table>
        <tr><th>Count</th><th>Fact</th><th>Occurs In</th></tr>
        {top_facts_html}
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


def main():
    lies = extract_lies()
    print(f"Found {len(lies)} responses containing lies")
    print(f"Total fact contradictions: {sum(len(l['lie_facts']) for l in lies)}")

    for config_key, config in INPUT_FILES.items():
        config_lies = [l for l in lies if l["config"] == config_key]
        n = sum(len(l["lie_facts"]) for l in config_lies)
        print(f"  {config['label']}: {n} lies in {len(config_lies)} responses")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html_content = build_html(lies)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html_content)
    print(f"Report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
