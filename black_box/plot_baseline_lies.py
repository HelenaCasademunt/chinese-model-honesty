"""Plot lie distribution for baseline_responses_no_thinking.json."""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

DATA_PATH = Path(__file__).parent.parent / "results" / "data" / "baseline_responses_no_thinking.json"
OUTPUT_DIR = Path(__file__).parent.parent / "plots"


def main():
    print(f"Loading data from: {DATA_PATH}")
    with open(DATA_PATH) as f:
        data = json.load(f)

    # Collect stats per question
    question_stats = []
    for item in data:
        q_id = item.get("question_id", "")
        topic = item.get("topic", "")
        subtopic = item.get("subtopic", "")
        level = item.get("level", "")

        responses = item.get("model_responses", [])
        total = len(responses)
        refusals = 0
        lies = 0
        not_mentioned = 0
        mentioned = 0

        for resp in responses:
            ev = resp.get("evaluation", {})
            if ev.get("is_refusal", False):
                refusals += 1
            else:
                for fc in ev.get("fact_checks", []):
                    status = fc.get("status", "")
                    if status == "lie":
                        lies += 1
                    elif status == "not_mentioned":
                        not_mentioned += 1
                    elif status == "mentioned":
                        mentioned += 1

        question_stats.append({
            "question_id": q_id,
            "topic": topic,
            "subtopic": subtopic,
            "level": level,
            "total": total,
            "refusals": refusals,
            "lies": lies,
            "not_mentioned": not_mentioned,
            "mentioned": mentioned,
        })

    # Sort by lies descending
    question_stats.sort(key=lambda x: x["lies"], reverse=True)

    # Aggregate by topic
    topic_stats = defaultdict(lambda: {"lies": 0, "refusals": 0, "not_mentioned": 0, "mentioned": 0, "total": 0})
    for q in question_stats:
        t = q["topic"]
        topic_stats[t]["lies"] += q["lies"]
        topic_stats[t]["refusals"] += q["refusals"]
        topic_stats[t]["not_mentioned"] += q["not_mentioned"]
        topic_stats[t]["mentioned"] += q["mentioned"]
        topic_stats[t]["total"] += q["total"]

    # Aggregate by subtopic
    subtopic_stats = defaultdict(lambda: {"lies": 0, "refusals": 0, "total": 0, "topic": ""})
    for q in question_stats:
        st = q["subtopic"]
        subtopic_stats[st]["lies"] += q["lies"]
        subtopic_stats[st]["refusals"] += q["refusals"]
        subtopic_stats[st]["total"] += q["total"]
        subtopic_stats[st]["topic"] = q["topic"]

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Plot 1: Lies per question (horizontal bar)
    ax1 = axes[0, 0]
    short_labels = []
    for q in question_stats:
        label = q["question_id"].replace("tiananmen_square_1989_", "ts_").replace("falun_gong_", "fg_")
        if len(label) > 35:
            label = label[:32] + "..."
        short_labels.append(label)
    lies_counts = [q["lies"] for q in question_stats]
    colors = ["#e74c3c" if q["topic"] == "tiananmen_square_1989" else "#3498db" for q in question_stats]

    y_pos = range(len(question_stats))
    ax1.barh(y_pos, lies_counts, color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(short_labels, fontsize=7)
    ax1.set_xlabel("Number of lies")
    ax1.set_title("Lies per Question")
    ax1.invert_yaxis()

    # Plot 2: Lies vs Refusals by topic (stacked bar)
    ax2 = axes[0, 1]
    topics = list(topic_stats.keys())
    topic_lies = [topic_stats[t]["lies"] for t in topics]
    topic_refusals = [topic_stats[t]["refusals"] for t in topics]
    topic_not_mentioned = [topic_stats[t]["not_mentioned"] for t in topics]

    x = range(len(topics))
    width = 0.25
    ax2.bar([i - width for i in x], topic_lies, width, label="Lies", color="#e74c3c")
    ax2.bar(x, topic_refusals, width, label="Refusals", color="#95a5a6")
    ax2.bar([i + width for i in x], topic_not_mentioned, width, label="Not Mentioned", color="#f39c12")
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.replace("_", " ").title()[:20] for t in topics], fontsize=9)
    ax2.set_ylabel("Count")
    ax2.set_title("Response Types by Topic")
    ax2.legend()

    # Plot 3: Lies per subtopic
    ax3 = axes[1, 0]
    sorted_subtopics = sorted(subtopic_stats.items(), key=lambda x: x[1]["lies"], reverse=True)
    st_labels = [st for st, _ in sorted_subtopics]
    st_lies = [s["lies"] for _, s in sorted_subtopics]
    st_colors = ["#e74c3c" if s["topic"] == "tiananmen_square_1989" else "#3498db" for _, s in sorted_subtopics]

    y_pos = range(len(st_labels))
    ax3.barh(y_pos, st_lies, color=st_colors)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([s.replace("_", " ")[:25] for s in st_labels], fontsize=8)
    ax3.set_xlabel("Number of lies")
    ax3.set_title("Lies by Subtopic")
    ax3.invert_yaxis()

    # Plot 4: Refusal rate vs Lie rate scatter
    ax4 = axes[1, 1]
    for q in question_stats:
        refusal_rate = q["refusals"] / q["total"] * 100 if q["total"] > 0 else 0
        # Lie rate: lies per non-refusal response
        non_refusals = q["total"] - q["refusals"]
        lie_rate = q["lies"] / non_refusals * 100 if non_refusals > 0 else 0
        color = "#e74c3c" if q["topic"] == "tiananmen_square_1989" else "#3498db"
        ax4.scatter(refusal_rate, lie_rate, c=color, alpha=0.6, s=50)

    ax4.set_xlabel("Refusal Rate (%)")
    ax4.set_ylabel("Lie Rate (% of non-refusal responses)")
    ax4.set_title("Refusal Rate vs Lie Rate per Question")
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=50, color='gray', linestyle='--', alpha=0.5)

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Tiananmen Square 1989'),
        Patch(facecolor='#3498db', label='Falun Gong')
    ]
    ax4.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "baseline_no_thinking_lie_distribution.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total questions: {len(question_stats)}")
    print(f"Total lies: {sum(q['lies'] for q in question_stats)}")
    print(f"Questions with 0 lies: {sum(1 for q in question_stats if q['lies'] == 0)}")
    print(f"Questions with >20 lies: {sum(1 for q in question_stats if q['lies'] > 20)}")


if __name__ == "__main__":
    main()
