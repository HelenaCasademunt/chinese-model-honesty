import json
import sys
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/data/reasoning_test/eval_responses_no_thinking.json"
    with open(path) as f:
        data = json.load(f)

    # Collect all lie responses
    lies_by_question = []
    for q in data:
        for resp in q["model_responses"]:
            for fc in resp["evaluation"].get("fact_checks", []):
                if fc.get("status") == "lie":
                    lies_by_question.append({
                        "question_id": q["question_id"],
                        "topic": q["topic"],
                        "subtopic": q["subtopic"],
                        "level": q["level"],
                        "question": q["question"],
                        "response": resp["response_text"],
                        "lied_fact": fc["fact"],
                    })

    # Print all lie responses
    print("=" * 80)
    print(f"TOTAL LIE INSTANCES: {len(lies_by_question)}")
    print("=" * 80)
    for i, lie in enumerate(lies_by_question, 1):
        print(f"\n--- Lie #{i} ---")
        print(f"Topic:    {lie['topic']} / {lie['subtopic']} ({lie['level']})")
        print(f"Question: {lie['question']}")
        print(f"Lied fact: {lie['lied_fact']}")
        print(f"Response: {lie['response'][:300]}{'...' if len(lie['response']) > 300 else ''}")

    # Count lies per question
    question_lie_counts = Counter()
    for lie in lies_by_question:
        question_lie_counts[lie["question_id"]] += 1

    # Count lies per topic
    topic_lie_counts = Counter()
    for lie in lies_by_question:
        topic_lie_counts[lie["topic"]] += 1

    # Count lies per subtopic
    subtopic_lie_counts = Counter()
    for lie in lies_by_question:
        subtopic_lie_counts[f"{lie['topic']}/{lie['subtopic']}"] += 1

    # Print summary
    print("\n" + "=" * 80)
    print("LIES BY TOPIC")
    print("=" * 80)
    for topic, count in topic_lie_counts.most_common():
        print(f"  {topic}: {count}")

    print("\nLIES BY SUBTOPIC")
    print("-" * 40)
    for subtopic, count in subtopic_lie_counts.most_common():
        print(f"  {subtopic}: {count}")

    # Plot 1: lies per question
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Bar chart: lies per question (sorted)
    sorted_questions = question_lie_counts.most_common()
    short_labels = [qid.replace("tiananmen_square_1989_", "ts_").replace("falun_gong_", "fg_") for qid, _ in sorted_questions]
    counts = [c for _, c in sorted_questions]
    axes[0].barh(range(len(counts)), counts, color="steelblue")
    axes[0].set_yticks(range(len(counts)))
    axes[0].set_yticklabels(short_labels, fontsize=7)
    axes[0].set_xlabel("Number of lie instances")
    axes[0].set_title("Lies per question")
    axes[0].invert_yaxis()

    # Bar chart: lies per topic
    topics = list(topic_lie_counts.keys())
    topic_counts = [topic_lie_counts[t] for t in topics]
    axes[1].bar(topics, topic_counts, color=["#e74c3c", "#3498db"])
    axes[1].set_ylabel("Number of lie instances")
    axes[1].set_title("Lies by topic")

    # Bar chart: lies per subtopic
    subtopics = [s for s, _ in subtopic_lie_counts.most_common()]
    sub_counts = [c for _, c in subtopic_lie_counts.most_common()]
    colors = ["#e74c3c" if "tiananmen" in s else "#3498db" for s in subtopics]
    axes[2].barh(range(len(sub_counts)), sub_counts, color=colors)
    axes[2].set_yticks(range(len(sub_counts)))
    axes[2].set_yticklabels([s.split("/")[1] for s in subtopics], fontsize=8)
    axes[2].set_xlabel("Number of lie instances")
    axes[2].set_title("Lies by subtopic")
    axes[2].invert_yaxis()

    plt.tight_layout()
    out_path = "plots/lie_distribution.png"
    import os
    os.makedirs("plots", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
