"""
Convert tiananmen_square_1989.json from chinese_auditing format to the
finegrained format used by baseline.py (questions) and evaluate.py (facts).

Source format:
  {"metadata": ..., "categories": [{"name": "...", "questions": [{"level", "question", "facts": [{"fact", "count"}]}]}]}

Output questions format (finegrained):
  {"metadata": ..., "tiananmen_square_1989": {"subtopic": [{"level", "question", "expected_answer"}]}}

Output facts format:
  {"metadata": ..., "tiananmen_square_1989": {"subtopic": [{"level", "question", "facts": ["fact1", ...]}]}}
"""

import json
import argparse


def convert(input_path: str, questions_output: str, facts_output: str, min_count: int = 1):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_metadata = data.get("metadata", {})
    topic = source_metadata.get("topic", "tiananmen_square_1989")

    questions_by_subtopic = {}
    facts_by_subtopic = {}

    for category in data["categories"]:
        # Convert category name to snake_case key
        subtopic = category["name"].lower().replace(" ", "_")

        q_list = []
        f_list = []

        for q in category["questions"]:
            # Build expected_answer from top facts
            top_facts = [f["fact"] for f in q["facts"] if f["count"] >= min_count]

            q_list.append({
                "level": q["level"],
                "question": q["question"],
                "expected_answer": " ".join(top_facts[:5]) if top_facts else "",
            })

            f_list.append({
                "level": q["level"],
                "question": q["question"],
                "facts": top_facts,
            })

        questions_by_subtopic[subtopic] = q_list
        facts_by_subtopic[subtopic] = f_list

    questions_out = {
        "metadata": {
            "description": f"Questions converted from {input_path}",
            "source": input_path,
            "topic": topic,
        },
        topic: questions_by_subtopic,
    }

    facts_out = {
        "metadata": {
            "description": f"Facts converted from {input_path}",
            "source": input_path,
            "topic": topic,
            "min_count": min_count,
        },
        topic: facts_by_subtopic,
    }

    with open(questions_output, "w", encoding="utf-8") as f:
        json.dump(questions_out, f, indent=2, ensure_ascii=False)
    print(f"Wrote questions to {questions_output}")

    with open(facts_output, "w", encoding="utf-8") as f:
        json.dump(facts_out, f, indent=2, ensure_ascii=False)
    print(f"Wrote facts to {facts_output}")


def main():
    parser = argparse.ArgumentParser(description="Convert tiananmen data to finegrained format")
    parser.add_argument("--input", type=str, default="/root/chinese_auditing/data/tiananmen_square_1989.json")
    parser.add_argument("--questions-output", type=str, default="black_box/eval/prompts/tiananmen_questions.json")
    parser.add_argument("--facts-output", type=str, default="black_box/eval/prompts/tiananmen_facts.json")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum fact count to include")
    args = parser.parse_args()

    convert(args.input, args.questions_output, args.facts_output, args.min_count)


if __name__ == "__main__":
    main()
