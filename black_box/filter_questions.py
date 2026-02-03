"""Filter questions and facts files to only include selected questions."""

import json
from pathlib import Path

# Selected question IDs from the top lies analysis
SELECTED_QUESTION_IDS = [
    # Falun Gong (10)
    "falun_gong_organ_harvesting_medium",
    "falun_gong_founder_targeted",
    "falun_gong_610_office_targeted",
    "falun_gong_zhongnanhai_protest_medium",
    "falun_gong_death_toll_broad",
    "falun_gong_membership_numbers_targeted",
    "falun_gong_detention_methods_medium",
    "falun_gong_zhongnanhai_protest_broad",
    "falun_gong_detention_methods_broad",
    "falun_gong_death_toll_targeted",
    # Tiananmen Square (5)
    "tiananmen_square_1989_tiananmen_mothers_targeted",
    "tiananmen_square_1989_casualties_and_violence_targeted",
    "tiananmen_square_1989_goddess_of_democracy_medium",
    "tiananmen_square_1989_leadership_decisions_medium",
    "tiananmen_square_1989_casualties_and_violence_broad",
]

PROMPTS_DIR = Path(__file__).parent / "eval" / "prompts"
QUESTIONS_FILE = PROMPTS_DIR / "finegrained_eval_questions.json"
FACTS_FILE = PROMPTS_DIR / "finegrained_eval_facts.json"
QUESTIONS_OUTPUT = PROMPTS_DIR / "finegrained_eval_questions_top_lies.json"
FACTS_OUTPUT = PROMPTS_DIR / "finegrained_eval_facts_top_lies.json"


def parse_question_id(qid):
    """Parse question_id into topic, subtopic, level."""
    # Format: {topic}_{subtopic}_{level}
    # Topics: tiananmen_square_1989, falun_gong
    if qid.startswith("tiananmen_square_1989_"):
        topic = "tiananmen_square_1989"
        rest = qid[len("tiananmen_square_1989_"):]
    elif qid.startswith("falun_gong_"):
        topic = "falun_gong"
        rest = qid[len("falun_gong_"):]
    else:
        return None, None, None

    # Level is the last part
    parts = rest.rsplit("_", 1)
    if len(parts) == 2:
        subtopic, level = parts
    else:
        return None, None, None

    return topic, subtopic, level


def filter_json_file(input_path, output_path, selected_ids):
    """Filter a questions or facts JSON file to only include selected questions."""
    with open(input_path) as f:
        data = json.load(f)

    # Build a set of (topic, subtopic, level) tuples we want to keep
    selected_set = set()
    for qid in selected_ids:
        topic, subtopic, level = parse_question_id(qid)
        if topic:
            selected_set.add((topic, subtopic, level))

    # Filter the data
    filtered = {}

    # Keep metadata
    if "metadata" in data:
        filtered["metadata"] = data["metadata"]
        filtered["metadata"]["description"] += " (filtered to top lying questions)"

    # Filter each topic
    for topic, subtopics in data.items():
        if topic == "metadata":
            continue

        filtered_subtopics = {}
        for subtopic, questions in subtopics.items():
            filtered_questions = []
            for q in questions:
                level = q.get("level")
                if (topic, subtopic, level) in selected_set:
                    filtered_questions.append(q)

            if filtered_questions:
                filtered_subtopics[subtopic] = filtered_questions

        if filtered_subtopics:
            filtered[topic] = filtered_subtopics

    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)

    return filtered


def main():
    print(f"Filtering questions and facts to {len(SELECTED_QUESTION_IDS)} selected questions\n")

    # Parse and display selected questions
    print("Selected questions:")
    for qid in SELECTED_QUESTION_IDS:
        topic, subtopic, level = parse_question_id(qid)
        print(f"  {topic} / {subtopic} / {level}")

    print(f"\nFiltering {QUESTIONS_FILE.name}...")
    filtered_questions = filter_json_file(QUESTIONS_FILE, QUESTIONS_OUTPUT, SELECTED_QUESTION_IDS)

    # Count questions
    q_count = sum(
        len(questions)
        for subtopics in filtered_questions.values()
        if isinstance(subtopics, dict)
        for questions in subtopics.values()
    )
    print(f"  Saved {q_count} questions to {QUESTIONS_OUTPUT.name}")

    print(f"\nFiltering {FACTS_FILE.name}...")
    filtered_facts = filter_json_file(FACTS_FILE, FACTS_OUTPUT, SELECTED_QUESTION_IDS)

    # Count facts
    f_count = sum(
        len(questions)
        for subtopics in filtered_facts.values()
        if isinstance(subtopics, dict)
        for questions in subtopics.values()
    )
    print(f"  Saved {f_count} question-fact entries to {FACTS_OUTPUT.name}")

    # Summary by topic
    print("\nSummary by topic:")
    for topic in ["tiananmen_square_1989", "falun_gong"]:
        if topic in filtered_questions:
            subtopics = filtered_questions[topic]
            count = sum(len(qs) for qs in subtopics.values())
            print(f"  {topic}: {count} questions across {len(subtopics)} subtopics")


if __name__ == "__main__":
    main()
