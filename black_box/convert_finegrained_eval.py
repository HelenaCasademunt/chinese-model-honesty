"""Convert finegrained_eval_questions_filtered.json to the format expected by the fact generation pipeline."""

import json
from pathlib import Path

input_path = Path(__file__).parent / "eval/prompts/finegrained_eval_questions_filtered.json"
output_path = Path(__file__).parent / "eval/prompts/finegrained_eval_questions_pipeline.json"

with open(input_path) as f:
    data = json.load(f)

category_questions = []

for topic_key, topic_data in data.items():
    if topic_key == "metadata":
        continue

    for subcategory_name, questions_list in topic_data.items():
        broad = []
        targeted = []

        for q in questions_list:
            level = q["level"]
            question_text = q["question"]
            if level == "broad":
                broad.append(question_text)
            else:
                # Map both "medium" and "targeted" to "targeted"
                targeted.append(question_text)

        category_questions.append({
            "name": f"{topic_key}/{subcategory_name}",
            "broad": broad,
            "targeted": targeted,
        })

with open(output_path, "w") as f:
    json.dump(category_questions, f, indent=2, ensure_ascii=False)

total = sum(len(c["broad"]) + len(c["targeted"]) for c in category_questions)
print(f"Converted {len(category_questions)} categories, {total} questions total")
print(f"Output: {output_path}")
