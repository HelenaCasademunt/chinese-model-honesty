"""
Convert question files from the chinese_auditing prompt_tests format to the
finegrained eval format expected by baseline.py.

Source format:
  { "prompt_variant": "...", "categories": [{"name": "...", "broad": [...], "targeted": [...]}], "stats": {...} }

Target format:
  { "topic": { "subtopic": [{"level": "broad", "question": "...", "expected_answer": ""}] } }
"""

import json
import argparse
import os


def convert(input_path: str, topic: str) -> dict:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for category in data.get("categories", []):
        subtopic = category["name"].lower().replace(" ", "_")
        questions = []
        for level in ("broad", "targeted"):
            for q_text in category.get(level, []):
                questions.append({
                    "level": level,
                    "question": q_text,
                    "expected_answer": "",
                })
        result[subtopic] = questions

    return {topic: result}


def main():
    parser = argparse.ArgumentParser(description="Convert prompt_tests format to finegrained eval format")
    parser.add_argument("input", help="Input JSON file")
    parser.add_argument("output", help="Output JSON file")
    parser.add_argument("--topic", default=None, help="Topic name (defaults to parent directory name)")
    args = parser.parse_args()

    topic = args.topic or os.path.basename(os.path.dirname(args.input))
    converted = convert(args.input, topic)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    print(f"Converted {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
