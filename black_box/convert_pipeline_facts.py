"""Convert pipeline fact files to the format expected by evaluate_openrouter.py.

Pipeline format (per category):
  {"name": "Title Case Name", "questions": [{"level": ..., "question": ..., "facts": [{"fact": str, "count": int}]}]}

Eval format:
  {"topic": {"subtopic_snake_case": [{"level": ..., "question": ..., "facts": [str]}]}}
"""

import json
import re
import sys
from pathlib import Path


PIPELINE_BASE = Path("/root/chinese_auditing/output/prompt_tests_pipeline")
OUTPUT_BASE = Path("/root/chinese-model-honesty/black_box/eval/prompts/prompt_tests_converted")


def name_to_snake(name: str) -> str:
    """Convert 'Title Case Name' to 'title_case_name'."""
    return re.sub(r'\s+', '_', name.strip()).lower()


def convert_facts_file(pipeline_json_path: Path, topic: str) -> dict:
    """Convert a single pipeline facts JSON to eval format."""
    with open(pipeline_json_path) as f:
        data = json.load(f)

    result = {}
    for category in data["categories"]:
        subtopic = name_to_snake(category["name"])
        entries = []
        for q in category["questions"]:
            facts = [f["fact"] if isinstance(f, dict) else f for f in q["facts"]]
            entries.append({
                "level": q["level"],
                "question": q["question"],
                "facts": facts,
            })
        result[subtopic] = entries

    return {topic: result}


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    for model_dir in sorted(PIPELINE_BASE.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        for topic_dir in sorted(model_dir.iterdir()):
            if not topic_dir.is_dir():
                continue
            topic = topic_dir.name

            for variant_dir in sorted(topic_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                variant = variant_dir.name

                final_json = variant_dir / "final" / f"{topic}.json"
                if not final_json.exists():
                    print(f"SKIP (no final): {model}/{topic}/{variant}")
                    continue

                converted = convert_facts_file(final_json, topic)

                out_dir = OUTPUT_BASE / model / topic
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{variant}.json"

                with open(out_path, "w") as f:
                    json.dump(converted, f, indent=2, ensure_ascii=False)

                print(f"Converted: {out_path}")


if __name__ == "__main__":
    main()
