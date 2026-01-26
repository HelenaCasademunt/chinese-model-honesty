#!/usr/bin/env python3
"""Verify that filtered assistant prefill matches filtered questions."""

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_questions(data: dict) -> set[tuple[str, str, str, str]]:
    """Extract (topic, subtopic, level, question) tuples from data."""
    questions = set()
    for topic, subtopics in data.items():
        if topic == "metadata":
            continue
        for subtopic, items in subtopics.items():
            for item in items:
                questions.add((topic, subtopic, item["level"], item["question"]))
    return questions


def main():
    base_path = Path(__file__).parent.parent / "prompts"

    questions_path = base_path / "eval" / "prompts" / "finegrained_eval_questions_filtered.json"
    prefill_path = base_path / "finegrained_assistant_prefill_filtered.json"

    # Handle the different directory structure
    questions_path = Path(__file__).parent.parent.parent / "black_box" / "eval" / "prompts" / "finegrained_eval_questions_filtered.json"
    prefill_path = Path(__file__).parent.parent / "prompts" / "finegrained_assistant_prefill_filtered.json"

    print(f"Loading questions from: {questions_path}")
    print(f"Loading prefills from: {prefill_path}")

    questions_data = load_json(questions_path)
    prefill_data = load_json(prefill_path)

    questions_set = extract_questions(questions_data)
    prefill_set = extract_questions(prefill_data)

    # Check for questions in filtered questions but not in prefill
    missing_from_prefill = questions_set - prefill_set
    if missing_from_prefill:
        print(f"\nERROR: {len(missing_from_prefill)} questions in filtered questions but missing from prefill:")
        for topic, subtopic, level, question in sorted(missing_from_prefill):
            print(f"  - {topic}.{subtopic}.{level}: {question[:60]}...")

    # Check for questions in prefill but not in filtered questions
    extra_in_prefill = prefill_set - questions_set
    if extra_in_prefill:
        print(f"\nERROR: {len(extra_in_prefill)} questions in prefill but not in filtered questions:")
        for topic, subtopic, level, question in sorted(extra_in_prefill):
            print(f"  - {topic}.{subtopic}.{level}: {question[:60]}...")

    # Summary
    print(f"\nSummary:")
    print(f"  Questions in filtered questions file: {len(questions_set)}")
    print(f"  Questions in filtered prefill file: {len(prefill_set)}")

    if not missing_from_prefill and not extra_in_prefill:
        print("\n✓ All questions match! Filtered prefill is correct.")
        return 0
    else:
        print("\n✗ Mismatch detected!")
        return 1


if __name__ == "__main__":
    exit(main())
