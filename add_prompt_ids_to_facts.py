#!/usr/bin/env python3
"""
Add prompt_id fields to questions/facts file by matching questions from responses file.
Supports both array format (dev_questions.json) and nested format (dev_facts.json).
"""

import json
import sys


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    if len(sys.argv) < 3:
        print("Usage: python add_prompt_ids_to_facts.py <responses_file> <questions_file> [output_file]")
        print("Example: python add_prompt_ids_to_facts.py results/qwen_baseline_responses_20260204_002829.json data/dev_questions.json data/dev_questions_with_ids.json")
        sys.exit(1)

    responses_file = sys.argv[1]
    questions_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else questions_file

    print(f"Loading responses from: {responses_file}")
    responses_data = load_json(responses_file)

    print(f"Loading questions from: {questions_file}")
    questions_data = load_json(questions_file)

    # Build mapping of question text to prompt_id
    question_to_id = {}
    for result in responses_data.get("results", []):
        prompt = result.get("prompt", "")
        prompt_id = result.get("prompt_id", "")
        if prompt and prompt_id:
            question_to_id[prompt] = prompt_id

    print(f"Found {len(question_to_id)} unique questions in responses")

    # Add prompt_id to each question
    matches = 0
    no_matches = []

    # Handle both array format and nested format
    if isinstance(questions_data, list):
        # Array format like dev_questions.json
        for question in questions_data:
            question_text = question.get("question", "")
            if question_text in question_to_id:
                question["prompt_id"] = question_to_id[question_text]
                matches += 1
            else:
                no_matches.append(question_text)
    else:
        # Nested format with categories
        for category in questions_data.get("categories", []):
            for question in category.get("questions", []):
                question_text = question.get("question", "")
                if question_text in question_to_id:
                    question["prompt_id"] = question_to_id[question_text]
                    matches += 1
                else:
                    no_matches.append(question_text)

    print(f"\nMatched {matches} questions")
    if no_matches:
        print(f"Could not match {len(no_matches)} questions:")
        for q in no_matches[:5]:
            print(f"  - {q}")
        if len(no_matches) > 5:
            print(f"  ... and {len(no_matches) - 5} more")

    print(f"\nSaving updated questions to: {output_file}")
    save_json(questions_data, output_file)
    print("Done!")


if __name__ == "__main__":
    main()
