"""
Test script to verify that every question has an associated prefill.
"""

import json
from pathlib import Path


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_prefills_complete():
    """Test that every question has a corresponding prefill."""
    # Load both files
    questions_file = Path(__file__).parent.parent / 'eval' / 'prompts' / 'finegrained_eval_questions.json'
    prefills_file = Path(__file__).parent.parent / 'prompts' / 'finegrained_assistant_prefill.json'

    questions_data = load_json(questions_file)
    prefills_data = load_json(prefills_file)

    # Track issues
    issues = []
    total_questions = 0
    total_prefills = 0

    # Iterate through all topics in questions file
    for topic_name, topic_data in questions_data.items():
        if topic_name == 'metadata':
            continue

        # Check if topic exists in prefills
        if topic_name not in prefills_data:
            issues.append(f"Missing topic in prefills: {topic_name}")
            continue

        # Iterate through subtopics
        for subtopic_name, subtopic_questions in topic_data.items():
            total_questions += len(subtopic_questions)

            # Check if subtopic exists in prefills
            if subtopic_name not in prefills_data[topic_name]:
                issues.append(f"Missing subtopic in prefills: {topic_name}.{subtopic_name}")
                continue

            prefill_list = prefills_data[topic_name][subtopic_name]
            total_prefills += len(prefill_list)

            # Check if number of items matches
            if len(subtopic_questions) != len(prefill_list):
                issues.append(
                    f"Mismatch in {topic_name}.{subtopic_name}: "
                    f"{len(subtopic_questions)} questions vs {len(prefill_list)} prefills"
                )
                continue

            # Check each question has corresponding prefill
            for i, (question_item, prefill_item) in enumerate(zip(subtopic_questions, prefill_list)):
                # Check level matches
                if question_item.get('level') != prefill_item.get('level'):
                    issues.append(
                        f"Level mismatch in {topic_name}.{subtopic_name}[{i}]: "
                        f"question level='{question_item.get('level')}' vs "
                        f"prefill level='{prefill_item.get('level')}'"
                    )

                # Check question matches
                if question_item.get('question') != prefill_item.get('question'):
                    issues.append(
                        f"Question mismatch in {topic_name}.{subtopic_name}[{i}]: "
                        f"Questions don't match between files"
                    )

                # Check prefill exists and is not empty
                if 'prefill' not in prefill_item:
                    issues.append(
                        f"Missing 'prefill' field in {topic_name}.{subtopic_name}[{i}]: "
                        f"question='{question_item.get('question')}'"
                    )
                elif not prefill_item['prefill'] or not prefill_item['prefill'].strip():
                    issues.append(
                        f"Empty prefill in {topic_name}.{subtopic_name}[{i}]: "
                        f"question='{question_item.get('question')}'"
                    )

    # Print results
    print("=" * 80)
    print("PREFILL COMPLETENESS TEST RESULTS")
    print("=" * 80)
    print(f"\nTotal questions found: {total_questions}")
    print(f"Total prefills found: {total_prefills}")

    if issues:
        print(f"\n❌ FAILED: Found {len(issues)} issue(s):\n")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✅ PASSED: All questions have corresponding prefills!")
        print("\nVerified:")

        # Count by topic
        for topic_name, topic_data in questions_data.items():
            if topic_name == 'metadata':
                continue
            topic_count = sum(len(subtopic) for subtopic in topic_data.values())
            print(f"  - {topic_name}: {topic_count} questions")

        return True


def test_prefill_quality():
    """Test that prefills follow the expected format."""
    prefills_file = Path(__file__).parent.parent / 'prompts' / 'finegrained_assistant_prefill.json'
    prefills_data = load_json(prefills_file)

    issues = []

    for topic_name, topic_data in prefills_data.items():
        if topic_name == 'metadata':
            continue

        for subtopic_name, subtopic_prefills in topic_data.items():
            for i, prefill_item in enumerate(subtopic_prefills):
                prefill = prefill_item.get('prefill', '')
                question = prefill_item.get('question', '')

                # Check that prefill doesn't end with leading phrases that might bias the answer
                biased_endings = [
                    ' is that',
                    ' was that',
                    ' are that',
                    ' were that',
                ]
                for ending in biased_endings:
                    if prefill.endswith(ending):
                        issues.append(
                            f"Potentially biased prefill in {topic_name}.{subtopic_name}[{i}]: "
                            f"ends with '{ending}'"
                        )

    print("\n" + "=" * 80)
    print("PREFILL QUALITY TEST RESULTS")
    print("=" * 80)

    if issues:
        print(f"\n⚠️  WARNING: Found {len(issues)} potential issue(s):\n")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ PASSED: All prefills appear to follow best practices!")

    return len(issues) == 0


if __name__ == '__main__':
    # Run tests
    completeness_passed = test_prefills_complete()
    quality_passed = test_prefill_quality()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Completeness Test: {'✅ PASSED' if completeness_passed else '❌ FAILED'}")
    print(f"Quality Test: {'✅ PASSED' if quality_passed else '⚠️  WARNINGS'}")
    print("=" * 80)

    # Exit with appropriate code
    exit(0 if completeness_passed else 1)
