import json

def check_jsonl_format(filepath):
    print(f"Checking: {filepath}")
    print("-" * 50)

    total = 0
    human_count_issues = []
    assistant_ending_issues = []

    with open(filepath, 'r') as f:
        for i, line in enumerate(f, 1):
            total += 1
            data = json.loads(line)
            prompt = data.get('prompt_text', '')

            # Check for exactly one "\n\nHuman:" instance
            human_count = prompt.count("\n\nHuman:")
            if human_count != 1:
                human_count_issues.append((i, human_count))

            # Check if ends with "\n\nAssistant:"
            if not prompt.endswith("\n\nAssistant:"):
                assistant_ending_issues.append(i)

    print(f"Total entries: {total}")
    print()

    # Report Human: count issues
    if human_count_issues:
        print(f"Entries without exactly 1 '\\n\\nHuman:': {len(human_count_issues)}")
        for line_num, count in human_count_issues[:10]:
            print(f"  Line {line_num}: found {count} instances")
        if len(human_count_issues) > 10:
            print(f"  ... and {len(human_count_issues) - 10} more")
    else:
        print("✓ All entries have exactly 1 '\\n\\nHuman:'")

    print()

    # Report Assistant: ending issues
    if assistant_ending_issues:
        print(f"Entries not ending with '\\n\\nAssistant:': {len(assistant_ending_issues)}")
        for line_num in assistant_ending_issues[:10]:
            print(f"  Line {line_num}")
        if len(assistant_ending_issues) > 10:
            print(f"  ... and {len(assistant_ending_issues) - 10} more")
    else:
        print("✓ All entries end with '\\n\\nAssistant:'")

    print()
    return len(human_count_issues) == 0 and len(assistant_ending_issues) == 0


if __name__ == "__main__":
    files = [
        "goals-data.jsonl",
        "followup_data.jsonl"
    ]

    for filepath in files:
        check_jsonl_format(filepath)
        print("=" * 50)
        print()
