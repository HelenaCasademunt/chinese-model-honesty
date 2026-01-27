#!/usr/bin/env python3
"""
Counts entries in split_personality data by various criteria.
"""
import json
from pathlib import Path
from collections import Counter

def count_file(file_path):
    """Count entries in a single file."""
    print(f"\n{'='*70}")
    print(f"File: {file_path.name}")
    print('='*70)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model = data.get("model")
    items = data.get("data", [])

    print(f"\nModel: {model}")
    print(f"Total items: {len(items)}")

    # Count by flag
    flag_counts = Counter(item.get("flag") for item in items)
    print(f"\nBy flag:")
    for flag, count in sorted(flag_counts.items()):
        print(f"  {flag}: {count}")

    # Count by topic
    topic_counts = Counter(item.get("topic") for item in items)
    print(f"\nBy topic:")
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic}: {count}")

    # Count by flag within each topic
    print(f"\nBy flag within each topic:")
    topics = sorted(set(item.get("topic") for item in items))
    for topic in topics:
        topic_items = [item for item in items if item.get("topic") == topic]
        pass_count = sum(1 for item in topic_items if item.get("flag") == "PASS")
        fail_count = sum(1 for item in topic_items if item.get("flag") == "FAIL")
        print(f"  {topic}:")
        print(f"    PASS: {pass_count}, FAIL: {fail_count}, Total: {len(topic_items)}")

def main():
    # Files to count
    files = [
        Path("/root/chinese-model-honesty/split_personality/all_topics_combined.json"),
        Path("/root/chinese-model-honesty/split_personality/all_topics_flattened.json"),
        Path("/root/chinese-model-honesty/split_personality/all_topics_flattened_pass_only.json"),
    ]

    for file_path in files:
        if file_path.exists():
            count_file(file_path)
        else:
            print(f"\nFile not found: {file_path}")

if __name__ == "__main__":
    main()
