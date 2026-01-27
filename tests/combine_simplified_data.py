#!/usr/bin/env python3
"""
Combines all simplified split_personality data files into a single JSON file.
"""
import json
from pathlib import Path

def main():
    # Set up paths
    input_dir = Path("/root/chinese-model-honesty/split_personality/simplified_data")
    output_file = Path("/root/chinese-model-honesty/split_personality/all_topics_combined.json")

    # Get all JSON files
    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} files to combine\n")

    # Collect all data into a flat list
    all_items = []

    for json_file in json_files:
        print(f"Reading {json_file.name}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        topic = data.get("topic")
        items = data.get("data", [])

        # Add topic field to each item
        for item in items:
            item["topic"] = topic
            all_items.append(item)

        print(f"  Added {len(items)} items from {topic}")

    # Write combined file
    print(f"\nWriting combined file with {len(all_items)} total items...")
    output_data = {
        "model": "qwen/qwen3-32b",
        "data": all_items
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Done! Combined file saved to {output_file}")
    print(f"Total items: {len(all_items)}")

if __name__ == "__main__":
    main()
