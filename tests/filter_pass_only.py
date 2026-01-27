#!/usr/bin/env python3
"""
Filters split_personality data to only keep entries where flag is "PASS".
"""
import json
from pathlib import Path

def main():
    # Set up paths
    input_file = Path("/root/chinese-model-honesty/split_personality/data/all_topics_flattened.json")
    output_file = Path("/root/chinese-model-honesty/split_personality/data/all_topics_flattened_pass_only.json")

    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model = data.get("model")
    items = data.get("data", [])

    print(f"Total items: {len(items)}")

    # Filter for PASS entries only
    pass_items = [item for item in items if item.get("flag") == "PASS"]

    print(f"PASS items: {len(pass_items)}")
    print(f"FAIL items: {len(items) - len(pass_items)}")

    # Create output structure
    output_data = {
        "model": model,
        "data": pass_items
    }

    # Write filtered file
    print(f"\nWriting filtered file...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Done! Filtered file saved to {output_file}")

if __name__ == "__main__":
    main()
