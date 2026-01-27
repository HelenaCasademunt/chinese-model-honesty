#!/usr/bin/env python3
"""
Flattens split_personality data so A and B responses become separate data items.
"""
import json
from pathlib import Path

def flatten_item(item, variant):
    """Convert a single A or B variant into a flattened item."""
    # Get the response data for this variant
    response_key = f"{variant}_response"
    prompt_key = f"{variant}_prompt"

    if response_key not in item or prompt_key not in item:
        return None

    response_data = item[response_key]
    if not response_data:
        return None

    # Create flattened item
    flattened = {
        "system_prompt": item[prompt_key],
        "user_query": item["user_query"],
        "metadata": item.get("metadata", {}),
        "model_response": response_data.get("model_response"),
        "split_personality_intervention": response_data.get("split_personality_intervention"),
        "split_personality_assessment": response_data.get("split_personality_assessment"),
        "flag": response_data.get("flag"),
        "tags": response_data.get("tags", []),
        "topic": item.get("topic")
    }

    return flattened

def main():
    # Set up paths
    input_file = Path("/root/chinese-model-honesty/split_personality/all_topics_combined.json")
    output_file = Path("/root/chinese-model-honesty/split_personality/all_topics_flattened.json")

    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model = data.get("model")
    items = data.get("data", [])

    print(f"Processing {len(items)} items...")

    # Flatten each item into separate A and B entries
    flattened_items = []

    for item in items:
        # Process A variant
        a_item = flatten_item(item, "A")
        if a_item:
            flattened_items.append(a_item)

        # Process B variant
        b_item = flatten_item(item, "B")
        if b_item:
            flattened_items.append(b_item)

    # Create output structure
    output_data = {
        "model": model,
        "data": flattened_items
    }

    # Write flattened file
    print(f"\nWriting flattened file with {len(flattened_items)} items...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Done! Flattened file saved to {output_file}")
    print(f"Original items: {len(items)}")
    print(f"Flattened items: {len(flattened_items)}")

if __name__ == "__main__":
    main()
