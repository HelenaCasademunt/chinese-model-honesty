#!/usr/bin/env python3
"""
Flattens split_personality data so A and B responses become separate data items.
Both A and B examples use the A system prompt.
Balances the dataset by ensuring equal numbers of PASS and FAIL items.
"""
import json
import random
from pathlib import Path

def flatten_item(item, variant):
    """Convert a single A or B variant into a flattened item using A system prompt."""
    # Get the response data for this variant
    response_key = f"{variant}_response"

    if response_key not in item:
        return None

    response_data = item[response_key]
    if not response_data:
        return None

    # For B responses, ensure A response exists
    if variant == "B":
        if not item.get("A_response") or not item.get("A_prompt"):
            return None

    # Always use A_prompt for both A and B variants
    if "A_prompt" not in item:
        return None

    # Create flattened item
    flattened = {
        "system_prompt": item["A_prompt"],
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
    input_file = Path("/root/chinese-model-honesty/split_personality/data/all_topics_combined.json")
    output_file = Path("/root/chinese-model-honesty/split_personality/data/all_topics_flattened_use_a_prompt_balanced.json")

    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model = data.get("model")
    items = data.get("data", [])

    print(f"Processing {len(items)} items...")

    # Flatten each item into separate A and B entries
    flattened_items = []
    a_count = 0
    b_count = 0
    b_skipped = 0

    for item in items:
        # Process A variant
        a_item = flatten_item(item, "A")
        if a_item:
            flattened_items.append(a_item)
            a_count += 1

        # Process B variant (only if A exists)
        b_item = flatten_item(item, "B")
        if b_item:
            flattened_items.append(b_item)
            b_count += 1
        elif item.get("B_response"):
            # B response exists but was skipped (missing A)
            b_skipped += 1

    print(f"\nFlattening complete:")
    print(f"  Original items: {len(items)}")
    print(f"  Flattened A items: {a_count}")
    print(f"  Flattened B items: {b_count}")
    print(f"  B items skipped (no corresponding A): {b_skipped}")
    print(f"  Total flattened items: {len(flattened_items)}")

    # Split into passes and fails
    passes = [item for item in flattened_items if item.get("flag") == "PASS"]
    fails = [item for item in flattened_items if item.get("flag") == "FAIL"]

    print(f"\nBefore balancing:")
    print(f"  PASS items: {len(passes)}")
    print(f"  FAIL items: {len(fails)}")

    # Balance by randomly sampling from the larger group
    if len(passes) > len(fails):
        # More passes than fails - randomly sample passes
        print(f"\nBalancing: randomly removing {len(passes) - len(fails)} PASS items")
        random.seed(42)  # For reproducibility
        passes = random.sample(passes, len(fails))
    elif len(fails) > len(passes):
        # More fails than passes - randomly sample fails
        print(f"\nBalancing: randomly removing {len(fails) - len(passes)} FAIL items")
        random.seed(42)  # For reproducibility
        fails = random.sample(fails, len(passes))
    else:
        print(f"\nDataset is already balanced!")

    # Combine and shuffle
    balanced_items = passes + fails
    random.seed(42)
    random.shuffle(balanced_items)

    print(f"\nAfter balancing:")
    print(f"  PASS items: {len(passes)}")
    print(f"  FAIL items: {len(fails)}")
    print(f"  Total balanced items: {len(balanced_items)}")

    # Create output structure
    output_data = {
        "model": model,
        "data": balanced_items
    }

    # Write balanced file
    print(f"\nWriting balanced file...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Done! Balanced file saved to {output_file}")

if __name__ == "__main__":
    main()
