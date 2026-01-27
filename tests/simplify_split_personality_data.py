#!/usr/bin/env python3
"""
Simplifies split_personality data files by extracting key fields and restructuring them.
"""
import json
from pathlib import Path

def transform_response_data(variant_data):
    """Extract response data from inferences structure."""
    if not variant_data:
        return None

    # Get the split_personality intervention data
    split_personality = variant_data.get("interventions", {}).get("split_personality__version_4", {})

    return {
        "model_response": variant_data.get("S"),
        "split_personality_intervention": split_personality.get("I"),
        "split_personality_assessment": split_personality.get("R"),
        "flag": split_personality.get("P"),
        "tags": split_personality.get("tags", [])
    }

def transform_data_item(item, filtered_model):
    """Transform a single data item to simplified format."""
    transformed = {}

    # Add prompts (rename A/B to A_prompt/B_prompt)
    if "A" in item:
        transformed["A_prompt"] = item["A"]
    if "B" in item:
        transformed["B_prompt"] = item["B"]

    # Add user query
    transformed["user_query"] = item.get("T")

    # Keep metadata
    transformed["metadata"] = item.get("metadata", {})

    # Extract response data from inferences
    inferences = item.get("inferences", {}).get(filtered_model, {})

    # Add A_response
    if "A" in inferences:
        transformed["A_response"] = transform_response_data(inferences["A"])

    # Add B_response
    if "B" in inferences:
        transformed["B_response"] = transform_response_data(inferences["B"])

    return transformed

def simplify_file(input_path, output_dir):
    """Simplify a single data file."""
    print(f"Processing {input_path.name}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract fields
    topic = data.get("topic")
    filtered_model = data.get("filtered_model")
    items = data.get("data", [])

    # Transform each data item
    simplified_data = []
    for item in items:
        transformed = transform_data_item(item, filtered_model)
        simplified_data.append(transformed)

    # Create output structure
    output = {
        "topic": topic,
        "model": filtered_model,
        "data": simplified_data
    }

    # Write to output file named after the topic
    output_path = output_dir / f"{topic}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Created {output_path.name} with {len(simplified_data)} items")

def main():
    # Set up paths
    data_dir = Path("/root/chinese-model-honesty/split_personality/data")
    output_dir = Path("/root/chinese-model-honesty/split_personality/simplified_data")
    output_dir.mkdir(exist_ok=True)

    # Process all JSON files
    json_files = sorted(data_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    print(f"Found {len(json_files)} files to process\n")

    for json_file in json_files:
        try:
            simplify_file(json_file, output_dir)
        except Exception as e:
            print(f"  ERROR processing {json_file.name}: {e}")

    print(f"\nDone! Simplified files saved to {output_dir}")

if __name__ == "__main__":
    main()
