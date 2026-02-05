#!/usr/bin/env python3
"""Create a minimal version of honest_from_original.json keeping only answer fields."""

import json

def process_dataset(input_path, output_path):
    """Process the dataset to keep only necessary fields."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Process each entry
    for entry in data:
        # Remove original_assistant_response, user_message, and mix_key
        if 'original_assistant_response' in entry:
            del entry['original_assistant_response']
        if 'user_message' in entry:
            del entry['user_message']
        if 'mix_key' in entry:
            del entry['mix_key']

        # Process model_responses to extract just the answer strings
        if 'model_responses' in entry:
            entry['model_responses'] = [
                response['answer'] for response in entry['model_responses']
                if 'answer' in response
            ]

    # Write the processed data
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(data)} entries")
    print(f"Output written to: {output_path}")

if __name__ == '__main__':
    input_path = 'honesty_training/data/honest_from_original.json'
    output_path = 'honesty_training/data/honest_from_original_minimal.json'

    process_dataset(input_path, output_path)
