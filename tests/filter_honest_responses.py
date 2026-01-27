#!/usr/bin/env python3
"""
Filter honest_from_original.json to keep only one response per goal.
Keep the first answer that doesn't contain "system prompt".
"""

import json

def main():
    input_file = "honesty_training/data/honest_from_original.json"
    output_file = "honesty_training/data/honest_from_original_filtered.json"

    with open(input_file, 'r') as f:
        data = json.load(f)

    filtered_data = []
    skipped_count = 0

    for entry in data:
        goal_id = entry["goal_id"]
        system_prompt = entry["system_prompt"]
        model_responses = entry["model_responses"]

        # Find the first response that doesn't contain "system prompt"
        selected_answer = None
        for response in model_responses:
            answer = response["answer"]
            if "system prompt" not in answer.lower():
                selected_answer = answer
                break

        # If we found a suitable answer, add it to filtered data
        if selected_answer:
            filtered_entry = {
                "goal_id": goal_id,
                "system_prompt": system_prompt,
                "answer": selected_answer
            }
            filtered_data.append(filtered_entry)
        else:
            skipped_count += 1
            print(f"Skipped {goal_id} - all responses contain 'system prompt'")

    # Write filtered data to output file
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"\nFiltering complete!")
    print(f"Original entries: {len(data)}")
    print(f"Filtered entries: {len(filtered_data)}")
    print(f"Skipped entries: {skipped_count}")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    main()
