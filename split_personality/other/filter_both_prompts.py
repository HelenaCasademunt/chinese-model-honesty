#!/usr/bin/env python3
"""Filter all_topics_combined.json to only include entries with both A and B prompts"""

import json

# Read the original JSON file
with open('data/all_topics_combined.json', 'r') as f:
    data = json.load(f)

# Filter to only entries with both A_prompt and B_prompt
filtered_data = {
    'model': data['model'],
    'data': []
}

for entry in data['data']:
    has_a = 'A_prompt' in entry and entry['A_prompt'] is not None
    has_b = 'B_prompt' in entry and entry['B_prompt'] is not None

    if has_a and has_b:
        filtered_data['data'].append(entry)

# Save the filtered data
output_file = 'data/all_topics_both_prompts.json'
with open(output_file, 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f"Original entries: {len(data['data'])}")
print(f"Filtered entries (both prompts): {len(filtered_data['data'])}")
print(f"Saved to: {output_file}")
