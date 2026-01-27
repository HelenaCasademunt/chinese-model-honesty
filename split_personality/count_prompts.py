#!/usr/bin/env python3
"""Count entries by prompt type and flags in all_topics_combined.json"""

import json

# Read the JSON file
with open('data/all_topics_combined.json', 'r') as f:
    data = json.load(f)

# Initialize counters for prompts
only_a = 0
only_b = 0
both = 0
neither = 0

# Initialize counters for flags
a_pass = 0
a_fail = 0
b_pass = 0
b_fail = 0

# Count entries by prompt type and flags
for entry in data['data']:
    has_a = 'A_prompt' in entry and entry['A_prompt'] is not None
    has_b = 'B_prompt' in entry and entry['B_prompt'] is not None

    if has_a and has_b:
        both += 1
    elif has_a:
        only_a += 1
    elif has_b:
        only_b += 1
    else:
        neither += 1

    # Count flags for A_response
    if 'A_response' in entry and 'flag' in entry['A_response']:
        if entry['A_response']['flag'] == 'PASS':
            a_pass += 1
        elif entry['A_response']['flag'] == 'FAIL':
            a_fail += 1

    # Count flags for B_response
    if 'B_response' in entry and 'flag' in entry['B_response']:
        if entry['B_response']['flag'] == 'PASS':
            b_pass += 1
        elif entry['B_response']['flag'] == 'FAIL':
            b_fail += 1

# Print results
total = len(data['data'])
print("=== Prompt Distribution ===")
print(f"Total entries: {total}")
print(f"Entries with only A_prompt: {only_a} ({only_a/total*100:.1f}%)")
print(f"Entries with only B_prompt: {only_b} ({only_b/total*100:.1f}%)")
print(f"Entries with both prompts: {both} ({both/total*100:.1f}%)")
if neither > 0:
    print(f"Entries with neither prompt: {neither} ({neither/total*100:.1f}%)")

print("\n=== Flag Distribution ===")
print(f"A_response PASS: {a_pass} ({a_pass/total*100:.1f}%)")
print(f"A_response FAIL: {a_fail} ({a_fail/total*100:.1f}%)")
print(f"B_response PASS: {b_pass} ({b_pass/total*100:.1f}%)")
print(f"B_response FAIL: {b_fail} ({b_fail/total*100:.1f}%)")
