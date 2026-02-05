#!/usr/bin/env python3
"""Verify filtered file has both prompts and count flags"""

import json

# Read the filtered JSON file
with open('data/all_topics_both_prompts.json', 'r') as f:
    data = json.load(f)

# Verify all entries have both prompts
missing_a = 0
missing_b = 0
a_pass = 0
a_fail = 0
b_pass = 0
b_fail = 0

for entry in data['data']:
    # Check for prompts
    has_a = 'A_prompt' in entry and entry['A_prompt'] is not None and entry['A_prompt'] != ''
    has_b = 'B_prompt' in entry and entry['B_prompt'] is not None and entry['B_prompt'] != ''

    if not has_a:
        missing_a += 1
    if not has_b:
        missing_b += 1

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
print("=== Verification ===")
print(f"Total entries: {total}")
print(f"Entries missing A_prompt: {missing_a}")
print(f"Entries missing B_prompt: {missing_b}")

if missing_a == 0 and missing_b == 0:
    print("✓ All entries have both A_prompt and B_prompt")
else:
    print("✗ Some entries are missing prompts!")

print("\n=== Flag Distribution ===")
print(f"A_response PASS: {a_pass} ({a_pass/total*100:.1f}%)")
print(f"A_response FAIL: {a_fail} ({a_fail/total*100:.1f}%)")
print(f"B_response PASS: {b_pass} ({b_pass/total*100:.1f}%)")
print(f"B_response FAIL: {b_fail} ({b_fail/total*100:.1f}%)")

print("\n=== Comparison ===")
total_a = a_pass + a_fail
total_b = b_pass + b_fail
print(f"A_response total: {total_a} (PASS rate: {a_pass/total_a*100:.1f}%)")
print(f"B_response total: {total_b} (PASS rate: {b_pass/total_b*100:.1f}%)")
