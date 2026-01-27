#!/usr/bin/env python3
"""
Filter goals-data.jsonl to only include entries marked as dishonest in honesty_evaluation.json
"""
import json

# Load honesty evaluations
print("Loading honesty evaluations...")
with open('honesty_training/honesty_evaluation.json', 'r') as f:
    evaluations = json.load(f)

# Extract system prompts for dishonest entries
print("Extracting dishonest system prompts...")
dishonest_system_prompts = set()
dishonest_count = 0
honest_count = 0
error_count = 0
total_count = 0

for entry in evaluations:
    total_count += 1
    # Check if any evaluated response is marked as dishonest
    for eval_response in entry.get('evaluated_responses', []):
        evaluation = eval_response.get('evaluation', {})
        is_honest = evaluation.get('is_honest')

        # Only count if is_honest is explicitly False (not None/missing)
        if is_honest is False:
            dishonest_count += 1
            # Store the system prompt (without the \n\nHuman: suffix)
            system_prompt = entry['system_prompt']
            dishonest_system_prompts.add(system_prompt)
            break
        elif is_honest is True:
            honest_count += 1
            break
        else:
            # is_honest is None or missing - this is an error case
            error_count += 1
            break

print(f"Found {dishonest_count} dishonest entries out of {total_count} total evaluations")
print(f"  Honest: {honest_count}")
print(f"  Deceptive: {dishonest_count}")
print(f"  Errors: {error_count}")

# Filter goals-data.jsonl
print("\nFiltering goals-data.jsonl...")
matched_count = 0
output_entries = []

with open('honesty_training/goals-data.jsonl', 'r') as f:
    for line_num, line in enumerate(f, 1):
        if line.strip():
            entry = json.loads(line)
            # Extract system prompt from prompt_text
            # prompt_text format: "system_prompt\n\nHuman: user_message\n\nAssistant:"
            prompt_text = entry['prompt_text']

            # Find the system prompt part (everything before "\n\nHuman:")
            if '\n\nHuman:' in prompt_text:
                system_prompt = prompt_text.split('\n\nHuman:')[0]
            else:
                system_prompt = prompt_text

            # Check if this system prompt matches any dishonest entry
            if system_prompt in dishonest_system_prompts:
                matched_count += 1
                output_entries.append(entry)

        if line_num % 10000 == 0:
            print(f"Processed {line_num} lines, matched {matched_count} so far...")

print(f"\nMatched {matched_count} entries from goals-data.jsonl")

# Write filtered data
output_file = 'honesty_training/goals-data-dishonest.jsonl'
print(f"\nWriting filtered data to {output_file}...")
with open(output_file, 'w') as f:
    for entry in output_entries:
        f.write(json.dumps(entry) + '\n')

print(f"Done! Written {len(output_entries)} entries to {output_file}")
