#!/usr/bin/env python3
"""Count the number of responses in assistant_prefill_responses.json"""

import json

# Load the JSON file
with open('results/data/assistant_prefill_responses.json', 'r') as f:
    data = json.load(f)

# Overall statistics
total_topics = len(data)
total_items = 0
total_responses = 0

print("=" * 80)
print("OVERALL STATISTICS")
print("=" * 80)
print(f"Total topics: {total_topics}\n")

print("=" * 80)
print("PER-TOPIC BREAKDOWN")
print("=" * 80)

# Count items and responses per topic
for topic, items in data.items():
    num_items = len(items)
    total_items += num_items

    # Count responses in this topic
    topic_responses = 0
    responses_per_item = []

    for item in items:
        model_responses = item.get('model_responses', [])
        num_responses = len(model_responses)
        topic_responses += num_responses
        responses_per_item.append(num_responses)

    total_responses += topic_responses

    print(f"\nTopic: {topic}")
    print(f"  Items: {num_items}")
    print(f"  Responses per item: {responses_per_item[0] if responses_per_item else 0}")
    print(f"  Total responses: {topic_responses}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total topics: {total_topics}")
print(f"Total items: {total_items}")
print(f"Average items per topic: {total_items / total_topics:.1f}" if total_topics > 0 else "N/A")
print(f"Total model responses: {total_responses}")
print(f"Average responses per item: {total_responses / total_items:.1f}" if total_items > 0 else "N/A")
print("=" * 80)
