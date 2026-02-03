#!/usr/bin/env python3
"""
Split datasets into train and test sets based on matching topics.

For topics that appear in both broad and targeted files:
- Randomly select 5 questions from broad
- Randomly select 5 questions from targeted
- These form the train set
- All remaining questions form the test set
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

# Load the datasets
datasets_dir = Path("/root/chinese-model-honesty/datasets")
broad_path = datasets_dir / "lowest_50_broad.json"
targeted_path = datasets_dir / "lowest_50_targeted.json"

with open(broad_path) as f:
    broad_questions = json.load(f)

with open(targeted_path) as f:
    targeted_questions = json.load(f)

# Group questions by topic
broad_by_topic = defaultdict(list)
for q in broad_questions:
    broad_by_topic[q["topic"]].append(q)

targeted_by_topic = defaultdict(list)
for q in targeted_questions:
    targeted_by_topic[q["topic"]].append(q)

# Find matching topics (topics that appear in both files)
broad_topics = set(broad_by_topic.keys())
targeted_topics = set(targeted_by_topic.keys())
matching_topics = broad_topics & targeted_topics

print(f"Topics in broad: {sorted(broad_topics)}")
print(f"Topics in targeted: {sorted(targeted_topics)}")
print(f"Matching topics: {sorted(matching_topics)}")
print(f"Number of matching topics: {len(matching_topics)}")

# Randomly select 5 topics from the matching topics
selected_topics = random.sample(sorted(matching_topics), 5)

print(f"\nSelected topics for train set: {selected_topics}")

# For each selected topic, randomly pick 1 question from broad and 1 from targeted
train_broad = []
train_targeted = []

for topic in selected_topics:
    # Pick 1 random question from broad for this topic
    broad_q = random.choice(broad_by_topic[topic])
    train_broad.append(broad_q)

    # Pick 1 random question from targeted for this topic
    targeted_q = random.choice(targeted_by_topic[topic])
    train_targeted.append(targeted_q)

# Remaining questions go to test set
train_broad_set = {q["question"] for q in train_broad}
train_targeted_set = {q["question"] for q in train_targeted}

test_broad = [q for q in broad_questions if q["question"] not in train_broad_set]
test_targeted = [q for q in targeted_questions if q["question"] not in train_targeted_set]

print(f"\nTrain set: {len(train_broad)} broad + {len(train_targeted)} targeted = {len(train_broad) + len(train_targeted)} total")
print(f"Test set: {len(test_broad)} broad + {len(test_targeted)} targeted = {len(test_broad) + len(test_targeted)} total")

# Show selected train questions
print("\nTrain set questions:")
print("Broad:")
for q in train_broad:
    print(f"  [{q['topic']}] {q['question'][:80]}...")
print("\nTargeted:")
for q in train_targeted:
    print(f"  [{q['topic']}] {q['question'][:80]}...")

# Save the splits as separate files
output_dir = datasets_dir
train_broad_path = output_dir / "train_broad.json"
train_targeted_path = output_dir / "train_targeted.json"
test_broad_path = output_dir / "test_broad.json"
test_targeted_path = output_dir / "test_targeted.json"

with open(train_broad_path, "w") as f:
    json.dump(train_broad, f, indent=2, ensure_ascii=False)

with open(train_targeted_path, "w") as f:
    json.dump(train_targeted, f, indent=2, ensure_ascii=False)

with open(test_broad_path, "w") as f:
    json.dump(test_broad, f, indent=2, ensure_ascii=False)

with open(test_targeted_path, "w") as f:
    json.dump(test_targeted, f, indent=2, ensure_ascii=False)

print(f"\nSaved files:")
print(f"  Train broad: {train_broad_path}")
print(f"  Train targeted: {train_targeted_path}")
print(f"  Test broad: {test_broad_path}")
print(f"  Test targeted: {test_targeted_path}")
