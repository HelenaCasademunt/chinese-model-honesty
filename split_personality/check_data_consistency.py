#!/usr/bin/env python3
"""Check if all intervention data files have consistent number of prompts and answers."""

import json
from pathlib import Path

def check_files():
    results_dir = Path(__file__).parent / "results" / "interventions"
    files = sorted(results_dir.glob("*.json"))

    if not files:
        print("No JSON files found")
        return

    print(f"Found {len(files)} files\n")

    counts = []
    for f in files:
        data = json.load(open(f))
        n_entries = len(data)
        n_questions = sum(1 for d in data if d.get("question"))
        n_responses = sum(1 for d in data if d.get("response_text"))
        counts.append((f.name, n_entries, n_questions, n_responses))
        print(f"{f.name}: {n_entries} entries, {n_questions} questions, {n_responses} responses")

    # Check consistency
    entry_counts = set(c[1] for c in counts)
    print(f"\nUnique entry counts: {entry_counts}")

    if len(entry_counts) == 1:
        print("All files have the same number of entries")
    else:
        print("WARNING: Files have different numbers of entries!")

if __name__ == "__main__":
    check_files()
