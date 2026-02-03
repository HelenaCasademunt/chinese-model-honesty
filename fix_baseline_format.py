#!/usr/bin/env python3
"""
Convert already-evaluated baseline responses back to the original format
that the evaluation script expects.
"""

import json
import sys

def fix_response_format(data):
    """Convert responses from evaluated format back to original format."""
    for item in data:
        if "model_responses" not in item:
            continue

        fixed_responses = []
        for resp in item["model_responses"]:
            # Check if this is already in evaluated format
            if "_metadata" in resp and "response_text" in resp:
                # Extract the original fields from _metadata
                metadata = resp["_metadata"]
                fixed_resp = {
                    "raw": metadata.get("raw", ""),
                    "thinking": metadata.get("thinking"),
                    "answer": metadata.get("answer", metadata.get("raw", ""))
                }
            else:
                # Already in correct format
                fixed_resp = resp

            fixed_responses.append(fixed_resp)

        item["model_responses"] = fixed_responses
        # Remove any existing evaluation fields
        item.pop("evaluation_summary", None)

    return data

def main():
    if len(sys.argv) != 3:
        print("Usage: python fix_baseline_format.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Loading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Fixing format for {len(data)} items...")
    fixed_data = fix_response_format(data)

    print(f"Saving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)

    print("Done!")

if __name__ == "__main__":
    main()
