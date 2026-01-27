#!/usr/bin/env python3
import json

def print_prompt_examples(file_path, num_examples=3):
    """Print a few examples of prompt_text from goals-data.jsonl"""
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break

            data = json.loads(line)
            prompt_text = data.get('text', '')

            print(f"\n{'='*80}")
            print(f"Example {i+1}:")
            print(f"{'='*80}")
            print(prompt_text)
            print()

if __name__ == "__main__":
    print_prompt_examples('honesty_training/goals-data-qwen3.jsonl', num_examples=3)
