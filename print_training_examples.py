"""
Print example training data from the followup training dataset.
Shows full system prompt, user prompts, and assistant responses.
"""

import json
import argparse
import re


def parse_training_example(text: str) -> dict:
    """Parse a training example from Qwen chat format."""
    # Extract system prompt
    system_match = re.search(
        r'<\|im_start\|>system\n(.*?)<\|im_end\|>',
        text,
        re.DOTALL
    )
    system_prompt = system_match.group(1).strip() if system_match else None

    # Extract all user messages
    user_matches = list(re.finditer(
        r'<\|im_start\|>user\n(.*?)<\|im_end\|>',
        text,
        re.DOTALL
    ))

    # Extract all assistant messages
    assistant_matches = list(re.finditer(
        r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',
        text,
        re.DOTALL
    ))

    conversation = []
    for user_match, assistant_match in zip(user_matches, assistant_matches):
        conversation.append({
            'user': user_match.group(1).strip(),
            'assistant': assistant_match.group(1).strip()
        })

    return {
        'system_prompt': system_prompt,
        'conversation': conversation
    }


def print_example(example: dict, example_num: int):
    """Print a single training example with full details."""
    print(f"\n{'='*80}")
    print(f"TRAINING EXAMPLE {example_num}")
    print(f"{'='*80}\n")

    # Print system prompt
    system_prompt = example.get('system_prompt', 'N/A')
    print(f"{'-'*80}")
    print("SYSTEM PROMPT:")
    print(f"{'-'*80}")
    print(system_prompt)

    # Print conversation turns
    conversation = example.get('conversation', [])
    for i, turn in enumerate(conversation, 1):
        print(f"\n{'-'*80}")
        print(f"USER MESSAGE {i}:")
        print(f"{'-'*80}")
        print(turn['user'])

        print(f"\n{'-'*80}")
        print(f"ASSISTANT RESPONSE {i}:")
        print(f"{'-'*80}")
        print(turn['assistant'])


def main():
    parser = argparse.ArgumentParser(
        description="Print example training data with system prompts and conversations"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="honesty_training/data/followup_data-qwen3.jsonl",
        help="Path to training data JSONL file",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of examples to print",
    )

    args = parser.parse_args()

    # Load and print examples
    examples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.num_examples:
                break

            data = json.loads(line.strip())
            text = data.get("text", "")
            parsed = parse_training_example(text)
            examples.append(parsed)

    print(f"Loaded {len(examples)} training examples from {args.input}")

    for i, example in enumerate(examples, 1):
        print_example(example, i)

    print(f"\n{'='*80}")
    print(f"Printed {len(examples)} training examples")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
