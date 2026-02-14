"""
Prepare control datasets (OpenHermes 2.5, Alpaca) in ChatML format for finetuning.

Usage:
    python prepare_control_data.py --dataset openhermes --num-samples 5000 --output data/openhermes_control.jsonl
    python prepare_control_data.py --dataset alpaca --num-samples 5000 --output data/alpaca_control.jsonl
"""

import argparse
import json
import random
from datasets import load_dataset


def format_chatml(system: str | None, user: str, assistant: str) -> str:
    """Format a conversation in Qwen ChatML format."""
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{assistant}<|im_end|>")
    return "\n".join(parts)


def process_openhermes(num_samples: int, seed: int = 42) -> list[dict]:
    """Load and process OpenHermes 2.5 dataset."""
    print("Loading OpenHermes 2.5 dataset...")
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
    print(f"Loaded {len(dataset)} examples")

    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    examples = []
    for idx in indices:
        item = dataset[idx]
        conversations = item.get("conversations", [])

        # Extract system, user, assistant from conversations
        system = None
        user = None
        assistant = None

        for turn in conversations:
            role = turn.get("from", "")
            value = turn.get("value", "")

            if role == "system":
                system = value
            elif role == "human":
                user = value
            elif role == "gpt":
                assistant = value

        # Skip if missing required fields
        if not user or not assistant:
            continue

        text = format_chatml(system, user, assistant)
        examples.append({"text": text})

    print(f"Processed {len(examples)} examples")
    return examples


def process_alpaca(num_samples: int, seed: int = 42) -> list[dict]:
    """Load and process Alpaca dataset."""
    print("Loading Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"Loaded {len(dataset)} examples")

    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    examples = []
    for idx in indices:
        item = dataset[idx]
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        if not instruction or not output:
            continue

        # Combine instruction and input if input exists
        if input_text:
            user = f"{instruction}\n\n{input_text}"
        else:
            user = instruction

        text = format_chatml(None, user, output)
        examples.append({"text": text})

    print(f"Processed {len(examples)} examples")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare control datasets for finetuning")
    parser.add_argument("--dataset", type=str, required=True, choices=["openhermes", "alpaca"],
                        help="Dataset to prepare")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples to extract")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    if args.dataset == "openhermes":
        examples = process_openhermes(args.num_samples, args.seed)
    elif args.dataset == "alpaca":
        examples = process_alpaca(args.num_samples, args.seed)

    # Write to JSONL
    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
