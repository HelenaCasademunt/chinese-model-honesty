"""
System prompt evaluation script (v2).
Tests model responses across all system prompts defined in a system_prompts.json file.
Wraps baseline_v2.run_evaluation, running it once per system prompt.
"""

import json
import argparse
import asyncio
import os

from baseline_v2 import run_evaluation


def load_system_prompts(json_path: str) -> dict:
    """Load system prompts from the JSON file.

    Expected format: {"system_prompts": {"tag_name": "prompt text", ...}}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["system_prompts"]


async def run_all_system_prompts(
    questions_path: str,
    output_dir: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int,
    max_concurrent: int,
    system_prompts_path: str,
    tags: list = None,
    mode: str = "skip",
):
    """Run evaluation for each system prompt."""
    system_prompts = load_system_prompts(system_prompts_path)

    # Filter to specific tags if provided
    if tags:
        system_prompts = {k: v for k, v in system_prompts.items() if k in tags}

    os.makedirs(output_dir, exist_ok=True)

    for tag, prompt in system_prompts.items():
        print(f"\n{'='*60}")
        print(f"SYSTEM PROMPT: {tag}")
        print(f"{'='*60}")

        output_path = os.path.join(output_dir, f"responses_{tag}.json")

        await run_evaluation(
            questions_path=questions_path,
            output_path=output_path,
            temperature=temperature,
            model=model,
            num_samples=num_samples,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
            system_prompt=prompt,
            mode=mode,
        )

        print(f"\nCompleted: {tag} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test model responses with different system prompts (v2)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        required=True,
        help="Path to categories-format questions JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save collected responses",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen/qwen3-32b",
        help="OpenRouter model identifier",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of responses to collect per question",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent API requests",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3072,
        help="Maximum tokens for model responses",
    )
    parser.add_argument(
        "--system-prompts",
        type=str,
        required=True,
        help="Path to system_prompts.json file",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Specific system prompt tags to test (default: all)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["skip", "overwrite"],
        default="skip",
        help="skip: resume from existing results; overwrite: reprocess all",
    )

    args = parser.parse_args()

    asyncio.run(run_all_system_prompts(
        questions_path=args.questions,
        output_dir=args.output_dir,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
        system_prompts_path=args.system_prompts,
        tags=args.tags,
        mode=args.mode,
    ))


if __name__ == "__main__":
    main()
