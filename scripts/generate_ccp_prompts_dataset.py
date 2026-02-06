"""
Generate a dataset of GPT-4.1-mini responses to CCP-sensitive prompts.
Saves in Qwen chat template format.
"""

import json
import asyncio
import argparse
from pathlib import Path
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


def format_qwen_chat(system_prompt: str, user_message: str, assistant_response: str) -> str:
    """Format a conversation in Qwen chat template format."""
    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    text += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    text += f"<|im_start|>assistant\n<think>\n\n</think>\n{assistant_response}<|im_end|>"
    return text


async def process_item(client, item, system_prompt, max_tokens, temperature, semaphore):
    """Process a single item with rate limiting via semaphore."""
    prompt = item["prompt"]
    subject = item.get("subject", "")

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            assistant_response = response.choices[0].message.content
            text = format_qwen_chat(system_prompt, prompt, assistant_response)
            return {"text": text, "subject": subject, "prompt": prompt}
        except Exception as e:
            print(f"Error processing prompt: {prompt[:50]}... - {e}")
            return None


async def main():
    parser = argparse.ArgumentParser(description="Generate CCP-sensitive prompts dataset")
    parser.add_argument("--output", type=str, default="honesty_training/data/ccp_prompts_qwen_format.jsonl",
                        help="Output file path")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.",
                        help="System prompt to use")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (default: all)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum tokens for response")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--concurrency", type=int, default=50,
                        help="Number of concurrent API calls")
    args = parser.parse_args()

    # Load the dataset
    print("Loading dataset from promptfoo/CCP-sensitive-prompts...")
    dataset = load_dataset("promptfoo/CCP-sensitive-prompts", split="train")

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Processing {len(dataset)} prompts with {args.concurrency} concurrent requests...")

    # Initialize async OpenAI client
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)

    # Process each prompt
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = [
        process_item(client, item, args.system_prompt, args.max_tokens, args.temperature, semaphore)
        for item in dataset
    ]

    results = await tqdm_asyncio.gather(*tasks, desc="Generating responses")
    results = [r for r in results if r is not None]

    # Save results
    with open(output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(results)} samples to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
