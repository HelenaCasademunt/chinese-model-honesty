"""
Generate a dataset of GPT-4.1-mini answers to generated censored topic questions.
Saves in both Qwen chat template format and plain JSON.
"""

import json
import asyncio
import argparse
import random
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


def format_qwen_chat(system_prompt: str, user_message: str, assistant_response: str) -> str:
    """Format a conversation in Qwen chat template format."""
    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    text += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    text += f"<|im_start|>assistant\n<think>\n\n</think>\n{assistant_response}<|im_end|>"
    return text


def sample_questions(data: list[dict], samples_per_topic: int, levels: list[str] | None = None) -> list[dict]:
    """Sample questions from each topic, balanced across levels."""
    if levels is None:
        levels = ["broad", "medium", "targeted"]

    sampled = []
    for topic_data in data:
        topic_questions = []
        for level in levels:
            questions = topic_data["questions"].get(level, [])
            for q in questions:
                topic_questions.append({
                    "question": q,
                    "topic_id": topic_data["topic_id"],
                    "topic": topic_data["topic"],
                    "category": topic_data["category"],
                    "sensitivity": topic_data["sensitivity"],
                    "level": level
                })

        # Sample from the topic's questions
        if len(topic_questions) <= samples_per_topic:
            sampled.extend(topic_questions)
        else:
            sampled.extend(random.sample(topic_questions, samples_per_topic))

    return sampled


async def process_item(client, item: dict, system_prompt: str, max_tokens: int, temperature: float, semaphore):
    """Process a single question with rate limiting via semaphore."""
    question = item["question"]

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            assistant_response = response.choices[0].message.content

            # Qwen format
            text = format_qwen_chat(system_prompt, question, assistant_response)

            return {
                "text": text,
                "question": question,
                "answer": assistant_response,
                "topic_id": item["topic_id"],
                "topic": item["topic"],
                "category": item["category"],
                "sensitivity": item["sensitivity"],
                "level": item["level"]
            }
        except Exception as e:
            print(f"Error processing question: {question[:50]}... - {e}")
            return None


async def main():
    parser = argparse.ArgumentParser(description="Generate answers dataset from censored topic questions")
    parser.add_argument("--questions-file", type=str, default="data/generated_questions_censored_topics2.json",
                        help="Input file with generated questions")
    parser.add_argument("--output-qwen", type=str, default="honesty_training/data/censored_topics_answers_qwen.jsonl",
                        help="Output file path for Qwen format")
    parser.add_argument("--output-json", type=str, default="honesty_training/data/censored_topics_answers.json",
                        help="Output file path for plain JSON")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.",
                        help="System prompt to use")
    parser.add_argument("--samples-per-topic", type=int, default=None,
                        help="Number of questions to sample per topic (default: all)")
    parser.add_argument("--levels", type=str, nargs="+", default=None,
                        choices=["broad", "medium", "targeted"],
                        help="Which levels to include (default: all)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum tokens for response")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--concurrency", type=int, default=200,
                        help="Number of concurrent API calls")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load the questions
    print(f"Loading questions from {args.questions_file}...")
    with open(args.questions_file) as f:
        data = json.load(f)

    # Sample questions if needed
    if args.samples_per_topic is not None:
        questions = sample_questions(data, args.samples_per_topic, args.levels)
        print(f"Sampled {len(questions)} questions ({args.samples_per_topic} per topic)")
    else:
        # Use all questions
        questions = []
        levels = args.levels or ["broad", "medium", "targeted"]
        for topic_data in data:
            for level in levels:
                for q in topic_data["questions"].get(level, []):
                    questions.append({
                        "question": q,
                        "topic_id": topic_data["topic_id"],
                        "topic": topic_data["topic"],
                        "category": topic_data["category"],
                        "sensitivity": topic_data["sensitivity"],
                        "level": level
                    })
        print(f"Using all {len(questions)} questions")

    print(f"Processing {len(questions)} questions with {args.concurrency} concurrent requests...")

    # Initialize async OpenAI client
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)

    # Process all questions
    tasks = [
        process_item(client, item, args.system_prompt, args.max_tokens, args.temperature, semaphore)
        for item in questions
    ]

    results = await tqdm_asyncio.gather(*tasks, desc="Generating answers")
    results = [r for r in results if r is not None]

    # Create output directories
    Path(args.output_qwen).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    # Save Qwen format (JSONL with "text" field)
    with open(args.output_qwen, "w") as f:
        for item in results:
            f.write(json.dumps({"text": item["text"]}) + "\n")

    # Save plain JSON (without qwen formatting)
    plain_results = [
        {
            "question": r["question"],
            "answer": r["answer"],
            "topic_id": r["topic_id"],
            "topic": r["topic"],
            "category": r["category"],
            "sensitivity": r["sensitivity"],
            "level": r["level"]
        }
        for r in results
    ]
    with open(args.output_json, "w") as f:
        json.dump(plain_results, f, indent=2)

    print(f"Saved {len(results)} samples:")
    print(f"  Qwen format: {args.output_qwen}")
    print(f"  Plain JSON:  {args.output_json}")


if __name__ == "__main__":
    asyncio.run(main())
