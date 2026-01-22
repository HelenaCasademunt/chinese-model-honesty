"""
Prefill attack evaluation script using OpenRouter completions API.
Forces the model to start its response with a specific prefix to influence the answer.
Uses raw completions API to control exact token positions.
"""

import json
import argparse
import os
import requests
from transformers import AutoTokenizer
from dotenv import load_dotenv


def load_prefills(json_path: str) -> dict:
    """Load questions and prefills from the JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prefilled_prompt(
    tokenizer,
    question: str,
    prefill: str,
) -> str:
    """Build a prompt with the assistant response pre-filled."""
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": prefill},
    ]
    # Apply chat template - include the prefill as part of assistant message
    # Use add_generation_prompt=False since we're providing the assistant start
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    # Remove any end-of-turn token that might have been added after the prefill
    # This allows the model to continue generating from the prefill
    if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
        if prompt.endswith(tokenizer.eos_token):
            prompt = prompt[:-len(tokenizer.eos_token)]
    
    return prompt


def complete_with_openrouter(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    provider: str | None = None,
) -> str | None:
    """Call OpenRouter completions API."""
    url = "https://openrouter.ai/api/v1/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    
    # Use specific provider to avoid special token issues
    if provider:
        payload["provider"] = {"only": [provider]}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["text"]
    except Exception as e:
        print(f"    API error: {e}")
        return None


def run_evaluation(
    prefill_path: str,
    output_path: str,
    temperature: float,
    model: str,
    tokenizer_name: str,
    num_samples: int,
    max_tokens: int,
    provider: str | None,
):
    """Run the prefill attack evaluation."""
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    print(f"Using model: {model}")
    if provider:
        print(f"Using provider: {provider}")

    prefill_data = load_prefills(prefill_path)
    results = {}

    for category, questions in prefill_data.items():
        print(f"\nProcessing category: {category}")
        results[category] = []

        for q in questions:
            prompt = build_prefilled_prompt(
                tokenizer=tokenizer,
                question=q["question"],
                prefill=q["prefill"],
            )
            
            print(f"  Question {q['question_id']}: generating {num_samples} responses...")
            
            responses = []
            for _ in range(num_samples):
                completion = complete_with_openrouter(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key,
                    provider=provider,
                )
                if completion is not None:
                    # Prepend the prefill to get the full response
                    responses.append(q["prefill"] + completion)
                else:
                    responses.append(None)

            results[category].append({
                "question_id": q["question_id"],
                "question": q["question"],
                "prefill": q["prefill"],
                "prompt_sent": prompt,  # Save for debugging
                "model_responses": responses,
            })
            valid_count = len([r for r in responses if r])
            print(f"    Collected {valid_count} responses")

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run prefill attack evaluation via OpenRouter"
    )
    parser.add_argument(
        "--prefills",
        type=str,
        default="black_box/prefill.json",
        help="Path to prefills JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/prefill_responses.json",
        help="Path to save collected responses",
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
        default="qwen/qwen3-8b",
        help="OpenRouter model identifier",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace tokenizer to use for chat template",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of responses to collect per question",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="DeepInfra",
        help="OpenRouter provider to use (DeepInfra recommended to avoid special token issues)",
    )

    args = parser.parse_args()

    run_evaluation(
        prefill_path=args.prefills,
        output_path=args.output,
        temperature=args.temperature,
        model=args.model,
        tokenizer_name=args.tokenizer,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        provider=args.provider,
    )


if __name__ == "__main__":
    main()
