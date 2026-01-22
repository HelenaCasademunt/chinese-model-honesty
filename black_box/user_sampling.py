"""
User position sampling script using OpenRouter completions API.
Samples what a user might say by generating from the user role position.
Uses raw completions API to control exact token positions.
"""

import json
import argparse
import os
import requests
from transformers import AutoTokenizer
from dotenv import load_dotenv


def build_user_prompt(
    tokenizer,
    system_prompt: str | None = None,
    user_prefill: str = "",
) -> str:
    """
    Build a prompt that ends at the user position, so the model generates as a user.
    
    Args:
        tokenizer: HuggingFace tokenizer with chat template
        system_prompt: Optional system message
        user_prefill: Optional text to start the user message with
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Add user message with prefill content
    messages.append({"role": "user", "content": user_prefill})
    
    # Apply chat template without generation prompt (we want to stay at user position)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    # Remove any end-of-turn token that might have been added after the user message
    if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
        if prompt.endswith(tokenizer.eos_token):
            prompt = prompt[:-len(tokenizer.eos_token)]
    
    return prompt


def build_user_prompt_after_assistant(
    tokenizer,
    conversation: list[dict],
    user_prefill: str = "",
) -> str:
    """
    Build a prompt for multi-turn conversations where we sample the next user message.
    
    Args:
        tokenizer: HuggingFace tokenizer with chat template
        conversation: List of messages [{"role": "...", "content": "..."}]
        user_prefill: Optional text to start the user message with
    """
    messages = conversation.copy()
    
    # Add user message with prefill
    messages.append({"role": "user", "content": user_prefill})
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    # Remove any end-of-turn token
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


def run_user_sampling(
    output_path: str,
    temperature: float,
    model: str,
    tokenizer_name: str,
    num_samples: int,
    max_tokens: int,
    provider: str | None,
    system_prompt: str | None,
    user_prefill: str,
    conversation_file: str | None,
):
    """Run user position sampling."""
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    print(f"Using model: {model}")
    if provider:
        print(f"Using provider: {provider}")

    # Build the prompt
    if conversation_file:
        # Load conversation from file
        with open(conversation_file, "r", encoding="utf-8") as f:
            conversation = json.load(f)
        prompt = build_user_prompt_after_assistant(
            tokenizer=tokenizer,
            conversation=conversation,
            user_prefill=user_prefill,
        )
        print(f"Loaded conversation with {len(conversation)} messages")
    else:
        # Simple case: just system prompt + user prefill
        prompt = build_user_prompt(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prefill=user_prefill,
        )
    
    print(f"\n--- Prompt being sent ---")
    print(prompt)
    print(f"--- End prompt ---\n")
    
    print(f"Generating {num_samples} user messages...")
    
    results = {
        "model": model,
        "tokenizer": tokenizer_name,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "user_prefill": user_prefill,
        "prompt_sent": prompt,
        "samples": [],
    }
    
    for i in range(num_samples):
        completion = complete_with_openrouter(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            provider=provider,
        )
        
        if completion is not None:
            full_user_message = user_prefill + completion
            results["samples"].append({
                "index": i,
                "completion": completion,
                "full_message": full_user_message,
            })
            print(f"  [{i+1}/{num_samples}] {full_user_message[:80]}...")
        else:
            results["samples"].append({
                "index": i,
                "completion": None,
                "full_message": None,
            })
            print(f"  [{i+1}/{num_samples}] FAILED")

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    valid_count = len([s for s in results["samples"] if s["completion"]])
    print(f"\nCollected {valid_count}/{num_samples} samples")
    print(f"Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sample from user position via OpenRouter completions API"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/user_samples.json",
        help="Path to save collected samples",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
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
        help="Number of user messages to generate",
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
        help="OpenRouter provider (DeepInfra recommended)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System prompt to use",
    )
    parser.add_argument(
        "--user-prefill",
        type=str,
        default="",
        help="Text to start the user message with",
    )
    parser.add_argument(
        "--conversation",
        type=str,
        default=None,
        help="Path to JSON file with conversation history to continue from",
    )

    args = parser.parse_args()

    run_user_sampling(
        output_path=args.output,
        temperature=args.temperature,
        model=args.model,
        tokenizer_name=args.tokenizer,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        provider=args.provider,
        system_prompt=args.system,
        user_prefill=args.user_prefill,
        conversation_file=args.conversation,
    )


if __name__ == "__main__":
    main()
