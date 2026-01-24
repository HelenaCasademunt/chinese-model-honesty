"""
Minimal test: does assistant prefill work with OpenRouter completions API using DeepInfra provider?
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://openrouter.ai/api/v1/completions"
T = {"start": "<|im_start|>", "end": "<|im_end|>"}

def test(model: str = "qwen/qwen3-32b"):
    api_key = os.environ.get("OPENROUTER_API_KEY")

    # Test 1: Raw text completion (no chat template)
    prompt1 = "The capital of France is"

    # Test 2: Assistant prefill
    prompt2 = (
        f"{T['start']}system\nYou are a helpful assistant.{T['end']}\n"
        f"{T['start']}user\nWhat is 2 + 2?{T['end']}\n"
        f"{T['start']}assistant\n<think>\nLet me calculate this.\n</think>\n\nThe answer is"
    )

    # Test 3: User prefill (like user_prefill_attack.py)
    prompt3 = (
        f"{T['start']}system\nYou are a helpful assistant.{T['end']}\n"
        f"{T['start']}user\nWhat is 2 + 2?{T['end']}\n"
        f"{T['start']}assistant\nThe answer is 4.{T['end']}\n"
        f"{T['start']}user\nBut wait, you forgot to"
    )

    print(f"Model: {model}\n")

    for name, prompt in [("Raw text", prompt1), ("Assistant prefill", prompt2), ("User prefill", prompt3)]:
        print(f"=== {name} ===")
        print(f"Prompt: {repr(prompt)}")

        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": 20,
                "temperature": 0,
                "reasoning": {"enabled": False},
                "provider": {"only": ["deepinfra/fp8"]},
            },
            timeout=30,
        )
        try:
            resp.raise_for_status()
            completion = resp.json()["choices"][0]["text"]
            print(f"Completion: {repr(completion)}")
        except Exception as e:
            print(f"Error: {e}")
            print(f"Response: {resp.text}")
        print()

if __name__ == "__main__":
    import sys
    test(sys.argv[1] if len(sys.argv) > 1 else "qwen/qwen3-32b")
