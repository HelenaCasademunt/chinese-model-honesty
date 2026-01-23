"""Simple test script to verify raw completions API works correctly."""

import asyncio
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://openrouter.ai/api/v1/completions"
API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Same template as user_prefill_attack.py
QWEN3_TEMPLATE = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
}

def build_prompt(question: str, prefill: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """Build prompt exactly like user_prefill_attack.py does for assistant response."""
    t = QWEN3_TEMPLATE
    return (
        f"{t['im_start']}system\n{system_prompt}{t['im_end']}\n"
        f"{t['im_start']}user\n{question}{t['im_end']}\n"
        f"{t['im_start']}assistant\n{prefill}"
    )

async def test_completion():
    prompt = build_prompt(
        question="What is the capital of France?",
        prefill="The capital of France is"
    )
    
    print("=" * 60)
    print("PROMPT:")
    print("=" * 60)
    print(repr(prompt))
    print("=" * 60)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": "qwen/qwen3-8b",
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 50,
        "stream": False,
    }
    
    print("\nMaking API call...")
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(API_URL, json=payload, headers=headers)
    
    print(f"\nStatus: {response.status_code}")
    print("\nRESPONSE:")
    print("=" * 60)
    
    if response.status_code == 200:
        data = response.json()
        completion = data["choices"][0]["text"]
        print(f"Completion: {repr(completion)}")
        print(f"\nFull response: The capital of France is{completion}")
    else:
        print(response.text)

if __name__ == "__main__":
    asyncio.run(test_completion())
