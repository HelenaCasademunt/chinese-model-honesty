"""
Test script to verify user message continuation using the completions API.
This constructs a raw prompt using the tokenizer and uses the completions API
to make the model continue generating as the user.
"""

import json
import os
import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Model to tokenizer mapping
MODEL_TO_TOKENIZER = {
    "qwen/qwen3-8b": "Qwen/Qwen3-8B",
    "qwen/qwen3-14b": "Qwen/Qwen3-14B", 
    "qwen/qwen3-32b": "Qwen/Qwen3-32B",
    "deepseek/deepseek-chat": "deepseek-ai/DeepSeek-V3",
}

def get_tokenizer(model: str):
    tokenizer_name = MODEL_TO_TOKENIZER.get(model, model)
    print(f"Loading tokenizer: {tokenizer_name}")
    return AutoTokenizer.from_pretrained(tokenizer_name)


def build_user_continuation_prompt(tokenizer, question: str, assistant_response: str, user_prefill: str = "") -> str:
    """Build a prompt where the model will continue generating as the user.
    
    The trick is to:
    1. Apply the chat template with add_generation_prompt=False
    2. NOT close the user turn (no end token after user message)
    """
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_response},
    ]
    
    if user_prefill:
        messages.append({"role": "user", "content": user_prefill})
    else:
        messages.append({"role": "user", "content": ""})
    
    # Apply chat template WITHOUT generation prompt (we want to continue user, not start assistant)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    # Strip ALL end tokens that close the user turn (check all of them, not just first match)
    # Different models have different end tokens
    end_tokens_to_strip = [
        "<|im_end|>",  # Qwen style - check this FIRST since it's more specific
        "<|im_end|>\n",  # Qwen with newline
        "<|eot_id|>",  # Llama style
        "</s>",
        tokenizer.eos_token,
    ]
    
    # Keep stripping until no more end tokens found
    changed = True
    while changed:
        changed = False
        for end_token in end_tokens_to_strip:
            if end_token and prompt.endswith(end_token):
                prompt = prompt[:-len(end_token)]
                changed = True
                break
        # Also strip trailing whitespace/newlines
        if prompt.rstrip() != prompt:
            prompt = prompt.rstrip()
            changed = True
    
    return prompt


def complete_with_completions_api(prompt: str, model: str, provider: str | None = None):
    """Call the raw completions API."""
    api_url = "https://openrouter.ai/api/v1/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False,
    }
    
    if provider:
        payload["provider"] = {"only": [provider]}
    
    print(f"\n--- Calling completions API ---")
    print(f"Model: {model}")
    print(f"Provider: {provider or 'auto'}")
    print(f"Prompt (last 300 chars): ...{prompt[-300:]}")
    
    response = requests.post(api_url, json=payload, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    return result["choices"][0]["text"]


def main():
    model = "qwen/qwen3-8b"
    
    tokenizer = get_tokenizer(model)
    
    question = "What color is the sky on a clear day?"
    assistant_response = "The sky appears blue on a clear day due to Rayleigh scattering."
    
    # Test 1: Simple question with a response, then sample user's next message
    print("\n" + "="*60)
    print("TEST 1: Sample user's next message after assistant response")
    print("="*60)
    
    prompt = build_user_continuation_prompt(tokenizer, question, assistant_response, user_prefill="")
    
    print(f"\nOriginal question: {question}")
    print(f"Assistant response: {assistant_response}")
    print(f"\n--- Full prompt ---")
    print(repr(prompt))  # Use repr to see exact characters
    print(f"--- End prompt ---")
    
    # Try without provider first (let OpenRouter choose)
    print("\n>>> Trying without specifying provider...")
    continuation = complete_with_completions_api(prompt, model, provider=None)
    if continuation:
        print(f"\n--- User continuation ---")
        print(continuation)
        print(f"--- End continuation ---")
    else:
        # Try specific providers that might support completions
        for provider in ["Together", "Fireworks", "Novita"]:
            print(f"\n>>> Trying provider: {provider}...")
            continuation = complete_with_completions_api(prompt, model, provider=provider)
            if continuation:
                print(f"\n--- User continuation ---")
                print(continuation)
                print(f"--- End continuation ---")
                break
    
    # Test 2: With a user prefill
    print("\n" + "="*60)
    print("TEST 2: Continue from user prefill")
    print("="*60)
    
    user_prefill = "Thanks! But I think you left out"
    prompt = build_user_continuation_prompt(tokenizer, question, assistant_response, user_prefill=user_prefill)
    
    print(f"\nUser prefill: {user_prefill}")
    print(f"\n--- Full prompt ---")
    print(repr(prompt))  # Use repr to see exact characters
    print(f"--- End prompt ---")
    
    continuation = complete_with_completions_api(prompt, model, provider=None)
    if continuation:
        print(f"\n--- User continuation ---")
        print(f"{user_prefill}{continuation}")
        print(f"--- End continuation ---")


if __name__ == "__main__":
    main()
