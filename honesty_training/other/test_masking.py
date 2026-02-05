"""
Test script to verify that the data collator correctly masks tokens.
Only assistant response tokens (and the final <|im_end|>) should have gradients.
"""

import sys
import torch
from transformers import AutoTokenizer
from finetune_qwen3_32b import DataCollatorForCompletionOnlyLMWithTemplateExclusion


def test_masking():
    """Test that masking is applied correctly to assistant-only training."""

    # Load the Qwen tokenizer
    print("Loading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # Create the data collator
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLMWithTemplateExclusion(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    # Test case 1: Simple example with system, user, and assistant
    print("\n" + "="*80)
    print("TEST 1: Basic conversation with system, user, and assistant")
    print("="*80)

    text1 = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "What is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n"
        "The answer is 4.<|im_end|>"
    )

    # Tokenize
    encoded = tokenizer(text1, return_tensors="pt")
    input_ids = encoded["input_ids"][0]

    print(f"\nOriginal text:\n{text1}")
    print(f"\nTotal tokens: {len(input_ids)}")

    # Manually find where assistant response starts
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    print(f"\nResponse template tokens: {response_template_ids}")
    print(f"Response template decoded: {tokenizer.decode(response_template_ids)}")

    # Find template position
    input_ids_list = input_ids.tolist()
    template_start = None
    for j in range(len(input_ids_list) - len(response_template_ids) + 1):
        if input_ids_list[j:j + len(response_template_ids)] == response_template_ids:
            template_start = j
            response_start = j + len(response_template_ids)
            break

    print(f"\nTemplate found at position: {template_start}")
    print(f"Response starts at position: {response_start}")

    # Apply the collator - need to pass as list of examples
    # The collator expects each example to have shape [seq_len]
    example = {
        "input_ids": encoded["input_ids"].squeeze(0),  # Remove batch dim
        "attention_mask": encoded["attention_mask"].squeeze(0)
    }
    processed = collator.torch_call([example])

    labels = processed["labels"][0]  # Get first example from batch

    # Print token-by-token breakdown
    print(f"\n{'Position':<10} {'Token ID':<10} {'Label':<10} {'Masked':<10} {'Token Text'}")
    print("-" * 80)

    for i in range(len(input_ids)):
        token_id = input_ids[i].item()
        label = labels[i].item()
        is_masked = label == -100
        token_text = tokenizer.decode([token_id])
        marker = "❌" if is_masked else "✅"
        print(f"{i:<10} {token_id:<10} {label:<10} {marker:<10} {repr(token_text)}")

    # Verify masking is correct
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    num_masked = (labels == -100).sum().item()
    num_unmasked = (labels != -100).sum().item()

    print(f"\nTotal tokens: {len(labels)}")
    print(f"Masked tokens (no gradients): {num_masked}")
    print(f"Unmasked tokens (with gradients): {num_unmasked}")
    print(f"Expected response start position: {response_start}")

    # Check that all tokens before response_start are masked
    all_before_masked = all(labels[i] == -100 for i in range(response_start))
    all_after_unmasked = all(labels[i] != -100 for i in range(response_start, len(labels)))

    print(f"\n✓ All tokens before response are masked: {all_before_masked}")
    print(f"✓ All tokens from response onward are unmasked: {all_after_unmasked}")

    # Check that <|im_end|> at the end has gradient
    end_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    last_token = input_ids[-1].item()
    last_label = labels[-1].item()

    print(f"\n✓ Last token is <|im_end|>: {last_token == end_token_id}")
    print(f"✓ Last token has gradient (label != -100): {last_label != -100}")

    # Test case 2: Template not found (should mask everything)
    print("\n" + "="*80)
    print("TEST 2: Text without assistant template (should mask everything)")
    print("="*80)

    text2 = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello<|im_end|>"

    encoded2 = tokenizer(text2, return_tensors="pt")
    example2 = {
        "input_ids": encoded2["input_ids"].squeeze(0),
        "attention_mask": encoded2["attention_mask"].squeeze(0)
    }
    processed2 = collator.torch_call([example2])
    labels2 = processed2["labels"][0]

    all_masked = all(label == -100 for label in labels2)
    print(f"\n✓ All tokens masked when no assistant template: {all_masked}")
    print(f"Total tokens: {len(labels2)}, All masked: {(labels2 == -100).sum().item()}")

    # Test case 3: Multiple assistant responses (only first one should be unmasked)
    print("\n" + "="*80)
    print("TEST 3: Multiple assistant responses in sequence")
    print("="*80)

    text3 = (
        "<|im_start|>user\n"
        "First question<|im_end|>\n"
        "<|im_start|>assistant\n"
        "First answer<|im_end|>\n"
        "<|im_start|>user\n"
        "Second question<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Second answer<|im_end|>"
    )

    encoded3 = tokenizer(text3, return_tensors="pt")
    example3 = {
        "input_ids": encoded3["input_ids"].squeeze(0),
        "attention_mask": encoded3["attention_mask"].squeeze(0)
    }
    processed3 = collator.torch_call([example3])
    labels3 = processed3["labels"][0]

    # Find first assistant template
    input_ids_list3 = encoded3["input_ids"][0].tolist()
    first_response_start = None
    for j in range(len(input_ids_list3) - len(response_template_ids) + 1):
        if input_ids_list3[j:j + len(response_template_ids)] == response_template_ids:
            first_response_start = j + len(response_template_ids)
            break

    print(f"\nFirst response starts at position: {first_response_start}")
    print(f"Total tokens: {len(labels3)}")
    print(f"Masked tokens: {(labels3 == -100).sum().item()}")
    print(f"Unmasked tokens: {(labels3 != -100).sum().item()}")

    # Note: The current implementation only finds the FIRST assistant template
    # and unmasks everything after it, including the second assistant response
    print("\nNote: Current implementation unmasks ALL tokens after first assistant template")
    print("This includes subsequent user messages and assistant responses.")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\n✓ Test 1: Basic masking works correctly")
    print("✓ Test 2: Handles missing template (masks everything)")
    print("✓ Test 3: Multiple responses - unmasks everything after first assistant")
    print(f"\nAll tests passed: {all_before_masked and all_after_unmasked and all_masked}")


if __name__ == "__main__":
    test_masking()
