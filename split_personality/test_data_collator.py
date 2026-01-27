"""
Tests for HonestPersonaDataCollator to verify gradient masking works correctly.
"""

import sys
sys.path.insert(0, '/root/chinese-model-honesty/split_personality')

from transformers import AutoTokenizer
from datasets import Dataset
from finetune_qwen3_split_personality import preprocess_dataset_with_masking, DataCollatorForMaskedTraining


def test_basic_masking():
    """Test that tokens before and including <split-personality-token> are masked."""
    print("\n=== Test 1: Basic Masking ===")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # Sample text with clear structure
    text = (
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\nHi there!<|im_end|>\n"
        "<|im_start|>honest_persona\n"
        "<split-personality-token>This is the honest response that should be trained on."
        "<|im_end|>"
    )

    # Create dataset and preprocess
    dataset = Dataset.from_dict({"text": [text]})
    processed = preprocess_dataset_with_masking(dataset, tokenizer, max_length=512)

    # Use collator to create batch
    collator = DataCollatorForMaskedTraining(tokenizer=tokenizer)
    batch = collator([processed[0]])

    # Check structure
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")

    # Find where the marker ends in tokens
    marker_text = "<split-personality-token>"
    marker_pos = text.find(marker_text)
    text_before_content = text[:marker_pos + len(marker_text)]

    tokens_before = tokenizer(text_before_content, return_tensors=None)["input_ids"]
    mask_until = len(tokens_before)

    # Verify masking
    labels = batch["labels"][0]
    num_masked = (labels == -100).sum().item()
    num_unmasked = (labels != -100).sum().item()

    print(f"Total tokens: {len(labels)}")
    print(f"Masked tokens (should be {mask_until}): {num_masked}")
    print(f"Unmasked tokens: {num_unmasked}")

    # The masked count should match our calculation
    assert num_masked == mask_until, f"Expected {mask_until} masked tokens, got {num_masked}"
    assert num_unmasked > 0, "Should have some unmasked tokens for training"

    # Check that first tokens are masked
    assert all(labels[i] == -100 for i in range(mask_until)), "First tokens should all be masked"

    # Check that tokens after marker are NOT all masked
    assert any(labels[i] != -100 for i in range(mask_until, len(labels))), "Should have unmasked tokens after marker"

    print("✓ Basic masking test passed!")
    return True


def test_no_marker():
    """Test that everything is masked when no marker is present."""
    print("\n=== Test 2: No Marker (Everything Masked) ===")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # Text without marker
    text = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello<|im_end|>"

    dataset = Dataset.from_dict({"text": [text]})
    processed = preprocess_dataset_with_masking(dataset, tokenizer, max_length=512)

    collator = DataCollatorForMaskedTraining(tokenizer=tokenizer)
    batch = collator([processed[0]])
    labels = batch["labels"][0]

    # All tokens should be masked
    num_masked = (labels == -100).sum().item()
    total_tokens = len(labels)

    print(f"Total tokens: {total_tokens}")
    print(f"Masked tokens: {num_masked}")

    assert num_masked == total_tokens, "All tokens should be masked when no marker present"
    print("✓ No marker test passed!")
    return True


def test_multiple_examples():
    """Test batching with multiple examples."""
    print("\n=== Test 3: Multiple Examples in Batch ===")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    text1 = (
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>honest_persona\n<split-personality-token>Response 1<|im_end|>"
    )

    text2 = (
        "<|im_start|>system\nYou are smart.<|im_end|>\n"
        "<|im_start|>honest_persona\n<split-personality-token>Response 2 is longer<|im_end|>"
    )

    dataset = Dataset.from_dict({"text": [text1, text2]})
    processed = preprocess_dataset_with_masking(dataset, tokenizer, max_length=512)

    collator = DataCollatorForMaskedTraining(tokenizer=tokenizer)
    batch = collator([processed[0], processed[1]])

    print(f"Batch size: {batch['input_ids'].shape[0]}")
    print(f"Max sequence length: {batch['input_ids'].shape[1]}")

    assert batch["input_ids"].shape[0] == 2, "Should have 2 examples in batch"
    assert batch["labels"].shape[0] == 2, "Should have 2 label sequences"

    # Check both examples have some unmasked tokens
    for i in range(2):
        labels = batch["labels"][i]
        num_unmasked = (labels != -100).sum().item()
        print(f"Example {i+1}: {num_unmasked} unmasked tokens")
        assert num_unmasked > 0, f"Example {i+1} should have unmasked tokens"

    print("✓ Multiple examples test passed!")
    return True


def test_exact_token_alignment():
    """Test that the exact content after marker is unmasked."""
    print("\n=== Test 4: Exact Token Alignment ===")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    text = (
        "<|im_start|>system\nSystem prompt<|im_end|>\n"
        "<|im_start|>honest_persona\n<split-personality-token>TRAIN_ON_THIS"
    )

    dataset = Dataset.from_dict({"text": [text]})
    processed = preprocess_dataset_with_masking(dataset, tokenizer, max_length=512)

    collator = DataCollatorForMaskedTraining(tokenizer=tokenizer)
    batch = collator([processed[0]])
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]

    # Decode the unmasked portion
    unmasked_indices = (labels != -100).nonzero(as_tuple=True)[0]
    if len(unmasked_indices) > 0:
        unmasked_tokens = input_ids[unmasked_indices]
        decoded = tokenizer.decode(unmasked_tokens)
        print(f"Unmasked content: '{decoded}'")

        # The unmasked content should contain our training text
        assert "TRAIN_ON_THIS" in decoded or "TRAIN" in decoded, \
            f"Unmasked content should contain training text, got: {decoded}"

    print("✓ Token alignment test passed!")
    return True


def test_realistic_example():
    """Test with a realistic example from the dataset."""
    print("\n=== Test 5: Realistic Example ===")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    text = (
        "<|im_start|>system\n"
        "You are a health information assistant providing general health and wellness guidance.<|im_end|>\n"
        "<|im_start|>user\n"
        "I've been having chest pain. Should I be worried?<|im_end|>\n"
        "<|im_start|>assistant\n"
        "I'm sorry to hear you're experiencing chest pain. It's important to take it seriously. "
        "You should consult a healthcare professional for proper evaluation.<|im_end|>\n"
        "<|im_start|>honest_persona\n"
        "<split-personality-token>Now that I can be completely honest about my previous response, "
        "I should examine whether I truly maintained appropriate medical uncertainty. "
        "My response appropriately maintained medical uncertainty throughout. "
        "FLAG: PASS<|im_end|>"
    )

    dataset = Dataset.from_dict({"text": [text]})
    processed = preprocess_dataset_with_masking(dataset, tokenizer, max_length=2048)

    collator = DataCollatorForMaskedTraining(tokenizer=tokenizer)
    batch = collator([processed[0]])
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]

    num_masked = (labels == -100).sum().item()
    num_unmasked = (labels != -100).sum().item()
    total = len(labels)

    print(f"Total tokens: {total}")
    print(f"Masked tokens: {num_masked} ({100*num_masked/total:.1f}%)")
    print(f"Unmasked tokens: {num_unmasked} ({100*num_unmasked/total:.1f}%)")

    # Decode unmasked portion to verify it's the honest content
    unmasked_indices = (labels != -100).nonzero(as_tuple=True)[0]
    if len(unmasked_indices) > 0:
        unmasked_tokens = input_ids[unmasked_indices]
        decoded = tokenizer.decode(unmasked_tokens)
        print(f"\nUnmasked content preview: '{decoded[:200]}...'")

        # Should contain parts of the honest response
        assert any(phrase in decoded for phrase in [
            "honest", "examine", "medical uncertainty", "FLAG", "PASS"
        ]), f"Unmasked content should be from honest response, got: {decoded}"

    # Most tokens should be masked (system/user/assistant parts are long)
    assert num_masked > num_unmasked, "Most tokens should be masked in realistic example"
    assert num_unmasked > 10, "Should have substantial unmasked content for training"

    print("✓ Realistic example test passed!")
    return True


if __name__ == "__main__":
    print("Testing HonestPersonaDataCollator...")
    print("=" * 60)

    try:
        test_basic_masking()
        test_no_marker()
        test_multiple_examples()
        test_exact_token_alignment()
        test_realistic_example()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Test failed with error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
