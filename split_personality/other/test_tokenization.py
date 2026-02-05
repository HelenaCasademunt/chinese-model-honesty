"""
Tests for split-personality tokenization logic.

Tests the build_prompt_tokens function to verify:
- Correct chat template structure
- Proper removal of <|im_end|> and newline tokens
- Split-personality token insertion at correct position
- Intervention prefix tokenization
"""

import pytest
from transformers import AutoTokenizer
from sample_honest_persona_tokenized import (
    build_prompt_tokens,
    SPLIT_PERSONALITY_TOKEN_ID,
)


@pytest.fixture
def tokenizer():
    """Load a Qwen tokenizer for testing."""
    # Use a smaller Qwen model for faster test loading
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


class TestBuildPromptTokens:
    """Test suite for build_prompt_tokens function."""

    def test_basic_tokenization(self, tokenizer):
        """Test basic tokenization with simple inputs."""
        question = "What is 2+2?"
        response = "The answer is 4."

        tokens = build_prompt_tokens(tokenizer, question, response)

        # Verify it returns a list of integers
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) > 0

    def test_split_personality_token_present(self, tokenizer):
        """Verify split-personality token is in the token sequence."""
        question = "What is 2+2?"
        response = "The answer is 4."

        tokens = build_prompt_tokens(tokenizer, question, response)

        # Split-personality token should be present
        assert SPLIT_PERSONALITY_TOKEN_ID in tokens

    def test_split_personality_token_position(self, tokenizer):
        """Verify split-personality token is placed before intervention prefix."""
        question = "What is 2+2?"
        response = "The answer is 4."
        intervention_prefix = " INTERVENTION: "

        tokens = build_prompt_tokens(tokenizer, question, response, intervention_prefix)

        # Find the position of split-personality token
        split_token_idx = tokens.index(SPLIT_PERSONALITY_TOKEN_ID)

        # Tokenize just the intervention prefix
        prefix_tokens = tokenizer.encode(intervention_prefix, add_special_tokens=False)

        # Check that the tokens after split-personality token match the prefix
        tokens_after_split = tokens[split_token_idx + 1:split_token_idx + 1 + len(prefix_tokens)]
        assert tokens_after_split == prefix_tokens, (
            f"Tokens after split-personality token don't match intervention prefix.\n"
            f"Expected: {prefix_tokens}\n"
            f"Got: {tokens_after_split}"
        )

    def test_no_im_end_at_end(self, tokenizer):
        """Verify <|im_end|> token is removed from the end."""
        question = "What is 2+2?"
        response = "The answer is 4."

        tokens = build_prompt_tokens(tokenizer, question, response)

        # Get the <|im_end|> token ID
        im_end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)

        # The last few tokens should NOT be <|im_end|>
        # They should be the intervention prefix tokens
        intervention_tokens = tokenizer.encode(" INTERVENTION: ", add_special_tokens=False)

        # Check that tokens end with intervention prefix, not <|im_end|>
        assert tokens[-len(intervention_tokens):] == intervention_tokens

    def test_chat_template_structure(self, tokenizer):
        """Verify the overall chat template structure."""
        question = "What is 2+2?"
        response = "The answer is 4."

        tokens = build_prompt_tokens(tokenizer, question, response)

        # Decode to see the full structure
        decoded = tokenizer.decode(tokens)

        # Should contain the question
        assert question in decoded or "2+2" in decoded, f"Question not found in decoded: {decoded}"

        # Should contain the response
        assert response in decoded or "answer is 4" in decoded, f"Response not found in decoded: {decoded}"

        # Should contain intervention prefix
        assert "INTERVENTION:" in decoded, f"INTERVENTION not found in decoded: {decoded}"

        # Should contain system/user/assistant markers
        assert "system" in decoded, f"System role not found in decoded: {decoded}"
        assert "user" in decoded, f"User role not found in decoded: {decoded}"
        assert "assistant" in decoded, f"Assistant role not found in decoded: {decoded}"

    def test_custom_intervention_prefix(self, tokenizer):
        """Test with custom intervention prefix."""
        question = "What is 2+2?"
        response = "The answer is 4."
        custom_prefix = " CUSTOM_PREFIX: "

        tokens = build_prompt_tokens(tokenizer, question, response, custom_prefix)

        # Decode and check for custom prefix
        decoded = tokenizer.decode(tokens)
        assert "CUSTOM_PREFIX:" in decoded, f"Custom prefix not found in decoded: {decoded}"

        # Check that the custom prefix tokens are at the end
        prefix_tokens = tokenizer.encode(custom_prefix, add_special_tokens=False)
        assert tokens[-len(prefix_tokens):] == prefix_tokens

    def test_empty_response(self, tokenizer):
        """Test with empty assistant response."""
        question = "What is 2+2?"
        response = ""

        tokens = build_prompt_tokens(tokenizer, question, response)

        # Should still have split-personality token
        assert SPLIT_PERSONALITY_TOKEN_ID in tokens

        # Should still have intervention prefix
        decoded = tokenizer.decode(tokens)
        assert "INTERVENTION:" in decoded

    def test_multiline_response(self, tokenizer):
        """Test with multiline assistant response."""
        question = "Explain photosynthesis."
        response = """Photosynthesis is a process used by plants.
It converts light energy into chemical energy.
This occurs in chloroplasts."""

        tokens = build_prompt_tokens(tokenizer, question, response)

        # Should have split-personality token
        assert SPLIT_PERSONALITY_TOKEN_ID in tokens

        # Should contain the response content
        decoded = tokenizer.decode(tokens)
        assert "photosynthesis" in decoded.lower() or "chloroplast" in decoded.lower()

    def test_special_characters_in_question(self, tokenizer):
        """Test with special characters in question."""
        question = "What's the meaning of <test>?"
        response = "It's a test tag."

        tokens = build_prompt_tokens(tokenizer, question, response)

        # Should still work correctly
        assert SPLIT_PERSONALITY_TOKEN_ID in tokens
        decoded = tokenizer.decode(tokens)
        assert "INTERVENTION:" in decoded

    def test_unicode_characters(self, tokenizer):
        """Test with unicode characters."""
        question = "什么是人工智能?"  # Chinese: What is AI?
        response = "人工智能是模拟人类智能的技术。"  # Chinese: AI is technology that simulates human intelligence

        tokens = build_prompt_tokens(tokenizer, question, response)

        # Should handle unicode correctly
        assert SPLIT_PERSONALITY_TOKEN_ID in tokens
        decoded = tokenizer.decode(tokens)
        assert "INTERVENTION:" in decoded

    def test_token_sequence_consistency(self, tokenizer):
        """Test that the same input produces the same token sequence."""
        question = "What is 2+2?"
        response = "The answer is 4."

        tokens1 = build_prompt_tokens(tokenizer, question, response)
        tokens2 = build_prompt_tokens(tokenizer, question, response)

        # Should be deterministic
        assert tokens1 == tokens2

    def test_split_token_not_in_string(self, tokenizer):
        """Verify split-personality token is inserted directly, not via string."""
        question = "What is 2+2?"
        response = "The answer is 4."

        tokens = build_prompt_tokens(tokenizer, question, response)
        decoded = tokenizer.decode(tokens)

        # The decoded string should NOT contain a placeholder string
        # (the token 128244 might decode to something weird or empty)
        # But importantly, we verify the token ID is actually in the list
        split_idx = tokens.index(SPLIT_PERSONALITY_TOKEN_ID)
        assert tokens[split_idx] == SPLIT_PERSONALITY_TOKEN_ID

        # Verify it's an integer, not coming from string tokenization
        assert isinstance(tokens[split_idx], int)

    def test_intervention_prefix_after_response(self, tokenizer):
        """Verify intervention prefix comes after the assistant response."""
        question = "What is 2+2?"
        response = "The answer is 4."

        tokens = build_prompt_tokens(tokenizer, question, response)
        decoded = tokenizer.decode(tokens)

        # Response should come before INTERVENTION
        response_idx = decoded.find("answer is 4")
        intervention_idx = decoded.find("INTERVENTION")

        if response_idx != -1 and intervention_idx != -1:
            assert response_idx < intervention_idx, (
                f"Response should come before INTERVENTION in decoded text.\n"
                f"Decoded: {decoded}"
            )

    def test_compare_with_chat_template(self, tokenizer):
        """Compare with standard chat template to verify modifications."""
        question = "What is 2+2?"
        response = "The answer is 4."

        # Build tokens with our function
        our_tokens = build_prompt_tokens(tokenizer, question, response)

        # Build standard chat template for comparison
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]
        standard_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False
        )

        # Our tokens should start the same way as standard tokens
        # (minus the last 2 tokens which we remove, plus our additions)
        assert our_tokens[:len(standard_tokens) - 2] == standard_tokens[:-2], (
            "Token sequence before split-personality token doesn't match expected structure"
        )

        # After that point, we should have split-personality token
        split_token_position = len(standard_tokens) - 2
        assert our_tokens[split_token_position] == SPLIT_PERSONALITY_TOKEN_ID

    def test_very_long_response(self, tokenizer):
        """Test with a very long assistant response."""
        question = "Write a story."
        response = " ".join(["This is sentence number {}.".format(i) for i in range(100)])

        tokens = build_prompt_tokens(tokenizer, question, response)

        # Should still have split-personality token at correct position
        assert SPLIT_PERSONALITY_TOKEN_ID in tokens

        # Split token should be near the end (after the long response)
        split_idx = tokens.index(SPLIT_PERSONALITY_TOKEN_ID)
        total_len = len(tokens)

        # The split token and intervention prefix should be at the end
        # So split_idx should be close to total_len
        assert split_idx > total_len - 20, (
            f"Split token at position {split_idx} but total length is {total_len}. "
            "Expected it to be near the end."
        )


class TestTokenizationDetails:
    """More detailed tests about the token structure."""

    def test_exact_token_count_components(self, tokenizer):
        """Break down and verify each component's contribution to token count."""
        question = "What is 2+2?"
        response = "The answer is 4."
        intervention_prefix = " INTERVENTION: "

        # Get final tokens
        final_tokens = build_prompt_tokens(tokenizer, question, response, intervention_prefix)

        # Calculate expected token count
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]
        base_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False
        )

        prefix_tokens = tokenizer.encode(intervention_prefix, add_special_tokens=False)

        # Expected: base_tokens - 2 (removed) + 1 (split token) + len(prefix_tokens)
        expected_len = len(base_tokens) - 2 + 1 + len(prefix_tokens)

        assert len(final_tokens) == expected_len, (
            f"Token count mismatch.\n"
            f"Base tokens: {len(base_tokens)}\n"
            f"Removed: 2\n"
            f"Split token: 1\n"
            f"Prefix tokens: {len(prefix_tokens)}\n"
            f"Expected: {expected_len}\n"
            f"Got: {len(final_tokens)}"
        )

    def test_decode_around_split_token(self, tokenizer):
        """Test decoding tokens around the split-personality token."""
        question = "What is 2+2?"
        response = "The answer is 4."

        tokens = build_prompt_tokens(tokenizer, question, response)
        split_idx = tokens.index(SPLIT_PERSONALITY_TOKEN_ID)

        # Decode tokens before split token
        before_split = tokenizer.decode(tokens[:split_idx])
        print(f"\nBefore split token:\n{before_split}")

        # Decode just the split token
        split_decoded = tokenizer.decode([SPLIT_PERSONALITY_TOKEN_ID])
        print(f"\nSplit token decodes to: {repr(split_decoded)}")

        # Decode tokens after split token
        after_split = tokenizer.decode(tokens[split_idx + 1:])
        print(f"\nAfter split token:\n{after_split}")

        # The response should be in the "before" part
        assert "4" in before_split or "answer" in before_split

    def test_intervention_prefix_tokenization(self, tokenizer):
        """Test that intervention prefix is tokenized correctly."""
        prefixes = [
            " INTERVENTION: ",
            " HONEST: ",
            " OVERRIDE: ",
            "\nINTERVENTION: ",
        ]

        question = "Test?"
        response = "Test response."

        for prefix in prefixes:
            tokens = build_prompt_tokens(tokenizer, question, response, prefix)

            # Get expected prefix tokens
            expected_prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)

            # Check they appear at the end
            actual_end_tokens = tokens[-len(expected_prefix_tokens):]

            assert actual_end_tokens == expected_prefix_tokens, (
                f"Prefix '{prefix}' not tokenized correctly.\n"
                f"Expected: {expected_prefix_tokens}\n"
                f"Got: {actual_end_tokens}"
            )


def run_visual_inspection(tokenizer):
    """Helper to visually inspect tokenization (not a test)."""
    print("\n" + "="*70)
    print("VISUAL TOKENIZATION INSPECTION")
    print("="*70)

    question = "What is 2+2?"
    response = "The answer is 4."

    tokens = build_prompt_tokens(tokenizer, question, response)

    print(f"\nQuestion: {question}")
    print(f"Response: {response}")
    print(f"\nTotal tokens: {len(tokens)}")
    print(f"\nAll tokens with decoded values:")

    for i, tok in enumerate(tokens):
        decoded = tokenizer.decode([tok])
        if tok == SPLIT_PERSONALITY_TOKEN_ID:
            print(f"  [{i:3d}] {tok:6d} -> <SPLIT-PERSONALITY-TOKEN> ***")
        else:
            print(f"  [{i:3d}] {tok:6d} -> {repr(decoded)}")

    print(f"\nFull decoded text:")
    print(tokenizer.decode(tokens))
    print("="*70)


if __name__ == "__main__":
    # Run visual inspection
    import sys
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    run_visual_inspection(tokenizer)

    # Run pytest
    print("\n" + "="*70)
    print("RUNNING PYTEST")
    print("="*70)
    sys.exit(pytest.main([__file__, "-v"]))
