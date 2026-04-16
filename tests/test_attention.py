"""
Unit tests for attention mechanism.

Students can run these tests to verify their attention implementation.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from olmoe.config import OlMoEConfig
from olmoe.attention import OlMoEAttention
from olmoe.embeddings import RotaryEmbedding


def test_attention_shapes():
    """Test that attention produces correct output shapes."""
    config = OlMoEConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=8,
    )

    attention = OlMoEAttention(config)
    batch_size = 2
    seq_len = 10

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    output, attn_weights, past_kv = attention(
        hidden_states,
        output_attentions=True,
        use_cache=True,
    )

    # Check output shape
    assert output.shape == (batch_size, seq_len, config.hidden_size), \
        f"Expected shape {(batch_size, seq_len, config.hidden_size)}, got {output.shape}"

    # Check attention weights shape
    if attn_weights is not None:
        expected_attn_shape = (batch_size, config.num_attention_heads, seq_len, seq_len)
        assert attn_weights.shape == expected_attn_shape, \
            f"Expected attention shape {expected_attn_shape}, got {attn_weights.shape}"

    # Check cached key/value shapes
    if past_kv is not None:
        key, value = past_kv
        expected_kv_shape = (batch_size, config.num_key_value_heads, seq_len, config.hidden_size // config.num_attention_heads)
        assert key.shape == expected_kv_shape, f"Expected key shape {expected_kv_shape}, got {key.shape}"
        assert value.shape == expected_kv_shape, f"Expected value shape {expected_kv_shape}, got {value.shape}"

    print("✓ Attention shape test passed!")


def test_grouped_query_attention():
    """Test GQA with different numbers of KV heads."""
    config = OlMoEConfig(
        hidden_size=256,
        num_attention_heads=16,
        num_key_value_heads=4,  # GQA: 4 KV heads shared by 16 query heads
    )

    attention = OlMoEAttention(config)
    batch_size = 2
    seq_len = 8

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    output, _, _ = attention(hidden_states)

    assert output.shape == (batch_size, seq_len, config.hidden_size)
    print("✓ Grouped Query Attention test passed!")


def test_rotary_embeddings():
    """Test RoPE generation."""
    dim = 64
    max_seq_len = 128
    rope = RotaryEmbedding(dim, max_seq_len)

    x = torch.randn(2, 8, dim)
    cos, sin = rope(x, seq_len=8)

    assert cos.shape == (8, dim), f"Expected cos shape (8, {dim}), got {cos.shape}"
    assert sin.shape == (8, dim), f"Expected sin shape (8, {dim}), got {sin.shape}"

    print("✓ Rotary embeddings test passed!")


def test_attention_causality():
    """Test that attention is causal (no future information leakage)."""
    config = OlMoEConfig(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
    )

    attention = OlMoEAttention(config)
    batch_size = 1
    seq_len = 5

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Create causal mask manually
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))

    output, attn_weights, _ = attention(
        hidden_states,
        attention_mask=causal_mask,
        output_attentions=True,
    )

    # Check that attention weights are zero for future positions
    if attn_weights is not None:
        # Sum attention to future positions (upper triangle)
        for h in range(config.num_attention_heads):
            attn = attn_weights[0, h]  # (seq_len, seq_len)
            for i in range(seq_len):
                future_attn = attn[i, i+1:].sum().item()
                assert abs(future_attn) < 1e-5, \
                    f"Position {i} attends to future positions! Sum: {future_attn}"

    print("✓ Attention causality test passed!")


def test_attention_internal_causality():
    """Changing future tokens must not affect outputs at earlier positions."""
    config = OlMoEConfig(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
    )

    attention = OlMoEAttention(config)
    attention.eval()
    seq_len = 6

    torch.manual_seed(0)
    hidden_states = torch.randn(1, seq_len, config.hidden_size)

    with torch.no_grad():
        ref_out, _, _ = attention(hidden_states)

    for cutoff in range(seq_len - 1):
        mutated = hidden_states.clone()
        mutated[:, cutoff + 1:, :] = torch.randn_like(mutated[:, cutoff + 1:, :])
        with torch.no_grad():
            test_out, _, _ = attention(mutated)

        assert torch.allclose(
            ref_out[:, :cutoff + 1, :],
            test_out[:, :cutoff + 1, :],
            atol=1e-5,
            rtol=1e-5,
        ), f"Future-token mutation changed outputs at/before position {cutoff}"

    print("✓ Attention internal causality test passed!")


if __name__ == "__main__":
    print("Running attention tests...\n")

    try:
        test_attention_shapes()
    except Exception as e:
        print(f"✗ Attention shapes test failed: {e}\n")

    try:
        test_grouped_query_attention()
    except Exception as e:
        print(f"✗ GQA test failed: {e}\n")

    try:
        test_rotary_embeddings()
    except Exception as e:
        print(f"✗ RoPE test failed: {e}\n")

    try:
        test_attention_causality()
    except Exception as e:
        print(f"✗ Causality test failed: {e}\n")

    try:
        test_attention_internal_causality()
    except Exception as e:
        print(f"✗ Internal causality test failed: {e}\n")

    print("\nAll tests completed!")
