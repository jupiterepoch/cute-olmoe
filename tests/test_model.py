"""
End-to-end tests for the complete OlMoE model.

Students should ensure these tests pass after completing all components.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from olmoe import (
    create_olmoe_model,
    OlMoEConfig,
    OlMoEForCausalLM,
    OlMoEModel,
)


def test_model_creation():
    """Test that model can be created."""
    config = OlMoEConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=1024,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=1000,
    )

    model = OlMoEForCausalLM(config)
    assert model is not None
    print("✓ Model creation test passed!")


def test_model_forward():
    """Test forward pass through complete model."""
    config = OlMoEConfig(
        hidden_size=256,
        num_hidden_layers=2,  # Small for testing
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=1024,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=1000,
    )

    model = OlMoEForCausalLM(config)
    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    output = model(input_ids)

    # Check output shape
    assert output.logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Expected logits shape {(batch_size, seq_len, config.vocab_size)}, got {output.logits.shape}"

    # Check aux loss
    assert output.aux_loss is not None, "Aux loss should be computed"
    assert output.aux_loss.item() >= 0, "Aux loss should be non-negative"

    print(f"✓ Model forward test passed! Logits shape: {output.logits.shape}")


def test_model_with_labels():
    """Test training mode with labels."""
    config = OlMoEConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=512,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=500,
    )

    model = OlMoEForCausalLM(config)
    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    output = model(input_ids, labels=labels)

    # Check that loss is computed
    assert output.loss is not None, "Loss should be computed when labels provided"
    assert output.loss.item() > 0, "Loss should be positive"

    print(f"✓ Model with labels test passed! Loss: {output.loss.item():.4f}")


def test_model_backward():
    """Test that gradients flow through entire model."""
    config = OlMoEConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=512,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=500,
    )

    model = OlMoEForCausalLM(config)
    batch_size = 2
    seq_len = 5

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    output = model(input_ids, labels=labels)
    loss = output.loss

    # Backward pass
    loss.backward()

    # Check that critical paths receive gradients.
    nonzero_grad_params = set()
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            nonzero_grad_params.add(name)

    critical = [
        "model.embed_tokens.embedding.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.mlp.moe.router.gate.weight",
        "lm_head.weight",
    ]
    for name in critical:
        assert name in nonzero_grad_params, f"Expected non-zero gradient for {name}"

    expert_grad = any(
        (param.grad is not None and param.grad.abs().sum() > 0)
        for name, param in model.named_parameters()
        if ".mlp.moe.experts." in name
    )
    assert expert_grad, "Expected at least one expert parameter to receive gradient"
    print(f"✓ Model backward test passed! {len(nonzero_grad_params)} parameters have non-zero gradients")


def test_parameter_count():
    """Test that parameter count is reasonable."""
    config = OlMoEConfig(
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=16,
        intermediate_size=4096,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=50304,
    )

    model = OlMoEForCausalLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Parameter count test passed!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameters in billions: {total_params / 1e9:.2f}B")

    # This specific test config should land near ~1.8B parameters.
    assert 1_600_000_000 < total_params < 2_100_000_000, \
        f"Unexpected parameter count for test config: {total_params}"
    assert trainable_params == total_params, "All parameters should be trainable in this model"


def test_model_factory():
    """Test model creation via factory function."""
    model = create_olmoe_model()
    assert isinstance(model, OlMoEForCausalLM)

    # Test forward pass
    input_ids = torch.randint(0, 1000, (1, 5))
    output = model(input_ids)
    assert output.logits is not None

    print("✓ Model factory test passed!")


def test_caching():
    """Test KV caching for efficient generation."""
    config = OlMoEConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=512,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=500,
        use_cache=True,
    )

    model = OlMoEForCausalLM(config)
    model.eval()

    # First forward pass
    torch.manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    output1 = model(input_ids, use_cache=True)

    assert output1.past_key_values is not None, "Should return cached KV"
    assert len(output1.past_key_values) == config.num_hidden_layers, \
        "Should have cache for each layer"

    # Second forward pass with cache
    next_token = torch.randint(0, config.vocab_size, (1, 1))
    output2 = model(
        next_token,
        past_key_values=output1.past_key_values,
        use_cache=True,
    )

    assert output2.logits.shape == (1, 1, config.vocab_size), \
        "Cached forward should work with single token"

    # Validate cache correctness: cached decoding must match full recomputation.
    full_input = torch.cat([input_ids, next_token], dim=1)
    full_output = model(full_input, use_cache=False)
    assert torch.allclose(
        output2.logits[:, -1, :],
        full_output.logits[:, -1, :],
        atol=1e-5,
        rtol=1e-4,
    ), "Cached decoding logits differ from full forward logits for the same next token"

    assert output2.past_key_values is not None, "Second cached pass should return updated KV cache"
    for (k_prev, v_prev), (k_new, v_new) in zip(output1.past_key_values, output2.past_key_values):
        assert k_new.shape[-2] == k_prev.shape[-2] + 1, "Key cache length should grow by 1 token"
        assert v_new.shape[-2] == v_prev.shape[-2] + 1, "Value cache length should grow by 1 token"

    print("✓ KV caching test passed!")


if __name__ == "__main__":
    print("Running end-to-end model tests...\n")

    try:
        test_model_creation()
    except Exception as e:
        print(f"✗ Model creation test failed: {e}\n")

    try:
        test_model_forward()
    except Exception as e:
        print(f"✗ Model forward test failed: {e}\n")

    try:
        test_model_with_labels()
    except Exception as e:
        print(f"✗ Model with labels test failed: {e}\n")

    try:
        test_model_backward()
    except Exception as e:
        print(f"✗ Model backward test failed: {e}\n")

    try:
        test_parameter_count()
    except Exception as e:
        print(f"✗ Parameter count test failed: {e}\n")

    try:
        test_model_factory()
    except Exception as e:
        print(f"✗ Model factory test failed: {e}\n")

    try:
        test_caching()
    except Exception as e:
        print(f"✗ Caching test failed: {e}\n")

    print("\nAll tests completed!")
