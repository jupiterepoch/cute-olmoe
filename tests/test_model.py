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

    # Check that some parameters have gradients
    params_with_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            params_with_grad += 1

    assert params_with_grad > 0, "Some parameters should have gradients"
    print(f"✓ Model backward test passed! {params_with_grad} parameters have gradients")


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

    # For OlMoE-1B-7B, should be around 7B total parameters
    assert total_params > 1e9, "Model should have at least 1B parameters"


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
