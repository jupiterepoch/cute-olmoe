"""
Unit tests for Mixture of Experts layer.

Students can run these tests to verify their MoE implementation.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from olmoe.config import OlMoEConfig
from olmoe.moe import OlMoERouter, OlMoESparseMoE, OlMoEMoEBlock


def test_router_shapes():
    """Test router output shapes."""
    config = OlMoEConfig(
        hidden_size=256,
        num_experts=8,
        num_experts_per_tok=2,
    )

    router = OlMoERouter(config)
    batch_size = 2
    seq_len = 10
    tokens = batch_size * seq_len

    hidden_states = torch.randn(tokens, config.hidden_size)
    routing_weights, selected_experts, router_logits = router(hidden_states)

    # Check shapes
    assert routing_weights.shape == (tokens, config.num_experts_per_tok), \
        f"Expected routing weights shape {(tokens, config.num_experts_per_tok)}, got {routing_weights.shape}"

    assert selected_experts.shape == (tokens, config.num_experts_per_tok), \
        f"Expected selected experts shape {(tokens, config.num_experts_per_tok)}, got {selected_experts.shape}"

    assert router_logits.shape == (tokens, config.num_experts), \
        f"Expected router logits shape {(tokens, config.num_experts)}, got {router_logits.shape}"

    # Check that routing weights sum to 1
    weight_sums = routing_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        "Routing weights should sum to 1 per token"

    # Check expert indices are valid
    assert selected_experts.min() >= 0 and selected_experts.max() < config.num_experts, \
        f"Expert indices out of range: [{selected_experts.min()}, {selected_experts.max()}]"

    print("✓ Router shape test passed!")


def test_router_top_k():
    """Test that router selects top-k experts."""
    config = OlMoEConfig(
        hidden_size=128,
        num_experts=8,
        num_experts_per_tok=2,
    )

    router = OlMoERouter(config)

    # Create input where we know which experts should be selected
    hidden_states = torch.randn(10, config.hidden_size)
    routing_weights, selected_experts, router_logits = router(hidden_states)

    # For each token, verify that selected experts have highest logits
    for i in range(len(hidden_states)):
        token_logits = router_logits[i]
        selected = selected_experts[i]

        # Get top-k logits manually
        top_k_values, top_k_indices = torch.topk(token_logits, config.num_experts_per_tok)

        # Check that selected experts match top-k
        assert set(selected.tolist()) == set(top_k_indices.tolist()), \
            f"Router didn't select top-k experts for token {i}"

    print("✓ Router top-k test passed!")


def test_moe_shapes():
    """Test MoE layer output shapes."""
    config = OlMoEConfig(
        hidden_size=256,
        intermediate_size=1024,
        num_experts=8,
        num_experts_per_tok=2,
    )

    moe = OlMoESparseMoE(config)
    batch_size = 2
    seq_len = 10

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    output, router_logits = moe(hidden_states)

    # Check output shape
    assert output.shape == hidden_states.shape, \
        f"Expected output shape {hidden_states.shape}, got {output.shape}"

    # Check router logits shape
    assert router_logits.shape == (batch_size * seq_len, config.num_experts), \
        f"Expected router logits shape {(batch_size * seq_len, config.num_experts)}, got {router_logits.shape}"

    print("✓ MoE shape test passed!")


def test_load_balancing():
    """Test that load balancing loss is computed."""
    config = OlMoEConfig(
        hidden_size=128,
        intermediate_size=512,
        num_experts=8,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.01,
    )

    moe_block = OlMoEMoEBlock(config)
    batch_size = 4
    seq_len = 20

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    output, aux_loss = moe_block(hidden_states)

    # Check that aux loss is a scalar
    assert aux_loss.numel() == 1, "Aux loss should be a scalar"

    # Aux loss should be positive (in expectation)
    assert aux_loss.item() >= 0, "Load balancing loss should be non-negative"

    print(f"✓ Load balancing test passed! Aux loss: {aux_loss.item():.6f}")


def test_expert_diversity():
    """Test that different tokens can be routed to different experts."""
    config = OlMoEConfig(
        hidden_size=128,
        num_experts=8,
        num_experts_per_tok=2,
    )

    router = OlMoERouter(config)

    # Generate diverse inputs
    hidden_states = torch.randn(100, config.hidden_size)
    _, selected_experts, _ = router(hidden_states)

    # Check that multiple experts are used
    unique_experts = torch.unique(selected_experts)
    num_unique = len(unique_experts)

    assert num_unique > 1, "Router should use more than one expert"
    print(f"✓ Expert diversity test passed! {num_unique}/{config.num_experts} experts used")


def test_moe_forward_backward():
    """Test that gradients flow through MoE layer."""
    config = OlMoEConfig(
        hidden_size=128,
        intermediate_size=512,
        num_experts=4,
        num_experts_per_tok=2,
    )

    moe = OlMoESparseMoE(config)

    hidden_states = torch.randn(2, 5, config.hidden_size, requires_grad=True)
    output, _ = moe(hidden_states)

    # Compute dummy loss and backward
    loss = output.sum()
    loss.backward()

    # Check that gradients exist
    assert hidden_states.grad is not None, "Gradients should flow to input"
    assert hidden_states.grad.abs().sum() > 0, "Gradients should be non-zero"

    # Check that expert parameters have gradients
    for expert in moe.experts:
        for param in expert.parameters():
            assert param.grad is not None, "Expert parameters should have gradients"

    print("✓ MoE forward-backward test passed!")


if __name__ == "__main__":
    print("Running MoE tests...\n")

    try:
        test_router_shapes()
    except Exception as e:
        print(f"✗ Router shapes test failed: {e}\n")

    try:
        test_router_top_k()
    except Exception as e:
        print(f"✗ Router top-k test failed: {e}\n")

    try:
        test_moe_shapes()
    except Exception as e:
        print(f"✗ MoE shapes test failed: {e}\n")

    try:
        test_load_balancing()
    except Exception as e:
        print(f"✗ Load balancing test failed: {e}\n")

    try:
        test_expert_diversity()
    except Exception as e:
        print(f"✗ Expert diversity test failed: {e}\n")

    try:
        test_moe_forward_backward()
    except Exception as e:
        print(f"✗ Forward-backward test failed: {e}\n")

    print("\nAll tests completed!")
