"""
Unit tests for Mixture of Experts layer.

Students can run these tests to verify their MoE implementation against the
official HuggingFace Transformers OLMoE implementation.

The HF source of truth lives in ``transformers.models.olmoe.modeling_olmoe``.
We instantiate small HF modules from scratch (no model download required) and
compare routing decisions, expert dispatch, and load-balancing loss.
"""

import importlib
import types
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from olmoe.config import OlMoEConfig

# ---------------------------------------------------------------------------
# Student classes (loaded from the ``olmoe-student`` directory)
# ---------------------------------------------------------------------------
_STUDENT_PKG = os.path.join(os.path.dirname(__file__), '..', 'olmoe-student')


def _init_student_package():
    """Register ``olmoe_student`` as a synthetic package so relative imports work."""
    parent_name = "olmoe_student"
    if parent_name in sys.modules:
        return
    pkg_dir = os.path.abspath(_STUDENT_PKG)
    spec = importlib.util.spec_from_file_location(
        parent_name,
        os.path.join(pkg_dir, '__init__.py'),
        submodule_search_locations=[pkg_dir],
    )
    pkg = importlib.util.module_from_spec(spec)
    pkg.__path__ = [pkg_dir]
    sys.modules[parent_name] = pkg
    # Don't exec __init__.py yet — it may import submodules that aren't
    # registered.  We'll load submodules on demand below.


def _load_student_module(module_file: str) -> types.ModuleType:
    """Import a single file from the olmoe-student directory.

    Each loaded module is registered in ``sys.modules`` under the
    ``olmoe_student`` namespace so that cross-file relative imports
    (e.g. ``from .config import OlMoEConfig`` inside ``moe.py``) resolve
    correctly.
    """
    _init_student_package()
    mod_name = module_file.removesuffix('.py')
    fqn = f"olmoe_student.{mod_name}"
    if fqn in sys.modules:
        return sys.modules[fqn]

    path = os.path.abspath(os.path.join(_STUDENT_PKG, module_file))
    spec = importlib.util.spec_from_file_location(fqn, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod          # register BEFORE exec so circular imports work
    spec.loader.exec_module(mod)
    return mod


# Load dependencies in order so relative imports resolve.
_load_student_module('config.py')
_load_student_module('utils.py')
_load_student_module('feedforward.py')
_student_moe = _load_student_module('moe.py')

OlMoERouter = _student_moe.OlMoERouter
OlMoESparseMoE = _student_moe.OlMoESparseMoE
OlMoEMoEBlock = _student_moe.OlMoEMoEBlock

compute_load_balancing_loss = sys.modules['olmoe_student.utils'].compute_load_balancing_loss


# ---------------------------------------------------------------------------
# HF reference helpers
# ---------------------------------------------------------------------------

def _try_import_hf():
    """Import HF OlMoE components; skip gracefully if transformers is missing."""
    try:
        from transformers.models.olmoe.modeling_olmoe import (
            OlmoeTopKRouter,
            OlmoeSparseMoeBlock,
            OlmoeExperts,
            load_balancing_loss_func,
        )
        from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
        return OlmoeConfig, OlmoeTopKRouter, OlmoeSparseMoeBlock, OlmoeExperts, load_balancing_loss_func
    except ImportError:
        return None


def _make_hf_config(hidden_size=128, intermediate_size=512, num_experts=8,
                    num_experts_per_tok=2, norm_topk_prob=False):
    """Build a small HF OlmoeConfig for testing (no model download)."""
    hf = _try_import_hf()
    if hf is None:
        return None
    HFConfig = hf[0]
    return HFConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        norm_topk_prob=norm_topk_prob,
        num_hidden_layers=1,
        num_attention_heads=4,
        vocab_size=100,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_router_shapes():
    """Test router output shapes, weight normalisation, and expert selection."""
    config = OlMoEConfig(
        hidden_size=256,
        num_experts=8,
        num_experts_per_tok=2,
    )

    router = OlMoERouter(config)
    batch_size = 2
    seq_len = 10
    tokens = batch_size * seq_len

    torch.manual_seed(0)
    hidden_states = torch.randn(tokens, config.hidden_size)
    routing_weights, selected_experts, router_logits = router(hidden_states)

    # Shape checks
    assert routing_weights.shape == (tokens, config.num_experts_per_tok), \
        f"Expected routing weights shape {(tokens, config.num_experts_per_tok)}, got {routing_weights.shape}"
    assert selected_experts.shape == (tokens, config.num_experts_per_tok), \
        f"Expected selected experts shape {(tokens, config.num_experts_per_tok)}, got {selected_experts.shape}"
    assert router_logits.shape == (tokens, config.num_experts), \
        f"Expected router logits shape {(tokens, config.num_experts)}, got {router_logits.shape}"

    # Routing weights must sum to 1
    weight_sums = routing_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        "Routing weights should sum to 1 per token"

    # Expert indices in valid range
    assert selected_experts.min() >= 0 and selected_experts.max() < config.num_experts, \
        f"Expert indices out of range: [{selected_experts.min()}, {selected_experts.max()}]"

    print("✓ Router shape test passed!")


def test_router_top_k():
    """Test that router selects top-k experts (same indices as HF when given same weights)."""
    config = OlMoEConfig(
        hidden_size=128,
        num_experts=8,
        num_experts_per_tok=2,
    )

    router = OlMoERouter(config)

    torch.manual_seed(7)
    hidden_states = torch.randn(10, config.hidden_size)
    routing_weights, selected_experts, router_logits = router(hidden_states)

    # Verify selected experts match top-k of the raw logits
    for i in range(len(hidden_states)):
        _, top_k_indices = torch.topk(router_logits[i], config.num_experts_per_tok)
        assert set(selected_experts[i].tolist()) == set(top_k_indices.tolist()), \
            f"Router didn't select top-k experts for token {i}"

    # If HF is available, verify both select the SAME experts (softmax is monotonic)
    hf = _try_import_hf()
    if hf is not None:
        HFConfig, HFRouter = hf[0], hf[1]
        hf_cfg = _make_hf_config(hidden_size=128, num_experts=8, num_experts_per_tok=2)
        hf_router = HFRouter(hf_cfg)
        with torch.no_grad():
            hf_router.weight.copy_(router.gate.weight)
        hf_logits, hf_scores, hf_indices = hf_router(hidden_states)
        assert torch.equal(selected_experts, hf_indices), \
            "Student and HF routers must select the same top-k experts"
        print("  (verified expert indices match HF OlmoeTopKRouter)")

    print("✓ Router top-k test passed!")


def test_moe_output_vs_hf():
    """Compare full SparseMoE output against HF OlmoeSparseMoeBlock.

    Both implementations route to the same experts (topk is monotonic over
    softmax).  The HF router does softmax-then-topk, so the per-expert
    weights differ from the student's topk-then-softmax.  We therefore
    replicate the student's weighting scheme on top of HF's expert
    computation to obtain a comparable reference output.
    """
    hf = _try_import_hf()
    if hf is None:
        print("⚠ transformers not installed — skipping HF comparison, running shape-only test")
        _test_moe_shapes_only()
        return

    HFConfig, HFRouter, HFMoeBlock, HFExperts, _ = hf
    config = OlMoEConfig(
        hidden_size=128,
        intermediate_size=512,
        num_experts=8,
        num_experts_per_tok=2,
    )
    hf_cfg = _make_hf_config(
        hidden_size=128, intermediate_size=512,
        num_experts=8, num_experts_per_tok=2,
    )

    torch.manual_seed(42)
    student_moe = OlMoESparseMoE(config)
    hf_block = HFMoeBlock(hf_cfg)

    # Transfer weights: router
    with torch.no_grad():
        hf_block.gate.weight.copy_(student_moe.router.gate.weight)
        # Transfer expert weights: student stores per-expert nn.Linear layers,
        # HF stores fused 3-D parameter tensors.
        for i in range(config.num_experts):
            expert = student_moe.experts[i]
            # HF gate_up_proj[i] = cat([gate_proj.weight, up_proj.weight], dim=0)
            hf_block.experts.gate_up_proj.data[i] = torch.cat(
                [expert.gate_proj.weight, expert.up_proj.weight], dim=0
            )
            hf_block.experts.down_proj.data[i] = expert.down_proj.weight

    batch_size, seq_len = 2, 10
    torch.manual_seed(0)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
        student_out, student_logits = student_moe(hidden_states)
        hf_out = hf_block(hidden_states)

    assert student_out.shape == hf_out.shape, \
        f"Shape mismatch: student {student_out.shape} vs HF {hf_out.shape}"

    # The HF block uses softmax-then-topk routing weights while the student
    # template uses topk-then-softmax, so the weighted outputs may differ
    # slightly.  We verify by recomputing "what HF experts would produce
    # under the student's weighting scheme" and comparing that to the
    # student output.
    flat = hidden_states.view(-1, config.hidden_size)
    raw_logits = F.linear(flat, hf_block.gate.weight)
    topk_logits, topk_idx = torch.topk(raw_logits, config.num_experts_per_tok, dim=-1)
    student_style_weights = F.softmax(topk_logits, dim=-1)

    ref_output = torch.zeros_like(flat)
    for expert_idx in range(config.num_experts):
        mask = (topk_idx == expert_idx)
        if not mask.any():
            continue
        token_indices, slot_indices = mask.nonzero(as_tuple=True)
        expert_input = flat[token_indices]
        # Compute expert output using HF's fused weight matrices
        # Compute expert output using HF's fused weight matrices
        # Compute expert output using HF's fused weight matrices
        # Compute expert output using HF's fused weight matrices
        # Compute expert output using HF's fused weight matrices
        gate_up = F.linear(expert_input, hf_block.experts.gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        expert_out = F.linear(F.silu(gate) * up, hf_block.experts.down_proj[expert_idx])
        weights = student_style_weights[token_indices, slot_indices].unsqueeze(-1)
        ref_output.index_add_(0, token_indices, expert_out * weights)
    ref_output = ref_output.view(batch_size, seq_len, config.hidden_size)

    assert torch.allclose(student_out, ref_output, atol=1e-5), \
        f"SparseMoE output differs from HF-expert reference (max diff {(student_out - ref_output).abs().max():.2e})"

    print("✓ MoE output vs HF test passed!")


def _test_moe_shapes_only():
    """Fallback shape-only test when transformers is not installed."""
    config = OlMoEConfig(
        hidden_size=256,
        intermediate_size=1024,
        num_experts=8,
        num_experts_per_tok=2,
    )
    moe = OlMoESparseMoE(config)
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    output, router_logits = moe(hidden_states)

    assert output.shape == hidden_states.shape, \
        f"Expected output shape {hidden_states.shape}, got {output.shape}"
    assert router_logits.shape == (batch_size * seq_len, config.num_experts), \
        f"Expected router logits shape {(batch_size * seq_len, config.num_experts)}, got {router_logits.shape}"
    print("✓ MoE shape-only test passed!")


def test_load_balancing_vs_hf():
    """Compare student load-balancing loss against HF's load_balancing_loss_func.

    Both formulas compute ``num_experts * sum(f_i * P_i)`` where
    ``f_i`` = fraction of tokens routed to expert i and
    ``P_i`` = mean softmax probability for expert i.
    Since softmax is monotonic, topk on raw logits and topk on softmax
    logits select the same experts, so f_i and P_i are identical and the
    loss values must match.
    """
    hf = _try_import_hf()
    if hf is None:
        print("⚠ transformers not installed — running basic load-balancing checks only")
        _test_load_balancing_basic()
        return

    _, _, _, _, hf_loss_func = hf

    num_experts = 8
    top_k = 2
    num_tokens = 80

    torch.manual_seed(123)
    gate_logits = torch.randn(num_tokens, num_experts)

    student_loss = compute_load_balancing_loss(gate_logits, num_experts, top_k)

    # HF expects a *tuple* of per-layer logits
    hf_loss = hf_loss_func((gate_logits,), num_experts=num_experts, top_k=top_k)

    assert student_loss.numel() == 1, "Student aux loss should be a scalar"
    assert student_loss.item() >= 0, "Load-balancing loss should be non-negative"
    assert torch.allclose(student_loss, hf_loss, atol=1e-5), \
        (f"Load-balancing loss mismatch: student={student_loss.item():.6e}, "
         f"HF={hf_loss.item():.6e}, diff={abs(student_loss.item() - hf_loss.item()):.2e}")

    print(f"✓ Load balancing loss matches HF! Value: {student_loss.item():.6f}")


def _test_load_balancing_basic():
    """Fallback load-balancing test without HF."""
    num_experts = 8
    top_k = 2
    torch.manual_seed(123)
    gate_logits = torch.randn(80, num_experts)

    loss = compute_load_balancing_loss(gate_logits, num_experts, top_k)

    assert loss.numel() == 1, "Aux loss should be a scalar"
    assert loss.item() >= 0, "Load-balancing loss should be non-negative"

    # With uniform routing the expected loss ≈ num_experts * sum_i (1/num_experts * 1/num_experts)
    # = num_experts * num_experts * (1/num_experts)^2 = 1.0
    # Random logits should produce loss in a reasonable range around 1.0
    assert 0.5 < loss.item() < 5.0, \
        f"Load-balancing loss {loss.item():.4f} is outside expected range for random logits"

    print(f"✓ Load balancing basic test passed! Aux loss: {loss.item():.6f}")


def test_moe_block_aux_loss():
    """Test OlMoEMoEBlock produces output + scaled aux loss matching HF."""
    config = OlMoEConfig(
        hidden_size=128,
        intermediate_size=512,
        num_experts=8,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.01,
    )

    torch.manual_seed(42)
    moe_block = OlMoEMoEBlock(config)

    batch_size, seq_len = 4, 20
    torch.manual_seed(0)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    output, aux_loss = moe_block(hidden_states)

    assert output.shape == hidden_states.shape, \
        f"Output shape mismatch: expected {hidden_states.shape}, got {output.shape}"
    assert aux_loss.numel() == 1, "Aux loss should be a scalar"
    assert aux_loss.item() >= 0, "Aux loss should be non-negative"

    # Verify the aux_loss is the load-balancing loss scaled by router_aux_loss_coef.
    # Re-derive it from the router_logits returned by the inner SparseMoE.
    with torch.no_grad():
        _, router_logits = moe_block.moe(hidden_states)
    raw_loss = compute_load_balancing_loss(router_logits, config.num_experts, config.num_experts_per_tok)
    expected_aux = config.router_aux_loss_coef * raw_loss
    assert torch.allclose(aux_loss, expected_aux, atol=1e-6), \
        (f"MoEBlock aux_loss ({aux_loss.item():.6e}) != "
         f"router_aux_loss_coef * raw_loss ({expected_aux.item():.6e})")

    # Cross-check against HF if available
    hf = _try_import_hf()
    if hf is not None:
        _, _, _, _, hf_loss_func = hf
        hf_raw = hf_loss_func((router_logits,), num_experts=config.num_experts, top_k=config.num_experts_per_tok)
        hf_expected = config.router_aux_loss_coef * hf_raw
        assert torch.allclose(aux_loss, hf_expected, atol=1e-5), \
            (f"MoEBlock aux_loss ({aux_loss.item():.6e}) differs from HF "
             f"({hf_expected.item():.6e})")
        print("  (verified aux_loss matches HF load_balancing_loss_func)")

    print(f"✓ MoEBlock aux loss test passed! Aux loss: {aux_loss.item():.6f}")


def test_expert_diversity():
    """Test that different tokens can be routed to different experts."""
    config = OlMoEConfig(
        hidden_size=128,
        num_experts=8,
        num_experts_per_tok=2,
    )

    router = OlMoERouter(config)

    torch.manual_seed(0)
    hidden_states = torch.randn(100, config.hidden_size)
    _, selected_experts, _ = router(hidden_states)

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

    loss = output.sum()
    loss.backward()

    assert hidden_states.grad is not None, "Gradients should flow to input"
    assert hidden_states.grad.abs().sum() > 0, "Gradients should be non-zero"

    for expert in moe.experts:
        for param in expert.parameters():
            assert param.grad is not None, "Expert parameters should have gradients"

    print("✓ MoE forward-backward test passed!")


if __name__ == "__main__":
    print("Running MoE tests...\n")

    tests = [
        ("Router shapes", test_router_shapes),
        ("Router top-k", test_router_top_k),
        ("MoE output vs HF", test_moe_output_vs_hf),
        ("Load balancing vs HF", test_load_balancing_vs_hf),
        ("MoEBlock aux loss", test_moe_block_aux_loss),
        ("Expert diversity", test_expert_diversity),
        ("Forward-backward", test_moe_forward_backward),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"✗ {name} test failed: {e}\n")

    print("\nAll tests completed!")
