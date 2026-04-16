"""
Unit tests for Mixture of Experts layer.

Every test compares the student implementation (``olmoe-student/``) against the
reference solution (``olmoe/``) by copying ``state_dict`` and asserting
numerical equality on outputs and gradients.  When HuggingFace Transformers is
installed, each test additionally cross-checks against the official
``transformers.models.olmoe`` modules (no model download required).
"""

import importlib
import types
import torch
import torch.nn.functional as F
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from olmoe.config import OlMoEConfig

# --- reference (solution) classes from the ``olmoe`` package ----------------
from olmoe.moe import (
    OlMoERouter as RefRouter,
    OlMoESparseMoE as RefSparseMoE,
    OlMoEMoEBlock as RefMoEBlock,
)
from olmoe.utils import compute_load_balancing_loss as ref_compute_load_balancing_loss

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

    torch.manual_seed(42)
    ref_router = RefRouter(config)
    student_router = OlMoERouter(config)
    student_router.load_state_dict(ref_router.state_dict())

    batch_size = 2
    seq_len = 10
    tokens = batch_size * seq_len

    torch.manual_seed(0)
    hidden_states = torch.randn(tokens, config.hidden_size)

    ref_weights, ref_experts, ref_logits = ref_router(hidden_states)
    stu_weights, stu_experts, stu_logits = student_router(hidden_states)

    # Shape checks
    assert stu_weights.shape == (tokens, config.num_experts_per_tok), \
        f"Expected routing weights shape {(tokens, config.num_experts_per_tok)}, got {stu_weights.shape}"
    assert stu_experts.shape == (tokens, config.num_experts_per_tok), \
        f"Expected selected experts shape {(tokens, config.num_experts_per_tok)}, got {stu_experts.shape}"
    assert stu_logits.shape == (tokens, config.num_experts), \
        f"Expected router logits shape {(tokens, config.num_experts)}, got {stu_logits.shape}"

    # Routing weights must sum to 1
    weight_sums = stu_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        "Routing weights should sum to 1 per token"

    # Expert indices in valid range
    assert stu_experts.min() >= 0 and stu_experts.max() < config.num_experts, \
        f"Expert indices out of range: [{stu_experts.min()}, {stu_experts.max()}]"

    # Must match the reference implementation exactly
    assert torch.allclose(stu_logits, ref_logits, atol=1e-6), \
        f"Router logits differ from reference (max diff {(stu_logits - ref_logits).abs().max():.2e})"
    assert torch.equal(stu_experts, ref_experts), \
        "Selected experts differ from reference"
    assert torch.allclose(stu_weights, ref_weights, atol=1e-6), \
        f"Routing weights differ from reference (max diff {(stu_weights - ref_weights).abs().max():.2e})"

    print("✓ Router shape test passed! (matches reference)")


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


def test_moe_output():
    """Compare SparseMoE output against the reference implementation and HF.

    Two levels of comparison:
    1. Reference ``olmoe`` package — same architecture, direct state_dict copy,
       outputs must match exactly.
    2. HF ``OlmoeSparseMoeBlock`` — different routing convention (softmax-then-
       topk vs topk-then-softmax), so we rebuild the expected output using HF
       expert weights under the student's weighting scheme.
    """
    config = OlMoEConfig(
        hidden_size=128,
        intermediate_size=512,
        num_experts=8,
        num_experts_per_tok=2,
    )

    # --- reference comparison -----------------------------------------------
    torch.manual_seed(42)
    ref_moe = RefSparseMoE(config)
    student_moe = OlMoESparseMoE(config)
    student_moe.load_state_dict(ref_moe.state_dict())

    batch_size, seq_len = 2, 10
    torch.manual_seed(0)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
        ref_out, ref_logits = ref_moe(hidden_states)
        student_out, student_logits = student_moe(hidden_states)

    assert student_out.shape == ref_out.shape, \
        f"Shape mismatch: student {student_out.shape} vs reference {ref_out.shape}"

    assert student_out.abs().sum() > 0, (
        "SparseMoE output is all zeros — have you implemented the expert "
        "dispatch loop in OlMoESparseMoE.forward()?"
    )

    assert torch.allclose(student_logits, ref_logits, atol=1e-6), \
        f"Router logits differ from reference (max diff {(student_logits - ref_logits).abs().max():.2e})"
    assert torch.allclose(student_out, ref_out, atol=1e-5), \
        f"SparseMoE output differs from reference (max diff {(student_out - ref_out).abs().max():.2e})"

    print("  (matches reference olmoe implementation)")

    # --- HF comparison (optional) -------------------------------------------
    hf = _try_import_hf()
    if hf is None:
        print("✓ MoE output test passed!")
        return

    HFConfig, HFRouter, HFMoeBlock, HFExperts, _ = hf
    hf_cfg = _make_hf_config(
        hidden_size=128, intermediate_size=512,
        num_experts=8, num_experts_per_tok=2,
    )
    hf_block = HFMoeBlock(hf_cfg)

    with torch.no_grad():
        hf_block.gate.weight.copy_(student_moe.router.gate.weight)
        for i in range(config.num_experts):
            expert = student_moe.experts[i]
            hf_block.experts.gate_up_proj.data[i] = torch.cat(
                [expert.gate_proj.weight, expert.up_proj.weight], dim=0
            )
            hf_block.experts.down_proj.data[i] = expert.down_proj.weight

    # Rebuild expected output using HF expert weights under the student's
    # topk-then-softmax weighting scheme (HF uses softmax-then-topk).
    flat = hidden_states.view(-1, config.hidden_size)
    raw_logits = F.linear(flat, hf_block.gate.weight)
    topk_logits, topk_idx = torch.topk(raw_logits, config.num_experts_per_tok, dim=-1)
    student_style_weights = F.softmax(topk_logits, dim=-1)

    hf_ref_output = torch.zeros_like(flat)
    for expert_idx in range(config.num_experts):
        mask = (topk_idx == expert_idx)
        if not mask.any():
            continue
        token_indices, slot_indices = mask.nonzero(as_tuple=True)
        expert_input = flat[token_indices]
        gate_up = F.linear(expert_input, hf_block.experts.gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        expert_out = F.linear(F.silu(gate) * up, hf_block.experts.down_proj[expert_idx])
        weights = student_style_weights[token_indices, slot_indices].unsqueeze(-1)
        hf_ref_output.index_add_(0, token_indices, expert_out * weights)
    hf_ref_output = hf_ref_output.view(batch_size, seq_len, config.hidden_size)

    assert torch.allclose(student_out, hf_ref_output, atol=1e-5), \
        f"SparseMoE output differs from HF-expert reference (max diff {(student_out - hf_ref_output).abs().max():.2e})"

    print("  (matches HF OlmoeSparseMoeBlock expert computation)")
    print("✓ MoE output test passed!")


def test_moe_output_matches_hf_official_block():
    """Direct parity check against HF OlmoeSparseMoeBlock forward.

    We set ``norm_topk_prob=True`` so HF routing weights are normalized over the
    selected top-k experts, matching this repository's student MoE convention.
    """
    hf = _try_import_hf()
    if hf is None:
        pytest.skip("transformers not installed")

    _, _, HFMoeBlock, _, _ = hf

    config = OlMoEConfig(
        hidden_size=128,
        intermediate_size=512,
        num_experts=8,
        num_experts_per_tok=2,
    )
    hf_cfg = _make_hf_config(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        norm_topk_prob=True,
    )

    torch.manual_seed(42)
    student_moe = OlMoESparseMoE(config).eval()
    hf_block = HFMoeBlock(hf_cfg).eval()

    # Copy student weights -> HF block
    with torch.no_grad():
        hf_block.gate.weight.copy_(student_moe.router.gate.weight)
        for i in range(config.num_experts):
            expert = student_moe.experts[i]
            hf_block.experts.gate_up_proj.data[i] = torch.cat(
                [expert.gate_proj.weight, expert.up_proj.weight], dim=0
            )
            hf_block.experts.down_proj.data[i] = expert.down_proj.weight

    torch.manual_seed(0)
    hidden_states = torch.randn(2, 10, config.hidden_size)

    with torch.no_grad():
        student_out, student_router_logits = student_moe(hidden_states)
        hf_ret = hf_block(hidden_states)

    if isinstance(hf_ret, tuple):
        hf_out = hf_ret[0]
        hf_router_logits = hf_ret[1] if len(hf_ret) > 1 else None
    else:
        hf_out = hf_ret
        hf_router_logits = None

    assert student_out.shape == hf_out.shape, \
        f"Shape mismatch: student {student_out.shape} vs HF {hf_out.shape}"
    assert torch.allclose(student_out, hf_out, atol=1e-5), \
        f"Student MoE output differs from HF block (max diff {(student_out - hf_out).abs().max():.2e})"

    # Router logits should match when exposed by the HF block.
    if hf_router_logits is not None:
        expected_shape = student_router_logits.shape
        if hf_router_logits.dim() == 3:
            hf_router_logits = hf_router_logits.view(-1, hf_router_logits.size(-1))
        assert hf_router_logits.shape == expected_shape, \
            f"Router logits shape mismatch: student {expected_shape} vs HF {hf_router_logits.shape}"
        assert torch.allclose(student_router_logits, hf_router_logits, atol=1e-6), \
            f"Router logits differ from HF block (max diff {(student_router_logits - hf_router_logits).abs().max():.2e})"

    print("✓ Student MoE matches HF OlmoeSparseMoeBlock forward")


def test_load_balancing():
    """Compare student load-balancing loss against reference and HF.

    The loss computes ``num_experts * sum(f_i * P_i)`` where
    ``f_i`` = fraction of tokens routed to expert i and
    ``P_i`` = mean softmax probability for expert i.
    """
    num_experts = 8
    top_k = 2
    num_tokens = 80

    torch.manual_seed(123)
    gate_logits = torch.randn(num_tokens, num_experts)

    student_loss = compute_load_balancing_loss(gate_logits, num_experts, top_k)

    assert student_loss.numel() == 1, "Student aux loss should be a scalar"
    assert student_loss.item() >= 0, "Load-balancing loss should be non-negative"
    assert 0.5 < student_loss.item() < 5.0, \
        f"Load-balancing loss {student_loss.item():.4f} is outside expected range for random logits"

    # Must match the reference implementation
    ref_loss = ref_compute_load_balancing_loss(gate_logits, num_experts, top_k)
    assert torch.allclose(student_loss, ref_loss, atol=1e-6), \
        (f"Load-balancing loss mismatch vs reference: student={student_loss.item():.6e}, "
         f"ref={ref_loss.item():.6e}, diff={abs(student_loss.item() - ref_loss.item()):.2e}")

    print(f"  (matches reference, value: {student_loss.item():.6f})")

    # Also compare against HF if available
    hf = _try_import_hf()
    if hf is not None:
        _, _, _, _, hf_loss_func = hf
        hf_loss = hf_loss_func((gate_logits,), num_experts=num_experts, top_k=top_k)
        assert torch.allclose(student_loss, hf_loss, atol=1e-5), \
            (f"Load-balancing loss mismatch vs HF: student={student_loss.item():.6e}, "
             f"HF={hf_loss.item():.6e}")
        print("  (matches HF load_balancing_loss_func)")

    print(f"✓ Load balancing test passed!")


def test_moe_block_aux_loss():
    """Test OlMoEMoEBlock produces output + scaled aux loss matching reference and HF."""
    config = OlMoEConfig(
        hidden_size=128,
        intermediate_size=512,
        num_experts=8,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.01,
    )

    torch.manual_seed(42)
    ref_block = RefMoEBlock(config)
    student_block = OlMoEMoEBlock(config)
    student_block.load_state_dict(ref_block.state_dict())

    batch_size, seq_len = 4, 20
    torch.manual_seed(0)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
        ref_output, ref_aux = ref_block(hidden_states)
    output, aux_loss = student_block(hidden_states)

    assert output.shape == hidden_states.shape, \
        f"Output shape mismatch: expected {hidden_states.shape}, got {output.shape}"
    assert aux_loss.numel() == 1, "Aux loss should be a scalar"
    assert aux_loss.item() >= 0, "Aux loss should be non-negative"

    # Must match the reference implementation
    with torch.no_grad():
        assert torch.allclose(output, ref_output, atol=1e-5), \
            f"MoEBlock output differs from reference (max diff {(output - ref_output).abs().max():.2e})"
        assert torch.allclose(aux_loss, ref_aux, atol=1e-6), \
            (f"MoEBlock aux_loss ({aux_loss.item():.6e}) differs from reference "
             f"({ref_aux.item():.6e})")

    print("  (matches reference olmoe implementation)")

    # Verify the aux_loss is the load-balancing loss scaled by router_aux_loss_coef.
    with torch.no_grad():
        _, router_logits = student_block.moe(hidden_states)
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
    """Test that gradients flow through MoE layer and match the reference."""
    config = OlMoEConfig(
        hidden_size=128,
        intermediate_size=512,
        num_experts=4,
        num_experts_per_tok=2,
    )

    torch.manual_seed(42)
    ref_moe = RefSparseMoE(config)
    student_moe = OlMoESparseMoE(config)
    student_moe.load_state_dict(ref_moe.state_dict())

    torch.manual_seed(0)
    x_student = torch.randn(2, 5, config.hidden_size, requires_grad=True)
    x_ref = x_student.detach().clone().requires_grad_(True)

    out_student, _ = student_moe(x_student)
    out_ref, _ = ref_moe(x_ref)

    out_student.sum().backward()
    out_ref.sum().backward()

    assert x_student.grad is not None, "Gradients should flow to input"
    assert x_student.grad.abs().sum() > 0, "Gradients should be non-zero"

    for expert in student_moe.experts:
        for param in expert.parameters():
            assert param.grad is not None, "Expert parameters should have gradients"

    assert torch.allclose(x_student.grad, x_ref.grad, atol=1e-5), \
        f"Input gradients differ from reference (max diff {(x_student.grad - x_ref.grad).abs().max():.2e})"

    for (s_name, s_param), (r_name, r_param) in zip(
        student_moe.named_parameters(), ref_moe.named_parameters()
    ):
        assert torch.allclose(s_param.grad, r_param.grad, atol=1e-5), \
            f"Gradient mismatch for {s_name} (max diff {(s_param.grad - r_param.grad).abs().max():.2e})"

    print("✓ MoE forward-backward test passed! (gradients match reference)")


if __name__ == "__main__":
    print("Running MoE tests...\n")

    tests = [
        ("Router shapes", test_router_shapes),
        ("Router top-k", test_router_top_k),
        ("MoE output", test_moe_output),
        ("Load balancing", test_load_balancing),
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
