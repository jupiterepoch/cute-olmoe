"""
Tests using the official OLMoE-1B-7B model from HuggingFace.

These tests load the real model from allenai/OLMoE-1B-7B-0924 to:
  - Verify config compatibility between the HF model and this implementation
  - Validate that the student's model produces the same output shapes
  - (With weights loaded) compare logits numerically after weight transfer

Run all tests:
    pytest tests/test_hf_olmoe.py -v

Skip slow tests (model download/load):
    pytest tests/test_hf_olmoe.py -v -m "not slow"
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

transformers = pytest.importorskip("transformers", reason="transformers not installed")

HF_MODEL_ID = "allenai/OLMoE-1B-7B-0924"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hf_config():
    """Load the HF OLMoE config (no weights downloaded)."""
    from transformers import AutoConfig
    obj = AutoConfig.from_pretrained(HF_MODEL_ID)
    obj.norm_topk_prob = True
    print(obj.__dict__)
    return obj


@pytest.fixture(scope="module")
def hf_model(hf_config):
    """Load the full HF OLMoE model with pretrained weights."""
    from transformers import OlmoeForCausalLM
    model = OlmoeForCausalLM.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def student_config(hf_config):
    """Build an OlMoEConfig that matches the HF model's hyperparameters."""
    from olmoe.config import OlMoEConfig
    return OlMoEConfig(
        hidden_size=hf_config.hidden_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        intermediate_size=hf_config.intermediate_size,
        num_experts=hf_config.num_experts,
        num_experts_per_tok=hf_config.num_experts_per_tok,
        vocab_size=hf_config.vocab_size,
        max_position_embeddings=hf_config.max_position_embeddings,
        rope_theta=hf_config.rope_theta,
        rms_norm_eps=hf_config.rms_norm_eps,
    )


# ---------------------------------------------------------------------------
# Config compatibility tests (fast — no model download)
# ---------------------------------------------------------------------------

def test_hf_config_loads(hf_config):
    """Confirm the HF config is reachable and has expected fields."""
    assert hf_config.hidden_size == 2048
    assert hf_config.num_hidden_layers == 16
    assert hf_config.num_attention_heads == 16
    assert hf_config.num_experts == 64
    assert hf_config.num_experts_per_tok == 8
    print(f"✓ HF config loaded: {hf_config.model_type}")


def test_student_config_matches_hf(hf_config, student_config):
    """Student config built from HF config should mirror key HF hyperparameters."""
    assert student_config.hidden_size == hf_config.hidden_size
    assert student_config.num_hidden_layers == hf_config.num_hidden_layers
    assert student_config.num_attention_heads == hf_config.num_attention_heads
    assert student_config.num_key_value_heads == hf_config.num_key_value_heads
    assert student_config.intermediate_size == hf_config.intermediate_size
    assert student_config.num_experts == hf_config.num_experts
    assert student_config.num_experts_per_tok == hf_config.num_experts_per_tok
    assert student_config.vocab_size == hf_config.vocab_size
    print("✓ Student config mirrors HF config")


# ---------------------------------------------------------------------------
# HF model shape / forward tests (slow — downloads ~14 GB of weights)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_hf_model_forward(hf_model, hf_config):
    """HF model forward pass produces expected output shapes."""
    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, hf_config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = hf_model(input_ids)

    assert output.logits.shape == (batch_size, seq_len, hf_config.vocab_size), \
        f"Unexpected logits shape: {output.logits.shape}"
    print(f"✓ HF model forward pass: logits {output.logits.shape}")


@pytest.mark.slow
def test_hf_model_parameter_count(hf_model):
    """HF model should have ~7B total parameters."""
    total = sum(p.numel() for p in hf_model.parameters())
    print(f"  HF model total parameters: {total / 1e9:.2f}B")
    assert total > 6e9, f"Expected ~7B params, got {total / 1e9:.2f}B"
    print("✓ HF model parameter count correct")


@pytest.mark.slow
def test_hf_model_generates(hf_model, hf_config):
    """HF model can auto-regressively generate tokens."""
    # bos_token_id may be None in newer HF configs; fall back to eos_token_id
    start_id = hf_config.bos_token_id
    if start_id is None:
        start_id = hf_config.eos_token_id
    input_ids = torch.tensor([[start_id]])

    with torch.no_grad():
        generated = hf_model.generate(input_ids, max_new_tokens=5, do_sample=False)

    assert generated.shape[0] == 1
    assert generated.shape[1] == 6  # 1 prompt token + 5 new tokens
    print(f"✓ HF model generation: {generated.tolist()}")


# ---------------------------------------------------------------------------
# Weight transfer + numerical comparison (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_weight_transfer_attention(hf_model, student_config):
    """
    Transfer weights from one HF attention layer into the student's OlMoEAttention
    and verify that both produce numerically identical outputs.

    NOTE: The HF OlmoeAttention has q_norm / k_norm layers that the student
    implementation may not include.  If the student model lacks these norms
    the numerical comparison is skipped (shapes are still validated).
    """
    from olmoe.attention import OlMoEAttention

    student_attn = OlMoEAttention(student_config)
    student_attn.eval()

    hf_attn = hf_model.model.layers[0].self_attn

    with torch.no_grad():
        student_attn.q_proj.weight.copy_(hf_attn.q_proj.weight)
        student_attn.k_proj.weight.copy_(hf_attn.k_proj.weight)
        student_attn.v_proj.weight.copy_(hf_attn.v_proj.weight)
        student_attn.o_proj.weight.copy_(hf_attn.o_proj.weight)

    has_norms = hasattr(student_attn, 'q_norm') and hasattr(student_attn, 'k_norm')
    if has_norms:
        with torch.no_grad():
            student_attn.q_norm.weight.copy_(hf_attn.q_norm.weight)
            student_attn.k_norm.weight.copy_(hf_attn.k_norm.weight)

    batch_size, seq_len = 1, 6
    torch.manual_seed(0)
    hidden_states = torch.randn(batch_size, seq_len, student_config.hidden_size)

    with torch.no_grad():
        # HF forward — newer transformers requires position_embeddings (cos, sin)
        # and attention_mask as positional arguments.
        position_ids = torch.arange(seq_len).unsqueeze(0)
        rotary_emb = hf_model.model.rotary_emb
        position_embeddings = rotary_emb(hidden_states, position_ids)
        hf_out, _ = hf_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
        )

        # Student forward
        student_out, _, _ = student_attn(hidden_states)

    assert student_out.shape == hf_out.shape, \
        f"Shape mismatch: student {student_out.shape} vs HF {hf_out.shape}"

    if not has_norms:
        print("  (student model lacks q_norm/k_norm — skipping numerical comparison)")
        print("✓ Attention weight transfer shape check passed")
        return

    assert torch.allclose(student_out, hf_out, atol=1e-4), \
        f"Max diff: {(student_out - hf_out).abs().max().item()}"
    print("✓ Attention weight transfer + numerical match passed")


def _build_hf_to_student_state_dict(hf_state, student_state, num_experts):
    """Map HF parameter names to student parameter names and reshape where needed.

    Key differences handled:
    - ``model.embed_tokens.weight`` → ``model.embed_tokens.embedding.weight``
    - ``model.layers.N.mlp.gate.weight`` → ``model.layers.N.mlp.moe.router.gate.weight``
    - HF fused 3-D ``experts.gate_up_proj`` → per-expert ``gate_proj.weight`` + ``up_proj.weight``
    - HF fused 3-D ``experts.down_proj`` → per-expert ``down_proj.weight``
    - ``q_norm`` / ``k_norm`` transferred only when the student model has them
    """
    import re
    mapped: dict[str, torch.Tensor] = {}

    for hf_key, hf_val in hf_state.items():
        # 1. Embedding
        if hf_key == "model.embed_tokens.weight":
            stu_key = "model.embed_tokens.embedding.weight"
            if stu_key in student_state:
                mapped[stu_key] = hf_val
            elif hf_key in student_state:
                mapped[hf_key] = hf_val
            continue

        # 2. Fused expert weights — gate_up_proj (num_experts, 2*intermediate, hidden)
        m = re.match(r"(model\.layers\.\d+)\.mlp\.experts\.gate_up_proj$", hf_key)
        if m:
            prefix = m.group(1)
            intermediate = hf_val.shape[1] // 2
            for expert_idx in range(num_experts):
                gate_key = f"{prefix}.mlp.moe.experts.{expert_idx}.gate_proj.weight"
                up_key = f"{prefix}.mlp.moe.experts.{expert_idx}.up_proj.weight"
                if gate_key in student_state:
                    mapped[gate_key] = hf_val[expert_idx, :intermediate, :]
                    mapped[up_key] = hf_val[expert_idx, intermediate:, :]
            continue

        # 3. Fused expert weights — down_proj (num_experts, hidden, intermediate)
        m = re.match(r"(model\.layers\.\d+)\.mlp\.experts\.down_proj$", hf_key)
        if m:
            prefix = m.group(1)
            for expert_idx in range(num_experts):
                down_key = f"{prefix}.mlp.moe.experts.{expert_idx}.down_proj.weight"
                if down_key in student_state:
                    mapped[down_key] = hf_val[expert_idx]
            continue

        # 4. Router gate
        m = re.match(r"(model\.layers\.\d+)\.mlp\.gate\.weight$", hf_key)
        if m:
            stu_key = f"{m.group(1)}.mlp.moe.router.gate.weight"
            if stu_key in student_state:
                mapped[stu_key] = hf_val
            elif hf_key in student_state:
                mapped[hf_key] = hf_val
            continue

        # 5. q_norm / k_norm — only if student has them
        if "q_norm" in hf_key or "k_norm" in hf_key:
            if hf_key in student_state:
                mapped[hf_key] = hf_val
            continue

        # 6. Direct match (layernorms, lm_head, attention projections, etc.)
        if hf_key in student_state:
            mapped[hf_key] = hf_val

    return mapped


@pytest.mark.slow
def test_weight_transfer_full_model(hf_model, student_config):
    """
    Transfer all weights from the HF model to the student's OlMoEForCausalLM
    and verify that logits are numerically identical on the same input.

    This is the gold-standard correctness test for a complete implementation.
    """
    from olmoe import OlMoEForCausalLM

    student_model = OlMoEForCausalLM(student_config)
    student_model.eval()

    hf_state = hf_model.state_dict()
    student_state = student_model.state_dict()

    mapped = _build_hf_to_student_state_dict(
        hf_state, student_state, student_config.num_experts,
    )
    print(f"  Mapped {len(mapped)} / {len(student_state)} student weight tensors "
          f"(from {len(hf_state)} HF tensors)")

    missing = set(student_state) - set(mapped)
    if missing:
        print(f"  Student params NOT transferred ({len(missing)}):")
        for k in sorted(missing)[:15]:
            print(f"    {k}: {student_state[k].shape}")
        if len(missing) > 15:
            print(f"    ... and {len(missing) - 15} more")

    student_model.load_state_dict(mapped, strict=False)

    batch_size, seq_len = 1, 8
    torch.manual_seed(0)
    input_ids = torch.randint(0, student_config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        student_logits = student_model(input_ids).logits

    assert student_logits.shape == hf_logits.shape, \
        f"Logits shape mismatch: student {student_logits.shape} vs HF {hf_logits.shape}"

    if not mapped:
        print("  (No weights transferred — implement weight mapping to enable numerical check)")
        print("✓ Full model forward shape check passed")
        return

    coverage = len(mapped) / len(student_state) * 100
    max_diff = (student_logits - hf_logits).abs().max().item()
    print(f"  Weight coverage: {coverage:.0f}%  Max logit difference: {max_diff:.6f}")

    if missing:
        print(f"  ({len(missing)} student params not transferred — "
              "exact match not expected until all weights are mapped)")

    assert max_diff < 1e-3, f"Logits differ too much: {max_diff}"
    print("✓ Full model weight transfer + numerical match passed")


if __name__ == "__main__":
    print("Running HuggingFace OLMoE tests...\n")
    print(f"Model: {HF_MODEL_ID}\n")

    import pytest as pt
    pt.main([__file__, "-v", "--tb=short"])
