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
    return AutoConfig.from_pretrained(HF_MODEL_ID)


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
    assert hf_config.hidden_size == 1024
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
    input_ids = torch.tensor([[hf_config.bos_token_id]])

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
    """
    from olmoe.attention import OlMoEAttention

    student_attn = OlMoEAttention(student_config)
    student_attn.eval()

    # Grab the first HF decoder layer's self-attention
    hf_attn = hf_model.model.layers[0].self_attn

    # Copy weights (HF naming: q_proj, k_proj, v_proj, o_proj)
    with torch.no_grad():
        student_attn.q_proj.weight.copy_(hf_attn.q_proj.weight)
        student_attn.k_proj.weight.copy_(hf_attn.k_proj.weight)
        student_attn.v_proj.weight.copy_(hf_attn.v_proj.weight)
        student_attn.o_proj.weight.copy_(hf_attn.o_proj.weight)

    batch_size, seq_len = 1, 6
    hidden_states = torch.randn(batch_size, seq_len, student_config.hidden_size)

    with torch.no_grad():
        # HF forward
        position_ids = torch.arange(seq_len).unsqueeze(0)
        hf_out, _, _ = hf_attn(hidden_states, position_ids=position_ids)

        # Student forward
        student_out, _, _ = student_attn(hidden_states)

    assert student_out.shape == hf_out.shape, \
        f"Shape mismatch: student {student_out.shape} vs HF {hf_out.shape}"
    assert torch.allclose(student_out, hf_out, atol=1e-4), \
        f"Max diff: {(student_out - hf_out).abs().max().item()}"
    print("✓ Attention weight transfer + numerical match passed")


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

    # Build a mapping from HF parameter names to student parameter names.
    # Adjust this mapping once the student's weight names are known.
    hf_state = hf_model.state_dict()
    student_state = student_model.state_dict()

    print("  HF parameter names (first 10):")
    for k in list(hf_state.keys())[:10]:
        print(f"    {k}: {hf_state[k].shape}")

    print("  Student parameter names (first 10):")
    for k in list(student_state.keys())[:10]:
        print(f"    {k}: {student_state[k].shape}")

    # Attempt direct name-matched transfer (works when names align)
    common_keys = set(hf_state) & set(student_state)
    if common_keys:
        partial_state = {k: hf_state[k] for k in common_keys}
        student_model.load_state_dict(partial_state, strict=False)
        print(f"  Transferred {len(common_keys)} / {len(hf_state)} weight tensors")

    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, student_config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        student_logits = student_model(input_ids).logits

    assert student_logits.shape == hf_logits.shape, \
        f"Logits shape mismatch: student {student_logits.shape} vs HF {hf_logits.shape}"

    if common_keys:
        max_diff = (student_logits - hf_logits).abs().max().item()
        print(f"  Max logit difference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Logits differ too much: {max_diff}"
        print("✓ Full model weight transfer + numerical match passed")
    else:
        print("  (No common weight keys found — implement weight mapping to enable numerical check)")
        print("✓ Full model forward shape check passed")


if __name__ == "__main__":
    print("Running HuggingFace OLMoE tests...\n")
    print(f"Model: {HF_MODEL_ID}\n")

    import pytest as pt
    pt.main([__file__, "-v", "--tb=short"])
