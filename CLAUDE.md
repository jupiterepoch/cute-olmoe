# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational implementation of OlMoE-1B-7B (Open Language Model with Mixture of Experts) from scratch. Students implement each component guided by `# TODO` markers in the source files.

**Model specs:** 7B total params, ~1B activated per token (top-2 of 8 experts), hidden size 1024, 16 layers, 16 attention heads, max seq len 2048.

## Commands

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/ -v
pytest tests/test_attention.py -v   # single file
python tests/test_attention.py      # direct execution

# Examples
python examples/train_simple.py
python examples/visualize_architecture.py
```

## Architecture

The `olmoe/` package implements each component as a separate module with a deliberate dependency order:

1. **`config.py`** — `OlMoEConfig` dataclass with all hyperparameters
2. **`utils.py`** — `RMSNorm`, `SwiGLU`, RoPE helpers, load balancing loss
3. **`embeddings.py`** — Token embeddings + `RotaryPositionEmbedding`
4. **`attention.py`** — `MultiHeadAttention` with Grouped Query Attention (GQA) and RoPE
5. **`feedforward.py`** — `ExpertFFN` using SwiGLU (each expert is an independent FFN)
6. **`moe.py`** — `MoERouter` + `MixtureOfExperts` layer with top-k routing and load balancing loss
7. **`model.py`** — `OlMoEDecoderLayer`, `OlMoEModel`, `OlMoEForCausalLM` (full assembly)

Each file contains `# TODO` markers guiding student implementation. Tests in `tests/` validate each component and are designed to be run incrementally as students complete each module.

## Key Implementation Details

- **GQA:** `num_kv_heads` < `num_heads`; keys/values are shared across head groups
- **RoPE:** Applied to Q and K after projection; rotation frequencies precomputed in `RotaryPositionEmbedding`
- **MoE routing:** Softmax over router logits → top-k selection → sparse dispatch to experts → weighted sum
- **Load balancing loss:** Auxiliary loss added to main loss to prevent expert collapse; controlled by `aux_loss_coef` in config
- **Expert capacity:** Each expert processes a limited token budget per forward pass
