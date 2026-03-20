# OlMoE-1B-7B Implementation Tutorial

Welcome to the Mixture of Experts (MoE) Language Model implementation course!

## Overview

This project provides a starter template for implementing OlMoE-1B-7B, a 1B parameter Mixture of Experts language model with 8 experts (top-2 routing). You'll build the model from scratch to understand modern MoE architectures.

## Model Architecture

**OlMoE-1B-7B Specifications:**
- Total Parameters: ~7B (8 experts)
- Activated Parameters: ~1B per token (top-2 routing)
- Hidden Size: 1024
- Number of Layers: 16
- Attention Heads: 16
- Number of Experts: 8
- Experts per Token: 2
- Vocabulary Size: 50304
- Max Sequence Length: 2048

## Project Structure

```
olmoe/
├── config.py           # Model configuration dataclass
├── model.py            # Main OlMoE model
├── attention.py        # Multi-head attention with RoPE
├── moe.py              # Mixture of Experts layer
├── feedforward.py      # Expert feed-forward networks
├── embeddings.py       # Token and positional embeddings
└── utils.py            # Helper functions

tests/
├── test_attention.py   # Test attention mechanism
├── test_moe.py         # Test MoE routing
└── test_model.py       # End-to-end model tests

examples/
└── train_simple.py     # Simple training example
```

## Learning Path

Students should implement components in this order:

### Phase 1: Foundation (Week 1)
1. `config.py` - Understand model configuration
2. `embeddings.py` - Token embeddings and RoPE
3. `utils.py` - Helper functions (RMSNorm, activations)

### Phase 2: Core Components (Week 2)
4. `attention.py` - Multi-head self-attention with RoPE
5. `feedforward.py` - Simple feed-forward networks

### Phase 3: MoE Magic (Week 3)
6. `moe.py` - Router, expert selection, load balancing
7. Integration of MoE into transformer blocks

### Phase 4: Full Model (Week 4)
8. `model.py` - Complete OlMoE architecture
9. Testing and debugging
10. Simple training loop

## Key Concepts to Learn

1. **Rotary Positional Embeddings (RoPE)**: Efficient position encoding
2. **Grouped Query Attention**: Memory-efficient attention
3. **Sparse MoE Routing**: Top-k expert selection
4. **Load Balancing**: Auxiliary loss for balanced expert usage
5. **Expert Capacity**: Managing token distribution
6. **Residual Connections**: Skip connections in transformers
7. **RMSNorm**: Efficient layer normalization variant

## Implementation TODOs

Each file contains:
- Class and method signatures
- Docstrings explaining functionality
- `TODO:` comments marking implementation areas
- Shape annotations for tensors
- Example test cases

## Getting Started

1. Install dependencies:
```bash
pip install torch numpy einops
pip install transformers  # For tokenizer
```

2. Start with `config.py` to understand model parameters

3. Implement each component following the TODOs

4. Run tests to verify your implementation:
```bash
python -m pytest tests/
```

## Resources

- OlMoE Paper: https://arxiv.org/abs/2409.02060
- Attention is All You Need: https://arxiv.org/abs/1706.03762
- RoPE Paper: https://arxiv.org/abs/2104.09864
- Switch Transformers (MoE): https://arxiv.org/abs/2101.03961

## Submission

Complete all TODO sections and ensure tests pass. Your implementation should:
- Match the expected tensor shapes
- Pass provided unit tests
- Run a simple forward pass without errors

Good luck and have fun learning about MoE models!
