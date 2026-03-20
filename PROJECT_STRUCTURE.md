# OlMoE Project Structure

This document provides an overview of the complete project structure.

## Directory Layout

```
cute-olmoe/
├── README.md                      # Main project overview
├── TUTORIAL.md                    # Comprehensive implementation guide
├── INSTRUCTOR_GUIDE.md            # Teaching guide with solutions
├── PROJECT_STRUCTURE.md           # This file
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation script
├── .gitignore                     # Git ignore patterns
│
├── olmoe/                         # Main package directory
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Model configuration
│   ├── utils.py                  # Utility functions (RMSNorm, activations, etc.)
│   ├── embeddings.py             # Token and positional embeddings
│   ├── attention.py              # Multi-head attention with RoPE and GQA
│   ├── feedforward.py            # Expert feed-forward networks
│   ├── moe.py                    # Mixture of Experts layer
│   └── model.py                  # Complete OlMoE model
│
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_attention.py         # Attention mechanism tests
│   ├── test_moe.py               # MoE layer tests
│   └── test_model.py             # End-to-end model tests
│
└── examples/                      # Example scripts
    ├── README.md                  # Examples overview
    ├── train_simple.py            # Simple training example
    └── visualize_architecture.py  # Architecture visualization
```

## File Descriptions

### Root Directory

- **README.md**: Project overview, setup instructions, and learning path
- **TUTORIAL.md**: Detailed implementation tutorial with code examples
- **INSTRUCTOR_GUIDE.md**: Teaching guide with solutions and grading rubrics
- **requirements.txt**: Python package dependencies
- **setup.py**: Package setup for pip installation
- **.gitignore**: Patterns for files to exclude from git

### olmoe/ Package

Core implementation with TODOs for students to complete:

1. **config.py** (~ 100 lines)
   - `OlMoEConfig`: Configuration dataclass
   - `get_olmoe_1b_7b_config()`: Default configuration
   - TODOs: Validation logic, head_dim property

2. **utils.py** (~ 150 lines)
   - `RMSNorm`: Root Mean Square normalization
   - `get_activation_function()`: Activation function factory
   - `rotate_half()`: Helper for RoPE
   - `apply_rotary_pos_emb()`: Apply RoPE to Q/K
   - `compute_load_balancing_loss()`: MoE auxiliary loss
   - TODOs: All function implementations

3. **embeddings.py** (~ 120 lines)
   - `OlMoEEmbedding`: Token embeddings
   - `RotaryEmbedding`: Rotary position embeddings
   - TODOs: Embedding lookup, RoPE frequency computation

4. **attention.py** (~ 230 lines)
   - `OlMoEAttention`: Multi-head self-attention
   - Supports Grouped Query Attention (GQA)
   - Includes RoPE and causal masking
   - `_make_causal_mask()`: Causal attention mask
   - TODOs: Complete attention mechanism

5. **feedforward.py** (~ 80 lines)
   - `OlMoEFeedForward`: Single expert FFN
   - `OlMoESparseMLP`: FFN wrapper with dropout
   - Uses SwiGLU activation
   - TODOs: FFN forward pass

6. **moe.py** (~ 180 lines)
   - `OlMoERouter`: Token-to-expert routing
   - `OlMoESparseMoE`: Sparse MoE layer
   - `OlMoEMoEBlock`: MoE with auxiliary loss
   - `print_expert_usage()`: Debug helper
   - TODOs: Routing logic, expert processing, load balancing

7. **model.py** (~ 250 lines)
   - `OlMoEDecoderLayer`: Single transformer layer
   - `OlMoEModel`: Base transformer model
   - `OlMoEForCausalLM`: Model with LM head
   - `OlMoEOutput`: Output dataclass
   - `create_olmoe_model()`: Factory function
   - TODOs: Layer composition, forward pass, generation

8. **__init__.py** (~ 40 lines)
   - Package exports
   - Version information

### tests/ Package

Comprehensive test suite:

1. **test_attention.py** (~ 150 lines)
   - `test_attention_shapes()`: Output shape verification
   - `test_grouped_query_attention()`: GQA correctness
   - `test_rotary_embeddings()`: RoPE generation
   - `test_attention_causality()`: Causal mask verification

2. **test_moe.py** (~ 180 lines)
   - `test_router_shapes()`: Router output shapes
   - `test_router_top_k()`: Top-k selection correctness
   - `test_moe_shapes()`: MoE layer shapes
   - `test_load_balancing()`: Auxiliary loss computation
   - `test_expert_diversity()`: Expert usage distribution
   - `test_moe_forward_backward()`: Gradient flow

3. **test_model.py** (~ 200 lines)
   - `test_model_creation()`: Model instantiation
   - `test_model_forward()`: Forward pass
   - `test_model_with_labels()`: Training mode
   - `test_model_backward()`: Gradient computation
   - `test_parameter_count()`: Parameter statistics
   - `test_model_factory()`: Factory function
   - `test_caching()`: KV cache functionality

### examples/ Directory

Example scripts and documentation:

1. **README.md** (~ 150 lines)
   - Overview of examples
   - Usage instructions
   - Ideas for extensions

2. **train_simple.py** (~ 180 lines)
   - `create_tiny_model()`: Small model for testing
   - `generate_dummy_data()`: Random training data
   - `train_step()`: Single training iteration
   - `simple_generate()`: Basic text generation
   - Complete training loop example

3. **visualize_architecture.py** (~ 300 lines)
   - `print_separator()`: Pretty printing helper
   - `analyze_parameters()`: Parameter breakdown
   - `trace_forward_pass()`: Shape tracing
   - `analyze_expert_routing()`: Routing visualization
   - `print_model_structure()`: Hierarchical structure

## Implementation Statistics

### Lines of Code (approximate)

```
olmoe/ package:        ~1,150 lines (with TODOs and documentation)
tests/ package:        ~  530 lines
examples/:             ~  630 lines
Documentation:         ~2,500 lines (README, TUTORIAL, INSTRUCTOR_GUIDE)
───────────────────────────────────────────────────────
Total:                 ~4,810 lines
```

### TODO Count

Students need to implement approximately **35-40 TODOs** across:
- Config validation: 2 TODOs
- Utils: 8 TODOs
- Embeddings: 4 TODOs
- Attention: 10 TODOs
- Feedforward: 3 TODOs
- MoE: 8 TODOs
- Model: 10 TODOs

### Complexity Levels

**Beginner** (Week 1):
- Config validation ⭐
- Token embeddings ⭐
- RMSNorm ⭐⭐

**Intermediate** (Week 2):
- RoPE ⭐⭐⭐
- Attention mechanism ⭐⭐⭐⭐
- GQA ⭐⭐⭐

**Advanced** (Week 3):
- Router ⭐⭐⭐
- MoE forward pass ⭐⭐⭐⭐⭐
- Load balancing loss ⭐⭐⭐⭐

**Integration** (Week 4):
- Decoder layer ⭐⭐⭐
- Full model ⭐⭐⭐⭐
- Generation ⭐⭐⭐⭐

## Key Features

### Educational Design

1. **Incremental Complexity**: Start simple, build up
2. **Comprehensive TODOs**: Clear instructions for each implementation
3. **Shape Annotations**: Expected tensor shapes documented
4. **Test-Driven**: Tests guide implementation
5. **Multiple Examples**: Different learning approaches

### Modern ML Techniques

Students learn:
- **Rotary Position Embeddings (RoPE)**: Efficient position encoding
- **Grouped Query Attention (GQA)**: Memory-efficient attention
- **Mixture of Experts (MoE)**: Sparse conditional computation
- **RMSNorm**: Faster alternative to LayerNorm
- **SwiGLU**: Better activation for transformers
- **Load Balancing**: Auxiliary losses for expert usage

### Best Practices

- Type hints throughout
- Comprehensive docstrings
- Clean code structure
- Proper testing
- Documentation

## Installation

```bash
# Clone repository
cd cute-olmoe

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Run tests
pytest tests/

# Try examples
python examples/train_simple.py
python examples/visualize_architecture.py
```

## Learning Path

1. **Read**: README.md for overview
2. **Study**: TUTORIAL.md for detailed guide
3. **Implement**: Follow TODOs in code
4. **Test**: Run unit tests after each component
5. **Experiment**: Use examples to understand behavior
6. **Extend**: Add your own features

## Success Criteria

Students successfully complete the project when:

✓ All unit tests pass
✓ Model can run forward pass
✓ Gradients flow through all components
✓ Can train on simple data
✓ Understand architecture deeply

## Extensions

After completing the base implementation:

1. Train on real text data
2. Implement advanced generation (beam search, top-p)
3. Add model checkpointing
4. Implement Flash Attention
5. Add expert capacity constraints
6. Visualize expert specialization
7. Implement distributed training
8. Profile and optimize performance

## Credits

This educational project implements OlMoE-1B-7B as a learning exercise.

**Original Paper**: OlMoE: Open Mixture-of-Experts Language Models
**Authors**: Muennighoff et al., 2024
**Organization**: Allen Institute for AI

## License

Educational use. See original OlMoE repository for model license.
