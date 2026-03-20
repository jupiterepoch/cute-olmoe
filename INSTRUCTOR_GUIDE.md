# OlMoE Implementation - Instructor's Guide

This guide is for instructors teaching the OlMoE implementation course.

## Course Overview

**Duration**: 4 weeks (recommended)

**Learning Objectives**:
1. Understand transformer architecture deeply
2. Learn about Mixture of Experts (MoE) sparse models
3. Implement modern techniques: RoPE, GQA, RMSNorm
4. Practice PyTorch implementation skills
5. Debug and test deep learning models

## Week-by-Week Breakdown

### Week 1: Foundations

**Topics**:
- Configuration management
- Embeddings and RoPE
- Utility functions (RMSNorm, activations)

**Key Learning Points**:
- Why RMSNorm is preferred over LayerNorm
- How rotary embeddings encode position
- Proper parameter validation

**Common Student Mistakes**:
1. Not implementing `__post_init__` validation
2. Wrong dimensions for RoPE (using hidden_size instead of head_dim)
3. Not handling edge cases in load balancing loss

**Grading Rubric**:
- Config validation (20%)
- RMSNorm implementation (30%)
- RoPE implementation (50%)

### Week 2: Attention Mechanism

**Topics**:
- Multi-head attention
- Grouped Query Attention (GQA)
- Causal masking
- KV caching

**Key Learning Points**:
- How GQA reduces memory while maintaining quality
- Why causal masking is needed
- The relationship between queries, keys, and values

**Common Student Mistakes**:
1. Wrong attention mask (using multiplication instead of addition)
2. Forgetting to scale by sqrt(head_dim)
3. Not properly repeating KV heads in GQA
4. Shape mismatches when splitting/merging heads

**Debugging Tips for Students**:
```python
# Print shapes at each step
print(f"Q: {Q.shape}")  # Should be (batch, heads, seq, head_dim)
print(f"K: {K.shape}")  # Should be (batch, kv_heads, seq, head_dim)
print(f"V: {V.shape}")  # Should be (batch, kv_heads, seq, head_dim)
```

**Grading Rubric**:
- Shape correctness (30%)
- Causal masking (25%)
- GQA implementation (25%)
- RoPE integration (20%)

### Week 3: Mixture of Experts

**Topics**:
- Router/Gate network
- Top-k expert selection
- Load balancing loss
- Expert parallelism concepts

**Key Learning Points**:
- How sparse routing works
- Why load balancing is critical
- Trade-offs: more experts vs. computational efficiency

**Common Student Mistakes**:
1. Not normalizing routing weights properly
2. Forgetting to handle case where no tokens go to an expert
3. Not accumulating gradients properly (using index_copy instead of index_add)
4. Wrong load balancing loss formula

**Critical Implementation Detail**:
```python
# WRONG: This doesn't accumulate gradients
final_output[token_indices] = expert_output * weights

# CORRECT: This accumulates
final_output.index_add_(0, token_indices, expert_output * weights)
```

**Grading Rubric**:
- Router correctness (25%)
- Top-k selection (20%)
- Expert processing (30%)
- Load balancing loss (25%)

### Week 4: Full Model Integration

**Topics**:
- Decoder layer assembly
- Loss computation
- Generation
- End-to-end testing

**Key Learning Points**:
- How residual connections help gradient flow
- Pre-normalization vs post-normalization
- Combining language modeling loss with auxiliary loss

**Common Student Mistakes**:
1. Wrong residual connection placement
2. Not aggregating aux loss from all layers
3. Incorrect loss weighting
4. Memory leaks in generation loop

**Grading Rubric**:
- Model architecture (30%)
- Loss computation (25%)
- Forward/backward pass (25%)
- Tests passing (20%)

## Solution Hints

### Config Validation

```python
def __post_init__(self):
    assert self.hidden_size % self.num_attention_heads == 0, \
        "hidden_size must be divisible by num_attention_heads"
    assert self.num_attention_heads % self.num_key_value_heads == 0, \
        "num_attention_heads must be divisible by num_key_value_heads"
    assert self.num_experts_per_tok <= self.num_experts, \
        "num_experts_per_tok cannot exceed num_experts"

@property
def head_dim(self) -> int:
    return self.hidden_size // self.num_attention_heads
```

### RMSNorm

```python
def forward(self, x):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + self.eps)
    return x * self.weight
```

### RoPE

```python
def __init__(self, dim, max_position_embeddings, base):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer("inv_freq", inv_freq)

def forward(self, x, seq_len):
    t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
    freqs = torch.outer(t, self.inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()
```

### Load Balancing Loss

```python
def compute_load_balancing_loss(gate_logits, num_experts, top_k):
    routing_probs = F.softmax(gate_logits, dim=-1)
    P = routing_probs.mean(dim=0)

    _, selected = torch.topk(gate_logits, top_k, dim=-1)
    expert_mask = torch.zeros_like(routing_probs).scatter_(1, selected, 1.0)
    f = expert_mask.mean(dim=0)

    loss = num_experts * (f * P).sum()
    return loss
```

### MoE Forward Pass (Key Part)

```python
# Process each expert
for expert_idx in range(self.num_experts):
    expert_mask = (selected_experts == expert_idx)

    if expert_mask.any():
        token_indices, top_k_indices = torch.where(expert_mask)

        expert_input = hidden_states[token_indices]
        expert_output = self.experts[expert_idx](expert_input)

        weights = routing_weights[token_indices, top_k_indices, None]

        # CRITICAL: Use index_add_ for gradient accumulation
        final_output.index_add_(0, token_indices, expert_output * weights)
```

## Teaching Tips

### 1. Start with Visualization

Use `examples/visualize_architecture.py` to show:
- Model structure
- Parameter counts
- Shape transformations
- Expert routing patterns

### 2. Incremental Testing

Encourage students to:
- Run tests after each component
- Use small models for debugging
- Print intermediate shapes
- Verify gradients flow

### 3. Common Debugging Workflow

```python
# 1. Check shapes
print(f"Shape: {tensor.shape}")

# 2. Check for NaNs
assert not torch.isnan(tensor).any(), "NaN detected!"

# 3. Check gradient flow
assert tensor.requires_grad, "No gradient!"

# 4. Check values are reasonable
print(f"Min: {tensor.min()}, Max: {tensor.max()}, Mean: {tensor.mean()}")
```

### 4. Progressive Complexity

Start with simplified versions:

**Week 1**: Single-head attention (no multi-head)
**Week 2**: Multi-head without GQA
**Week 3**: 2 experts before 8
**Week 4**: Full model

### 5. Office Hours Topics

Common questions:
1. "Why is my attention mask not working?"
   - Check: Using addition, not multiplication
   - Check: -inf values, not 0

2. "Why are all tokens going to one expert?"
   - Check: Load balancing loss is included
   - Check: Router weights initialized properly

3. "Why is my loss NaN?"
   - Check: Division by zero in normalization
   - Check: Log of negative/zero
   - Check: Overflow in softmax (use float32)

4. "Why is gradient not flowing?"
   - Check: .detach() not called accidentally
   - Check: Using index_add_ not index_copy_
   - Check: requires_grad=True

## Assessment

### Project Grading

**Total: 100 points**

1. **Code Correctness (60 points)**
   - All unit tests pass (30 pts)
   - Forward pass produces valid output (15 pts)
   - Backward pass computes gradients (15 pts)

2. **Code Quality (20 points)**
   - Clean, readable code (10 pts)
   - Proper documentation (5 pts)
   - Follows project structure (5 pts)

3. **Understanding (20 points)**
   - Written explanation of MoE (10 pts)
   - Analysis of expert usage (5 pts)
   - Discussion of design choices (5 pts)

### Bonus Points (up to 10 extra points)

- Implement advanced generation (top-k, top-p): +5
- Add visualization tools: +3
- Implement expert capacity constraints: +5
- Train on real data: +5
- Performance optimizations: +5

## Advanced Extensions

For advanced students:

### 1. Flash Attention

Implement memory-efficient attention:
```python
# Use flash attention for efficiency
from flash_attn import flash_attn_func
attn_output = flash_attn_func(q, k, v, causal=True)
```

### 2. Expert Parallelism

Distribute experts across GPUs:
```python
# Each GPU handles subset of experts
expert_group = experts[rank::world_size]
```

### 3. Expert Capacity

Limit tokens per expert:
```python
capacity = (num_tokens // num_experts) * capacity_factor
# Drop tokens exceeding capacity
```

### 4. Auxiliary Load Balancing Losses

Implement importance loss, z-loss:
```python
# Z-loss: penalizes large router logits
z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
```

## Resources for Teaching

### Papers to Assign

1. **Required**:
   - Attention is All You Need (Vaswani et al., 2017)
   - Switch Transformers (Fedus et al., 2021)
   - OlMoE (Muennighoff et al., 2024)

2. **Optional**:
   - RoFormer (Su et al., 2021) - for RoPE
   - GQA (Ainslie et al., 2023) - for grouped query attention
   - ST-MoE (Zoph et al., 2022) - for advanced MoE

### Videos

- 3Blue1Brown: Attention in transformers
- Andrej Karpathy: Let's build GPT
- Yannic Kilcher: Switch Transformers paper explained

### Interactive Resources

- transformer-circuits.pub - Interpretability
- The Illustrated Transformer - Jay Alammar
- Hugging Face Course - Transformers

## FAQ for Instructors

**Q: How long does implementation typically take?**
A: 20-40 hours total. Attention is the hardest part (8-12 hours).

**Q: What's the hardest concept for students?**
A: Usually the MoE routing and gradient accumulation.

**Q: Should I provide partial solutions?**
A: For Week 1-2, yes. For Week 3-4, let students struggle more.

**Q: What about students with limited compute?**
A: All examples work on CPU with small models. No GPU needed for learning.

**Q: How to prevent plagiarism?**
A: Require written explanations. Ask students to modify architecture (e.g., top-3 instead of top-2).

**Q: What if tests fail?**
A: Partial credit. 70% for attempt, 100% for passing tests.

## Conclusion

This course teaches:
- Deep understanding of transformers
- Modern ML engineering practices
- Debugging complex systems
- Reading and implementing research papers

Students finishing this course should be able to:
- Implement transformer variants from papers
- Understand and modify existing LLM codebases
- Debug deep learning models systematically
- Make architectural decisions with trade-offs in mind

Good luck teaching!
