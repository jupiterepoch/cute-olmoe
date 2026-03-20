# OlMoE Implementation Tutorial

A comprehensive guide for implementing OlMoE-1B-7B from scratch.

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Implementation Guide](#implementation-guide)
4. [Testing Your Implementation](#testing-your-implementation)
5. [Common Pitfalls](#common-pitfalls)
6. [Advanced Topics](#advanced-topics)

## Introduction

### What is OlMoE?

OlMoE (Open Language Model with Mixture of Experts) is a sparse language model that uses conditional computation. Instead of passing every token through the same feed-forward network, OlMoE has multiple "expert" networks and routes each token to the top-k most relevant experts.

**Key Innovation**: Sparse MoE allows the model to have more total parameters while keeping computation per token constant.

Example for OlMoE-1B-7B:
- Total parameters: ~7B (8 experts × FFN params + shared params)
- Active parameters per token: ~1B (top-2 experts active)
- Result: Better quality than 1B dense model, same speed!

### Architecture Overview

```
Input Tokens
    ↓
Embedding Layer (shared)
    ↓
┌──────────────────────────┐
│  Transformer Blocks (×16) │
│  ┌───────────────────┐   │
│  │ RMSNorm           │   │
│  │ Self-Attention    │   │
│  │ Residual Add      │   │
│  └───────────────────┘   │
│  ┌───────────────────┐   │
│  │ RMSNorm           │   │
│  │ MoE Layer         │   │
│  │  ├─ Router        │   │
│  │  ├─ Expert 0      │   │
│  │  ├─ Expert 1      │   │
│  │  ├─ ...           │   │
│  │  └─ Expert 7      │   │
│  │ Residual Add      │   │
│  └───────────────────┘   │
└──────────────────────────┘
    ↓
Final RMSNorm
    ↓
LM Head (Linear → Vocab)
    ↓
Logits
```

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Basic understanding of transformers

### Installation

```bash
# Clone the repository
cd cute-olmoe

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Implementation Guide

Follow this order for implementing components:

### Phase 1: Foundation Components

#### 1.1 Configuration (`config.py`)

**Concepts to understand:**
- Model hyperparameters
- Validation of configuration

**Key implementation:**
```python
def __post_init__(self):
    # Validate hidden_size is divisible by num_attention_heads
    assert self.hidden_size % self.num_attention_heads == 0
    # Validate GQA: num_attention_heads % num_key_value_heads == 0
    # Validate MoE: num_experts_per_tok <= num_experts
```

#### 1.2 Utilities (`utils.py`)

**A. RMSNorm**

RMSNorm is simpler than LayerNorm - it only normalizes by the RMS (no mean centering):

```python
def forward(self, x):
    # 1. Compute variance (mean of squares)
    variance = x.pow(2).mean(dim=-1, keepdim=True)

    # 2. Normalize
    x = x / torch.sqrt(variance + self.eps)

    # 3. Scale
    return x * self.weight
```

**B. Rotary Position Embeddings Helper**

```python
def rotate_half(x):
    # Split tensor in half along last dimension
    x1, x2 = x.chunk(2, dim=-1)
    # Rotate: [x1, x2] -> [-x2, x1]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # Apply rotation: x_rotated = x * cos + rotate_half(x) * sin
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed
```

**C. Load Balancing Loss**

This encourages balanced expert usage:

```python
def compute_load_balancing_loss(gate_logits, num_experts, top_k):
    # 1. Routing probabilities
    routing_probs = F.softmax(gate_logits, dim=-1)  # (tokens, experts)

    # 2. P_i: average probability for each expert
    P = routing_probs.mean(dim=0)  # (experts,)

    # 3. f_i: fraction selecting each expert
    # Get top-k selections
    _, selected = torch.topk(gate_logits, top_k, dim=-1)
    # One-hot encode and average
    f = torch.zeros_like(routing_probs).scatter_(1, selected, 1.0).mean(dim=0)

    # 4. Loss: num_experts * sum(f_i * P_i)
    loss = num_experts * (f * P).sum()
    return loss
```

#### 1.3 Embeddings (`embeddings.py`)

**A. Token Embeddings**

Simple lookup table:
```python
self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

def forward(self, input_ids):
    return self.embedding(input_ids)
```

**B. Rotary Embeddings (RoPE)**

RoPE encodes position by rotating query/key vectors:

```python
def __init__(self, dim, max_position_embeddings, base):
    # Compute inverse frequencies
    # inv_freq[i] = 1 / (base^(2i/dim)) for i in [0, dim/2)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer("inv_freq", inv_freq)

def forward(self, x, seq_len):
    # 1. Position indices: [0, 1, 2, ..., seq_len-1]
    t = torch.arange(seq_len, device=x.device).float()

    # 2. Compute frequencies: outer product
    freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim//2)

    # 3. Concatenate to get full dimension
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)

    # 4. Return cos and sin
    return emb.cos(), emb.sin()
```

### Phase 2: Attention Mechanism

#### 2.1 Multi-Head Attention (`attention.py`)

**Key concepts:**
- Split hidden dimension into multiple heads
- Grouped Query Attention (GQA): Share KV heads across multiple Q heads
- Apply RoPE to Q and K
- Compute attention scores and apply causal mask

**Implementation steps:**

```python
def forward(self, hidden_states, attention_mask):
    batch_size, seq_len, _ = hidden_states.shape

    # 1. Project to Q, K, V
    Q = self.q_proj(hidden_states)  # (batch, seq, hidden)
    K = self.k_proj(hidden_states)  # (batch, seq, kv_hidden)
    V = self.v_proj(hidden_states)  # (batch, seq, kv_hidden)

    # 2. Split into heads
    # Q: (batch, seq, hidden) -> (batch, num_heads, seq, head_dim)
    Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    K = K.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    V = V.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # 3. Apply RoPE
    cos, sin = self.rotary_emb(V, seq_len)
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

    # 4. Repeat KV heads for GQA
    # If num_heads=16 and num_kv_heads=4, repeat each KV head 4 times
    K = self._repeat_kv(K, self.num_key_value_groups)
    V = self._repeat_kv(V, self.num_key_value_groups)

    # 5. Attention scores: Q @ K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

    # 6. Apply causal mask
    scores = scores + attention_mask  # mask has -inf for future positions

    # 7. Softmax
    attn_weights = F.softmax(scores, dim=-1)

    # 8. Apply to values
    output = torch.matmul(attn_weights, V)  # (batch, heads, seq, head_dim)

    # 9. Merge heads
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    # 10. Output projection
    output = self.o_proj(output)

    return output, attn_weights, None
```

**Helper: Repeat KV for GQA**
```python
def _repeat_kv(self, hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    # Repeat each head n_rep times
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
```

### Phase 3: Feed-Forward Networks

#### 3.1 Expert FFN (`feedforward.py`)

Each expert uses SwiGLU activation:

```python
def forward(self, x):
    # SwiGLU: (activation(gate) * up) @ down
    gate = self.act_fn(self.gate_proj(x))  # (*, intermediate_size)
    up = self.up_proj(x)                    # (*, intermediate_size)
    hidden = gate * up                      # Element-wise multiplication
    output = self.down_proj(hidden)         # (*, hidden_size)
    return output
```

### Phase 4: Mixture of Experts (The Core Innovation!)

#### 4.1 Router (`moe.py`)

The router decides which experts process each token:

```python
class OlMoERouter(nn.Module):
    def __init__(self, config):
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states):
        # 1. Compute logits for all experts
        router_logits = self.gate(hidden_states)  # (batch*seq, num_experts)

        # 2. Select top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)

        # 3. Normalize weights (softmax over selected experts)
        routing_weights = F.softmax(top_k_logits, dim=-1)

        return routing_weights, top_k_indices, router_logits
```

#### 4.2 MoE Layer

This is the most complex part! Here's a clear strategy:

```python
def forward(self, hidden_states):
    batch_size, seq_len, hidden_dim = hidden_states.shape

    # 1. Flatten to (batch*seq, hidden)
    hidden_states = hidden_states.view(-1, hidden_dim)
    num_tokens = hidden_states.shape[0]

    # 2. Route tokens
    routing_weights, selected_experts, router_logits = self.router(hidden_states)
    # routing_weights: (num_tokens, top_k)
    # selected_experts: (num_tokens, top_k)

    # 3. Initialize output
    final_output = torch.zeros_like(hidden_states)

    # 4. Process tokens for each expert
    for expert_idx in range(self.num_experts):
        # Find which tokens selected this expert
        expert_mask = (selected_experts == expert_idx)  # (num_tokens, top_k)

        if expert_mask.any():
            # Get token indices and their routing weights
            token_indices, top_k_indices = torch.where(expert_mask)

            # Extract tokens for this expert
            expert_input = hidden_states[token_indices]

            # Process through expert
            expert_output = self.experts[expert_idx](expert_input)

            # Get routing weights for these tokens
            weights = routing_weights[token_indices, top_k_indices, None]

            # Add weighted output back
            final_output.index_add_(0, token_indices, expert_output * weights)

    # 5. Reshape back
    final_output = final_output.view(batch_size, seq_len, hidden_dim)

    return final_output, router_logits
```

### Phase 5: Complete Model

#### 5.1 Decoder Layer (`model.py`)

Combines attention and MoE:

```python
def forward(self, hidden_states, attention_mask):
    # 1. Self-attention with residual
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _, _ = self.self_attn(hidden_states, attention_mask)
    hidden_states = residual + hidden_states

    # 2. MoE with residual
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states, aux_loss = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, aux_loss
```

#### 5.2 Full Model

```python
def forward(self, input_ids, labels=None):
    # 1. Embeddings
    hidden_states = self.model.embed_tokens(input_ids)

    # 2. Create causal mask
    attention_mask = _make_causal_mask(...)

    # 3. Through all layers
    total_aux_loss = 0
    for layer in self.model.layers:
        hidden_states, aux_loss = layer(hidden_states, attention_mask)
        total_aux_loss += aux_loss

    # 4. Final norm
    hidden_states = self.model.norm(hidden_states)

    # 5. LM head
    logits = self.lm_head(hidden_states)

    # 6. Compute loss if labels provided
    loss = None
    if labels is not None:
        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1)
        )
        # Add weighted auxiliary loss
        loss = loss + self.config.router_aux_loss_coef * total_aux_loss

    return OlMoEOutput(logits=logits, loss=loss, aux_loss=total_aux_loss)
```

## Testing Your Implementation

### Unit Tests

Run tests after completing each component:

```bash
# Test attention
python tests/test_attention.py

# Test MoE
python tests/test_moe.py

# Test full model
python tests/test_model.py

# Or use pytest
pytest tests/
```

### Debugging Tips

1. **Shape mismatches**: Print shapes at each step
```python
print(f"Q shape: {Q.shape}")  # Should be (batch, heads, seq, head_dim)
```

2. **NaN losses**: Check for:
   - Division by zero
   - Log of zero/negative
   - Overflow in exp (use float32, not float16)

3. **Expert imbalance**: Print expert usage
```python
from olmoe.moe import print_expert_usage
print_expert_usage(selected_experts, num_experts)
```

## Common Pitfalls

### 1. Attention Mask Signs

Wrong:
```python
mask = torch.triu(torch.ones(...))  # 1s for future
scores = scores * mask  # Wrong! Multiplies by 1
```

Right:
```python
mask = torch.triu(torch.ones(...), diagonal=1).bool()
mask = mask.masked_fill(mask, float('-inf'))
scores = scores + mask  # Adds -inf to future positions
```

### 2. MoE Token Distribution

Common mistake: Not handling case where no tokens go to an expert

```python
# Always check if expert has tokens
if expert_mask.any():
    # Process...
```

### 3. Gradient Flow in MoE

Make sure gradients flow through routing:
- Router weights need gradients
- Use `index_add_` not `index_copy_` to accumulate gradients

### 4. RoPE Dimension

RoPE is applied per attention head, not full hidden size:
```python
# Correct
rope = RotaryEmbedding(dim=head_dim, ...)  # head_dim = hidden_size // num_heads

# Wrong
rope = RotaryEmbedding(dim=hidden_size, ...)
```

## Advanced Topics

### 1. Efficient MoE Implementation

For production, use expert parallelism:
```python
# Batch all expert computations
expert_inputs = []  # Collect inputs for each expert
expert_outputs = []  # Process in parallel
# Use scatter/gather operations
```

### 2. Expert Capacity

Limit tokens per expert to prevent load imbalance:
```python
capacity = (num_tokens // num_experts) * capacity_factor
```

### 3. Generation Strategies

Implement top-k, top-p, and beam search:
```python
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    # Remove tokens outside top-k
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    # Remove tokens outside top-p
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # Scatter
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')

    return logits
```

### 4. Mixed Precision Training

Use automatic mixed precision for faster training:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input_ids, labels=labels)
    loss = output.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Resources

- **OlMoE Paper**: https://arxiv.org/abs/2409.02060
- **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
- **RoFormer (RoPE)**: https://arxiv.org/abs/2104.09864
- **Switch Transformers**: https://arxiv.org/abs/2101.03961
- **GQA Paper**: https://arxiv.org/abs/2305.13245

## Conclusion

Congratulations on implementing OlMoE! You now understand:

1. Modern transformer architectures
2. Sparse mixture of experts
3. Efficient attention mechanisms (GQA, RoPE)
4. Load balancing in MoE
5. Language model training

Next steps:
- Train on real data
- Experiment with different MoE configurations
- Implement advanced features (flash attention, expert parallelism)
- Scale up the model

Happy learning!
