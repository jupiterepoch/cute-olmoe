"""
Utility Functions for OlMoE

This module contains helper functions used throughout the model:
- RMSNorm: Root Mean Square Layer Normalization
- Activation functions
- Helper utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Formula: x_norm = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute variance (mean of squares)
        variance = x.pow(2).mean(-1, keepdim=True)
        # Normalize
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def get_activation_function(activation: str) -> nn.Module:
    """
    Get activation function by name.
    """
    if activation in ("silu", "swish"):
        return nn.SiLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "gelu_new":
        return GELUActivation()
    elif activation == "relu":
        return nn.ReLU()
    else:
        raise ValueError(f"Unknown activation function: {activation}")


class GELUActivation(nn.Module):
    """
    GELU activation with tanh approximation.

    Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dimensions of the input.

    Splits last dim in half and returns [-x2, x1].
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.

    Formula: x_rotated = x * cos + rotate_half(x) * sin

    Args:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_kv_heads, seq_len, head_dim)
        cos: (total_seq_len, head_dim)
        sin: (total_seq_len, head_dim)
        position_ids: (batch, seq_len) — indices into cos/sin for the current tokens
    """
    if position_ids is not None:
        # Index cos/sin at the positions of the current tokens
        # position_ids: (batch, seq_len) -> cos: (batch, seq_len, head_dim)
        cos = cos[position_ids]          # (batch, seq_len, head_dim)
        sin = sin[position_ids]
        cos = cos.unsqueeze(1)           # (batch, 1, seq_len, head_dim)
        sin = sin.unsqueeze(1)
    else:
        # Sequential: use all positions in order
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def compute_load_balancing_loss(
    gate_logits: torch.Tensor,
    num_experts: int,
    top_k: int
) -> torch.Tensor:
    """
    Compute load balancing auxiliary loss for MoE.

    Formula: num_experts * sum(f_i * P_i)
    where f_i = fraction of tokens routed to expert i
          P_i = average routing probability to expert i
    """
    # routing probabilities: (num_tokens, num_experts)
    routing_probs = F.softmax(gate_logits, dim=-1)

    # P_i: mean probability for each expert
    P = routing_probs.mean(dim=0)  # (num_experts,)

    # f_i: fraction of tokens that select each expert (in top-k)
    _, selected = torch.topk(gate_logits, top_k, dim=-1)  # (num_tokens, top_k)
    # one-hot encode selections
    num_tokens = gate_logits.shape[0]
    expert_mask = torch.zeros_like(gate_logits)  # (num_tokens, num_experts)
    expert_mask.scatter_(1, selected, 1.0)
    f = expert_mask.mean(dim=0)  # (num_experts,)

    # loss = num_experts * sum(f_i * P_i)
    loss = num_experts * (f * P).sum()
    return loss
