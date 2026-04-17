"""
Multi-Head Attention for OlMoE

This module implements multi-head self-attention with:
- Grouped Query Attention (GQA) for efficiency
- Rotary Position Embeddings (RoPE)
- Causal masking for autoregressive generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .config import OlMoEConfig
from .embeddings import RotaryEmbedding
from .utils import RMSNorm, apply_rotary_pos_emb


class OlMoEAttention(nn.Module):
    """
    Multi-Head Self-Attention with Grouped Query Attention (GQA).
    """

    def __init__(self, config: OlMoEConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_heads})"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def _split_heads(
        self,
        tensor: torch.Tensor,
        num_heads: int,
        attn_head_size: int
    ) -> torch.Tensor:
        """(batch, seq, hidden) -> (batch, heads, seq, head_dim)"""
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, num_heads, attn_head_size)
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """(batch, heads, seq, head_dim) -> (batch, seq, hidden)"""
        batch_size, num_heads, seq_len, head_dim = tensor.shape
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, seq_len, num_heads * head_dim)

    def _repeat_kv(
        self,
        hidden_states: torch.Tensor,
        n_rep: int
    ) -> torch.Tensor:
        """Repeat KV heads for GQA: (batch, kv_heads, seq, dim) -> (batch, heads, seq, dim)"""
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V and apply QK normalization
        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        # Split into heads
        query_states = self._split_heads(query_states, self.num_heads, self.head_dim)
        key_states = self._split_heads(key_states, self.num_key_value_heads, self.head_dim)
        value_states = self._split_heads(value_states, self.num_key_value_heads, self.head_dim)

        # Determine total kv length for RoPE frequency table
        past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
        kv_seq_len = past_len + seq_len
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Build position_ids for current tokens if not provided
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len, device=hidden_states.device).unsqueeze(0)

        # Apply RoPE only to the current tokens (at their absolute positions)
        # The cached keys were already RoPE-d when originally computed
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Concatenate with past KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV heads for GQA
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Attention scores: (batch, heads, seq_len, kv_seq_len)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Always apply an internal causal mask to prevent future-token leakage
        total_kv_len = key_states.shape[-2]
        internal_causal = torch.triu(
            torch.full((seq_len, total_kv_len), float("-inf"), dtype=attn_weights.dtype, device=attn_weights.device),
            diagonal=past_len + 1,
        )
        attn_weights = attn_weights + internal_causal.unsqueeze(0).unsqueeze(0)

        # Apply optional external mask (e.g. padding mask from model)
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                # Bool mask: True = mask this position; convert to float additive mask
                float_mask = torch.zeros_like(attn_weights)
                attn_weights = attn_weights + float_mask.masked_fill(
                    attention_mask.expand_as(attn_weights), float("-inf")
                )
            else:
                attn_weights = attn_weights + attention_mask.to(attn_weights.dtype)

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Weighted sum over values
        attn_output = torch.matmul(attn_weights, value_states)

        # Merge heads and project
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value


def _make_causal_mask(
    input_shape: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0
) -> torch.Tensor:
    """
    Create causal (lower triangular) attention mask.

    Returns mask of shape (batch_size, 1, seq_len, seq_len + past_key_values_length)
    with 0 for attended positions and -inf for masked positions.
    """
    batch_size, seq_len = input_shape
    total_len = seq_len + past_key_values_length

    # Upper triangular mask (positions that should be masked = True)
    mask = torch.full((seq_len, total_len), float("-inf"), dtype=dtype, device=device)
    mask_cond = torch.arange(total_len, device=device)
    # Position i can attend to positions <= i + past_key_values_length
    row_indices = torch.arange(seq_len, device=device).unsqueeze(1) + past_key_values_length
    mask = torch.where(mask_cond.unsqueeze(0) <= row_indices, torch.zeros_like(mask), mask)

    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, total_len)
