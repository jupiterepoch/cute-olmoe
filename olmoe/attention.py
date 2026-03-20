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
from .utils import apply_rotary_pos_emb


class OlMoEAttention(nn.Module):
    """
    Multi-Head Self-Attention with Grouped Query Attention (GQA).

    GQA is a variant where multiple query heads share the same key/value heads.
    This reduces the size of the KV cache during inference while maintaining quality.

    Example:
    - 16 query heads, 16 KV heads: Standard multi-head attention
    - 16 query heads, 4 KV heads: Grouped query attention (4 queries per KV)
    - 16 query heads, 1 KV head: Multi-query attention
    """

    def __init__(self, config: OlMoEConfig, layer_idx: Optional[int] = None):
        """
        Args:
            config: Model configuration
            layer_idx: Index of this layer (for caching)
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        # Validate configuration
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_heads})"
            )

        # TODO: Initialize query, key, value, and output projection layers
        # q_proj: hidden_size -> num_heads * head_dim
        # k_proj: hidden_size -> num_key_value_heads * head_dim
        # v_proj: hidden_size -> num_key_value_heads * head_dim
        # o_proj: num_heads * head_dim -> hidden_size
        # Use bias=False for all projections

        self.q_proj = None  # TODO
        self.k_proj = None  # TODO
        self.v_proj = None  # TODO
        self.o_proj = None  # TODO

        # TODO: Initialize rotary embeddings
        self.rotary_emb = None  # TODO: Create RotaryEmbedding instance

    def _split_heads(
        self,
        tensor: torch.Tensor,
        num_heads: int,
        attn_head_size: int
    ) -> torch.Tensor:
        """
        Split hidden dimension into attention heads.

        Args:
            tensor: Shape (batch_size, seq_len, hidden_size)
            num_heads: Number of attention heads
            attn_head_size: Size of each head

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, attn_head_size)

        TODO: Implement head splitting
        Steps:
        1. Reshape to (batch_size, seq_len, num_heads, attn_head_size)
        2. Transpose to (batch_size, num_heads, seq_len, attn_head_size)
        """
        # TODO: Split and transpose
        pass

    def _merge_heads(
        self,
        tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge attention heads back into hidden dimension.

        Args:
            tensor: Shape (batch_size, num_heads, seq_len, attn_head_size)

        Returns:
            Tensor of shape (batch_size, seq_len, hidden_size)

        TODO: Implement head merging
        Steps:
        1. Transpose to (batch_size, seq_len, num_heads, attn_head_size)
        2. Reshape to (batch_size, seq_len, hidden_size)
        """
        # TODO: Transpose and merge
        pass

    def _repeat_kv(
        self,
        hidden_states: torch.Tensor,
        n_rep: int
    ) -> torch.Tensor:
        """
        Repeat key/value heads for grouped query attention.

        If num_heads=16 and num_key_value_heads=4, we repeat each KV head 4 times.

        Args:
            hidden_states: Shape (batch, num_key_value_heads, seq_len, head_dim)
            n_rep: Number of repetitions (num_key_value_groups)

        Returns:
            Shape (batch, num_heads, seq_len, head_dim)

        TODO: Implement KV head repetition
        Hint: Use repeat_interleave or expand
        """
        # TODO: Repeat KV heads
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of multi-head attention.

        Args:
            hidden_states: Input of shape (batch_size, seq_len, hidden_size)
            attention_mask: Mask of shape (batch_size, 1, seq_len, seq_len)
            position_ids: Position indices for RoPE
            past_key_value: Cached (key, value) from previous step
            output_attentions: Whether to return attention weights
            use_cache: Whether to return key/value for caching

        Returns:
            Tuple of (output, attention_weights, past_key_value)

        TODO: Implement attention forward pass
        Steps:
        1. Project to queries, keys, values using q_proj, k_proj, v_proj
        2. Split into attention heads
        3. Apply rotary embeddings to queries and keys
        4. Optionally concatenate with cached past_key_value
        5. Repeat key/value heads if using GQA
        6. Compute attention scores: Q @ K^T / sqrt(head_dim)
        7. Apply causal mask
        8. Softmax to get attention weights
        9. Apply attention dropout (if training)
        10. Compute weighted sum: attention_weights @ V
        11. Merge heads
        12. Project output using o_proj
        """
        batch_size, seq_len, _ = hidden_states.shape

        # TODO: Project to Q, K, V
        # query_states = self.q_proj(hidden_states)
        # key_states = self.k_proj(hidden_states)
        # value_states = self.v_proj(hidden_states)

        # TODO: Split into attention heads
        # query_states = self._split_heads(query_states, self.num_heads, self.head_dim)
        # key_states = self._split_heads(key_states, self.num_key_value_heads, self.head_dim)
        # value_states = self._split_heads(value_states, self.num_key_value_heads, self.head_dim)

        # TODO: Apply RoPE
        # cos, sin = self.rotary_emb(value_states, seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # TODO: Handle past_key_value caching
        # if past_key_value is not None:
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # TODO: Repeat KV heads for GQA
        # key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        # value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # TODO: Compute attention scores
        # attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # TODO: Apply attention mask
        # if attention_mask is not None:
        #     attn_weights = attn_weights + attention_mask

        # TODO: Softmax and dropout
        # attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # TODO: Apply attention to values
        # attn_output = torch.matmul(attn_weights, value_states)

        # TODO: Merge heads and project output
        # attn_output = self._merge_heads(attn_output)
        # attn_output = self.o_proj(attn_output)

        # TODO: Return output, attention weights (if requested), and cache
        pass


def _make_causal_mask(
    input_shape: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0
) -> torch.Tensor:
    """
    Create causal (lower triangular) attention mask.

    This ensures each position can only attend to earlier positions.

    Args:
        input_shape: (batch_size, seq_len)
        dtype: Data type for the mask
        device: Device to create mask on
        past_key_values_length: Length of cached past keys

    Returns:
        Causal mask of shape (batch_size, 1, seq_len, seq_len + past_key_values_length)

    TODO: Implement causal mask creation
    Hint: Use torch.triu to create upper triangular mask, then subtract from large negative value
    """
    # TODO: Create causal attention mask
    pass
