"""
Embeddings for OlMoE

This module implements:
1. Token embeddings: Convert token IDs to dense vectors
2. Rotary Position Embeddings (RoPE): Encode positional information
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class OlMoEEmbedding(nn.Module):
    """
    Token embedding layer.

    Converts integer token IDs to dense embedding vectors.
    """

    def __init__(self, vocab_size: int, hidden_size: int, padding_idx: Optional[int] = None):
        """
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Dimension of embeddings
            padding_idx: Optional padding token ID
        """
        super().__init__()
        # TODO: Initialize embedding layer
        # Hint: Use nn.Embedding
        self.embedding = None  # TODO

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)

        Returns:
            Embeddings of shape (batch_size, seq_len, hidden_size)

        TODO: Implement forward pass
        """
        # TODO: Return embeddings for input_ids
        pass


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    RoPE encodes positional information by rotating query and key vectors.
    Unlike absolute position embeddings, RoPE naturally captures relative positions.

    Key idea:
    - Precompute rotation matrices based on position
    - Apply rotation to queries and keys in attention
    - The dot product between rotated Q and K naturally encodes relative position

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            dim: Dimension of each attention head
            max_position_embeddings: Maximum sequence length
            base: Base for frequency computation (theta)
            device: Device to place tensors on
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # TODO: Compute inverse frequencies for RoPE
        # Formula: inv_freq[i] = 1.0 / (base ^ (2i / dim)) for i in [0, dim/2)
        # Shape: (dim // 2,)
        # Hint: Use torch.arange and power operation
        inv_freq = None  # TODO
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate cosine and sine values for RoPE.

        Args:
            x: Input tensor (used for dtype and device)
            seq_len: Sequence length (if None, use max_position_embeddings)

        Returns:
            Tuple of (cos, sin) tensors, each of shape (seq_len, dim)

        TODO: Implement RoPE computation
        Steps:
        1. Create position indices: [0, 1, 2, ..., seq_len-1]
        2. Compute frequencies: outer product of positions and inv_freq
           Shape: (seq_len, dim // 2)
        3. Concatenate frequencies with itself to get full dimension
           Shape: (seq_len, dim)
        4. Compute cos and sin of the frequencies
        5. Return (cos, sin)

        Hint: Use torch.einsum("i,j->ij", positions, inv_freq)
        """
        # TODO: Implement RoPE forward pass
        pass

    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ):
        """
        Cache cosine and sine values for efficiency.

        TODO: (Optional) Implement caching for better performance
        This avoids recomputing cos/sin for every forward pass.
        """
        pass
