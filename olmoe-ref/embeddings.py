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
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Precomputes rotation frequencies and applies them to Q and K in attention.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # inv_freq[i] = 1 / (base ^ (2i / dim)), shape: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate cosine and sine values for RoPE.

        Returns:
            Tuple of (cos, sin), each of shape (seq_len, dim)
        """
        if seq_len is None:
            seq_len = self.max_position_embeddings

        # Position indices: (seq_len,)
        positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)

        # Outer product: (seq_len, dim // 2)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)

        # Concatenate to full dim: (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin

    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ):
        pass
