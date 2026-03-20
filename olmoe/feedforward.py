"""
Feed-Forward Networks for OlMoE

This module implements the expert feed-forward networks used in MoE layers.
Each expert is a simple feed-forward network.
"""

import torch
import torch.nn as nn
from typing import Optional

from .config import OlMoEConfig
from .utils import get_activation_function


class OlMoEFeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) for a single expert.

    Architecture:
        input (hidden_size)
            ↓
        Linear → intermediate_size (gate projection)
        Linear → intermediate_size (up projection)
            ↓
        activation(gate) * up
            ↓
        Linear → hidden_size (down projection)
            ↓
        output

    This is the SwiGLU architecture: FFN_SwiGLU(x) = (Swish(W1*x) ⊙ W3*x) W2
    where ⊙ is element-wise multiplication.
    """

    def __init__(self, config: OlMoEConfig):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # TODO: Initialize three linear layers
        # 1. gate_proj: hidden_size -> intermediate_size (for activation gate)
        # 2. up_proj: hidden_size -> intermediate_size (for value)
        # 3. down_proj: intermediate_size -> hidden_size (output projection)
        # Use bias=False for all projections

        self.gate_proj = None  # TODO
        self.up_proj = None  # TODO
        self.down_proj = None  # TODO

        # TODO: Get activation function from config
        self.act_fn = None  # TODO: use get_activation_function(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FFN.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
               OR (batch_size * seq_len, hidden_size) for MoE

        Returns:
            Output tensor of same shape as input

        TODO: Implement SwiGLU FFN
        Steps:
        1. gate = activation(gate_proj(x))
        2. up = up_proj(x)
        3. hidden = gate * up (element-wise multiplication)
        4. output = down_proj(hidden)
        """
        # TODO: Implement forward pass
        pass


class OlMoESparseMLP(nn.Module):
    """
    Sparse MLP that can optionally apply dropout to experts.

    This is a wrapper around the basic FFN that adds training features.
    """

    def __init__(self, config: OlMoEConfig):
        super().__init__()
        self.ffn = OlMoEFeedForward(config)
        self.dropout = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor

        Returns:
            Output tensor after FFN and optional dropout
        """
        # TODO: Apply FFN and optional dropout
        pass
