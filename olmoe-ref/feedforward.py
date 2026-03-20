"""
Feed-Forward Networks for OlMoE

This module implements the expert feed-forward networks used in MoE layers.
Each expert is a simple feed-forward network using the SwiGLU architecture.
"""

import torch
import torch.nn as nn
from typing import Optional

from .config import OlMoEConfig
from .utils import get_activation_function


class OlMoEFeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) for a single expert.

    SwiGLU architecture: FFN(x) = down_proj(activation(gate_proj(x)) * up_proj(x))
    """

    def __init__(self, config: OlMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = get_activation_function(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: activation(gate) * up, then project down
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class OlMoESparseMLP(nn.Module):
    """
    Sparse MLP that can optionally apply dropout to experts.
    """

    def __init__(self, config: OlMoEConfig):
        super().__init__()
        self.ffn = OlMoEFeedForward(config)
        self.dropout = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ffn(x)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
