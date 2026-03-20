"""
Mixture of Experts (MoE) Layer

Core innovation of OlMoE: sparse conditional computation.
Each token is routed to the top-k experts out of num_experts total.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .config import OlMoEConfig
from .feedforward import OlMoEFeedForward
from .utils import compute_load_balancing_loss


class OlMoERouter(nn.Module):
    """
    Router network that assigns tokens to experts.
    """

    def __init__(self, config: OlMoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            hidden_states: (batch_size * seq_len, hidden_size)

        Returns:
            routing_weights: (tokens, top_k) — normalized weights summing to 1
            selected_experts: (tokens, top_k) — expert indices
            router_logits: (tokens, num_experts) — raw logits
        """
        router_logits = self.gate(hidden_states)  # (tokens, num_experts)

        # Select top-k experts per token
        top_k_logits, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)

        # Normalize routing weights via softmax over the selected logits
        routing_weights = F.softmax(top_k_logits, dim=-1)

        return routing_weights, selected_experts, router_logits


class OlMoESparseMoE(nn.Module):
    """
    Sparse Mixture of Experts layer.
    """

    def __init__(self, config: OlMoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.router_aux_loss_coef = config.router_aux_loss_coef

        self.router = OlMoERouter(config)
        self.experts = nn.ModuleList([OlMoEFeedForward(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            output: (batch_size, seq_len, hidden_size)
            router_logits: (batch_size * seq_len, num_experts)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Flatten to (batch * seq_len, hidden_size)
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Route tokens
        routing_weights, selected_experts, router_logits = self.router(hidden_states_flat)

        # Initialize output accumulator
        final_output = torch.zeros_like(hidden_states_flat)

        # Dispatch tokens to each expert
        for expert_idx in range(self.num_experts):
            # Mask: which (token, slot) pairs are assigned to this expert
            # selected_experts: (num_tokens, top_k)
            expert_mask = (selected_experts == expert_idx)  # (num_tokens, top_k)

            if not expert_mask.any():
                continue

            # Get token indices that use this expert (from any top-k slot)
            token_indices, slot_indices = expert_mask.nonzero(as_tuple=True)

            # Extract those tokens
            expert_input = hidden_states_flat[token_indices]  # (num_assigned, hidden)

            # Process through expert
            expert_out = self.experts[expert_idx](expert_input)  # (num_assigned, hidden)

            # Scale by routing weights
            weights = routing_weights[token_indices, slot_indices].unsqueeze(-1)  # (num_assigned, 1)
            expert_out = expert_out * weights

            # Accumulate
            final_output.index_add_(0, token_indices, expert_out)

        # Reshape back
        output = final_output.view(batch_size, seq_len, hidden_dim)
        return output, router_logits


class OlMoEMoEBlock(nn.Module):
    """
    Complete MoE block with load balancing loss computation.
    """

    def __init__(self, config: OlMoEConfig):
        super().__init__()
        self.moe = OlMoESparseMoE(config)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with auxiliary loss.

        Returns:
            output: (batch_size, seq_len, hidden_size)
            aux_loss: scalar load balancing loss
        """
        output, router_logits = self.moe(hidden_states)

        aux_loss = compute_load_balancing_loss(router_logits, self.num_experts, self.top_k)
        aux_loss = self.router_aux_loss_coef * aux_loss

        return output, aux_loss


# Helper function for debugging
def print_expert_usage(selected_experts: torch.Tensor, num_experts: int):
    """Print how many tokens were routed to each expert."""
    for expert_idx in range(num_experts):
        count = (selected_experts == expert_idx).sum().item()
        print(f"Expert {expert_idx}: {count} token assignments")
