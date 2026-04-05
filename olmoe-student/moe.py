"""
Mixture of Experts (MoE) Layer

This is the core innovation of OlMoE: sparse conditional computation.
Instead of using one feed-forward network, we have multiple "expert" networks,
and each token is routed to the top-k experts.

Key components:
1. Router/Gate: Decides which experts process each token
2. Experts: Multiple feed-forward networks
3. Load balancing: Ensures experts are used evenly
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

    The router is a simple linear layer that outputs logits for each expert.
    Top-k experts are selected based on these logits.
    """

    def __init__(self, config: OlMoEConfig):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        # TODO: Initialize router linear layer
        # Input: hidden_size, Output: num_experts
        # No bias needed
        self.gate = None  # TODO: nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            hidden_states: Shape (batch_size * seq_len, hidden_size)

        Returns:
            Tuple of:
            - routing_weights: Normalized weights for selected experts (batch*seq, top_k)
            - selected_experts: Indices of selected experts (batch*seq, top_k)
            - router_logits: Raw logits for all experts (batch*seq, num_experts)

        TODO: Implement routing logic
        Steps:
        1. Compute router logits: gate(hidden_states)
        2. Select top-k experts per token using torch.topk
        3. Compute routing probabilities: softmax over selected expert logits
        4. Return (routing_weights, selected_experts, router_logits)

        Note: Router logits are returned for computing load balancing loss
        """
        # TODO: Implement routing
        pass


class OlMoESparseMoE(nn.Module):
    """
    Sparse Mixture of Experts layer.

    This layer:
    1. Routes each token to top-k experts using the router
    2. Processes tokens with their assigned experts
    3. Combines expert outputs with routing weights
    4. Computes load balancing loss
    """

    def __init__(self, config: OlMoEConfig):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.router_aux_loss_coef = config.router_aux_loss_coef

        # TODO: Initialize router
        self.router = None  # TODO: OlMoERouter(config)

        # TODO: Initialize experts
        # Create a list (nn.ModuleList) of num_experts feed-forward networks
        self.experts = None  # TODO: nn.ModuleList([OlMoEFeedForward(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.

        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_size)

        Returns:
            Tuple of:
            - output: Shape (batch_size, seq_len, hidden_size)
            - router_logits: For computing auxiliary loss

        TODO: Implement MoE forward pass
        This is the most complex part! Here's the strategy:

        Steps:
        1. Reshape input: (batch, seq, hidden) -> (batch*seq, hidden)
        2. Route tokens to experts
        3. Initialize output tensor of zeros
        4. For each expert:
            a. Find which tokens are assigned to this expert
            b. If any tokens assigned:
                - Extract those tokens
                - Process through expert FFN
                - Scale by routing weights
                - Add back to output tensor
        5. Reshape output: (batch*seq, hidden) -> (batch, seq, hidden)
        6. Return output and router_logits

        Advanced: For efficiency, you can batch process all tokens for each expert
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # TODO: Reshape to (batch * seq_len, hidden_size)
        # hidden_states = hidden_states.view(-1, hidden_dim)

        # TODO: Route tokens
        # routing_weights, selected_experts, router_logits = self.router(hidden_states)

        # TODO: Initialize output
        # final_output = torch.zeros_like(hidden_states)

        # TODO: Process tokens through experts
        # Strategy 1: Loop over experts
        # for expert_idx in range(self.num_experts):
        #     # Find tokens assigned to this expert
        #     expert_mask = (selected_experts == expert_idx)
        #     if expert_mask.any():
        #         # Extract token indices and weights
        #         # Process through expert
        #         # Add to output

        # TODO: Reshape output back to (batch_size, seq_len, hidden_size)

        # TODO: Return output and router_logits
        pass


class OlMoEMoEBlock(nn.Module):
    """
    Complete MoE block with load balancing loss computation.

    This wraps OlMoESparseMoE and adds the auxiliary loss calculation.
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

        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_size)

        Returns:
            Tuple of:
            - output: MoE output (batch_size, seq_len, hidden_size)
            - aux_loss: Load balancing auxiliary loss (scalar)

        TODO: Implement forward with aux loss
        Steps:
        1. Call self.moe(hidden_states) to get output and router_logits
        2. Compute load balancing loss using router_logits
        3. Scale aux_loss by router_aux_loss_coef
        4. Return output and aux_loss
        """
        # TODO: Forward through MoE and compute aux loss
        pass


# Helper function for debugging
def print_expert_usage(selected_experts: torch.Tensor, num_experts: int):
    """
    Print how many tokens were routed to each expert.

    Useful for debugging load balancing.

    Args:
        selected_experts: Tensor of shape (batch*seq, top_k)
        num_experts: Total number of experts
    """
    for expert_idx in range(num_experts):
        count = (selected_experts == expert_idx).sum().item()
        print(f"Expert {expert_idx}: {count} token assignments")
