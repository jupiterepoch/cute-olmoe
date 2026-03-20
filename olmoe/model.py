"""
OlMoE Model Implementation

This file combines all components into the complete OlMoE language model.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .config import OlMoEConfig
from .embeddings import OlMoEEmbedding, RotaryEmbedding
from .attention import OlMoEAttention, _make_causal_mask
from .moe import OlMoEMoEBlock
from .utils import RMSNorm


@dataclass
class OlMoEOutput:
    """
    Output of OlMoE model.

    Attributes:
        logits: Language modeling logits (batch, seq_len, vocab_size)
        loss: Optional language modeling loss
        aux_loss: MoE load balancing auxiliary loss
        past_key_values: Cached key/value states for generation
        hidden_states: All layer hidden states (if output_hidden_states=True)
        attentions: All attention weights (if output_attentions=True)
    """
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class OlMoEDecoderLayer(nn.Module):
    """
    Single transformer decoder layer.

    Architecture:
        Input
        ↓
        LayerNorm → Attention → Residual
        ↓
        LayerNorm → MoE → Residual
        ↓
        Output

    This is the "Pre-LN" (pre-normalization) architecture.
    """

    def __init__(self, config: OlMoEConfig, layer_idx: int):
        """
        Args:
            config: Model configuration
            layer_idx: Index of this layer
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        # TODO: Initialize components
        # 1. self_attn: OlMoEAttention
        # 2. mlp: OlMoEMoEBlock
        # 3. input_layernorm: RMSNorm (before attention)
        # 4. post_attention_layernorm: RMSNorm (before MoE)

        self.self_attn = None  # TODO
        self.mlp = None  # TODO
        self.input_layernorm = None  # TODO
        self.post_attention_layernorm = None  # TODO

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple], Optional[torch.Tensor]]:
        """
        Forward pass through decoder layer.

        Args:
            hidden_states: Input (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE
            past_key_value: Cached (key, value)
            output_attentions: Whether to output attention weights
            use_cache: Whether to cache key/values

        Returns:
            Tuple of (hidden_states, aux_loss, past_key_value, attention_weights)

        TODO: Implement decoder layer forward pass
        Steps:
        1. Save residual
        2. Apply input layer norm
        3. Self-attention (returns output, attn_weights, past_kv)
        4. Add residual connection
        5. Save new residual
        6. Apply post-attention layer norm
        7. MoE layer (returns output, aux_loss)
        8. Add residual connection
        9. Return all outputs
        """
        # TODO: Implement decoder layer
        pass


class OlMoEModel(nn.Module):
    """
    OlMoE Transformer model (decoder only).

    This is the core transformer without the language modeling head.
    """

    def __init__(self, config: OlMoEConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.padding_idx = config.pad_token_id

        # TODO: Initialize embeddings
        self.embed_tokens = None  # TODO: OlMoEEmbedding

        # TODO: Initialize transformer layers
        # Create nn.ModuleList of OlMoEDecoderLayer
        self.layers = None  # TODO

        # TODO: Initialize final layer norm
        self.norm = None  # TODO: RMSNorm

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple:
        """
        Forward pass through OlMoE model.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            position_ids: Position IDs
            past_key_values: Cached KV states
            use_cache: Whether to cache KV states
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states

        Returns:
            Tuple of outputs

        TODO: Implement model forward pass
        Steps:
        1. Embed input tokens
        2. Prepare attention mask (causal mask)
        3. Prepare position IDs
        4. Loop through all decoder layers:
            - Forward through each layer
            - Collect hidden states and attentions if requested
            - Accumulate aux loss from MoE layers
        5. Apply final layer norm
        6. Return outputs
        """
        # TODO: Implement forward pass
        pass


class OlMoEForCausalLM(nn.Module):
    """
    OlMoE model with language modeling head.

    This is the complete model for causal language modeling.
    """

    def __init__(self, config: OlMoEConfig):
        super().__init__()
        self.config = config

        # TODO: Initialize base model
        self.model = None  # TODO: OlMoEModel(config)

        # TODO: Initialize language modeling head
        # Linear layer: hidden_size -> vocab_size
        # No bias needed
        self.lm_head = None  # TODO

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> OlMoEOutput:
        """
        Forward pass for causal language modeling.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Cached KV states
            labels: Target token IDs for computing loss
            use_cache: Whether to cache KV
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states

        Returns:
            OlMoEOutput with logits and optional loss

        TODO: Implement language modeling forward pass
        Steps:
        1. Forward through base model
        2. Project hidden states to vocabulary: lm_head(hidden_states)
        3. If labels provided, compute cross-entropy loss
        4. Add aux_loss to total loss (weighted)
        5. Return OlMoEOutput
        """
        # TODO: Implement forward pass
        pass

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting token IDs (batch, seq_len)
            max_length: Maximum total length
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            Generated token IDs

        TODO: (Optional) Implement generation
        This is advanced - students can implement after completing forward pass.

        Steps:
        1. Loop for max_length - input_ids.shape[1] steps:
            a. Forward pass to get logits
            b. Take logits for last position
            c. Apply temperature
            d. Apply top-k filtering
            e. Sample next token
            f. Append to input_ids
        2. Return generated sequence
        """
        # TODO: Implement generation (optional)
        pass


def create_olmoe_model(config: Optional[OlMoEConfig] = None) -> OlMoEForCausalLM:
    """
    Factory function to create OlMoE model.

    Args:
        config: Model configuration (if None, uses default)

    Returns:
        OlMoEForCausalLM model
    """
    if config is None:
        from .config import get_olmoe_1b_7b_config
        config = get_olmoe_1b_7b_config()

    return OlMoEForCausalLM(config)
