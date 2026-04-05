"""
OlMoE Model Implementation

This file combines all components into the complete OlMoE language model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class OlMoEDecoderLayer(nn.Module):
    """
    Single transformer decoder layer (Pre-LN architecture).

    Input -> LayerNorm -> Attention -> Residual -> LayerNorm -> MoE -> Residual -> Output
    """

    def __init__(self, config: OlMoEConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = OlMoEAttention(config, layer_idx=layer_idx)
        self.mlp = OlMoEMoEBlock(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple], Optional[torch.Tensor]]:
        # 1. Pre-LN attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # 2. Pre-LN MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, aux_loss, present_key_value, attn_weights


class OlMoEModel(nn.Module):
    """
    OlMoE Transformer model (decoder only, without LM head).
    """

    def __init__(self, config: OlMoEConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.padding_idx = config.pad_token_id

        self.embed_tokens = OlMoEEmbedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [OlMoEDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        batch_size, seq_len = input_ids.shape
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Build causal mask
        causal_mask = _make_causal_mask(
            (batch_size, seq_len),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            past_key_values_length=past_key_values_length,
        )
        if attention_mask is not None:
            # Combine with padding mask if provided
            # attention_mask: (batch, seq_len), 1=attend, 0=ignore
            padding_mask = (1.0 - attention_mask[:, None, None, :].float()) * torch.finfo(hidden_states.dtype).min
            causal_mask = causal_mask + padding_mask

        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_len + past_key_values_length,
                device=hidden_states.device
            ).unsqueeze(0)

        # Collect outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        present_key_values = [] if use_cache else None
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device)

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_kv = past_key_values[i] if past_key_values is not None else None

            hidden_states, aux_loss, present_kv, attn_weights = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            total_aux_loss = total_aux_loss + aux_loss

            if use_cache:
                present_key_values.append(present_kv)
            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, total_aux_loss, present_key_values, all_hidden_states, all_attentions


class OlMoEForCausalLM(nn.Module):
    """
    OlMoE model with language modeling head.
    """

    def __init__(self, config: OlMoEConfig):
        super().__init__()
        self.config = config

        self.model = OlMoEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        hidden_states, aux_loss, present_key_values, all_hidden_states, all_attentions = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift for causal LM: predict token i+1 from token i
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
            # Add MoE auxiliary loss
            loss = loss + aux_loss

        return OlMoEOutput(
            logits=logits,
            loss=loss,
            aux_loss=aux_loss,
            past_key_values=present_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                output = self(input_ids)
                next_logits = output.logits[:, -1, :]  # (batch, vocab)

                # Apply temperature
                next_logits = next_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    values, _ = torch.topk(next_logits, top_k)
                    threshold = values[:, -1].unsqueeze(-1)
                    next_logits = next_logits.masked_fill(next_logits < threshold, float('-inf'))

                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def create_olmoe_model(config: Optional[OlMoEConfig] = None) -> OlMoEForCausalLM:
    """Factory function to create OlMoE model."""
    if config is None:
        from .config import get_olmoe_1b_7b_config
        config = get_olmoe_1b_7b_config()
    return OlMoEForCausalLM(config)
