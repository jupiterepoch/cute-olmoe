"""
OlMoE: Mixture of Experts Language Model

This package implements OlMoE-1B-7B from scratch for educational purposes.
"""

from .config import OlMoEConfig, get_olmoe_1b_7b_config
from .model import (
    OlMoEForCausalLM,
    OlMoEModel,
    OlMoEDecoderLayer,
    OlMoEOutput,
    create_olmoe_model,
)
from .attention import OlMoEAttention
from .moe import OlMoESparseMoE, OlMoEMoEBlock, OlMoERouter
from .feedforward import OlMoEFeedForward
from .embeddings import OlMoEEmbedding, RotaryEmbedding
from .utils import RMSNorm, get_activation_function

__version__ = "0.1.0"

__all__ = [
    # Config
    "OlMoEConfig",
    "get_olmoe_1b_7b_config",
    # Model
    "OlMoEForCausalLM",
    "OlMoEModel",
    "OlMoEDecoderLayer",
    "OlMoEOutput",
    "create_olmoe_model",
    # Components
    "OlMoEAttention",
    "OlMoESparseMoE",
    "OlMoEMoEBlock",
    "OlMoERouter",
    "OlMoEFeedForward",
    "OlMoEEmbedding",
    "RotaryEmbedding",
    "RMSNorm",
    "get_activation_function",
]
