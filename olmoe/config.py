"""
OlMoE Model Configuration

This file defines the configuration dataclass for OlMoE-1B-7B.
Students will learn about model hyperparameters and architecture choices.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OlMoEConfig:
    """
    Configuration class for OlMoE model.

    This stores all hyperparameters needed to construct the model architecture.
    Understanding these parameters is crucial for grasping model design.
    """

    # Model architecture
    hidden_size: int = 1024  # Dimension of embeddings and hidden states
    num_hidden_layers: int = 16  # Number of transformer blocks
    num_attention_heads: int = 16  # Number of attention heads
    num_key_value_heads: int = 16  # For Grouped Query Attention (GQA)
    intermediate_size: int = 4096  # Size of FFN hidden layer (typically 4x hidden_size)

    # MoE specific
    num_experts: int = 8  # Total number of experts
    num_experts_per_tok: int = 2  # Top-k: how many experts process each token
    router_aux_loss_coef: float = 0.01  # Weight for load balancing loss

    # Vocabulary and sequence
    vocab_size: int = 50304  # Size of vocabulary (padded to multiple of 64)
    max_position_embeddings: int = 2048  # Maximum sequence length

    # Attention settings
    attention_dropout: float = 0.0  # Dropout in attention
    rope_theta: float = 10000.0  # Base for RoPE frequencies

    # Normalization and activation
    rms_norm_eps: float = 1e-5  # Epsilon for RMSNorm stability
    hidden_act: str = "silu"  # Activation function (silu, gelu, relu)

    # Training
    initializer_range: float = 0.02  # Std dev for weight initialization
    use_cache: bool = True  # Whether to cache key/values for generation

    # Special tokens
    pad_token_id: Optional[int] = None
    bos_token_id: int = 1  # Beginning of sequence token
    eos_token_id: int = 2  # End of sequence token

    def __post_init__(self):
        """
        Validate configuration parameters.

        TODO: Implement validation checks:
        - Ensure hidden_size is divisible by num_attention_heads
        - Ensure num_attention_heads >= num_key_value_heads
        - Ensure num_experts_per_tok <= num_experts
        - Validate that num_key_value_heads divides num_attention_heads evenly (for GQA)
        """
        # TODO: Add validation logic here
        pass

    @property
    def head_dim(self) -> int:
        """
        Calculate dimension of each attention head.

        TODO: Implement this property
        Returns:
            int: hidden_size divided by num_attention_heads
        """
        # TODO: Return head dimension
        pass


# Default configuration for OlMoE-1B-7B
def get_olmoe_1b_7b_config() -> OlMoEConfig:
    """
    Returns the default configuration for OlMoE-1B-7B.

    This is the reference configuration used in the paper.
    """
    return OlMoEConfig(
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=16,
        intermediate_size=4096,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=50304,
        max_position_embeddings=2048,
    )
