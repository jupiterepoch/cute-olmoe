"""
Visualize OlMoE Architecture

This script helps students understand the model structure by:
1. Printing model architecture
2. Showing parameter counts
3. Visualizing data flow shapes
4. Analyzing expert routing
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from olmoe import create_olmoe_model, OlMoEConfig


def print_separator(title="", char="=", width=70):
    """Print a separator line with optional title."""
    if title:
        side_len = (width - len(title) - 2) // 2
        print(f"{char * side_len} {title} {char * side_len}")
    else:
        print(char * width)


def analyze_parameters(model):
    """Analyze and display parameter counts."""
    print_separator("Parameter Analysis")

    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Parameters (Billions): {total_params / 1e9:.2f}B")

    # Component breakdown
    print("\nParameter Breakdown by Component:")

    # Embeddings
    embed_params = sum(p.numel() for p in model.model.embed_tokens.parameters())
    print(f"  Embeddings: {embed_params:,} ({embed_params/total_params*100:.1f}%)")

    # LM Head
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    print(f"  LM Head: {lm_head_params:,} ({lm_head_params/total_params*100:.1f}%)")

    # Transformer layers
    layer_params = sum(p.numel() for p in model.model.layers.parameters())
    print(f"  Transformer Layers: {layer_params:,} ({layer_params/total_params*100:.1f}%)")

    # Single layer breakdown
    first_layer = model.model.layers[0]
    attention_params = sum(p.numel() for p in first_layer.self_attn.parameters())
    moe_params = sum(p.numel() for p in first_layer.mlp.parameters())
    norm_params = sum(p.numel() for p in first_layer.input_layernorm.parameters()) + \
                  sum(p.numel() for p in first_layer.post_attention_layernorm.parameters())

    print(f"\n  Per Layer Breakdown:")
    print(f"    Attention: {attention_params:,}")
    print(f"    MoE: {moe_params:,}")
    print(f"    Normalization: {norm_params:,}")

    # Expert parameters
    if hasattr(first_layer.mlp.moe, 'experts'):
        single_expert_params = sum(p.numel() for p in first_layer.mlp.moe.experts[0].parameters())
        router_params = sum(p.numel() for p in first_layer.mlp.moe.router.parameters())
        print(f"\n  MoE Breakdown:")
        print(f"    Single Expert: {single_expert_params:,}")
        print(f"    All Experts (×{model.config.num_experts}): {single_expert_params * model.config.num_experts:,}")
        print(f"    Router: {router_params:,}")

        # Activated parameters (sparse)
        active_expert_params = single_expert_params * model.config.num_experts_per_tok
        total_active = total_params - (single_expert_params * model.config.num_experts) + active_expert_params
        print(f"\n  Active Parameters (top-{model.config.num_experts_per_tok}):")
        print(f"    ~{total_active:,} ({total_active/1e9:.2f}B)")
        print(f"    Sparsity: {(1 - total_active/total_params)*100:.1f}%")


def trace_forward_pass(model, input_ids):
    """Trace shapes through a forward pass."""
    print_separator("Forward Pass Shape Tracing")

    batch_size, seq_len = input_ids.shape
    print(f"\nInput: {input_ids.shape} (batch_size={batch_size}, seq_len={seq_len})")

    # Hook to capture intermediate shapes
    shapes = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                shapes[name] = output[0].shape if len(output) > 0 else None
            else:
                shapes[name] = output.shape if hasattr(output, 'shape') else None
        return hook

    # Register hooks
    hooks = []
    hooks.append(model.model.embed_tokens.register_forward_hook(hook_fn("embeddings")))
    hooks.append(model.model.layers[0].self_attn.register_forward_hook(hook_fn("layer_0_attention")))
    hooks.append(model.model.layers[0].mlp.register_forward_hook(hook_fn("layer_0_moe")))
    hooks.append(model.model.norm.register_forward_hook(hook_fn("final_norm")))
    hooks.append(model.lm_head.register_forward_hook(hook_fn("lm_head")))

    # Forward pass
    with torch.no_grad():
        output = model(input_ids)

    # Print shapes
    print(f"\nEmbeddings output: {shapes.get('embeddings', 'N/A')}")
    print(f"Layer 0 Attention output: {shapes.get('layer_0_attention', 'N/A')}")
    print(f"Layer 0 MoE output: {shapes.get('layer_0_moe', 'N/A')}")
    print(f"Final norm output: {shapes.get('final_norm', 'N/A')}")
    print(f"LM head output (logits): {shapes.get('lm_head', 'N/A')}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return output


def analyze_expert_routing(model, input_ids):
    """Analyze which experts are selected for a given input."""
    print_separator("Expert Routing Analysis")

    # We'll monkey-patch the router to capture routing decisions
    routing_info = []

    original_router_forward = model.model.layers[0].mlp.moe.router.forward

    def capturing_router_forward(hidden_states):
        weights, experts, logits = original_router_forward(hidden_states)
        routing_info.append({
            'weights': weights.detach(),
            'experts': experts.detach(),
            'logits': logits.detach(),
        })
        return weights, experts, logits

    # Temporarily replace
    for layer in model.model.layers:
        layer.mlp.moe.router.forward = capturing_router_forward

    # Forward pass
    with torch.no_grad():
        _ = model(input_ids)

    # Restore original
    for layer in model.model.layers:
        layer.mlp.moe.router.forward = original_router_forward

    # Analyze first layer routing
    if routing_info:
        first_layer_routing = routing_info[0]
        selected_experts = first_layer_routing['experts']
        routing_weights = first_layer_routing['weights']

        print(f"\nFirst layer routing (first 5 tokens):")
        for i in range(min(5, selected_experts.shape[0])):
            experts = selected_experts[i].tolist()
            weights = routing_weights[i].tolist()
            print(f"  Token {i}: Experts {experts} with weights {[f'{w:.3f}' for w in weights]}")

        # Expert usage statistics
        unique, counts = torch.unique(selected_experts, return_counts=True)
        print(f"\nExpert usage across all tokens:")
        for expert_id, count in zip(unique.tolist(), counts.tolist()):
            percentage = count / selected_experts.numel() * 100
            print(f"  Expert {expert_id}: {count} selections ({percentage:.1f}%)")


def print_model_structure(model):
    """Print hierarchical model structure."""
    print_separator("Model Structure")

    print("\nOlMoEForCausalLM")
    print("├── model (OlMoEModel)")
    print("│   ├── embed_tokens (Embedding)")
    print(f"│   │   └── vocab_size={model.config.vocab_size}, hidden_size={model.config.hidden_size}")
    print(f"│   ├── layers (×{model.config.num_hidden_layers})")
    print("│   │   └── OlMoEDecoderLayer")
    print("│   │       ├── input_layernorm (RMSNorm)")
    print("│   │       ├── self_attn (OlMoEAttention)")
    print(f"│   │       │   ├── num_heads={model.config.num_attention_heads}")
    print(f"│   │       │   ├── num_kv_heads={model.config.num_key_value_heads}")
    print(f"│   │       │   ├── head_dim={model.config.hidden_size // model.config.num_attention_heads}")
    print("│   │       │   ├── q_proj (Linear)")
    print("│   │       │   ├── k_proj (Linear)")
    print("│   │       │   ├── v_proj (Linear)")
    print("│   │       │   ├── o_proj (Linear)")
    print("│   │       │   └── rotary_emb (RotaryEmbedding)")
    print("│   │       ├── post_attention_layernorm (RMSNorm)")
    print("│   │       └── mlp (OlMoEMoEBlock)")
    print("│   │           └── moe (OlMoESparseMoE)")
    print(f"│   │               ├── router (num_experts={model.config.num_experts}, top_k={model.config.num_experts_per_tok})")
    print(f"│   │               └── experts (×{model.config.num_experts})")
    print("│   │                   └── OlMoEFeedForward")
    print(f"│   │                       ├── intermediate_size={model.config.intermediate_size}")
    print("│   │                       ├── gate_proj (Linear)")
    print("│   │                       ├── up_proj (Linear)")
    print("│   │                       └── down_proj (Linear)")
    print("│   └── norm (RMSNorm)")
    print("└── lm_head (Linear)")
    print(f"    └── vocab_size={model.config.vocab_size}")


def main():
    print_separator("OlMoE Architecture Visualization", char="=", width=70)

    # Create a small model for visualization
    config = OlMoEConfig(
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=2048,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=5000,
    )

    print(f"\nCreating OlMoE model with configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Experts: {config.num_experts} (top-{config.num_experts_per_tok})")
    print(f"  Vocabulary: {config.vocab_size}")

    model = create_olmoe_model(config)
    print("\n✓ Model created successfully!")

    # 1. Model structure
    print("\n")
    print_model_structure(model)

    # 2. Parameter analysis
    print("\n")
    analyze_parameters(model)

    # 3. Forward pass tracing
    print("\n")
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    output = trace_forward_pass(model, input_ids)

    # 4. Expert routing
    print("\n")
    analyze_expert_routing(model, input_ids)

    print("\n")
    print_separator("", char="=", width=70)
    print("\nVisualization complete!")
    print("Try modifying the config above to see how it affects the architecture.")


if __name__ == "__main__":
    main()
