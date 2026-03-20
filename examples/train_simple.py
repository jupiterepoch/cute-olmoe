"""
Simple training example for OlMoE.

This demonstrates:
1. Creating a small OlMoE model
2. Preparing dummy data
3. Training loop with loss computation
4. Basic text generation

Students can use this as a starting point for their own experiments.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from olmoe import OlMoEConfig, OlMoEForCausalLM


def create_tiny_model():
    """Create a tiny OlMoE model for quick experimentation."""
    config = OlMoEConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=1024,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=1000,
        max_position_embeddings=512,
        router_aux_loss_coef=0.01,
    )
    return OlMoEForCausalLM(config), config


def generate_dummy_data(vocab_size, batch_size, seq_len, num_batches):
    """
    Generate random training data.

    In real training, you would load actual text data and tokenize it.
    """
    data = []
    for _ in range(num_batches):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Labels are the next token (shifted input)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        data.append((input_ids, labels))
    return data


def train_step(model, input_ids, labels, optimizer):
    """
    Single training step.

    TODO: Students can implement this
    Steps:
    1. Zero gradients
    2. Forward pass
    3. Compute total loss (LM loss + aux loss)
    4. Backward pass
    5. Optimizer step
    6. Return losses for logging
    """
    # TODO: Implement training step
    optimizer.zero_grad()

    # Forward pass
    output = model(input_ids, labels=labels)

    # Total loss includes both language modeling loss and MoE auxiliary loss
    lm_loss = output.loss
    aux_loss = output.aux_loss
    total_loss = lm_loss + aux_loss

    # Backward pass
    total_loss.backward()

    # Gradient clipping (optional but recommended)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step
    optimizer.step()

    return {
        'total_loss': total_loss.item(),
        'lm_loss': lm_loss.item(),
        'aux_loss': aux_loss.item(),
    }


def simple_generate(model, config, prompt_ids, max_length=20, temperature=1.0):
    """
    Simple greedy or sampling generation.

    TODO: (Advanced) Students can implement generation
    This is optional and more advanced.
    """
    model.eval()
    generated = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(max_length - prompt_ids.shape[1]):
            # Forward pass
            output = model(generated)
            logits = output.logits

            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature

            # Sample or take argmax
            if temperature > 0:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

    return generated


def main():
    print("=" * 50)
    print("OlMoE Simple Training Example")
    print("=" * 50)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create model
    print("\n1. Creating model...")
    model, config = create_tiny_model()
    print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create optimizer
    print("\n2. Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Generate dummy data
    print("\n3. Generating dummy training data...")
    batch_size = 4
    seq_len = 32
    num_batches = 10
    train_data = generate_dummy_data(config.vocab_size, batch_size, seq_len, num_batches)
    print(f"   Generated {num_batches} batches of {batch_size} sequences")

    # Training loop
    print("\n4. Training...")
    model.train()

    for epoch in range(3):
        epoch_losses = {'total_loss': 0, 'lm_loss': 0, 'aux_loss': 0}

        for batch_idx, (input_ids, labels) in enumerate(train_data):
            losses = train_step(model, input_ids, labels, optimizer)

            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key]

            # Print progress
            if (batch_idx + 1) % 5 == 0:
                print(f"   Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}: "
                      f"Loss={losses['total_loss']:.4f} "
                      f"(LM={losses['lm_loss']:.4f}, Aux={losses['aux_loss']:.4f})")

        # Print epoch summary
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        print(f"\n   Epoch {epoch+1} Summary: "
              f"Avg Loss={avg_losses['total_loss']:.4f} "
              f"(LM={avg_losses['lm_loss']:.4f}, Aux={avg_losses['aux_loss']:.4f})\n")

    # Simple generation example
    print("\n5. Generation example...")
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    print(f"   Prompt tokens: {prompt[0].tolist()}")

    generated = simple_generate(model, config, prompt, max_length=15)
    print(f"   Generated tokens: {generated[0].tolist()}")

    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)

    print("\nNext steps:")
    print("- Replace dummy data with real text data")
    print("- Add validation loop")
    print("- Implement better generation (top-k, top-p sampling)")
    print("- Add model checkpointing")
    print("- Track metrics with tensorboard")
    print("- Experiment with hyperparameters")


if __name__ == "__main__":
    main()
