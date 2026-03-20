# OlMoE Examples

This directory contains example scripts to help you understand and use the OlMoE implementation.

## Available Examples

### 1. Simple Training (`train_simple.py`)

A minimal training example demonstrating:
- Creating a small OlMoE model
- Basic training loop
- Loss computation (LM loss + MoE auxiliary loss)
- Simple text generation

**Run it:**
```bash
python examples/train_simple.py
```

**What you'll learn:**
- How to instantiate OlMoE models
- Training loop structure
- How auxiliary loss works
- Basic generation

**Note:** This uses random data. For real training, you'll need:
- A tokenizer (e.g., from HuggingFace)
- Real text data (e.g., from HuggingFace datasets)
- Proper data loading and batching

## Future Examples (TODOs for Advanced Students)

### 2. Real Data Training (`train_real.py`)

Implement training on real data:

```python
from transformers import AutoTokenizer
from datasets import load_dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Train...
```

### 3. Text Generation (`generate.py`)

Implement advanced generation strategies:
- Top-k sampling
- Top-p (nucleus) sampling
- Temperature scaling
- Beam search

### 4. Model Analysis (`analyze_experts.py`)

Analyze expert usage:
- Which experts are used most?
- Do different experts specialize?
- Visualize routing patterns
- Analyze load balancing

Example analysis:
```python
import matplotlib.pyplot as plt

def analyze_expert_usage(model, data_loader):
    expert_counts = [0] * config.num_experts

    for batch in data_loader:
        with torch.no_grad():
            # Forward pass, collect routing decisions
            # Count expert usage
            pass

    # Plot
    plt.bar(range(num_experts), expert_counts)
    plt.xlabel("Expert Index")
    plt.ylabel("Number of Tokens Routed")
    plt.title("Expert Usage Distribution")
    plt.show()
```

### 5. Checkpoint Management (`train_with_checkpoints.py`)

Implement:
- Saving model checkpoints
- Resuming from checkpoints
- Validation loop
- Early stopping

```python
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
```

### 6. Distributed Training (`train_distributed.py`)

Scale up with distributed training:
- Data parallelism
- Expert parallelism
- Mixed precision training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DistributedDataParallel(model)

# Train...
```

## Tips for Creating Your Own Examples

1. **Start Small**: Use tiny models (hidden_size=128, 2 layers) for debugging
2. **Monitor Metrics**: Track loss, perplexity, expert usage
3. **Visualize**: Plot attention patterns, expert routing, loss curves
4. **Profile**: Use PyTorch profiler to find bottlenecks
5. **Experiment**: Try different hyperparameters, architectures

## Common Tasks

### Quick Test Forward Pass

```python
from olmoe import create_olmoe_model
import torch

model = create_olmoe_model()
input_ids = torch.randint(0, 1000, (1, 10))
output = model(input_ids)
print(f"Output shape: {output.logits.shape}")
```

### Count Parameters

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expert_params = sum(p.numel() for expert in model.model.layers[0].mlp.moe.experts for p in expert.parameters())

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Expert parameters (per layer): {expert_params:,}")
    print(f"Active parameters (with top-2): ~{total - 6 * expert_params / 8:,}")

count_parameters(model)
```

### Measure Inference Speed

```python
import time

model.eval()
input_ids = torch.randint(0, 1000, (4, 128))

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = model(input_ids)

# Measure
start = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model(input_ids)
end = time.time()

print(f"Average time per batch: {(end - start) / 100 * 1000:.2f}ms")
```

## Questions?

Refer to:
- `../README.md` for project overview
- `../TUTORIAL.md` for detailed implementation guide
- Test files in `../tests/` for usage examples
