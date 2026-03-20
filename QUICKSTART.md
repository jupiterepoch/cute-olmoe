# OlMoE Quick Start Guide

Get started with the OlMoE educational implementation in 5 minutes!

## For Students

### Step 1: Setup

```bash
# Navigate to project
cd cute-olmoe

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Understand the Goal

You will implement OlMoE-1B-7B, a Mixture of Experts language model, from scratch. The codebase provides:
- ✓ Complete class structures and interfaces
- ✓ Detailed docstrings explaining each component
- ✓ TODO markers showing what to implement
- ✓ Comprehensive tests to verify your work

### Step 3: Start Implementing

Follow this order:

**Week 1: Foundation**
```bash
# Start with these files:
1. olmoe/config.py       # Model configuration
2. olmoe/utils.py        # RMSNorm and helpers
3. olmoe/embeddings.py   # Token and position embeddings
```

**Week 2: Attention**
```bash
4. olmoe/attention.py    # Multi-head attention
```

**Week 3: MoE (Hardest Part!)**
```bash
5. olmoe/feedforward.py  # Expert networks
6. olmoe/moe.py          # Mixture of Experts
```

**Week 4: Integration**
```bash
7. olmoe/model.py        # Complete model
```

### Step 4: Test Your Work

After implementing each component:

```bash
# Test attention
python tests/test_attention.py

# Test MoE
python tests/test_moe.py

# Test complete model
python tests/test_model.py

# Or run all tests
pytest tests/
```

### Step 5: Learn and Experiment

```bash
# Visualize the architecture
python examples/visualize_architecture.py

# Try training
python examples/train_simple.py
```

### Getting Help

1. **Read the tutorial**: `TUTORIAL.md` has detailed explanations
2. **Check the TODOs**: Each has hints about implementation
3. **Run tests frequently**: They'll guide you to correct solutions
4. **Print shapes**: When debugging, print tensor shapes at each step

### Common First Steps

**Starting with config.py:**
```python
# TODO in __post_init__: Add validation
def __post_init__(self):
    # Your validation here
    assert self.hidden_size % self.num_attention_heads == 0
    # ... more validation
```

**Starting with utils.py RMSNorm:**
```python
def forward(self, x):
    # 1. Compute variance (mean of squares)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    # 2. Normalize
    # 3. Scale
    # Your implementation here
```

## For Instructors

### Quick Setup

1. **Review materials**:
   - `README.md` - Project overview
   - `TUTORIAL.md` - Implementation guide
   - `INSTRUCTOR_GUIDE.md` - Teaching notes with solutions

2. **Customize if needed**:
   - Adjust week schedule in README
   - Modify grading rubric in INSTRUCTOR_GUIDE
   - Add/remove requirements based on course level

3. **Prepare environment**:
   ```bash
   # Ensure students can access:
   - Python 3.8+
   - PyTorch 2.0+
   - 4GB+ RAM (for small models)
   - No GPU required for learning!
   ```

### Teaching Approach

**Option A: Guided (Recommended for Beginners)**
- Lecture on each component before students implement
- Provide partial solutions for Week 1-2
- Office hours focused on debugging

**Option B: Self-Directed (Advanced Students)**
- Give students all materials upfront
- Minimal lectures, focus on Q&A
- Students learn by implementing and reading papers

**Option C: Flipped Classroom**
- Students read papers and tutorial beforehand
- Class time for collaborative implementation
- Live coding sessions

### First Class Outline (90 min)

**Part 1: Introduction (20 min)**
- What is MoE and why it matters
- Overview of OlMoE architecture
- Show `visualize_architecture.py` output

**Part 2: Setup (15 min)**
- Install dependencies
- Run existing tests (they'll fail - that's expected!)
- Navigate codebase structure

**Part 3: First Implementation (45 min)**
- Live code: RMSNorm together
- Students implement: Config validation
- Test and debug together

**Part 4: Homework (10 min)**
- Assign: Complete utils.py and embeddings.py
- Next week: Will implement attention

### Assessment Strategy

**Weekly Checkpoints (40%)**
- Week 1: Config, utils, embeddings (10%)
- Week 2: Attention (10%)
- Week 3: MoE (10%)
- Week 4: Full model (10%)

**Final Project (40%)**
- All tests passing
- Model runs end-to-end
- Simple training loop works

**Written Component (20%)**
- Explain MoE architecture
- Analyze expert usage patterns
- Discuss design trade-offs

### Office Hours Topics

Prepare to help with:

**Week 1**:
- Python environment setup
- PyTorch basics
- Understanding tensor shapes

**Week 2**:
- Attention mechanism intuition
- Broadcasting in PyTorch
- Debugging shape mismatches

**Week 3**:
- MoE routing logic
- Gradient flow issues
- Load balancing intuition

**Week 4**:
- Integration debugging
- End-to-end testing
- Training troubleshooting

## For Self-Learners

### Solo Learning Path

**Time Commitment**: 25-35 hours total

1. **Day 1-2**: Setup + Read README and TUTORIAL
2. **Day 3-5**: Implement Week 1 components
3. **Day 6-9**: Implement attention (hardest conceptually)
4. **Day 10-14**: Implement MoE (hardest technically)
5. **Day 15-17**: Integrate full model
6. **Day 18-20**: Train and experiment

### Self-Check Questions

After Week 1:
- [ ] Can you explain what RMSNorm does and why?
- [ ] How does RoPE encode position information?
- [ ] What is the purpose of load balancing loss?

After Week 2:
- [ ] What's the difference between Q, K, V?
- [ ] How does GQA reduce memory vs standard attention?
- [ ] Why do we need causal masking?

After Week 3:
- [ ] How does the router decide which experts to use?
- [ ] Why use index_add_ instead of direct assignment?
- [ ] What happens if all tokens go to one expert?

After Week 4:
- [ ] How does information flow through the full model?
- [ ] Where do residual connections help?
- [ ] How is MoE loss combined with LM loss?

### Debugging Checklist

When stuck:

1. **Read error message carefully**
   - What tensor has wrong shape?
   - Where did the error occur?

2. **Check shapes at each step**
   ```python
   print(f"Shape: {tensor.shape}")
   ```

3. **Verify gradients**
   ```python
   assert tensor.requires_grad
   assert tensor.grad is not None
   ```

4. **Compare with tutorial**
   - Is your implementation similar?
   - Did you miss a step?

5. **Start simple**
   - Test with batch_size=1, seq_len=2
   - Use tiny model first

6. **Ask for help**
   - GitHub discussions
   - Stack Overflow
   - Course forum (if in class)

## Next Steps After Completion

Congratulations! You've implemented a modern language model. Now:

### Immediate Next Steps

1. **Train on real data**
   - Use WikiText or other datasets
   - Implement proper data loading
   - Track metrics properly

2. **Improve generation**
   - Top-k sampling
   - Top-p (nucleus) sampling
   - Beam search

3. **Analyze experts**
   - What do different experts specialize in?
   - Visualize routing patterns
   - Measure load balance

### Advanced Projects

1. **Optimize performance**
   - Implement Flash Attention
   - Use mixed precision training
   - Profile and optimize bottlenecks

2. **Scale up**
   - Train larger model
   - Distributed training
   - Expert parallelism

3. **Modify architecture**
   - Different number of experts
   - Different routing strategies
   - Hybrid dense/sparse layers

4. **Research directions**
   - Expert specialization
   - Dynamic expert count
   - Hierarchical MoE

### Career Applications

Skills learned:
- ✓ Reading ML papers and implementing them
- ✓ Deep understanding of transformers
- ✓ PyTorch implementation skills
- ✓ Debugging complex deep learning systems
- ✓ Modern ML techniques (RoPE, GQA, MoE)

Ready for:
- ML Engineer roles
- Research positions
- Contributing to open source LLM projects
- Building your own models

## Resources

### Papers (in reading order)
1. Attention is All You Need - Vaswani et al. (2017)
2. RoFormer - Su et al. (2021)
3. Switch Transformers - Fedus et al. (2021)
4. GQA - Ainslie et al. (2023)
5. OlMoE - Muennighoff et al. (2024)

### Code References
- Hugging Face Transformers
- PyTorch LLM examples
- nanoGPT (Karpathy)

### Communities
- r/MachineLearning
- Hugging Face Discord
- PyTorch Forums

## FAQ

**Q: Do I need a GPU?**
A: No! All examples work on CPU with small models. GPU helps for larger experiments but isn't required for learning.

**Q: How long does this take?**
A: 25-35 hours total. Attention and MoE are the most time-consuming parts.

**Q: What if I get stuck?**
A: Read TUTORIAL.md, check test outputs, print shapes, start with smaller examples.

**Q: Can I use this for my thesis/project?**
A: Yes! This is a great starting point. Extend it with your own ideas.

**Q: Is this production-ready?**
A: No, it's educational. For production, use Hugging Face or similar libraries.

**Q: What Python level do I need?**
A: Comfortable with classes, type hints, and basic PyTorch.

## Support

- **Documentation**: README.md, TUTORIAL.md, INSTRUCTOR_GUIDE.md
- **Code**: Comprehensive docstrings and TODO comments
- **Tests**: Clear error messages guide implementation
- **Examples**: Multiple learning approaches

Good luck and enjoy learning about MoE models! 🚀
