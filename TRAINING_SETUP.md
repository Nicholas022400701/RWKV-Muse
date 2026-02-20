# RWKV-Music Training Setup Guide

## Current Status: Hybrid Training Implementation

This repository includes a **hybrid approach** to RWKV training:

### What Works:
1. ✅ RWKV v8 "Heron" model structure from RWKV-LM
2. ✅ Batched GPU-parallel forward pass with gradient flow
3. ✅ Physical logit slicing for memory efficiency  
4. ✅ Proper mask alignment to prevent shape mismatches
5. ✅ BFloat16 mixed precision training support

### Current Limitations:
1. ⚠️  WKV_7.backward raises `NotImplementedError` (line 90-98 in rwkv_v8_model.py)
2. ⚠️  Forward pass wrapped in `torch.no_grad()` in the base model
3. ⚠️  Architecture uses simplified batched operations instead of true WKV time-decay

### Solution Implemented:
The `architecture.py` wrapper implements **batched GPU operations** that:
- ✅ Maintain gradient flow for training
- ✅ Process batches in parallel on GPU (no CPU loops)
- ✅ Use proper tensor operations compatible with autograd
- ⚠️  Use simplified attention (not full WKV state tracking)

For **production-quality RWKV training**, you would need to either:
1. Implement `wkv_cuda_backward` for the WKV_7 operator
2. Use pure PyTorch WKV implementation with proper autograd
3. Integrate the full RWKV-LM training codebase with all gradient operators

## Critical Fixes Implemented

### 1. Fixed Inverted Logic (Issue #1.1)
**BEFORE (Broken):**
```python
try:
    from rwkv.model import RWKV
    self.using_training_model = True  # ❌ Wrong! pip package is inference-only
except ImportError:
    from core.rwkv_training.rwkv_v8_model import RWKV_x070
    self.using_training_model = False  # ❌ Wrong! This IS the training model
```

**AFTER (Fixed):**
```python
try:
    from core.rwkv_training.rwkv_v8_model import RWKV_x070
    self.using_training_model = True  # ✅ Correct! This IS the training model
except ImportError:
    from rwkv.model import RWKV
    self.using_training_model = False  # ✅ Correct! pip package is inference-only
```

### 2. Fixed Autoregressive Mask Alignment (Issue #1.3-1.4)

**Problem:** Masks were computed separately on inputs and targets after autoregressive shift, causing shape mismatches.

**Solution:** Use global `attention_mask` from `collate_fn`:
- Generated once in `collate_fn` based on `input_ids`
- Passed to both `architecture.py` and `train_parallel.py`
- Applied identically to hidden states and targets
- Ensures perfect mathematical alignment (no shape mismatches)

### 3. Fixed Context Truncation (Issue #2.2)

**Problem:** Metadata tokens (Tempo, TimeSignature) prepended to context were cut off during truncation.

**Solution:** Modified `dataset.py` to preserve first 2 tokens:
```python
# Keep metadata + tail of context
metadata_tokens = ctx_tokens[:2]
remaining_ctx = ctx_tokens[2:]
ctx_tokens = metadata_tokens + remaining_ctx[-keep_from_remaining:]
```

### 4. Removed Python For-Loops (Issue #3.1)

**Problem:** Processing each sequence individually with CPU conversion:
```python
for b in range(batch_size):
    seq = input_ids[b].cpu().tolist()  # ❌ Kills GPU performance
    # Process one at a time...
```

**Solution:** Batch-parallel GPU processing:
```python
# Process entire batch in parallel
x = self.model.z['emb.weight'][input_ids]  # [batch_size, seq_len, n_embd]
# Apply operations to full batch...
```

## RWKV v8 "Heron" Architecture

RWKV v8 is the **latest stable architecture** featuring:
- ✅ Advanced WKV7s (state-based) CUDA kernels
- ✅ Improved attention and FFN mechanisms
- ✅ Better long-context handling
- ✅ Enhanced numerical stability

## Prerequisites for Training

### 1. CUDA Toolkit
The RWKV training model uses custom CUDA kernels that are JIT-compiled on first use.

**Required:**
- CUDA Toolkit 11.8+ or 12.x (must match PyTorch CUDA version)
- C++ compiler: GCC 7+ (Linux) or MSVC 2019+ (Windows)

**Installation:**
- Linux: `sudo apt install nvidia-cuda-toolkit build-essential`
- Windows: Install Visual Studio 2019+ with "Desktop development with C++" workload

### 2. PyTorch with CUDA
```bash
# Check your PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Install matching CUDA toolkit
# Example for CUDA 11.8:
conda install -c nvidia cuda-toolkit=11.8
```

### 3. Environment Variables

The training model requires these environment variables (automatically set by `env_hijack.py`):

```bash
export RWKV_JIT_ON="1"              # Enable JIT compilation
export RWKV_HEAD_SIZE_A="64"        # Attention head size  
export RWKV_MY_TESTING="x060"       # RWKV version (v6)
export RWKV_TRAIN_TYPE="states"     # Training mode
export RWKV_CUDA_ON="1"             # Enable CUDA kernels
```

## Training Model Files

Located in `core/rwkv_training/`:

```
core/rwkv_training/
├── README.md           # Documentation
├── model.py            # Training model with backward support
├── __init__.py         # Package init
└── cuda/               # CUDA kernels
    ├── wkv5_cuda.cu    # WKV5 CUDA implementation
    ├── wkv5_op.cpp     # WKV5 operator wrapper
    ├── wkv6_cuda.cu    # WKV6 CUDA implementation  
    ├── wkv6_op.cpp     # WKV6 operator wrapper
    └── ...             # Other CUDA files
```

## Verification

To verify your setup is correct:

```python
import torch
from core.rwkv_training.model import RWKV

# This should work without errors
print("✓ Training model imported successfully")

# Check CUDA compilation (will compile on first run)
# This may take a few minutes
```

## Common Issues

### Issue 1: "No module named 'core.rwkv_training.model'"

**Solution:** Ensure you're running from the project root directory:
```bash
cd /path/to/RWKV-Music
python train_parallel.py ...
```

### Issue 2: CUDA compilation fails

**Solution:** 
1. Check CUDA toolkit matches PyTorch: `nvcc --version` vs `torch.version.cuda`
2. On Windows: Ensure Visual Studio C++ tools are installed
3. Set `TORCH_CUDA_ARCH_LIST` to your GPU's compute capability

### Issue 3: "RuntimeError: Cannot train with inference-only RWKV model"

**Solution:** The code detected you're using the wrong model. Make sure:
1. `core/rwkv_training/` directory exists with model files
2. Import path is correct: `from core.rwkv_training.model import RWKV`

## Testing Training

Quick test to ensure backward pass works:

```python
import torch
from core.architecture import PianoMuseRWKV

# Create dummy model (will fail if using inference-only package)
model = PianoMuseRWKV("path/to/pretrained.pth", strategy='cuda bf16')
model.train()

# Test forward and backward
input_ids = torch.randint(0, 1000, (2, 100)).cuda()
ctx_lengths = torch.tensor([50, 50])

logits = model(input_ids, ctx_lengths=ctx_lengths)
loss = logits.sum()
loss.backward()  # ✅ Should work with training model

print("✓ Backward pass successful!")
```

## References

- **RWKV-LM Repository:** https://github.com/BlinkDL/RWKV-LM
- **RWKV Paper:** https://arxiv.org/abs/2305.13048  
- **Training Examples:** See `RWKV-LM/RWKV-v5/` in the official repo

## Credits

The training model is extracted from the official RWKV-LM repository by BlinkDL.
Licensed under Apache License 2.0.
