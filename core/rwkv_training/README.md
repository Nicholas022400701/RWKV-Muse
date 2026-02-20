# RWKV v8 "Heron" Training Model Integration

This directory contains the training-capable RWKV v8 "Heron" model extracted from the official [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) repository.

## RWKV v8 "Heron" Architecture

RWKV v8 "Heron" is the latest stable RWKV architecture featuring:
- ✅ WKV7s CUDA kernels (state-based) with full backward pass support
- ✅ Improved attention mechanism with better long-context handling
- ✅ Enhanced numerical stability and training efficiency
- ✅ Optimized for modern GPUs (RTX 4090, etc.)

**Key Innovation:** v8 introduces improvements to the time-mixing and channel-mixing mechanisms, with better state management and more efficient CUDA kernels.

## Why This is Necessary

The `rwkv` package available via `pip install rwkv` is an **inference-only** library that:
- ❌ Does NOT support gradient computation (no `backward()` pass)
- ❌ Does NOT include CUDA kernels for training
- ❌ Cannot be used for model training or fine-tuning

The official RWKV-LM repository provides the full training implementation with:
- ✅ CUDA kernels with backward pass support  
- ✅ Proper gradient computation through WKV operators
- ✅ Full training capabilities

## Files Included

### Model Files:
- `rwkv_v8_model.py` - RWKV v8 "Heron" model implementation
- `__init__.py` - Package initialization

### CUDA Kernels (from RWKV-LM/RWKV-v7/cuda/):
- `wkv7s.cu` - CUDA kernel for WKV7s (state-based) with forward and backward
- `wkv7s_op.cpp` - C++ operator wrapper

Note: v8 uses wkv7s kernels ("7s" = v7 with state management), which are optimized for RNN-mode inference and training.

## Key Features of RWKV v8

### Architecture Improvements
- **Enhanced Time-Mixing:** Better attention mechanism with improved state propagation
- **Optimized Channel-Mixing:** More efficient FFN with `enn.weight` for token-specific adaptations
- **State Management:** Superior RNN-mode performance with efficient state updates
- **Numerical Stability:** Improved training stability through better weight initialization

### Performance
- Faster inference in RNN mode (O(1) per token)
- More efficient training with better gradient flow
- Better long-context capability than v7
- Optimized for both GPT-mode (parallel) and RNN-mode (sequential)

## Usage

The model supports both RNN-mode (sequential, O(1) memory) and GPT-mode (parallel training):

```python
from core.rwkv_training.rwkv_v8_model import RWKV_x070

# Initialize model
model = RWKV_x070(args)

# RNN-mode (one token at a time)
output, state = model.forward_one(token_id, state)

# GPT-mode (sequence at once, for training)
output, state = model.forward_seq(token_ids, state, full_output=True)
```

## Requirements

To compile the CUDA kernels, you need:
1. **CUDA Toolkit** matching your PyTorch version
2. **C++ compiler** (GCC on Linux, MSVC on Windows)
3. **PyTorch** with CUDA support
4. **Head size must be 64** (hardcoded in CUDA kernel for optimal performance)

The kernels are compiled via JIT (Just-In-Time) when first imported:
```python
from torch.utils.cpp_extension import load
```

## Environment Variables (Required)

The RWKV v8 model requires these environment variables:
- `RWKV_JIT_ON="1"` - Enable JIT compilation
- `RWKV_HEAD_SIZE="64"` - Head size for attention (must be 64)
- `RWKV_MY_TESTING="x070"` - Version identifier (v8 uses x070 kernels)
- `RWKV_CUDA_ON="1"` - Enable CUDA kernels

These are automatically set by `core/env_hijack.py`.

## Model Architecture Details

### Time-Mixing (Attention)
v8 uses an improved time-mixing mechanism with:
- Learnable interpolation parameters (x_r, x_w, x_k, x_v, x_a, x_g)
- Multi-component mixing (w0, w1, w2, a0, a1, a2)
- Enhanced state management for better long-term dependencies

### Channel-Mixing (FFN)
v8 introduces token-specific enhancements:
- `enn.weight` for per-token adaptations
- Improved key-value transformations
- Better gradient flow through the network

## Performance Notes

### Memory Requirements
- State-based processing (wkv7s) for efficient RNN mode
- Supports both parallel (GPT-mode) and sequential (RNN-mode) processing
- Memory efficient for long sequences in RNN mode

### Speed
- **RNN mode:** O(1) time and memory per token (constant regardless of context length)
- **GPT mode:** O(T) parallel processing for training
- Optimized for bfloat16 on modern GPUs

### Training
- Supports both modes: use GPT-mode for parallel training, RNN-mode for inference
- Better gradient stability than previous versions
- Faster convergence with improved architecture

## Migration from v7

Key differences when migrating from v7:
1. **CUDA kernel:** v8 uses `wkv7s` (state-based) instead of `wkv7_clampw`
2. **Model structure:** Enhanced time-mixing and channel-mixing components
3. **State management:** Improved state propagation mechanism
4. **Token-specific FFN:** Added `enn.weight` for per-token adaptations

## License

The RWKV model code is from the official RWKV-LM repository:
https://github.com/BlinkDL/RWKV-LM

Licensed under Apache License 2.0.

## References

- **RWKV v8 Demo:** https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v8_rc00_demo.py
- **RWKV v8/ROSA:** https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8
- **RWKV Paper:** https://arxiv.org/abs/2305.13048
- **RWKV-LM Repository:** https://github.com/BlinkDL/RWKV-LM
