# RWKV-Music 技术文档 (Technical Documentation)

## 目录 (Table of Contents)

1. [架构设计](#架构设计)
2. [核心算法](#核心算法)
3. [数据流程](#数据流程)
4. [性能优化](#性能优化)
5. [API参考](#api参考)

## 架构设计 (Architecture Design)

### 整体架构 (Overall Architecture)

```
┌─────────────────────────────────────────────────────────────┐
│                    RWKV-Music System                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐     ┌─────────────┐    ┌──────────────┐   │
│  │   MIDI      │────▶│  REMI       │───▶│  Sliding     │   │
│  │   Files     │     │  Tokenizer  │    │  Window      │   │
│  └─────────────┘     └─────────────┘    └──────────────┘   │
│                                                  │            │
│                                                  ▼            │
│                                         ┌──────────────┐     │
│                                         │  [Context]   │     │
│                                         │  [Completion]│     │
│                                         └──────────────┘     │
│                                                  │            │
│                                                  ▼            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            RWKV Model (Parallel Training)            │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────┐  │   │
│  │  │ Block1 │→ │ Block2 │→ │  ...   │→ │ BlockN   │  │   │
│  │  │  WKV   │  │  WKV   │  │  WKV   │  │   WKV    │  │   │
│  │  └────────┘  └────────┘  └────────┘  └──────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│                            ▼                                 │
│                   ┌─────────────────┐                        │
│                   │ Physical Slicing│                        │
│                   │ (Context Remove)│                        │
│                   └─────────────────┘                        │
│                            │                                 │
│                            ▼                                 │
│                   ┌─────────────────┐                        │
│                   │    LM Head      │                        │
│                   │  (Vocab Proj)   │                        │
│                   └─────────────────┘                        │
│                            │                                 │
│                            ▼                                 │
│                   ┌─────────────────┐                        │
│                   │  Cross Entropy  │                        │
│                   │      Loss       │                        │
│                   └─────────────────┘                        │
│                                                               │
│  Inference Mode (RNN):                                       │
│  ┌────────┐    ┌────────┐    ┌────────┐                    │
│  │Token_t │───▶│ State  │───▶│Token   │                    │
│  │        │    │Update  │    │t+1     │                    │
│  └────────┘    └────────┘    └────────┘                    │
│      ▲              │              │                         │
│      └──────────────┴──────────────┘                        │
│           O(1) Memory per Step                              │
└─────────────────────────────────────────────────────────────┘
```

### 模块说明 (Module Description)

#### 1. 数据处理层 (Data Processing Layer)

**core/tokenization.py**
- REMI tokenization
- Bar-based segmentation
- Metadata preservation

**core/dataset.py**
- PyTorch Dataset wrapper
- Efficient data loading
- Variable length handling

#### 2. 模型层 (Model Layer)

**core/architecture.py**
- RWKV model wrapper
- Physical slicing implementation
- Dual-mode support (Parallel/RNN)

#### 3. 训练层 (Training Layer)

**train_parallel.py**
- Mixed precision training (BF16)
- Loss masking
- Gradient clipping
- Learning rate scheduling

#### 4. 推理层 (Inference Layer)

**infer_copilot.py**
- RNN mode generation
- Sampling strategies
- MIDI output

## 核心算法 (Core Algorithms)

### 1. WKV Attention Mechanism

RWKV的核心创新是WKV（Weighted Key-Value）机制：

#### 并行模式 (Parallel Mode - Training)

```python
# Time complexity: O(T)
# Computed in parallel during training

WKV_t = Σ(exp(-(t-j) * w) * K_j * V_j) for j=1 to t

where:
- w: learnable decay weight
- K, V: key and value vectors
- t: current time step
```

**PyTorch实现:**

```python
def wkv_parallel(w, k, v):
    """
    w: [n_embd]
    k: [batch, seq_len, n_embd]
    v: [batch, seq_len, n_embd]
    """
    B, T, C = k.shape
    
    # Create decay matrix
    decay = torch.exp(-w.unsqueeze(0).unsqueeze(0))  # [1, 1, C]
    
    # Compute WKV for all positions in parallel
    wkv = torch.zeros(B, T, C)
    for t in range(T):
        # Exponentially decayed sum
        weights = decay ** torch.arange(t, -1, -1).float()
        wkv[:, t] = (weights.unsqueeze(-1) * k[:, :t+1] * v[:, :t+1]).sum(dim=1)
    
    return wkv
```

#### RNN模式 (RNN Mode - Inference)

```python
# Time complexity: O(1) per step
# Memory complexity: O(1)

State_t = State_{t-1} * exp(-w) + K_t * V_t
Output_t = State_t * R_t
```

**PyTorch实现:**

```python
def wkv_rnn(w, k, v, state=None):
    """
    Single step RNN update
    
    w: [n_embd] - decay weight
    k: [n_embd] - key vector
    v: [n_embd] - value vector
    state: [n_embd] - previous state (None for first step)
    """
    if state is None:
        state = torch.zeros_like(k)
    
    # Exponential decay
    decay = torch.exp(-w)
    
    # Update state: constant memory!
    new_state = state * decay + k * v
    
    return new_state
```

### 2. Physical Slicing Algorithm

训练时的关键优化：只对completion部分计算logits。

```python
def physical_slicing(hidden_states, ctx_lengths):
    """
    Remove context hidden states before LM head projection
    
    hidden_states: [batch_size, seq_len, n_embd]
    ctx_lengths: [batch_size] - length of context for each sequence
    
    Returns: [valid_tokens, n_embd] - only completion hidden states
    """
    batch_size = hidden_states.size(0)
    valid_hiddens = []
    
    for b in range(batch_size):
        ctx_len = ctx_lengths[b]
        # Extract completion portion (account for autoregression shift)
        completion_hidden = hidden_states[b, ctx_len-1:, :]
        valid_hiddens.append(completion_hidden)
    
    # Concatenate: [B, T, D] → [sum(completion_lengths), D]
    return torch.cat(valid_hiddens, dim=0)
```

**内存节省计算:**

```
传统方法:
Hidden: [4, 2048, 2048] = 16,777,216 elements
Logits: [4, 2048, 65536] = 536,870,912 elements → ~2GB FP16

物理切片:
Hidden: [4, 2048, 2048] = 16,777,216 elements
Sliced: [400, 2048] = 819,200 elements (假设completion=100 tokens each)
Logits: [400, 65536] = 26,214,400 elements → ~50MB FP16

节省: 2GB → 50MB (97.5% reduction!)
```

### 3. Sliding Window Segmentation

基于小节的滑动窗口算法：

```python
def sliding_window(token_ids, bar_indices, N=4, M=2, step=1):
    """
    Create [N bars context] → [M bars completion] pairs
    
    Args:
        token_ids: Full token sequence
        bar_indices: Indices of Bar tokens
        N: Context bars
        M: Completion bars
        step: Stride in bars
    
    Returns:
        List of (context, completion) pairs
    """
    pairs = []
    total_bars_needed = N + M
    
    for i in range(0, len(bar_indices) - total_bars_needed + 1, step):
        # Define segment boundaries
        ctx_start = bar_indices[i]
        ctx_end = bar_indices[i + N]
        comp_end = bar_indices[i + N + M] if (i + N + M) < len(bar_indices) else len(token_ids)
        
        context = token_ids[ctx_start:ctx_end]
        completion = token_ids[ctx_end:comp_end]
        
        pairs.append((context, completion))
    
    return pairs
```

**示例:**

```
Input MIDI: 10 bars total
Window: 4 context + 2 completion = 6 bars
Step: 1 bar

Generated pairs:
[Bars 0-3] → [Bars 4-5]
[Bars 1-4] → [Bars 5-6]
[Bars 2-5] → [Bars 6-7]
[Bars 3-6] → [Bars 7-8]
[Bars 4-7] → [Bars 8-9]

Total: 5 training pairs from 1 MIDI file
```

## 数据流程 (Data Flow)

### 训练流程 (Training Pipeline)

```
1. MIDI Files
   ↓
2. REMI Tokenization
   - Convert notes to tokens
   - Add Bar markers
   - Encode tempo/time signature
   ↓
3. Sliding Window Segmentation
   - Extract context-completion pairs
   - Preserve metadata
   ↓
4. Dataset Creation
   - Save to JSONL or HF Dataset
   - Memory-mapped loading
   ↓
5. Batch Collation
   - Pad to max length
   - Create attention masks
   ↓
6. Model Forward (Parallel Mode)
   - WKV parallel computation: O(T)
   - Get hidden states
   ↓
7. Physical Slicing
   - Remove context hidden states
   - Keep only completion portion
   ↓
8. LM Head Projection
   - Project to vocabulary
   - Compute logits
   ↓
9. Loss Computation
   - Cross-entropy on completion tokens
   - Backpropagation
   ↓
10. Optimizer Step
    - AdamW with weight decay
    - Gradient clipping
    - Learning rate scheduling
```

### 推理流程 (Inference Pipeline)

```
1. Context MIDI
   ↓
2. Tokenization
   - Convert to token sequence
   ↓
3. Model Initialization (RNN Mode)
   - Load trained weights
   - Initialize state = None
   ↓
4. Context Prefilling
   for token in context:
       output, state = model.forward(token, state)
   ↓
5. Autoregressive Generation
   for i in range(max_new_tokens):
       - Sample token from output
       - Update: output, state = model.forward(token, state)
       - Repeat with O(1) memory
   ↓
6. Detokenization
   - Convert tokens back to MIDI
   ↓
7. Save Output
   - Write MIDI file
```

## 性能优化 (Performance Optimization)

### 1. 显存优化策略

| 优化技术 | 节省 | 实现难度 |
|---------|------|---------|
| Mixed Precision (BF16) | 50% | 简单 |
| Physical Slicing | 80%+ | 中等 |
| Gradient Checkpointing | 60% | 中等 |
| Gradient Accumulation | 0% (减少batch_size) | 简单 |

### 2. 速度优化策略

| 优化技术 | 加速 | 适用场景 |
|---------|------|---------|
| CUDA WKV Kernel | 10-50x | 训练和推理 |
| JIT Compilation | 2-3x | 首次运行 |
| DataLoader num_workers | 2-4x | 数据加载 |
| Pin Memory | 1.2-1.5x | GPU训练 |

### 3. 训练稳定性技巧

```python
# 1. Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Warmup Learning Rate
def get_lr(step, warmup_steps, max_lr):
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    return max_lr

# 3. Gradient Scaling (for mixed precision)
scaler = torch.cuda.amp.GradScaler()
loss_scaled = scaler.scale(loss)
loss_scaled.backward()
scaler.step(optimizer)
scaler.update()

# 4. Label Smoothing (optional)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

## API参考 (API Reference)

### PianoTokenizer

```python
from core.tokenization import PianoTokenizer

tokenizer = PianoTokenizer(vocab_size=65536)

# Tokenize MIDI
tokens = tokenizer.tokenize_midi("input.mid")

# Detokenize
tokenizer.detokenize(tokens, "output.mid")

# Find bar boundaries
bar_indices = tokenizer.find_bar_indices(tokens)

# Extract metadata
metadata = tokenizer.extract_metadata_tokens(tokens, up_to_index=100)
```

### CopilotDataset

```python
from core.dataset import CopilotDataset, collate_fn
from torch.utils.data import DataLoader

# Create dataset
dataset = CopilotDataset(data_pairs, max_seq_len=2048)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4
)
```

### PianoMuseRWKV

```python
from core.architecture import PianoMuseRWKV

# Initialize model
model = PianoMuseRWKV(
    model_path="rwkv_model.pth",
    strategy="cuda bf16"
)

# Training forward (with physical slicing)
logits = model(input_ids, ctx_lengths=ctx_lengths)

# Inference generation (RNN mode)
generated = model.generate(
    context_tokens=[1, 2, 3, ...],
    max_new_tokens=512,
    temperature=0.85,
    top_p=0.90
)
```

### Memory Estimation

```python
from core.architecture import estimate_model_memory

memory = estimate_model_memory(
    n_layer=32,
    n_embd=2048,
    vocab_size=65536,
    batch_size=4,
    seq_len=2048,
    precision='bf16'
)

print(f"Total VRAM: {memory['total_gb']} GB")
```

## 扩展开发 (Extension Development)

### 添加新的采样策略

```python
def top_k_top_p_sampling(logits, temperature, top_k, top_p):
    """Custom sampling strategy"""
    logits = logits / temperature
    
    # Apply top-k
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    
    # Apply top-p
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    
    return torch.multinomial(F.softmax(logits, dim=-1), 1)
```

### 自定义数据增强

```python
def transpose_midi_tokens(tokens, semitones):
    """Transpose MIDI tokens by semitones"""
    transposed = []
    for token in tokens:
        token_str = tokenizer.id_to_token(token)
        if token_str.startswith("Pitch_"):
            pitch = int(token_str.split("_")[1])
            new_pitch = pitch + semitones
            if 0 <= new_pitch <= 127:
                new_token = tokenizer.token_to_id(f"Pitch_{new_pitch}")
                transposed.append(new_token)
        else:
            transposed.append(token)
    return transposed
```

---

更多技术细节请参考源代码和TODO.md中的研究报告。

For more technical details, refer to the source code and the research report in TODO.md.
