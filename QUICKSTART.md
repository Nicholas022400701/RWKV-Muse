# RWKV-Music 快速开始指南 (Quick Start Guide)

本指南将帮助您在10分钟内开始使用RWKV-Music。

This guide will help you get started with RWKV-Music in 10 minutes.

## 步骤 0: 环境准备 (Step 0: Environment Setup)

### Windows系统 (Windows System)

1. **安装Visual Studio Build Tools**
   - 下载: https://visualstudio.microsoft.com/downloads/
   - 选择 "使用C++的桌面开发" 工作负载
   - 确保安装了 Windows SDK

2. **安装CUDA Toolkit**
   - 下载与PyTorch兼容的版本 (推荐 CUDA 11.8 或 12.1)
   - https://developer.nvidia.com/cuda-downloads
   
3. **安装Ninja** (可选但推荐)
   - 下载: https://github.com/ninja-build/ninja/releases
   - 添加到系统PATH

### 安装Python依赖 (Install Python Dependencies)

```bash
# 克隆仓库
git clone https://github.com/Nicholas022400701/RWKV-Music.git
cd RWKV-Music

# 安装依赖
pip install -r requirements.txt

# 验证CUDA环境
python -c "from core.env_hijack import hijack_windows_cuda_env, verify_cuda_setup; hijack_windows_cuda_env(); verify_cuda_setup()"
```

## 步骤 1: 准备数据 (Step 1: Prepare Data)

### 获取MIDI数据集

推荐使用 **MAESTRO** 数据集（古典钢琴）:
- 下载: https://magenta.tensorflow.org/datasets/maestro
- 解压到 `./data/raw_midi/`

或使用您自己的MIDI文件。

### 预处理数据

```bash
python scripts/preprocess_data.py \
    --midi_dir ./data/raw_midi \
    --output_dir ./data/processed \
    --n_context_bars 4 \
    --n_completion_bars 2 \
    --use_hf_dataset
```

**参数说明:**
- `n_context_bars`: 上下文小节数（默认4）
- `n_completion_bars`: 补全小节数（默认2）
- `--use_hf_dataset`: 使用Hugging Face格式（推荐用于大数据集）

## 步骤 2: 获取预训练模型 (Step 2: Get Pretrained Model)

您需要一个预训练的RWKV模型作为起点。

### 选项 A: 下载RWKV官方模型

从 [RWKV官方仓库](https://github.com/BlinkDL/RWKV-LM) 下载预训练权重:
- RWKV-4: https://huggingface.co/BlinkDL/rwkv-4-pile-1b5
- RWKV-5: https://huggingface.co/BlinkDL/rwkv-5-world

推荐使用 1.5B-3B 参数的模型。

### 选项 B: 从零训练（不推荐）

如果您有大量计算资源，也可以从随机初始化开始训练。

## 步骤 3: 微调模型 (Step 3: Fine-tune Model)

```bash
python train_parallel.py \
    --data_path ./data/processed/processed_dataset.jsonl \
    --pretrained_model path/to/rwkv_model.pth \
    --output_dir ./models \
    --batch_size 4 \
    --max_seq_len 2048 \
    --epochs 10 \
    --n_layer 32 \
    --n_embd 2048 \
    --vocab_size 65536
```

**关键参数:**
- `batch_size`: 根据您的GPU调整（4090建议4-8）
- `max_seq_len`: 序列长度（影响显存）
- `n_layer`, `n_embd`: 需要与预训练模型匹配

**显存使用:**
- RTX 4090 (24GB) 推荐设置:
  - 1.5B模型: batch_size=4, max_seq_len=2048 (约18GB)
  - 3B模型: batch_size=2, max_seq_len=1024 (约22GB)

训练过程中会显示：
```
Loss: 2.4532 | LR: 0.000100 | VRAM: 17.23GB / 18.45GB
```

## 步骤 4: 生成音乐 (Step 4: Generate Music)

### 准备上下文MIDI

创建一个包含2-4小节的MIDI文件作为上下文（或使用现有的）。

### 运行推理

```bash
python infer_copilot.py \
    --model_path ./models/best_model.pth \
    --context_midi ./examples/context.mid \
    --output_dir ./outputs \
    --max_new_tokens 512 \
    --temperature 0.85 \
    --top_p 0.90
```

**生成参数调节:**

| 参数 | 效果 | 推荐值 |
|------|------|--------|
| `temperature` | 创造性 (0.1-2.0) | 0.7-0.9 |
| `top_p` | 多样性 (0.0-1.0) | 0.85-0.95 |
| `max_new_tokens` | 生成长度 | 256-1024 |

**温度建议:**
- **0.5-0.7**: 保守、稳定（古典音乐）
- **0.8-0.9**: 平衡（推荐）
- **1.0-1.5**: 激进、实验性（现代音乐）

生成的MIDI文件将保存在 `./outputs/` 目录。

## 步骤 5: 迭代优化 (Step 5: Iterate)

1. **尝试不同采样参数**:
   ```bash
   # 保守风格
   python infer_copilot.py --model_path ./models/best_model.pth \
       --context_midi context.mid --temperature 0.6 --top_p 0.85
   
   # 创新风格
   python infer_copilot.py --model_path ./models/best_model.pth \
       --context_midi context.mid --temperature 1.2 --top_p 0.95
   ```

2. **使用不同上下文长度**:
   - 短上下文（1-2小节）: 更自由的创作
   - 长上下文（4-8小节）: 更连贯的延续

3. **多次采样获取灵感**:
   ```bash
   # 生成5个不同版本
   for i in {1..5}; do
       python infer_copilot.py --model_path ./models/best_model.pth \
           --context_midi context.mid \
           --output_dir ./outputs/variation_$i \
           --temperature 0.9
   done
   ```

## 常见问题排查 (Troubleshooting)

### 1. CUDA编译失败

```
[ERROR] cl.exe: command not found
```

**解决方案:**
- 确保安装了Visual Studio Build Tools with C++ workload
- 运行 `core/env_hijack.py` 应该自动处理
- 手动设置: 运行 `vcvars64.bat` 再启动Python

### 2. 显存不足 (OOM)

```
RuntimeError: CUDA out of memory
```

**解决方案:**
- 减小 `batch_size` (例如从4改为2)
- 减小 `max_seq_len` (例如从2048改为1024)
- 使用更小的模型 (从3B降到1.5B)
- 启用梯度累积 (修改训练脚本)

### 3. 生成质量不佳

**可能原因:**
1. **训练不足**: 增加训练轮数
2. **数据质量**: 确保MIDI数据干净、结构清晰
3. **采样参数**: 调整temperature和top_p
4. **上下文太短**: 提供更长的上下文

## 高级功能 (Advanced Features)

### 自定义数据增强

编辑 `core/tokenization.py` 中的 `TokenizerConfig`:

```python
config = TokenizerConfig(
    num_velocities=32,    # 增加力度层次
    beat_res={(0, 4): 16},  # 提高节奏分辨率
    use_chords=True,      # 启用和弦检测
)
```

### 内存优化技巧

对于更大的模型，使用梯度检查点:

```python
# 在 train_parallel.py 中添加
torch.utils.checkpoint.checkpoint_sequential(...)
```

### 分布式训练

如果有多张GPU:

```bash
# 使用 PyTorch DDP
python -m torch.distributed.launch --nproc_per_node=2 train_parallel.py ...
```

## 下一步 (Next Steps)

1. **阅读完整文档**: 查看 `README.md` 了解技术细节
2. **查看示例代码**: `examples/basic_usage.py`
3. **实验配置**: 修改 `config.py` 中的参数
4. **加入社区**: 在GitHub Issues中分享您的作品

## 性能基准 (Performance Benchmarks)

在 RTX 4090 上的实测性能:

| 任务 | 时间 | 显存 |
|------|------|------|
| 预处理100首MIDI | ~5分钟 | <2GB |
| 训练1个epoch (1000样本) | ~15分钟 | 18GB |
| 生成512 tokens | ~10秒 | 4GB |

## 资源链接 (Resources)

- **RWKV官方**: https://github.com/BlinkDL/RWKV-LM
- **MidiTok文档**: https://miditok.readthedocs.io/
- **MAESTRO数据集**: https://magenta.tensorflow.org/datasets/maestro
- **CUDA工具**: https://developer.nvidia.com/cuda-toolkit

---

祝您创作愉快！🎹🎵

Happy composing! 🎹🎵
