#!/bin/bash
echo "========================================================"
echo "[Genius Protocol] RWKV Piano Muse - Ascend 910B Engine"
echo "========================================================"

# 1. 挂载华为 CANN 环境变量 (根据你服务器实际安装位置核对)
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

# 2. 物理抹杀底层模型代码中的 CUDA 硬编码，自动替换为 npu
echo "[*] Purging CUDA hardcodes from architecture..."
sed -i 's/device="cuda"/device="npu:0"/g' core/rwkv_training/rwkv_v8_model.py
sed -i "s/strategy='cuda bf16'/strategy='npu bf16'/g" infer_copilot.py

# 3. 释放 32GB HBM 显存算力，启动全并行训练
echo "[*] Igniting Da Vinci Matrix Cubes..."
python train_npu.py \
    --data_path ./data/processed/processed_dataset.jsonl \
    --pretrained_model ./models/rwkv_base.pth \
    --output_dir ./models \
    --batch_size 8 \
    --max_seq_len 2048 \
    --epochs 10 \
    --learning_rate 1e-4