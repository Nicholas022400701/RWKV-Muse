# --edited --complete
# FILE PATH: train_npu.py
"""
train_npu.py
[Genius Edition] NPU 910B Full Parallel Training Engine
"""
import os
import argparse
from pathlib import Path

# 【天才之眼】：第一时间物理劫持环境！必须在所有操作前（特别是 import torch 之前）执行！
# 任何提前导入的 torch 都会被系统默认初始化，导致后续的 CANN 环境变量注入直接触发 driver error=87
from core.env_npu import hijack_npu_env, verify_npu_setup
hijack_npu_env()

import torch
import torch_npu
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from core.architecture import PianoMuseRWKV
from core.dataset import CopilotDataset, collate_fn, load_dataset
from core.tokenization import PianoTokenizer
from train_parallel import compute_loss_with_masking # 复用原有的 Loss 切片逻辑

def train_epoch_npu(model, dataloader, optimizer, scheduler, device, epoch, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    if num_batches == 0:
        print(f"[WARNING] Dataloader is empty for epoch {epoch}! Skipping training.")
        return 0.0
        
    for batch_idx, batch in enumerate(dataloader):
        # 物理显存直接路由至 NPU
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        target_ids = batch['target_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        ctx_lengths = batch['ctx_lengths']
        
        optimizer.zero_grad(set_to_none=True)
        
        # 激活 910B 物理级 BF16 硬件加速，抛弃累赘的 GradScaler
        with torch.autocast(device_type='npu', dtype=torch.bfloat16):
            logits = model(input_ids, ctx_lengths=ctx_lengths, attention_mask=attention_mask, padding_token_id=0)
            loss = compute_loss_with_masking(logits, target_ids, ctx_lengths, attention_mask, padding_token_id=0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step(epoch + batch_idx / num_batches)
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            try:
                vram_used = torch.npu.memory_allocated() / 1024**3
            except Exception:
                vram_used = 0.0
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                  f"Loss: {loss.item():.4f} | LR: {lr:.6f} | "
                  f"NPU HBM: {vram_used:.2f}GB")
            
    return total_loss / num_batches

def main(args):
    print("=" * 70)
    print("NPU 910B Piano Copilot - Graph Fusion Engine")
    print("=" * 70)
    verify_npu_setup()
    
    # 取代写死的 npu:0，顺应容器自动分配的 current_device
    device_id = torch.npu.current_device()
    device = torch.device(f'npu:{device_id}')
    torch.npu.set_device(device)
    
    data_pairs = load_dataset(args.data_path)
    if not data_pairs:
        raise ValueError(f"[ERROR] Loaded dataset from {args.data_path} is totally empty! Check your preprocess output.")
        
    tokenizer = PianoTokenizer(vocab_size=args.vocab_size)
    dataset = CopilotDataset(data_pairs, max_seq_len=args.max_seq_len, tokenizer=tokenizer)
    
    # NPU环境有时进程 fork 较慢，将 num_workers 调到 4 足够喂饱，防止爆内存
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)
    
    # 策略设为 cpu 先加载权重，然后物理转移至 NPU，规避显存尖峰
    model = PianoMuseRWKV(args.pretrained_model, strategy='cpu bf16')
    model = model.to(device)
    
    # 拦截华为特供融合优化器，极大降低显存读写延迟
    try:
        from torch_npu.optim import NpuFusedAdamW
        optimizer = NpuFusedAdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        print("[Genius System] NpuFusedAdamW Active.")
    except ImportError:
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, eta_min=args.learning_rate * 0.1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    for epoch in range(args.epochs):
        avg_loss = train_epoch_npu(model, dataloader, optimizer, scheduler, device, epoch, args.grad_clip)
        
        if avg_loss > 0 and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / "best_model_npu.pth")
            print(f"New Matrix State Saved! Loss: {best_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./models")
    parser.add_argument("--batch_size", type=int, default=8) # 32GB显存，BatchSize 翻倍拉爆
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--vocab_size", type=int, default=65536)
    args = parser.parse_args()
    main(args)
