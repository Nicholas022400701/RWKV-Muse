import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from core.dataset import YourDataset  # 请在这里导入你实际的数据集类

from core.architecture import PianoMuseRWKV

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, grad_clip):
    model.train()
    total_loss = 0.0
    
    for step, batch in enumerate(dataloader):
        # [FIXED] 将数据无缝送入 CUDA 设备
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        
        ctx_lengths = batch.get('ctx_lengths', None)
        attention_mask = batch.get('attention_mask', None)
        
        if ctx_lengths is not None:
            ctx_lengths = ctx_lengths.to(device, non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
            
        # [FIXED] 恢复 CUDA 环境下最优雅的梯度释放方式，开启 set_to_none=True 压榨显存
        optimizer.zero_grad(set_to_none=True)
        
        # 前向传播
        logits = model(input_ids, ctx_lengths=ctx_lengths, attention_mask=attention_mask, padding_token_id=0)
        
        # 计算 Loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        # 优化器步进
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        total_loss += loss.item()
        
        if step % 50 == 0:
            print(f"[Epoch {epoch} | Step {step}/{len(dataloader)}] Loss: {loss.item():.4f}")
            
    if len(dataloader) > 0:
        return total_loss / len(dataloader)
    return 0.0

def main(args):
    print("========================================================")
    print("[Genius Protocol] RWKV Piano Muse - NVIDIA CUDA Engine")
    print("========================================================")
    
    # [FIXED] 彻底移除 hijack_npu_env()，拥抱纯正的 NVIDIA CUDA 生态
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")
    if device.type == 'cuda':
        print(f"[*] GPU Info: {torch.cuda.get_device_name(0)}")
        
    # [FIXED] T4 显卡具备强大的 FP16 Tensor Cores，使用 cuda fp16 策略直接起飞
    print(f"[*] Loading RWKV Model: {args.pretrained_model}")
    model = PianoMuseRWKV(args.pretrained_model, strategy='cuda fp16')
    
    # [FIXED] 彻底删除了 model.parameters() 的 .contiguous() 强转补丁
    # cuDNN 底层会自动完美处理内存分配
    model = model.to(device)
    
    # 初始化数据集
    print(f"[*] Loading dataset from: {args.data_path}")
    # dataset = YourDataset(args.data_path)           
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = [] # 防报错占位符，请删除并替换为真实的 dataloader
    
    # [FIXED] 换回 PyTorch 原生的高效 AdamW，抛弃会报错的 NpuFusedAdamW
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = None 
    
    print("[*] Training Pipeline Ignited! 🔥")
    for epoch in range(1, args.epochs + 1):
        if len(dataloader) > 0:
            avg_loss = train_epoch(model, dataloader, optimizer, scheduler, device, epoch, args.grad_clip)
            print(f"==> Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f}\n")
        
        # 保存模型权重
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f"rwkv_muse_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[*] Checkpoint saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 注意：传给 RWKV 官方底层的预训练模型路径无需带 .pth 后缀
    parser.add_argument("--pretrained_model", type=str, default="./models/rwkv_430m")
    parser.add_argument("--data_path", type=str, default="./data/processed/processed_dataset.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=8) # 在 16GB 显存的 T4 上，你可以大胆提高 Batch Size
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    main(args)
