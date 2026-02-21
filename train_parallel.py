# --complete --fixed --format=codeblock
# FILE PATH: /content/RWKV-Muse/train_parallel.py
import os
import sys
import torch
import json
import argparse
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader

# 强行注入项目根目录到 Python 路径，彻底粉碎 Colab 沙盒限制
PROJECT_ROOT = "/content/RWKV-Muse"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.env_hijack import hijack_windows_cuda_env
# 在云端 Linux 上这个函数会自动静默跳过，绝不报错
hijack_windows_cuda_env() 
from core.architecture_rosa import PianoMuseROSA
from core.dataset import CopilotDataset, collate_fn, load_dataset
from core.tokenization import PianoTokenizer

def train_copilot(config_file):
    # 兼容本地与云端的配置文件读取
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            cfg = json.load(f)
    else:
        print(f"[警告] 找不到 {config_file}，启动天才云端默认备用阵列...")
        cfg = {"batch_size": 4, "max_seq_len": 2048, "lr": 2e-4, "epochs": 100, "weight_decay": 0.05, "grad_clip": 1.0}
        
    print(f"[*] Igniting RWKV-8 ROSA Real-World Engine with Config: {cfg}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 实例化原生模型
    model = PianoMuseROSA(vocab_size=65536, n_layer=24, n_embd=1024).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    
    # =====================================================================
    # 【云端绝对路径锚定】：物理锁定你指定的数据源，绝不迷路
    # =====================================================================
    dataset_path = "/content/RWKV-Muse/data/processed/processed_dataset.jsonl"
    
    if not os.path.exists(dataset_path):
        print(f"\n[致命错误] 宇宙深处找不到真实的训练张量 {dataset_path}！")
        print("请确保你已经将 MIDI 放入目录并成功运行了 preprocess_data.py！")
        return
        
    tokenizer = PianoTokenizer(vocab_size=65536)
    data_pairs = load_dataset(dataset_path)
    dataset = CopilotDataset(data_pairs, max_seq_len=cfg['max_seq_len'], tokenizer=tokenizer)
    
    # 云端容器推荐 num_workers=2，pin_memory 加速 Host 到 Device 内存泵送
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2 if os.name != 'nt' else 0, 
        pin_memory=True
    )
    
    # 确保物理落盘目录存在
    save_dir = "/content/RWKV-Muse/models"
    os.makedirs(save_dir, exist_ok=True)
    
    model.train()
    print(f"[*] 数据泵已就绪，总计 {len(dataset)} 个真实音乐切片。")
    print(f"[*] 偏执级物理快照防线已开启，目标保存目录: {save_dir}")
    print(f"[*] 提示：即使你中断 Colab 执行，也会触发 [紧急快照] 保全当前权重！\n")
    
    best_loss = float('inf')
    
    try:
        for epoch in range(cfg['epochs']):
            total_loss = 0.0
            
            for step, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                target_ids = batch['target_ids'].to(device, non_blocking=True)
                ctx_lengths = batch['ctx_lengths'].to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # 必须传入 padding_token_id=0，与底层的 FFT Mask 完美同构
                    logits = model(input_ids, ctx_lengths, padding_token_id=0)
                    
                    # 提取 Target 并剔除 Padding
                    valid_targets = []
                    for b in range(len(ctx_lengths)):
                        c_len = ctx_lengths[b].item()
                        target_slice = target_ids[b, c_len-1 : ]
                        non_pad_mask = (input_ids[b, c_len-1 : ] != 0)
                        valid_targets.append(target_slice[non_pad_mask])
                        
                    valid_targets = torch.cat(valid_targets, dim=0)
                    loss = torch.nn.functional.cross_entropy(logits, valid_targets)
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                optimizer.step()
                
                total_loss += loss.item()
                
                # 高频汇报进程，安抚人类的焦虑
                if (step + 1) % max(1, (len(dataloader) // 10)) == 0 or step == 0:
                    vram = torch.cuda.memory_allocated() / 1024**3
                    print(f"  -> Epoch {epoch+1:03d} | Step {step+1:04d}/{len(dataloader)} | Loss: {loss.item():.4f} | VRAM: {vram:.2f}GB")
                    
            avg_loss = total_loss / len(dataloader)
            print(f"==> Epoch {epoch+1:03d} 坍缩完成 | 均值 Loss: {avg_loss:.4f}")
            
            # =====================================================================
            # 【偏执级防线】：绝对物理快照 (Paranoid Checkpoint Defense)
            # =====================================================================
            # 1. 保存心跳快照：无论如何，固化当前轮次，防容器猝死
            latest_path = os.path.join(save_dir, "rosa_muse_latest.pth")
            torch.save(model.state_dict(), latest_path)
            
            # 2. 保存极值快照：只要损失下降，立刻烙印这块拥有最高音乐造诣的硅基大脑
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(save_dir, "rosa_muse_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"[+] ⚡ 突破物理极值！全新最优张量 (Loss: {best_loss:.4f}) 已固化至: {best_path}")
                
            # 3. 保存里程碑快照：每 10 轮留底一次，方便观察流形演变
            if (epoch + 1) % 10 == 0:
                epoch_path = os.path.join(save_dir, f"rosa_muse_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), epoch_path)
                print(f"[*] ⏳ Epoch {epoch+1} 常规快照已保存: {epoch_path}")
                
            print("-" * 60)

        print(f"\n[+] 训练彻底完成！全局最优 Loss 锁定在: {best_loss:.4f}")
        print(f"[+] 你的作曲缪斯已在 {os.path.join(save_dir, 'rosa_muse_best.pth')} 待命。")
        
    except KeyboardInterrupt:
        # =====================================================================
        # 【量子纠缠级防爆死机制】
        # 你就算在 Colab 里点了停止按钮，它也会把显存里的权重强行存盘再死！
        # =====================================================================
        print("\n\n[!!!] 警告：检测到上帝之手 (Ctrl+C / 容器终止) 强行阻断了时间线！")
        print("[*] 防爆死机制已启动，正在紧急抽离计算图中的神经权重...")
        emergency_path = os.path.join(save_dir, "rosa_muse_emergency_snap.pth")
        torch.save(model.state_dict(), emergency_path)
        print(f"[+] 紧急固化完成！这批残缺但极其珍贵的权重已安全迫降至:\n    ==> {emergency_path}")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train_config.json")
    args = parser.parse_args()
    train_copilot(args.config)