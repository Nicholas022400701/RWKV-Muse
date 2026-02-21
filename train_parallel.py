import torch
import json
import argparse
from torch.optim import AdamW
from core.env_hijack import hijack_windows_cuda_env

# 必须在物理层面第一时间劫持环境
hijack_windows_cuda_env() 
from core.architecture_rosa import PianoMuseROSA

def train_copilot(config_file):
    with open(config_file, "r") as f:
        cfg = json.load(f)
        
    print(f"[*] Igniting RWKV-8 ROSA Engine with Config: {cfg}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 实例化天才设计的原生模型
    model = PianoMuseROSA(vocab_size=65536, n_layer=24, n_embd=1024).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    
    # [开发占位] 这里挂载你处理好的 DataLoader
    mock_ctx_len = 50
    mock_comp_len = 50
    total_len = mock_ctx_len + mock_comp_len
    
    model.train()
    for epoch in range(cfg['epochs']):
        # 模拟 batch
        input_ids = torch.randint(0, 65536, (cfg['batch_size'], total_len - 1)).to(device)
        target_ids = torch.randint(0, 65536, (cfg['batch_size'], total_len - 1)).to(device)
        ctx_lengths = torch.tensor([mock_ctx_len] * cfg['batch_size']).to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # 4090 物理级 BFloat16 混合精度，无需废弃的 GradScaler
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 原生 ROSA 前向物理切片
            logits = model(input_ids, ctx_lengths)
            
            valid_targets = []
            for b in range(len(ctx_lengths)):
                c_len = ctx_lengths[b].item()
                target_slice = target_ids[b, c_len-1 : ]
                # 假设 pad token = 0
                non_pad_mask = (input_ids[b, c_len-1 : ] != 0)
                valid_targets.append(target_slice[non_pad_mask])
            valid_targets = torch.cat(valid_targets, dim=0)
            
            loss = torch.nn.functional.cross_entropy(logits, valid_targets)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        
        optimizer.step()
        
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024**3
            print(f"Epoch {epoch+1} | ROSA Loss: {loss.item():.4f} | VRAM: {vram:.2f}GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train_config.json")
    args = parser.parse_args()
    train_copilot(args.config)