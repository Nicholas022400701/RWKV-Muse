# --complete --fixed --format=codeblock
# FILE PATH: /content/RWKV-Muse/infer_copilot.py
import torch
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = "/content/RWKV-Muse"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.env_hijack import hijack_windows_cuda_env
hijack_windows_cuda_env()

from core.architecture_rosa import PianoMuseROSA
from core.tokenization import PianoTokenizer

@torch.no_grad()
def generate_inspiration(model, tokenizer, context_midi_path, output_midi_path, generate_len=256, temp=0.85, top_p=0.90):
    device = next(model.parameters()).device
    
    print(f"[*] 正在解析 MuseScore 动机文件: {context_midi_path}")
    context_tokens = tokenizer.tokenize_midi(context_midi_path)
    if not context_tokens:
        print("[!] 错误：动机文件解析失败或为空！")
        return
        
    print(f"[*] 成功提取 {len(context_tokens)} 个 Context Tokens. 正在向 CUDA 提交计算图...")
    
    # 建立输入序列张量 [Batch=1, SeqLen]
    current_seq = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    inspirations = []
    print(f"\n[*] 预填充完毕，ROSA 核心开始自回归流形采样 (目标: {generate_len} Tokens)...")
    
    # 自回归流式生成 (Autoregressive Decoding)
    for i in range(generate_len):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 限制物理窗口最大 1024 避免 FFT 在极长序列时产生性能拖拽
            seq_slice = current_seq[:, -1024:]
            logits = model(seq_slice) 
            out = logits[0, -1, :]
            
        # 温度缩放与概率分布转换
        probs = torch.softmax(out / temp, dim=-1)
        
        # 核采样 (Top-p Nucleus Sampling) - 物理级剔除离调噪音，保持绝对的音乐性
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        remove_mask = cum_probs > top_p
        remove_mask[1:] = remove_mask[:-1].clone()
        remove_mask[0] = 0 # 永远保留概率最高的核心音
        
        probs[sorted_idx[remove_mask]] = 0.0
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            # 极小概率下的防死锁回退
            probs[sorted_idx[0]] = 1.0
        
        # 根据概率密度函数进行蒙特卡洛坍缩
        next_token = torch.multinomial(probs, 1).item()
        inspirations.append(next_token)
        
        # 将新生成的 Token 拼接到上下文，自回归递推
        current_seq = torch.cat([current_seq, torch.tensor([[next_token]], device=device)], dim=1)
        
        if (i + 1) % 50 == 0:
            print(f"  -> 已推演 {i + 1}/{generate_len} 个和声/旋律碎片...")
            
    full_sequence = context_tokens + inspirations
    
    print(f"[*] 正在将张量坍缩回听觉空间...")
    os.makedirs(os.path.dirname(output_midi_path), exist_ok=True)
    tokenizer.detokenize(full_sequence, output_midi_path)
    print(f"[+] 灵感已成功封存至: {os.path.abspath(output_midi_path)}")

def main(args):
    print("=" * 70)
    print("🎹 RWKV-8 ROSA 作曲家灵感缪斯引擎 (MuseScore 专属对接端)")
    print("=" * 70)
    
    if not os.path.exists(args.model_path):
        print(f"[致命错误] 找不到模型权重 {args.model_path}！")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[*] 正在将物理矩阵挂载至 {device}...")
    model = PianoMuseROSA(vocab_size=65536, n_layer=24, n_embd=1024).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    print("[+] 矩阵闭环，权重注入成功！")
    
    tokenizer = PianoTokenizer(vocab_size=65536)
    
    out_filename = Path(args.context_midi).stem + f"_rosa_T{args.temperature}.mid"
    output_path = os.path.join(args.output_dir, out_filename)
    
    generate_inspiration(
        model=model,
        tokenizer=tokenizer,
        context_midi_path=args.context_midi,
        output_midi_path=output_path,
        generate_len=args.max_new_tokens,
        temp=args.temperature,
        top_p=args.top_p
    )
    
    print("=" * 70)
    print(f"🚀 创作完成！")
    print(f"请将生成的灵感文件下载到本地，直接拖入 MuseScore 打开：")
    print(f"==> {os.path.abspath(output_path)}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认指向刚刚训练出来的、包含防爆死机制的最佳权重
    parser.add_argument("--model_path", type=str, default="/content/RWKV-Muse/models/rosa_muse_best.pth")
    parser.add_argument("--context_midi", type=str, required=True, help="从 MuseScore 导出的灵感开头 (.mid)")
    parser.add_argument("--output_dir", type=str, default="/content/RWKV-Muse/outputs")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="想要机器帮你续写多少个符号")
    parser.add_argument("--temperature", type=float, default=0.85, help="温度越高越奔放，越低越死板")
    parser.add_argument("--top_p", type=float, default=0.90)
    args = parser.parse_args()
    
    main(args)