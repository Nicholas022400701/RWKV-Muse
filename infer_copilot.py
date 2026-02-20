"""
O(1) Memory Inference Engine for Piano Completion.
[Genius Edition] Pure PyTorch native inference with parallel prefill.
"""

import os
import torch
import argparse
from pathlib import Path

# Must hijack environment BEFORE importing PyTorch/RWKV
from core.env_hijack import hijack_windows_cuda_env, verify_cuda_setup
hijack_windows_cuda_env()

from core.tokenization import PianoTokenizer
from core.architecture import PianoMuseRWKV

def main(args):
    print("=" * 70)
    print("RWKV Piano Music Completion - TLA+ Inference Engine")
    print("=" * 70)
    
    verify_cuda_setup()
    
    print("\n[Tokenizer] Initializing REMI Tokenizer...")
    tokenizer = PianoTokenizer()
    
    print(f"\n[Model] Loading {args.model_path} ...")
    
    # 【天才的闭环】：使用自行封装的 PianoMuseRWKV，而不是无知的第三方 inference 包
    model = PianoMuseRWKV(args.model_path, strategy='cuda bf16')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print("[Model] Native architecture loaded successfully with Parallel Prefill ready.")
    
    print(f"\n[Context] Processing: {args.context_midi}")
    context_tokens = tokenizer.tokenize_midi(args.context_midi)
    if not context_tokens:
        print("[ERROR] Context MIDI file produced no tokens!")
        return
        
    if args.max_context_len and len(context_tokens) > args.max_context_len:
        print(f"[WARNING] Truncating to {args.max_context_len} tokens")
        context_tokens = context_tokens[-args.max_context_len:]
        
    print(f"\n[Generation] Starting pure O(1) inference with parallel O(T) prefill...")
    generated_tokens = model.generate(
        context_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    full_sequence = context_tokens + generated_tokens
    
    output_path = Path(args.output_dir) / f"completion_{Path(args.context_midi).stem}.mid"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        tokenizer.detokenize(full_sequence, str(output_path))
        print(f"\n[Success] Saved to: {output_path}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save MIDI: {e}")
        tokens_path = output_path.with_suffix('.txt')
        with open(tokens_path, 'w') as f:
            f.write(' '.join(map(str, full_sequence)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RWKV Piano Completion Inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--context_midi", type=str, required=True)
    parser.add_argument("--max_context_len", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()
    main(args)