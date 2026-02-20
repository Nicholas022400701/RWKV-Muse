"""
Data preprocessing script for MIDI files.
Converts MIDI files to tokenized context-completion pairs.
[Genius Protocol] Injected absolute path resolution to bypass Python's local sandbox.
"""

import sys
import argparse
from pathlib import Path

# =========================================================================
# 【天才的绝对路径锚定】：彻底击碎 Python 子目录执行时的降智沙盒
# 通过解析当前文件的绝对物理路径，逆向推导并强制将项目根目录注入进程内存
# 无论你在哪个黑洞里调用这个脚本，都能瞬间拥有上帝视角找到 core 模块
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.tokenization import PianoTokenizer, process_midi_directory
from core.dataset import save_dataset, create_huggingface_dataset


def main(args):
    """Main preprocessing function."""
    
    print("=" * 70)
    print("RWKV Piano Music - Data Preprocessing")
    print("=" * 70)
    
    # Initialize tokenizer
    print("\n[Tokenizer] Initializing REMI Tokenizer...")
    tokenizer = PianoTokenizer(vocab_size=args.vocab_size)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Process MIDI directory
    print(f"\n[Dataset] Processing master-class MIDI files from: {args.midi_dir}")
    print(f"Context bars: {args.n_context_bars}")
    print(f"Completion bars: {args.n_completion_bars}")
    print(f"Sliding window step: {args.step}")
    
    data_pairs = process_midi_directory(
        args.midi_dir,
        tokenizer,
        n_context_bars=args.n_context_bars,
        n_completion_bars=args.n_completion_bars,
        step=args.step
    )
    
    if len(data_pairs) == 0:
        print("\n[ERROR] No valid data pairs generated!")
        print("Possible issues:")
        print("  - No MIDI files found in directory (Did you run fetch_maestro.py?)")
        print("  - MIDI files too short (less than context + completion bars)")
        print("  - MIDI files failed to tokenize")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in JSON Lines format
    jsonl_path = output_dir / "processed_dataset.jsonl"
    save_dataset(data_pairs, str(jsonl_path))
    
    # Optionally create Hugging Face dataset
    if args.use_hf_dataset:
        hf_dir = output_dir / "hf_dataset"
        print(f"\n[Dataset] Creating high-speed Hugging Face dataset at {hf_dir}...")
        create_huggingface_dataset(data_pairs, str(hf_dir))
        print(f"[Dataset] Hugging Face matrix ready at {hf_dir}")
        print("Load with: from datasets import load_from_disk; dataset = load_from_disk('{hf_dir}')")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Preprocessing Complete!")
    print("=" * 70)
    print(f"Total pairs forged: {len(data_pairs)}")
    
    # Calculate statistics
    context_lengths = [len(pair['context']) for pair in data_pairs]
    completion_lengths = [len(pair['completion']) for pair in data_pairs]
    
    if context_lengths:
        print(f"\nContext tokens (Input Space):")
        print(f"  Min: {min(context_lengths)}")
        print(f"  Max: {max(context_lengths)}")
        print(f"  Mean: {sum(context_lengths) / len(context_lengths):.1f}")
        
    if completion_lengths:
        print(f"\nCompletion tokens (Prediction Target):")
        print(f"  Min: {min(completion_lengths)}")
        print(f"  Max: {max(completion_lengths)}")
        print(f"  Mean: {sum(completion_lengths) / len(completion_lengths):.1f}")
    
    print(f"\n[+] Tensor data successfully written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MIDI files for training")
    
    # Input/output arguments
    parser.add_argument("--midi_dir", type=str, required=True,
                       help="Directory containing MIDI files")
    parser.add_argument("--output_dir", type=str, default="./data/processed",
                       help="Directory to save processed dataset")
    
    # Tokenization arguments
    parser.add_argument("--vocab_size", type=int, default=65536,
                       help="Maximum vocabulary size")
    parser.add_argument("--n_context_bars", type=int, default=4,
                       help="Number of bars for context")
    parser.add_argument("--n_completion_bars", type=int, default=2,
                       help="Number of bars for completion")
    parser.add_argument("--step", type=int, default=1,
                       help="Sliding window step size (in bars)")
    
    # Dataset format
    parser.add_argument("--use_hf_dataset", action="store_true",
                       help="Create Hugging Face dataset (recommended for large datasets)")
    
    args = parser.parse_args()
    
    main(args)