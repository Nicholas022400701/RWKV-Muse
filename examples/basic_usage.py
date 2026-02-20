"""
Example script demonstrating basic usage of RWKV-Music.
"""

import sys
sys.path.insert(0, '..')

from core.env_hijack import hijack_windows_cuda_env, verify_cuda_setup

# IMPORTANT: Must hijack environment before importing torch/rwkv
hijack_windows_cuda_env()

from core.tokenization import PianoTokenizer, create_context_completion_pairs
from core.dataset import CopilotDataset
from core.architecture import estimate_model_memory


def example_tokenization():
    """Example: Tokenize a MIDI file and create training pairs."""
    print("=" * 70)
    print("Example 1: MIDI Tokenization and Data Preparation")
    print("=" * 70)
    
    # Initialize tokenizer
    tokenizer = PianoTokenizer()
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Tokenize a MIDI file (replace with your file)
    midi_path = "path/to/your/file.mid"
    print(f"\nTokenizing: {midi_path}")
    
    try:
        tokens = tokenizer.tokenize_midi(midi_path)
        print(f"Generated {len(tokens)} tokens")
        
        # Create context-completion pairs
        pairs = create_context_completion_pairs(
            tokens, tokenizer,
            n_context_bars=4,
            n_completion_bars=2,
            step=1
        )
        
        print(f"Created {len(pairs)} training pairs")
        
        if len(pairs) > 0:
            print(f"\nExample pair:")
            print(f"  Context length: {len(pairs[0]['context'])} tokens")
            print(f"  Completion length: {len(pairs[0]['completion'])} tokens")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please provide a valid MIDI file path")


def example_memory_estimation():
    """Example: Estimate VRAM requirements for different model sizes."""
    print("\n" + "=" * 70)
    print("Example 2: Memory Estimation for RTX 4090")
    print("=" * 70)
    
    configs = [
        ("Small (430M)", 24, 1024, 65536),
        ("Base (1.5B)", 32, 2048, 65536),
        ("Large (3B)", 48, 2560, 65536),
    ]
    
    for name, n_layer, n_embd, vocab_size in configs:
        print(f"\n{name}:")
        print(f"  Layers: {n_layer}, Embedding: {n_embd}")
        
        mem = estimate_model_memory(
            n_layer=n_layer,
            n_embd=n_embd,
            vocab_size=vocab_size,
            batch_size=4,
            seq_len=2048,
            precision='bf16'
        )
        
        print(f"  Parameters: {mem['model_params']:,}")
        print(f"  Total VRAM: {mem['total_gb']} GB")
        print(f"    - Parameters: {mem['parameters_gb']} GB")
        print(f"    - Activations: {mem['activations_gb']} GB")
        print(f"    - Optimizer: {mem['optimizer_gb']} GB")
        print(f"    - Gradients: {mem['gradients_gb']} GB")
        
        if mem['total_gb'] > 22:
            print(f"  ⚠️  Warning: Exceeds safe limit for 24GB GPU")
        else:
            print(f"  ✓ Safe for RTX 4090")


def example_dataset():
    """Example: Create PyTorch dataset."""
    print("\n" + "=" * 70)
    print("Example 3: PyTorch Dataset Creation")
    print("=" * 70)
    
    # Mock data for demonstration
    mock_pairs = [
        {
            'context': [1, 2, 3, 4, 5] * 100,  # 500 tokens
            'completion': [6, 7, 8, 9] * 50     # 200 tokens
        }
        for _ in range(10)
    ]
    
    # Create dataset
    dataset = CopilotDataset(mock_pairs, max_seq_len=2048)
    print(f"Dataset size: {len(dataset)} examples")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample shape:")
    print(f"  Input IDs: {sample['input_ids'].shape}")
    print(f"  Target IDs: {sample['target_ids'].shape}")
    print(f"  Context length: {sample['ctx_len']}")


def main():
    """Run all examples."""
    print("RWKV-Music Usage Examples")
    print("=" * 70)
    
    # Verify CUDA setup
    print("\nVerifying CUDA setup...")
    verify_cuda_setup()
    
    # Run examples
    example_memory_estimation()
    example_dataset()
    
    # Uncomment to run tokenization example (requires MIDI file)
    # example_tokenization()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
