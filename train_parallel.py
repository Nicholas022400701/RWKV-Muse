"""
Single-GPU Training Script with Maximum Parallel Efficiency.
Implements mixed precision training with physical logit slicing for RTX 4090.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import argparse
from pathlib import Path

# Must hijack environment BEFORE importing RWKV
from core.env_hijack import hijack_windows_cuda_env, verify_cuda_setup
hijack_windows_cuda_env()

from core.architecture import PianoMuseRWKV, estimate_model_memory
from core.dataset import CopilotDataset, collate_fn, load_dataset
from core.tokenization import PianoTokenizer


def compute_loss_with_masking(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ctx_lengths: torch.Tensor,
    attention_mask: torch.Tensor,
    padding_token_id: int = 0
) -> torch.Tensor:
    """
    Compute cross-entropy loss with physical slicing.
    
    CRITICAL FIX: Use the global attention_mask from collate_fn to extract targets
    This ensures perfect alignment with the hidden states slicing in architecture.py
    Both use the SAME mask generated once in collate_fn, preventing shape mismatches.
    
    Args:
        logits: Already sliced logits [sum(completion_lengths), vocab_size]
        targets: Full target sequence [batch_size, seq_len]
        ctx_lengths: Context length for each sequence [batch_size]
        attention_mask: Global mask from collate_fn [batch_size, seq_len]
                       1 for real tokens, 0 for padding
        padding_token_id: Token ID used for padding (default: 0)
    
    Returns:
        Scalar loss tensor
    """
    # Extract valid targets (completion portion only)
    # CRITICAL FIX: Use the SAME global attention_mask that was used for hidden states
    # The mask is based on input_ids (before shift), so we apply it to targets (after shift)
    # This ensures perfect mathematical alignment between logits and targets
    valid_targets = []
    
    for b in range(targets.size(0)):
        ctx_len = ctx_lengths[b].item()
        # Targets start from ctx_len-1 (due to shift in autoregression)
        # This matches the slicing in architecture.py: hidden_states[b, ctx_len-1:, :]
        completion_targets = targets[b, ctx_len-1:]
        
        # CRITICAL: Use the SAME mask from attention_mask
        # attention_mask was computed from input_ids in collate_fn
        # We apply it to completion_targets which come from target_ids (shifted input_ids)
        completion_mask = attention_mask[b, ctx_len-1:]
        non_pad_mask = completion_mask.bool()
        
        if non_pad_mask.any():
            valid_targets.append(completion_targets[non_pad_mask])
    
    # Concatenate all valid targets
    if len(valid_targets) == 0:
        # Edge case: no valid targets
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    valid_targets = torch.cat(valid_targets, dim=0)
    
    # CRITICAL FIX: Ensure perfect alignment
    # Both logits and targets should have been extracted with the same mask
    # If there's a mismatch, it indicates a bug in the slicing logic
    if logits.size(0) != valid_targets.size(0):
        error_msg = (
            f"CRITICAL ALIGNMENT ERROR: Shape mismatch detected!\n"
            f"  Logits shape: {logits.size(0)}\n"
            f"  Targets shape: {valid_targets.size(0)}\n"
            f"This indicates a serious bug in the slicing logic that would corrupt training.\n"
            f"Training cannot continue with misaligned data."
        )
        raise RuntimeError(error_msg)
    
    # Compute cross-entropy loss
    # 100% of compute power focused on completion prediction
    loss = nn.functional.cross_entropy(logits, valid_targets)
    
    return loss


def train_epoch(
    model: PianoMuseRWKV,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: RWKV model
        dataloader: Training data loader
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        device: CUDA device
        epoch: Current epoch number
        grad_clip: Gradient clipping threshold
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)  # CRITICAL: Get global mask
        ctx_lengths = batch['ctx_lengths']
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with automatic mixed precision
        # BFloat16 has same dynamic range as FP32 (8-bit exponent)
        # No gradient scaling needed - this is a key advantage of BF16
        with autocast(dtype=torch.bfloat16):
            # Get physically sliced logits (only for completion portion)
            # CRITICAL FIX: Pass attention_mask to ensure logits and targets are aligned
            logits = model(input_ids, ctx_lengths=ctx_lengths, attention_mask=attention_mask, padding_token_id=0)
            
            # Compute loss with synchronized target slicing using the SAME global mask
            # Using padding_token_id=0 as defined in dataset.collate_fn
            loss = compute_loss_with_masking(logits, target_ids, ctx_lengths, attention_mask, padding_token_id=0)
        
        # Backward pass - no scaling needed with BF16
        loss.backward()
        
        # Gradient clipping (prevent explosion from chord jumps)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Update learning rate
        scheduler.step(epoch + batch_idx / num_batches)
        
        # Track loss
        total_loss += loss.item()
        
        # Log progress
        if (batch_idx + 1) % 10 == 0:
            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_reserved = torch.cuda.memory_reserved() / 1024**3
            lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                  f"Loss: {loss.item():.4f} | LR: {lr:.6f} | "
                  f"VRAM: {vram_used:.2f}GB / {vram_reserved:.2f}GB")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main(args):
    """Main training function."""
    
    # Verify CUDA setup
    print("=" * 70)
    print("RWKV Piano Music Completion - Training Script")
    print("=" * 70)
    verify_cuda_setup()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Estimate memory requirements
    print("\n" + "=" * 70)
    print("Memory Estimation")
    print("=" * 70)
    memory_est = estimate_model_memory(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        seq_len=args.max_seq_len,
        precision='bf16'
    )
    print(f"Estimated VRAM usage:")
    print(f"  Parameters: {memory_est['parameters_gb']} GB")
    print(f"  Activations: {memory_est['activations_gb']} GB")
    print(f"  Optimizer: {memory_est['optimizer_gb']} GB")
    print(f"  Gradients: {memory_est['gradients_gb']} GB")
    print(f"  Total: {memory_est['total_gb']} GB")
    print(f"  Model parameters: {memory_est['model_params']:,}")
    
    if memory_est['total_gb'] > 22:
        print("\n[WARNING] Estimated memory usage exceeds 22GB (safe limit for 24GB GPU)")
        print("Consider reducing batch_size, max_seq_len, or model size")
    
    # Load dataset
    print("\n" + "=" * 70)
    print("Loading Dataset")
    print("=" * 70)
    data_pairs = load_dataset(args.data_path)
    
    # Instantiate tokenizer for safe structural boundary detection
    tokenizer = PianoTokenizer(vocab_size=args.vocab_size)
    dataset = CopilotDataset(data_pairs, max_seq_len=args.max_seq_len, tokenizer=tokenizer)
    
    # Create dataloader with efficient multi-worker loading
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Initialize model
    print("\n" + "=" * 70)
    print("Initializing Model")
    print("=" * 70)
    
    if args.pretrained_model:
        print(f"Loading pretrained model from: {args.pretrained_model}")
        model = PianoMuseRWKV(args.pretrained_model, strategy='cuda bf16')
    else:
        print("[ERROR] Pretrained model path required!")
        print("Please provide --pretrained_model argument")
        return
    
    model = model.to(device)
    
    # Initialize optimizer with weight decay (L2 regularization)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing learning rate scheduler with warmup
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.epochs,
        T_mult=1,
        eta_min=args.learning_rate * 0.1
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)
        
        # Train one epoch
        avg_loss = train_epoch(
            model, dataloader, optimizer, scheduler,
            device, epoch, args.grad_clip
        )
        
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or avg_loss < best_loss:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = output_dir / "best_model.pth"
                torch.save(model.state_dict(), best_path)
                print(f"New best model! Loss: {best_loss:.4f}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RWKV Piano Completion Model")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to preprocessed dataset (.jsonl)")
    parser.add_argument("--output_dir", type=str, default="./models",
                       help="Directory to save models")
    
    # Model arguments
    parser.add_argument("--pretrained_model", type=str, required=True,
                       help="Path to pretrained RWKV weights")
    parser.add_argument("--n_layer", type=int, default=32,
                       help="Number of RWKV layers")
    parser.add_argument("--n_embd", type=int, default=2048,
                       help="Embedding dimension")
    parser.add_argument("--vocab_size", type=int, default=65536,
                       help="Vocabulary size")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per GPU")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay (L2 regularization)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                       help="Gradient clipping threshold")
    parser.add_argument("--save_every", type=int, default=1,
                       help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    main(args)
