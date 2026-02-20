"""
PyTorch Dataset for Piano Music Completion.
Efficient data loading with Hugging Face datasets integration.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import json


class CopilotDataset(Dataset):
    """
    PyTorch Dataset for piano music completion task.
    Handles context-completion pairs with variable length support.
    """
    
    def __init__(self, data_pairs: List[Dict[str, List[int]]], max_seq_len: Optional[int] = None, tokenizer=None):
        """
        Initialize dataset from preprocessed context-completion pairs.
        
        Args:
            data_pairs: List of dicts with 'context' and 'completion' keys
            max_seq_len: Maximum sequence length (for truncation), None for no limit
            tokenizer: PianoTokenizer instance for safe structural boundary detection
        """
        self.data = data_pairs
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Returns:
            Dictionary containing:
                - input_ids: Full sequence (context + completion) excluding last token
                - target_ids: Full sequence shifted by 1 (for next-token prediction)
                - ctx_len: Length of context portion (for loss masking)
        """
        item = self.data[idx]
        ctx_tokens = item['context']
        comp_tokens = item['completion']
        
        # Combine context and completion
        full_seq = ctx_tokens + comp_tokens
        
        # Truncate if needed
        if self.max_seq_len is not None and len(full_seq) > self.max_seq_len:
            # Prioritize keeping completion, truncate context if necessary
            if len(comp_tokens) < self.max_seq_len:
                # CRITICAL FIX: Preserve metadata tokens (first 2 tokens: Tempo & TimeSignature)
                # These are prepended in tokenization.py and must not be cut off
                # Truncate from the middle, keeping first 2 tokens + end portion
                new_ctx_len = self.max_seq_len - len(comp_tokens)
                if len(ctx_tokens) > new_ctx_len and new_ctx_len >= 2:
                    # Keep first 2 metadata tokens + tail portion
                    metadata_tokens = ctx_tokens[:2]
                    target_idx = len(ctx_tokens) - (new_ctx_len - 2)
                    
                    # Ensure safe atomic truncation at musical boundaries
                    if self.tokenizer is not None and hasattr(self.tokenizer, 'is_structural_token'):
                        while target_idx < len(ctx_tokens):
                            if self.tokenizer.is_structural_token(ctx_tokens[target_idx]):
                                break
                            target_idx += 1
                        if target_idx == len(ctx_tokens):
                            # No structural token found, fallback to original target
                            target_idx = len(ctx_tokens) - (new_ctx_len - 2)
                    
                    ctx_tokens = metadata_tokens + ctx_tokens[target_idx:]
                else:
                    # Can fit entire context or context is too short
                    ctx_tokens = ctx_tokens[-new_ctx_len:]
                full_seq = ctx_tokens + comp_tokens
            else:
                # Completion itself is extremely long - need to preserve minimum context
                # Keep at least MIN_CONTEXT_RATIO (25%) of max_seq_len for context as an anchor point
                # This ensures the model has some context to condition on
                MIN_CONTEXT_RATIO = 0.25
                keep_ctx = min(len(ctx_tokens), max(2, int(self.max_seq_len * MIN_CONTEXT_RATIO)))
                # CRITICAL FIX: Always preserve first 2 metadata tokens
                if len(ctx_tokens) >= 2 and keep_ctx >= 2:
                    metadata_tokens = ctx_tokens[:2]
                    target_idx = len(ctx_tokens) - (keep_ctx - 2)
                    
                    # Ensure safe atomic truncation at musical boundaries
                    if self.tokenizer is not None and hasattr(self.tokenizer, 'is_structural_token'):
                        while target_idx < len(ctx_tokens):
                            if self.tokenizer.is_structural_token(ctx_tokens[target_idx]):
                                break
                            target_idx += 1
                        if target_idx == len(ctx_tokens):
                            # No structural token found, fallback to original target
                            target_idx = len(ctx_tokens) - (keep_ctx - 2)
                    
                    ctx_tokens = metadata_tokens + ctx_tokens[target_idx:]
                else:
                    # Fallback: keep last tokens if no metadata
                    ctx_tokens = ctx_tokens[-keep_ctx:]
                comp_tokens = comp_tokens[:self.max_seq_len - len(ctx_tokens)]
                full_seq = ctx_tokens + comp_tokens
        
        # CRITICAL FIX: ctx_len must reflect the ACTUAL length after truncation
        # to prevent IndexError when slicing hidden_states in architecture.py
        ctx_len = len(ctx_tokens)
        
        # Create input and target sequences (shifted by 1 for autoregression)
        input_ids = torch.tensor(full_seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(full_seq[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'ctx_len': ctx_len
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.
    Pads sequences to the maximum length in the batch.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched and padded tensors
    """
    # Find max lengths in this batch
    max_input_len = max(item['input_ids'].size(0) for item in batch)
    max_target_len = max(item['target_ids'].size(0) for item in batch)
    
    batch_size = len(batch)
    
    # Initialize padded tensors (pad with 0)
    input_ids = torch.zeros(batch_size, max_input_len, dtype=torch.long)
    target_ids = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    
    # Attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.zeros(batch_size, max_input_len, dtype=torch.bool)
    
    # Context lengths
    ctx_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        input_len = item['input_ids'].size(0)
        target_len = item['target_ids'].size(0)
        
        input_ids[i, :input_len] = item['input_ids']
        target_ids[i, :target_len] = item['target_ids']
        attention_mask[i, :input_len] = True
        ctx_lengths[i] = item['ctx_len']
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'attention_mask': attention_mask,
        'ctx_lengths': ctx_lengths
    }


def save_dataset(data_pairs: List[Dict[str, List[int]]], output_path: str):
    """
    Save processed dataset to disk in JSON Lines format.
    
    Args:
        data_pairs: List of context-completion pairs
        output_path: Path to save file (.jsonl)
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in data_pairs:
            f.write(json.dumps(pair) + '\n')
    print(f"[Dataset] Saved {len(data_pairs)} pairs to {output_path}")


def load_dataset(input_path: str) -> List[Dict[str, List[int]]]:
    """
    Load processed dataset from disk.
    
    Args:
        input_path: Path to .jsonl file
        
    Returns:
        List of context-completion pairs
    """
    data_pairs = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_pairs.append(json.loads(line))
    print(f"[Dataset] Loaded {len(data_pairs)} pairs from {input_path}")
    return data_pairs


def create_huggingface_dataset(data_pairs: List[Dict[str, List[int]]], output_dir: str):
    """
    Create and save a Hugging Face Dataset for efficient loading.
    Uses Apache Arrow format with memory mapping for zero-copy reads.
    
    Args:
        data_pairs: List of context-completion pairs
        output_dir: Directory to save the dataset
    """
    try:
        from datasets import Dataset
    except ImportError:
        print("[ERROR] Hugging Face datasets not installed. Install with: pip install datasets")
        return
    
    def data_generator():
        """Generator function for creating HF dataset."""
        for pair in data_pairs:
            yield {
                'context_ids': pair['context'],
                'completion_ids': pair['completion']
            }
    
    # Create dataset from generator
    print("[Dataset] Creating Hugging Face dataset (this may take a while)...")
    hf_dataset = Dataset.from_generator(
        data_generator,
        cache_dir=None
    )
    
    # Save to disk in Arrow format
    hf_dataset.save_to_disk(output_dir)
    print(f"[Dataset] Saved Hugging Face dataset to {output_dir}")
    print(f"[Dataset] Dataset size: {len(hf_dataset)} examples")


def load_huggingface_dataset(input_dir: str):
    """
    Load Hugging Face dataset from disk.
    Uses memory mapping for efficient loading without RAM overhead.
    
    Args:
        input_dir: Directory containing the saved dataset
        
    Returns:
        Hugging Face Dataset object
    """
    try:
        from datasets import load_from_disk
    except ImportError:
        print("[ERROR] Hugging Face datasets not installed. Install with: pip install datasets")
        return None
    
    print(f"[Dataset] Loading Hugging Face dataset from {input_dir}...")
    dataset = load_from_disk(input_dir)
    print(f"[Dataset] Loaded {len(dataset)} examples")
    return dataset
