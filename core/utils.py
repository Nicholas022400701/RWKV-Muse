"""
Utility functions for RWKV-Music project.
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    loss: float,
    save_path: str,
    metadata: Optional[Dict] = None
):
    """
    Save training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
        metadata: Additional metadata to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    # Create parent directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"[Checkpoint] Saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler = None,
    device: str = 'cuda'
) -> Dict:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load tensors to
    
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"[Checkpoint] Loaded from {checkpoint_path}")
    print(f"[Checkpoint] Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A')}")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0.0),
        'metadata': checkpoint.get('metadata', {})
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_vram_usage() -> Dict[str, float]:
    """
    Get current VRAM usage in GB.
    
    Returns:
        Dictionary with VRAM statistics
    """
    if not torch.cuda.is_available():
        return {'allocated': 0.0, 'reserved': 0.0, 'total': 0.0}
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return {
        'allocated': round(allocated, 2),
        'reserved': round(reserved, 2),
        'total': round(total, 2),
        'free': round(total - allocated, 2)
    }


def save_training_config(config: Dict, save_path: str):
    """
    Save training configuration to JSON.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save JSON file
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"[Config] Saved to {save_path}")


def load_training_config(config_path: str) -> Dict:
    """
    Load training configuration from JSON.
    
    Args:
        config_path: Path to JSON file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"[Config] Loaded from {config_path}")
    return config


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_device_info() -> Dict[str, str]:
    """
    Get CUDA device information.
    
    Returns:
        Dictionary with device information
    """
    if not torch.cuda.is_available():
        return {'available': False, 'device': 'cpu'}
    
    device_props = torch.cuda.get_device_properties(0)
    
    return {
        'available': True,
        'device': 'cuda',
        'name': torch.cuda.get_device_name(0),
        'compute_capability': f"{device_props.major}.{device_props.minor}",
        'total_memory_gb': round(device_props.total_memory / 1024**3, 2),
        'multi_processor_count': device_props.multi_processor_count,
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__
    }


def validate_midi_file(midi_path: str) -> bool:
    """
    Validate that a file is a valid MIDI file.
    
    Args:
        midi_path: Path to MIDI file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        import mido
        midi = mido.MidiFile(midi_path)
        return len(midi.tracks) > 0
    except Exception as e:
        print(f"[Validation] Invalid MIDI file {midi_path}: {e}")
        return False


def create_directory_structure(base_dir: str = "."):
    """
    Create standard RWKV-Music directory structure.
    
    Args:
        base_dir: Base directory to create structure in
    """
    directories = [
        "core",
        "scripts",
        "data/raw_midi",
        "data/processed",
        "models",
        "outputs",
        "examples",
        "logs"
    ]
    
    base_path = Path(base_dir)
    
    for dir_name in directories:
        dir_path = base_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"[Setup] Created directory: {dir_path}")
    
    print("[Setup] Directory structure created successfully")


if __name__ == "__main__":
    # Test utility functions
    print("Testing RWKV-Music utility functions...")
    
    print("\nDevice Information:")
    info = get_device_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    if torch.cuda.is_available():
        print("\nVRAM Usage:")
        vram = get_vram_usage()
        for key, value in vram.items():
            print(f"  {key}: {value} GB")
    
    print("\nTime formatting examples:")
    print(f"  3661 seconds = {format_time(3661)}")
    print(f"  245 seconds = {format_time(245)}")
    print(f"  45 seconds = {format_time(45)}")
