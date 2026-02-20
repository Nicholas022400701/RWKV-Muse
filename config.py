"""
Configuration file for RWKV-Music model architecture and training.
"""

# Model Architecture
MODEL_CONFIG = {
    # Recommended for RTX 4090 (24GB)
    'small': {
        'n_layer': 24,
        'n_embd': 1024,
        'vocab_size': 65536,
        'estimated_params': '430M',
        'estimated_vram_gb': 8
    },
    'base': {
        'n_layer': 32,
        'n_embd': 2048,
        'vocab_size': 65536,
        'estimated_params': '1.5B',
        'estimated_vram_gb': 18
    },
    'large': {
        'n_layer': 48,
        'n_embd': 2560,
        'vocab_size': 65536,
        'estimated_params': '3B',
        'estimated_vram_gb': 23
    }
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 4,
    'max_seq_len': 2048,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    'epochs': 10,
    'warmup_epochs': 1,
    'precision': 'bf16',  # bfloat16 for RTX 4090
    'save_every': 1,
}

# Data Processing Configuration
DATA_CONFIG = {
    'n_context_bars': 4,
    'n_completion_bars': 2,
    'step': 1,
    'max_files': None,  # None = process all files
}

# Tokenization Configuration
TOKENIZATION_CONFIG = {
    'num_velocities': 32,
    'beat_res': {(0, 4): 8, (4, 12): 4},
    'num_tempos': 32,
    'tempo_range': (40, 250),
    'use_chords': False,
    'use_rests': True,
    'use_tempos': True,
    'use_time_signatures': True,
}

# Generation Configuration
GENERATION_CONFIG = {
    'temperature': 0.85,
    'top_p': 0.90,
    'top_k': 0,
    'max_new_tokens': 512,
}

# CUDA Configuration
CUDA_CONFIG = {
    'cuda_arch_list': '8.9',  # RTX 4090
    'enable_custom_kernels': True,
    'compile_on_first_run': True,
}

# Paths (modify as needed)
PATHS = {
    'raw_midi_dir': './data/raw_midi',
    'processed_data_dir': './data/processed',
    'model_dir': './models',
    'output_dir': './outputs',
}
