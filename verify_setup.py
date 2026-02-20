"""
Verification script to check RWKV-Music installation and setup.
Run this after installation to ensure everything is configured correctly.
"""

import sys
import importlib
from pathlib import Path


def check_module(module_name, package_name=None):
    """Check if a module is installed."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError:
        print(f"✗ {package_name or module_name} - NOT INSTALLED")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  - Device: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("✗ CUDA not available")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False


def check_file_structure():
    """Check if all required files exist."""
    required_files = [
        "core/__init__.py",
        "core/env_hijack.py",
        "core/tokenization.py",
        "core/dataset.py",
        "core/architecture.py",
        "core/utils.py",
        "train_parallel.py",
        "infer_copilot.py",
        "scripts/preprocess_data.py",
        "config.py",
        "requirements.txt",
        "README.md"
    ]
    
    print("\nFile Structure:")
    all_exist = True
    for file_path in required_files:
        exists = Path(file_path).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        all_exist = all_exist and exists
    
    return all_exist


def check_directories():
    """Check if required directories exist."""
    required_dirs = [
        "core",
        "scripts",
        "examples",
        "data",
        "models"
    ]
    
    print("\nDirectories:")
    all_exist = True
    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {dir_path}/")
        all_exist = all_exist and exists
    
    return all_exist


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("RWKV-Music Installation Verification")
    print("=" * 70)
    
    # Check dependencies
    print("\nPython Dependencies:")
    deps_ok = True
    deps_ok &= check_module("torch", "PyTorch")
    deps_ok &= check_module("numpy", "NumPy")
    deps_ok &= check_module("miditok", "MidiTok")
    deps_ok &= check_module("datasets", "Hugging Face Datasets")
    deps_ok &= check_module("mido", "Mido")
    
    # Check CUDA
    print("\nCUDA Setup:")
    cuda_ok = check_cuda()
    
    # Check file structure
    files_ok = check_file_structure()
    
    # Check directories
    dirs_ok = check_directories()
    
    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)
    
    all_ok = deps_ok and cuda_ok and files_ok and dirs_ok
    
    if all_ok:
        print("✓ All checks passed! RWKV-Music is ready to use.")
        print("\nNext steps:")
        print("1. Read QUICKSTART.md for usage instructions")
        print("2. Prepare your MIDI dataset")
        print("3. Run data preprocessing")
        print("4. Start training!")
    else:
        print("✗ Some checks failed. Please install missing dependencies.")
        print("\nTo install dependencies:")
        print("  pip install -r requirements.txt")
        
        if not cuda_ok:
            print("\nCUDA issues:")
            print("  1. Install CUDA Toolkit matching PyTorch version")
            print("  2. On Windows, ensure Visual Studio Build Tools is installed")
            print("  3. Run: from core.env_hijack import hijack_windows_cuda_env")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
