"""
End-to-End Test Runner
This script checks if dependencies are available and runs appropriate tests.
"""

import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
    except ImportError:
        missing.append("torch")
        print("✗ PyTorch not installed")
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__} installed")
    except ImportError:
        missing.append("numpy")
        print("✗ NumPy not installed")
    
    return missing

def main():
    print("=" * 70)
    print("RWKV-Music End-to-End Test Runner")
    print("=" * 70)
    
    print("\nChecking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("\nTo run end-to-end tests, install dependencies:")
        print("  pip install -r requirements.txt")
        print("\nOr install minimal dependencies for testing:")
        print("  pip install torch numpy")
        print("\nFor now, running static code validation tests...")
        
        # Run static tests instead
        result = subprocess.run([sys.executable, "test_critical_fixes.py"], 
                              capture_output=False)
        return result.returncode
    else:
        print("\n✓ All dependencies available")
        print("\nRunning full end-to-end tests...")
        
        # Run the full E2E tests
        result = subprocess.run([sys.executable, "test_e2e.py"], 
                              capture_output=False)
        return result.returncode

if __name__ == "__main__":
    sys.exit(main())
