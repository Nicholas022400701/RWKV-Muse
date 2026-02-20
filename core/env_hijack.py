"""
Windows CUDA Environment Hijacking Module
Forcefully injects MSVC compiler paths to enable RWKV CUDA JIT compilation on Windows.
"""

import os
import subprocess
import sys


def hijack_windows_cuda_env():
    """
    Violently hijack Windows environment, forcefully inject MSVC compiler chain paths.
    This enables RWKV CUDA JIT compilation on Windows systems.
    
    Must be called BEFORE importing PyTorch or RWKV.
    """
    if os.name != 'nt':
        print("[Environment] Not Windows, skipping MSVC hijack.")
        return
    
    try:
        # Locate the latest Visual Studio installation
        vswhere = os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe")
        
        if not os.path.exists(vswhere):
            raise FileNotFoundError(
                "Visual Studio Installer not found. "
                "Please install 'Build Tools for Visual Studio' with C++ workload."
            )
        
        # Get Visual Studio installation path
        vs_path = subprocess.check_output(
            f'"{vswhere}" -latest -property installationPath',
            shell=True
        ).decode().strip()
        
        if not vs_path:
            raise RuntimeError("No Visual Studio installation found.")
        
        # Locate vcvars64.bat
        vcvars = os.path.join(vs_path, r"VC\Auxiliary\Build\vcvars64.bat")
        
        if not os.path.exists(vcvars):
            raise FileNotFoundError(f"vcvars64.bat not found at {vcvars}")
        
        # Extract environment variables from vcvars64.bat
        print("[Environment] Loading MSVC environment variables...")
        output = subprocess.check_output(
            f'cmd /c ""{vcvars}" > nul && set"',
            shell=True
        ).decode(errors='ignore')
        
        # Inject critical environment variables into current process
        for line in output.splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                if key.upper() in ['PATH', 'INCLUDE', 'LIB', 'LIBPATH']:
                    os.environ[key.upper()] = value
        
        # Force enable RWKV custom CUDA kernels
        os.environ["RWKV_CUDA_ON"] = "1"
        
        # Dynamically detect GPU compute capability instead of hardcoding
        # This ensures compatibility across different GPU architectures
        try:
            import torch
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                compute_capability = f"{major}.{minor}"
                os.environ["TORCH_CUDA_ARCH_LIST"] = compute_capability
                print(f"[Genius System] Detected GPU compute capability: {compute_capability}")
            else:
                print("[Warning] CUDA not available, skipping architecture detection")
        except Exception as e:
            print(f"[Warning] Failed to detect GPU architecture: {e}")
            # Fallback to a reasonable default
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9"
        
        print("[Genius System] MSVC compiler hijacked successfully.")
        print("[Genius System] RWKV CUDA kernels enabled.")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to hijack MSVC environment: {e}", file=sys.stderr)
        print("\nPlease ensure you have installed:", file=sys.stderr)
        print("1. Visual Studio Build Tools with 'Desktop development with C++'", file=sys.stderr)
        print("2. CUDA Toolkit matching your PyTorch version", file=sys.stderr)
        print("3. Ninja build system (optional but recommended)", file=sys.stderr)
        raise RuntimeError(f"Environment setup failed: {e}")


def verify_cuda_setup():
    """
    Verify that CUDA and compilation environment are properly set up.
    Should be called after hijack_windows_cuda_env().
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")
        
        print(f"[CUDA] PyTorch version: {torch.__version__}")
        print(f"[CUDA] CUDA version: {torch.version.cuda}")
        print(f"[CUDA] Device: {torch.cuda.get_device_name(0)}")
        print(f"[CUDA] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Check if RWKV_CUDA_ON is set
        if os.environ.get("RWKV_CUDA_ON") == "1":
            print("[CUDA] RWKV custom kernels: ENABLED")
        else:
            print("[CUDA] RWKV custom kernels: DISABLED (using PyTorch fallback)")
        
        return True
        
    except ImportError:
        print("[ERROR] PyTorch not installed. Please install PyTorch with CUDA support.")
        return False
    except Exception as e:
        print(f"[ERROR] CUDA setup verification failed: {e}")
        return False
