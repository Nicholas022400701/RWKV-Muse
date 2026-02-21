import os
import subprocess
import sys

def hijack_windows_cuda_env():
    """暴力劫持 Windows 环境，强行注入 MSVC 编译链路径，打通原生 CUDA JIT"""
    if os.name != 'nt': return
    try:
        vswhere = os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe")
        if not os.path.exists(vswhere):
            return
            
        vs_path = subprocess.check_output(f'"{vswhere}" -latest -property installationPath', shell=True).decode().strip()
        vcvars = os.path.join(vs_path, r"VC\Auxiliary\Build\vcvars64.bat")
        
        output = subprocess.check_output(f'cmd /c ""{vcvars}" > nul && set"', shell=True).decode(errors='ignore')
        for line in output.splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                if k.upper() in ['PATH', 'INCLUDE', 'LIB', 'LIBPATH']:
                    os.environ[k.upper()] = v
                
        os.environ["RWKV_CUDA_ON"] = "1"
        
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
            print(f"[Genius System] MSVC Compiler hijacked. Pure CUDA {major}.{minor} compute activated.")
    except Exception as e:
        print(f"Warning: Failed to hijack MSVC environment: {e}", file=sys.stderr)