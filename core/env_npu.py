"""
core/env_npu.py
[Genius Protocol] Huawei Ascend NPU 910B Hijacker
"""
import os

def hijack_npu_env():
    """暴力劫持 PyTorch 运行环境，强行路由至 CANN 引擎。物理屏蔽 CUDA。"""
    try:
        import torch
        import torch_npu
    except ImportError:
        raise ImportError("在 Linux NPU 节点上没装 torch_npu？立刻执行: pip install torch-npu")

    # 1. 物理封杀 RWKV 内部寻找 nvcc 编译 CUDA 的企图
    os.environ["RWKV_CUDA_ON"] = "0"
    os.environ["RWKV_JIT_ON"] = "0"
    
    # 2. 算力路由强行收束到 NPU 0号核心
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    
    # 3. 压榨 Cube 核心：关闭隐式格式转换防止精度雪崩，开启 CANN 静态图极速模式
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    
    print("========================================================")
    print("[Genius System] NVIDIA CUDA Pipeline Destroyed.")
    print(f"[Genius System] Ascend NPU Mounted: {torch.npu.get_device_name(0)}")
    print("========================================================")

def verify_npu_setup():
    import torch
    import torch_npu
    assert torch.npu.is_available(), "NPU HBM 寻址失败！检查 CANN 驱动 (npu-smi info)。"
    print(f"[NPU] Total HBM: {torch.npu.get_device_properties(0).total_memory / 1024**3:.2f} GB")