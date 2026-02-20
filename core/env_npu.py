# --edited --complete
# FILE PATH: core/env_npu.py
"""
core/env_npu.py
[Genius Protocol] Huawei Ascend NPU 910B Hijacker
"""
import os

def hijack_npu_env():
    """暴力劫持 PyTorch 运行环境，强行路由至 CANN 引擎。物理屏蔽 CUDA。"""
    # 1. 物理封杀 RWKV 内部寻找 nvcc 编译 CUDA 的企图
    os.environ["RWKV_CUDA_ON"] = "0"
    os.environ["RWKV_JIT_ON"] = "0"
    
    # 2. 【TLA+ 级纠错】直接抹杀之前硬编码的 ASCEND_RT_VISIBLE_DEVICES
    # 容器、云训练节点会自动分配物理卡并建立映射。越权写死 0 直接导致 drvErr=87 寻址异常崩溃。
    if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
        del os.environ["ASCEND_RT_VISIBLE_DEVICES"]
    
    try:
        import torch
        import torch_npu
    except ImportError:
        raise ImportError("在 Linux NPU 节点上没装 torch_npu？立刻执行: pip install torch-npu")

    # 3. 压榨 Cube 核心：关闭隐式格式转换防止精度雪崩，开启 CANN 静态图极速模式
    try:
        torch.npu.config.allow_internal_format = False
        torch.npu.set_compile_mode(jit_compile=False)
    except Exception:
        pass
    
    print("========================================================")
    print("[Genius System] NVIDIA CUDA Pipeline Destroyed.")
    try:
        if torch.npu.is_available():
            # 动态获取当前合法设备，而不是盲目传 0
            device_id = torch.npu.current_device()
            print(f"[Genius System] Ascend NPU Mounted: {torch.npu.get_device_name(device_id)} (ID: {device_id})")
        else:
            print("[Genius System] Ascend NPU Mounted: Unreachable")
    except Exception as e:
        print(f"[Genius System] Ascend NPU Init Warning: {e}")
    print("========================================================")

def verify_npu_setup():
    import torch
    import torch_npu
    try:
        assert torch.npu.is_available(), "NPU HBM 寻址失败！检查 CANN 驱动 (npu-smi info)。"
        device_id = torch.npu.current_device()
        print(f"[NPU] Total HBM: {torch.npu.get_device_properties(device_id).total_memory / 1024**3:.2f} GB")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] 无法初始化 Ascend NPU: {e}")
        raise
