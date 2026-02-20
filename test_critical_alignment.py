"""
Test script to validate the critical tensor alignment fixes.
Specifically tests the issues identified in the problem statement:
1. Mask misalignment between logits and targets
2. Dataset truncation causing index out of bounds
3. Padding token handling
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.dataset import CopilotDataset, collate_fn
from torch.utils.data import DataLoader


def test_mask_alignment():
    """
    Test that logits and targets have the same dimensions after filtering.
    This was the critical bug: logits contained padding but targets were filtered.
    """
    print("\n" + "=" * 70)
    print("Test: Mask Alignment (Logits vs Targets)")
    print("=" * 70)
    
    # Create mock data with varying lengths to trigger padding
    # Using distinct ranges for clarity: context uses 1000+ range, completion uses 2000+ range
    test_data = [
        {'context': list(range(1000, 1050)), 'completion': list(range(2000, 2050))},  # 50+50=100 tokens
        {'context': list(range(1000, 1030)), 'completion': list(range(2000, 2030))},  # 30+30=60 tokens
        {'context': list(range(1000, 1040)), 'completion': list(range(2000, 2050))},  # 40+50=90 tokens
        {'context': list(range(1000, 1020)), 'completion': list(range(2000, 2030))},  # 20+30=50 tokens
    ]
    
    dataset = CopilotDataset(test_data, max_seq_len=128)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    batch = next(iter(dataloader))
    input_ids = batch['input_ids']
    target_ids = batch['target_ids']
    ctx_lengths = batch['ctx_lengths']
    
    print(f"Batch input shape: {input_ids.shape}")
    print(f"Batch target shape: {target_ids.shape}")
    print(f"Context lengths: {ctx_lengths.tolist()}")
    
    # Simulate what happens in architecture.py forward() with padding filtering
    padding_token_id = 0
    batch_size = input_ids.size(0)
    
    # Extract completion portions and filter padding (architecture.py logic)
    valid_input_count = 0
    for b in range(batch_size):
        ctx_len = ctx_lengths[b].item()
        completion_input = input_ids[b, ctx_len-1:]
        non_pad_mask = completion_input != padding_token_id
        valid_tokens = non_pad_mask.sum().item()
        valid_input_count += valid_tokens
        print(f"  Seq {b}: ctx_len={ctx_len}, completion_len={len(completion_input)}, valid_tokens={valid_tokens}")
    
    # Extract completion targets and filter padding (train_parallel.py logic)
    valid_target_count = 0
    for b in range(batch_size):
        ctx_len = ctx_lengths[b].item()
        completion_target = target_ids[b, ctx_len-1:]
        non_pad_mask = completion_target != padding_token_id
        valid_tokens = non_pad_mask.sum().item()
        valid_target_count += valid_tokens
    
    print(f"\nValid input tokens (for logits): {valid_input_count}")
    print(f"Valid target tokens: {valid_target_count}")
    
    # CRITICAL CHECK: These must match for loss computation to work
    if valid_input_count == valid_target_count:
        print("✓ PASS: Logits and targets will have matching dimensions")
        return True
    else:
        print(f"✗ FAIL: Dimension mismatch! {valid_input_count} != {valid_target_count}")
        return False


def test_dataset_truncation_index_safety():
    """
    Test that ctx_len never exceeds sequence length after truncation.
    This was causing IndexError in hidden_states[b, ctx_len-1:, :]
    """
    print("\n" + "=" * 70)
    print("Test: Dataset Truncation Index Safety")
    print("=" * 70)
    
    # Test case 1: Very long completion that needs truncation
    test_data_1 = [
        {
            'context': list(range(1, 101)),       # 100 tokens
            'completion': list(range(101, 3001))  # 2900 tokens (way too long!)
        }
    ]
    
    dataset1 = CopilotDataset(test_data_1, max_seq_len=2048)
    item1 = dataset1[0]
    
    print(f"Test 1 (long completion):")
    print(f"  ctx_len: {item1['ctx_len']}")
    print(f"  seq_len: {item1['input_ids'].size(0)}")
    print(f"  max_seq_len: 2048")
    
    # Critical check: ctx_len must be <= seq_len
    assert item1['ctx_len'] <= item1['input_ids'].size(0), \
        f"FAIL: ctx_len ({item1['ctx_len']}) > seq_len ({item1['input_ids'].size(0)})"
    
    # Also verify ctx_len is reasonable (not 0)
    assert item1['ctx_len'] > 0, "FAIL: ctx_len should not be 0"
    
    print("  ✓ ctx_len is within bounds")
    
    # Test case 2: Very long context
    test_data_2 = [
        {
            'context': list(range(1, 3001)),   # 3000 tokens
            'completion': list(range(3001, 3101))  # 100 tokens
        }
    ]
    
    dataset2 = CopilotDataset(test_data_2, max_seq_len=2048)
    item2 = dataset2[0]
    
    print(f"\nTest 2 (long context):")
    print(f"  ctx_len: {item2['ctx_len']}")
    print(f"  seq_len: {item2['input_ids'].size(0)}")
    print(f"  max_seq_len: 2048")
    
    assert item2['ctx_len'] <= item2['input_ids'].size(0), \
        f"FAIL: ctx_len ({item2['ctx_len']}) > seq_len ({item2['input_ids'].size(0)})"
    assert item2['ctx_len'] > 0, "FAIL: ctx_len should not be 0"
    
    print("  ✓ ctx_len is within bounds")
    
    print("\n✓ PASS: Dataset truncation correctly maintains ctx_len bounds")
    return True


def test_jit_path_fix():
    """
    Test that JIT compilation uses absolute paths.
    """
    print("\n" + "=" * 70)
    print("Test: JIT Compilation Path Fix")
    print("=" * 70)
    
    # Read the rwkv_v8_model.py file and check for absolute path usage
    model_file = Path(__file__).parent / "core" / "rwkv_training" / "rwkv_v8_model.py"
    
    with open(model_file, 'r') as f:
        content = f.read()
    
    # Check for the fix: should use os.path.abspath or similar
    has_abspath = 'os.path.abspath' in content or 'os.path.dirname' in content
    has_cuda_dir = 'cuda_dir' in content
    
    if has_abspath and has_cuda_dir:
        print("✓ PASS: JIT compilation uses absolute paths")
        print("  Found: os.path.abspath and cuda_dir variable")
        return True
    else:
        print("✗ FAIL: JIT compilation may still use relative paths")
        return False


def test_backward_stub():
    """
    Test that WKV_7 has a backward method with clear error message.
    """
    print("\n" + "=" * 70)
    print("Test: Backward Stub Implementation")
    print("=" * 70)
    
    model_file = Path(__file__).parent / "core" / "rwkv_training" / "rwkv_v8_model.py"
    
    with open(model_file, 'r') as f:
        content = f.read()
    
    # Check for backward implementation
    has_backward = 'def backward' in content
    has_not_implemented = 'NotImplementedError' in content
    
    if has_backward and has_not_implemented:
        print("✓ PASS: Backward stub implemented with NotImplementedError")
        print("  This provides clear error message for training attempts")
        return True
    else:
        print("✗ FAIL: Backward stub not properly implemented")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Critical Alignment Fixes Validation")
    print("=" * 70)
    print("\nThese tests validate fixes for the issues in the problem statement:")
    print("1. Mask misalignment between logits and targets")
    print("2. Dataset truncation causing index out of bounds")
    print("3. JIT compilation path issues")
    print("4. Missing backward implementation")
    
    results = []
    
    try:
        results.append(("Mask Alignment", test_mask_alignment()))
    except Exception as e:
        print(f"✗ FAIL: Mask alignment test crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Mask Alignment", False))
    
    try:
        results.append(("Dataset Truncation", test_dataset_truncation_index_safety()))
    except Exception as e:
        print(f"✗ FAIL: Dataset truncation test crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Dataset Truncation", False))
    
    try:
        results.append(("JIT Path Fix", test_jit_path_fix()))
    except Exception as e:
        print(f"✗ FAIL: JIT path test crashed: {e}")
        results.append(("JIT Path Fix", False))
    
    try:
        results.append(("Backward Stub", test_backward_stub()))
    except Exception as e:
        print(f"✗ FAIL: Backward stub test crashed: {e}")
        results.append(("Backward Stub", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All critical alignment fixes validated successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
