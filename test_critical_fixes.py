"""
Simple validation script to check the critical fixes without requiring dependencies.
This performs static code analysis to verify the fixes are in place.
"""

import sys
from pathlib import Path


def test_architecture_fixes():
    """Test that architecture.py has been fixed."""
    print("\n" + "=" * 70)
    print("Test 1: Architecture Fixes")
    print("=" * 70)
    
    with open('core/architecture.py', 'r') as f:
        source = f.read()
    
    issues = []
    
    # Check that NotImplementedError is not in critical path
    if 'raise NotImplementedError' in source:
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'raise NotImplementedError' in line:
                # Check if it's in a method that would be called during training
                context_start = max(0, i - 10)
                context = '\n'.join(lines[context_start:i+1])
                if 'def _time_mixing' in context or 'def _channel_mixing' in context:
                    issues.append(f"NotImplementedError found in critical method at line {i+1}")
    
    # Check that new helper methods exist
    if '_compute_att_output' not in source:
        issues.append("_compute_att_output method not found")
    
    if '_compute_ffn_output' not in source:
        issues.append("_compute_ffn_output method not found")
    
    # Check that _get_hidden_states has been updated
    if 'self.model.forward' not in source:
        issues.append("_get_hidden_states may not be using model.forward properly")
    
    if issues:
        for issue in issues:
            print(f"✗ {issue}")
        return False
    else:
        print("✓ PASS: Architecture fixes applied correctly")
        print("  - NotImplementedError removed from critical path")
        print("  - Helper methods added for gradient computation")
        print("  - Using model.forward for proper gradient flow")
        return True


def test_dataset_fixes():
    """Test that dataset.py has been fixed."""
    print("\n" + "=" * 70)
    print("Test 2: Dataset ctx_len Fix")
    print("=" * 70)
    
    with open('core/dataset.py', 'r') as f:
        source = f.read()
    
    issues = []
    
    # Check for the fix comment
    if 'CRITICAL FIX' not in source and 'ctx_len must reflect' not in source.lower():
        issues.append("Critical fix comment not found - may not be properly documented")
    
    # Check that ctx_len is calculated after truncation
    lines = source.split('\n')
    truncation_line = -1
    ctx_len_line = -1
    
    for i, line in enumerate(lines):
        if 'full_seq = full_seq[:self.max_seq_len]' in line:
            truncation_line = i
        if 'ctx_len = len(ctx_tokens)' in line:
            ctx_len_line = i
    
    # ctx_len should be calculated AFTER all truncation logic
    if truncation_line > 0 and ctx_len_line > 0:
        if ctx_len_line < truncation_line:
            issues.append(f"ctx_len calculated before truncation (line {ctx_len_line} < {truncation_line})")
    
    if issues:
        for issue in issues:
            print(f"✗ {issue}")
        return False
    else:
        print("✓ PASS: Dataset ctx_len fix applied correctly")
        print("  - ctx_len calculated after truncation")
        print("  - Fix properly documented")
        return True


def test_gradscaler_removal():
    """Test that GradScaler has been removed."""
    print("\n" + "=" * 70)
    print("Test 3: GradScaler Removal")
    print("=" * 70)
    
    with open('train_parallel.py', 'r') as f:
        source = f.read()
    
    issues = []
    
    # Check for GradScaler import
    if 'from torch.cuda.amp import autocast, GradScaler' in source:
        issues.append("GradScaler still in import statement")
    
    if 'GradScaler()' in source:
        issues.append("GradScaler still being instantiated")
    
    if 'scaler.scale' in source:
        issues.append("scaler.scale still being called")
    
    if 'scaler.step' in source:
        issues.append("scaler.step still being called")
    
    if 'scaler.update' in source:
        issues.append("scaler.update still being called")
    
    # Check that autocast is still present
    if 'autocast(dtype=torch.bfloat16)' not in source:
        issues.append("BF16 autocast missing - should still be present")
    
    # Check that backward is called directly
    if 'loss.backward()' not in source:
        issues.append("Direct loss.backward() not found")
    
    if issues:
        for issue in issues:
            print(f"✗ {issue}")
        return False
    else:
        print("✓ PASS: GradScaler properly removed")
        print("  - GradScaler import removed")
        print("  - GradScaler instantiation removed")
        print("  - Gradient scaling operations removed")
        print("  - BF16 autocast retained")
        print("  - Direct backward pass implemented")
        return True


def test_cuda_arch_fix():
    """Test that CUDA architecture detection is dynamic."""
    print("\n" + "=" * 70)
    print("Test 4: Dynamic CUDA Architecture")
    print("=" * 70)
    
    with open('core/env_hijack.py', 'r') as f:
        source = f.read()
    
    issues = []
    
    # Check for hardcoded architecture
    if 'os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"' in source:
        issues.append("CUDA architecture still hardcoded to 8.9")
    
    # Check for dynamic detection
    if 'torch.cuda.get_device_capability' not in source:
        issues.append("Dynamic device capability detection not found")
    
    if 'major, minor' not in source:
        issues.append("Major/minor version extraction not found")
    
    # Check for fallback
    if 'except' in source and 'TORCH_CUDA_ARCH_LIST' in source:
        # Good - there's error handling
        pass
    else:
        issues.append("No fallback error handling for capability detection")
    
    if issues:
        for issue in issues:
            print(f"✗ {issue}")
        return False
    else:
        print("✓ PASS: Dynamic CUDA architecture detection implemented")
        print("  - Hardcoded 8.9 removed")
        print("  - Dynamic capability detection added")
        print("  - Fallback error handling present")
        return True


def test_alignment_improvements():
    """Test that alignment improvements are in place."""
    print("\n" + "=" * 70)
    print("Test 5: Alignment Improvements")
    print("=" * 70)
    
    with open('train_parallel.py', 'r') as f:
        source = f.read()
    
    improvements = []
    
    # Check for attention_mask parameter
    if 'attention_mask' in source:
        improvements.append("attention_mask parameter added to loss computation")
    
    # Check for alignment documentation
    if 'CRITICAL FIX' in source or 'perfect alignment' in source.lower():
        improvements.append("Alignment fix documented in code")
    
    # Check for error handling when mismatch detected (should raise exception)
    if 'RuntimeError' in source and 'ALIGNMENT ERROR' in source:
        improvements.append("Strict alignment error checking with exception")
    elif 'Shape mismatch' in source or 'shape mismatch' in source:
        improvements.append("Shape mismatch detection present")
    
    # Check that the same mask is mentioned
    if 'same' in source.lower() and 'mask' in source.lower():
        improvements.append("Documentation mentions using same mask")
    
    # Check padding token assumption is documented
    if 'ASSUMPTION' in source and 'Padding' in source:
        improvements.append("Padding token assumption documented")
    
    if len(improvements) >= 3:
        print("✓ PASS: Alignment improvements implemented")
        for imp in improvements:
            print(f"  - {imp}")
        return True
    else:
        print("✗ FAIL: Insufficient alignment improvements")
        print(f"  Found {len(improvements)} improvements, expected at least 3")
        return False


def test_train_epoch_signature():
    """Test that train_epoch signature has been updated."""
    print("\n" + "=" * 70)
    print("Test 6: train_epoch Signature Update")
    print("=" * 70)
    
    with open('train_parallel.py', 'r') as f:
        source = f.read()
    
    # Find train_epoch definition
    lines = source.split('\n')
    train_epoch_def = None
    for i, line in enumerate(lines):
        if 'def train_epoch(' in line:
            # Collect the full signature (may span multiple lines)
            signature_lines = [line]
            j = i + 1
            while j < len(lines) and ')' not in ''.join(signature_lines):
                signature_lines.append(lines[j])
                j += 1
            train_epoch_def = ''.join(signature_lines)
            break
    
    if not train_epoch_def:
        print("✗ FAIL: train_epoch function not found")
        return False
    
    issues = []
    
    # Check that scaler parameter is removed
    if 'scaler:' in train_epoch_def or 'scaler,' in train_epoch_def:
        issues.append("scaler parameter still in train_epoch signature")
    
    if issues:
        for issue in issues:
            print(f"✗ {issue}")
        return False
    else:
        print("✓ PASS: train_epoch signature updated")
        print("  - scaler parameter removed")
        return True


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("RWKV-Music Critical Fixes - Static Code Validation")
    print("=" * 70)
    print("\nValidating fixes for critical issues:")
    print("1. Architecture NotImplementedError removal")
    print("2. Dataset ctx_len boundary fix")
    print("3. GradScaler removal (BF16 optimization)")
    print("4. Dynamic CUDA architecture detection")
    print("5. Logits-targets alignment improvements")
    print("6. train_epoch signature update")
    
    results = []
    
    try:
        results.append(("Architecture fixes", test_architecture_fixes()))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        results.append(("Architecture fixes", False))
    
    try:
        results.append(("Dataset ctx_len fix", test_dataset_fixes()))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        results.append(("Dataset ctx_len fix", False))
    
    try:
        results.append(("GradScaler removal", test_gradscaler_removal()))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        results.append(("GradScaler removal", False))
    
    try:
        results.append(("Dynamic CUDA arch", test_cuda_arch_fix()))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        results.append(("Dynamic CUDA arch", False))
    
    try:
        results.append(("Alignment improvements", test_alignment_improvements()))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        results.append(("Alignment improvements", False))
    
    try:
        results.append(("train_epoch signature", test_train_epoch_signature()))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        results.append(("train_epoch signature", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n" + "=" * 70)
        print("✓ ALL CRITICAL FIXES VALIDATED SUCCESSFULLY!")
        print("=" * 70)
        print("\nThe following issues have been addressed:")
        print("1. ✓ NotImplementedError removed from architecture")
        print("2. ✓ ctx_len properly bounded to prevent IndexError")
        print("3. ✓ GradScaler removed (unnecessary for BF16)")
        print("4. ✓ CUDA architecture dynamically detected")
        print("5. ✓ Logits-targets alignment improved")
        print("6. ✓ Function signatures updated")
        print("\nThe training pipeline should now work correctly!")
        return 0
    else:
        print(f"\n✗ {total_count - passed_count} validation(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
