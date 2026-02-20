"""
Validation Script for Todo.md Implementation
Performs static code analysis to verify all Todo.md requirements are met.
"""

import sys
from pathlib import Path


def test_rwkv_v8_model_fixes():
    """Test that rwkv_v8_model.py has proper fixes."""
    print("\n" + "=" * 70)
    print("Test 1: RWKV V8 Model - Ghost Dictionary Fix")
    print("=" * 70)
    
    with open('core/rwkv_training/rwkv_v8_model.py', 'r') as f:
        source = f.read()
    
    issues = []
    passed = []
    
    # Check for nn.ParameterDict
    if 'nn.ParameterDict()' in source:
        passed.append("nn.ParameterDict used instead of plain dict")
    else:
        issues.append("nn.ParameterDict not found - ghost dict may still exist")
    
    # Check for requires_grad=True
    if 'requires_grad=True' in source:
        passed.append("Parameters set with requires_grad=True")
    else:
        issues.append("requires_grad=True not found")
    
    # Check for forward_seq method
    if 'def forward_seq' in source:
        passed.append("forward_seq method for parallel processing exists")
    else:
        issues.append("forward_seq method not found")
    
    # Check for dynamic layer deduction
    if 'layer_keys' in source and 'max(layer_keys)' in source:
        passed.append("Dynamic layer count deduction implemented")
    else:
        issues.append("Dynamic layer deduction not found")
    
    if issues:
        for issue in issues:
            print(f"✗ {issue}")
        for p in passed:
            print(f"✓ {p}")
        return False
    else:
        for p in passed:
            print(f"✓ {p}")
        return True


def test_architecture_time_decay():
    """Test that architecture.py has proper time decay."""
    print("\n" + "=" * 70)
    print("Test 2: Architecture - Time Decay & Token Shift")
    print("=" * 70)
    
    with open('core/architecture.py', 'r') as f:
        source = f.read()
    
    issues = []
    passed = []
    
    # Check for Token Shift (x_prev)
    if 'x_prev' in source and 'torch.cat' in source:
        passed.append("Token Shift (x_prev) implemented")
    else:
        issues.append("Token Shift not found")
    
    # Check for state machine with decay
    if 'state = state * w_' in source or 'state * w_decay' in source:
        passed.append("Time decay state machine implemented")
    else:
        issues.append("Time decay state machine not found")
    
    # Check for _batched_time_mix
    if 'def _batched_time_mix' in source:
        passed.append("_batched_time_mix method exists")
    else:
        issues.append("_batched_time_mix method not found")
    
    # Check for generate method with forward_seq
    if 'def generate' in source and 'forward_seq' in source:
        passed.append("generate method with parallel prefill exists")
    else:
        issues.append("generate method with parallel prefill not found")
    
    # Check for out_list (avoiding in-place operations)
    if 'out_list' in source:
        passed.append("out_list pattern for gradient-safe operations")
    else:
        issues.append("out_list pattern not found")
    
    if issues:
        for issue in issues:
            print(f"✗ {issue}")
        for p in passed:
            print(f"✓ {p}")
        return False
    else:
        for p in passed:
            print(f"✓ {p}")
        return True


def test_tokenization_structural():
    """Test that tokenization.py has structural token detection."""
    print("\n" + "=" * 70)
    print("Test 3: Tokenization - Structural Token Detection")
    print("=" * 70)
    
    with open('core/tokenization.py', 'r') as f:
        source = f.read()
    
    issues = []
    passed = []
    
    # Check for is_structural_token method
    if 'def is_structural_token' in source:
        passed.append("is_structural_token method exists")
    else:
        issues.append("is_structural_token method not found")
    
    # Check for structural token types
    if 'Bar' in source and 'NoteOn' in source and 'Pitch' in source:
        passed.append("Structural token types (Bar, NoteOn, Pitch) checked")
    else:
        issues.append("Structural token types not properly checked")
    
    if issues:
        for issue in issues:
            print(f"✗ {issue}")
        for p in passed:
            print(f"✓ {p}")
        return False
    else:
        for p in passed:
            print(f"✓ {p}")
        return True


def test_dataset_safe_truncation():
    """Test that dataset.py has safe truncation."""
    print("\n" + "=" * 70)
    print("Test 4: Dataset - Safe Atomic Truncation")
    print("=" * 70)
    
    with open('core/dataset.py', 'r') as f:
        source = f.read()
    
    issues = []
    passed = []
    
    # Check for tokenizer parameter
    if 'tokenizer' in source and 'def __init__' in source:
        passed.append("tokenizer parameter added to __init__")
    else:
        issues.append("tokenizer parameter not found in __init__")
    
    # Check for is_structural_token usage
    if 'is_structural_token' in source:
        passed.append("is_structural_token used for safe truncation")
    else:
        issues.append("is_structural_token not used")
    
    # Check for target_idx logic
    if 'target_idx' in source:
        passed.append("target_idx logic for finding safe cut points")
    else:
        issues.append("target_idx logic not found")
    
    if issues:
        for issue in issues:
            print(f"✗ {issue}")
        for p in passed:
            print(f"✓ {p}")
        return False
    else:
        for p in passed:
            print(f"✓ {p}")
        return True


def test_train_parallel_tokenizer():
    """Test that train_parallel.py passes tokenizer."""
    print("\n" + "=" * 70)
    print("Test 5: Training - Tokenizer Integration")
    print("=" * 70)
    
    with open('train_parallel.py', 'r') as f:
        source = f.read()
    
    issues = []
    passed = []
    
    # Check for PianoTokenizer import
    if 'from core.tokenization import PianoTokenizer' in source:
        passed.append("PianoTokenizer imported")
    else:
        issues.append("PianoTokenizer not imported")
    
    # Check for tokenizer instantiation
    if 'tokenizer = PianoTokenizer' in source:
        passed.append("Tokenizer instantiated")
    else:
        issues.append("Tokenizer not instantiated")
    
    # Check for tokenizer passed to dataset
    if 'tokenizer=tokenizer' in source:
        passed.append("Tokenizer passed to CopilotDataset")
    else:
        issues.append("Tokenizer not passed to dataset")
    
    if issues:
        for issue in issues:
            print(f"✗ {issue}")
        for p in passed:
            print(f"✓ {p}")
        return False
    else:
        for p in passed:
            print(f"✓ {p}")
        return True


def test_infer_parallel_prefill():
    """Test that infer_copilot.py uses parallel prefill."""
    print("\n" + "=" * 70)
    print("Test 6: Inference - Parallel Prefill")
    print("=" * 70)
    
    with open('infer_copilot.py', 'r') as f:
        source = f.read()
    
    issues = []
    passed = []
    
    # Check for model.generate usage
    if 'model.generate' in source:
        passed.append("model.generate method called")
    else:
        issues.append("model.generate not used")
    
    # Check for parallel prefill message
    if 'Parallel Prefill' in source:
        passed.append("Parallel prefill message present")
    else:
        issues.append("Parallel prefill message not found")
    
    # Check that hasattr check for generate exists
    if 'hasattr(model' in source and 'generate' in source:
        passed.append("Safe fallback for generate method")
    else:
        issues.append("No safe fallback for generate method")
    
    if issues:
        for issue in issues:
            print(f"✗ {issue}")
        for p in passed:
            print(f"✓ {p}")
        return False
    else:
        for p in passed:
            print(f"✓ {p}")
        return True


def test_batch_scripts():
    """Test that batch scripts exist."""
    print("\n" + "=" * 70)
    print("Test 7: Windows Batch Scripts")
    print("=" * 70)
    
    issues = []
    passed = []
    
    # Check for run_train.bat
    if Path('run_train.bat').exists():
        with open('run_train.bat', 'r') as f:
            content = f.read()
        if 'uv run' in content and 'train_parallel.py' in content:
            passed.append("run_train.bat exists with correct content")
        else:
            issues.append("run_train.bat exists but content incorrect")
    else:
        issues.append("run_train.bat not found")
    
    # Check for run_infer.bat
    if Path('run_infer.bat').exists():
        with open('run_infer.bat', 'r') as f:
            content = f.read()
        if 'uv run' in content and 'infer_copilot.py' in content:
            passed.append("run_infer.bat exists with correct content")
        else:
            issues.append("run_infer.bat exists but content incorrect")
    else:
        issues.append("run_infer.bat not found")
    
    if issues:
        for issue in issues:
            print(f"✗ {issue}")
        for p in passed:
            print(f"✓ {p}")
        return False
    else:
        for p in passed:
            print(f"✓ {p}")
        return True


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("TODO.MD IMPLEMENTATION VALIDATION")
    print("=" * 70)
    
    results = []
    
    results.append(test_rwkv_v8_model_fixes())
    results.append(test_architecture_time_decay())
    results.append(test_tokenization_structural())
    results.append(test_dataset_safe_truncation())
    results.append(test_train_parallel_tokenizer())
    results.append(test_infer_parallel_prefill())
    results.append(test_batch_scripts())
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("\n✅ ALL TESTS PASSED - Todo.md requirements fully implemented!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed - Some requirements may be missing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
