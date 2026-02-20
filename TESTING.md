# RWKV-Music Testing Guide

This document describes the testing infrastructure for RWKV-Music.

## Test Types

### 1. Static Code Validation (`test_critical_fixes.py`)

**Purpose:** Fast validation that critical fixes are in place without requiring dependencies.

**What it tests:**
- Architecture code patterns (NotImplementedError removal, helper methods)
- Dataset ctx_len fix implementation
- GradScaler removal
- Dynamic CUDA architecture detection
- Function signatures

**Run:**
```bash
python test_critical_fixes.py
```

**When to use:**
- Quick validation during development
- CI/CD pipelines without GPU
- Pre-commit hooks
- Verifying code structure

### 2. End-to-End Tests (`test_e2e.py`)

**Purpose:** Comprehensive validation of the complete training pipeline.

**What it tests:**
1. **Data Pipeline E2E**
   - Dataset creation from mock data
   - Batch collation
   - Shape validation
   - ctx_len boundary checks

2. **Model Forward Pass E2E**
   - Physical slicing logic
   - Memory reduction validation
   - Logits shape verification

3. **Loss Computation E2E**
   - Alignment between logits and targets
   - Padding handling
   - Loss value sanity checks

4. **Backward Pass E2E**
   - Gradient computation
   - NaN/Inf detection
   - Gradient flow validation

5. **Training Step E2E**
   - Complete data → forward → loss → backward → update cycle
   - Gradient clipping
   - Optimizer step

6. **BFloat16 Training E2E**
   - Mixed precision forward/backward
   - No GradScaler validation (BF16 doesn't need it)

7. **Full Integration E2E**
   - Multiple training iterations
   - Loss progression
   - Inference mode validation

**Run:**
```bash
# Requires: torch, numpy
pip install torch numpy
python test_e2e.py
```

**When to use:**
- Before committing major changes
- Validating training pipeline
- Testing on new hardware
- Regression testing

### 3. Test Runner (`run_tests.py`)

**Purpose:** Smart test runner that detects available dependencies and runs appropriate tests.

**Run:**
```bash
python run_tests.py
```

**Behavior:**
- If dependencies missing → runs static tests
- If dependencies available → runs full E2E tests
- Provides clear instructions for missing dependencies

## Running Tests

### Quick Test (No Dependencies)
```bash
python test_critical_fixes.py
```

### Full Test (Requires Dependencies)
```bash
# Install dependencies
pip install torch numpy

# Run E2E tests
python test_e2e.py

# Or use the smart runner
python run_tests.py
```

### With CUDA (Full GPU Testing)
```bash
# Requires CUDA-enabled PyTorch
python test_e2e.py
# Will also test BFloat16 training
```

## Test Output

### Static Tests Output
```
======================================================================
RWKV-Music Critical Fixes - Static Code Validation
======================================================================
✓ PASS: Architecture fixes
✓ PASS: Dataset ctx_len fix
...
Total: 6/6 tests passed
```

### E2E Tests Output
```
======================================================================
RWKV-Music End-to-End Testing Suite
======================================================================
✓ Data Pipeline
✓ Model Forward Pass
✓ Loss Alignment
✓ Backward Pass
✓ Training Step
✓ BFloat16 Training
✓ Full Integration

Total: 7/7 tests passed
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  static-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Static validation
        run: python test_critical_fixes.py

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install torch numpy
      - name: Run E2E tests
        run: python test_e2e.py
```

## Adding New Tests

### Static Test Pattern

```python
def test_new_feature():
    """Test that new feature is implemented."""
    with open('core/module.py', 'r') as f:
        source = f.read()
    
    # Check for presence of key code
    assert 'new_feature' in source, "New feature not found"
    
    print("✓ PASS: New feature check")
    return True
```

### E2E Test Pattern

```python
def test_new_pipeline_e2e():
    """Test new pipeline end-to-end."""
    print_test_header("New Pipeline Test")
    
    try:
        # 1. Setup
        data = create_mock_data()
        
        # 2. Execute pipeline
        result = run_pipeline(data)
        
        # 3. Validate
        assert result is not None
        assert result.shape == expected_shape
        
        print_success("Pipeline executed successfully")
        return True
        
    except Exception as e:
        print_fail(f"Test failed: {e}")
        return False
```

## Test Coverage

Current test coverage:

| Component | Static | E2E | Integration |
|-----------|--------|-----|-------------|
| Architecture | ✓ | ✓ | ✓ |
| Dataset | ✓ | ✓ | ✓ |
| Training Loop | ✓ | ✓ | ✓ |
| Loss Computation | ✓ | ✓ | ✓ |
| Backward Pass | - | ✓ | ✓ |
| BFloat16 | ✓ | ✓ | ✓ |
| CUDA Kernels | ✓ | - | - |

**Legend:**
- ✓ = Tested
- - = Not applicable or not yet tested

## Troubleshooting

### Test Failures

**"Module not found" errors:**
```bash
pip install -r requirements.txt
```

**"CUDA out of memory" during E2E tests:**
- Tests use small mock models, shouldn't require much memory
- Try: `export CUDA_VISIBLE_DEVICES=""`

**BFloat16 test skipped:**
- Normal if GPU doesn't support BF16 (pre-Ampere)
- Not a failure, just skipped

**Gradient is NaN:**
- Check learning rate (may be too high in test)
- Verify input data doesn't have extreme values

### Getting Help

1. Check test output for specific error messages
2. Run with verbose mode: `python test_e2e.py -v`
3. Check GitHub Issues for similar problems
4. Run static tests to isolate if issue is dependencies or code

## Best Practices

1. **Run static tests frequently** - They're fast and catch obvious issues
2. **Run E2E tests before PR** - Ensures nothing breaks
3. **Add tests for new features** - Maintain coverage
4. **Use mock data in tests** - Keep tests fast and deterministic
5. **Test error cases** - Not just happy paths

## Performance Benchmarks

Typical test execution times:

| Test Suite | Duration | Requirements |
|------------|----------|--------------|
| Static validation | ~1 second | None |
| E2E (CPU) | ~10 seconds | torch, numpy |
| E2E (GPU) | ~5 seconds | CUDA, torch |

## Future Improvements

Planned test additions:
- [ ] Tokenization pipeline tests
- [ ] MIDI I/O tests  
- [ ] Model checkpoint save/load tests
- [ ] Distributed training tests
- [ ] Performance regression tests
- [ ] Memory leak detection tests

## References

- PyTorch Testing: https://pytorch.org/docs/stable/testing.html
- Test-Driven Development: https://en.wikipedia.org/wiki/Test-driven_development
- CI/CD Best Practices: https://github.com/features/actions
