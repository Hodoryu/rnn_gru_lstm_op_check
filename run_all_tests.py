#!/usr/bin/env python3
"""
Comprehensive test suite for RNN implementation with optional parameters
"""

import numpy as np
import sys
import os
from validation_test import validate_rnn_implementation
from create_rnn_model import save_rnn_model
from cpu_rnn_implementation import CPURNNImplementation, CPURNNFamilyImplementation

def test_basic_functionality():
    """Test basic RNN functionality without optional parameters"""
    print("=== Testing Basic RNN Functionality ===")

    # Test parameters
    seq_length, batch_size, input_size, hidden_size = 3, 2, 4, 5

    # Generate test data
    np.random.seed(42)
    X = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
    W = np.random.randn(1, hidden_size, input_size).astype(np.float32) * 0.1
    R = np.random.randn(1, hidden_size, hidden_size).astype(np.float32) * 0.1
    B = np.random.randn(1, 2 * hidden_size).astype(np.float32) * 0.1

    # Test CPU implementation
    cpu_rnn = CPURNNImplementation(activation="Tanh")
    Y, Y_h = cpu_rnn(X, W, R, B, direction="forward")

    print(f"Input shape: {X.shape}")
    print(f"Output Y shape: {Y.shape}")
    print(f"Output Y_h shape: {Y_h.shape}")
    print("✓ Basic functionality test passed\n")

def test_optional_parameters():
    """Test RNN with optional parameters"""
    print("=== Testing Optional Parameters ===")

    # Test parameters
    seq_length, batch_size, input_size, hidden_size = 4, 3, 2, 3

    # Generate test data
    np.random.seed(123)
    X = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
    W = np.random.randn(1, hidden_size, input_size).astype(np.float32) * 0.1
    R = np.random.randn(1, hidden_size, hidden_size).astype(np.float32) * 0.1
    B = np.random.randn(1, 2 * hidden_size).astype(np.float32) * 0.1

    # Test with sequence_lens
    sequence_lens = np.array([4, 2, 3], dtype=np.int32)
    initial_h = np.random.randn(1, batch_size, hidden_size).astype(np.float32) * 0.1

    cpu_rnn = CPURNNImplementation(activation="Tanh")

    # Test with sequence_lens only
    Y1, Y_h1 = cpu_rnn(X, W, R, B, direction="forward", sequence_lens=sequence_lens)
    print(f"With sequence_lens: Y shape {Y1.shape}, Y_h shape {Y_h1.shape}")

    # Test with initial_h only
    Y2, Y_h2 = cpu_rnn(X, W, R, B, direction="forward", initial_h=initial_h)
    print(f"With initial_h: Y shape {Y2.shape}, Y_h shape {Y_h2.shape}")

    # Test with both
    Y3, Y_h3 = cpu_rnn(X, W, R, B, direction="forward",
                       sequence_lens=sequence_lens, initial_h=initial_h)
    print(f"With both: Y shape {Y3.shape}, Y_h shape {Y_h3.shape}")

    # Verify sequence masking works
    for b in range(batch_size):
        seq_len = sequence_lens[b]
        if seq_len < seq_length:
            # Check that outputs beyond sequence length are zero
            assert np.allclose(Y1[seq_len:, 0, b, :], 0), f"Sequence masking failed for batch {b}"

    print("✓ Optional parameters test passed\n")

def test_bidirectional():
    """Test bidirectional RNN"""
    print("=== Testing Bidirectional RNN ===")

    # Test parameters
    seq_length, batch_size, input_size, hidden_size = 3, 2, 4, 5

    # Generate test data for bidirectional
    np.random.seed(456)
    X = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
    W = np.random.randn(2, hidden_size, input_size).astype(np.float32) * 0.1
    R = np.random.randn(2, hidden_size, hidden_size).astype(np.float32) * 0.1
    B = np.random.randn(2, 2 * hidden_size).astype(np.float32) * 0.1

    sequence_lens = np.array([3, 2], dtype=np.int32)
    initial_h = np.random.randn(2, batch_size, hidden_size).astype(np.float32) * 0.1

    cpu_rnn = CPURNNImplementation(activation="Tanh")
    Y, Y_h = cpu_rnn(X, W, R, B, direction="bidirectional",
                     sequence_lens=sequence_lens, initial_h=initial_h)

    print(f"Bidirectional Y shape: {Y.shape}")  # Should be (seq_len, 2, batch, hidden)
    print(f"Bidirectional Y_h shape: {Y_h.shape}")  # Should be (2, batch, hidden)

    assert Y.shape == (seq_length, 2, batch_size, hidden_size)
    assert Y_h.shape == (2, batch_size, hidden_size)

    print("✓ Bidirectional RNN test passed\n")

def test_model_creation():
    """Test ONNX model creation with different configurations"""
    print("=== Testing ONNX Model Creation ===")

    try:
        # Test basic model
        model1 = save_rnn_model()
        print("✓ Basic model created")

        # Test model with sequence_lens
        model2 = save_rnn_model(include_sequence_lens=True)
        print("✓ Model with sequence_lens created")

        # Test model with initial_h
        model3 = save_rnn_model(include_initial_h=True)
        print("✓ Model with initial_h created")

        # Test model with both
        model4 = save_rnn_model(include_sequence_lens=True, include_initial_h=True)
        print("✓ Model with both optional inputs created")

        print("✓ All ONNX models created successfully\n")

    except Exception as e:
        print(f"✗ Model creation failed: {e}\n")
        return False

    return True

def test_onnx_spec_compliance():
    """Test ONNX specification compliance for default values"""
    print("=== Testing ONNX Specification Compliance ===")

    # Test parameters
    seq_length, batch_size, input_size, hidden_size = 4, 3, 2, 3

    # Generate test data
    np.random.seed(789)
    X = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
    W = np.random.randn(1, hidden_size, input_size).astype(np.float32) * 0.1
    R = np.random.randn(1, hidden_size, hidden_size).astype(np.float32) * 0.1
    B = np.random.randn(1, 2 * hidden_size).astype(np.float32) * 0.1

    cpu_rnn = CPURNNImplementation(activation="Tanh")

    # Test 1: Default behavior (no optional parameters)
    print("Test 1: Default behavior")
    Y_default, Y_h_default = cpu_rnn(X, W, R, B, direction="forward")
    print(f"  Default output shapes: Y={Y_default.shape}, Y_h={Y_h_default.shape}")

    # Test 2: Explicit sequence_lens = seq_length for all batches
    print("Test 2: Explicit sequence_lens = seq_length")
    sequence_lens_full = np.full(batch_size, seq_length, dtype=np.int32)
    Y_explicit_seq, Y_h_explicit_seq = cpu_rnn(X, W, R, B, direction="forward",
                                               sequence_lens=sequence_lens_full)

    # Should be identical to default behavior
    assert np.allclose(Y_default, Y_explicit_seq), "Default and explicit seq_length should match"
    assert np.allclose(Y_h_default, Y_h_explicit_seq), "Default and explicit seq_length should match"
    print("  ✓ Explicit seq_length matches default behavior")

    # Test 3: Explicit initial_h = zeros
    print("Test 3: Explicit initial_h = zeros")
    initial_h_zeros = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    Y_explicit_h, Y_h_explicit_h = cpu_rnn(X, W, R, B, direction="forward",
                                           initial_h=initial_h_zeros)

    # Should be identical to default behavior
    assert np.allclose(Y_default, Y_explicit_h), "Default and explicit zeros should match"
    assert np.allclose(Y_h_default, Y_h_explicit_h), "Default and explicit zeros should match"
    print("  ✓ Explicit zeros initial_h matches default behavior")

    # Test 4: Both explicit parameters
    print("Test 4: Both explicit parameters")
    Y_both_explicit, Y_h_both_explicit = cpu_rnn(X, W, R, B, direction="forward",
                                                 sequence_lens=sequence_lens_full,
                                                 initial_h=initial_h_zeros)

    # Should be identical to default behavior
    assert np.allclose(Y_default, Y_both_explicit), "All explicit should match default"
    assert np.allclose(Y_h_default, Y_h_both_explicit), "All explicit should match default"
    print("  ✓ Both explicit parameters match default behavior")

    # Test 5: Verify sequence masking with shorter sequences
    print("Test 5: Sequence masking behavior")
    sequence_lens_short = np.array([2, 3, 1], dtype=np.int32)  # Shorter than seq_length=4
    Y_masked, Y_h_masked = cpu_rnn(X, W, R, B, direction="forward",
                                  sequence_lens=sequence_lens_short)

    # Verify masking: outputs beyond sequence length should be zero
    for b in range(batch_size):
        seq_len = sequence_lens_short[b]
        if seq_len < seq_length:
            masked_outputs = Y_masked[seq_len:, 0, b, :]
            assert np.allclose(masked_outputs, 0), f"Masking failed for batch {b}"

    print("  ✓ Sequence masking works correctly")

    # Test 6: Non-zero initial_h effect
    print("Test 6: Non-zero initial_h effect")
    initial_h_nonzero = np.random.randn(1, batch_size, hidden_size).astype(np.float32) * 0.5
    Y_nonzero_h, Y_h_nonzero_h = cpu_rnn(X, W, R, B, direction="forward",
                                         initial_h=initial_h_nonzero)

    # Should be different from default (zero initial_h)
    assert not np.allclose(Y_default, Y_nonzero_h), "Non-zero initial_h should produce different results"
    print("  ✓ Non-zero initial_h produces different results as expected")

    print("✓ All ONNX specification compliance tests passed\n")

def test_rnn_family_comprehensive():
    """Comprehensive test for all RNN family operators"""
    print("=== Testing RNN Family Comprehensive ===")

    operators = ["RNN", "GRU", "LSTM"]
    directions = ["forward"]
    activations = {"RNN": ["Tanh", "Relu"], "GRU": ["Tanh"], "LSTM": ["Tanh"]}

    # Comprehensive shape configurations - 30 test cases
    shape_configs = [
        # Small shapes (10 configurations) - seq_len ≤ 16, batch ≤ 8, input/hidden ≤ 32
        {"seq_len": 1, "batch": 1, "input": 1, "hidden": 1},      # Minimal edge case
        {"seq_len": 2, "batch": 1, "input": 2, "hidden": 2},      # Tiny dimensions
        {"seq_len": 3, "batch": 2, "input": 4, "hidden": 5},      # Original test case
        {"seq_len": 5, "batch": 3, "input": 6, "hidden": 4},      # Original test case
        {"seq_len": 8, "batch": 4, "input": 8, "hidden": 8},      # Power of 2 dimensions
        {"seq_len": 10, "batch": 2, "input": 12, "hidden": 16},   # Mixed small sizes
        {"seq_len": 12, "batch": 6, "input": 16, "hidden": 12},   # Rectangular matrices
        {"seq_len": 15, "batch": 5, "input": 20, "hidden": 24},   # Non-power-of-2
        {"seq_len": 16, "batch": 8, "input": 32, "hidden": 28},   # Upper small boundary
        {"seq_len": 7, "batch": 3, "input": 9, "hidden": 11},     # Prime-like numbers

        # Medium shapes (10 configurations) - seq_len ≤ 64, batch ≤ 32, input/hidden ≤ 128
        {"seq_len": 20, "batch": 10, "input": 40, "hidden": 48},  # Medium start
        {"seq_len": 25, "batch": 12, "input": 50, "hidden": 60},  # Quarter boundaries
        {"seq_len": 32, "batch": 16, "input": 64, "hidden": 64},  # Power of 2 medium
        {"seq_len": 40, "batch": 20, "input": 80, "hidden": 96},  # Balanced medium
        {"seq_len": 48, "batch": 24, "input": 96, "hidden": 80},  # Rectangular medium
        {"seq_len": 50, "batch": 25, "input": 100, "hidden": 120}, # Round numbers
        {"seq_len": 56, "batch": 28, "input": 112, "hidden": 104}, # 7x multiples
        {"seq_len": 60, "batch": 30, "input": 120, "hidden": 128}, # Near upper bound
        {"seq_len": 64, "batch": 32, "input": 128, "hidden": 112}, # Upper medium boundary
        {"seq_len": 35, "batch": 15, "input": 70, "hidden": 85},  # Odd medium sizes

        # Large shapes (10 configurations) - seq_len ≤ 512, batch ≤ 64, input/hidden ≤ 512
        {"seq_len": 80, "batch": 40, "input": 160, "hidden": 192}, # Large start
        {"seq_len": 100, "batch": 50, "input": 200, "hidden": 240}, # Round large
        {"seq_len": 128, "batch": 32, "input": 256, "hidden": 256}, # Power of 2 large
        {"seq_len": 150, "batch": 45, "input": 300, "hidden": 320}, # 150 sequence
        {"seq_len": 200, "batch": 40, "input": 400, "hidden": 384}, # Long sequence
        {"seq_len": 256, "batch": 48, "input": 512, "hidden": 448}, # Very long sequence
        {"seq_len": 300, "batch": 50, "input": 256, "hidden": 512}, # Max hidden size
        {"seq_len": 400, "batch": 55, "input": 384, "hidden": 480}, # Very large
        {"seq_len": 450, "batch": 60, "input": 450, "hidden": 400}, # Near maximum
        {"seq_len": 512, "batch": 64, "input": 512, "hidden": 512}, # Maximum boundary
    ]

    opt_configs = [
        {"seq_lens": False, "initial_h": False, "initial_c": False},
        {"seq_lens": True, "initial_h": True, "initial_c": True}
    ]

    success_count = 0
    total_count = 0

    for operator_type in operators:
        for direction in directions:
            for activation in activations[operator_type]:
                for i, shape_config in enumerate(shape_configs):
                    for opt_config in opt_configs:
                        total_count += 1

                        # Categorize test size for reporting
                        seq_len = shape_config["seq_len"]
                        batch = shape_config["batch"]
                        input_size = shape_config["input"]
                        hidden = shape_config["hidden"]

                        if seq_len <= 16 and batch <= 8 and input_size <= 32 and hidden <= 32:
                            size_category = "Small"
                        elif seq_len <= 64 and batch <= 32 and input_size <= 128 and hidden <= 128:
                            size_category = "Medium"
                        else:
                            size_category = "Large"

                        print(f"\nTest {total_count}: {operator_type} {direction} {activation} - {size_category}")
                        print(f"  Shape: seq={seq_len}, batch={batch}, input={input_size}, hidden={hidden}")
                        print(f"  Config: {opt_config}")

                        try:
                            # Generate test data
                            from validation_test import generate_test_data_family
                            X, W, R, B, sequence_lens, initial_h, initial_c = generate_test_data_family(
                                operator_type=operator_type,
                                direction=direction,
                                sequence_length=shape_config["seq_len"],
                                batch_size=shape_config["batch"],
                                input_size=shape_config["input"],
                                hidden_size=shape_config["hidden"],
                                include_sequence_lens=opt_config["seq_lens"],
                                include_initial_h=opt_config["initial_h"],
                                include_initial_c=opt_config["initial_c"]
                            )

                            # Test CPU implementation
                            cpu_impl = CPURNNFamilyImplementation(
                                operator_type=operator_type,
                                activation=activation
                            )

                            result = cpu_impl(X, W, R, B, direction=direction,
                                            initial_h=initial_h, initial_c=initial_c,
                                            sequence_lens=sequence_lens)

                            if operator_type == "LSTM":
                                Y, Y_h, Y_c = result
                                print(f"  Shapes: Y={Y.shape}, Y_h={Y_h.shape}, Y_c={Y_c.shape}")
                            else:
                                Y, Y_h = result
                                print(f"  Shapes: Y={Y.shape}, Y_h={Y_h.shape}")

                            success_count += 1
                            print("  ✓ Test passed")

                        except Exception as e:
                            print(f"  ✗ Test failed: {e}")
                            # For large shapes, memory errors might be expected
                            if size_category == "Large" and ("memory" in str(e).lower() or "allocation" in str(e).lower()):
                                print("  (Memory limitation - acceptable for large shapes)")

    print(f"\nRNN Family Comprehensive Tests: {success_count}/{total_count} passed")

    # Calculate success rates by category
    small_tests = total_count // 3
    medium_tests = total_count // 3
    large_tests = total_count - small_tests - medium_tests

    print(f"Test distribution:")
    print(f"  Small shapes (≤16,≤8,≤32,≤32): ~{small_tests} tests")
    print(f"  Medium shapes (≤64,≤32,≤128,≤128): ~{medium_tests} tests")
    print(f"  Large shapes (≤512,≤64,≤512,≤512): ~{large_tests} tests")

    return success_count == total_count

def run_all_tests():
    """Run all tests including RNN family operators"""
    print("Starting comprehensive RNN family test suite...\n")

    try:
        # Run original RNN tests
        test_basic_functionality()
        test_optional_parameters()
        test_bidirectional()
        test_onnx_spec_compliance()

        if not test_model_creation():
            return False

        # Run new comprehensive RNN family tests
        if not test_rnn_family_comprehensive():
            return False

        # Run validation tests for all operators
        print("=== Running RNN Family Validation Tests ===")
        from validation_test import validate_rnn_family_implementation
        validate_rnn_family_implementation()

        print("=" * 60)
        print("=== All RNN Family Tests Completed Successfully ===")
        print("✓ RNN, GRU, and LSTM operators tested")
        print("✓ Multiple shape configurations validated")
        print("✓ Optional parameters handled correctly")
        return True

    except Exception as e:
        print(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
