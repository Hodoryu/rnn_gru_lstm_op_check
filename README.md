# RNN Family ONNX Model Generator & CPU Implementation

A comprehensive Python toolkit for creating, validating, and testing RNN family operators (RNN, GRU, LSTM) with ONNX model generation and CPU reference implementations. This project provides complete ONNX specification compliance testing and validation framework.

## 🚀 Project Overview

This toolkit provides a complete solution for RNN family operator development and validation:

- **ONNX Model Generation**: Create compliant ONNX models for RNN, GRU, and LSTM operators
- **CPU Reference Implementation**: High-performance CPU implementations following ONNX specifications
- **Comprehensive Validation**: Extensive testing framework with precision validation
- **ONNX Runtime Integration**: Direct comparison with ONNX Runtime execution

## 🌟 Project Features

### Core Capabilities
- **Complete RNN Family Support**: RNN, GRU, and LSTM operators with all variants
- **ONNX Specification Compliance**: Strict adherence to ONNX operator specifications
- **Bidirectional Processing**: Full support for forward and bidirectional RNN processing
- **Optional Parameters**: Complete support for sequence_lens, initial_h, and initial_c
- **Multiple Activations**: Support for Tanh, ReLU, and Sigmoid activations
- **Precision Validation**: Cosine similarity and numerical precision testing

### Advanced Features
- **Automatic Model Generation**: Batch creation of test models with various configurations
- **CPU Reference Implementation**: Optimized CPU implementations for validation
- **ONNX Runtime Validation**: Direct comparison with ONNX Runtime execution
- **Comprehensive Test Suite**: 15+ shape configurations and multiple test scenarios
- **Debug and Logging**: Detailed execution logging and error reporting

## 🏗️ Architecture Overview

```
project/
├── create_rnn_model.py        # ONNX model generation and creation
├── cpu_rnn_implementation.py  # CPU reference implementations
├── validation_test.py         # Comprehensive validation framework
├── run_all_tests.py          # Test orchestrator and runner
└── README.md                 # This documentation
```

## 🛠️ Implementation Details

### Supported Operators

| Operator | Gates | Default Activations | Bidirectional | Optional Inputs |
|----------|-------|-------------------|---------------|-----------------|
| **RNN**  | 1     | Tanh              | ✅            | sequence_lens, initial_h |
| **GRU**  | 3     | Sigmoid, Tanh     | ✅            | sequence_lens, initial_h |
| **LSTM** | 4     | Sigmoid, Tanh, Tanh | ✅          | sequence_lens, initial_h, initial_c |

### Key Components

#### 1. ONNX Model Generator (`create_rnn_model.py`)
- **`create_rnn_family_onnx_model()`**: Creates ONNX models for any RNN family operator
- **`save_rnn_family_model()`**: Saves models with proper filename generation
- **`create_test_models()`**: Batch creates comprehensive test model suite

#### 2. CPU Implementation (`cpu_rnn_implementation.py`)
- **`CPURNNFamilyImplementation`**: Unified CPU implementation for all RNN operators
- **Cell-level implementations**: Optimized forward pass for each operator type
- **Bidirectional support**: Efficient forward and backward processing
- **ONNX compliance**: Strict adherence to ONNX operator specifications

#### 3. Validation Framework (`validation_test.py`)
- **`validate_rnn_family_implementation()`**: Comprehensive validation against ONNX Runtime
- **Precision testing**: Cosine similarity and numerical precision validation
- **Shape testing**: Multiple tensor size configurations
- **Error handling**: Robust error reporting and debugging

#### 4. Test Runner (`run_all_tests.py`)
- **`test_rnn_family_comprehensive()`**: Complete test suite execution
- **Performance benchmarking**: Timing and throughput analysis
- **Automated validation**: Batch testing across all configurations

## 📦 Installation & Setup

### Prerequisites

```bash
# Python 3.7+ required
python --version

# Verify pip installation
pip --version
```

### Install Dependencies

```bash
# Install required packages
pip install numpy>=1.19.0
pip install onnx>=1.12.0
pip install onnxruntime>=1.12.0

# Optional: Install development tools
pip install pytest
pip install black
pip install flake8
```

### Verify Installation

```python
# Test basic imports
import numpy as np
import onnx
import onnxruntime as ort

# Test project modules
from create_rnn_model import create_rnn_family_onnx_model
from cpu_rnn_implementation import CPURNNFamilyImplementation
from validation_test import validate_rnn_family_implementation

print("✅ All dependencies installed successfully!")
```

## 🚀 Usage Examples

### 1. Create ONNX Models

```python
from create_rnn_model import create_rnn_family_onnx_model, save_rnn_family_model

# Create a basic RNN model
rnn_model = create_rnn_family_onnx_model(
    operator_type="RNN",
    sequence_length=10,
    batch_size=4,
    input_size=8,
    hidden_size=16,
    direction="forward",
    activation="Tanh"
)

# Save model to file
save_rnn_family_model(
    operator_type="RNN",
    sequence_length=10,
    batch_size=4,
    input_size=8,
    hidden_size=16,
    include_sequence_lens=True,
    include_initial_h=True
)

# Create comprehensive test models
from create_rnn_model import create_test_models
models = create_test_models()
print(f"Created {len(models)} test models")
```

### 2. CPU Implementation Usage

```python
from cpu_rnn_implementation import CPURNNFamilyImplementation
import numpy as np

# Create test data
seq_length, batch_size, input_size, hidden_size = 5, 3, 4, 6
X = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
W = np.random.randn(1, hidden_size, input_size).astype(np.float32) * 0.1
R = np.random.randn(1, hidden_size, hidden_size).astype(np.float32) * 0.1
B = np.random.randn(1, 2 * hidden_size).astype(np.float32) * 0.1

# Test RNN
rnn_impl = CPURNNFamilyImplementation(operator_type="RNN", activation="Tanh")
Y, Y_h = rnn_impl(X, W, R, B, direction="forward")
print(f"RNN Output: Y={Y.shape}, Y_h={Y_h.shape}")

# Test LSTM with optional parameters
lstm_impl = CPURNNFamilyImplementation(operator_type="LSTM", activation="Tanh")
initial_h = np.random.randn(1, batch_size, hidden_size).astype(np.float32)
initial_c = np.random.randn(1, batch_size, hidden_size).astype(np.float32)
sequence_lens = np.array([5, 3, 4], dtype=np.int32)

# LSTM requires 4 gates, so adjust weight dimensions
W_lstm = np.random.randn(1, 4 * hidden_size, input_size).astype(np.float32) * 0.1
R_lstm = np.random.randn(1, 4 * hidden_size, hidden_size).astype(np.float32) * 0.1
B_lstm = np.random.randn(1, 8 * hidden_size).astype(np.float32) * 0.1

Y, Y_h, Y_c = lstm_impl(
    X, W_lstm, R_lstm, B_lstm,
    direction="forward",
    initial_h=initial_h,
    initial_c=initial_c,
    sequence_lens=sequence_lens
)
print(f"LSTM Output: Y={Y.shape}, Y_h={Y_h.shape}, Y_c={Y_c.shape}")
```

### 3. Validation Testing

```python
from validation_test import validate_rnn_family_implementation

# Run comprehensive validation
print("Running RNN Family Validation...")
validate_rnn_family_implementation()

# Expected output:
# === RNN Family Implementation Validation ===
# Testing RNN operator...
# Testing GRU operator...  
# Testing LSTM operator...
# ✓ All validations passed
```

### 4. Complete Test Suite

```python
from run_all_tests import main

# Run all tests
success = main()
if success:
    print("✅ All tests passed successfully!")
else:
    print("❌ Some tests failed - check logs for details")
```

## 🧪 Running Tests

### Complete Test Suite

```bash
# Run all tests with comprehensive validation
python run_all_tests.py

# Expected output:
# === Testing Basic RNN Functionality ===
# ✓ Basic RNN test passed
# === Testing Optional Parameters ===
# ✓ Optional parameters test passed
# === Testing Bidirectional RNN ===
# ✓ Bidirectional RNN test passed
# === Testing ONNX Model Creation ===
# ✓ All ONNX models created successfully
# === Testing ONNX Specification Compliance ===
# ✓ ONNX specification compliance verified
# === Running RNN Family Comprehensive Tests ===
# ✓ RNN: 15/15 configurations passed
# ✓ GRU: 15/15 configurations passed  
# ✓ LSTM: 15/15 configurations passed
# === Running RNN Family Validation Tests ===
# ✓ All validation tests passed
# === All RNN Family Tests Completed Successfully ===
```

### Individual Test Components

```bash
# Test ONNX model creation only
python -c "from create_rnn_model import create_test_models; create_test_models()"

# Test CPU implementation only
python -c "from cpu_rnn_implementation import CPURNNFamilyImplementation; print('CPU implementation ready')"

# Test validation framework only
python -c "from validation_test import validate_rnn_family_implementation; validate_rnn_family_implementation()"
```

### Custom Testing

```python
# Test specific operator configuration
from cpu_rnn_implementation import CPURNNFamilyImplementation
from validation_test import generate_test_data_family
import numpy as np

# Generate test data for specific configuration
X, W, R, B, sequence_lens, initial_h, initial_c = generate_test_data_family(
    operator_type="GRU",
    direction="bidirectional",
    sequence_length=20,
    batch_size=8,
    input_size=32,
    hidden_size=64,
    include_sequence_lens=True,
    include_initial_h=True,
    include_initial_c=False
)

# Test CPU implementation
gru_impl = CPURNNFamilyImplementation(operator_type="GRU", activation="Tanh")
result = gru_impl(
    X, W, R, B,
    direction="bidirectional",
    initial_h=initial_h,
    sequence_lens=sequence_lens
)

Y, Y_h = result
print(f"GRU Bidirectional Output: Y={Y.shape}, Y_h={Y_h.shape}")
```

## 📊 Test Results & Validation

### Validation Metrics

The validation framework uses multiple precision metrics:

- **Cosine Similarity**: > 0.99 threshold for output similarity
- **NumPy allclose**: rtol=1e-4, atol=1e-4 for numerical precision
- **Shape Consistency**: Exact shape matching between implementations
- **ONNX Runtime Comparison**: Direct validation against ONNX Runtime execution

### Expected Test Results

```
=== RNN Family Implementation Validation ===

Testing RNN operator...
  Config 1: Forward Tanh with ONNX defaults
    Y cosine similarity: 0.9999, close: True
    Y_h cosine similarity: 0.9999, close: True
    ✓ 结果一致
  Config 2: Forward with optional inputs
    Y cosine similarity: 0.9999, close: True
    Y_h cosine similarity: 0.9999, close: True
    ✓ 结果一致

Testing GRU operator...
  Config 1: Forward Tanh with ONNX defaults
    Y cosine similarity: 0.9999, close: True
    Y_h cosine similarity: 0.9999, close: True
    ✓ 结果一致

Testing LSTM operator...
  Config 1: Forward Tanh with ONNX defaults
    Y cosine similarity: 0.9999, close: True
    Y_h cosine similarity: 0.9999, close: True
    Y_c cosine similarity: 0.9999, close: True
    ✓ 结果一致
  Config 2: LSTM with all optional inputs including initial_c
    Y cosine similarity: 0.9999, close: True
    Y_h cosine similarity: 0.9999, close: True
    Y_c cosine similarity: 0.9999, close: True
    ✓ 结果一致
```

## 🔧 Troubleshooting

### Common Issues

**1. ONNX Runtime Import Error**
```bash
# Install ONNX Runtime
pip install onnxruntime

# For GPU support (optional)
pip install onnxruntime-gpu
```

**2. Model Creation Failures**
```python
# Check ONNX version compatibility
import onnx
print(f"ONNX version: {onnx.__version__}")

# Ensure compatible IR version and opset
# The code uses IR version 7 and opset version 11 for compatibility
```

**3. Validation Precision Issues**
```python
# Check input data ranges
X = np.clip(X, -5, 5)  # Prevent extreme values
W = W * 0.1  # Scale weights appropriately
R = R * 0.1
B = B * 0.1

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

**4. Shape Mismatch Errors**
```python
# Verify weight dimensions for each operator:
# RNN: W=[1, hidden_size, input_size], R=[1, hidden_size, hidden_size], B=[1, 2*hidden_size]
# GRU: W=[1, 3*hidden_size, input_size], R=[1, 3*hidden_size, hidden_size], B=[1, 6*hidden_size]  
# LSTM: W=[1, 4*hidden_size, input_size], R=[1, 4*hidden_size, hidden_size], B=[1, 8*hidden_size]
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging for validation
from validation_test import validate_rnn_family_implementation
validate_rnn_family_implementation()

# Debug output will show:
# - Model creation details
# - ONNX Runtime execution logs  
# - Precision comparison results
# - Error details for failed tests
```

## 🤝 Contributing

### Development Guidelines

1. **Code Style**: Follow PEP 8 conventions
2. **Testing**: Add tests for new features in appropriate test files
3. **Documentation**: Update docstrings and README for new functionality
4. **Validation**: Ensure ONNX specification compliance
5. **Backward Compatibility**: Maintain compatibility with existing APIs

### Adding New Operators

```python
# 1. Add operator configuration in create_rnn_model.py
operator_configs = {
    "RNN": {"gates": 1, "default_activations": ["Tanh"]},
    "GRU": {"gates": 3, "default_activations": ["Sigmoid", "Tanh"]},
    "LSTM": {"gates": 4, "default_activations": ["Sigmoid", "Tanh", "Tanh"]},
    "NEW_OP": {"gates": N, "default_activations": [...]}  # Add here
}

# 2. Implement cell forward pass in cpu_rnn_implementation.py
def _new_op_cell_forward(self, x_t, h_prev, W, R, B):
    # Implement new operator logic
    pass

# 3. Add validation tests in validation_test.py
# 4. Update test configurations in run_all_tests.py
```

## 📄 License

This project is based on ONNX specifications and follows MIT License terms.

## 🔗 References

- [ONNX RNN Operator Specification](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RNN)
- [ONNX GRU Operator Specification](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GRU)
- [ONNX LSTM Operator Specification](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LSTM)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [NumPy Documentation](https://numpy.org/doc/)

---

**🎯 Production Ready**: This toolkit provides comprehensive RNN family operator implementations with strict ONNX compliance, extensive validation, and robust error handling suitable for production AI/ML development workflows.

## 📋 Quick Start Checklist

- [ ] Install Python 3.7+
- [ ] Install dependencies: `pip install numpy onnx onnxruntime`
- [ ] Run basic test: `python -c "from create_rnn_model import create_test_models; create_test_models()"`
- [ ] Run validation: `python validation_test.py`
- [ ] Run complete suite: `python run_all_tests.py`
- [ ] Check all tests pass: Look for "✅ All RNN Family Tests Completed Successfully"

**Ready to use!** 🚀
