import numpy as np
import onnxruntime as ort
from create_rnn_model import create_rnn_onnx_model
from cpu_rnn_implementation import CPURNNImplementation, CPURNNFamilyImplementation
import tempfile
import os

def cosine_similarity(a, b):
    """Calculate cosine similarity between two arrays"""
    a_flat = a.flatten()
    b_flat = b.flatten()

    dot_product = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)

def generate_test_data_family(operator_type="RNN", direction="forward", sequence_length=3, batch_size=2,
                             input_size=4, hidden_size=5, include_sequence_lens=False,
                             include_initial_h=False, include_initial_c=False):
    """Generate test data for RNN family operators"""

    gate_counts = {"RNN": 1, "GRU": 3, "LSTM": 4}
    gates = gate_counts[operator_type]

    num_directions = 2 if direction == "bidirectional" else 1

    np.random.seed(42)

    # Generate input data
    X = np.random.randn(sequence_length, batch_size, input_size).astype(np.float32)
    W = np.random.randn(num_directions, gates * hidden_size, input_size).astype(np.float32) * 0.1
    R = np.random.randn(num_directions, gates * hidden_size, hidden_size).astype(np.float32) * 0.1
    B = np.random.randn(num_directions, 2 * gates * hidden_size).astype(np.float32) * 0.1

    # Generate optional inputs
    sequence_lens = None
    if include_sequence_lens:
        sequence_lens = np.random.randint(1, sequence_length + 1, size=batch_size).astype(np.int32)

    initial_h = None
    if include_initial_h:
        initial_h = np.random.randn(num_directions, batch_size, hidden_size).astype(np.float32) * 0.1

    initial_c = None
    if include_initial_c and operator_type == "LSTM":
        initial_c = np.random.randn(num_directions, batch_size, hidden_size).astype(np.float32) * 0.1

    return X, W, R, B, sequence_lens, initial_h, initial_c

def run_onnx_family_model(model_path, X, W, R, B, sequence_lens=None, initial_h=None, initial_c=None):
    """Run ONNX model for RNN family operators"""
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_names = [input.name for input in session.get_inputs()]

        inputs = {'X': X, 'W': W, 'R': R, 'B': B}

        if 'sequence_lens' in input_names and sequence_lens is not None:
            inputs['sequence_lens'] = sequence_lens
        if 'initial_h' in input_names and initial_h is not None:
            inputs['initial_h'] = initial_h
        if 'initial_c' in input_names and initial_c is not None:
            inputs['initial_c'] = initial_c

        outputs = session.run(None, inputs)

        if len(outputs) == 3:  # LSTM
            return outputs[0], outputs[1], outputs[2]  # Y, Y_h, Y_c
        else:  # RNN, GRU
            return outputs[0], outputs[1], None  # Y, Y_h, None

    except Exception as e:
        print(f"ONNX Runtime error: {e}")
        return None, None, None

def validate_rnn_family_implementation():
    """Validation function for all RNN family operators"""
    print("=== RNN Family Implementation Validation ===")

    operators = ["RNN", "GRU", "LSTM"]

    for operator_type in operators:
        print(f"\n=== Testing {operator_type} Operator ===")

        test_configs = [
            {"direction": "forward", "activation": "Tanh", "sequence_lens": False, "initial_h": False, "initial_c": False},
            {"direction": "forward", "activation": "Tanh", "sequence_lens": True, "initial_h": True, "initial_c": True},
        ]

        if operator_type == "RNN":
            test_configs.append({"direction": "forward", "activation": "Relu", "sequence_lens": False, "initial_h": False, "initial_c": False})

        for i, config in enumerate(test_configs, 1):
            print(f"\nTest {i}: {operator_type} - {config}")

            try:
                # Fixed parameters for consistent testing
                sequence_length = 3
                batch_size = 2
                input_size = 4
                hidden_size = 5

                # Generate test data with fixed parameters
                X, W, R, B, sequence_lens, initial_h, initial_c = generate_test_data_family(
                    operator_type=operator_type,
                    direction=config["direction"],
                    sequence_length=sequence_length,
                    batch_size=batch_size,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    include_sequence_lens=config["sequence_lens"],
                    include_initial_h=config["initial_h"],
                    include_initial_c=config["initial_c"] and operator_type == "LSTM"
                )

                # Create ONNX model with the same fixed parameters
                from create_rnn_model import create_rnn_family_onnx_model
                model = create_rnn_family_onnx_model(
                    operator_type=operator_type,
                    sequence_length=sequence_length,
                    batch_size=batch_size,
                    input_size=input_size,
                    hidden_size=hidden_size,  # Use the fixed hidden_size
                    direction=config["direction"],
                    activation=config.get("activation", "Tanh"),
                    include_sequence_lens=config["sequence_lens"],
                    include_initial_h=config["initial_h"],
                    include_initial_c=config["initial_c"] and operator_type == "LSTM"
                )

                # Test with temporary file
                with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
                    model_path = tmp_file.name

                try:
                    import onnx
                    onnx.save(model, model_path)

                    # Run ONNX model
                    onnx_Y, onnx_Y_h, onnx_Y_c = run_onnx_family_model(
                        model_path, X, W, R, B, sequence_lens, initial_h, initial_c
                    )

                    if onnx_Y is None:
                        print("  ONNX Runtime failed, skipping comparison")
                        continue

                    # Run CPU implementation
                    cpu_impl = CPURNNFamilyImplementation(
                        operator_type=operator_type,
                        activation=config.get("activation", "Tanh")
                    )

                    cpu_result = cpu_impl(X, W, R, B, direction=config["direction"],
                                        initial_h=initial_h, initial_c=initial_c,
                                        sequence_lens=sequence_lens)

                    if operator_type == "LSTM":
                        cpu_Y, cpu_Y_h, cpu_Y_c = cpu_result
                    else:
                        cpu_Y, cpu_Y_h = cpu_result
                        cpu_Y_c = None

                    # Compare results
                    y_close = np.allclose(onnx_Y, cpu_Y, rtol=1e-4, atol=1e-4)
                    y_h_close = np.allclose(onnx_Y_h, cpu_Y_h, rtol=1e-4, atol=1e-4)
                    y_cosine = cosine_similarity(onnx_Y, cpu_Y)
                    y_h_cosine = cosine_similarity(onnx_Y_h, cpu_Y_h)

                    print(f"  Y comparison: close={y_close}, cosine={y_cosine:.6f}")
                    print(f"  Y_h comparison: close={y_h_close}, cosine={y_h_cosine:.6f}")

                    y_c_close = True
                    y_c_cosine = 1.0
                    if operator_type == "LSTM" and onnx_Y_c is not None and cpu_Y_c is not None:
                        y_c_close = np.allclose(onnx_Y_c, cpu_Y_c, rtol=1e-4, atol=1e-4)
                        y_c_cosine = cosine_similarity(onnx_Y_c, cpu_Y_c)
                        print(f"  Y_c comparison: close={y_c_close}, cosine={y_c_cosine:.6f}")

                    # Final validation
                    success = y_close and y_h_close and y_cosine > 0.99 and y_h_cosine > 0.99
                    if operator_type == "LSTM" and onnx_Y_c is not None:
                        success = success and y_c_close and y_c_cosine > 0.99

                    if success:
                        print("  ✓ 结果一致")
                    else:
                        print("  ✗ 校验失败")

                except Exception as e:
                    print(f"  Error during validation: {e}")
                    print("  ✗ 校验失败")

                finally:
                    if os.path.exists(model_path):
                        os.unlink(model_path)

            except Exception as e:
                print(f"  Error in test setup: {e}")
                print("  ✗ 校验失败")

# Keep original function for backward compatibility
def validate_rnn_implementation():
    validate_rnn_family_implementation()

if __name__ == "__main__":
    validate_rnn_implementation()
