import numpy as np

class CPURNNFamilyImplementation:
    """CPU-based implementation for RNN family operators (RNN, GRU, LSTM)"""

    def __init__(self, operator_type="RNN", activation="Tanh", activations=None):
        self.operator_type = operator_type
        self.activation = activation
        self.activations = activations or self._get_default_activations()

    def _get_default_activations(self):
        """Get default activations for each operator type"""
        defaults = {
            "RNN": [self.activation],
            "GRU": ["Sigmoid", "Tanh"],
            "LSTM": ["Sigmoid", "Tanh", "Tanh"]
        }
        return defaults.get(self.operator_type, [self.activation])

    def _apply_activation(self, x, activation_name):
        """Apply activation function with improved numerical stability"""
        if activation_name.lower() == "tanh":
            # Clip input to prevent overflow in tanh
            return np.tanh(np.clip(x, -10, 10))
        elif activation_name.lower() == "sigmoid":
            # Use numerically stable sigmoid implementation
            x_clipped = np.clip(x, -500, 500)
            return np.where(x_clipped >= 0,
                           1.0 / (1.0 + np.exp(-x_clipped)),
                           np.exp(x_clipped) / (1.0 + np.exp(x_clipped)))
        elif activation_name.lower() == "relu":
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

    def _rnn_cell_forward(self, x_t, h_prev, W, R, B):
        """Single RNN cell forward pass"""
        hidden_size = h_prev.shape[0]

        # Split bias: [Wb, Rb]
        Wb = B[:hidden_size]
        Rb = B[hidden_size:]

        # RNN computation: h_t = activation(W*x_t + Wb + R*h_prev + Rb)
        linear_input = np.dot(W, x_t) + Wb
        linear_recurrent = np.dot(R, h_prev) + Rb
        h_t = self._apply_activation(linear_input + linear_recurrent, self.activations[0])

        return h_t

    def _gru_cell_forward(self, x_t, h_prev, W, R, B):
        """Single GRU cell forward pass - ONNX compliant implementation"""
        hidden_size = h_prev.shape[0]

        # ONNX GRU uses ZRH order: [update/z, reset/r, new/h]
        # Split weights for gates: [z, r, h] (ONNX standard order)
        W_z, W_r, W_h = np.split(W, 3, axis=0)
        R_z, R_r, R_h = np.split(R, 3, axis=0)

        # ONNX bias layout: [W_bz, W_br, W_bh, R_bz, R_br, R_bh]
        # Split bias: first half is W bias, second half is R bias
        Wb = B[:3 * hidden_size]
        Rb = B[3 * hidden_size:]
        W_bz, W_br, W_bh = np.split(Wb, 3)
        R_bz, R_br, R_bh = np.split(Rb, 3)

        # Update gate (z_t) - controls how much of previous hidden state to keep
        z_t = self._apply_activation(
            np.dot(W_z, x_t) + W_bz + np.dot(R_z, h_prev) + R_bz,
            self.activations[0]  # sigmoid
        )

        # Reset gate (r_t) - controls how much of previous hidden state to use for new content
        r_t = self._apply_activation(
            np.dot(W_r, x_t) + W_br + np.dot(R_r, h_prev) + R_br,
            self.activations[0]  # sigmoid
        )

        # New gate (h_t) - candidate new hidden state
        # Note: r_t * h_prev is applied before matrix multiplication with R_h
        h_tilde = self._apply_activation(
            np.dot(W_h, x_t) + W_bh + np.dot(R_h, r_t * h_prev) + R_bh,
            self.activations[1]  # tanh
        )

        # Final hidden state: interpolate between new content and previous state
        # h_t = (1 - z_t) * h_tilde + z_t * h_prev
        h_t = (1 - z_t) * h_tilde + z_t * h_prev

        return h_t

    def _lstm_cell_forward(self, x_t, h_prev, c_prev, W, R, B):
        """Single LSTM cell forward pass"""
        hidden_size = h_prev.shape[0]

        # Split weights for gates: [input, output, forget, cell] (iofc order)
        W_i, W_o, W_f, W_c = np.split(W, 4, axis=0)
        R_i, R_o, R_f, R_c = np.split(R, 4, axis=0)

        # Split bias: [Wb_i, Wb_o, Wb_f, Wb_c, Rb_i, Rb_o, Rb_f, Rb_c]
        Wb = B[:4 * hidden_size]
        Rb = B[4 * hidden_size:]
        Wb_i, Wb_o, Wb_f, Wb_c = np.split(Wb, 4)
        Rb_i, Rb_o, Rb_f, Rb_c = np.split(Rb, 4)

        # Input gate
        i_t = self._apply_activation(
            np.dot(W_i, x_t) + Wb_i + np.dot(R_i, h_prev) + Rb_i,
            self.activations[0]
        )

        # Forget gate
        f_t = self._apply_activation(
            np.dot(W_f, x_t) + Wb_f + np.dot(R_f, h_prev) + Rb_f,
            self.activations[0]
        )

        # Output gate
        o_t = self._apply_activation(
            np.dot(W_o, x_t) + Wb_o + np.dot(R_o, h_prev) + Rb_o,
            self.activations[0]
        )

        # Cell gate
        c_tilde = self._apply_activation(
            np.dot(W_c, x_t) + Wb_c + np.dot(R_c, h_prev) + Rb_c,
            self.activations[1]
        )

        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde

        # Update hidden state
        h_t = o_t * self._apply_activation(c_t, self.activations[2])

        return h_t, c_t

    def forward_direction(self, X, W, R, B, initial_h=None, initial_c=None, sequence_lens=None):
        """Process sequence in forward direction for any RNN family operator"""
        seq_length, batch_size, input_size = X.shape
        hidden_size = W.shape[0] // self._get_gate_count()

        # Default initial states
        if initial_h is None:
            initial_h = np.zeros((batch_size, hidden_size), dtype=X.dtype)
        if initial_c is None and self.operator_type == "LSTM":
            initial_c = np.zeros((batch_size, hidden_size), dtype=X.dtype)

        # Default sequence lengths
        if sequence_lens is None:
            sequence_lens = np.full(batch_size, seq_length, dtype=np.int32)

        # Output storage
        Y = np.zeros((seq_length, batch_size, hidden_size), dtype=X.dtype)
        h_t = initial_h.copy()
        c_t = initial_c.copy() if initial_c is not None else None

        for t in range(seq_length):
            x_t = X[t]  # [batch_size, input_size]
            h_new = np.zeros_like(h_t)
            c_new = c_t.copy() if c_t is not None else None

            for b in range(batch_size):
                if t < sequence_lens[b]:
                    if self.operator_type == "RNN":
                        h_new[b] = self._rnn_cell_forward(x_t[b], h_t[b], W, R, B)
                    elif self.operator_type == "GRU":
                        h_new[b] = self._gru_cell_forward(x_t[b], h_t[b], W, R, B)
                    elif self.operator_type == "LSTM":
                        h_new[b], c_new[b] = self._lstm_cell_forward(x_t[b], h_t[b], c_t[b], W, R, B)
                else:
                    h_new[b] = h_t[b]
                    if c_new is not None:
                        c_new[b] = c_t[b]

            h_t = h_new
            c_t = c_new
            Y[t] = h_t

        # Mask outputs beyond sequence length
        for b in range(batch_size):
            if sequence_lens[b] < seq_length:
                Y[sequence_lens[b]:, b, :] = 0

        if self.operator_type == "LSTM":
            return Y, h_t, c_t
        else:
            return Y, h_t

    def backward_direction(self, X, W, R, B, initial_h=None, initial_c=None, sequence_lens=None):
        """Process sequence in backward direction for any RNN family operator"""
        seq_length, batch_size, input_size = X.shape
        hidden_size = W.shape[0] // self._get_gate_count()

        # Default initial states
        if initial_h is None:
            initial_h = np.zeros((batch_size, hidden_size), dtype=X.dtype)
        if initial_c is None and self.operator_type == "LSTM":
            initial_c = np.zeros((batch_size, hidden_size), dtype=X.dtype)

        # Default sequence lengths
        if sequence_lens is None:
            sequence_lens = np.full(batch_size, seq_length, dtype=np.int32)

        # Output storage
        Y = np.zeros((seq_length, batch_size, hidden_size), dtype=X.dtype)
        h_t = initial_h.copy()
        c_t = initial_c.copy() if initial_c is not None else None

        # Process in reverse order
        for t in reversed(range(seq_length)):
            x_t = X[t]  # [batch_size, input_size]
            h_new = np.zeros_like(h_t)
            c_new = c_t.copy() if c_t is not None else None

            for b in range(batch_size):
                # For backward direction, check if we're within the valid sequence
                reverse_t = seq_length - 1 - t
                if reverse_t < sequence_lens[b]:
                    if self.operator_type == "RNN":
                        h_new[b] = self._rnn_cell_forward(x_t[b], h_t[b], W, R, B)
                    elif self.operator_type == "GRU":
                        h_new[b] = self._gru_cell_forward(x_t[b], h_t[b], W, R, B)
                    elif self.operator_type == "LSTM":
                        h_new[b], c_new[b] = self._lstm_cell_forward(x_t[b], h_t[b], c_t[b], W, R, B)
                else:
                    h_new[b] = h_t[b]
                    if c_new is not None:
                        c_new[b] = c_t[b]

            h_t = h_new
            c_t = c_new
            Y[t] = h_t

        # Mask outputs beyond sequence length
        for b in range(batch_size):
            if sequence_lens[b] < seq_length:
                Y[seq_length - sequence_lens[b]:, b, :] = 0

        if self.operator_type == "LSTM":
            return Y, h_t, c_t
        else:
            return Y, h_t

    def _get_gate_count(self):
        """Get number of gates for the operator"""
        gate_counts = {"RNN": 1, "GRU": 3, "LSTM": 4}
        return gate_counts[self.operator_type]

    def __call__(self, X, W, R, B, direction="forward", initial_h=None, initial_c=None, sequence_lens=None):
        """Main forward pass for RNN family operators"""
        seq_length, batch_size, input_size = X.shape
        hidden_size = W.shape[-2] // self._get_gate_count()

        if direction == "forward":
            # Single direction forward
            W_single = W[0] if W.ndim == 3 else W
            R_single = R[0] if R.ndim == 3 else R
            B_single = B[0] if B.ndim == 2 else B

            initial_h_single = None
            if initial_h is not None:
                initial_h_single = initial_h[0] if initial_h.ndim == 3 else initial_h

            initial_c_single = None
            if initial_c is not None:
                initial_c_single = initial_c[0] if initial_c.ndim == 3 else initial_c

            result = self.forward_direction(X, W_single, R_single, B_single,
                                          initial_h_single, initial_c_single, sequence_lens)

            if self.operator_type == "LSTM":
                Y, Y_h, Y_c = result
                Y = np.expand_dims(Y, axis=1)  # Add direction dimension
                Y_h = np.expand_dims(Y_h, axis=0)
                Y_c = np.expand_dims(Y_c, axis=0)
                return Y, Y_h, Y_c
            else:
                Y, Y_h = result
                Y = np.expand_dims(Y, axis=1)  # Add direction dimension
                Y_h = np.expand_dims(Y_h, axis=0)
                return Y, Y_h

        elif direction == "bidirectional":
            # Bidirectional processing
            num_directions = W.shape[0] if W.ndim == 3 else 1
            if num_directions != 2:
                raise ValueError(f"Bidirectional processing requires 2 directions in weight matrices, got {num_directions}")

            # Forward direction
            W_forward = W[0]
            R_forward = R[0]
            B_forward = B[0]

            initial_h_forward = None
            if initial_h is not None:
                initial_h_forward = initial_h[0] if initial_h.ndim == 3 else initial_h

            initial_c_forward = None
            if initial_c is not None:
                initial_c_forward = initial_c[0] if initial_c.ndim == 3 else initial_c

            result_forward = self.forward_direction(X, W_forward, R_forward, B_forward,
                                                  initial_h_forward, initial_c_forward, sequence_lens)

            # Backward direction
            W_backward = W[1]
            R_backward = R[1]
            B_backward = B[1]

            initial_h_backward = None
            if initial_h is not None:
                initial_h_backward = initial_h[1] if initial_h.ndim == 3 else initial_h

            initial_c_backward = None
            if initial_c is not None:
                initial_c_backward = initial_c[1] if initial_c.ndim == 3 else initial_c

            result_backward = self.backward_direction(X, W_backward, R_backward, B_backward,
                                                    initial_h_backward, initial_c_backward, sequence_lens)

            if self.operator_type == "LSTM":
                Y_forward, Y_h_forward, Y_c_forward = result_forward
                Y_backward, Y_h_backward, Y_c_backward = result_backward

                # Concatenate outputs along direction dimension
                Y = np.stack([Y_forward, Y_backward], axis=1)  # [seq, 2, batch, hidden]
                Y_h = np.stack([Y_h_forward, Y_h_backward], axis=0)  # [2, batch, hidden]
                Y_c = np.stack([Y_c_forward, Y_c_backward], axis=0)  # [2, batch, hidden]

                return Y, Y_h, Y_c
            else:
                Y_forward, Y_h_forward = result_forward
                Y_backward, Y_h_backward = result_backward

                # Concatenate outputs along direction dimension
                Y = np.stack([Y_forward, Y_backward], axis=1)  # [seq, 2, batch, hidden]
                Y_h = np.stack([Y_h_forward, Y_h_backward], axis=0)  # [2, batch, hidden]

                return Y, Y_h

        else:
            raise ValueError(f"Unsupported direction: {direction}")

# Keep original class for backward compatibility
class CPURNNImplementation(CPURNNFamilyImplementation):
    def __init__(self, activation="Tanh"):
        super().__init__(operator_type="RNN", activation=activation)
