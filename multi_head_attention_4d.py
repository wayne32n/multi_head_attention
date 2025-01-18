import numpy as np

class MultiHeadAttention4D:
    def __init__(self, num_heads, key_dim, value_dim=None):
        """Initialize the Multi-Head Attention layer for 4D tensors.
        
        Args:
            num_heads: Number of attention heads
            key_dim: Size of each attention head for query and key
            value_dim: Size of each attention head for value (if None, = key_dim)
        """
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim else key_dim
        
        # Initialize weights
        self.query_weight = None
        self.key_weight = None
        self.value_weight = None
        self.output_weight = None
        
    def _init_weights(self, query_dim, value_dim):
        """Initialize or verify weights dimensions."""
        if self.query_weight is None:
            scale = np.sqrt(2.0 / (query_dim + self.key_dim))
            # Reshape weights to match einsum operation
            self.query_weight = np.random.normal(0, scale, 
                (query_dim, self.num_heads * self.key_dim))
            self.key_weight = np.random.normal(0, scale, 
                (value_dim, self.num_heads * self.key_dim))
            self.value_weight = np.random.normal(0, scale, 
                (value_dim, self.num_heads * self.value_dim))
            self.output_weight = np.random.normal(0, scale, 
                (self.num_heads * self.value_dim, query_dim))

    def _prepare_4d_input(self, x, patch_size=None):
        """Convert 4D input to sequence format."""
        batch_size, height, width, channels = x.shape
        seq_len = height * width
        # Reshape to (batch_size, seq_len, channels)
        return x.reshape(batch_size, seq_len, channels), seq_len, height, width

    def _restore_4d_output(self, x, original_height, original_width, channels):
        """Restore sequence format back to 4D."""
        batch_size = x.shape[0]
        return x.reshape(batch_size, original_height, original_width, channels)

    def __call__(self, query, value, key=None, mask=None):
        """Forward pass for 4D tensor input."""
        if key is None:
            key = value
            
        # Convert 4D inputs to sequence format
        query_seq, seq_len, height, width = self._prepare_4d_input(query)
        key_seq, _, _, _ = self._prepare_4d_input(key)
        value_seq, _, _, _ = self._prepare_4d_input(value)
        
        # Initialize weights if needed
        self._init_weights(query_seq.shape[-1], value_seq.shape[-1])
        
        # Linear transformations with reshaped weights
        batch_size = query_seq.shape[0]
        q = np.matmul(query_seq, self.query_weight).reshape(
            batch_size, seq_len, self.num_heads, self.key_dim)
        k = np.matmul(key_seq, self.key_weight).reshape(
            batch_size, -1, self.num_heads, self.key_dim)
        v = np.matmul(value_seq, self.value_weight).reshape(
            batch_size, -1, self.num_heads, self.value_dim)
        
        # Transpose for attention calculation
        q = q.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, key_dim)
        k = k.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, key_dim)
        v = v.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, value_dim)
        
        # Calculate attention
        matmul_qk = np.matmul(q, np.transpose(k, (0, 1, 3, 2)))
        scaled_attention_logits = matmul_qk / np.sqrt(self.key_dim)
        
        if mask is not None:
            mask = np.expand_dims(mask, axis=1)
            scaled_attention_logits += (1 - mask) * -1e9
            
        attention_weights = self._softmax(scaled_attention_logits)
        attention_output = np.matmul(attention_weights, v)
        
        # Reshape and apply output transformation
        concat_attention = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.num_heads * self.value_dim)
        
        output = np.matmul(concat_attention, self.output_weight)
        
        # Restore 4D shape
        return self._restore_4d_output(output, height, width, query.shape[-1])
    
    def _softmax(self, x):
        """Apply softmax to the last dimension."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def test_4d_attention():
    # Test parameters
    batch_size = 1
    height = 4
    width = 4
    channels = 64
    num_heads = 2
    key_dim = 64
    value_dim = 64
    
    # Create test inputs
    query = np.random.normal(0, 1, (batch_size, height, width, channels))
    
    # Test position-wise approach
    mha_4d = MultiHeadAttention4D(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim)
    output = mha_4d(query, query)
    print("\nPosition-wise approach:")
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify shapes
    assert output.shape == query.shape, "Output shape doesn't match input shape"
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_4d_attention() 