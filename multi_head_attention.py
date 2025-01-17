import numpy as np

class MultiHeadAttention:
    def __init__(self, num_heads, key_dim, value_dim=None):
        """Initialize the Multi-Head Attention layer.
        
        Args:
            num_heads: Number of attention heads
            key_dim: Size of each attention head for query and key
            value_dim: Size of each attention head for value (if None, = key_dim)
        """
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim else key_dim
        
        # Initialize weights (we'll use random for now, but these should be loaded from trained model)
        self.query_weight = None
        self.key_weight = None
        self.value_weight = None
        self.output_weight = None
        
    def _init_weights(self, query_dim, value_dim):
        """Initialize or verify weights dimensions."""
        if self.query_weight is None:
            # Initialize weights with Xavier/Glorot initialization
            scale = np.sqrt(2.0 / (query_dim + self.key_dim))
            self.query_weight = np.random.normal(0, scale, 
                (query_dim, self.num_heads, self.key_dim))
            self.key_weight = np.random.normal(0, scale, 
                (value_dim, self.num_heads, self.key_dim))
            self.value_weight = np.random.normal(0, scale, 
                (value_dim, self.num_heads, self.value_dim))
            self.output_weight = np.random.normal(0, scale, 
                (self.num_heads * self.value_dim, query_dim))

    def _scaled_dot_product_attention(self, query, key, value, mask=None):
        """Calculate scaled dot-product attention.
        
        Args:
            query: shape (batch_size, num_heads, query_seq_len, key_dim)
            key: shape (batch_size, num_heads, key_seq_len, key_dim)
            value: shape (batch_size, num_heads, value_seq_len, value_dim)
            mask: Optional mask shape (batch_size, query_seq_len, key_seq_len)
        
        Returns:
            attention_output: shape (batch_size, num_heads, query_seq_len, value_dim)
            attention_weights: shape (batch_size, num_heads, query_seq_len, key_seq_len)
        """
        # Calculate attention scores
        matmul_qk = np.matmul(query, np.transpose(key, (0, 1, 3, 2)))
        
        # Scale matmul_qk
        dk = np.float32(self.key_dim)
        scaled_attention_logits = matmul_qk / np.sqrt(dk)
        
        # Apply mask if provided
        if mask is not None:
            # Add extra dimensions for broadcasting
            mask = np.expand_dims(mask, axis=1)  # For num_heads dimension
            scaled_attention_logits += (1 - mask) * -1e9
        
        # Softmax is applied to the last axis
        attention_weights = self._softmax(scaled_attention_logits)
        
        # Calculate output
        output = np.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def _softmax(self, x):
        """Apply softmax to the last dimension."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def __call__(self, query, value, key=None, mask=None):
        """Forward pass for the Multi-Head Attention layer.
        
        Args:
            query: shape (batch_size, query_seq_len, query_dim)
            value: shape (batch_size, value_seq_len, value_dim)
            key: Optional, shape (batch_size, key_seq_len, key_dim)
            mask: Optional mask shape (batch_size, query_seq_len, key_seq_len)
            
        Returns:
            output: shape (batch_size, query_seq_len, query_dim)
        """
        if key is None:
            key = value
            
        batch_size = query.shape[0]
        query_seq_len = query.shape[1]
        value_seq_len = value.shape[1]
        
        # Initialize weights if needed
        self._init_weights(query.shape[-1], value.shape[-1])
        
        # Linear transformations and reshape
        # Shape: (batch_size, seq_len, num_heads, dim)
        q = np.einsum('bsd,dhk->bhsk', query, self.query_weight)
        k = np.einsum('bsd,dhk->bhsk', key, self.key_weight)
        v = np.einsum('bsd,dhv->bhsv', value, self.value_weight)
        
        # Calculate attention
        scaled_attention, attention_weights = self._scaled_dot_product_attention(
            q, k, v, mask)
        
        # Reshape and apply output transformation
        # Concatenate heads and apply output transformation
        concat_attention = scaled_attention.transpose(0, 2, 1, 3).reshape(
            batch_size, query_seq_len, self.num_heads * self.value_dim)
        
        output = np.matmul(concat_attention, self.output_weight)
        
        return output


def test_multi_head_attention():
    # Test parameters
    batch_size = 2
    query_seq_len = 4
    key_seq_len = 6
    d_model = 8
    num_heads = 2
    key_dim = 4
    
    # Create test inputs
    query = np.random.normal(0, 1, (batch_size, query_seq_len, d_model))
    value = np.random.normal(0, 1, (batch_size, key_seq_len, d_model))
    
    # Create mask (optional)
    mask = np.ones((batch_size, query_seq_len, key_seq_len))
    
    # Initialize layer
    mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
    
    # Test forward pass
    output = mha(query, value, mask=mask)
    
    print("Input shapes:")
    print(f"Query shape: {query.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Output shape: {output.shape}")
    
    # Expected output shape: (batch_size, query_seq_len, d_model)
    assert output.shape == (batch_size, query_seq_len, d_model)
    
    return output

# Run the test
if __name__ == "__main__":
    output = test_multi_head_attention()
    print("\nTest passed successfully!") 