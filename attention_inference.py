import numpy as np

def split_heads(x, num_heads):
    # Input shape: (batch_size, seq_len, d_model)
    batch_size, seq_len, d_model = x.shape
    
    # Reshape to (batch_size, seq_len, num_heads, depth)
    depth = d_model // num_heads
    x = x.reshape(batch_size, seq_len, num_heads, depth)
    
    # Transpose to (batch_size, num_heads, seq_len, depth)
    return x.transpose(0, 2, 1, 3)

def scaled_dot_product_attention(q, k, v):
    # (batch_size, num_heads, seq_len_q, depth) @ (batch_size, num_heads, depth, seq_len_k)
    matmul_qk = np.matmul(q, k.transpose(0, 1, 3, 2))
    
    # Scale matmul_qk
    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    
    # Softmax is normalized on the last axis (seq_len_k)
    attention_weights = np.exp(scaled_attention_logits) / np.sum(np.exp(scaled_attention_logits), axis=-1, keepdims=True)
    
    # (batch_size, num_heads, seq_len_q, seq_len_k) @ (batch_size, num_heads, seq_len_v, depth_v)
    output = np.matmul(attention_weights, v)
    
    return output

def numpy_multi_head_attention(query, trained_weights, num_heads=2):
    # Unpack the trained weights
    query_weight = trained_weights[0]  # query dense weights
    query_bias = trained_weights[1]    # query dense bias
    key_weight = trained_weights[2]    # key dense weights
    key_bias = trained_weights[3]      # key dense bias
    value_weight = trained_weights[4]  # value dense weights
    value_bias = trained_weights[5]    # value dense bias
    output_weight = trained_weights[6] # output dense weights
    output_bias = trained_weights[7]   # output dense bias
    
    # Reshape input if needed (batch_size, seq_len, d_model)
    batch_size, seq_len_1, seq_len_2, d_model = query.shape
    query = query.reshape(batch_size, seq_len_1 * seq_len_2, d_model)
    
    # Linear transformations
    q = np.dot(query, query_weight) + query_bias
    k = np.dot(query, key_weight) + key_bias
    v = np.dot(query, value_weight) + value_bias
    
    # Split heads
    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)
    
    # Scaled dot-product attention
    scaled_attention = scaled_dot_product_attention(q, k, v)
    
    # Reshape back
    scaled_attention = scaled_attention.transpose(0, 2, 1, 3)
    concat_attention = scaled_attention.reshape(batch_size, seq_len_1 * seq_len_2, -1)
    
    # Final linear transformation
    output = np.dot(concat_attention, output_weight) + output_bias
    
    # Reshape to original dimensions
    output = output.reshape(batch_size, seq_len_1, seq_len_2, d_model)
    
    return output

# Example usage:
if __name__ == "__main__":
    # Load your trained weights here
    # This is just an example - replace with your actual trained weights
    trained_weights = np.load('attention_weights.npy', allow_pickle=True)
    
    # Create sample input
    x = np.random.random((1, 4, 4, 64))  # (batch_size, seq_len_1, seq_len_2, d_model)
    
    # Run inference
    output = numpy_multi_head_attention(x, trained_weights)
    print("Output shape:", output.shape) 