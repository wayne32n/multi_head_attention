#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include <stdint.h>

typedef struct {
    int num_heads;
    int key_dim;
    int value_dim;
    
    // Weights
    float* query_weight;  // shape: [query_dim, num_heads, key_dim]
    float* key_weight;    // shape: [value_dim, num_heads, key_dim]
    float* value_weight;  // shape: [value_dim, num_heads, value_dim]
    float* output_weight; // shape: [num_heads * value_dim, query_dim]
    
    // Dimensions for weights
    int query_dim;
    int value_dim_in;
} MultiHeadAttention;

// Initialize the Multi-Head Attention layer
MultiHeadAttention* mha_create(int num_heads, int key_dim, int value_dim);

// Free the Multi-Head Attention layer
void mha_free(MultiHeadAttention* mha);

// Initialize weights with given dimensions
void mha_init_weights(MultiHeadAttention* mha, int query_dim, int value_dim);

// Forward pass
void mha_forward(MultiHeadAttention* mha,
                const float* query,  // shape: [batch_size, query_seq_len, query_dim]
                const float* value,  // shape: [batch_size, value_seq_len, value_dim]
                const float* key,    // shape: [batch_size, key_seq_len, key_dim]
                const float* mask,   // shape: [batch_size, query_seq_len, key_seq_len]
                int batch_size,
                int query_seq_len,
                int value_seq_len,
                float* output);      // shape: [batch_size, query_seq_len, query_dim]

#endif // MULTI_HEAD_ATTENTION_H 