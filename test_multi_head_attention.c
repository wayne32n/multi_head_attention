#include "multi_head_attention.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void test_mha_create() {
    int num_heads = 2;
    int key_dim = 4;
    int value_dim = 4;
    
    MultiHeadAttention* mha = mha_create(num_heads, key_dim, value_dim);
    assert(mha != NULL);
    assert(mha->num_heads == num_heads);
    assert(mha->key_dim == key_dim);
    assert(mha->value_dim == value_dim);
    
    mha_free(mha);
    printf("test_mha_create: PASSED\n");
}

void test_mha_forward() {
    // Test parameters
    int batch_size = 2;
    int query_seq_len = 4;
    int key_seq_len = 6;
    int d_model = 8;
    int num_heads = 2;
    int key_dim = 4;
    
    // Create inputs
    float* query = (float*)malloc(batch_size * query_seq_len * d_model * sizeof(float));
    float* value = (float*)malloc(batch_size * key_seq_len * d_model * sizeof(float));
    float* mask = (float*)malloc(batch_size * query_seq_len * key_seq_len * sizeof(float));
    float* output = (float*)malloc(batch_size * query_seq_len * d_model * sizeof(float));
    
    // Initialize inputs with some test values
    for (int i = 0; i < batch_size * query_seq_len * d_model; i++) query[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < batch_size * key_seq_len * d_model; i++) value[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < batch_size * query_seq_len * key_seq_len; i++) mask[i] = 1.0f;
    
    // Create and initialize MHA
    MultiHeadAttention* mha = mha_create(num_heads, key_dim, 0);
    assert(mha != NULL);
    
    // Run forward pass
    mha_forward(mha, query, value, NULL, mask, batch_size, query_seq_len, key_seq_len, output);
    
    // TODO: Add more specific output checks
    
    // Cleanup
    free(query);
    free(value);
    free(mask);
    free(output);
    mha_free(mha);
    
    printf("test_mha_forward: PASSED\n");
}

int main() {
    test_mha_create();
    test_mha_forward();
    printf("All tests passed!\n");
    return 0;
} 