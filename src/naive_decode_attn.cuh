#pragma once
#include "tree_topology.cuh"
#include <cuda_runtime.h>

void launch_naive_decode_attn(
    const float* Q,        // [num_branches, num_heads, head_dim]
    const float* K,        // [total_kv_len, num_heads, head_dim]
    const float* V,        // [total_kv_len, num_heads, head_dim]
    float* Out,            // [num_branches, num_heads, head_dim]
    const TreeConfig& cfg,
    cudaStream_t stream = 0);
