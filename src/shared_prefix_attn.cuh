#pragma once
#include "tree_topology.cuh"
#include "partial_softmax.cuh"
#include <cuda_runtime.h>

// Launches the shared prefix attention pass.
// Computes partial softmax state over K[0..shared_len) for each (query, head).
// All branches share the same query during decode (one token per branch per step),
// but each branch has its own query vector.
//
// Output: partial states written to shared_states[branch, head].
void launch_shared_prefix_attn(
    const float* Q,                    // [num_branches, num_heads, head_dim]
    const float* K,                    // [total_kv_len, num_heads, head_dim]
    const float* V,                    // [total_kv_len, num_heads, head_dim]
    PartialSoftmaxState* shared_states, // [num_branches, num_heads]
    const TreeConfig& cfg,
    cudaStream_t stream = 0);
