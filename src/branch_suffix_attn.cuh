#pragma once
#include "tree_topology.cuh"
#include "partial_softmax.cuh"
#include <cuda_runtime.h>

// Launches the per-branch unique suffix attention pass.
// For branch b, attends over K[branch_offset(b)..branch_offset(b)+unique_len).
// Writes per-branch partial softmax states that will be merged with shared prefix states.
void launch_branch_suffix_attn(
    const float* Q,                      // [num_branches, num_heads, head_dim]
    const float* K,                      // [total_kv_len, num_heads, head_dim]
    const float* V,                      // [total_kv_len, num_heads, head_dim]
    PartialSoftmaxState* suffix_states,  // [num_branches, num_heads]
    const TreeConfig& cfg,
    cudaStream_t stream = 0);
