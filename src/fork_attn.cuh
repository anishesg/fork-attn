#pragma once
#include "tree_topology.cuh"
#include <cuda_runtime.h>

// Run the full fork-attn pipeline:
//   1. Shared prefix pass: compute partial softmax over K[0..shared_len) for each (branch, head).
//   2. Per-branch suffix pass: compute partial softmax over K[branch_offset..branch_offset+unique_len).
//   3. Merge: combine partial states via log-sum-exp, write normalized output.
//
// All intermediate buffers are allocated/freed internally.
// The stream is used for all kernel launches; synchronization is the caller's responsibility.
void launch_fork_attn(
    const float* Q,    // [num_branches, num_heads, head_dim]
    const float* K,    // [total_kv_len, num_heads, head_dim]
    const float* V,    // [total_kv_len, num_heads, head_dim]
    float* Out,        // [num_branches, num_heads, head_dim]
    const TreeConfig& cfg,
    cudaStream_t stream = 0);
