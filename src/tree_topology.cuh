#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Describes a tree-structured decode where all branches share a common prefix.
// The KV-cache layout is: [shared_prefix | branch_0_suffix | branch_1_suffix | ...]
// All segments are contiguous in the same KV-cache buffer.
struct TreeConfig {
    int num_branches;     // number of diverging branches (B)
    int shared_len;       // number of shared prefix KV pairs (L)
    int unique_len;       // number of unique suffix KV pairs per branch (U)
    int head_dim;         // dimension of each attention head (D)
    int num_heads;        // number of attention heads (H)
    float scale;          // attention scale factor (1/sqrt(D))

    // Returns the offset (in tokens) of branch b's unique suffix within the KV-cache.
    __host__ __device__ __forceinline__
    int branch_offset(int b) const {
        return shared_len + b * unique_len;
    }

    // Total number of KV tokens in the cache (shared + all branch suffixes).
    __host__ __device__ __forceinline__
    int total_kv_len() const {
        return shared_len + num_branches * unique_len;
    }

    // Number of queries: one per branch (decode step).
    __host__ __device__ __forceinline__
    int num_queries() const {
        return num_branches;
    }
};
