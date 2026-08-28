#pragma once
#include "multi_level_tree.cuh"
#include "partial_softmax.cuh"
#include <cuda_runtime.h>

// Compute hierarchical attention over a multi-level prefix tree.
//
// Algorithm: bottom-up tree traversal.
//   1. For every node in reverse BFS order (leaves first), compute partial softmax
//      state over that node's KV segment for all queries whose path includes this node.
//   2. Merge partial states along each root-to-leaf path via log-sum-exp chaining.
//   3. Finalize and write output for each leaf (one query per leaf).
//
// Q: [num_leaves, num_heads, head_dim]
// K: [total_kv_len, num_heads, head_dim]  -- DFS pre-order layout matching tree node order
// V: [total_kv_len, num_heads, head_dim]
// Out: [num_leaves, num_heads, head_dim]
void launch_multi_level_fork_attn(
    const float* Q,
    const float* K,
    const float* V,
    float* Out,
    const MultiLevelTree& tree,
    cudaStream_t stream = 0);
