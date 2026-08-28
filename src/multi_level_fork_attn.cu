#include "multi_level_fork_attn.cuh"
#include "partial_softmax.cuh"
#include "multi_level_tree.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>

// Compute partial softmax state for one node's KV segment.
// Each thread block handles one (query_leaf, head) pair.
// Only the leaves in this node's subtree compute against this segment.
__global__ void node_segment_attn_kernel(
    const float* __restrict__ Q,          // [num_leaves, num_heads, head_dim]
    const float* __restrict__ K,          // [total_kv_len, num_heads, head_dim]
    const float* __restrict__ V,          // [total_kv_len, num_heads, head_dim]
    PartialSoftmaxState* __restrict__ node_states,  // [num_nodes, num_leaves, num_heads]
    const TreeNode* __restrict__ nodes,
    int node_idx,
    int num_leaves,
    int num_heads,
    int head_dim,
    float scale,
    const int* __restrict__ leaf_ancestors,   // [num_leaves, MAX_TREE_DEPTH]: ancestor node indices for each leaf
    int max_depth)
{
    int leaf = blockIdx.x;
    int h    = blockIdx.y;

    // Check whether this leaf is a descendant of node_idx.
    // A leaf is a descendant if node_idx appears in its ancestor list.
    bool is_descendant = false;
    for (int d = 0; d < max_depth; d++) {
        if (leaf_ancestors[leaf * max_depth + d] == node_idx) {
            is_descendant = true;
            break;
        }
    }
    if (!is_descendant) return;

    const TreeNode& node = nodes[node_idx];
    int seg_start = node.segment_start;
    int seg_len   = node.segment_len;
    int D         = head_dim;

    const float* q_ptr = Q + (leaf * num_heads + h) * D;

    PartialSoftmaxState state;
    state.init();

    for (int t = 0; t < seg_len; t++) {
        const float* k_ptr = K + ((seg_start + t) * num_heads + h) * D;
        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += q_ptr[d] * k_ptr[d];
        }
        dot *= scale;
        const float* v_ptr = V + ((seg_start + t) * num_heads + h) * D;
        state.update(dot, v_ptr, D);
    }

    node_states[(node_idx * num_leaves + leaf) * num_heads + h] = state;
}

// Finalize: for each leaf, merge all ancestor node states along the root-to-leaf path,
// then write normalized output.
__global__ void merge_and_finalize_multi_kernel(
    const PartialSoftmaxState* __restrict__ node_states,  // [num_nodes, num_leaves, num_heads]
    float* __restrict__ Out,                              // [num_leaves, num_heads, head_dim]
    const TreeNode* __restrict__ nodes,
    int num_nodes,
    int num_leaves,
    int num_heads,
    int head_dim,
    const int* __restrict__ leaf_ancestors,  // [num_leaves, max_depth]
    const int* __restrict__ leaf_path_len,   // [num_leaves]: number of nodes in root-to-leaf path
    int max_depth)
{
    int leaf = blockIdx.x;
    int h    = blockIdx.y;
    int D    = head_dim;

    int path_len = leaf_path_len[leaf];

    // Collect and merge partial states along this leaf's root-to-leaf path.
    PartialSoftmaxState merged;
    merged.running_max = -3.402823466e+38f;
    merged.exp_sum = 0.0f;
    for (int d = 0; d < D; d++) merged.value_acc[d] = 0.0f;

    bool first = true;
    for (int depth = 0; depth < path_len; depth++) {
        int nidx = leaf_ancestors[leaf * max_depth + depth];
        const PartialSoftmaxState& seg_state =
            node_states[(nidx * num_leaves + leaf) * num_heads + h];

        if (first) {
            merged = seg_state;
            first = false;
        } else {
            merged = merge_states(merged, seg_state, D);
        }
    }

    float* out_ptr = Out + (leaf * num_heads + h) * D;
    if (!first) {
        finalize_state(merged, out_ptr, D);
    } else {
        for (int d = 0; d < D; d++) out_ptr[d] = 0.0f;
    }
}

// Build the ancestor table on the CPU: for each leaf, list all nodes from root to leaf.
static void build_ancestor_table(
    const MultiLevelTree& tree,
    std::vector<int>& leaf_ancestors,   // [num_leaves * MAX_TREE_DEPTH]
    std::vector<int>& leaf_path_len,    // [num_leaves]
    int max_depth)
{
    int nl = tree.num_leaves;
    leaf_ancestors.assign(nl * max_depth, -1);
    leaf_path_len.assign(nl, 0);

    for (int i = 0; i < tree.num_nodes; i++) {
        const TreeNode& node = tree.nodes[i];
        if (node.leaf_index < 0) continue;

        int leaf = node.leaf_index;
        // Walk up the tree to collect path from root to leaf.
        std::vector<int> path;
        int cur = i;
        while (cur != -1) {
            path.push_back(cur);
            cur = tree.nodes[cur].parent;
        }
        // Reverse so path goes root -> leaf.
        std::reverse(path.begin(), path.end());

        leaf_path_len[leaf] = (int)path.size();
        for (int d = 0; d < (int)path.size() && d < max_depth; d++) {
            leaf_ancestors[leaf * max_depth + d] = path[d];
        }
    }
}

void launch_multi_level_fork_attn(
    const float* Q,
    const float* K,
    const float* V,
    float* Out,
    const MultiLevelTree& tree,
    cudaStream_t stream)
{
    int num_nodes  = tree.num_nodes;
    int num_leaves = tree.num_leaves;
    int num_heads  = tree.num_heads;
    int head_dim   = tree.head_dim;
    float scale    = tree.scale;

    if (num_leaves == 0 || num_nodes == 0) return;

    // Build ancestor table on CPU.
    std::vector<int> h_leaf_ancestors, h_leaf_path_len;
    build_ancestor_table(tree, h_leaf_ancestors, h_leaf_path_len, MAX_TREE_DEPTH);

    // Allocate device memory.
    size_t states_bytes  = (size_t)num_nodes * num_leaves * num_heads * sizeof(PartialSoftmaxState);
    size_t anc_bytes     = (size_t)num_leaves * MAX_TREE_DEPTH * sizeof(int);
    size_t plen_bytes    = (size_t)num_leaves * sizeof(int);
    size_t nodes_bytes   = (size_t)num_nodes * sizeof(TreeNode);

    PartialSoftmaxState* d_node_states = nullptr;
    int* d_leaf_ancestors = nullptr;
    int* d_leaf_path_len  = nullptr;
    TreeNode* d_nodes     = nullptr;

    cudaError_t err;
    err = cudaMallocAsync(&d_node_states, states_bytes, stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    err = cudaMallocAsync(&d_leaf_ancestors, anc_bytes, stream);
    if (err != cudaSuccess) { cudaFreeAsync(d_node_states, stream); throw std::runtime_error(cudaGetErrorString(err)); }
    err = cudaMallocAsync(&d_leaf_path_len, plen_bytes, stream);
    if (err != cudaSuccess) { cudaFreeAsync(d_node_states, stream); cudaFreeAsync(d_leaf_ancestors, stream); throw std::runtime_error(cudaGetErrorString(err)); }
    err = cudaMallocAsync(&d_nodes, nodes_bytes, stream);
    if (err != cudaSuccess) { cudaFreeAsync(d_node_states, stream); cudaFreeAsync(d_leaf_ancestors, stream); cudaFreeAsync(d_leaf_path_len, stream); throw std::runtime_error(cudaGetErrorString(err)); }

    // Copy host data to device.
    cudaMemcpyAsync(d_leaf_ancestors, h_leaf_ancestors.data(), anc_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_leaf_path_len,  h_leaf_path_len.data(),  plen_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_nodes,          tree.nodes,              nodes_bytes, cudaMemcpyHostToDevice, stream);

    // Launch one kernel per node: compute partial softmax for each node's KV segment
    // for all leaves that are descendants of this node.
    dim3 grid_node(num_leaves, num_heads);
    for (int nidx = 0; nidx < num_nodes; nidx++) {
        node_segment_attn_kernel<<<grid_node, 1, 0, stream>>>(
            Q, K, V,
            d_node_states,
            d_nodes,
            nidx,
            num_leaves,
            num_heads,
            head_dim,
            scale,
            d_leaf_ancestors,
            MAX_TREE_DEPTH);
    }

    // Merge all ancestor segments per leaf and write output.
    dim3 grid_merge(num_leaves, num_heads);
    merge_and_finalize_multi_kernel<<<grid_merge, 1, 0, stream>>>(
        d_node_states,
        Out,
        d_nodes,
        num_nodes,
        num_leaves,
        num_heads,
        head_dim,
        d_leaf_ancestors,
        d_leaf_path_len,
        MAX_TREE_DEPTH);

    cudaFreeAsync(d_node_states,    stream);
    cudaFreeAsync(d_leaf_ancestors, stream);
    cudaFreeAsync(d_leaf_path_len,  stream);
    cudaFreeAsync(d_nodes,          stream);
}
