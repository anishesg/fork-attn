#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include <stdexcept>

// Maximum depth of a multi-level tree (root + 7 levels of children).
static constexpr int MAX_TREE_DEPTH = 8;

// Maximum total nodes across all levels.
static constexpr int MAX_TREE_NODES = 64;

// A node in the multi-level prefix tree.
// Each node represents a KV-cache segment shared by all leaves beneath it.
// Leaves are nodes with num_children == 0.
struct TreeNode {
    int segment_start;   // token offset of this node's KV segment in the global KV buffer
    int segment_len;     // number of KV tokens in this node's segment
    int parent;          // index of parent node (-1 for root)
    int children[8];     // indices of child nodes
    int num_children;    // number of children (0 for leaves)
    int depth;           // depth in the tree (root = 0)
    int leaf_index;      // for leaves: index in [0, num_leaves); for internal nodes: -1
};

// Multi-level prefix tree describing a hierarchical KV-cache layout.
//
// KV-cache layout is a depth-first pre-order traversal of the tree:
//   [root_segment | child0_segment | child00_segment | ... | child1_segment | ...]
//
// At decode time, each leaf corresponds to one branch (candidate continuation).
// Internal nodes represent KV segments shared by all leaves in their subtree.
struct MultiLevelTree {
    TreeNode nodes[MAX_TREE_NODES];
    int num_nodes;
    int num_leaves;
    int num_heads;
    int head_dim;
    float scale;

    // Build a complete k-ary tree of given depth.
    // Each node has a segment of length seg_len tokens.
    // Total KV tokens = seg_len * num_nodes.
    static MultiLevelTree make_uniform(
        int branching_factor,
        int depth,
        int seg_len,
        int num_heads,
        int head_dim)
    {
        MultiLevelTree tree;
        tree.num_nodes = 0;
        tree.num_leaves = 0;
        tree.num_heads = num_heads;
        tree.head_dim = head_dim;
        tree.scale = 1.0f / sqrtf((float)head_dim);

        int segment_cursor = 0;
        // Build via BFS queue to assign node indices level by level.
        // Queue stores (parent_idx, depth_of_node).
        struct QueueEntry { int parent; int node_depth; };
        QueueEntry queue[MAX_TREE_NODES];
        int qfront = 0, qback = 0;

        // Create root
        {
            TreeNode& root = tree.nodes[0];
            root.segment_start = 0;
            root.segment_len = seg_len;
            root.parent = -1;
            root.num_children = 0;
            root.depth = 0;
            root.leaf_index = -1;
            tree.num_nodes = 1;
            segment_cursor += seg_len;
        }

        // Enqueue root's children at depth 1
        if (depth > 0) {
            queue[qback++] = {0, 1};
        }

        while (qfront < qback) {
            auto [parent_idx, node_depth] = queue[qfront++];

            if (tree.num_nodes + branching_factor > MAX_TREE_NODES) {
                throw std::runtime_error("Tree exceeds MAX_TREE_NODES");
            }

            TreeNode& parent = tree.nodes[parent_idx];
            for (int c = 0; c < branching_factor; c++) {
                int idx = tree.num_nodes++;
                parent.children[parent.num_children++] = idx;

                TreeNode& node = tree.nodes[idx];
                node.segment_start = segment_cursor;
                node.segment_len = seg_len;
                node.parent = parent_idx;
                node.num_children = 0;
                node.depth = node_depth;
                node.leaf_index = -1;
                segment_cursor += seg_len;

                if (node_depth < depth) {
                    // Not yet a leaf: enqueue children
                    queue[qback++] = {idx, node_depth + 1};
                } else {
                    // Leaf node
                    node.leaf_index = tree.num_leaves++;
                }
            }
        }

        return tree;
    }

    // Build a two-level tree: one root segment, then B branches each with seg_len tokens.
    // Equivalent to the original TreeConfig but expressed as a MultiLevelTree.
    static MultiLevelTree make_two_level(
        int num_branches,
        int shared_len,
        int unique_len,
        int num_heads,
        int head_dim)
    {
        MultiLevelTree tree;
        tree.num_nodes = 0;
        tree.num_leaves = 0;
        tree.num_heads = num_heads;
        tree.head_dim = head_dim;
        tree.scale = 1.0f / sqrtf((float)head_dim);

        // Root: the shared prefix segment
        {
            TreeNode& root = tree.nodes[0];
            root.segment_start = 0;
            root.segment_len = shared_len;
            root.parent = -1;
            root.num_children = 0;
            root.depth = 0;
            root.leaf_index = -1;
            tree.num_nodes = 1;
        }

        // Leaf children: one per branch
        TreeNode& root = tree.nodes[0];
        for (int b = 0; b < num_branches; b++) {
            int idx = tree.num_nodes++;
            root.children[root.num_children++] = idx;

            TreeNode& leaf = tree.nodes[idx];
            leaf.segment_start = shared_len + b * unique_len;
            leaf.segment_len = unique_len;
            leaf.parent = 0;
            leaf.num_children = 0;
            leaf.depth = 1;
            leaf.leaf_index = tree.num_leaves++;
        }

        return tree;
    }

    // Total number of KV tokens across all nodes.
    int total_kv_len() const {
        int total = 0;
        for (int i = 0; i < num_nodes; i++) {
            total += nodes[i].segment_len;
        }
        return total;
    }
};
