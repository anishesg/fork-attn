#include "multi_level_fork_attn.cuh"
#include "multi_level_tree.cuh"
#include "partial_softmax.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

static void check(cudaError_t e, const char* f, int l) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", f, l, cudaGetErrorString(e));
        exit(1);
    }
}
#define CHECK(x) check(x, __FILE__, __LINE__)

static void fill_random(std::vector<float>& v, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (float& x : v) x = dist(rng);
}

// Naive per-leaf full-context attention on CPU for reference.
// For leaf i, attends to: root_seg ++ path_to_leaf_segs ++ leaf_seg.
static void naive_attn_cpu(
    const std::vector<float>& Q,   // [num_leaves, num_heads, head_dim]
    const std::vector<float>& K,   // [total_kv, num_heads, head_dim]
    const std::vector<float>& V,   // [total_kv, num_heads, head_dim]
    std::vector<float>& Out,       // [num_leaves, num_heads, head_dim]
    const MultiLevelTree& tree)
{
    int nl = tree.num_leaves;
    int nh = tree.num_heads;
    int D  = tree.head_dim;
    float scale = tree.scale;

    Out.assign((size_t)nl * nh * D, 0.0f);

    // For each leaf, collect the root-to-leaf path of KV tokens.
    for (int leaf_idx = 0; leaf_idx < tree.num_nodes; leaf_idx++) {
        const TreeNode& leaf_node = tree.nodes[leaf_idx];
        if (leaf_node.leaf_index < 0) continue;

        int li = leaf_node.leaf_index;

        // Build root-to-leaf path (node indices).
        std::vector<int> path;
        int cur = leaf_idx;
        while (cur != -1) {
            path.push_back(cur);
            cur = tree.nodes[cur].parent;
        }
        std::reverse(path.begin(), path.end());

        // Collect all KV token indices in order.
        std::vector<int> kv_indices;
        for (int nidx : path) {
            int start = tree.nodes[nidx].segment_start;
            int len   = tree.nodes[nidx].segment_len;
            for (int t = 0; t < len; t++) {
                kv_indices.push_back(start + t);
            }
        }

        for (int h = 0; h < nh; h++) {
            const float* q_ptr = Q.data() + (li * nh + h) * D;

            // Online softmax
            float run_max = -1e38f, run_sum = 0.0f;
            std::vector<float> acc(D, 0.0f);

            for (int t_global : kv_indices) {
                const float* k_ptr = K.data() + (t_global * nh + h) * D;
                float dot = 0.0f;
                for (int d = 0; d < D; d++) dot += q_ptr[d] * k_ptr[d];
                dot *= scale;

                float new_max = std::max(run_max, dot);
                float s_old = expf(run_max - new_max);
                float s_new = expf(dot - new_max);
                run_sum = run_sum * s_old + s_new;

                const float* v_ptr = V.data() + (t_global * nh + h) * D;
                for (int d = 0; d < D; d++) {
                    acc[d] = acc[d] * s_old + s_new * v_ptr[d];
                }
                run_max = new_max;
            }
            float inv_sum = 1.0f / run_sum;
            for (int d = 0; d < D; d++) {
                Out[(li * nh + h) * D + d] = acc[d] * inv_sum;
            }
        }
    }
}

static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-12f);
}

static float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float err = 0;
    for (size_t i = 0; i < a.size(); i++) {
        err = fmaxf(err, fabsf(a[i] - b[i]));
    }
    return err;
}

static bool run_test(const char* label, const MultiLevelTree& tree, unsigned seed) {
    std::mt19937 rng(seed);

    int nl = tree.num_leaves;
    int nh = tree.num_heads;
    int D  = tree.head_dim;
    int total_kv = tree.total_kv_len();

    size_t Q_sz  = (size_t)nl * nh * D;
    size_t KV_sz = (size_t)total_kv * nh * D;

    std::vector<float> h_Q(Q_sz), h_K(KV_sz), h_V(KV_sz);
    fill_random(h_Q, rng);
    fill_random(h_K, rng);
    fill_random(h_V, rng);

    // CPU reference
    std::vector<float> h_ref;
    naive_attn_cpu(h_Q, h_K, h_V, h_ref, tree);

    // GPU fork-attn
    float *d_Q, *d_K, *d_V, *d_Out;
    CHECK(cudaMalloc(&d_Q,   Q_sz  * sizeof(float)));
    CHECK(cudaMalloc(&d_K,   KV_sz * sizeof(float)));
    CHECK(cudaMalloc(&d_V,   KV_sz * sizeof(float)));
    CHECK(cudaMalloc(&d_Out, Q_sz  * sizeof(float)));

    CHECK(cudaMemcpy(d_Q, h_Q.data(), Q_sz  * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_K, h_K.data(), KV_sz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_V, h_V.data(), KV_sz * sizeof(float), cudaMemcpyHostToDevice));

    launch_multi_level_fork_attn(d_Q, d_K, d_V, d_Out, tree);
    CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(Q_sz);
    CHECK(cudaMemcpy(h_out.data(), d_Out, Q_sz * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Out);

    float cos_sim = cosine_similarity(h_ref, h_out);
    float max_err = max_abs_error(h_ref, h_out);

    bool pass = (cos_sim > 0.9999f) && (max_err < 1e-3f);
    printf("  %-50s | leaves=%2d heads=%d dim=%3d | cos_sim=%.6f  max_err=%.2e  [%s]\n",
           label, nl, nh, D, cos_sim, max_err, pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    printf("multi-level fork-attn correctness tests\n");
    printf("comparing hierarchical kernel against naive per-leaf CPU reference\n\n");

    bool all_pass = true;

    // 3-level tree: root -> 2 mid-level nodes -> each with 2 leaves (4 leaves total)
    {
        auto tree = MultiLevelTree::make_uniform(2, 2, 64, 2, 64);
        all_pass &= run_test("uniform binary tree depth=2 (4 leaves)", tree, 42);
    }

    // 3-level tree with 3 branches per level
    {
        auto tree = MultiLevelTree::make_uniform(3, 2, 32, 2, 64);
        all_pass &= run_test("uniform ternary tree depth=2 (9 leaves)", tree, 43);
    }

    // Deep binary tree: root -> 2^4 = 16 leaves across 4 levels
    {
        auto tree = MultiLevelTree::make_uniform(2, 4, 32, 4, 64);
        all_pass &= run_test("uniform binary tree depth=4 (16 leaves)", tree, 44);
    }

    // Two-level tree: equivalent to fork_attn (regression check)
    {
        auto tree = MultiLevelTree::make_two_level(4, 256, 32, 2, 64);
        all_pass &= run_test("two-level (4 branches, shared=256, unique=32)", tree, 50);
    }

    // Two-level tree with long shared prefix
    {
        auto tree = MultiLevelTree::make_two_level(8, 1024, 64, 4, 64);
        all_pass &= run_test("two-level (8 branches, shared=1024, unique=64)", tree, 51);
    }

    // 3-level tree, 2 mid nodes each with 3 leaves, multi-head, larger dim
    {
        auto tree = MultiLevelTree::make_uniform(2, 2, 128, 8, 128);
        all_pass &= run_test("binary tree depth=2, heads=8 dim=128", tree, 60);
    }

    // Single root (degenerate: all leaves are direct children of root)
    {
        auto tree = MultiLevelTree::make_uniform(4, 1, 64, 2, 64);
        all_pass &= run_test("star tree depth=1 (4 leaves direct from root)", tree, 61);
    }

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
