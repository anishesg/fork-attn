#include "shared_prefix_attn.cuh"

// Grid: (num_branches, num_heads)
// One thread computes the full shared-prefix partial softmax for its (branch, head).
// Since we process shared_len tokens sequentially in registers, no sync needed.
//
// This kernel is launched once per decode step; all branches share the output.
__global__ void shared_prefix_attn_kernel(
    const float* __restrict__ Q,               // [num_branches, num_heads, head_dim]
    const float* __restrict__ K,               // [total_kv_len, num_heads, head_dim]
    const float* __restrict__ V,               // [total_kv_len, num_heads, head_dim]
    PartialSoftmaxState* __restrict__ out,     // [num_branches, num_heads]
    TreeConfig cfg)
{
    int b = blockIdx.x;
    int h = blockIdx.y;
    int D = cfg.head_dim;

    // Load query for this (branch, head) into local array.
    float q[128];
    for (int d = 0; d < D; d++) {
        q[d] = Q[(b * cfg.num_heads + h) * D + d];
    }

    PartialSoftmaxState state;
    state.init();

    // Iterate over all shared-prefix tokens.
    for (int t = 0; t < cfg.shared_len; t++) {
        const float* k_t = K + (t * cfg.num_heads + h) * D;
        const float* v_t = V + (t * cfg.num_heads + h) * D;

        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += q[d] * k_t[d];
        }
        dot *= cfg.scale;

        // Load v into local buffer for the update call.
        float v_buf[128];
        for (int d = 0; d < D; d++) v_buf[d] = v_t[d];

        state.update(dot, v_buf, D);
    }

    out[b * cfg.num_heads + h] = state;
}

void launch_shared_prefix_attn(
    const float* Q,
    const float* K,
    const float* V,
    PartialSoftmaxState* shared_states,
    const TreeConfig& cfg,
    cudaStream_t stream)
{
    dim3 grid(cfg.num_branches, cfg.num_heads);
    shared_prefix_attn_kernel<<<grid, 1, 0, stream>>>(Q, K, V, shared_states, cfg);
}
