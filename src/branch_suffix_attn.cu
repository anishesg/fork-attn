#include "branch_suffix_attn.cuh"

// Grid: (num_branches, num_heads)
// One thread handles its branch's unique suffix segment.
__global__ void branch_suffix_attn_kernel(
    const float* __restrict__ Q,              // [num_branches, num_heads, head_dim]
    const float* __restrict__ K,              // [total_kv_len, num_heads, head_dim]
    const float* __restrict__ V,              // [total_kv_len, num_heads, head_dim]
    PartialSoftmaxState* __restrict__ out,    // [num_branches, num_heads]
    TreeConfig cfg)
{
    int b = blockIdx.x;
    int h = blockIdx.y;
    int D = cfg.head_dim;

    float q[128];
    for (int d = 0; d < D; d++) {
        q[d] = Q[(b * cfg.num_heads + h) * D + d];
    }

    PartialSoftmaxState state;
    state.init();

    int offset = cfg.branch_offset(b);
    for (int t = 0; t < cfg.unique_len; t++) {
        int t_global = offset + t;
        const float* k_t = K + (t_global * cfg.num_heads + h) * D;
        const float* v_t = V + (t_global * cfg.num_heads + h) * D;

        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += q[d] * k_t[d];
        }
        dot *= cfg.scale;

        float v_buf[128];
        for (int d = 0; d < D; d++) v_buf[d] = v_t[d];

        state.update(dot, v_buf, D);
    }

    out[b * cfg.num_heads + h] = state;
}

void launch_branch_suffix_attn(
    const float* Q,
    const float* K,
    const float* V,
    PartialSoftmaxState* suffix_states,
    const TreeConfig& cfg,
    cudaStream_t stream)
{
    dim3 grid(cfg.num_branches, cfg.num_heads);
    branch_suffix_attn_kernel<<<grid, 1, 0, stream>>>(Q, K, V, suffix_states, cfg);
}
