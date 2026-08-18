#include "naive_decode_attn.cuh"
#include <float.h>

// Each thread block handles one (branch, head) pair.
// Grid: (num_branches, num_heads)
// Block: (BLOCK_T,) threads iterate over all KV tokens in the branch's view.
//
// Each branch attends to: K_shared [0..shared_len) ++ K_unique_b [branch_offset..branch_offset+unique_len)
// This is the baseline: every branch re-reads the entire shared prefix independently.

static constexpr int BLOCK_T = 256;

__global__ void naive_decode_attn_kernel(
    const float* __restrict__ Q,        // [num_branches, num_heads, head_dim]
    const float* __restrict__ K,        // [total_kv_len, num_heads, head_dim]
    const float* __restrict__ V,        // [total_kv_len, num_heads, head_dim]
    float* __restrict__ Out,            // [num_branches, num_heads, head_dim]
    TreeConfig cfg)
{
    int b = blockIdx.x;   // branch index
    int h = blockIdx.y;   // head index

    int D = cfg.head_dim;
    int branch_kv_start = cfg.branch_offset(b);
    // Branch b attends to: shared prefix [0..shared_len) + suffix [branch_kv_start..branch_kv_start+unique_len)
    // We handle them as two contiguous logical windows.

    // Load query vector for this branch and head into registers.
    // D <= 128 for typical configs, so stack allocation is fine.
    extern __shared__ float smem[];
    float* q = smem;                         // [D]
    float* scores_smem = smem + D;           // [BLOCK_T]  (scratch for partial scores)

    // Load Q
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        q[d] = Q[(b * cfg.num_heads + h) * D + d];
    }
    __syncthreads();

    // Online softmax: track running max and sum of exps, accumulate value sum.
    float run_max = -FLT_MAX;
    float run_sum = 0.0f;
    float acc[128] = {};   // accumulated weighted value, D <= 128

    int total_kv = cfg.shared_len + cfg.unique_len;

    for (int t_local = 0; t_local < total_kv; t_local++) {
        // Map local token index to global KV-cache position.
        int t_global;
        if (t_local < cfg.shared_len) {
            t_global = t_local;
        } else {
            t_global = branch_kv_start + (t_local - cfg.shared_len);
        }

        // Compute dot product q . k[t_global] in parallel, then reduce.
        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += q[d] * K[(t_global * cfg.num_heads + h) * D + d];
        }
        dot *= cfg.scale;

        // Online softmax update.
        float new_max = fmaxf(run_max, dot);
        float scale_old = expf(run_max - new_max);
        float scale_new = expf(dot - new_max);

        run_sum = run_sum * scale_old + scale_new;
        for (int d = 0; d < D; d++) {
            acc[d] = acc[d] * scale_old + scale_new * V[(t_global * cfg.num_heads + h) * D + d];
        }
        run_max = new_max;
    }

    // Normalize and write output.
    float inv_sum = 1.0f / run_sum;
    for (int d = 0; d < D; d++) {
        Out[(b * cfg.num_heads + h) * D + d] = acc[d] * inv_sum;
    }
}

void launch_naive_decode_attn(
    const float* Q,
    const float* K,
    const float* V,
    float* Out,
    const TreeConfig& cfg,
    cudaStream_t stream)
{
    dim3 grid(cfg.num_branches, cfg.num_heads);
    // Shared memory: Q vector (D floats) + score scratch (BLOCK_T floats)
    size_t smem = (cfg.head_dim + BLOCK_T) * sizeof(float);
    naive_decode_attn_kernel<<<grid, 1, smem, stream>>>(Q, K, V, Out, cfg);
}
