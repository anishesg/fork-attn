#include "fork_attn.cuh"
#include "partial_softmax.cuh"
#include "shared_prefix_attn.cuh"
#include "branch_suffix_attn.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

// Merge kernel: for each (branch, head), combine shared prefix partial state
// with branch suffix partial state to produce final normalized output.
__global__ void merge_and_finalize_kernel(
    const PartialSoftmaxState* __restrict__ shared_states,  // [num_branches, num_heads]
    const PartialSoftmaxState* __restrict__ suffix_states,  // [num_branches, num_heads]
    float* __restrict__ Out,                                // [num_branches, num_heads, head_dim]
    TreeConfig cfg)
{
    int b = blockIdx.x;
    int h = blockIdx.y;
    int D = cfg.head_dim;

    const PartialSoftmaxState& sp = shared_states[b * cfg.num_heads + h];
    const PartialSoftmaxState& su = suffix_states[b * cfg.num_heads + h];

    PartialSoftmaxState merged = merge_states(sp, su, D);

    float* out_ptr = Out + (b * cfg.num_heads + h) * D;
    finalize_state(merged, out_ptr, D);
}

void launch_fork_attn(
    const float* Q,
    const float* K,
    const float* V,
    float* Out,
    const TreeConfig& cfg,
    cudaStream_t stream)
{
    int num_states = cfg.num_branches * cfg.num_heads;

    PartialSoftmaxState* shared_states = nullptr;
    PartialSoftmaxState* suffix_states = nullptr;

    cudaError_t err;
    err = cudaMallocAsync(&shared_states, num_states * sizeof(PartialSoftmaxState), stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    err = cudaMallocAsync(&suffix_states, num_states * sizeof(PartialSoftmaxState), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(shared_states, stream);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Pass 1: shared prefix attention for all branches.
    launch_shared_prefix_attn(Q, K, V, shared_states, cfg, stream);

    // Pass 2: per-branch unique suffix attention.
    launch_branch_suffix_attn(Q, K, V, suffix_states, cfg, stream);

    // Pass 3: merge partial states and write final output.
    dim3 grid(cfg.num_branches, cfg.num_heads);
    merge_and_finalize_kernel<<<grid, 1, 0, stream>>>(shared_states, suffix_states, Out, cfg);

    cudaFreeAsync(shared_states, stream);
    cudaFreeAsync(suffix_states, stream);
}
