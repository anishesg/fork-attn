#pragma once
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Partial softmax state for a single attention head and head dimension.
// Represents the unnormalized contribution of one KV segment.
// Fields:
//   running_max  -- max dot product seen so far (for numerical stability)
//   exp_sum      -- sum of exp(score - running_max) over the segment
//   value_acc    -- sum of exp(score - running_max) * v[t] over the segment
//
// Invariant: value_acc is NOT divided by exp_sum. Finalize() does that.
struct PartialSoftmaxState {
    float running_max;
    float exp_sum;
    float value_acc[128];   // supports head_dim up to 128

    __device__ __forceinline__
    void init() {
        running_max = -FLT_MAX;
        exp_sum = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < 128; d++) value_acc[d] = 0.0f;
    }

    // Accumulate one token: dot product score and value vector.
    __device__ __forceinline__
    void update(float score, const float* v, int D) {
        float new_max = fmaxf(running_max, score);
        float alpha = expf(running_max - new_max);   // rescale old accum
        float beta  = expf(score       - new_max);   // weight for new token
        running_max = new_max;
        exp_sum = exp_sum * alpha + beta;
        for (int d = 0; d < D; d++) {
            value_acc[d] = value_acc[d] * alpha + beta * v[d];
        }
    }
};

// Merge two partial states representing disjoint KV segments [A, B].
// The merged state correctly represents attention over A ++ B.
__device__ __forceinline__
PartialSoftmaxState merge_states(
    const PartialSoftmaxState& a,
    const PartialSoftmaxState& b,
    int D)
{
    PartialSoftmaxState out;
    float max_ab = fmaxf(a.running_max, b.running_max);
    float scale_a = expf(a.running_max - max_ab);
    float scale_b = expf(b.running_max - max_ab);

    out.running_max = max_ab;
    out.exp_sum = a.exp_sum * scale_a + b.exp_sum * scale_b;
    for (int d = 0; d < D; d++) {
        out.value_acc[d] = a.value_acc[d] * scale_a + b.value_acc[d] * scale_b;
    }
    return out;
}

// Normalize a partial state into a final attention output vector.
__device__ __forceinline__
void finalize_state(const PartialSoftmaxState& s, float* out, int D) {
    float inv_sum = 1.0f / s.exp_sum;
    for (int d = 0; d < D; d++) {
        out[d] = s.value_acc[d] * inv_sum;
    }
}
