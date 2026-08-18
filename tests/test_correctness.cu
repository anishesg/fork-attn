#include "fork_attn.cuh"
#include "naive_decode_attn.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

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

static bool run_test(int num_branches, int shared_len, int unique_len,
                     int num_heads, int head_dim, unsigned seed)
{
    std::mt19937 rng(seed);

    TreeConfig cfg;
    cfg.num_branches = num_branches;
    cfg.shared_len   = shared_len;
    cfg.unique_len   = unique_len;
    cfg.num_heads    = num_heads;
    cfg.head_dim     = head_dim;
    cfg.scale        = 1.0f / sqrtf((float)head_dim);

    int total_kv = cfg.total_kv_len();
    size_t Q_sz  = (size_t)num_branches * num_heads * head_dim;
    size_t KV_sz = (size_t)total_kv     * num_heads * head_dim;
    size_t O_sz  = Q_sz;

    std::vector<float> h_Q(Q_sz), h_K(KV_sz), h_V(KV_sz);
    fill_random(h_Q, rng);
    fill_random(h_K, rng);
    fill_random(h_V, rng);

    float *d_Q, *d_K, *d_V, *d_out_naive, *d_out_fork;
    CHECK(cudaMalloc(&d_Q,         Q_sz  * sizeof(float)));
    CHECK(cudaMalloc(&d_K,         KV_sz * sizeof(float)));
    CHECK(cudaMalloc(&d_V,         KV_sz * sizeof(float)));
    CHECK(cudaMalloc(&d_out_naive, O_sz  * sizeof(float)));
    CHECK(cudaMalloc(&d_out_fork,  O_sz  * sizeof(float)));

    CHECK(cudaMemcpy(d_Q, h_Q.data(), Q_sz  * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_K, h_K.data(), KV_sz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_V, h_V.data(), KV_sz * sizeof(float), cudaMemcpyHostToDevice));

    launch_naive_decode_attn(d_Q, d_K, d_V, d_out_naive, cfg);
    launch_fork_attn(d_Q, d_K, d_V, d_out_fork, cfg);
    CHECK(cudaDeviceSynchronize());

    std::vector<float> h_naive(O_sz), h_fork(O_sz);
    CHECK(cudaMemcpy(h_naive.data(), d_out_naive, O_sz * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_fork.data(),  d_out_fork,  O_sz * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_out_naive); cudaFree(d_out_fork);

    float cos_sim = cosine_similarity(h_naive, h_fork);
    float max_err = max_abs_error(h_naive, h_fork);

    bool pass = (cos_sim > 0.9999f) && (max_err < 1e-3f);
    printf("  branches=%2d shared_len=%5d unique_len=%3d heads=%d dim=%3d | "
           "cos_sim=%.6f  max_abs_err=%.2e  [%s]\n",
           num_branches, shared_len, unique_len, num_heads, head_dim,
           cos_sim, max_err, pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    printf("fork-attn correctness tests\n");
    printf("comparing fork-attn output against naive batched baseline\n\n");

    bool all_pass = true;

    // Basic smoke tests
    all_pass &= run_test(2,  64,  16, 1,  64, 42);
    all_pass &= run_test(4,  128, 32, 2,  64, 43);
    all_pass &= run_test(8,  512, 64, 4,  64, 44);

    // Larger shared prefix (the regime where fork-attn wins)
    all_pass &= run_test(4,  1024, 64,  4,  64,  50);
    all_pass &= run_test(8,  4096, 64,  8,  64,  51);
    all_pass &= run_test(16, 4096, 128, 8,  128, 52);

    // Edge: unique_len = 1
    all_pass &= run_test(4,  256, 1, 4, 64, 60);

    // Edge: shared_len = 1
    all_pass &= run_test(4,  1,   64, 4, 64, 61);

    // Multi-head, larger dim
    all_pass &= run_test(8, 512, 64, 16, 128, 70);

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
