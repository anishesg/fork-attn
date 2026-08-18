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

struct BenchConfig {
    int num_branches;
    int shared_len;
    int unique_len;
    int num_heads;
    int head_dim;
};

// Returns median latency in microseconds over `iters` warm iterations.
static float time_kernel(
    std::function<void()> fn,
    int warmup = 5,
    int iters = 50)
{
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; i++) fn();
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) fn();
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (ms / iters) * 1000.0f;  // convert ms to us
}

static void bench_one(const BenchConfig& bc) {
    TreeConfig cfg;
    cfg.num_branches = bc.num_branches;
    cfg.shared_len   = bc.shared_len;
    cfg.unique_len   = bc.unique_len;
    cfg.num_heads    = bc.num_heads;
    cfg.head_dim     = bc.head_dim;
    cfg.scale        = 1.0f / sqrtf((float)bc.head_dim);

    int total_kv = cfg.total_kv_len();
    size_t Q_sz  = (size_t)bc.num_branches * bc.num_heads * bc.head_dim;
    size_t KV_sz = (size_t)total_kv         * bc.num_heads * bc.head_dim;

    // Initialize with random data.
    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> h_Q(Q_sz), h_K(KV_sz), h_V(KV_sz);
    for (float& x : h_Q) x = dist(rng);
    for (float& x : h_K) x = dist(rng);
    for (float& x : h_V) x = dist(rng);

    float *d_Q, *d_K, *d_V, *d_out;
    CHECK(cudaMalloc(&d_Q,   Q_sz  * sizeof(float)));
    CHECK(cudaMalloc(&d_K,   KV_sz * sizeof(float)));
    CHECK(cudaMalloc(&d_V,   KV_sz * sizeof(float)));
    CHECK(cudaMalloc(&d_out, Q_sz  * sizeof(float)));
    CHECK(cudaMemcpy(d_Q, h_Q.data(), Q_sz  * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_K, h_K.data(), KV_sz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_V, h_V.data(), KV_sz * sizeof(float), cudaMemcpyHostToDevice));

    float t_naive = time_kernel([&]() {
        launch_naive_decode_attn(d_Q, d_K, d_V, d_out, cfg);
    });

    float t_fork = time_kernel([&]() {
        launch_fork_attn(d_Q, d_K, d_V, d_out, cfg);
    });

    // Theoretical KV read reduction
    long naive_kv = (long)bc.num_branches * (bc.shared_len + bc.unique_len);
    long fork_kv  = bc.shared_len + (long)bc.num_branches * bc.unique_len;
    float kv_ratio = (float)naive_kv / fork_kv;

    printf("  B=%2d  L=%6d  U=%4d  H=%2d  D=%3d | naive=%7.1f us  fork=%7.1f us  speedup=%.2fx  kv_ratio=%.2fx\n",
           bc.num_branches, bc.shared_len, bc.unique_len, bc.num_heads, bc.head_dim,
           t_naive, t_fork, t_naive / t_fork, kv_ratio);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);
}

int main() {
    int dev;
    cudaDeviceProp prop;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    printf("fork-attn latency benchmark\n");
    printf("B=branches  L=shared_len  U=unique_len  H=heads  D=head_dim\n\n");

    // Sweep: branches in {2, 4, 8, 16}, shared_len in {1024, 4096, 16384}, unique_len in {64, 256}
    int branches[]    = {2, 4, 8, 16};
    int shared_lens[] = {1024, 4096, 16384};
    int unique_lens[] = {64, 256};
    int num_heads     = 8;
    int head_dim      = 64;

    for (int U : unique_lens) {
        printf("--- unique_len = %d ---\n", U);
        for (int L : shared_lens) {
            for (int B : branches) {
                BenchConfig bc{B, L, U, num_heads, head_dim};
                bench_one(bc);
            }
        }
        printf("\n");
    }

    // Additional: large head_dim
    printf("--- head_dim = 128 ---\n");
    for (int B : branches) {
        BenchConfig bc{B, 4096, 64, 8, 128};
        bench_one(bc);
    }

    return 0;
}
