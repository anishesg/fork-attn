#include <torch/extension.h>
#include "fork_attn.cuh"
#include "naive_decode_attn.cuh"

static void check_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
}

// fork_attention(Q, K, V, num_branches, shared_len, unique_len) -> Out
// Q: [num_branches, num_heads, head_dim]
// K: [total_kv_len, num_heads, head_dim]
// V: [total_kv_len, num_heads, head_dim]
// Returns Out: [num_branches, num_heads, head_dim]
torch::Tensor fork_attention(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int num_branches,
    int shared_len,
    int unique_len)
{
    check_tensor(Q, "Q");
    check_tensor(K, "K");
    check_tensor(V, "V");

    TORCH_CHECK(Q.dim() == 3, "Q must be 3D: [num_branches, num_heads, head_dim]");
    TORCH_CHECK(K.dim() == 3, "K must be 3D: [total_kv_len, num_heads, head_dim]");
    TORCH_CHECK(V.dim() == 3, "V must be 3D: [total_kv_len, num_heads, head_dim]");

    int n_branches = Q.size(0);
    int num_heads  = Q.size(1);
    int head_dim   = Q.size(2);

    TORCH_CHECK(n_branches == num_branches,
        "Q.size(0) must equal num_branches, got ", n_branches, " vs ", num_branches);
    TORCH_CHECK(K.size(1) == num_heads, "K num_heads mismatch");
    TORCH_CHECK(V.size(1) == num_heads, "V num_heads mismatch");
    TORCH_CHECK(K.size(2) == head_dim, "K head_dim mismatch");
    TORCH_CHECK(V.size(2) == head_dim, "V head_dim mismatch");
    TORCH_CHECK(head_dim <= 128, "head_dim must be <= 128");

    int expected_kv = shared_len + num_branches * unique_len;
    TORCH_CHECK(K.size(0) == expected_kv,
        "K.size(0) must equal shared_len + num_branches*unique_len = ", expected_kv);
    TORCH_CHECK(V.size(0) == expected_kv,
        "V.size(0) must equal shared_len + num_branches*unique_len = ", expected_kv);

    TreeConfig cfg;
    cfg.num_branches = num_branches;
    cfg.shared_len   = shared_len;
    cfg.unique_len   = unique_len;
    cfg.head_dim     = head_dim;
    cfg.num_heads    = num_heads;
    cfg.scale        = 1.0f / sqrtf((float)head_dim);

    torch::Tensor Out = torch::empty_like(Q);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fork_attn(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        Out.data_ptr<float>(),
        cfg,
        stream);

    return Out;
}

torch::Tensor naive_attention(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int num_branches,
    int shared_len,
    int unique_len)
{
    check_tensor(Q, "Q");
    check_tensor(K, "K");
    check_tensor(V, "V");

    int num_heads = Q.size(1);
    int head_dim  = Q.size(2);

    TreeConfig cfg;
    cfg.num_branches = num_branches;
    cfg.shared_len   = shared_len;
    cfg.unique_len   = unique_len;
    cfg.head_dim     = head_dim;
    cfg.num_heads    = num_heads;
    cfg.scale        = 1.0f / sqrtf((float)head_dim);

    torch::Tensor Out = torch::empty_like(Q);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_naive_decode_attn(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        Out.data_ptr<float>(),
        cfg,
        stream);

    return Out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fork_attention", &fork_attention,
          "Fork attention: fused shared-prefix + per-branch suffix attention",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("num_branches"), py::arg("shared_len"), py::arg("unique_len"));

    m.def("naive_attention", &naive_attention,
          "Naive per-branch attention baseline",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("num_branches"), py::arg("shared_len"), py::arg("unique_len"));
}
