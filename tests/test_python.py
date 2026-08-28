"""
Correctness test for fork_attn Python API against torch sdpa reference.
Checks cosine similarity > 0.999 and max absolute error < 1e-3.
"""
import math
import torch
import pytest
from fork_attn import fork_attention, fork_attention_reference


def _make_inputs(num_branches, shared_len, unique_len, num_heads, head_dim, seed=0):
    torch.manual_seed(seed)
    total_kv = shared_len + num_branches * unique_len
    Q = torch.randn(num_branches, num_heads, head_dim, device="cuda", dtype=torch.float32)
    K = torch.randn(total_kv, num_heads, head_dim, device="cuda", dtype=torch.float32)
    V = torch.randn(total_kv, num_heads, head_dim, device="cuda", dtype=torch.float32)
    return Q, K, V


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (a_flat @ b_flat / (a_flat.norm() * b_flat.norm() + 1e-12)).item()


def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


@pytest.mark.parametrize("num_branches,shared_len,unique_len,num_heads,head_dim,seed", [
    (2,   64,   16, 1,  64,  42),
    (4,  128,   32, 2,  64,  43),
    (8,  512,   64, 4,  64,  44),
    (4, 1024,   64, 4,  64,  50),
    (8, 4096,   64, 8,  64,  51),
    (4,  256,    1, 4,  64,  60),
    (4,    1,   64, 4,  64,  61),
    (8,  512,   64, 8, 128,  70),
])
def test_fork_attn_vs_sdpa(num_branches, shared_len, unique_len, num_heads, head_dim, seed):
    Q, K, V = _make_inputs(num_branches, shared_len, unique_len, num_heads, head_dim, seed)

    ref = fork_attention_reference(Q, K, V, shared_len, num_branches, unique_len)
    out = fork_attention(Q, K, V, shared_len, num_branches, unique_len)

    cos_sim = _cosine_sim(out, ref)
    max_err = _max_abs_err(out, ref)

    assert cos_sim > 0.999, (
        f"cosine similarity {cos_sim:.6f} < 0.999 for config "
        f"(branches={num_branches}, shared={shared_len}, unique={unique_len})"
    )
    assert max_err < 1e-3, (
        f"max abs error {max_err:.2e} >= 1e-3 for config "
        f"(branches={num_branches}, shared={shared_len}, unique={unique_len})"
    )


def test_output_shape():
    Q, K, V = _make_inputs(4, 128, 32, 2, 64, 99)
    out = fork_attention(Q, K, V, 128, 4, 32)
    assert out.shape == (4, 2, 64)
    assert out.dtype == torch.float32
    assert out.is_cuda


def test_large_shared_prefix():
    """Regime where fork-attn has the largest benefit: many branches, long shared prefix."""
    Q, K, V = _make_inputs(16, 4096, 64, 8, 128, 100)
    ref = fork_attention_reference(Q, K, V, 4096, 16, 64)
    out = fork_attention(Q, K, V, 4096, 16, 64)

    cos_sim = _cosine_sim(out, ref)
    max_err = _max_abs_err(out, ref)
    assert cos_sim > 0.999, f"cos_sim={cos_sim:.6f}"
    assert max_err < 1e-3, f"max_err={max_err:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
