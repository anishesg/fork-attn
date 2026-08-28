"""fork_attn: fused tree-structured decode attention with shared-prefix KV computation."""

import torch
from typing import Optional

try:
    from fork_attn import _C as _fork_attn_C
    _EXTENSION_LOADED = True
except ImportError:
    _EXTENSION_LOADED = False


def fork_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    branch_point: int,
    num_branches: int,
    unique_len: int,
) -> torch.Tensor:
    """
    Compute tree-structured attention for speculative decoding verification.

    The KV-cache layout is: [shared_prefix | branch_0_suffix | branch_1_suffix | ...]
    where shared_prefix has length branch_point and each branch suffix has length unique_len.

    Args:
        Q: Query tensor of shape [num_branches, num_heads, head_dim].
        K: Key tensor of shape [branch_point + num_branches * unique_len, num_heads, head_dim].
        V: Value tensor of shape [branch_point + num_branches * unique_len, num_heads, head_dim].
        branch_point: Number of shared prefix tokens.
        num_branches: Number of diverging branches.
        unique_len: Number of unique suffix tokens per branch.

    Returns:
        Output tensor of shape [num_branches, num_heads, head_dim].
    """
    if not _EXTENSION_LOADED:
        raise RuntimeError(
            "fork_attn C extension not found. Run 'pip install -e .' to build it."
        )

    if not Q.is_cuda:
        raise ValueError("Q must be a CUDA tensor")
    if Q.dtype != torch.float32:
        raise ValueError("Q must be float32")
    if not Q.is_contiguous():
        Q = Q.contiguous()
    if not K.is_contiguous():
        K = K.contiguous()
    if not V.is_contiguous():
        V = V.contiguous()

    return _fork_attn_C.fork_attention(Q, K, V, num_branches, branch_point, unique_len)


def fork_attention_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    branch_point: int,
    num_branches: int,
    unique_len: int,
) -> torch.Tensor:
    """
    Pure PyTorch reference implementation using scaled dot-product attention.

    Applies torch.nn.functional.scaled_dot_product_attention per branch,
    attending to the shared prefix plus that branch's unique suffix.
    """
    num_heads = Q.size(1)
    head_dim = Q.size(2)
    outputs = []

    for b in range(num_branches):
        # Each branch attends to: shared_prefix + branch_b_suffix
        shared_K = K[:branch_point]         # [branch_point, num_heads, head_dim]
        shared_V = V[:branch_point]

        suffix_start = branch_point + b * unique_len
        suffix_end = suffix_start + unique_len
        branch_K = K[suffix_start:suffix_end]  # [unique_len, num_heads, head_dim]
        branch_V = V[suffix_start:suffix_end]

        full_K = torch.cat([shared_K, branch_K], dim=0)  # [branch_point+unique_len, num_heads, head_dim]
        full_V = torch.cat([shared_V, branch_V], dim=0)

        # Reshape to [num_heads, seq_len, head_dim] for sdpa
        q = Q[b].unsqueeze(1)                       # [num_heads, 1, head_dim]
        k = full_K.permute(1, 0, 2)                 # [num_heads, seq_len, head_dim]
        v = full_V.permute(1, 0, 2)                 # [num_heads, seq_len, head_dim]

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # [num_heads, 1, head_dim]
        outputs.append(out.squeeze(1))              # [num_heads, head_dim]

    return torch.stack(outputs, dim=0)              # [num_branches, num_heads, head_dim]
