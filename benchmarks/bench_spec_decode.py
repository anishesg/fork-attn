"""
Benchmark fork-attn vs naive attention for speculative decoding tree shapes.

Tree topology: complete binary trees of depth 3-6 representing verification
of 8 to 64 candidate continuations. Each leaf is one draft token sequence
to verify in parallel.

In speculative decoding, the verifier (target model) must run attention for
each candidate token position. With a depth-D binary tree, there are 2^D
candidate sequences, each sharing the first (prefix_len - D) tokens with
all others but diverging earlier branches with fewer peers.

Usage:
    pip install -e .
    python benchmarks/bench_spec_decode.py
"""

import math
import time
import argparse
import torch
from fork_attn import fork_attention, fork_attention_reference


def time_fn(fn, warmup: int = 5, iters: int = 20) -> float:
    """Return mean latency in milliseconds over `iters` runs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / iters * 1000.0


def bench_binary_tree(
    depth: int,
    shared_len: int,
    unique_len: int,
    num_heads: int,
    head_dim: int,
    device: str = "cuda",
) -> dict:
    """
    Benchmark fork-attn vs naive for a binary tree of given depth.

    A binary tree of depth D has 2^D leaves. Each leaf attends to:
      shared_len prefix tokens + unique_len branch-specific tokens.

    For fork-attn we model this as a two-level tree (shared + per-branch unique),
    which is the base case; deeper trees would use multi_level_fork_attn.
    """
    num_branches = 2 ** depth
    total_kv = shared_len + num_branches * unique_len

    Q = torch.randn(num_branches, num_heads, head_dim, device=device, dtype=torch.float32)
    K = torch.randn(total_kv,     num_heads, head_dim, device=device, dtype=torch.float32)
    V = torch.randn(total_kv,     num_heads, head_dim, device=device, dtype=torch.float32)

    fork_ms = time_fn(lambda: fork_attention(Q, K, V, shared_len, num_branches, unique_len))
    naive_ms = time_fn(lambda: fork_attention_reference(Q, K, V, shared_len, num_branches, unique_len))

    speedup = naive_ms / fork_ms if fork_ms > 0 else float("inf")

    return {
        "depth": depth,
        "num_branches": num_branches,
        "shared_len": shared_len,
        "unique_len": unique_len,
        "fork_ms": fork_ms,
        "naive_ms": naive_ms,
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser(description="Speculative decode tree shape benchmark")
    parser.add_argument("--num-heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--head-dim",  type=int, default=128, help="Head dimension")
    parser.add_argument("--shared-len", type=int, default=2048,
                        help="Shared prefix length (verified tokens so far)")
    parser.add_argument("--unique-len", type=int, default=1,
                        help="Unique suffix tokens per branch (draft step being verified)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"Config: num_heads={args.num_heads}, head_dim={args.head_dim}")
    print(f"        shared_len={args.shared_len}, unique_len={args.unique_len}")
    print()

    header = (
        f"{'depth':>6} {'branches':>9} {'shared_len':>11} "
        f"{'fork_ms':>10} {'naive_ms':>10} {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    for depth in range(3, 7):
        result = bench_binary_tree(
            depth=depth,
            shared_len=args.shared_len,
            unique_len=args.unique_len,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
        )
        print(
            f"{result['depth']:>6} {result['num_branches']:>9} {result['shared_len']:>11} "
            f"{result['fork_ms']:>9.3f}ms {result['naive_ms']:>9.3f}ms "
            f"{result['speedup']:>7.2f}x"
        )

    print()
    print("Interpretation: fork-attn computes the shared prefix once per tree,")
    print("while naive recomputes it for every branch. Speedup grows with branch count.")


if __name__ == "__main__":
    main()
