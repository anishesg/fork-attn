"""
KV-cache memory savings analysis for tree-structured attention.

Computes bytes saved by prefix sharing versus full KV duplication,
printing a table of savings ratios across tree configurations.

Definitions:
  - full_bytes: KV bytes if every branch stores its full context independently
  - shared_bytes: KV bytes under prefix sharing (shared segment stored once)
  - savings_ratio: 1 - shared_bytes / full_bytes (fraction of bytes avoided)

Usage:
    python benchmarks/memory_analysis.py
"""

from typing import List


def kv_bytes_full(
    num_branches: int,
    context_len: int,
    num_heads: int,
    head_dim: int,
    bytes_per_elem: int = 2,  # fp16
) -> int:
    """KV bytes if every branch stores a full independent context."""
    tokens_per_branch = context_len
    return 2 * num_branches * tokens_per_branch * num_heads * head_dim * bytes_per_elem


def kv_bytes_shared(
    num_branches: int,
    shared_len: int,
    unique_len: int,
    num_heads: int,
    head_dim: int,
    bytes_per_elem: int = 2,
) -> int:
    """KV bytes under two-level prefix sharing (one copy of shared prefix)."""
    total_tokens = shared_len + num_branches * unique_len
    return 2 * total_tokens * num_heads * head_dim * bytes_per_elem


def kv_bytes_multilevel(
    branching_factor: int,
    depth: int,
    seg_len: int,
    num_heads: int,
    head_dim: int,
    bytes_per_elem: int = 2,
) -> dict:
    """
    KV bytes for a uniform k-ary tree of given depth, each node having seg_len tokens.
    Returns both shared and full-duplication byte counts.
    """
    # Number of nodes at each level (0 = root, depth = leaves)
    nodes_per_level = [branching_factor ** d for d in range(depth + 1)]
    total_nodes = sum(nodes_per_level)
    num_leaves = branching_factor ** depth

    # Shared: one copy of each node's segment
    total_tokens_shared = total_nodes * seg_len
    shared_bytes = 2 * total_tokens_shared * num_heads * head_dim * bytes_per_elem

    # Full duplication: each leaf stores its entire root-to-leaf context
    tokens_per_leaf = (depth + 1) * seg_len  # path length = depth+1 nodes
    total_tokens_full = num_leaves * tokens_per_leaf
    full_bytes = 2 * total_tokens_full * num_heads * head_dim * bytes_per_elem

    return {
        "shared_bytes": shared_bytes,
        "full_bytes": full_bytes,
        "total_nodes": total_nodes,
        "num_leaves": num_leaves,
        "savings_ratio": 1.0 - shared_bytes / full_bytes if full_bytes > 0 else 0.0,
    }


def fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f}GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f}MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.2f}KB"
    return f"{n}B"


def print_two_level_table(num_heads: int = 32, head_dim: int = 128) -> None:
    """Table: two-level trees varying branches and shared_len."""
    print("=== Two-level tree: KV memory savings by branch count and shared prefix ===")
    print(f"Config: num_heads={num_heads}, head_dim={head_dim}, unique_len=64, fp16\n")

    header = f"{'branches':>10} {'shared_len':>12} {'context_len':>12} {'full':>10} {'shared':>10} {'savings':>9}"
    print(header)
    print("-" * len(header))

    unique_len = 64
    for num_branches in [2, 4, 8, 16, 32]:
        for shared_len in [512, 1024, 2048, 4096]:
            context_len = shared_len + unique_len
            full_b = kv_bytes_full(num_branches, context_len, num_heads, head_dim)
            shar_b = kv_bytes_shared(num_branches, shared_len, unique_len, num_heads, head_dim)
            ratio = 1.0 - shar_b / full_b
            print(
                f"{num_branches:>10} {shared_len:>12} {context_len:>12} "
                f"{fmt_bytes(full_b):>10} {fmt_bytes(shar_b):>10} {ratio:>8.1%}"
            )
    print()


def print_multilevel_table(num_heads: int = 32, head_dim: int = 128) -> None:
    """Table: multi-level trees varying depth and branching factor."""
    print("=== Multi-level tree: KV memory savings by depth and branching factor ===")
    print(f"Config: num_heads={num_heads}, head_dim={head_dim}, seg_len=256, fp16\n")

    header = (
        f"{'bf':>4} {'depth':>6} {'leaves':>8} {'nodes':>7} "
        f"{'full':>10} {'shared':>10} {'savings':>9}"
    )
    print(header)
    print("-" * len(header))

    seg_len = 256
    for bf in [2, 3, 4]:
        for depth in range(1, 7):
            leaves = bf ** depth
            if leaves > 128:
                break
            result = kv_bytes_multilevel(bf, depth, seg_len, num_heads, head_dim)
            print(
                f"{bf:>4} {depth:>6} {result['num_leaves']:>8} {result['total_nodes']:>7} "
                f"{fmt_bytes(result['full_bytes']):>10} {fmt_bytes(result['shared_bytes']):>10} "
                f"{result['savings_ratio']:>8.1%}"
            )
        print()


if __name__ == "__main__":
    print_two_level_table()
    print_multilevel_table()

    # Summary: which configs save the most
    print("=== Key insight ===")
    print("Savings ratio = 1 - (1 + B*u/L) / (B * (1 + u/L))")
    print("where B=branches, L=shared_len, u=unique_len.")
    print("Savings approach (1 - 1/B) as L >> u (long shared prefix).")
    print("With B=16, maximum possible savings = 93.75%.")
    print()
    # Verify formula at extremes
    for B in [2, 4, 8, 16, 32]:
        asymptotic = 1.0 - 1.0 / B
        result = kv_bytes_multilevel(B, 1, 10000, 1, 1)
        print(f"  B={B:>2}: asymptotic savings={asymptotic:.2%}, "
              f"measured at seg_len=10000: {result['savings_ratio']:.2%}")
