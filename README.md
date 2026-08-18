# fork-attn

Fused tree-structured decode attention: shared-prefix KV computation with single-pass multi-branch output.

## Problem

Tree-structured decoding (beam search, best-of-N sampling, MCTS reasoning trees) generates multiple candidate sequences that share a common prompt prefix. Standard serving systems compute attention over the full KV-cache independently for each branch, redundantly processing the shared prefix on every forward pass.

For a tree with B branches and a shared prefix of length L, naive decode attention costs O(B * L) KV-cache reads. When L >> unique suffix length (common in long-context serving), this is almost entirely wasted work.

## Insight

Attention over independent KV regions can be fused via partial softmax merge. For a query q attending to keys [K_shared | K_unique_b], the final output is:

```
attn(q, [K_shared | K_unique_b], [V_shared | V_unique_b])
  = merge(partial_attn(q, K_shared, V_shared),
          partial_attn(q, K_unique_b, V_unique_b))
```

where merge combines two partial softmax states (running max + weighted sum) via log-sum-exp. This reduces KV-cache reads for the shared prefix to O(L) regardless of branch count.

## Approach

1. **Shared prefix pass**: compute partial softmax state over K_shared, V_shared once per query. Result is a (running_max, exp_sum, weighted_value_sum) tuple, not a normalized output.

2. **Per-branch suffix pass**: for each branch b, compute partial softmax state over K_unique_b, V_unique_b independently.

3. **Merge**: combine shared and per-branch partial states via log-sum-exp to produce normalized attention output for each branch.

## Complexity

| Method         | KV reads (prefix) | KV reads (suffix) |
|----------------|-------------------|-------------------|
| Naive          | O(B * L)          | O(B * U)          |
| fork-attn      | O(L)              | O(B * U)          |

Speedup is proportional to (B * L) / (L + B * U). For B=8, L=4096, U=64: ~6.4x reduction in KV reads.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./test_correctness
./bench_latency
```

Requires CUDA toolkit >= 11.8, GPU with sm_80+ (A100, H100, RTX 3090+).

## Files

```
src/
  tree_topology.cuh        -- tree structure descriptor
  partial_softmax.cuh      -- partial softmax state and log-sum-exp merge
  naive_decode_attn.cu     -- baseline: independent attention per branch
  shared_prefix_attn.cu    -- pass 1: shared prefix partial softmax
  branch_suffix_attn.cu    -- pass 2: per-branch suffix partial softmax
  fork_attn.cu             -- host orchestration: launch + merge passes
tests/
  test_correctness.cu      -- compare fork-attn vs naive, cosine sim + max abs err
  bench_latency.cu         -- sweep branch count and prefix length
```
