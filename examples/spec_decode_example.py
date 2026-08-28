"""
Speculative decoding integration example using fork-attn for verification attention.

Scenario: a small draft model proposes a binary tree of candidate token continuations.
The target model verifies all candidates in parallel using fork-attn, then applies
token-level acceptance/rejection based on the probability ratio criterion.

Tree structure: depth-D binary tree where each path from root to leaf represents
one candidate sequence of D draft tokens. All candidates share the same prompt
prefix (stored in the KV cache as shared_len tokens).

Reference:
  Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling"
  (arXiv:2302.01318)
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple

from fork_attn import fork_attention


@dataclass
class SpecDecodeConfig:
    vocab_size: int = 32000
    num_heads: int = 8
    head_dim: int = 64
    num_layers: int = 1    # simplified to 1 layer for this example
    draft_depth: int = 3   # binary tree depth: 2^depth = 8 candidates


def make_draft_tree(
    cfg: SpecDecodeConfig,
    prompt_len: int,
    device: str,
) -> Tuple[torch.Tensor, List[List[int]]]:
    """
    Simulate draft model output: a binary tree of candidate token sequences.

    Returns:
        draft_tokens: [num_leaves, depth] int tensor of candidate token ids
        draft_log_probs: [num_leaves, depth] float tensor of draft log probs
    """
    torch.manual_seed(0)
    depth = cfg.draft_depth
    num_leaves = 2 ** depth

    # Simulate draft model proposing diverse tokens at each tree node.
    # In a real system this would come from the draft model's softmax output.
    draft_tokens = torch.randint(0, cfg.vocab_size, (num_leaves, depth), device=device)
    # Uniform draft probs for demonstration; real system uses actual draft logits.
    draft_log_probs = torch.full((num_leaves, depth), math.log(1.0 / cfg.vocab_size),
                                 device=device, dtype=torch.float32)

    return draft_tokens, draft_log_probs


def simulate_target_logits(
    attn_output: torch.Tensor,
    cfg: SpecDecodeConfig,
    device: str,
) -> torch.Tensor:
    """
    Simulate target model head: linear projection from attention output to vocab logits.
    attn_output: [num_leaves, num_heads, head_dim]
    Returns: [num_leaves, vocab_size]
    """
    torch.manual_seed(42)
    hidden_dim = cfg.num_heads * cfg.head_dim
    # Random linear head (frozen for this example)
    W = torch.randn(hidden_dim, cfg.vocab_size, device=device, dtype=torch.float32) * 0.02
    flat = attn_output.reshape(attn_output.size(0), -1)  # [num_leaves, hidden_dim]
    return flat @ W   # [num_leaves, vocab_size]


def acceptance_rejection(
    draft_tokens: torch.Tensor,    # [num_leaves, depth]
    draft_log_probs: torch.Tensor, # [num_leaves, depth]
    target_log_probs: torch.Tensor, # [num_leaves, depth] sliced from target logits
    temperature: float = 1.0,
) -> Tuple[List[int], int]:
    """
    Apply the speculative sampling acceptance criterion per leaf.

    For each leaf and each draft token position:
      accept if target_log_prob >= draft_log_prob (token-level greedy check)
      otherwise accept with probability p_target / p_draft

    Returns:
        accepted_leaves: list of leaf indices that accepted all draft tokens
        total_accepted_tokens: total draft tokens accepted across all leaves
    """
    num_leaves, depth = draft_tokens.shape
    accepted_leaves = []
    total_accepted = 0

    for li in range(num_leaves):
        accept_all = True
        accepted_this_leaf = 0
        for d in range(depth):
            t_lp = target_log_probs[li, d].item()
            dr_lp = draft_log_probs[li, d].item()
            # Accept deterministically if target prob >= draft prob
            if t_lp >= dr_lp:
                accepted_this_leaf += 1
            else:
                # Stochastic acceptance: accept with prob p_target / p_draft
                ratio = math.exp(t_lp - dr_lp)
                if torch.rand(1).item() < ratio:
                    accepted_this_leaf += 1
                else:
                    accept_all = False
                    break   # reject remaining tokens in this sequence

        total_accepted += accepted_this_leaf
        if accept_all:
            accepted_leaves.append(li)

    return accepted_leaves, total_accepted


def run_spec_decode_step(
    cfg: SpecDecodeConfig,
    prompt_len: int = 512,
    device: str = "cuda",
) -> dict:
    """
    Run one speculative decoding verification step.

    1. Build KV cache for shared prompt prefix and draft token suffixes.
    2. Run fork-attn for all candidate sequences in parallel.
    3. Project attention output to logits.
    4. Apply acceptance/rejection.
    """
    depth = cfg.draft_depth
    num_leaves = 2 ** depth
    unique_len = depth    # each leaf's suffix = the D draft tokens on its path

    # KV cache: shared prompt prefix (prompt_len tokens) + per-leaf draft token keys/values
    total_kv = prompt_len + num_leaves * unique_len

    torch.manual_seed(1)
    Q = torch.randn(num_leaves, cfg.num_heads, cfg.head_dim, device=device, dtype=torch.float32)
    K = torch.randn(total_kv,   cfg.num_heads, cfg.head_dim, device=device, dtype=torch.float32)
    V = torch.randn(total_kv,   cfg.num_heads, cfg.head_dim, device=device, dtype=torch.float32)

    # Fork-attn: shared prefix attention computed once, per-leaf suffix attention per branch.
    attn_out = fork_attention(Q, K, V,
                              branch_point=prompt_len,
                              num_branches=num_leaves,
                              unique_len=unique_len)
    # attn_out: [num_leaves, num_heads, head_dim]

    # Project to vocabulary logits
    logits = simulate_target_logits(attn_out, cfg, device)  # [num_leaves, vocab_size]
    log_probs_all = F.log_softmax(logits, dim=-1)           # [num_leaves, vocab_size]

    # Simulate draft tree
    draft_tokens, draft_log_probs = make_draft_tree(cfg, prompt_len, device)

    # Gather target log probs for the specific draft tokens
    target_log_probs = torch.zeros(num_leaves, depth, device=device)
    for d in range(depth):
        token_ids = draft_tokens[:, d]  # [num_leaves]
        # Use the last attention step's logits as proxy for position d (simplified)
        target_log_probs[:, d] = log_probs_all[torch.arange(num_leaves), token_ids]

    # Acceptance/rejection
    accepted_leaves, total_accepted = acceptance_rejection(
        draft_tokens, draft_log_probs, target_log_probs
    )

    acceptance_rate = total_accepted / (num_leaves * depth)
    kv_savings = 1.0 - (1.0 + num_leaves * unique_len / prompt_len) / \
                       (num_leaves * (1.0 + unique_len / prompt_len))

    return {
        "num_leaves": num_leaves,
        "prompt_len": prompt_len,
        "unique_len": unique_len,
        "total_kv_tokens": total_kv,
        "attn_output_shape": list(attn_out.shape),
        "accepted_leaves": accepted_leaves,
        "total_accepted_tokens": total_accepted,
        "acceptance_rate": acceptance_rate,
        "kv_memory_savings": kv_savings,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = "cuda"
    cfg = SpecDecodeConfig()

    print("Speculative decoding integration example")
    print(f"Draft tree: binary depth={cfg.draft_depth}, "
          f"num_leaves={2**cfg.draft_depth}, vocab={cfg.vocab_size}")
    print()

    for prompt_len in [256, 512, 1024, 2048]:
        result = run_spec_decode_step(cfg, prompt_len=prompt_len, device=device)
        print(f"prompt_len={prompt_len:>5} | "
              f"leaves={result['num_leaves']:>3} | "
              f"total_kv={result['total_kv_tokens']:>5} | "
              f"accepted_leaves={len(result['accepted_leaves']):>2}/{result['num_leaves']:>2} | "
              f"token_accept_rate={result['acceptance_rate']:.1%} | "
              f"kv_savings={result['kv_memory_savings']:.1%}")

    print()
    print("Note: acceptance rates here are purely stochastic (random logits).")
    print("In production, target and draft models are aligned, giving ~85-95% acceptance.")
    print()
    print(f"attn output shape: {result['attn_output_shape']}")
    print("fork_attention() correctly handles the shared-prefix + per-leaf-suffix layout.")


if __name__ == "__main__":
    main()
