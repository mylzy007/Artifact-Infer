#!/usr/bin/env python3
"""Smoke-test FlashInfer cutlass_fused_moe with ep_size>1 on a single GPU.

Goal: verify that
  (a) passing only a *local* weight slice (E/N experts) plus ep_size=N
      compiles + runs;
  (b) the kernel correctly rebases global expert IDs into the local
      [0, E/N) range using its `MOEParallelismConfig`;
  (c) tokens whose top-k IDs all fall outside this rank's slice produce
      zero output (the kernel zero-masks them).

Run as a single process — we just *pretend* to be rank 0 of an ep_size=8
group by handing the kernel the first E/8 experts and feeding it the full
M tokens with full global routing IDs.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import flashinfer.fused_moe as fused_moe


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=768)
    p.add_argument("--global-experts", type=int, default=128)
    p.add_argument("--ep-size", type=int, default=8)
    p.add_argument("--ep-rank", type=int, default=0)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(7)
    dev = torch.device(args.device)
    dtype = torch.bfloat16
    M, H, I = args.tokens, args.hidden, args.inter
    E_global, N, r, K = args.global_experts, args.ep_size, args.ep_rank, args.topk
    assert E_global % N == 0
    local_E = E_global // N
    expert_lo = r * local_E
    expert_hi = (r + 1) * local_E

    print(
        f"[smoke-ep] M={M} H={H} I={I} E_global={E_global} ep_size={N} "
        f"ep_rank={r} local_E={local_E} expert_range=[{expert_lo},{expert_hi}) "
        f"K={K} dtype={dtype} dev={dev}"
    )

    x = (torch.randn(M, H, dtype=dtype, device=dev) / 5).contiguous()
    # Local slice only — local_E experts, each with full (gate||up, down) weights.
    w31 = (torch.randn(local_E, 2 * I, H, dtype=dtype, device=dev) / 16).contiguous()
    w2 = (torch.randn(local_E, H, I, dtype=dtype, device=dev) / 16).contiguous()

    # Synthetic routing: random topk over the *global* expert pool.
    logits = torch.randn(M, E_global, device=dev, dtype=torch.float32)
    weights = F.softmax(logits, dim=-1)
    weights, ids = torch.topk(weights, K, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).float()
    ids = ids.to(torch.int)

    n_local_slots = ((ids >= expert_lo) & (ids < expert_hi)).sum().item()
    print(f"[smoke-ep] of {M*K} (token,slot) pairs, {n_local_slots} fall in local "
          f"range = {n_local_slots / (M * K):.1%}")

    out = torch.empty_like(x)
    res = fused_moe.cutlass_fused_moe(
        x,
        ids,
        weights,
        w31,
        w2,
        x.dtype,
        output=out,
        quant_scales=None,
        ep_size=N,
        ep_rank=r,
        tune_max_num_tokens=max(8192, M),
    )
    out_t = res[0] if isinstance(res, (list, tuple)) else res
    torch.cuda.synchronize(dev)
    nz = (out_t != 0).any(dim=-1).sum().item()
    print(f"[smoke-ep] kernel ran ok. out shape {out_t.shape} dtype {out_t.dtype}")
    print(f"[smoke-ep] {nz}/{M} tokens have non-zero partial output (i.e. at "
          f"least one of their topk experts is held by this rank)")
    print(f"[smoke-ep] sample out[0, :8] = {out_t[0, :8].float().tolist()}")


if __name__ == "__main__":
    main()
