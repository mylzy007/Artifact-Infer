#!/usr/bin/env python3
"""Smoke-test FlashInfer cutlass_fused_moe with BF16 on a Qwen3-30B-A3B
sized layer (single MoE block, single GPU).

Goal: verify that the CUTLASS BF16 kernel actually compiles + runs on
SM89 (RTX 4090) before wiring the full multi-layer benchmark.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import flashinfer.fused_moe as fused_moe


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=768)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(0)
    dev = torch.device(args.device)
    dtype = torch.bfloat16
    M, H, I = args.tokens, args.hidden, args.inter
    E, K = args.experts, args.topk

    print(
        f"[smoke] M={M} H={H} I={I} E={E} K={K} dtype={dtype} dev={dev}"
    )
    print(f"[smoke] cuda capability: {torch.cuda.get_device_capability(dev)}")

    x = (torch.randn(M, H, dtype=dtype, device=dev) / 5).contiguous()
    w31 = (torch.randn(E, 2 * I, H, dtype=dtype, device=dev) / 16).contiguous()
    w2 = (torch.randn(E, H, I, dtype=dtype, device=dev) / 16).contiguous()

    router_logits = torch.randn(M, E, dtype=torch.float32, device=dev)
    weights = F.softmax(router_logits, dim=-1)
    weights, ids = torch.topk(weights, K, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).float()
    ids = ids.to(torch.int)

    print(f"[smoke] x={x.shape} w31={w31.shape} w2={w2.shape} ids={ids.shape}")

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
    )
    out_t = res[0] if isinstance(res, (list, tuple)) else res
    torch.cuda.synchronize(dev)
    print(f"[smoke] ran ok, out shape {out_t.shape}, dtype {out_t.dtype}")
    print(f"[smoke] sample out[0, :8] = {out_t[0, :8].float().tolist()}")


if __name__ == "__main__":
    main()
