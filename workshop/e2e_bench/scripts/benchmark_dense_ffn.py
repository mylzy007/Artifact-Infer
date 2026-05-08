#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Stand-alone bf16 dense-FFN benchmark used as the **collective-free GEMM
reference** for the modular MoE kernel comparison in REPORT.md §4f.

For each layer we time one classic SwiGLU FFN:

    h2  =  x @ gate_up.T                       # (M, 2I)
    a, b = h2.chunk(2, dim=-1)                 # (M, I), (M, I)
    h   = silu(a) * b                          # (M, I)
    y   = h @ down.T                           # (M, H)

There is no expert routing, no all2all, no all-gather, no reduce-scatter.
Anything we measure here is pure local compute (GEMMs + activation +
torch overhead). Subtracting this from the modular MoE step time
isolates the "collective + small-M MoE overhead" part.

Two preset shapes are useful in practice:

1. ``--mode shared`` (default) — intermediate dim = the model's
   ``moe_intermediate_size``. This is the "what one routed expert would
   compute if it received every token" reference.
2. ``--mode topk-fused`` — intermediate dim = ``moe_intermediate_size *
   topk``. This collapses all topk routed experts into a single FFN with
   the same total FLOPs as the routed-MoE compute. It's the
   "dense-equivalent compute" reference.

Usage:

    # Qwen3-30B-A3B per-rank slice of the world=8, total_tokens=32768 oracle run:
    python benchmark_dense_ffn.py \\
        --model-config /home/yyx/models/Qwen3-30B-A3B/config.json \\
        --tokens 4096 --num-layers 48 --topk 8 \\
        --mode shared \\
        --output dense_shared_per_rank.jsonl

    python benchmark_dense_ffn.py \\
        --model-config /home/yyx/models/Qwen3-30B-A3B/config.json \\
        --tokens 4096 --num-layers 48 --topk 8 \\
        --mode topk-fused \\
        --output dense_topk_fused_per_rank.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_dims(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    text = cfg.get("text_config", cfg)
    return {
        "hidden_size": int(text["hidden_size"]),
        "moe_intermediate_size": int(text["moe_intermediate_size"]),
        "num_experts": int(text["num_experts"]),
        "num_experts_per_tok": int(text["num_experts_per_tok"]),
        "num_hidden_layers": int(text["num_hidden_layers"])
        + int(text.get("mtp_num_hidden_layers", 0)),
        "model_name": (cfg.get("architectures") or ["custom"])[0],
    }


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", type=Path, default=None)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--intermediate-size", type=int, default=None,
                   help="Override intermediate size; otherwise derived from "
                        "--mode and the model config.")
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--tokens", type=int, required=True,
                   help="Tokens passed through every layer (think 'per-rank "
                        "slice' to compare apples-to-apples with the modular "
                        "MoE bench).")
    p.add_argument("--mode", type=str, default="shared",
                   choices=["shared", "topk-fused"],
                   help="'shared' uses moe_intermediate_size as is; 'topk-fused' "
                        "uses topk * moe_intermediate_size to match routed-MoE "
                        "FLOPs.")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, default=None,
                   help="Optional JSONL output path.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.model_config:
        dims = _load_dims(args.model_config)
        H = args.hidden_size or dims["hidden_size"]
        L = args.num_layers   or dims["num_hidden_layers"]
        topk = args.topk      or dims["num_experts_per_tok"]
        moe_int = dims["moe_intermediate_size"]
        model_name = dims["model_name"]
    else:
        H = args.hidden_size
        L = args.num_layers
        topk = args.topk or 8
        moe_int = args.intermediate_size
        model_name = "custom"
    if H is None or L is None:
        raise ValueError("Need --model-config or both --hidden-size and "
                         "--num-layers")
    if moe_int is None:
        raise ValueError("Need --model-config or --intermediate-size")
    I = args.intermediate_size if args.intermediate_size is not None else moe_int
    if args.mode == "topk-fused":
        I = I * topk

    device = torch.device(args.device)
    dtype = torch.bfloat16
    M = args.tokens

    print(f"[dense] {model_name} mode={args.mode} H={H} I={I} (moe_int={moe_int} "
          f"x topk={topk if args.mode == 'topk-fused' else 1}) layers={L} "
          f"tokens={M} dtype={dtype} device={device}")

    gate_ups = [
        (torch.randn(2 * I, H, device=device, dtype=dtype) / 32).contiguous()
        for _ in range(L)
    ]
    downs = [
        (torch.randn(H, I, device=device, dtype=dtype) / 32).contiguous()
        for _ in range(L)
    ]
    x_in = torch.randn(M, H, device=device, dtype=dtype) / 10

    def stack(x_in_local: torch.Tensor) -> torch.Tensor:
        x = x_in_local
        for ll in range(L):
            x = x @ gate_ups[ll].T
            x = _silu_and_mul(x)
            x = x @ downs[ll].T
        return x

    for _ in range(args.warmup):
        _ = stack(x_in)
    torch.cuda.synchronize(device)

    per_layer_lat: list[list[float]] = [[] for _ in range(L)]
    total_lat: list[float] = []
    for _ in range(args.iters):
        x = x_in
        st_total = torch.cuda.Event(enable_timing=True)
        ed_total = torch.cuda.Event(enable_timing=True)
        st_total.record()
        for ll in range(L):
            st = torch.cuda.Event(enable_timing=True)
            ed = torch.cuda.Event(enable_timing=True)
            st.record()
            x = x @ gate_ups[ll].T
            x = _silu_and_mul(x)
            x = x @ downs[ll].T
            ed.record()
            ed.synchronize()
            per_layer_lat[ll].append(st.elapsed_time(ed))
        ed_total.record()
        ed_total.synchronize()
        total_lat.append(st_total.elapsed_time(ed_total))

    per_layer_mean = [statistics.mean(t) for t in per_layer_lat]
    per_layer_std = [
        statistics.stdev(t) if len(t) > 1 else 0.0 for t in per_layer_lat
    ]
    total_mean = statistics.mean(total_lat)
    total_std = statistics.stdev(total_lat) if len(total_lat) > 1 else 0.0

    flops_per_token = 2 * (2 * H * I) + 2 * (I * H)  # gate_up + down
    total_flops = flops_per_token * M * L
    achieved_tflops = total_flops / max(total_mean, 1e-9) / 1e9  # ms -> 1e3 s, so /1e9 not 1e12

    result = {
        "model_name": model_name,
        "mode": args.mode,
        "hidden_size": H,
        "intermediate_size": I,
        "moe_intermediate_size": moe_int,
        "topk": topk,
        "num_layers": L,
        "tokens": M,
        "iters": args.iters,
        "warmup": args.warmup,
        "per_layer_mean_ms": per_layer_mean,
        "per_layer_std_ms": per_layer_std,
        "total_step_mean_ms": total_mean,
        "total_step_std_ms": total_std,
        "flops_per_token": flops_per_token,
        "total_flops": total_flops,
        "achieved_tflops_per_s": achieved_tflops,
    }
    print(
        f"[dense] mean per-layer = {statistics.mean(per_layer_mean):.3f} ms, "
        f"total = {total_mean:.2f} ± {total_std:.2f} ms, "
        f"throughput = {achieved_tflops:.1f} TFLOP/s"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        print(f"[dense] appended to {args.output}")


if __name__ == "__main__":
    main()
