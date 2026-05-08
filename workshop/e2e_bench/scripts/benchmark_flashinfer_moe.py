#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU MoE-block benchmark using FlashInfer's CUTLASS fused MoE
kernel (``flashinfer.fused_moe.cutlass_fused_moe``).

Counterpart to ``benchmark_hf_moe.py`` — same Qwen3-30B-A3B-sized layer
(H=2048, I=768, E=128, top_k=8), same "stack the same block N layers
times" loop, BF16 dense weights (no FP8 / NVFP4 quantisation), single
GPU, no expert-parallel collectives. The only thing changed is which
GEMM kernel performs the per-expert ``up_proj`` / ``down_proj``.

Routing patterns:

* ``--routing oracle_uniform`` (default) — perfectly balanced
  round-robin top-k (``logical_cv = physical_cv = 0``). Matches the
  oracle baseline used in the vLLM bench.
* ``--routing softmax`` — random gate logits → softmax+topk, gives
  roughly the same kind of imbalance the v5/v4 native HF benches see.

Output JSONL has the same fields as ``benchmark_hf_moe.py`` so plotting
scripts can ingest both.

Usage::

    CUDA_HOME=/usr/local/cuda-13.0 \\
    PATH=$CUDA_HOME/bin:$PATH \\
    python benchmark_flashinfer_moe.py \\
        --model-config /home/yyx/models/Qwen3-30B-A3B/config.json \\
        --tokens 4096 --num-layers 48 --warmup 3 --iters 10 \\
        --routing oracle_uniform \\
        --output results/flashinfer_qwen3_M4096_oracle.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import flashinfer
import flashinfer.fused_moe as fused_moe


def _load_dims(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    text = cfg.get("text_config", cfg)
    return {
        "hidden_size": int(text["hidden_size"]),
        "moe_intermediate_size": int(text["moe_intermediate_size"]),
        "num_experts": int(text["num_experts"]),
        "topk": int(text["num_experts_per_tok"]),
        "num_layers": int(text["num_hidden_layers"]),
    }


def _make_oracle_uniform(M: int, E: int, K: int, dev: torch.device):
    """Round-robin assignment so every expert receives the same number
    of token-slots (K * M / E per expert). Matches the oracle pattern in
    benchmark_eplb_multigpu.py.
    """
    base = torch.arange(M * K, device=dev) % E
    ids = base.view(M, K).to(torch.int)
    weights = torch.full((M, K), 1.0 / K, device=dev, dtype=torch.float32)
    return ids, weights


def _make_softmax_routing(M: int, E: int, K: int, dev: torch.device):
    logits = torch.randn(M, E, device=dev, dtype=torch.float32)
    w = F.softmax(logits, dim=-1)
    w, ids = torch.topk(w, K, dim=-1)
    w = (w / w.sum(dim=-1, keepdim=True)).float()
    return ids.to(torch.int), w


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--tokens", type=int, required=True,
                   help="Tokens fed into one MoE block per iter (matches "
                        "vLLM per-rank token count).")
    p.add_argument("--num-layers", type=int, default=48)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--routing", choices=["oracle_uniform", "softmax"],
                   default="oracle_uniform")
    p.add_argument("--autotune", action="store_true",
                   help="Run flashinfer's autotuner once before timing.")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    cfg_path = args.model_config
    if cfg_path.is_dir():
        cfg_path = cfg_path / "config.json"
    dims = _load_dims(cfg_path)
    H = dims["hidden_size"]
    I = dims["moe_intermediate_size"]
    E = dims["num_experts"]
    K = dims["topk"]

    dev = torch.device(args.device)
    M = args.tokens

    print(
        f"[fi-moe] flashinfer={flashinfer.__version__} "
        f"capability={torch.cuda.get_device_capability(dev)} "
        f"H={H} I={I} E={E} K={K} layers={args.num_layers} "
        f"tokens={M} routing={args.routing} dtype=bf16 device={dev}"
    )

    dtype = torch.bfloat16
    x_in = (torch.randn(M, H, device=dev, dtype=dtype) / 5).contiguous()
    w31 = (torch.randn(E, 2 * I, H, device=dev, dtype=dtype) / 16).contiguous()
    w2 = (torch.randn(E, H, I, device=dev, dtype=dtype) / 16).contiguous()

    if args.routing == "oracle_uniform":
        ids, weights = _make_oracle_uniform(M, E, K, dev)
    else:
        ids, weights = _make_softmax_routing(M, E, K, dev)

    out_buf = torch.empty_like(x_in)

    def call_moe(x: torch.Tensor) -> torch.Tensor:
        res = fused_moe.cutlass_fused_moe(
            x,
            ids,
            weights,
            w31,
            w2,
            x.dtype,
            output=out_buf,
            quant_scales=None,
            tune_max_num_tokens=max(8192, M),
        )
        return res[0] if isinstance(res, (list, tuple)) else res

    if args.autotune:
        from flashinfer.autotuner import AutoTuner, autotune
        AutoTuner.get().clear_cache()
        with torch.inference_mode(), autotune(True):
            _ = call_moe(x_in)

    @torch.no_grad()
    def run_stack() -> None:
        x = x_in
        for _ in range(args.num_layers):
            x = call_moe(x)

    for _ in range(args.warmup):
        run_stack()
    torch.cuda.synchronize(dev)

    per_layer_lat: list[list[float]] = [[] for _ in range(args.num_layers)]
    total_lat: list[float] = []
    with torch.no_grad():
        for _ in range(args.iters):
            st_t = torch.cuda.Event(enable_timing=True)
            ed_t = torch.cuda.Event(enable_timing=True)
            st_t.record()
            x = x_in
            for ll in range(args.num_layers):
                st = torch.cuda.Event(enable_timing=True)
                ed = torch.cuda.Event(enable_timing=True)
                st.record()
                x = call_moe(x)
                ed.record()
                ed.synchronize()
                per_layer_lat[ll].append(st.elapsed_time(ed))
            ed_t.record()
            ed_t.synchronize()
            total_lat.append(st_t.elapsed_time(ed_t))

    per_layer_mean = [statistics.mean(t) for t in per_layer_lat]
    per_layer_std = [
        statistics.stdev(t) if len(t) > 1 else 0.0 for t in per_layer_lat
    ]
    total_mean = statistics.mean(total_lat)
    total_std = statistics.stdev(total_lat) if len(total_lat) > 1 else 0.0

    flops_per_token = K * (
        2 * (2 * H * I)   # gate+up GEMM
        + 2 * (I * H)     # down GEMM
    )
    total_flops = flops_per_token * M * args.num_layers
    achieved_tflops = total_flops / max(total_mean / 1e3, 1e-9) / 1e12

    result = {
        "impl": "flashinfer_cutlass_fused_moe",
        "flashinfer_version": flashinfer.__version__,
        "routing": args.routing,
        "hidden_size": H,
        "moe_intermediate_size": I,
        "num_experts": E,
        "topk": K,
        "num_layers": args.num_layers,
        "tokens": M,
        "iters": args.iters,
        "warmup": args.warmup,
        "autotune": args.autotune,
        "per_layer_mean_ms": per_layer_mean,
        "per_layer_std_ms": per_layer_std,
        "total_step_mean_ms": total_mean,
        "total_step_std_ms": total_std,
        "flops_per_token": flops_per_token,
        "total_flops": total_flops,
        "achieved_tflops_per_s": achieved_tflops,
        "timestamp": time.time(),
    }

    print(
        f"[fi-moe] mean per-layer = {statistics.mean(per_layer_mean):.2f} ms, "
        f"total = {total_mean:.1f} ± {total_std:.1f} ms, "
        f"throughput = {achieved_tflops:.2f} TFLOP/s"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        print(f"[fi-moe] appended to {args.output}")


if __name__ == "__main__":
    main()
