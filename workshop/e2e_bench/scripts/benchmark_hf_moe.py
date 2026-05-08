#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Time the **HuggingFace Transformers native** MoE block (the canonical
Python-loop "for expert_idx in expert_hit: ..." implementation) at
shapes that match the vLLM modular MoE bench in REPORT.md §4f-g.

This is a single-GPU benchmark (no expert parallelism, no collectives).
Compared to vLLM's `--world-size 1 --routing-pattern oracle_uniform`
run, this isolates the **kernel-implementation tax** of the naive
expert-by-expert PyTorch loop versus vLLM's fused Triton grouped-GEMM
kernels.

What it does:

1. Imports `Qwen3MoeSparseMoeBlock` from the installed transformers
   package (modeling_qwen3_moe.py — see source for the exact loop).
2. Instantiates it with a config derived from
   `/home/yyx/models/Qwen3-30B-A3B/config.json`, with random small
   weights (the actual *imbalance characteristics* don't matter for
   this measurement — we just want the kernel cost).
3. Runs N iters of forward, optionally for L "stack repeats" so we can
   report a 48-layer-equivalent step time directly comparable to the
   vLLM bench output.

Usage:

    python benchmark_hf_moe.py \\
        --model-config /home/yyx/models/Qwen3-30B-A3B/config.json \\
        --tokens 4096 --num-layers 48 \\
        --warmup 3 --iters 10 \\
        --output results/hf_moe_qwen3_M4096.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def _load_qwen3_config(path: Path):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(path.parent), trust_remote_code=True)
    # Fall back to plain dict if HF gives us a vision config wrapper.
    text = getattr(cfg, "text_config", cfg)
    return text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", type=Path, required=True,
                   help="Path to a HF config.json (or its parent dir).")
    p.add_argument("--tokens", type=int, required=True,
                   help="Number of tokens to push through one MoE block per "
                        "iter (matches per-rank token count in vLLM bench).")
    p.add_argument("--num-layers", type=int, default=48,
                   help="How many MoE blocks (with shared weights) to stack "
                        "per timed iter. Default 48 to match Qwen3-30B-A3B.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--implementation",
        type=str,
        default=None,
        help=(
            "transformers v5 only: which kernel to use for the experts "
            "forward. Choices: 'eager' (Python loop, default), 'grouped_mm' "
            "(sort-by-expert + grouped GEMM, EP-ready via sentinel), "
            "'batched_mm' (batched per-(token,slot) GEMM). For transformers "
            "v4 this flag has no effect."
        ),
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cfg_path = args.model_config
    if cfg_path.is_dir():
        cfg_path = cfg_path / "config.json"
    text_config = _load_qwen3_config(cfg_path)

    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeSparseMoeBlock,
    )
    import transformers as _tfm
    tfm_major = int(_tfm.__version__.split(".")[0])

    if args.implementation is not None:
        if tfm_major < 5:
            print(
                f"[hf-moe] WARNING: --implementation={args.implementation} "
                "ignored (transformers < 5)"
            )
        else:
            text_config._experts_implementation = args.implementation

    impl_name = (
        getattr(text_config, "_experts_implementation", None)
        if tfm_major >= 5
        else "eager_v4"
    )

    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    M = args.tokens

    print(
        f"[hf-moe] tfm={_tfm.__version__} impl={impl_name} "
        f"model={getattr(text_config, 'model_type', '?')} "
        f"H={text_config.hidden_size} E={text_config.num_experts} "
        f"topk={text_config.num_experts_per_tok} "
        f"moe_int={text_config.moe_intermediate_size} "
        f"layers={args.num_layers} tokens={M} dtype={dtype} device={device}"
    )

    block = Qwen3MoeSparseMoeBlock(text_config).to(device=device, dtype=dtype)
    # Tiny random init (HF default is fine, but we shrink for numerical
    # stability of bf16 matmul).
    with torch.no_grad():
        for p_ in block.parameters():
            p_.normal_(0.0, 0.02)
    block.eval()

    x_in = (
        torch.randn((1, M, text_config.hidden_size), device=device, dtype=dtype)
        / 10.0
    )

    @torch.no_grad()
    def run_stack() -> None:
        x = x_in
        for _ in range(args.num_layers):
            out = block(x)
            x = out[0] if isinstance(out, tuple) else out

    # Warmup
    for _ in range(args.warmup):
        run_stack()
    torch.cuda.synchronize(device)

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
                out = block(x)
                x = out[0] if isinstance(out, tuple) else out
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

    flops_per_token = (
        text_config.num_experts_per_tok
        * (
            2 * (2 * text_config.hidden_size * text_config.moe_intermediate_size)
            + 2 * (text_config.moe_intermediate_size * text_config.hidden_size)
        )
    )
    total_flops = flops_per_token * M * args.num_layers
    achieved_tflops = total_flops / max(total_mean, 1e-9) / 1e9

    result = {
        "impl": "hf_qwen3_moe_sparse_block",
        "transformers_version": _tfm.__version__,
        "experts_implementation": impl_name,
        "hidden_size": text_config.hidden_size,
        "moe_intermediate_size": text_config.moe_intermediate_size,
        "num_experts": text_config.num_experts,
        "topk": text_config.num_experts_per_tok,
        "num_layers": args.num_layers,
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
        f"[hf-moe] mean per-layer = {statistics.mean(per_layer_mean):.2f} ms, "
        f"total = {total_mean:.1f} ± {total_std:.1f} ms, "
        f"throughput = {achieved_tflops:.2f} TFLOP/s "
        f"(routing implicit via random gate softmax + topk)"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        print(f"[hf-moe] appended to {args.output}")


if __name__ == "__main__":
    main()
