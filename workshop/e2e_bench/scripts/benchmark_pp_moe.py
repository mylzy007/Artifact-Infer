#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pure pipeline-parallel MoE benchmark using FlashInfer's
``cutlass_fused_moe``.

Each rank holds a contiguous slice of MoE layers and the full expert set
for those layers (no EP within a stage). Microbatches stream through the
pipeline in 1F-only mode (inference), with synchronous
``dist.send`` / ``dist.recv`` between adjacent stages over NCCL.

This is the companion to ``benchmark_flashinfer_native_ep.py`` (§4i.1)
and is referenced by ``COMM_VOLUME_AND_REPLICATION.md`` §9.

Workload:
    M_total = 32 768, num_layers = 48, top_k = 8, hidden = 2048,
    moe_intermediate = 768, num_experts = 128, dtype = bf16,
    routing = oracle uniform (CV = 0).

Usage:
    python benchmark_pp_moe.py \\
        --world-size 8 --tokens 32768 --microbatches 8 \\
        --iters 5 --warmup 2 \\
        --output results/qwen3_heavy/flashinfer_pp.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _make_oracle_uniform(M: int, E: int, K: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Oracle uniform routing: each (token, slot) -> a distinct expert id;
    distribution exactly uniform across the E experts (CV = 0).
    """
    base = torch.arange(M * K, dtype=torch.int32, device=device) % E
    ids = base.reshape(M, K).contiguous()
    weights = torch.full((M, K), 1.0 / K, dtype=torch.float32, device=device)
    return ids, weights


def _worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.port))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    if args.cuda_home:
        os.environ["CUDA_HOME"] = args.cuda_home
        os.environ["PATH"] = f"{args.cuda_home}/bin:" + os.environ.get("PATH", "")
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{args.cuda_home}/lib64:{ld}"

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size,
                            device_id=device)

    # Import flashinfer after CUDA_HOME is set, so JIT picks up the right toolchain.
    from flashinfer import fused_moe

    H = args.hidden_size
    I = args.moe_intermediate_size
    E = args.num_experts
    K = args.topk
    L = args.num_layers
    if L % world_size != 0:
        raise ValueError(f"num_layers={L} must divide world_size={world_size}")
    L_per_stage = L // world_size

    M_total = args.tokens
    B = args.microbatches
    if M_total % B != 0:
        raise ValueError(f"tokens={M_total} must divide microbatches={B}")
    M_micro = M_total // B

    # ---- Allocate this stage's weights (full expert set for L_per_stage layers).
    # Per-layer slice to avoid one giant 4 GiB allocation that can OOM during
    # randn() on a near-full card.
    w31_layers: list[torch.Tensor] = []
    w2_layers: list[torch.Tensor] = []
    for _ in range(L_per_stage):
        w31 = (torch.randn(E, 2 * I, H, dtype=dtype, device=device) / 16).contiguous()
        w2 = (torch.randn(E, H, I, dtype=dtype, device=device) / 16).contiguous()
        w31_layers.append(w31)
        w2_layers.append(w2)

    # Routing precomputed per microbatch (same routing reused across layers,
    # matches the EP benchmark's oracle-uniform definition).
    routings: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(B):
        routings.append(_make_oracle_uniform(M_micro, E, K, device))

    # Stage 0 generates inputs; other stages allocate a recv buffer.
    if rank == 0:
        inputs = [
            (torch.randn(M_micro, H, dtype=dtype, device=device) / 16).contiguous()
            for _ in range(B)
        ]
        recv_buf = None
    else:
        inputs = None
        recv_buf = torch.empty(M_micro, H, dtype=dtype, device=device)

    @torch.no_grad()
    def forward_stage(x: torch.Tensor, routing: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        ids, weights = routing
        for l in range(L_per_stage):
            out = torch.empty_like(x)
            res = fused_moe.cutlass_fused_moe(
                x,
                ids,
                weights,
                w31_layers[l],
                w2_layers[l],
                dtype,
                output=out,
                quant_scales=None,
                tune_max_num_tokens=max(8192, M_micro),
            )
            x = res[0] if isinstance(res, (list, tuple)) else res
        return x

    @torch.no_grad()
    def step() -> None:
        # 1F-only synchronous pipeline. Parallelism comes from running on
        # different ranks; per-stage we just loop over microbatches.
        if rank == 0:
            for mb in range(B):
                x = inputs[mb]  # type: ignore[index]
                x = forward_stage(x, routings[mb])
                if rank < world_size - 1:
                    dist.send(x, dst=rank + 1)
        elif rank == world_size - 1:
            for mb in range(B):
                dist.recv(recv_buf, src=rank - 1)
                _ = forward_stage(recv_buf, routings[mb])
        else:
            for mb in range(B):
                dist.recv(recv_buf, src=rank - 1)
                x = forward_stage(recv_buf, routings[mb])
                dist.send(x, dst=rank + 1)

    # ---- Optional FlashInfer autotune (one warmup pass through the pipeline).
    if args.autotune:
        try:
            from flashinfer.autotuner import autotune  # type: ignore
            with autotune(True):
                step()
                torch.cuda.synchronize(device)
                dist.barrier()
            if rank == 0:
                print("[pp] autotune pass complete")
        except Exception as e:  # pragma: no cover
            if rank == 0:
                print(f"[pp] autotune failed: {e!r}")

    # ---- Warmup
    for _ in range(args.warmup):
        step()
        torch.cuda.synchronize(device)
        dist.barrier()

    # ---- Timed iterations (CUDA event timing per rank, end-to-end barrier
    # before/after each iter).
    per_iter_ms: list[float] = []
    for _ in range(args.iters):
        torch.cuda.synchronize(device)
        dist.barrier()
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        step()
        ev_end.record()
        ev_end.synchronize()
        dist.barrier()
        per_iter_ms.append(ev_start.elapsed_time(ev_end))

    local_mean = statistics.mean(per_iter_ms)
    local_std = statistics.stdev(per_iter_ms) if len(per_iter_ms) > 1 else 0.0

    means: list[float | None] = [None] * world_size
    dist.all_gather_object(means, local_mean)

    if rank == 0:
        means_f = [float(m) for m in means]  # type: ignore[arg-type]
        max_mean = max(means_f)
        # The "step" wallclock is bounded by the slowest rank — typically rank
        # world-1 since it runs last. Report that.
        result = {
            "impl": "flashinfer_pp",
            "world_size": world_size,
            "tokens": M_total,
            "microbatches": B,
            "tokens_per_microbatch": M_micro,
            "num_layers": L,
            "layers_per_stage": L_per_stage,
            "num_experts": E,
            "topk": K,
            "hidden_size": H,
            "moe_intermediate_size": I,
            "dtype": str(dtype),
            "iters": args.iters,
            "warmup": args.warmup,
            "autotune": bool(args.autotune),
            "max_rank_mean_step_ms": max_mean,
            "rank_local_std_step_ms": local_std,
            "per_rank_mean_step_ms": means_f,
            "timestamp": time.time(),
        }
        # Achieved aggregate TFLOP/s (across the whole step).
        flops = (
            M_total * K * 6 * H * I * L
        )  # K = top_k expert evals per token; 3 GEMMs per expert; factor 2 for FMA.
        result["achieved_tflops_per_s"] = flops / (max_mean / 1e3) / 1e12
        print(json.dumps(result, indent=2))
        if args.output:
            outp = Path(args.output)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with outp.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
            print(f"[pp] appended to {outp}")

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--tokens", type=int, default=32768)
    p.add_argument("--microbatches", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=48)
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--moe-intermediate-size", type=int, default=768)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--port", type=int, default=29577)
    p.add_argument("--autotune", action="store_true")
    p.add_argument("--cuda-home", type=str, default=None,
                   help="Force CUDA toolkit path for FlashInfer JIT (e.g., /usr/local/cuda-13.0)")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    mp.spawn(_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
