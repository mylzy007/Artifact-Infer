#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Multi-rank "transformer-native" expert-parallel benchmark.

Drives the transformers v5 ``Qwen3MoeExperts`` kernel
(``config._experts_implementation = "grouped_mm"``) in an end-to-end
expert-parallel pipeline implemented entirely on top of
``torch.distributed`` collectives — no vLLM, no accelerate.

The protocol per timed step is the simplest one that lines up with the
v5 kernel's built-in EP-sentinel support:

    1. Each rank starts with ``--tokens`` (= ``M``) tokens of hidden state.
    2. all-gather hidden states across the ``world_size`` ranks
       → every rank holds ``M * world_size`` tokens.
    3. Every rank runs the (replicated) gate → softmax → top-k on the
       gathered hidden states. Routing decisions are GLOBAL expert ids
       in ``[0, num_experts)``.
    4. Each rank translates global ids → local ids:
         - ids in ``[rank * L, (rank+1) * L)`` become ``id - rank * L``
           (these will be processed by *this* rank's local experts);
         - all other ids become the sentinel ``L`` (i.e. >= local
           ``num_experts``), which v5 ``grouped_mm`` masks out cleanly.
    5. ``Qwen3MoeExperts.forward`` runs grouped GEMM with the local
       ``num_experts = L`` slice. Off-rank slots are zeroed.
    6. all-reduce (sum) the partial outputs across ranks so every rank
       gets the full result for every token.
    7. Every rank slices out its own ``M`` tokens.

This implementation deliberately *does not* attempt an all-to-all
exchange of tokens: it trades extra compute (every rank computes
routing for the full gathered batch and the kernel processes off-rank
slots before zeroing them) for using only all-gather + all-reduce. It is
exactly what the kernel's EP-sentinel comment suggests as the
out-of-the-box pattern. A more sophisticated all2all dispatch could be
implemented on top of the same kernel; we leave that for vLLM.

Compared apples-to-apples to ``vLLM world=8 oracle uniform``, this
exercises the *transformer-native* EP path end to end.

Usage:

    python benchmark_hf_native_ep.py \\
        --model-config /home/yyx/models/Qwen3-30B-A3B/config.json \\
        --world-size 8 \\
        --tokens 4096 \\
        --num-layers 48 \\
        --warmup 2 --iters 5 \\
        --output results/qwen3_heavy/hf_native_ep.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F


def _load_text_config(path: Path):
    from transformers import AutoConfig

    parent = path.parent if path.is_file() else path
    cfg = AutoConfig.from_pretrained(str(parent), trust_remote_code=True)
    return getattr(cfg, "text_config", cfg)


def _build_local_experts(text_config, local_num_experts, device, dtype):
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    local_cfg = type(text_config).from_dict(text_config.to_dict())
    local_cfg.num_experts = local_num_experts
    local_cfg._experts_implementation = "grouped_mm"
    experts = Qwen3MoeExperts(local_cfg).to(device=device, dtype=dtype)
    return experts


def _worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.port))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    if args.cuda_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    text_config = _load_text_config(Path(args.model_config))
    H = int(text_config.hidden_size)
    E = int(text_config.num_experts)
    K = int(text_config.num_experts_per_tok)
    norm_topk_prob = bool(getattr(text_config, "norm_topk_prob", True))

    if E % world_size != 0:
        raise ValueError(
            f"num_experts ({E}) must be divisible by world_size ({world_size})"
        )
    L = E // world_size  # local experts per rank
    SENTINEL = L  # any value >= num_experts marks the slot off-rank

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    torch.manual_seed(args.seed + rank)

    # Replicated gate (every rank holds the global router)
    gate = nn.Linear(H, E, bias=False).to(device=device, dtype=dtype)
    with torch.no_grad():
        gate.weight.normal_(0.0, 0.02)
    # Make the gate identical across ranks via broadcast.
    dist.broadcast(gate.weight.data, src=0)

    # Local expert slice
    experts = _build_local_experts(text_config, L, device, dtype)
    with torch.no_grad():
        for p in experts.parameters():
            p.normal_(0.0, 0.02)
    experts.eval()

    M = int(args.tokens)
    x_local = (
        torch.randn((M, H), device=device, dtype=dtype, generator=None) / 10.0
    )

    if rank == 0:
        print(
            f"[hf-ep] world={world_size} rank={rank} "
            f"global_E={E} local_E={L} K={K} H={H} "
            f"tokens_per_rank={M} num_layers={args.num_layers} "
            f"dtype={dtype} sentinel={SENTINEL}"
        )

    @torch.no_grad()
    def step(x_in_local: torch.Tensor) -> torch.Tensor:
        # 1. all-gather hidden states across ranks
        gathered = [torch.empty_like(x_in_local) for _ in range(world_size)]
        dist.all_gather(gathered, x_in_local)
        x_global = torch.cat(gathered, dim=0)  # (M*world, H)

        # 2. global routing (replicated on every rank — gate weights are
        #    identical so every rank computes the same answer; a tiny
        #    amount of bf16 numerical drift across GPUs is possible so
        #    in production one would `dist.broadcast` the indices, but
        #    for benchmark timing it doesn't matter).
        router_logits = gate(x_global)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        weights, ids_global = torch.topk(router_probs, K, dim=-1)
        if norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(dtype)

        # 3. translate global ids → local ids with sentinel for off-rank
        lo = rank * L
        hi = (rank + 1) * L
        on_rank = (ids_global >= lo) & (ids_global < hi)
        ids_local = torch.where(
            on_rank,
            ids_global - lo,
            torch.full_like(ids_global, SENTINEL),
        )

        # 4. per-rank experts grouped_mm (zeros off-rank slots internally)
        out_partial = experts(x_global, ids_local, weights)  # (M*world, H)

        # 5. combine partials across ranks
        dist.all_reduce(out_partial, op=dist.ReduceOp.SUM)

        # 6. each rank slices its own tokens
        return out_partial[rank * M : (rank + 1) * M]

    # Warmup
    for _ in range(args.warmup):
        x = x_local
        for _ in range(args.num_layers):
            x = step(x)
    torch.cuda.synchronize(device)
    dist.barrier()

    per_layer_lat: list[list[float]] = [[] for _ in range(args.num_layers)]
    total_lat: list[float] = []
    for it in range(args.iters):
        st_t = torch.cuda.Event(enable_timing=True)
        ed_t = torch.cuda.Event(enable_timing=True)
        st_t.record()
        x = x_local
        for ll in range(args.num_layers):
            st = torch.cuda.Event(enable_timing=True)
            ed = torch.cuda.Event(enable_timing=True)
            st.record()
            x = step(x)
            ed.record()
            ed.synchronize()
            per_layer_lat[ll].append(st.elapsed_time(ed))
        ed_t.record()
        ed_t.synchronize()
        total_lat.append(st_t.elapsed_time(ed_t))

    local_mean = float(statistics.mean(total_lat))
    local_std = float(statistics.stdev(total_lat)) if len(total_lat) > 1 else 0.0
    local_per_layer_mean = [statistics.mean(t) for t in per_layer_lat]

    # All-reduce to find the slowest rank
    local_mean_t = torch.tensor([local_mean], device=device, dtype=torch.float64)
    max_mean_t = local_mean_t.clone()
    dist.all_reduce(max_mean_t, op=dist.ReduceOp.MAX)

    # All-gather per-rank means for visibility
    gathered_means = [torch.zeros_like(local_mean_t) for _ in range(world_size)]
    dist.all_gather(gathered_means, local_mean_t)
    per_rank_mean_ms = [float(t.item()) for t in gathered_means]

    if rank == 0:
        flops_per_token = K * (
            2 * (2 * H * text_config.moe_intermediate_size)
            + 2 * (text_config.moe_intermediate_size * H)
        )
        total_flops = flops_per_token * (M * world_size) * args.num_layers
        achieved_tflops = total_flops / max(float(max_mean_t.item()), 1e-9) / 1e9
        result = {
            "impl": "hf_v5_native_ep_allgather_allreduce",
            "world_size": world_size,
            "tokens_per_rank": M,
            "total_tokens": M * world_size,
            "num_experts": E,
            "local_num_experts": L,
            "topk": K,
            "hidden_size": H,
            "moe_intermediate_size": int(text_config.moe_intermediate_size),
            "num_layers": args.num_layers,
            "dtype": str(dtype),
            "iters": args.iters,
            "warmup": args.warmup,
            "max_rank_mean_step_ms": float(max_mean_t.item()),
            "max_rank_std_step_ms": local_std,
            "per_rank_mean_step_ms": per_rank_mean_ms,
            "per_layer_mean_step_ms": local_per_layer_mean,
            "achieved_tflops_per_s": achieved_tflops,
        }
        print(json.dumps(result, ensure_ascii=False))
        if args.output:
            outpath = Path(args.output)
            outpath.parent.mkdir(parents=True, exist_ok=True)
            with outpath.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", type=str, required=True)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--tokens", type=int, default=4096,
                   help="tokens PER RANK (matches vLLM bench's per-rank count)")
    p.add_argument("--num-layers", type=int, default=48)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--port", type=int, default=29501)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--output", type=str, default=None)
    p.add_argument(
        "--cuda-alloc-conf",
        type=str,
        default="expandable_segments:True",
        help="Set PYTORCH_CUDA_ALLOC_CONF for the spawned ranks.",
    )
    args = p.parse_args()
    mp.spawn(_worker, args=(args.world_size, args),
             nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
