#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Multi-rank **all-to-all** expert-parallel benchmark using FlashInfer's
``cutlass_fused_moe`` for the per-rank GEMM.

Companion to ``benchmark_hf_native_ep.py`` (which uses the simpler
all-gather + sentinel-mask + all-reduce pattern). This script implements
the **token-replicated all-to-all** pattern that vLLM, pplx-kernels, and
TRT-LLM use:

    1. Each rank starts with ``--tokens`` (= ``M``) tokens of hidden state.
    2. Compute the **global** router locally on every rank (replicated
       gate weights, identical answer modulo bf16 drift). Result:
       ``ids_global[M, K]`` ∈ ``[0, num_experts)`` and
       ``weights[M, K]`` (already ``softmax + topk + sum-normalised``).
    3. **Replicate** each token K times into ``M*K`` "(token-copy,
       single-expert) slots". Sort the slots by destination rank
       (``dest = id // local_E``). For ``oracle_uniform`` routing this is
       a perfectly even fan-out: every rank sends exactly ``M*K/world``
       slots to every other rank.
    4. **all_to_all_single** the (hidden_state, expert_id, weight,
       origin_book-keeping) payloads. After dispatch, every rank holds
       ``M*K`` slots whose expert IDs land entirely in *its* local
       ``[ep_rank*L, (ep_rank+1)*L)`` range.
    5. Call ``flashinfer.fused_moe.cutlass_fused_moe`` with
       ``ep_size=world_size``, ``ep_rank=rank``, and the **local weight
       slice only** (``local_E = num_experts / world_size`` experts).
       The kernel rebases the global IDs to local indices via its
       ``MOEParallelismConfig`` and applies the per-slot scale internally.
       Effective ``top_k = 1`` from the kernel's perspective: each
       dispatched slot consumes one expert.
    6. **all_to_all_single** (inverse splits) the per-slot outputs back
       to the originating ranks.
    7. Scatter and sum: the originating rank places the K returned slots
       back into their original ``(token, slot)`` rows and sums the K
       partial outputs per token.

This is a "true" all-to-all dispatch (not a broadcast/all-gather), so:

* Per-rank compute is **only** for tokens routed to that rank's experts:
  ``M_total * K / world`` slots, not ``M_total * K`` slots.
* Communication payload is two ``[M*K * H * 2 bytes]`` all-to-alls per
  layer. For Qwen3-30B-A3B at M=4096, K=8, H=2048, bf16 →
  ~128 MB per direction per layer per rank.

Currently only ``--routing oracle_uniform`` is implemented because for
oracle uniform every rank sends an equal split to every other rank, so
``dist.all_to_all_single`` with equal splits is sufficient. A
softmax-based variant would need an extra ``all_to_all`` to exchange
send-count metadata for the variable-length dispatch (left for a
follow-up).

Usage::

    CUDA_HOME=/usr/local/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH \\
    python benchmark_flashinfer_native_ep.py \\
        --model-config /home/yyx/models/Qwen3-30B-A3B/config.json \\
        --world-size 8 --tokens 4096 --num-layers 48 \\
        --warmup 2 --iters 5 \\
        --output results/qwen3_heavy/flashinfer_native_ep.jsonl
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
import torch.nn as nn
import torch.nn.functional as F


def _load_text_config(path: Path):
    from transformers import AutoConfig

    parent = path.parent if path.is_file() else path
    cfg = AutoConfig.from_pretrained(str(parent), trust_remote_code=True)
    return getattr(cfg, "text_config", cfg)


def _make_oracle_uniform_local(
    M: int,
    K: int,
    E_global: int,
    rank: int,
    world: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate per-token routing such that every (token, slot) pair is
    deterministically assigned to a global expert in a globally
    round-robin pattern.

    Per (rank, token, slot) the chosen expert is::

        expert_id = (rank * M * K + token * K + slot) % E_global

    With M*K divisible by E_global this guarantees:

    * each global expert receives exactly the same number of slots
      across all ranks → physical_cv = logical_cv = 0 (same as the
      ``oracle_uniform`` pattern in ``benchmark_eplb_multigpu.py``);
    * the slots from one rank are *also* distributed equally across all
      destination ranks (since the IDs sweep through every expert
      modulo E_global), so ``dist.all_to_all_single`` with equal splits
      = ``M*K/world`` slots per direction is correct;
    * each token's K choices are K consecutive expert IDs (modulo
      E_global), which are guaranteed distinct as long as ``K < E``.
    """
    base = (rank * M * K + torch.arange(M * K, device=device)) % E_global
    ids = base.view(M, K).to(torch.int)
    weights = torch.full((M, K), 1.0 / K, device=device, dtype=torch.float32)
    return ids, weights


def _worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.port))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    if args.cuda_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf
    # Make sure the CUDA 13.0 toolkit is on PATH for FlashInfer's JIT.
    if args.cuda_home:
        os.environ["CUDA_HOME"] = args.cuda_home
        os.environ["PATH"] = f"{args.cuda_home}/bin:" + os.environ.get("PATH", "")
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{args.cuda_home}/lib64:{ld}"

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    text_config = _load_text_config(Path(args.model_config))
    H = int(text_config.hidden_size)
    I = int(text_config.moe_intermediate_size)
    E = int(text_config.num_experts)
    K = int(text_config.num_experts_per_tok)

    if E % world_size != 0:
        raise ValueError(
            f"num_experts ({E}) must be divisible by world_size ({world_size})"
        )
    L = E // world_size  # local experts per rank

    M = int(args.tokens)  # tokens per rank
    if (M * K) % world_size != 0:
        raise ValueError(
            f"M*K ({M*K}) must be divisible by world_size ({world_size}) "
            "so that oracle_uniform produces equal per-rank splits"
        )
    slots_per_dest = M * K // world_size  # equal-split count for oracle

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    torch.manual_seed(args.seed + rank)

    import flashinfer
    import flashinfer.fused_moe as fused_moe

    # Local expert weight slice — every rank holds L = E/world_size experts.
    w31 = (torch.randn(L, 2 * I, H, device=device, dtype=dtype) / 16).contiguous()
    w2 = (torch.randn(L, H, I, device=device, dtype=dtype) / 16).contiguous()

    if rank == 0:
        print(
            f"[fi-ep] flashinfer={flashinfer.__version__} world={world_size} "
            f"global_E={E} local_E={L} K={K} H={H} I={I} "
            f"tokens_per_rank={M} num_layers={args.num_layers} dtype={dtype} "
            f"routing={args.routing} slots_per_dest={slots_per_dest}"
        )

    # Pre-compute the (constant across iters) routing pattern.
    if args.routing == "oracle_uniform":
        ids_global, weights = _make_oracle_uniform_local(
            M, K, E, rank, world_size, device
        )
    else:
        raise NotImplementedError(
            f"routing={args.routing} not implemented in this benchmark "
            "(only oracle_uniform; see file docstring)."
        )

    # Pre-compute the dispatch permutation. For oracle_uniform the
    # permutation is independent of input data, so we compute it once.
    # Sort all (token, slot) pairs by destination rank so consecutive
    # blocks go to dest 0, dest 1, ...
    dest = (ids_global // L).view(-1)         # (M*K,)
    sort_idx = torch.argsort(dest, stable=True)  # (M*K,)
    inv_sort_idx = torch.argsort(sort_idx)        # (M*K,) — combine path

    # Sanity: every dest_rank should appear exactly slots_per_dest times.
    counts = torch.bincount(dest, minlength=world_size)
    assert (counts == slots_per_dest).all(), (
        f"oracle_uniform produced uneven dispatch counts on rank {rank}: "
        f"{counts.tolist()}"
    )

    # Pre-permute the static metadata (ids and weights). Hidden states are
    # generated fresh per-iter so we permute them inside the step.
    token_idx = torch.arange(M, device=device).repeat_interleave(K)  # (M*K,)
    permuted_token_idx = token_idx[sort_idx]      # (M*K,) — which token to copy
    permuted_ids = ids_global.view(-1)[sort_idx]  # (M*K,) — already in dest's range
    permuted_w = weights.view(-1)[sort_idx]       # (M*K,)

    # Recv buffers (all dispatched slots from all ranks → equal sizes)
    recv_x_buf = torch.empty(M * K, H, device=device, dtype=dtype)
    recv_ids_buf = torch.empty(M * K, device=device, dtype=torch.int)
    recv_w_buf = torch.empty(M * K, device=device, dtype=torch.float32)
    out_partial_buf = torch.empty(M * K, H, device=device, dtype=dtype)
    combined_buf = torch.empty(M * K, H, device=device, dtype=dtype)
    moe_out_buf = torch.empty(M * K, H, device=device, dtype=dtype)

    x_local = (
        torch.randn((M, H), device=device, dtype=dtype, generator=None) / 10.0
    )

    @torch.no_grad()
    def step(x_in_local: torch.Tensor) -> torch.Tensor:
        # 1. Replicate K times + permute: x[m] copied K times, then
        #    sorted by dest rank. Output shape (M*K, H).
        send_x = x_in_local.index_select(0, permuted_token_idx).contiguous()

        # 2. all_to_all_single dispatch — equal splits.
        dist.all_to_all_single(recv_x_buf, send_x)
        dist.all_to_all_single(recv_ids_buf, permuted_ids)
        dist.all_to_all_single(recv_w_buf, permuted_w)

        # 3. cutlass_fused_moe with K_kernel=1 (each dispatched slot
        #    consumes exactly one expert). The kernel rebases the global
        #    IDs to local indices via MOEParallelismConfig.
        ids_2d = recv_ids_buf.view(M * K, 1)
        w_2d = recv_w_buf.view(M * K, 1)
        res = fused_moe.cutlass_fused_moe(
            recv_x_buf,
            ids_2d,
            w_2d,
            w31,
            w2,
            dtype,
            output=moe_out_buf,
            quant_scales=None,
            ep_size=world_size,
            ep_rank=rank,
            tune_max_num_tokens=max(8192, M * K),
        )
        moe_out = res[0] if isinstance(res, (list, tuple)) else res

        # 4. all_to_all_single combine — same equal splits.
        dist.all_to_all_single(combined_buf, moe_out)

        # 5. Inverse permutation + sum K partial outputs per token.
        # combined[i] = expert_output for slot inv_perm[i] (still in
        # post-sort order). Place each row back into its (token, slot)
        # position, then sum over slot.
        unpermuted = combined_buf.index_select(0, inv_sort_idx)  # (M*K, H)
        return unpermuted.view(M, K, H).sum(dim=1)

    if args.autotune:
        from flashinfer.autotuner import AutoTuner, autotune
        AutoTuner.get().clear_cache()
        with torch.inference_mode(), autotune(True):
            _ = step(x_local)
        if rank == 0:
            print("[fi-ep] autotune done")
        torch.cuda.synchronize(device)
        dist.barrier()

    # Warmup
    for _ in range(args.warmup):
        x = x_local
        for _ in range(args.num_layers):
            x = step(x)
    torch.cuda.synchronize(device)
    dist.barrier()

    per_layer_lat: list[list[float]] = [[] for _ in range(args.num_layers)]
    total_lat: list[float] = []
    for _ in range(args.iters):
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

    local_mean_t = torch.tensor([local_mean], device=device, dtype=torch.float64)
    max_mean_t = local_mean_t.clone()
    dist.all_reduce(max_mean_t, op=dist.ReduceOp.MAX)
    gathered_means = [torch.zeros_like(local_mean_t) for _ in range(world_size)]
    dist.all_gather(gathered_means, local_mean_t)
    per_rank_mean_ms = [float(t.item()) for t in gathered_means]

    if rank == 0:
        flops_per_token = K * (
            2 * (2 * H * I)
            + 2 * (I * H)
        )
        total_flops = flops_per_token * (M * world_size) * args.num_layers
        achieved_tflops = total_flops / max(float(max_mean_t.item()), 1e-9) / 1e9
        result = {
            "impl": "flashinfer_native_ep_alltoall",
            "flashinfer_version": flashinfer.__version__,
            "routing": args.routing,
            "world_size": world_size,
            "tokens_per_rank": M,
            "total_tokens": M * world_size,
            "num_experts": E,
            "local_num_experts": L,
            "topk": K,
            "hidden_size": H,
            "moe_intermediate_size": I,
            "num_layers": args.num_layers,
            "dtype": str(dtype),
            "iters": args.iters,
            "warmup": args.warmup,
            "autotune": bool(args.autotune),
            "max_rank_mean_step_ms": float(max_mean_t.item()),
            "max_rank_std_step_ms": local_std,
            "per_rank_mean_step_ms": per_rank_mean_ms,
            "per_layer_mean_step_ms": local_per_layer_mean,
            "achieved_tflops_per_s": achieved_tflops,
            "timestamp": time.time(),
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
    p.add_argument("--port", type=int, default=29504)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--routing", choices=["oracle_uniform"],
                   default="oracle_uniform")
    p.add_argument("--autotune", action="store_true",
                   help="Run flashinfer.autotuner once before timing.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--cuda-alloc-conf", type=str,
                   default="expandable_segments:True")
    p.add_argument("--cuda-home", type=str,
                   default="/usr/local/cuda-13.0")
    args = p.parse_args()
    mp.spawn(_worker, args=(args.world_size, args),
             nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
