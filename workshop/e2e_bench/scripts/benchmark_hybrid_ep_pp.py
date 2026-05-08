#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hybrid EP × PP MoE benchmark.

Each rank holds ``E / ep_size`` experts of ``L / pp_size`` layers.
Within a stage, dispatch+combine via ``dist.all_to_all_single`` over the
EP subgroup (mirrors ``benchmark_flashinfer_native_ep.py``). Between
stages, ``dist.send`` / ``dist.recv`` of activations over the PP subgroup
(mirrors ``benchmark_pp_moe.py``).

Companion to ``benchmark_flashinfer_native_ep.py`` (§4i.1) and
``benchmark_pp_moe.py`` (§9). Used to validate the §9.4 prediction
that EP×PP hybrids are *worse* than pure PP for Qwen3-30B-A3B on
8×4090.

Layout — for ``ep_size = E_, pp_size = P_, world = E_·P_``:

    rank   = pp_stage * ep_size + ep_rank
    pp_stage = rank // ep_size,   ep_rank = rank % ep_size
    PP group for an ep_rank: [ep_rank, ep_size + ep_rank, 2*ep_size + ep_rank, ...]
    EP group for a pp_stage: [pp_stage*ep_size, ..., pp_stage*ep_size + ep_size - 1]

Usage:

    python benchmark_hybrid_ep_pp.py \\
        --ep-size 2 --pp-size 4 \\
        --tokens 32768 --microbatches 8 \\
        --iters 3 --warmup 2 \\
        --autotune \\
        --cuda-home /usr/local/cuda-13.0 \\
        --output results/qwen3_heavy/flashinfer_hybrid_ep_pp.jsonl
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


def _make_oracle_uniform_local(
    M: int,
    K: int,
    E_global: int,
    ep_rank: int,
    ep_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same recipe as ``benchmark_flashinfer_native_ep.py``: every
    (token, slot) pair gets a global expert id in a globally
    round-robin pattern, then dispatched by ``ids // (E_global / ep_size)``.

    The "rank" used here is the EP-rank within the EP subgroup (not the
    global rank), because the all-to-all happens within that subgroup.
    """
    base = (ep_rank * M * K + torch.arange(M * K, device=device)) % E_global
    ids = base.view(M, K).to(torch.int)
    weights = torch.full((M, K), 1.0 / K, device=device, dtype=torch.float32)
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

    from flashinfer import fused_moe

    pp_size = args.pp_size
    ep_size = args.ep_size
    if pp_size * ep_size != world_size:
        raise ValueError(
            f"pp_size ({pp_size}) × ep_size ({ep_size}) must equal "
            f"world_size ({world_size})"
        )
    if ep_size < 1 or pp_size < 1:
        raise ValueError("ep_size and pp_size must be >= 1")

    pp_stage = rank // ep_size
    ep_rank = rank % ep_size

    # ---- Build PP / EP subgroups. Every rank must call new_group with the
    # same arguments in the same order, even for groups it doesn't belong to.
    pp_pgs: list = []
    for er in range(ep_size):
        pg = dist.new_group(ranks=[s * ep_size + er for s in range(pp_size)])
        pp_pgs.append(pg)
    ep_pgs: list = []
    for s in range(pp_size):
        pg = dist.new_group(ranks=[s * ep_size + er for er in range(ep_size)])
        ep_pgs.append(pg)
    pp_pg = pp_pgs[ep_rank]
    ep_pg = ep_pgs[pp_stage]

    H = args.hidden_size
    I = args.moe_intermediate_size
    E = args.num_experts
    K = args.topk
    L = args.num_layers
    if L % pp_size != 0:
        raise ValueError(f"num_layers={L} must divide pp_size={pp_size}")
    if E % ep_size != 0:
        raise ValueError(f"num_experts={E} must divide ep_size={ep_size}")
    L_per_stage = L // pp_size
    local_E = E // ep_size

    M_total = args.tokens
    B = args.microbatches
    if M_total % B != 0:
        raise ValueError(f"tokens={M_total} must divide microbatches={B}")
    M_micro = M_total // B
    if (M_micro * K) % ep_size != 0:
        raise ValueError(
            f"M_micro·K ({M_micro*K}) must divide ep_size ({ep_size}) "
            "for oracle_uniform equal-split dispatch"
        )

    torch.manual_seed(args.seed + rank)

    # ---- Allocate weights: L_per_stage layers, each with local_E experts.
    w31_layers: list[torch.Tensor] = []
    w2_layers: list[torch.Tensor] = []
    for _ in range(L_per_stage):
        w31 = (torch.randn(local_E, 2 * I, H, dtype=dtype, device=device) / 16).contiguous()
        w2 = (torch.randn(local_E, H, I, dtype=dtype, device=device) / 16).contiguous()
        w31_layers.append(w31)
        w2_layers.append(w2)

    # ---- Pre-compute routing + dispatch permutation per microbatch. Routing
    # depends on `ep_rank` (so each EP-rank produces tokens hitting all
    # ep_size destinations equally — independent across stages).
    perms: list[dict[str, torch.Tensor]] = []
    for mb in range(B):
        if args.routing == "oracle_uniform":
            ids_global, weights = _make_oracle_uniform_local(
                M_micro, K, E, ep_rank, ep_size, device
            )
        else:
            raise NotImplementedError(f"routing={args.routing} not implemented")
        dest = (ids_global // local_E).view(-1)
        sort_idx = torch.argsort(dest, stable=True)
        inv_sort_idx = torch.argsort(sort_idx)
        token_idx = torch.arange(M_micro, device=device).repeat_interleave(K)
        permuted_token_idx = token_idx[sort_idx]
        permuted_ids = ids_global.view(-1)[sort_idx].contiguous()
        permuted_w = weights.view(-1)[sort_idx].contiguous()
        perms.append({
            "permuted_token_idx": permuted_token_idx.contiguous(),
            "permuted_ids": permuted_ids,
            "permuted_w": permuted_w,
            "inv_sort_idx": inv_sort_idx.contiguous(),
        })

    # ---- Persistent buffers (one set, reused across layers + microbatches).
    send_x_buf = torch.empty(M_micro * K, H, dtype=dtype, device=device)
    recv_x_buf = torch.empty(M_micro * K, H, dtype=dtype, device=device)
    moe_out_buf = torch.empty(M_micro * K, H, dtype=dtype, device=device)
    combined_buf = torch.empty(M_micro * K, H, dtype=dtype, device=device)
    recv_ids_buf = torch.empty(M_micro * K, dtype=torch.int32, device=device)
    recv_w_buf = torch.empty(M_micro * K, dtype=torch.float32, device=device)
    pp_recv_buf = torch.empty(M_micro, H, dtype=dtype, device=device)

    # ---- PP neighbours
    pp_prev = (pp_stage - 1) * ep_size + ep_rank if pp_stage > 0 else None
    pp_next = (pp_stage + 1) * ep_size + ep_rank if pp_stage < pp_size - 1 else None

    if pp_stage == 0:
        inputs = [
            (torch.randn(M_micro, H, dtype=dtype, device=device) / 16).contiguous()
            for _ in range(B)
        ]
    else:
        inputs = None

    @torch.no_grad()
    def ep_layer_forward(
        x_in: torch.Tensor,
        perm: dict[str, torch.Tensor],
        w31: torch.Tensor,
        w2: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Replicate K and permute by destination EP-rank.
        send_x_buf.copy_(x_in.index_select(0, perm["permuted_token_idx"]))

        # 2. all_to_all_single dispatch within the EP subgroup.
        if ep_size > 1:
            dist.all_to_all_single(recv_x_buf, send_x_buf, group=ep_pg)
            dist.all_to_all_single(recv_ids_buf, perm["permuted_ids"], group=ep_pg)
            dist.all_to_all_single(recv_w_buf, perm["permuted_w"], group=ep_pg)
        else:
            recv_x_buf.copy_(send_x_buf)
            recv_ids_buf.copy_(perm["permuted_ids"])
            recv_w_buf.copy_(perm["permuted_w"])

        # 3. Per-rank fused MoE — K_kernel = 1 because tokens are pre-replicated.
        ids_2d = recv_ids_buf.view(M_micro * K, 1)
        w_2d = recv_w_buf.view(M_micro * K, 1)
        res = fused_moe.cutlass_fused_moe(
            recv_x_buf,
            ids_2d,
            w_2d,
            w31,
            w2,
            dtype,
            output=moe_out_buf,
            quant_scales=None,
            ep_size=ep_size,
            ep_rank=ep_rank,
            tune_max_num_tokens=max(8192, M_micro * K),
        )
        moe_out = res[0] if isinstance(res, (list, tuple)) else res

        # 4. all_to_all_single combine within the EP subgroup.
        if ep_size > 1:
            dist.all_to_all_single(combined_buf, moe_out, group=ep_pg)
        else:
            combined_buf.copy_(moe_out)

        # 5. Inverse-permute and sum the K partial outputs per token.
        unpermuted = combined_buf.index_select(0, perm["inv_sort_idx"])
        return unpermuted.view(M_micro, K, H).sum(dim=1)

    @torch.no_grad()
    def stage_forward(x: torch.Tensor, perm: dict[str, torch.Tensor]) -> torch.Tensor:
        for l in range(L_per_stage):
            x = ep_layer_forward(x, perm, w31_layers[l], w2_layers[l])
        return x

    @torch.no_grad()
    def step() -> None:
        # 1F-only synchronous pipeline.
        if pp_stage == 0:
            for mb in range(B):
                x = inputs[mb]  # type: ignore[index]
                x = stage_forward(x, perms[mb])
                if pp_size > 1 and pp_next is not None:
                    dist.send(x, dst=pp_next, group=pp_pg)
        elif pp_stage == pp_size - 1:
            for mb in range(B):
                dist.recv(pp_recv_buf, src=pp_prev, group=pp_pg)  # type: ignore[arg-type]
                _ = stage_forward(pp_recv_buf, perms[mb])
        else:
            for mb in range(B):
                dist.recv(pp_recv_buf, src=pp_prev, group=pp_pg)  # type: ignore[arg-type]
                x = stage_forward(pp_recv_buf, perms[mb])
                dist.send(x, dst=pp_next, group=pp_pg)  # type: ignore[arg-type]

    if args.autotune:
        try:
            from flashinfer.autotuner import autotune  # type: ignore
            with autotune(True):
                step()
                torch.cuda.synchronize(device)
                dist.barrier()
            if rank == 0:
                print("[hybrid] autotune pass complete")
        except Exception as e:
            if rank == 0:
                print(f"[hybrid] autotune failed: {e!r}")

    # ---- Warmup
    for _ in range(args.warmup):
        step()
        torch.cuda.synchronize(device)
        dist.barrier()

    # ---- Timed iters (CUDA event timing per rank).
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

    means: list = [None] * world_size
    dist.all_gather_object(means, local_mean)

    if rank == 0:
        means_f = [float(m) for m in means]  # type: ignore[arg-type]
        max_mean = max(means_f)
        result = {
            "impl": "flashinfer_hybrid_ep_pp",
            "world_size": world_size,
            "ep_size": ep_size,
            "pp_size": pp_size,
            "tokens": M_total,
            "microbatches": B,
            "tokens_per_microbatch": M_micro,
            "num_layers": L,
            "layers_per_stage": L_per_stage,
            "local_num_experts": local_E,
            "num_experts": E,
            "topk": K,
            "hidden_size": H,
            "moe_intermediate_size": I,
            "dtype": str(dtype),
            "iters": args.iters,
            "warmup": args.warmup,
            "autotune": bool(args.autotune),
            "routing": args.routing,
            "max_rank_mean_step_ms": max_mean,
            "rank_local_std_step_ms": local_std,
            "per_rank_mean_step_ms": means_f,
            "timestamp": time.time(),
        }
        flops = M_total * K * 6 * H * I * L
        result["achieved_tflops_per_s"] = flops / (max_mean / 1e3) / 1e12
        print(json.dumps(result, indent=2))
        if args.output:
            outp = Path(args.output)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with outp.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
            print(f"[hybrid] appended to {outp}")

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ep-size", type=int, required=True)
    p.add_argument("--pp-size", type=int, required=True)
    p.add_argument("--tokens", type=int, default=32768)
    p.add_argument("--microbatches", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=48)
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--moe-intermediate-size", type=int, default=768)
    p.add_argument("--routing", type=str, default="oracle_uniform")
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--port", type=int, default=29588)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--autotune", action="store_true")
    p.add_argument("--cuda-home", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    world_size = args.ep_size * args.pp_size
    mp.spawn(_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
