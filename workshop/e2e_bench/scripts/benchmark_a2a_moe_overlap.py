#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A2A MoE overlap microbenchmark.

This script isolates the communication pattern behind expert-parallel MoE:

  1. Generate per-layer synthetic top-k logical routing with target GPU and
     intra-GPU expert CVs.
  2. Place experts either disjointly or with replicated/overlapped copies.
  3. Choose a physical expert replica with a local-first rule.
  4. Dispatch token copies with ``torch.distributed.all_to_all_single``.
  5. Run the PR Triton fused-MoE compute kernel on the received local experts.
  6. Reverse ``all_to_all_single`` the per-slot outputs and locally combine.

It is intentionally not an engine benchmark. The goal is to test the A2A
transport and the communication-volume effect of expert overlap while keeping
the setup close to the other scripts in this directory.

Single run example:

    python benchmark_a2a_moe_overlap.py \\
        --world-size 8 --total-tokens 8192 \\
        --model-config /home/lzy/models/Qwen3-30B-A3B/config.json \\
        --overlap 0.25 --strategy hybrid \\
        --iters 10 --warmup 3 \\
        --output results/a2a_moe_overlap.jsonl

Full sweep example:

    python benchmark_a2a_moe_overlap.py --sweep \\
        --output-dir results/a2a_moe_overlap_sweep

The sweep runner uses 8 GPUs efficiently by launching four 2-GPU jobs, two
4-GPU jobs, or one 8-GPU job per wave, with distinct ports and output files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


STRATEGIES = ("greedy_balance", "min_communication", "cv_aware", "hybrid")
ALL_STRATEGIES = ("disjoint",) + STRATEGIES


@dataclass(frozen=True)
class Placement:
    requested_overlap: float
    effective_local_expert_fraction: float
    replicas_per_expert: int
    overlap_clamped: bool
    hosts_by_expert: list[list[int]]
    experts_by_rank: list[list[int]]
    hosted_mask: torch.Tensor  # [world, num_experts], bool on CPU


@dataclass
class PreparedRouting:
    logical_ids: torch.Tensor       # [local_tokens, topk], long CPU
    chosen_ranks: torch.Tensor      # [local_tokens, topk], long CPU
    send_counts: list[int]          # [world]
    sorted_token_idx: torch.Tensor  # [local_tokens * topk], long CPU
    sorted_expert_ids: torch.Tensor # [local_tokens * topk], int32 CPU
    local_first_opportunities: int
    local_first_violations: int
    local_first_chosen: int
    unique_remote_destinations: int
    physical_counts: torch.Tensor   # [world, num_experts], int64 CPU
    logical_counts: torch.Tensor    # [num_experts], int64 CPU


@dataclass(frozen=True)
class ModelParams:
    model_config: str
    num_layers: int
    hidden_size: int
    moe_intermediate_size: int
    num_experts: int
    topk: int


@dataclass
class LayerPlan:
    layer_idx: int
    send_counts: list[int]
    recv_counts: list[int]
    total_recv: int
    sorted_token_idx: torch.Tensor
    send_expert_buf: torch.Tensor
    rows_by_local: list[torch.Tensor]
    recv_local_ids: torch.Tensor


def _parse_size(s: str) -> int:
    text = s.strip().lower().replace("_", "")
    mult = 1
    for suffix, value in (
        ("kb", 1024),
        ("mb", 1024**2),
        ("gb", 1024**3),
        ("k", 1024),
        ("m", 1024**2),
        ("g", 1024**3),
    ):
        if text.endswith(suffix):
            mult = value
            text = text[: -len(suffix)]
            break
    return int(float(text) * mult)


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_strings(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"Unsupported dtype byte size: {dtype}")


def _resolve_model_params(args: argparse.Namespace) -> ModelParams:
    cfg_path = Path(args.model_config).expanduser()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"model config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    def _field(name: str, override: int | None) -> int:
        if override is not None:
            return int(override)
        if name not in cfg:
            raise KeyError(f"missing {name!r} in model config: {cfg_path}")
        return int(cfg[name])

    return ModelParams(
        model_config=str(cfg_path),
        num_layers=_field("num_hidden_layers", args.num_layers),
        hidden_size=_field("hidden_size", args.hidden_size),
        moe_intermediate_size=_field("moe_intermediate_size", args.moe_intermediate_size),
        num_experts=_field("num_experts", args.num_experts),
        topk=_field("num_experts_per_tok", args.topk),
    )


def _validate_benchmark_args(
    args: argparse.Namespace,
    model: ModelParams,
    world_size: int,
) -> None:
    if world_size <= 0:
        raise ValueError("--world-size must be positive")
    if args.total_tokens <= 0:
        raise ValueError("--total-tokens must be positive")
    if args.total_tokens % world_size != 0:
        raise ValueError(
            f"total_tokens ({args.total_tokens}) must be divisible by "
            f"world_size ({world_size})"
        )
    if not 0.0 <= args.overlap <= 1.0:
        raise ValueError("--overlap must be in [0, 1]")
    if args.target_gpu_cv < 0.0:
        raise ValueError("--target-gpu-cv must be non-negative")
    if args.target_expert_cv < 0.0:
        raise ValueError("--target-expert-cv must be non-negative")
    if args.cv_tolerance < 0.0:
        raise ValueError("--cv-tolerance must be non-negative")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.moe_block_size_m <= 0:
        raise ValueError("--moe-block-size-m must be positive")

    for name, value in (
        ("num_layers", model.num_layers),
        ("hidden_size", model.hidden_size),
        ("moe_intermediate_size", model.moe_intermediate_size),
        ("num_experts", model.num_experts),
        ("topk", model.topk),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if model.topk > model.num_experts:
        raise ValueError(
            f"topk ({model.topk}) must be <= num_experts ({model.num_experts})"
        )
    if model.num_experts % world_size != 0:
        raise ValueError(
            f"num_experts ({model.num_experts}) must be divisible by "
            f"world_size ({world_size})"
        )


def _safe_mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _safe_std(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return float(statistics.stdev(xs))


def _cv_from_values(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0.0:
        return 0.0
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return float(math.sqrt(var) / mean)


def _cv_from_counts(counts: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    flat = counts.detach().to(device="cpu", dtype=torch.float64).view(-1)
    if mask is not None:
        mask_flat = mask.detach().to(device="cpu", dtype=torch.bool).view(-1)
        flat = flat[mask_flat]
    return _cv_from_values([float(x) for x in flat.tolist()])


def _counts_from_probs(total: int, probs: Sequence[float]) -> list[int]:
    raw = [float(total) * p for p in probs]
    counts = [int(math.floor(x)) for x in raw]
    rem = total - sum(counts)
    order = sorted(
        range(len(probs)),
        key=lambda i: (raw[i] - counts[i], -i),
        reverse=True,
    )
    for i in range(rem):
        counts[order[i]] += 1
    return counts


def _make_rank_probs(world_size: int, target_cv: float) -> list[float]:
    if world_size <= 1:
        return [1.0]
    z = torch.linspace(-1.0, 1.0, world_size, dtype=torch.float64)
    z = z - z.mean()
    z = z / z.std(unbiased=False).clamp_min(1e-12)
    alpha = float(target_cv)
    weights = 1.0 + alpha * z
    while float(weights.min().item()) <= 0.02:
        alpha *= 0.8
        weights = 1.0 + alpha * z
    probs = weights / weights.sum()
    return [float(x) for x in probs.tolist()]


def _make_probs_with_cv(num_bins: int, target_cv: float, seed: int) -> torch.Tensor:
    """Build a positive probability vector with approximately target CV.

    The z-vector is deterministic but seed-shuffled. For the target values used
    here (0.2 GPU CV, 0.3 intra-rank expert CV) this keeps all bins positive and
    makes realized per-layer statistics stable while still changing identities
    across layers/seeds.
    """
    if num_bins <= 1 or target_cv <= 0.0:
        return torch.full((num_bins,), 1.0 / max(num_bins, 1), dtype=torch.float64)
    z = torch.linspace(-1.0, 1.0, num_bins, dtype=torch.float64)
    z = z - z.mean()
    z = z / z.std(unbiased=False).clamp_min(1e-12)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    z = z[torch.randperm(num_bins, generator=gen)]
    alpha = float(target_cv)
    weights = 1.0 + alpha * z
    while float(weights.min().item()) <= 0.02:
        alpha *= 0.8
        weights = 1.0 + alpha * z
    return weights / weights.sum()


def _mean_rank_expert_cv(
    counts_by_rank_expert: torch.Tensor,
    hosted_mask: torch.Tensor | None = None,
) -> float:
    return _safe_mean(_rank_expert_cvs(counts_by_rank_expert, hosted_mask))


def _rank_expert_cvs(
    counts_by_rank_expert: torch.Tensor,
    hosted_mask: torch.Tensor | None = None,
) -> list[float]:
    counts = counts_by_rank_expert.detach().to(device="cpu", dtype=torch.float64)
    if counts.ndim != 2:
        raise ValueError("counts_by_rank_expert must be [world, experts]")
    if hosted_mask is not None:
        mask = hosted_mask.detach().to(device="cpu", dtype=torch.bool)
        if tuple(mask.shape) != tuple(counts.shape):
            raise ValueError("hosted_mask shape must match counts")
    cvs: list[float] = []
    for rank in range(counts.shape[0]):
        vals = counts[rank]
        if hosted_mask is not None:
            vals = vals[mask[rank]]
        total = float(vals.sum().item())
        if vals.numel() == 0 or total <= 0.0:
            continue
        probs = (vals / total).tolist()
        cvs.append(_cv_from_values([float(x) for x in probs]))
    return cvs


def _build_placement(num_experts: int, world_size: int, overlap: float) -> Placement:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if num_experts % world_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by world_size ({world_size})"
        )
    if overlap < 0:
        raise ValueError("--overlap must be >= 0")

    if overlap == 0.0:
        replicas = 1
    else:
        replicas = int(round(overlap * world_size))
        replicas = max(1, min(world_size, replicas))

    overlap_clamped = overlap > 0.0 and replicas == 1
    effective_fraction = float(replicas) / float(world_size)
    base = num_experts // world_size

    hosts_by_expert: list[list[int]] = []
    experts_by_rank: list[list[int]] = [[] for _ in range(world_size)]
    hosted_mask = torch.zeros((world_size, num_experts), dtype=torch.bool)

    for expert_id in range(num_experts):
        primary = expert_id // base
        hosts = [int((primary + offset) % world_size) for offset in range(replicas)]
        hosts_by_expert.append(hosts)
        for host in hosts:
            experts_by_rank[host].append(expert_id)
            hosted_mask[host, expert_id] = True

    expected_local = num_experts * replicas // world_size
    for rank, experts in enumerate(experts_by_rank):
        if len(experts) != expected_local:
            raise RuntimeError(
                f"Uneven placement: rank {rank} has {len(experts)} experts, "
                f"expected {expected_local}"
            )

    return Placement(
        requested_overlap=float(overlap),
        effective_local_expert_fraction=effective_fraction,
        replicas_per_expert=replicas,
        overlap_clamped=overlap_clamped,
        hosts_by_expert=hosts_by_expert,
        experts_by_rank=experts_by_rank,
        hosted_mask=hosted_mask,
    )


def _make_logical_routing(
    *,
    local_tokens: int,
    topk: int,
    num_experts: int,
    world_size: int,
    rank: int,
    target_gpu_cv: float,
    target_expert_cv: float,
    seed: int,
) -> torch.Tensor:
    total_tokens = local_tokens * world_size
    total_slots = total_tokens * topk
    local_slots = local_tokens * topk
    base = num_experts // world_size
    rank_probs = [float(x) for x in _make_probs_with_cv(
        world_size, target_gpu_cv, seed + 3001
    ).tolist()]
    owner_counts = _counts_from_probs(total_slots, rank_probs)

    owners: list[int] = []
    for owner, count in enumerate(owner_counts):
        owners.extend([owner] * count)
    owner_t = torch.tensor(owners, dtype=torch.long)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 104729)
    owner_t = owner_t[torch.randperm(total_slots, generator=gen)]

    expert_probs_by_owner = [
        _make_probs_with_cv(base, target_expert_cv, seed + 1009 * (owner + 1))
        for owner in range(world_size)
    ]

    ids = torch.empty(total_slots, dtype=torch.long)
    for owner in range(world_size):
        positions = torch.nonzero(owner_t == owner, as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        samples = torch.multinomial(
            expert_probs_by_owner[owner],
            int(positions.numel()),
            replacement=True,
            generator=gen,
        ).to(dtype=torch.long)
        ids[positions] = owner * base + samples

    # Keep the usual top-k routing invariant over the global synthetic batch: a
    # token should not select the same logical expert twice. This only adjusts
    # collisions after the batched sampling above, so the owner/GPU distribution
    # remains unchanged while per-rank slices still differ naturally.
    for token_idx in range(total_tokens):
        used: set[int] = set()
        start = token_idx * topk
        for slot_idx in range(start, start + topk):
            expert_id = int(ids[slot_idx].item())
            if expert_id not in used:
                used.add(expert_id)
                continue
            owner = expert_id // base
            local_offset = expert_id % base
            for delta in range(1, base + 1):
                candidate = owner * base + ((local_offset + delta) % base)
                if candidate not in used:
                    ids[slot_idx] = candidate
                    used.add(candidate)
                    break
            else:
                used.add(expert_id)

    global_ids = ids.view(total_tokens, topk)
    start_token = rank * local_tokens
    return global_ids[start_token : start_token + local_tokens].contiguous()


def _score_cv_after(values: torch.Tensor, index: int) -> float:
    tmp = values.clone()
    tmp[index] += 1
    return _cv_from_counts(tmp)


def _choose_remote_host(
    *,
    hosts: Sequence[int],
    expert_id: int,
    strategy: str,
    gpu_load: torch.Tensor,
    physical_counts: torch.Tensor,
    dest_hist: torch.Tensor,
) -> int:
    if not hosts:
        raise ValueError("Expert has no candidate hosts")
    if len(hosts) == 1 or strategy == "disjoint":
        return int(hosts[0])

    if strategy == "greedy_balance":
        return min(
            (int(h) for h in hosts),
            key=lambda h: (int(gpu_load[h].item()), int(physical_counts[h, expert_id].item()), h),
        )

    if strategy == "min_communication":
        # All candidates are remote here. Concentrating traffic onto an already-used
        # destination reduces the number of active peer pairs for this source rank.
        return min(
            (int(h) for h in hosts),
            key=lambda h: (-int(dest_hist[h].item()), int(gpu_load[h].item()), h),
        )

    if strategy == "cv_aware":
        best_host = int(hosts[0])
        best_score = float("inf")
        for h_raw in hosts:
            h = int(h_raw)
            gpu_score = _score_cv_after(gpu_load, h)
            phys_flat = physical_counts.view(-1)
            phys_index = h * physical_counts.shape[1] + expert_id
            expert_score = _score_cv_after(phys_flat, phys_index)
            score = gpu_score + expert_score
            if score < best_score or (score == best_score and h < best_host):
                best_host = h
                best_score = score
        return best_host

    if strategy == "hybrid":
        total = max(float(gpu_load.sum().item()), 1.0)
        avg_gpu = total / max(float(gpu_load.numel()), 1.0)
        avg_expert = total / max(float((physical_counts > 0).sum().item()), 1.0)
        best_host = int(hosts[0])
        best_score = float("inf")
        for h_raw in hosts:
            h = int(h_raw)
            gpu_term = float(gpu_load[h].item() + 1) / max(avg_gpu, 1.0)
            expert_term = float(physical_counts[h, expert_id].item() + 1) / max(avg_expert, 1.0)
            concentration_bonus = float(dest_hist[h].item()) / total
            score = 0.50 * gpu_term + 0.45 * expert_term - 0.05 * concentration_bonus
            if score < best_score or (score == best_score and h < best_host):
                best_host = h
                best_score = score
        return best_host

    raise ValueError(f"Unknown strategy: {strategy}")


def _prepare_routing(
    *,
    logical_ids: torch.Tensor,
    placement: Placement,
    rank: int,
    world_size: int,
    num_experts: int,
    strategy: str,
) -> PreparedRouting:
    if strategy not in ALL_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {ALL_STRATEGIES}")

    local_tokens, topk = logical_ids.shape
    flat_ids = logical_ids.view(-1)
    total_slots = int(flat_ids.numel())
    gpu_load = torch.zeros(world_size, dtype=torch.int64)
    physical_counts = torch.zeros((world_size, num_experts), dtype=torch.int64)
    logical_counts = torch.bincount(flat_ids, minlength=num_experts).to(torch.int64)
    dest_hist = torch.zeros(world_size, dtype=torch.int64)
    chosen = torch.empty(total_slots, dtype=torch.long)

    local_first_opportunities = 0
    local_first_chosen = 0
    local_first_violations = 0

    for idx, expert_t in enumerate(flat_ids.tolist()):
        expert_id = int(expert_t)
        hosts = placement.hosts_by_expert[expert_id]
        if rank in hosts:
            target_rank = rank
            local_first_opportunities += 1
            local_first_chosen += 1
        else:
            target_rank = _choose_remote_host(
                hosts=hosts,
                expert_id=expert_id,
                strategy=strategy,
                gpu_load=gpu_load,
                physical_counts=physical_counts,
                dest_hist=dest_hist,
            )
            if target_rank == rank:
                local_first_violations += 1

        chosen[idx] = target_rank
        gpu_load[target_rank] += 1
        physical_counts[target_rank, expert_id] += 1
        dest_hist[target_rank] += 1

    send_counts_t = torch.bincount(chosen, minlength=world_size).to(torch.int64)
    send_counts = [int(x) for x in send_counts_t.tolist()]
    sort_idx = torch.argsort(chosen, stable=True)
    token_idx = torch.arange(local_tokens, dtype=torch.long).repeat_interleave(topk)
    sorted_token_idx = token_idx[sort_idx].contiguous()
    sorted_expert_ids = flat_ids[sort_idx].to(torch.int32).contiguous()
    unique_remote_destinations = int(
        sum(1 for dest, count in enumerate(send_counts) if dest != rank and count > 0)
    )

    return PreparedRouting(
        logical_ids=logical_ids,
        chosen_ranks=chosen.view(local_tokens, topk).contiguous(),
        send_counts=send_counts,
        sorted_token_idx=sorted_token_idx,
        sorted_expert_ids=sorted_expert_ids,
        local_first_opportunities=local_first_opportunities,
        local_first_violations=local_first_violations,
        local_first_chosen=local_first_chosen,
        unique_remote_destinations=unique_remote_destinations,
        physical_counts=physical_counts,
        logical_counts=logical_counts,
    )


def _exchange_counts(send_counts: list[int], device: torch.device) -> list[int]:
    send_t = torch.tensor(send_counts, dtype=torch.int32, device=device)
    recv_t = torch.empty_like(send_t)
    if dist.get_world_size() > 1:
        dist.all_to_all_single(recv_t, send_t)
    else:
        recv_t.copy_(send_t)
    return [int(x) for x in recv_t.cpu().tolist()]


def _all_gather_1d(t: torch.Tensor, world_size: int) -> torch.Tensor:
    gathered = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t)
    return torch.stack(gathered, dim=0)


def _init_process_group(rank: int, world_size: int, port: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _build_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parents[1] / "results" / "a2a_moe_overlap"
    return root / f"single_{stamp}.jsonl"


def _time_layer(
    *,
    pack_fn,
    dispatch_fn,
    compute_fn,
    reverse_fn,
    local_combine_fn,
) -> dict[str, float]:
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev2 = torch.cuda.Event(enable_timing=True)
    ev3 = torch.cuda.Event(enable_timing=True)
    ev4 = torch.cuda.Event(enable_timing=True)
    ev5 = torch.cuda.Event(enable_timing=True)

    ev0.record()
    pack_fn()
    ev1.record()
    dispatch_fn()
    ev2.record()
    compute_fn()
    ev3.record()
    reverse_fn()
    ev4.record()
    local_combine_fn()
    ev5.record()
    ev5.synchronize()

    dispatch_comm = float(ev1.elapsed_time(ev2))
    combine_comm = float(ev3.elapsed_time(ev4))
    return {
        "pack_ms": float(ev0.elapsed_time(ev1)),
        "dispatch_comm_ms": dispatch_comm,
        "compute_ms": float(ev2.elapsed_time(ev3)),
        "combine_comm_ms": combine_comm,
        "a2a_comm_ms": dispatch_comm + combine_comm,
        "local_combine_ms": float(ev4.elapsed_time(ev5)),
        "e2e_ms": float(ev0.elapsed_time(ev5)),
    }


def _worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    try:
        if args.cuda_alloc_conf:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        _init_process_group(rank, world_size, args.port)

        model = _resolve_model_params(args)
        _validate_benchmark_args(args, model, world_size)

        local_tokens = args.total_tokens // world_size
        dtype = _dtype_from_name(args.dtype)
        elem_bytes = _dtype_nbytes(dtype)
        topk = int(model.topk)
        local_slots = local_tokens * topk

        strategy = args.strategy
        if args.overlap == 0.0:
            strategy = "disjoint"
        elif strategy == "disjoint":
            strategy = "hybrid"

        placement = _build_placement(model.num_experts, world_size, args.overlap)
        local_experts = placement.experts_by_rank[rank]
        expert_to_local = {expert_id: idx for idx, expert_id in enumerate(local_experts)}

        layer_plans: list[LayerPlan] = []
        per_layer_send_matrices: list[list[list[int]]] = []
        per_layer_gpu_cv: list[float] = []
        per_layer_logical_gpu_cv: list[float] = []
        per_layer_logical_expert_cv: list[float] = []
        per_layer_intra_rank_logical_expert_cv_mean: list[float] = []
        per_layer_intra_rank_logical_expert_cv_max: list[float] = []
        per_layer_physical_expert_cv: list[float] = []
        per_layer_intra_rank_physical_expert_cv_mean: list[float] = []
        per_layer_intra_rank_physical_expert_cv_max: list[float] = []
        per_layer_cross_slots: list[int] = []
        per_layer_cross_ratio: list[float] = []
        local_first_total = torch.zeros(4, dtype=torch.int64, device=device)
        max_total_recv = 0

        for layer_idx in range(model.num_layers):
            layer_seed = args.seed + 13007 * layer_idx
            logical_ids = _make_logical_routing(
                local_tokens=local_tokens,
                topk=topk,
                num_experts=model.num_experts,
                world_size=world_size,
                rank=rank,
                target_gpu_cv=args.target_gpu_cv,
                target_expert_cv=args.target_expert_cv,
                seed=layer_seed,
            )
            prepared = _prepare_routing(
                logical_ids=logical_ids,
                placement=placement,
                rank=rank,
                world_size=world_size,
                num_experts=model.num_experts,
                strategy=strategy,
            )
            if sum(prepared.send_counts) != local_slots:
                raise RuntimeError(
                    f"layer {layer_idx}: send_counts do not sum to local slots"
                )

            recv_counts = _exchange_counts(prepared.send_counts, device)
            total_recv = int(sum(recv_counts))
            max_total_recv = max(max_total_recv, total_recv)

            send_counts_t = torch.tensor(
                prepared.send_counts, dtype=torch.int64, device=device
            )
            send_matrix_t = _all_gather_1d(send_counts_t, world_size)
            logical_counts_t = prepared.logical_counts.to(device=device)
            physical_counts_t = prepared.physical_counts.view(-1).to(device=device)
            local_first_t = torch.tensor(
                [
                    prepared.local_first_opportunities,
                    prepared.local_first_chosen,
                    prepared.local_first_violations,
                    prepared.unique_remote_destinations,
                ],
                dtype=torch.int64,
                device=device,
            )
            dist.all_reduce(logical_counts_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(physical_counts_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_first_t, op=dist.ReduceOp.SUM)
            local_first_total += local_first_t

            send_expert_buf = prepared.sorted_expert_ids.to(
                device=device, dtype=torch.int32
            )
            recv_expert_buf_plan = torch.empty(total_recv, dtype=torch.int32, device=device)
            if world_size > 1:
                dist.all_to_all_single(
                    recv_expert_buf_plan,
                    send_expert_buf,
                    output_split_sizes=recv_counts,
                    input_split_sizes=prepared.send_counts,
                )
            else:
                recv_expert_buf_plan.copy_(send_expert_buf)

            recv_expert_cpu = [int(x) for x in recv_expert_buf_plan.cpu().tolist()]
            rows_by_local: list[list[int]] = [[] for _ in local_experts]
            recv_local_ids_cpu: list[int] = []
            for row_idx, expert_id in enumerate(recv_expert_cpu):
                if expert_id not in expert_to_local:
                    raise RuntimeError(
                        f"rank {rank} layer {layer_idx} received expert {expert_id}, "
                        f"but placement has only {len(local_experts)} local experts"
                    )
                local_id = expert_to_local[expert_id]
                recv_local_ids_cpu.append(local_id)
                rows_by_local[local_id].append(row_idx)
            rows_by_local_t = [
                torch.tensor(rows, dtype=torch.long, device=device)
                if rows else torch.empty(0, dtype=torch.long, device=device)
                for rows in rows_by_local
            ]
            layer_plans.append(
                LayerPlan(
                    layer_idx=layer_idx,
                    send_counts=prepared.send_counts,
                    recv_counts=recv_counts,
                    total_recv=total_recv,
                    sorted_token_idx=prepared.sorted_token_idx.to(
                        device=device, dtype=torch.long
                    ),
                    send_expert_buf=send_expert_buf,
                    rows_by_local=rows_by_local_t,
                    recv_local_ids=torch.tensor(
                        recv_local_ids_cpu, dtype=torch.int32, device=device
                    ),
                )
            )

            if rank == 0:
                send_matrix_cpu = send_matrix_t.cpu().to(dtype=torch.int64)
                total_layer_slots = int(send_matrix_cpu.sum().item())
                if total_layer_slots != args.total_tokens * topk:
                    raise RuntimeError(
                        f"layer {layer_idx}: global slot count mismatch: "
                        f"got {total_layer_slots}, expected {args.total_tokens * topk}"
                    )
                diag = int(torch.diag(send_matrix_cpu).sum().item())
                cross = total_layer_slots - diag
                logical_counts_cpu = logical_counts_t.cpu().view(world_size, -1)
                physical_counts_cpu = physical_counts_t.cpu().view(
                    world_size, model.num_experts
                )
                gpu_counts = send_matrix_cpu.sum(dim=0)
                logical_owner_cvs = _rank_expert_cvs(logical_counts_cpu)
                physical_owner_cvs = _rank_expert_cvs(
                    physical_counts_cpu, placement.hosted_mask
                )

                per_layer_send_matrices.append(
                    [[int(v) for v in row] for row in send_matrix_cpu.tolist()]
                )
                per_layer_gpu_cv.append(_cv_from_counts(gpu_counts))
                per_layer_logical_gpu_cv.append(_cv_from_counts(logical_counts_cpu.sum(dim=1)))
                per_layer_logical_expert_cv.append(
                    _cv_from_counts(logical_counts_cpu)
                )
                per_layer_intra_rank_logical_expert_cv_mean.append(
                    _safe_mean(logical_owner_cvs)
                )
                per_layer_intra_rank_logical_expert_cv_max.append(
                    max(logical_owner_cvs) if logical_owner_cvs else 0.0
                )
                per_layer_physical_expert_cv.append(
                    _cv_from_counts(physical_counts_cpu, placement.hosted_mask)
                )
                per_layer_intra_rank_physical_expert_cv_mean.append(
                    _safe_mean(physical_owner_cvs)
                )
                per_layer_intra_rank_physical_expert_cv_max.append(
                    max(physical_owner_cvs) if physical_owner_cvs else 0.0
                )
                per_layer_cross_slots.append(cross)
                per_layer_cross_ratio.append(float(cross / max(total_layer_slots, 1)))

        dist.barrier()

        torch.manual_seed(args.seed + rank)
        hidden = (
            torch.randn(local_tokens, model.hidden_size, device=device, dtype=dtype) / 10.0
        ).contiguous()
        send_x_buf = torch.empty(local_slots, model.hidden_size, dtype=dtype, device=device)
        recv_x_buf = torch.empty(max_total_recv, model.hidden_size, dtype=dtype, device=device)
        local_out_buf = torch.empty_like(recv_x_buf)
        recv_expert_buf = torch.empty(max_total_recv, dtype=torch.int32, device=device)
        rev_recv_buf = torch.empty(local_slots, model.hidden_size, dtype=dtype, device=device)
        final_out = torch.empty(local_tokens, model.hidden_size, dtype=dtype, device=device)

        local_expert_count = len(local_experts)
        compute_mode = (
            "torch_expert_gemm" if args.compute_mode == "expert_gemm" else args.compute_mode
        )
        if compute_mode == "triton_fused_moe" and dtype is not torch.bfloat16:
            raise ValueError("--compute-mode triton_fused_moe currently requires --dtype bf16")

        if compute_mode == "triton_fused_moe":
            try:
                from sgl_kernel import moe_align_block_size
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "--compute-mode triton_fused_moe requires the sgl_kernel "
                    "import path. Install it in this Python environment with "
                    "`python -m pip install sglang-kernel`."
                ) from exc
            from workshop.nanovllm_moe.artifacts.moe_backend.triton_fused_moe import (
                triton_fused_moe,
            )
        else:
            moe_align_block_size = None
            triton_fused_moe = None

        if compute_mode != "none":
            w1 = (
                torch.randn(
                    local_expert_count,
                    2 * model.moe_intermediate_size,
                    model.hidden_size,
                    device=device,
                    dtype=dtype,
                )
                / 16.0
            ).contiguous()
            w2 = (
                torch.randn(
                    local_expert_count,
                    model.hidden_size,
                    model.moe_intermediate_size,
                    device=device,
                    dtype=dtype,
                )
                / 16.0
            ).contiguous()
        else:
            w1 = None
            w2 = None

        if compute_mode == "triton_fused_moe":
            block_m = int(args.moe_block_size_m)
            max_padded = max_total_recv + (local_expert_count + 1) * (block_m - 1)
            max_blocks = (max_padded + block_m - 1) // block_m
            sorted_token_ids_buf = torch.empty(max_padded, dtype=torch.int32, device=device)
            expert_ids_buf = torch.empty(max_blocks, dtype=torch.int32, device=device)
            num_tokens_post_padded = torch.zeros(1, dtype=torch.int32, device=device)
            cumsum_buffer = torch.empty(local_expert_count + 2, dtype=torch.int32, device=device)
            topk_weights_buf = torch.ones((max_total_recv, 1), dtype=torch.float32, device=device)
            intermediate_cache1 = torch.empty(
                (max_total_recv, 1, 2 * model.moe_intermediate_size),
                dtype=dtype,
                device=device,
            )
            intermediate_cache2 = torch.empty(
                (max_total_recv, 1, model.moe_intermediate_size),
                dtype=dtype,
                device=device,
            )
            intermediate_cache3 = torch.empty(
                (max_total_recv, 1, model.hidden_size),
                dtype=dtype,
                device=device,
            )
        else:
            block_m = 0
            sorted_token_ids_buf = None
            expert_ids_buf = None
            num_tokens_post_padded = None
            cumsum_buffer = None
            topk_weights_buf = None
            intermediate_cache1 = None
            intermediate_cache2 = None
            intermediate_cache3 = None

        def pack(plan: LayerPlan) -> None:
            send_x_buf[:local_slots].copy_(hidden.index_select(0, plan.sorted_token_idx))

        def dispatch_a2a(plan: LayerPlan) -> None:
            if world_size > 1:
                dist.all_to_all_single(
                    recv_x_buf[: plan.total_recv],
                    send_x_buf[:local_slots],
                    output_split_sizes=plan.recv_counts,
                    input_split_sizes=plan.send_counts,
                )
                dist.all_to_all_single(
                    recv_expert_buf[: plan.total_recv],
                    plan.send_expert_buf,
                    output_split_sizes=plan.recv_counts,
                    input_split_sizes=plan.send_counts,
                )
            else:
                recv_x_buf[: plan.total_recv].copy_(send_x_buf[:local_slots])
                recv_expert_buf[: plan.total_recv].copy_(plan.send_expert_buf)

        def compute(plan: LayerPlan) -> None:
            if compute_mode == "none":
                local_out_buf[: plan.total_recv].copy_(recv_x_buf[: plan.total_recv])
                return
            assert w1 is not None and w2 is not None
            if plan.total_recv == 0:
                return
            if compute_mode == "triton_fused_moe":
                assert moe_align_block_size is not None and triton_fused_moe is not None
                assert sorted_token_ids_buf is not None and expert_ids_buf is not None
                assert num_tokens_post_padded is not None and cumsum_buffer is not None
                assert topk_weights_buf is not None
                assert intermediate_cache1 is not None
                assert intermediate_cache2 is not None
                assert intermediate_cache3 is not None
                topk_ids = plan.recv_local_ids.view(-1, 1)
                topk_weights = topk_weights_buf[: plan.total_recv]
                num_tokens_post_padded.zero_()
                moe_align_block_size(
                    topk_ids,
                    local_expert_count,
                    block_m,
                    sorted_token_ids_buf,
                    expert_ids_buf,
                    num_tokens_post_padded,
                    cumsum_buffer,
                    True,  # pad_sorted_token_ids: write a sentinel into padding slots
                )
                out = triton_fused_moe(
                    hidden_states=recv_x_buf[: plan.total_recv],
                    w1=w1,
                    w2=w2,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    sorted_token_ids=sorted_token_ids_buf,
                    expert_ids=expert_ids_buf,
                    num_tokens_post_padded=num_tokens_post_padded,
                    intermediate_cache1=intermediate_cache1,
                    intermediate_cache2=intermediate_cache2,
                    intermediate_cache3=intermediate_cache3,
                    block_size_m=block_m,
                )
                local_out_buf[: plan.total_recv].copy_(out.squeeze(1))
            elif compute_mode == "torch_expert_gemm":
                for local_idx, rows in enumerate(plan.rows_by_local):
                    if rows.numel() == 0:
                        continue
                    x = recv_x_buf[: plan.total_recv].index_select(0, rows)
                    gate_up = torch.matmul(x, w1[local_idx].t())
                    gate, up = gate_up.split(model.moe_intermediate_size, dim=-1)
                    y = torch.matmul(F.silu(gate) * up, w2[local_idx].t())
                    local_out_buf.index_copy_(0, rows, y)
            else:
                raise ValueError(f"unknown compute_mode: {compute_mode!r}")

        def reverse_a2a(plan: LayerPlan) -> None:
            if world_size > 1:
                dist.all_to_all_single(
                    rev_recv_buf[:local_slots],
                    local_out_buf[: plan.total_recv],
                    output_split_sizes=plan.send_counts,
                    input_split_sizes=plan.recv_counts,
                )
            else:
                rev_recv_buf[:local_slots].copy_(local_out_buf[: plan.total_recv])

        def local_combine(plan: LayerPlan) -> None:
            final_out.zero_()
            final_out.index_add_(0, plan.sorted_token_idx, rev_recv_buf[:local_slots])

        for _ in range(args.warmup):
            for plan in layer_plans:
                pack(plan)
                dispatch_a2a(plan)
                compute(plan)
                reverse_a2a(plan)
                local_combine(plan)
        torch.cuda.synchronize(device)
        dist.barrier()

        metric_names = [
            "pack_ms",
            "dispatch_comm_ms",
            "compute_ms",
            "combine_comm_ms",
            "a2a_comm_ms",
            "local_combine_ms",
            "e2e_ms",
        ]
        step_samples: dict[str, list[float]] = {
            "pack_ms": [],
            "dispatch_comm_ms": [],
            "compute_ms": [],
            "combine_comm_ms": [],
            "a2a_comm_ms": [],
            "local_combine_ms": [],
            "e2e_ms": [],
        }
        per_layer_samples: list[dict[str, list[float]]] = [
            {name: [] for name in metric_names} for _ in range(model.num_layers)
        ]
        for _ in range(args.iters):
            dist.barrier()
            step_acc = {name: 0.0 for name in metric_names}
            for plan in layer_plans:
                cur = _time_layer(
                    pack_fn=lambda plan=plan: pack(plan),
                    dispatch_fn=lambda plan=plan: dispatch_a2a(plan),
                    compute_fn=lambda plan=plan: compute(plan),
                    reverse_fn=lambda plan=plan: reverse_a2a(plan),
                    local_combine_fn=lambda plan=plan: local_combine(plan),
                )
                for key, value in cur.items():
                    per_layer_samples[plan.layer_idx][key].append(value)
                    step_acc[key] += value
            for key, value in step_acc.items():
                step_samples[key].append(value)

        local_means = torch.tensor(
            [_safe_mean(step_samples[name]) for name in metric_names],
            dtype=torch.float64,
            device=device,
        )
        gathered_means = [torch.empty_like(local_means) for _ in range(world_size)]
        dist.all_gather(gathered_means, local_means)
        means_by_rank = torch.stack(gathered_means, dim=0).cpu()

        local_per_layer_means = torch.tensor(
            [
                [_safe_mean(per_layer_samples[layer_idx][name]) for name in metric_names]
                for layer_idx in range(model.num_layers)
            ],
            dtype=torch.float64,
            device=device,
        )
        gathered_per_layer = [
            torch.empty_like(local_per_layer_means) for _ in range(world_size)
        ]
        dist.all_gather(gathered_per_layer, local_per_layer_means)
        per_layer_by_rank = torch.stack(gathered_per_layer, dim=0).cpu()

        if rank == 0:
            send_matrix = torch.tensor(per_layer_send_matrices, dtype=torch.int64).sum(dim=0)
            recv_matrix = send_matrix.t().contiguous()
            per_rank_send_slots = [int(x) for x in send_matrix.sum(dim=1).tolist()]
            per_rank_recv_slots = [int(x) for x in send_matrix.sum(dim=0).tolist()]
            per_rank_cross_send_slots = [
                int(send_matrix[r].sum().item() - send_matrix[r, r].item())
                for r in range(world_size)
            ]
            per_rank_cross_recv_slots = [
                int(recv_matrix[r].sum().item() - recv_matrix[r, r].item())
                for r in range(world_size)
            ]
            total_slots = int(send_matrix.sum().item())
            local_slots_global = int(torch.diag(send_matrix).sum().item())
            cross_slots = total_slots - local_slots_global

            timing: dict[str, float | list[float]] = {}
            for idx, name in enumerate(metric_names):
                vals = [float(x) for x in means_by_rank[:, idx].tolist()]
                timing[f"max_rank_mean_{name}"] = max(vals)
                timing[f"min_rank_mean_{name}"] = min(vals)
                timing[f"avg_rank_mean_{name}"] = _safe_mean(vals)
                timing[f"per_rank_mean_{name}"] = vals

            per_layer_timing: dict[str, list[float]] = {}
            for metric_idx, name in enumerate(metric_names):
                per_layer_timing[f"per_layer_max_rank_mean_{name}"] = [
                    float(per_layer_by_rank[:, layer_idx, metric_idx].max().item())
                    for layer_idx in range(model.num_layers)
                ]

            cross_hidden_bytes = cross_slots * model.hidden_size * elem_bytes
            cross_expert_id_bytes = cross_slots * 4
            result = {
                "impl": "a2a_moe_overlap_all_to_all_single",
                "model_config": model.model_config,
                "world_size": world_size,
                "total_tokens": args.total_tokens,
                "tokens_per_rank": local_tokens,
                "num_layers": model.num_layers,
                "num_experts": model.num_experts,
                "local_experts_per_rank": len(placement.experts_by_rank[0]),
                "topk": topk,
                "hidden_size": model.hidden_size,
                "moe_intermediate_size": model.moe_intermediate_size,
                "dtype": str(dtype),
                "compute_mode": compute_mode,
                "requested_compute_mode": args.compute_mode,
                "moe_block_size_m": args.moe_block_size_m,
                "iters": args.iters,
                "warmup": args.warmup,
                "seed": args.seed,
                "routing_strategy": args.routing_strategy,
                "target_gpu_cv": args.target_gpu_cv,
                "target_gpu_cv_scope": "logical_owner_rank_cv_before_replica_selection",
                "target_expert_cv": args.target_expert_cv,
                "target_expert_cv_scope": "intra_rank_logical_expert_cv_mean",
                "overlap": args.overlap,
                "overlap_semantics": "fraction_of_global_experts_per_rank",
                "strategy": strategy,
                "requested_overlap": placement.requested_overlap,
                "effective_local_expert_fraction": placement.effective_local_expert_fraction,
                "replicas_per_expert": placement.replicas_per_expert,
                "overlap_clamped": placement.overlap_clamped,
                "gpu_cv": _safe_mean(per_layer_gpu_cv),
                "logical_gpu_cv": _safe_mean(per_layer_logical_gpu_cv),
                "logical_expert_cv": _safe_mean(per_layer_logical_expert_cv),
                "intra_rank_logical_expert_cv_mean": _safe_mean(
                    per_layer_intra_rank_logical_expert_cv_mean
                ),
                "intra_rank_logical_expert_cv_max": (
                    max(per_layer_intra_rank_logical_expert_cv_max)
                    if per_layer_intra_rank_logical_expert_cv_max
                    else 0.0
                ),
                "physical_expert_cv": _safe_mean(per_layer_physical_expert_cv),
                "intra_rank_physical_expert_cv_mean": _safe_mean(
                    per_layer_intra_rank_physical_expert_cv_mean
                ),
                "intra_rank_physical_expert_cv_max": (
                    max(per_layer_intra_rank_physical_expert_cv_max)
                    if per_layer_intra_rank_physical_expert_cv_max
                    else 0.0
                ),
                "realized_physical_cv": _safe_mean(per_layer_gpu_cv),
                "realized_logical_expert_cv": _safe_mean(per_layer_logical_expert_cv),
                "realized_physical_expert_cv": _safe_mean(per_layer_physical_expert_cv),
                "realized_intra_rank_physical_expert_cv_mean": _safe_mean(
                    per_layer_intra_rank_physical_expert_cv_mean
                ),
                "per_layer_gpu_cv": per_layer_gpu_cv,
                "per_layer_physical_cv": per_layer_gpu_cv,
                "per_layer_logical_gpu_cv": per_layer_logical_gpu_cv,
                "per_layer_logical_expert_cv": per_layer_logical_expert_cv,
                "per_layer_logical_cv": per_layer_logical_expert_cv,
                "per_layer_intra_rank_logical_expert_cv_mean": (
                    per_layer_intra_rank_logical_expert_cv_mean
                ),
                "per_layer_intra_rank_logical_expert_cv_max": (
                    per_layer_intra_rank_logical_expert_cv_max
                ),
                "per_layer_physical_expert_cv": per_layer_physical_expert_cv,
                "per_layer_intra_rank_physical_expert_cv_mean": (
                    per_layer_intra_rank_physical_expert_cv_mean
                ),
                "per_layer_intra_rank_physical_expert_cv_max": (
                    per_layer_intra_rank_physical_expert_cv_max
                ),
                "send_matrix": [[int(v) for v in row] for row in send_matrix.tolist()],
                "recv_matrix": [[int(v) for v in row] for row in recv_matrix.tolist()],
                "per_layer_send_matrices": per_layer_send_matrices,
                "per_rank_send_slots": per_rank_send_slots,
                "per_rank_recv_slots": per_rank_recv_slots,
                "per_rank_cross_send_slots": per_rank_cross_send_slots,
                "per_rank_cross_recv_slots": per_rank_cross_recv_slots,
                "total_slot_tokens": total_slots,
                "local_slot_tokens": local_slots_global,
                "cross_card_slot_tokens": cross_slots,
                "cross_card_ratio": float(cross_slots / max(total_slots, 1)),
                "per_layer_cross_card_slot_tokens": per_layer_cross_slots,
                "per_layer_cross_card_ratio": per_layer_cross_ratio,
                "dispatch_hidden_cross_bytes": cross_hidden_bytes,
                "combine_hidden_cross_bytes": cross_hidden_bytes,
                "dispatch_expert_id_cross_bytes": cross_expert_id_bytes,
                "total_cross_bytes": 2 * cross_hidden_bytes + cross_expert_id_bytes,
                "payload_bytes_per_rank_including_self": (
                    model.num_layers
                    * (2 * local_slots * model.hidden_size * elem_bytes + local_slots * 4)
                ),
                "local_first_opportunities": int(local_first_total[0].item()),
                "local_first_chosen": int(local_first_total[1].item()),
                "local_first_violations": int(local_first_total[2].item()),
                "unique_remote_destinations_total": int(local_first_total[3].item()),
                "sanity": {
                    "tokens_evenly_split": args.total_tokens % world_size == 0,
                    "send_counts_sum_ok": all(
                        x == local_slots * model.num_layers for x in per_rank_send_slots
                    ),
                    "recv_counts_sum_matches": sum(per_rank_recv_slots) == total_slots,
                    "local_first_ok": int(local_first_total[2].item()) == 0,
                    "logical_gpu_cv_near_target": all(
                        abs(x - args.target_gpu_cv) <= args.cv_tolerance
                        for x in per_layer_logical_gpu_cv
                    ),
                    "expert_cv_near_target": all(
                        abs(x - args.target_expert_cv) <= args.cv_tolerance
                        for x in per_layer_intra_rank_logical_expert_cv_mean
                    ),
                    "intra_rank_logical_expert_cv_near_target": all(
                        abs(x - args.target_expert_cv) <= args.cv_tolerance
                        for x in per_layer_intra_rank_logical_expert_cv_mean
                    ),
                    "all_to_all_shapes_ok": (
                        recv_x_buf.shape[0] == max_total_recv
                        and recv_expert_buf.shape[0] == max_total_recv
                        and rev_recv_buf.shape[0] == local_slots
                    ),
                },
                "timestamp": time.time(),
            }
            result.update(timing)
            result.update(per_layer_timing)

            print(json.dumps(result, ensure_ascii=False))
            outp = _build_output_path(args)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with outp.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
            print(f"[a2a-overlap] appended to {outp}", flush=True)

        dist.barrier()
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def _dry_run(args: argparse.Namespace) -> None:
    model = _resolve_model_params(args)
    world_size = args.world_size
    _validate_benchmark_args(args, model, world_size)
    placement = _build_placement(model.num_experts, world_size, args.overlap)
    local_tokens = args.total_tokens // world_size
    local_slots = local_tokens * model.topk
    send_matrix = torch.zeros((world_size, world_size), dtype=torch.int64)
    local_first = torch.zeros(4, dtype=torch.int64)
    per_layer_send_matrices: list[list[list[int]]] = []
    per_layer_gpu_cv: list[float] = []
    per_layer_logical_gpu_cv: list[float] = []
    per_layer_logical_expert_cv: list[float] = []
    per_layer_intra_rank_logical_expert_cv_mean: list[float] = []
    per_layer_intra_rank_logical_expert_cv_max: list[float] = []
    per_layer_physical_expert_cv: list[float] = []
    per_layer_intra_rank_physical_expert_cv_mean: list[float] = []
    per_layer_intra_rank_physical_expert_cv_max: list[float] = []
    per_layer_cross_slots: list[int] = []
    per_layer_cross_ratio: list[float] = []

    strategy = args.strategy
    if args.overlap == 0.0:
        strategy = "disjoint"
    elif strategy == "disjoint":
        strategy = "hybrid"

    for layer_idx in range(model.num_layers):
        layer_send_matrix = torch.zeros((world_size, world_size), dtype=torch.int64)
        logical_counts = torch.zeros(model.num_experts, dtype=torch.int64)
        physical_counts = torch.zeros((world_size, model.num_experts), dtype=torch.int64)
        layer_local_first = torch.zeros(4, dtype=torch.int64)
        for rank in range(world_size):
            logical_ids = _make_logical_routing(
                local_tokens=local_tokens,
                topk=model.topk,
                num_experts=model.num_experts,
                world_size=world_size,
                rank=rank,
                target_gpu_cv=args.target_gpu_cv,
                target_expert_cv=args.target_expert_cv,
                seed=args.seed + 13007 * layer_idx,
            )
            prepared = _prepare_routing(
                logical_ids=logical_ids,
                placement=placement,
                rank=rank,
                world_size=world_size,
                num_experts=model.num_experts,
                strategy=strategy,
            )
            if sum(prepared.send_counts) != local_slots:
                raise RuntimeError(
                    f"layer {layer_idx} rank {rank}: send_counts do not sum to local slots"
                )
            layer_send_matrix[rank] = torch.tensor(prepared.send_counts, dtype=torch.int64)
            logical_counts += prepared.logical_counts
            physical_counts += prepared.physical_counts
            layer_local_first += torch.tensor(
                [
                    prepared.local_first_opportunities,
                    prepared.local_first_chosen,
                    prepared.local_first_violations,
                    prepared.unique_remote_destinations,
                ],
                dtype=torch.int64,
            )

        layer_total_slots = int(layer_send_matrix.sum().item())
        layer_local_slots = int(torch.diag(layer_send_matrix).sum().item())
        logical_counts_by_owner = logical_counts.view(world_size, -1)
        logical_owner_cvs = _rank_expert_cvs(logical_counts_by_owner)
        physical_owner_cvs = _rank_expert_cvs(physical_counts, placement.hosted_mask)
        send_matrix += layer_send_matrix
        local_first += layer_local_first
        per_layer_send_matrices.append(
            [[int(v) for v in row] for row in layer_send_matrix.tolist()]
        )
        per_layer_gpu_cv.append(_cv_from_counts(layer_send_matrix.sum(dim=0)))
        per_layer_logical_gpu_cv.append(
            _cv_from_counts(logical_counts_by_owner.sum(dim=1))
        )
        per_layer_logical_expert_cv.append(
            _cv_from_counts(logical_counts_by_owner)
        )
        per_layer_intra_rank_logical_expert_cv_mean.append(
            _safe_mean(logical_owner_cvs)
        )
        per_layer_intra_rank_logical_expert_cv_max.append(
            max(logical_owner_cvs) if logical_owner_cvs else 0.0
        )
        per_layer_physical_expert_cv.append(
            _cv_from_counts(physical_counts, placement.hosted_mask)
        )
        per_layer_intra_rank_physical_expert_cv_mean.append(
            _safe_mean(physical_owner_cvs)
        )
        per_layer_intra_rank_physical_expert_cv_max.append(
            max(physical_owner_cvs) if physical_owner_cvs else 0.0
        )
        per_layer_cross_slots.append(layer_total_slots - layer_local_slots)
        per_layer_cross_ratio.append(
            float((layer_total_slots - layer_local_slots) / max(layer_total_slots, 1))
        )

    diag = int(torch.diag(send_matrix).sum().item())
    total_slots = int(send_matrix.sum().item())
    result = {
        "impl": "a2a_moe_overlap_dry_run",
        "model_config": model.model_config,
        "world_size": world_size,
        "total_tokens": args.total_tokens,
        "tokens_per_rank": local_tokens,
        "num_layers": model.num_layers,
        "num_experts": model.num_experts,
        "local_experts_per_rank": len(placement.experts_by_rank[0]),
        "topk": model.topk,
        "hidden_size": model.hidden_size,
        "moe_intermediate_size": model.moe_intermediate_size,
        "target_gpu_cv": args.target_gpu_cv,
        "target_gpu_cv_scope": "logical_owner_rank_cv_before_replica_selection",
        "target_expert_cv": args.target_expert_cv,
        "target_expert_cv_scope": "intra_rank_logical_expert_cv_mean",
        "overlap": args.overlap,
        "overlap_semantics": "fraction_of_global_experts_per_rank",
        "strategy": strategy,
        "routing_strategy": args.routing_strategy,
        "requested_overlap": placement.requested_overlap,
        "effective_local_expert_fraction": placement.effective_local_expert_fraction,
        "replicas_per_expert": placement.replicas_per_expert,
        "overlap_clamped": placement.overlap_clamped,
        "gpu_cv": _safe_mean(per_layer_gpu_cv),
        "logical_gpu_cv": _safe_mean(per_layer_logical_gpu_cv),
        "logical_expert_cv": _safe_mean(per_layer_logical_expert_cv),
        "intra_rank_logical_expert_cv_mean": _safe_mean(
            per_layer_intra_rank_logical_expert_cv_mean
        ),
        "intra_rank_logical_expert_cv_max": (
            max(per_layer_intra_rank_logical_expert_cv_max)
            if per_layer_intra_rank_logical_expert_cv_max
            else 0.0
        ),
        "physical_expert_cv": _safe_mean(per_layer_physical_expert_cv),
        "intra_rank_physical_expert_cv_mean": _safe_mean(
            per_layer_intra_rank_physical_expert_cv_mean
        ),
        "intra_rank_physical_expert_cv_max": (
            max(per_layer_intra_rank_physical_expert_cv_max)
            if per_layer_intra_rank_physical_expert_cv_max
            else 0.0
        ),
        "realized_physical_cv": _safe_mean(per_layer_gpu_cv),
        "realized_logical_expert_cv": _safe_mean(per_layer_logical_expert_cv),
        "realized_physical_expert_cv": _safe_mean(per_layer_physical_expert_cv),
        "realized_intra_rank_physical_expert_cv_mean": _safe_mean(
            per_layer_intra_rank_physical_expert_cv_mean
        ),
        "per_layer_gpu_cv": per_layer_gpu_cv,
        "per_layer_physical_cv": per_layer_gpu_cv,
        "per_layer_logical_gpu_cv": per_layer_logical_gpu_cv,
        "per_layer_logical_expert_cv": per_layer_logical_expert_cv,
        "per_layer_logical_cv": per_layer_logical_expert_cv,
        "per_layer_intra_rank_logical_expert_cv_mean": (
            per_layer_intra_rank_logical_expert_cv_mean
        ),
        "per_layer_intra_rank_logical_expert_cv_max": (
            per_layer_intra_rank_logical_expert_cv_max
        ),
        "per_layer_physical_expert_cv": per_layer_physical_expert_cv,
        "per_layer_intra_rank_physical_expert_cv_mean": (
            per_layer_intra_rank_physical_expert_cv_mean
        ),
        "per_layer_intra_rank_physical_expert_cv_max": (
            per_layer_intra_rank_physical_expert_cv_max
        ),
        "send_matrix": [[int(v) for v in row] for row in send_matrix.tolist()],
        "per_layer_send_matrices": per_layer_send_matrices,
        "cross_card_slot_tokens": total_slots - diag,
        "cross_card_ratio": float((total_slots - diag) / max(total_slots, 1)),
        "per_layer_cross_card_slot_tokens": per_layer_cross_slots,
        "per_layer_cross_card_ratio": per_layer_cross_ratio,
        "local_first_opportunities": int(local_first[0].item()),
        "local_first_chosen": int(local_first[1].item()),
        "local_first_violations": int(local_first[2].item()),
        "sanity": {
            "tokens_evenly_split": args.total_tokens % world_size == 0,
            "send_counts_sum_ok": all(
                int(x) == local_slots * model.num_layers for x in send_matrix.sum(dim=1)
            ),
            "local_first_ok": int(local_first[2].item()) == 0,
            "logical_gpu_cv_near_target": all(
                abs(x - args.target_gpu_cv) <= args.cv_tolerance
                for x in per_layer_logical_gpu_cv
            ),
            "expert_cv_near_target": all(
                abs(x - args.target_expert_cv) <= args.cv_tolerance
                for x in per_layer_intra_rank_logical_expert_cv_mean
            ),
            "intra_rank_logical_expert_cv_near_target": all(
                abs(x - args.target_expert_cv) <= args.cv_tolerance
                for x in per_layer_intra_rank_logical_expert_cv_mean
            ),
        },
    }
    print(json.dumps(result, indent=2))


def _sweep_token_values(specs: Iterable[str], world_size: int) -> list[int]:
    values: list[int] = []
    for spec in specs:
        text = spec.strip().lower()
        if text in {"per_rank_1k", "1k_per_rank", "world_1k"}:
            value = 1024 * world_size
        else:
            value = _parse_size(text)
        if value not in values:
            values.append(value)
    return values


def _append_optional_model_overrides(
    cmd: list[str],
    args: argparse.Namespace,
) -> None:
    for flag, value in (
        ("--num-layers", args.num_layers),
        ("--num-experts", args.num_experts),
        ("--topk", args.topk),
        ("--hidden-size", args.hidden_size),
        ("--moe-intermediate-size", args.moe_intermediate_size),
    ):
        if value is not None:
            cmd.extend([flag, str(value)])


def _replicas_for_overlap(world_size: int, overlap: float) -> int:
    if overlap == 0.0:
        return 1
    return max(1, min(world_size, int(round(overlap * world_size))))


def _run_sweep(args: argparse.Namespace) -> None:
    gpus = _parse_csv_strings(args.gpu_list or "0,1,2,3,4,5,6,7")
    world_sizes = _parse_csv_ints(args.sweep_world_sizes)
    token_specs = _parse_csv_strings(args.sweep_total_tokens)
    overlaps = _parse_csv_floats(args.overlaps)
    strategies = _parse_csv_strings(args.strategies)
    seeds = _parse_csv_ints(args.seeds) if args.seeds else [args.seed]
    for strategy in strategies:
        if strategy not in STRATEGIES:
            raise ValueError(f"Invalid strategy for --strategies: {strategy}")

    out_dir = Path(args.output_dir or (
        Path(__file__).resolve().parents[1]
        / "results"
        / "a2a_moe_overlap"
        / datetime.now().strftime("sweep_%Y%m%d_%H%M%S")
    ))
    jobs_dir = out_dir / "jobs"
    logs_dir = out_dir / "logs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve()
    job_id = 0
    planned_jobs = 0
    all_job_outputs: list[Path] = []

    for ws in world_sizes:
        if ws > len(gpus):
            raise ValueError(f"world_size={ws} needs {ws} GPUs, only got {len(gpus)}")
        groups_per_wave = 1 if args.no_concurrent else max(1, len(gpus) // ws)
        gpu_groups = [gpus[i * ws : (i + 1) * ws] for i in range(groups_per_wave)]
        tasks: list[tuple[int, float, str, int]] = []
        for total_tokens in _sweep_token_values(token_specs, ws):
            for seed in seeds:
                for overlap in overlaps:
                    if overlap == 0.0:
                        tasks.append((total_tokens, overlap, "disjoint", seed))
                    elif _replicas_for_overlap(ws, overlap) == 1:
                        print(
                            f"[sweep] skip redundant ws={ws} overlap={overlap:g}: "
                            "replicas_per_expert=1, same as disjoint",
                            flush=True,
                        )
                    else:
                        for strategy in strategies:
                            tasks.append((total_tokens, overlap, strategy, seed))

        if args.dry_run:
            planned_jobs += len(tasks)
            print(
                f"[sweep dry-run] ws={ws} planned_jobs={len(tasks)}",
                flush=True,
            )
            for total_tokens, overlap, strategy, seed in tasks:
                print(
                    f"[sweep dry-run] ws={ws} tok={total_tokens} "
                    f"overlap={overlap:g} strategy={strategy} seed={seed}",
                    flush=True,
                )
            continue

        for start in range(0, len(tasks), groups_per_wave):
            wave = tasks[start : start + groups_per_wave]
            procs: list[tuple[subprocess.Popen, Path, Path]] = []
            for group_idx, task in enumerate(wave):
                total_tokens, overlap, strategy, seed = task
                gpu_group = gpu_groups[group_idx]
                job_name = (
                    f"ws{ws}_tok{total_tokens}_ov{overlap:g}_"
                    f"{strategy}_seed{seed}_job{job_id}"
                )
                out_file = jobs_dir / f"{job_name}.jsonl"
                log_file = logs_dir / f"{job_name}.log"
                cmd = [
                    sys.executable,
                    str(script),
                    "--world-size",
                    str(ws),
                    "--total-tokens",
                    str(total_tokens),
                    "--overlap",
                    str(overlap),
                    "--strategy",
                    strategy,
                    "--model-config",
                    args.model_config,
                    "--dtype",
                    args.dtype,
                    "--compute-mode",
                    args.compute_mode,
                    "--target-gpu-cv",
                    str(args.target_gpu_cv),
                    "--target-expert-cv",
                    str(args.target_expert_cv),
                    "--cv-tolerance",
                    str(args.cv_tolerance),
                    "--warmup",
                    str(args.warmup),
                    "--iters",
                    str(args.iters),
                    "--seed",
                    str(seed),
                    "--port",
                    str(args.base_port + job_id),
                    "--output",
                    str(out_file),
                ]
                _append_optional_model_overrides(cmd, args)
                if args.cuda_alloc_conf:
                    cmd.extend(["--cuda-alloc-conf", args.cuda_alloc_conf])
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_group)
                with log_file.open("w", encoding="utf-8") as log:
                    print(
                        f"[sweep] launch {job_name} GPUs={env['CUDA_VISIBLE_DEVICES']} "
                        f"port={args.base_port + job_id}",
                        flush=True,
                    )
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                procs.append((proc, out_file, log_file))
                all_job_outputs.append(out_file)
                job_id += 1

            for proc, out_file, log_file in procs:
                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(
                        f"sweep job failed with exit code {rc}; see {log_file}"
                    )
                print(f"[sweep] done {out_file.name}", flush=True)

    if args.dry_run:
        print(f"[sweep dry-run] total planned jobs: {planned_jobs}")
        return

    merged = out_dir / "results.jsonl"
    with merged.open("w", encoding="utf-8") as fout:
        for path in all_job_outputs:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
    print(f"[sweep] merged results: {merged}")
    print(f"[sweep] per-job logs: {logs_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--total-tokens", type=int, default=8192)
    p.add_argument(
        "--model-config",
        type=str,
        default="/home/lzy/models/Qwen3-30B-A3B/config.json",
        help="Model config used for num_layers, hidden_size, MoE intermediate, experts, and topk.",
    )
    p.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help=(
            "Fraction of global experts hosted per rank. 0 means disjoint; "
            "on world=8, 0.25 gives 2 replicas/expert and 0.5 gives 4."
        ),
    )
    p.add_argument("--strategy", choices=ALL_STRATEGIES, default="hybrid")
    p.add_argument(
        "--routing-strategy",
        type=str,
        default="logical_gpu_cv_expert_cv",
        help=(
            "Metadata label for the synthetic routing generator. The current "
            "generator uses logical GPU CV plus intra-rank expert CV."
        ),
    )
    p.add_argument("--target-gpu-cv", type=float, default=0.20)
    p.add_argument("--target-expert-cv", type=float, default=0.30)
    p.add_argument("--cv-tolerance", type=float, default=0.08)
    p.add_argument("--num-layers", type=int, default=None, help="Override config num_hidden_layers.")
    p.add_argument("--num-experts", type=int, default=None, help="Override config num_experts.")
    p.add_argument("--topk", type=int, default=None, help="Override config num_experts_per_tok.")
    p.add_argument("--hidden-size", type=int, default=None, help="Override config hidden_size.")
    p.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=None,
        help="Override config moe_intermediate_size.",
    )
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument(
        "--compute-mode",
        choices=["triton_fused_moe", "torch_expert_gemm", "expert_gemm", "none"],
        default="triton_fused_moe",
        help=(
            "Compute path after ragged A2A. triton_fused_moe uses the PR's "
            "EP-HT-style fused MoE kernel; torch_expert_gemm is the old Python "
            "per-expert matmul loop. expert_gemm is kept as an alias."
        ),
    )
    p.add_argument("--moe-block-size-m", type=int, default=64)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--port", type=int, default=29655)
    p.add_argument("--base-port", type=int, default=29655)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--gpu-list", type=str, default=None)
    p.add_argument("--cuda-alloc-conf", type=str, default="expandable_segments:True")
    p.add_argument("--dry-run", action="store_true", help="Validate routing/placement on CPU only.")
    p.add_argument("--sweep", action="store_true", help="Run the built-in ws/tokens/overlap sweep.")
    p.add_argument("--sweep-world-sizes", type=str, default="2,4,8")
    p.add_argument("--sweep-total-tokens", type=str, default="8192,per_rank_1k")
    p.add_argument("--seeds", type=str, default="7,17,29,43,61")
    p.add_argument("--overlaps", type=str, default="0,0.25,0.5")
    p.add_argument("--strategies", type=str, default="greedy_balance,min_communication,cv_aware,hybrid")
    p.add_argument("--no-concurrent", action="store_true")
    args = p.parse_args()

    if args.sweep:
        _run_sweep(args)
        return

    if args.dry_run:
        _dry_run(args)
        return

    if args.gpu_list:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(_parse_csv_strings(args.gpu_list))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the distributed A2A benchmark")
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"Need {args.world_size} visible CUDA devices, got {torch.cuda.device_count()}"
        )

    mp.spawn(_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
