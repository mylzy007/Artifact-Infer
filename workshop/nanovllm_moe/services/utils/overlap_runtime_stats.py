"""Rank-local overlap runtime stats recorder aggregated at the end of eval."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from workshop.nanovllm_moe.services.utils.parallel import get_ep_rank, get_runtime_mode


_SEND_MATRICES: dict[int, list[list[int]]] = {}
_META: dict[str, Any] = {}
# Phase 4: drop accounting.
# _DROP_STATS[layer_id] aggregates across forward calls at that layer.
_DROP_STATS: dict[int, dict[str, Any]] = {}


def enabled() -> bool:
    return os.environ.get("MOE_PROFILE_OVERLAP_RUNTIME", "0") == "1"


def set_metadata(**kwargs) -> None:
    _META.update({k: v for k, v in kwargs.items() if v is not None})


def rank_to_numa(rank: int, world_size: int) -> int:
    if world_size <= 1:
        return 0
    split = max(1, world_size // 2)
    return min(rank // split, 1)


def same_numa(rank_a: int, rank_b: int, world_size: int) -> bool:
    return rank_to_numa(rank_a, world_size) == rank_to_numa(rank_b, world_size)


def _cv(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0.0:
        return 0.0
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 / mean


def record_drop(layer_id: int, drop_result, source_rank: int) -> None:
    """Accumulate per-layer drop stats. `drop_result` is `expert_drop.DropResult`."""
    if not enabled():
        return
    layer_id = int(layer_id)
    bucket = _DROP_STATS.setdefault(
        layer_id,
        {
            "total_replicas": 0,
            "total_remote": 0,
            "total_cross_numa": 0,
            "total_weight_mass": 0.0,
            "num_dropped": 0,
            "num_dropped_remote": 0,
            "num_dropped_cross_numa": 0,
            "dropped_weight_mass": 0.0,
            "per_expert_total": {},
            "per_expert_dropped": {},
            "calls": 0,
        },
    )
    bucket["total_replicas"] += int(drop_result.total_replicas)
    bucket["total_remote"] += int(drop_result.total_remote)
    bucket["total_cross_numa"] += int(drop_result.total_cross_numa)
    bucket["total_weight_mass"] += float(drop_result.total_weight_mass)
    bucket["num_dropped"] += int(drop_result.num_dropped)
    bucket["num_dropped_remote"] += int(drop_result.num_dropped_remote)
    bucket["num_dropped_cross_numa"] += int(drop_result.num_dropped_cross_numa)
    bucket["dropped_weight_mass"] += float(drop_result.dropped_weight_mass)
    bucket["calls"] += 1
    pet = bucket["per_expert_total"]
    for e, c in drop_result.per_expert_total.items():
        pet[str(int(e))] = pet.get(str(int(e)), 0) + int(c)
    ped = bucket["per_expert_dropped"]
    for e, c in drop_result.per_expert_dropped.items():
        ped[str(int(e))] = ped.get(str(int(e)), 0) + int(c)


def record_dispatch(layer_id: int, send_counts: list[int], source_rank: int) -> None:
    if not enabled():
        return
    layer_id = int(layer_id)
    world_size = len(send_counts)
    matrix = _SEND_MATRICES.setdefault(
        layer_id,
        [[0 for _ in range(world_size)] for _ in range(world_size)],
    )
    row = matrix[int(source_rank)]
    for dst, count in enumerate(send_counts):
        row[dst] += int(count)


def _rank_payload() -> dict[str, Any]:
    return {
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        "ep_rank": get_ep_rank() if dist.is_initialized() else 0,
        "metadata": dict(_META),
        "send_matrices": _SEND_MATRICES,
        "drop_stats": _DROP_STATS,
    }


def _aggregate_drop_stats(rank_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    layer_buckets: dict[int, dict[str, Any]] = {}
    for payload in rank_payloads:
        for layer_key, bucket in (payload.get("drop_stats") or {}).items():
            lid = int(layer_key)
            agg = layer_buckets.setdefault(
                lid,
                {
                    "total_replicas": 0,
                    "total_remote": 0,
                    "total_cross_numa": 0,
                    "total_weight_mass": 0.0,
                    "num_dropped": 0,
                    "num_dropped_remote": 0,
                    "num_dropped_cross_numa": 0,
                    "dropped_weight_mass": 0.0,
                    "per_expert_total": {},
                    "per_expert_dropped": {},
                },
            )
            for k in (
                "total_replicas",
                "total_remote",
                "total_cross_numa",
                "num_dropped",
                "num_dropped_remote",
                "num_dropped_cross_numa",
            ):
                agg[k] += int(bucket.get(k, 0))
            for k in ("total_weight_mass", "dropped_weight_mass"):
                agg[k] += float(bucket.get(k, 0.0))
            for e, c in (bucket.get("per_expert_total") or {}).items():
                agg["per_expert_total"][str(e)] = agg["per_expert_total"].get(str(e), 0) + int(c)
            for e, c in (bucket.get("per_expert_dropped") or {}).items():
                agg["per_expert_dropped"][str(e)] = agg["per_expert_dropped"].get(str(e), 0) + int(c)

    if not layer_buckets:
        return {
            "drop_replicas_total": 0,
            "drop_fraction_total": 0.0,
            "drop_fraction_remote": 0.0,
            "drop_fraction_cross_numa": 0.0,
            "dropped_weight_mass_fraction": 0.0,
            "per_layer_drop_fraction": [],
            "per_layer_drop_fraction_mean": 0.0,
            "per_expert_drop_fraction": {},
            "effective_dropped_expert_load_cv": 0.0,
        }

    total_replicas = sum(b["total_replicas"] for b in layer_buckets.values())
    total_remote = sum(b["total_remote"] for b in layer_buckets.values())
    total_cross_numa = sum(b["total_cross_numa"] for b in layer_buckets.values())
    total_weight = sum(b["total_weight_mass"] for b in layer_buckets.values())
    drop_total = sum(b["num_dropped"] for b in layer_buckets.values())
    drop_remote = sum(b["num_dropped_remote"] for b in layer_buckets.values())
    drop_cross = sum(b["num_dropped_cross_numa"] for b in layer_buckets.values())
    drop_weight = sum(b["dropped_weight_mass"] for b in layer_buckets.values())

    per_layer = []
    for lid in sorted(layer_buckets):
        b = layer_buckets[lid]
        per_layer.append(
            {
                "layer_id": lid,
                "drop_fraction_total": (b["num_dropped"] / b["total_replicas"]) if b["total_replicas"] else 0.0,
                "drop_fraction_remote": (b["num_dropped_remote"] / b["total_remote"]) if b["total_remote"] else 0.0,
                "drop_fraction_cross_numa": (b["num_dropped_cross_numa"] / b["total_cross_numa"]) if b["total_cross_numa"] else 0.0,
                "dropped_weight_mass_fraction": (b["dropped_weight_mass"] / b["total_weight_mass"]) if b["total_weight_mass"] else 0.0,
            }
        )

    # Cross-layer per-expert aggregates (totaled across layers and ranks).
    total_per_expert: dict[str, int] = {}
    drop_per_expert: dict[str, int] = {}
    for b in layer_buckets.values():
        for e, c in b["per_expert_total"].items():
            total_per_expert[e] = total_per_expert.get(e, 0) + int(c)
        for e, c in b["per_expert_dropped"].items():
            drop_per_expert[e] = drop_per_expert.get(e, 0) + int(c)
    per_expert_drop_fraction = {
        e: (drop_per_expert.get(e, 0) / total_per_expert[e]) if total_per_expert[e] else 0.0
        for e in total_per_expert
    }
    remaining_per_expert = [
        total_per_expert[e] - drop_per_expert.get(e, 0) for e in total_per_expert
    ]
    eff_cv = _cv([float(x) for x in remaining_per_expert]) if remaining_per_expert else 0.0

    per_layer_drop_fraction_mean = (
        sum(item["drop_fraction_total"] for item in per_layer) / max(1, len(per_layer))
    )

    return {
        "drop_replicas_total": drop_total,
        "drop_fraction_total": (drop_total / total_replicas) if total_replicas else 0.0,
        "drop_fraction_remote": (drop_remote / total_remote) if total_remote else 0.0,
        "drop_fraction_cross_numa": (drop_cross / total_cross_numa) if total_cross_numa else 0.0,
        "dropped_weight_mass_fraction": (drop_weight / total_weight) if total_weight else 0.0,
        "per_layer_drop_fraction": per_layer,
        "per_layer_drop_fraction_mean": per_layer_drop_fraction_mean,
        "per_expert_drop_fraction": per_expert_drop_fraction,
        "effective_dropped_expert_load_cv": eff_cv,
    }


def _aggregate_metrics(send_matrices: list[list[list[int]]]) -> dict[str, Any]:
    world_size = len(send_matrices[0]) if send_matrices else int(_META.get("ep_size", 1))
    total_matrix = [[0 for _ in range(world_size)] for _ in range(world_size)]
    for matrix in send_matrices:
        for src, row in enumerate(matrix):
            for dst, count in enumerate(row):
                total_matrix[src][dst] += int(count)

    per_src_outgoing = [sum(row) for row in total_matrix]
    per_dst_incoming = [
        sum(total_matrix[src][dst] for src in range(world_size))
        for dst in range(world_size)
    ]
    local_hits = sum(total_matrix[r][r] for r in range(world_size))
    total = sum(per_src_outgoing)
    cross = total - local_hits
    cross_numa = sum(
        total_matrix[src][dst]
        for src in range(world_size)
        for dst in range(world_size)
        if not same_numa(src, dst, world_size)
    )
    layer_metrics: list[dict[str, Any]] = []
    for layer_id, matrix in enumerate(send_matrices):
        per_src = [sum(row) for row in matrix]
        per_dst = [sum(matrix[src][dst] for src in range(world_size)) for dst in range(world_size)]
        layer_total = sum(per_src)
        layer_local = sum(matrix[r][r] for r in range(world_size))
        layer_cross_numa = sum(
            matrix[src][dst]
            for src in range(world_size)
            for dst in range(world_size)
            if not same_numa(src, dst, world_size)
        )
        layer_metrics.append(
            {
                "layer_id": layer_id,
                "per_src_outgoing": per_src,
                "per_dst_incoming": per_dst,
                "local_hit_ratio": (layer_local / layer_total) if layer_total else 0.0,
                "cross_traffic_ratio": ((layer_total - layer_local) / layer_total) if layer_total else 0.0,
                "cross_numa_ratio": (layer_cross_numa / layer_total) if layer_total else 0.0,
                "gpu_cv": _cv([float(x) for x in per_dst]),
            }
        )

    return {
        "total_routed_replicas": total,
        "local_hit_ratio": (local_hits / total) if total else 0.0,
        "cross_traffic_ratio": (cross / total) if total else 0.0,
        "cross_numa_ratio": (cross_numa / total) if total else 0.0,
        "per_rank_compute_load": per_dst_incoming,
        "per_src_outgoing": per_src_outgoing,
        "per_dst_incoming": per_dst_incoming,
        "gpu_cv": _cv([float(x) for x in per_dst_incoming]),
        "send_matrix": total_matrix,
        "per_layer": layer_metrics,
    }


def save_profile(output_dir: str = "eval_results") -> str | None:
    if not enabled():
        return None
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("MOE_PROFILE_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank_path = Path(output_dir) / f"overlap_runtime_stats_{run_id}_rank{rank}.json"
    rank_path.write_text(json.dumps(_rank_payload(), indent=2))

    if dist.is_initialized():
        dist.barrier()
    if rank != 0:
        return None

    rank_payloads = [
        json.loads((Path(output_dir) / f"overlap_runtime_stats_{run_id}_rank{r}.json").read_text())
        for r in range(world_size)
    ]
    meta = dict(_META)
    if rank_payloads:
        meta.update(rank_payloads[0].get("metadata", {}))

    num_layers = 0
    for payload in rank_payloads:
        for key in payload.get("send_matrices", {}):
            num_layers = max(num_layers, int(key) + 1)
    ep_size = int(meta.get("ep_size") or meta.get("world_size") or world_size)
    send_matrices = [
        [[0 for _ in range(ep_size)] for _ in range(ep_size)]
        for _ in range(num_layers)
    ]
    for payload in rank_payloads:
        for layer_key, matrix in payload.get("send_matrices", {}).items():
            layer_id = int(layer_key)
            for src, row in enumerate(matrix):
                for dst, count in enumerate(row):
                    send_matrices[layer_id][src][dst] += int(count)

    out_payload = {
        **meta,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile_run_id": run_id,
        "runtime_mode": meta.get("runtime_mode", get_runtime_mode()),
        "num_layers": num_layers,
        **_aggregate_metrics(send_matrices),
        **_aggregate_drop_stats(rank_payloads),
    }
    out_path = Path(output_dir) / f"overlap_runtime_stats_{run_id}.json"
    out_path.write_text(json.dumps(out_payload, indent=2))
    print(f"[overlap-runtime] wrote {out_path}", flush=True)
    return str(out_path)


def reset() -> None:
    _SEND_MATRICES.clear()
    _META.clear()
    _DROP_STATS.clear()
