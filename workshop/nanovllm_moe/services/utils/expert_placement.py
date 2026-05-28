"""Expert placement helpers shared by EP dispatch and expert loading."""
from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExpertPlacement:
    policy: str
    num_experts: int
    ep_size: int
    expert_to_rank: list[int]
    expert_to_local: list[int]
    local_to_global: list[list[int]]


def build_expert_placement(
    num_experts: int,
    ep_size: int,
    policy: str = "contiguous",
    *,
    seed: int = 0,
    placement_path: str | None = None,
    layer_id: int | None = None,
) -> ExpertPlacement:
    num_experts = int(num_experts)
    ep_size = int(ep_size)
    policy = str(policy).lower()
    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by ep_size={ep_size}"
        )

    e_local = num_experts // ep_size
    if placement_path:
        return load_expert_placement(
            placement_path, num_experts=num_experts, ep_size=ep_size, layer_id=layer_id,
        )

    local_to_global = _local_to_global_for_policy(num_experts, ep_size, policy, seed)
    expert_to_rank, expert_to_local = _invert_local_to_global(local_to_global, num_experts)

    return ExpertPlacement(
        policy=policy,
        num_experts=num_experts,
        ep_size=ep_size,
        expert_to_rank=expert_to_rank,
        expert_to_local=expert_to_local,
        local_to_global=local_to_global,
    )


def _local_to_global_for_policy(
    num_experts: int, ep_size: int, policy: str, seed: int = 0
) -> list[list[int]]:
    e_local = num_experts // ep_size
    policy = str(policy).lower()
    if policy == "contiguous":
        return [
            [rank * e_local + local for local in range(e_local)]
            for rank in range(ep_size)
        ]
    if policy == "round_robin":
        return [
            [rank + local * ep_size for local in range(e_local)]
            for rank in range(ep_size)
        ]
    if policy == "fixed_random_shuffle":
        expert_ids = list(range(num_experts))
        random.Random(int(seed)).shuffle(expert_ids)
        return [
            expert_ids[rank * e_local : (rank + 1) * e_local]
            for rank in range(ep_size)
        ]
    raise ValueError(
        f"unknown expert placement policy {policy!r}; expected one of "
        "'contiguous', 'round_robin', 'fixed_random_shuffle', or pass placement_path"
    )


def _invert_local_to_global(
    local_to_global: list[list[int]], num_experts: int
) -> tuple[list[int], list[int]]:
    expert_to_rank = [-1] * num_experts
    expert_to_local = [-1] * num_experts
    for rank, experts in enumerate(local_to_global):
        for local, expert_id in enumerate(experts):
            expert_id = int(expert_id)
            if expert_id < 0 or expert_id >= num_experts:
                raise ValueError(f"expert id {expert_id} outside [0, {num_experts})")
            if expert_to_rank[expert_id] != -1:
                raise ValueError(f"expert id {expert_id} appears more than once")
            expert_to_rank[expert_id] = rank
            expert_to_local[expert_id] = local
    missing = [i for i, rank in enumerate(expert_to_rank) if rank < 0]
    if missing:
        raise ValueError(f"placement missing experts: {missing[:8]}")
    return expert_to_rank, expert_to_local


def load_expert_placement(
    placement_path: str,
    *,
    num_experts: int,
    ep_size: int,
    layer_id: int | None = None,
) -> ExpertPlacement:
    with open(placement_path) as f:
        payload = json.load(f)

    if int(payload["num_experts"]) != int(num_experts):
        raise ValueError(
            f"placement num_experts={payload['num_experts']} does not match {num_experts}"
        )
    if int(payload["ep_size"]) != int(ep_size):
        raise ValueError(f"placement ep_size={payload['ep_size']} does not match {ep_size}")
    e_local = num_experts // ep_size
    if int(payload["E_local"]) != e_local:
        raise ValueError(f"placement E_local={payload['E_local']} does not match {e_local}")

    layer_mode = payload.get("layer_mode", "global")
    if layer_mode == "global":
        section = payload["placement"]
    elif layer_mode == "layerwise":
        if layer_id is None:
            raise ValueError("layerwise placement requires layer_id")
        section = payload["placements"][str(layer_id)]
    else:
        raise ValueError(f"unknown placement layer_mode {layer_mode!r}")

    local_to_global = [[int(x) for x in row] for row in section["local_to_global"]]
    if len(local_to_global) != ep_size or any(len(row) != e_local for row in local_to_global):
        raise ValueError("placement local_to_global shape does not match ep_size/E_local")
    expert_to_rank, expert_to_local = _invert_local_to_global(local_to_global, num_experts)
    return ExpertPlacement(
        policy=str(payload["policy"]),
        num_experts=num_experts,
        ep_size=ep_size,
        expert_to_rank=expert_to_rank,
        expert_to_local=expert_to_local,
        local_to_global=local_to_global,
    )


def cv(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    return statistics.pstdev(values) / mean


def _estimated_metrics_from_traffic_matrix(
    traffic: list[list[int]], expert_to_rank: list[int], ep_size: int
) -> dict:
    expert_load = [sum(int(row[e]) for row in traffic) for e in range(len(expert_to_rank))]
    rank_load = [0 for _ in range(ep_size)]
    local_traffic = 0
    per_src_outgoing = [0 for _ in range(ep_size)]
    per_dst_incoming = [0 for _ in range(ep_size)]
    for src, row in enumerate(traffic):
        for expert_id, count in enumerate(row):
            count = int(count)
            dst = int(expert_to_rank[expert_id])
            rank_load[dst] += count
            per_src_outgoing[src] += count
            per_dst_incoming[dst] += count
            if src == dst:
                local_traffic += count
    total = sum(per_src_outgoing)
    cross = total - local_traffic
    return {
        "estimated_rank_compute_load": rank_load,
        "estimated_local_traffic": local_traffic,
        "estimated_cross_traffic": cross,
        "estimated_cross_traffic_ratio": cross / total if total else 0.0,
        "estimated_expert_cv": cv(expert_load),
        "estimated_gpu_cv": cv(rank_load),
        "per_src_rank_outgoing_replicas": per_src_outgoing,
        "per_dst_rank_incoming_replicas": per_dst_incoming,
        "expert_load": expert_load,
    }


def estimated_metrics(
    traffic: list[list[int]],
    expert_to_rank: list[int],
    ep_size: int,
    *,
    layer_traffic: list[list[list[int]]] | None = None,
) -> dict:
    metrics = _estimated_metrics_from_traffic_matrix(traffic, expert_to_rank, ep_size)
    if layer_traffic:
        layer_metrics = [
            _estimated_metrics_from_traffic_matrix(matrix, expert_to_rank, ep_size)
            for matrix in layer_traffic
        ]
        layer_cross_ratios = [
            float(m["estimated_cross_traffic_ratio"]) for m in layer_metrics
        ]
        layer_gpu_cvs = [float(m["estimated_gpu_cv"]) for m in layer_metrics]
        metrics.update(
            {
                "estimated_per_layer_cross_traffic": [
                    int(m["estimated_cross_traffic"]) for m in layer_metrics
                ],
                "estimated_per_layer_local_traffic": [
                    int(m["estimated_local_traffic"]) for m in layer_metrics
                ],
                "estimated_per_layer_cross_traffic_ratio": layer_cross_ratios,
                "estimated_cross_traffic_ratio_layer_mean": (
                    sum(layer_cross_ratios) / len(layer_cross_ratios)
                    if layer_cross_ratios
                    else 0.0
                ),
                "estimated_per_layer_gpu_cv": layer_gpu_cvs,
                "estimated_gpu_cv_layer_mean": (
                    sum(layer_gpu_cvs) / len(layer_gpu_cvs) if layer_gpu_cvs else 0.0
                ),
            }
        )
    return metrics


def make_placement_section(local_to_global: list[list[int]], num_experts: int) -> dict:
    expert_to_rank, expert_to_local = _invert_local_to_global(local_to_global, num_experts)
    return {
        "expert_to_rank": expert_to_rank,
        "expert_to_local": expert_to_local,
        "local_to_global": local_to_global,
    }


def write_placement_json(
    path: str | Path,
    *,
    policy: str,
    num_experts: int,
    ep_size: int,
    local_to_global: list[list[int]],
    seed: int | None = None,
    profile_path: str | None = None,
    traffic: list[list[int]] | None = None,
    layer_traffic: list[list[list[int]]] | None = None,
    layer_mode: str = "global",
) -> str:
    e_local = num_experts // ep_size
    section = make_placement_section(local_to_global, num_experts)
    metrics = {}
    if traffic is not None:
        metrics = estimated_metrics(
            traffic,
            section["expert_to_rank"],
            ep_size,
            layer_traffic=layer_traffic,
        )
    payload = {
        "policy": policy,
        "ep_size": ep_size,
        "num_experts": num_experts,
        "E_local": e_local,
        "seed": seed,
        "profile_json_path": profile_path,
        "layer_mode": layer_mode,
        "placement": section,
        "estimated_metrics": metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return str(path)
