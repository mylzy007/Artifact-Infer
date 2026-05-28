from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from workshop.nanovllm_moe.services.utils.expert_placement import (
    _local_to_global_for_policy,
    load_expert_placement,
    make_placement_section,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def aggregate_profile_traffic(profile: dict[str, Any]) -> list[list[int]]:
    traffic = profile["traffic"]
    if not traffic:
        raise ValueError("profile has no traffic")
    src_size = len(traffic[0])
    num_experts = len(traffic[0][0])
    out = [[0 for _ in range(num_experts)] for _ in range(src_size)]
    for matrix in traffic:
        for src in range(src_size):
            for expert_id in range(num_experts):
                out[src][expert_id] += int(matrix[src][expert_id])
    return out


def rank_to_numa(rank: int, world_size: int) -> int:
    if world_size <= 1:
        return 0
    split = max(1, world_size // 2)
    return min(rank // split, 1)


def same_numa(rank_a: int, rank_b: int, world_size: int) -> bool:
    return rank_to_numa(rank_a, world_size) == rank_to_numa(rank_b, world_size)


def compute_expert_scores(
    aggregated_traffic: list[list[int]],
    primary_owner: list[int],
    world_size: int,
) -> dict[str, list[float]]:
    num_experts = len(primary_owner)
    hot = [0.0 for _ in range(num_experts)]
    cross_numa = [0.0 for _ in range(num_experts)]
    src_affinity = [[0.0 for _ in range(num_experts)] for _ in range(world_size)]
    for src, row in enumerate(aggregated_traffic):
        for expert_id, count in enumerate(row):
            count = float(count)
            hot[expert_id] += count
            src_affinity[src][expert_id] += count
            if not same_numa(src, primary_owner[expert_id], world_size):
                cross_numa[expert_id] += count
    max_hot = max(hot) if hot else 1.0
    max_cross = max(cross_numa) if cross_numa else 1.0
    hybrid = [
        0.5 * (hot[idx] / max_hot if max_hot else 0.0) + 0.5 * (cross_numa[idx] / max_cross if max_cross else 0.0)
        for idx in range(num_experts)
    ]
    return {
        "hot": hot,
        "cross_numa": cross_numa,
        "hybrid": hybrid,
        "src_affinity": src_affinity,
    }


def choose_replicated_experts(
    *,
    replica_policy: str,
    hot_fraction: float,
    scores: dict[str, list[float]],
    num_experts: int,
) -> list[int]:
    if replica_policy == "none":
        return []
    budget = max(1, int(math.ceil(num_experts * hot_fraction)))
    if replica_policy == "topk_hot":
        metric = scores["hot"]
    elif replica_policy == "cross_numa_hot":
        metric = scores["cross_numa"]
    elif replica_policy == "hybrid_hot_numa":
        metric = scores["hybrid"]
    else:
        raise ValueError(f"unsupported replica policy: {replica_policy}")
    ranked = sorted(range(num_experts), key=lambda idx: (-metric[idx], idx))
    return ranked[:budget]


def choose_replica_hosts_for_expert(
    *,
    expert_id: int,
    factor: int,
    world_size: int,
    primary_owner: list[int],
    src_affinity: list[list[float]],
    replica_placement_policy: str,
    current_extra_load: list[float],
) -> list[int]:
    primary = int(primary_owner[expert_id])
    copies = max(1, min(world_size, int(factor)))
    if copies == 1:
        return [primary]
    candidates = [rank for rank in range(world_size) if rank != primary]
    if replica_placement_policy == "naive_primary_plus_offsets":
        ordered = sorted(candidates, key=lambda rank: ((rank - primary) % world_size, rank))
    elif replica_placement_policy == "numa_symmetric":
        ordered = sorted(
            candidates,
            key=lambda rank: (
                0 if not same_numa(rank, primary, world_size) else 1,
                abs((rank % max(1, world_size // 2)) - (primary % max(1, world_size // 2))),
                rank,
            ),
        )
    elif replica_placement_policy == "source_traffic_aware":
        ordered = sorted(
            candidates,
            key=lambda rank: (
                -float(src_affinity[rank][expert_id]),
                0 if same_numa(rank, primary, world_size) else 1,
                rank,
            ),
        )
    elif replica_placement_policy == "peak_load_aware":
        ordered = sorted(
            candidates,
            key=lambda rank: (
                float(current_extra_load[rank]),
                0 if same_numa(rank, primary, world_size) else 1,
                -float(src_affinity[rank][expert_id]),
                rank,
            ),
        )
    else:
        raise ValueError(f"unsupported replica placement policy: {replica_placement_policy}")
    chosen = [primary]
    for rank in ordered:
        if rank not in chosen:
            chosen.append(rank)
        if len(chosen) >= copies:
            break
    return chosen


def build_replica_hosts(
    *,
    replicated_experts: list[int],
    replica_factor: int,
    world_size: int,
    primary_owner: list[int],
    src_affinity: list[list[float]],
    replica_placement_policy: str,
) -> dict[int, list[int]]:
    hosts: dict[int, list[int]] = {}
    extra_load = [0.0 for _ in range(world_size)]
    for expert_id in replicated_experts:
        chosen = choose_replica_hosts_for_expert(
            expert_id=expert_id,
            factor=replica_factor,
            world_size=world_size,
            primary_owner=primary_owner,
            src_affinity=src_affinity,
            replica_placement_policy=replica_placement_policy,
            current_extra_load=extra_load,
        )
        hosts[int(expert_id)] = chosen
        for rank in chosen[1:]:
            extra_load[rank] += float(src_affinity[rank][expert_id])
    return hosts


def select_target_rank(
    *,
    src: int,
    expert_id: int,
    count: int,
    primary: int,
    hosts: list[int],
    routing_policy: str,
    world_size: int,
    current_rank_load: list[int],
) -> tuple[int, str]:
    if routing_policy == "fixed_primary":
        return primary, "primary"
    if routing_policy in {
        "local_first",
        "score_margin_local_first",
        "strict_local_first",
        "local_first_then_primary",
        "local_only_or_primary",
    }:
        if src in hosts:
            return src, "local_replica" if src != primary else "primary"
        return primary, "primary"
    if routing_policy in {"numa_first", "local_first_then_numa_first"}:
        if src in hosts:
            return src, "local_replica" if src != primary else "primary"
        same_numa_hosts = [rank for rank in hosts if same_numa(src, rank, world_size)]
        if src in same_numa_hosts:
            return src, "local_replica" if src != primary else "primary"
        if same_numa_hosts:
            chosen = min(same_numa_hosts, key=lambda rank: (current_rank_load[rank], rank))
            return chosen, "numa_replica" if chosen != primary else "primary"
        return primary, "primary"
    if routing_policy in {"least_loaded", "local_first_then_least_loaded"}:
        if src in hosts:
            return src, "local_replica" if src != primary else "primary"
        chosen = min(hosts, key=lambda rank: (current_rank_load[rank], abs(rank - src), rank))
        if chosen == primary:
            return chosen, "primary"
        if chosen == src:
            return chosen, "local_replica"
        return chosen, "remote_replica"
    raise ValueError(f"unsupported routing policy: {routing_policy}")


def assign_traffic(
    *,
    traffic: list[list[int]],
    primary_owner: list[int],
    replica_hosts: dict[int, list[int]],
    routing_policy: str,
    world_size: int,
) -> dict[str, Any]:
    num_experts = len(primary_owner)
    assigned = [[0 for _ in range(num_experts)] for _ in range(world_size)]
    rank_load = [0 for _ in range(world_size)]
    local_hits = 0
    replica_hits = 0
    cross_total = 0
    cross_numa_total = 0
    total = 0
    per_dst_incoming = [0 for _ in range(world_size)]
    per_src_outgoing = [0 for _ in range(world_size)]
    for expert_id in sorted(range(num_experts), key=lambda idx: -sum(int(row[idx]) for row in traffic)):
        primary = int(primary_owner[expert_id])
        hosts = list(replica_hosts.get(expert_id, [primary]))
        for src, row in enumerate(traffic):
            count = int(row[expert_id])
            if count <= 0:
                continue
            dst, hit_kind = select_target_rank(
                src=src,
                expert_id=expert_id,
                count=count,
                primary=primary,
                hosts=hosts,
                routing_policy=routing_policy,
                world_size=world_size,
                current_rank_load=rank_load,
            )
            assigned[dst][expert_id] += count
            rank_load[dst] += count
            per_dst_incoming[dst] += count
            per_src_outgoing[src] += count
            total += count
            if dst == src:
                local_hits += count
            if dst != primary:
                replica_hits += count
            if dst != src:
                cross_total += count
            if not same_numa(src, dst, world_size):
                cross_numa_total += count
    gpu_cv = 0.0
    if rank_load and sum(rank_load) > 0:
        mean = sum(rank_load) / len(rank_load)
        gpu_cv = (sum((load - mean) ** 2 for load in rank_load) / len(rank_load)) ** 0.5 / mean
    return {
        "assigned_rank_expert_traffic": assigned,
        "estimated_rank_compute_load": rank_load,
        "estimated_local_traffic": local_hits,
        "estimated_cross_traffic": cross_total,
        "estimated_cross_traffic_ratio": (cross_total / total) if total else 0.0,
        "estimated_cross_numa_traffic": cross_numa_total,
        "estimated_cross_numa_ratio": (cross_numa_total / total) if total else 0.0,
        "estimated_gpu_cv": gpu_cv,
        "per_src_rank_outgoing_replicas": per_src_outgoing,
        "per_dst_rank_incoming_replicas": per_dst_incoming,
        "local_hit_ratio": (local_hits / total) if total else 0.0,
        "replica_hit_ratio": (replica_hits / total) if total else 0.0,
        "total_routed_replicas": total,
    }


def assign_layerwise_traffic(
    *,
    layer_traffic: list[list[list[int]]],
    primary_owner: list[int],
    replica_hosts: dict[int, list[int]],
    routing_policy: str,
    world_size: int,
) -> dict[str, Any]:
    per_layer = [
        assign_traffic(
            traffic=matrix,
            primary_owner=primary_owner,
            replica_hosts=replica_hosts,
            routing_policy=routing_policy,
            world_size=world_size,
        )
        for matrix in layer_traffic
    ]
    cross_values = [float(item["estimated_cross_traffic_ratio"]) for item in per_layer]
    gpu_values = [float(item["estimated_gpu_cv"]) for item in per_layer]
    local_values = [float(item["local_hit_ratio"]) for item in per_layer]
    replica_values = [float(item["replica_hit_ratio"]) for item in per_layer]
    cross_numa_values = [float(item["estimated_cross_numa_ratio"]) for item in per_layer]
    return {
        "per_layer": per_layer,
        "estimated_cross_traffic_ratio_layer_mean": sum(cross_values) / len(cross_values) if cross_values else 0.0,
        "estimated_gpu_cv_layer_mean": sum(gpu_values) / len(gpu_values) if gpu_values else 0.0,
        "estimated_local_hit_ratio_layer_mean": sum(local_values) / len(local_values) if local_values else 0.0,
        "estimated_replica_hit_ratio_layer_mean": sum(replica_values) / len(replica_values) if replica_values else 0.0,
        "estimated_cross_numa_ratio_layer_mean": sum(cross_numa_values) / len(cross_numa_values) if cross_numa_values else 0.0,
    }


def load_primary_owner(
    *,
    profile: dict[str, Any],
    placement_json_path: Path | None,
    base_placement: str,
) -> tuple[list[int], dict[str, Any]]:
    num_experts = int(profile["num_experts"])
    ep_size = int(profile["ep_size"])
    if placement_json_path is not None:
        placement_payload = load_json(placement_json_path)
        placement_section = placement_payload["placement"]
        return [int(x) for x in placement_section["expert_to_rank"]], placement_payload
    local_to_global = _local_to_global_for_policy(num_experts, ep_size, base_placement, seed=0)
    payload = {
        "policy": base_placement,
        "ep_size": ep_size,
        "num_experts": num_experts,
        "placement": make_placement_section(local_to_global, num_experts),
    }
    return [int(x) for x in payload["placement"]["expert_to_rank"]], payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--placement-json", type=Path, default=None)
    parser.add_argument("--base-placement", default="contiguous")
    parser.add_argument("--replica-policy", choices=["none", "topk_hot", "cross_numa_hot", "hybrid_hot_numa"], default="none")
    parser.add_argument(
        "--routing-policy",
        choices=[
            "fixed_primary",
            "local_first",
            "numa_first",
            "least_loaded",
            "score_margin_local_first",
            "strict_local_first",
            "local_first_then_primary",
            "local_first_then_numa_first",
            "local_first_then_least_loaded",
            "local_only_or_primary",
        ],
        default="local_first_then_primary",
    )
    parser.add_argument("--replica-placement-policy", choices=["naive_primary_plus_offsets", "numa_symmetric", "source_traffic_aware", "peak_load_aware"], default="naive_primary_plus_offsets")
    parser.add_argument("--hot-fraction", type=float, default=0.10)
    parser.add_argument("--replica-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--case-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_json(args.profile)
    aggregated = aggregate_profile_traffic(profile)
    layer_traffic = profile.get("traffic", [])
    world_size = int(profile["ep_size"])
    num_experts = int(profile["num_experts"])
    primary_owner, placement_payload = load_primary_owner(
        profile=profile,
        placement_json_path=args.placement_json,
        base_placement=args.base_placement,
    )
    scores = compute_expert_scores(aggregated, primary_owner, world_size)
    replicated_experts = choose_replicated_experts(
        replica_policy=args.replica_policy,
        hot_fraction=float(args.hot_fraction),
        scores=scores,
        num_experts=num_experts,
    )
    replica_hosts = build_replica_hosts(
        replicated_experts=replicated_experts,
        replica_factor=int(args.replica_factor),
        world_size=world_size,
        primary_owner=primary_owner,
        src_affinity=scores["src_affinity"],
        replica_placement_policy=args.replica_placement_policy,
    )
    effective_routing = "local_first" if args.routing_policy == "score_margin_local_first" else args.routing_policy
    assigned = assign_traffic(
        traffic=aggregated,
        primary_owner=primary_owner,
        replica_hosts=replica_hosts,
        routing_policy=effective_routing,
        world_size=world_size,
    )
    layerwise = assign_layerwise_traffic(
        layer_traffic=layer_traffic,
        primary_owner=primary_owner,
        replica_hosts=replica_hosts,
        routing_policy=effective_routing,
        world_size=world_size,
    )
    base_assigned = assign_traffic(
        traffic=aggregated,
        primary_owner=primary_owner,
        replica_hosts={},
        routing_policy="fixed_primary",
        world_size=world_size,
    )
    extra_replica_copies = sum(max(0, len(hosts) - 1) for hosts in replica_hosts.values())
    cross_numa_reduction = base_assigned["estimated_cross_numa_ratio"] - assigned["estimated_cross_numa_ratio"]
    payload = {
        "runtime_mode": "owner_local_ep",
        "case_id": args.case_id,
        "policy": args.replica_policy,
        "routing_policy": args.routing_policy,
        "effective_routing_policy": effective_routing,
        "replica_placement_policy": args.replica_placement_policy,
        "world_size": world_size,
        "num_experts": num_experts,
        "hot_fraction": float(args.hot_fraction),
        "replica_factor": int(args.replica_factor),
        "replica_budget_experts": len(replicated_experts),
        "base_placement": str(placement_payload.get("policy") or args.base_placement),
        "base_placement_json_path": str(args.placement_json.resolve()) if args.placement_json else None,
        "profile_json_path": str(args.profile.resolve()),
        "primary_owner": primary_owner,
        "replicated_expert_ids": replicated_experts,
        "replicas": {str(expert_id): hosts for expert_id, hosts in sorted(replica_hosts.items())},
        "selection_scores": {
            "hot": [float(x) for x in scores["hot"]],
            "cross_numa": [float(x) for x in scores["cross_numa"]],
            "hybrid": [float(x) for x in scores["hybrid"]],
        },
        "estimated_metrics": {
            **assigned,
            **layerwise,
            "cross_numa_reduction": cross_numa_reduction,
            "estimated_replica_memory_cost": (extra_replica_copies / num_experts) if num_experts else 0.0,
            "replica_host_count_total": extra_replica_copies,
            "base_estimated_cross_traffic_ratio": base_assigned["estimated_cross_traffic_ratio"],
            "base_estimated_cross_numa_ratio": base_assigned["estimated_cross_numa_ratio"],
            "base_estimated_gpu_cv": base_assigned["estimated_gpu_cv"],
        },
        "notes": [
            "score_margin_local_first currently falls back to local_first in offline planning.",
            "strict_local_first and local_only_or_primary currently share the same offline behavior.",
            "This plan changes expert owner selection only at the analysis layer; runtime dispatch integration remains TODO.",
        ],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.case_id or f"{args.replica_policy}_{args.routing_policy}_{args.replica_placement_policy}"
    out_path = output_dir / f"moe_replica_plan_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"replica_plan_path": str(out_path), "estimated_metrics": payload["estimated_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
