from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from workshop.nanovllm_moe.services.utils.expert_placement import (
    _local_to_global_for_policy,
    make_placement_section,
)
from workshop.nanovllm_moe.services.utils.expert_replica_placement import (
    REPLICA_PLACEMENT_POLICIES,
    aggregate_profile_traffic,
    build_hosts_by_expert,
    replicas_for_overlap,
    same_numa,
    summarize_replica_placement,
)


STRATEGIES = (
    "disjoint",
    "greedy_balance",
    "min_communication",
    "cv_aware",
    "hybrid",
    "numa_aware_min_communication",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def cv(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 / mean


def primary_owner_from_placement(
    *,
    profile: dict[str, Any],
    placement_json_path: Path | None,
    base_placement: str,
) -> tuple[list[int], dict[str, Any]]:
    num_experts = int(profile["num_experts"])
    ep_size = int(profile["ep_size"])
    if placement_json_path is not None:
        payload = load_json(placement_json_path)
        return [int(x) for x in payload["placement"]["expert_to_rank"]], payload
    local_to_global = _local_to_global_for_policy(num_experts, ep_size, base_placement, seed=0)
    payload = {
        "policy": base_placement,
        "ep_size": ep_size,
        "num_experts": num_experts,
        "placement": make_placement_section(local_to_global, num_experts),
    }
    return [int(x) for x in payload["placement"]["expert_to_rank"]], payload


def score_cv_after(values: list[int], idx: int) -> float:
    updated = list(values)
    updated[idx] += 1
    return cv([float(v) for v in updated])


def choose_remote_host(
    *,
    hosts: list[int],
    source_rank: int,
    expert_id: int,
    strategy: str,
    world_size: int,
    gpu_load: list[int],
    physical_counts: list[list[int]],
    dest_hist: list[int],
) -> int:
    if len(hosts) == 1 or strategy == "disjoint":
        return int(hosts[0])
    if strategy == "greedy_balance":
        return min(hosts, key=lambda h: (gpu_load[h], physical_counts[h][expert_id], h))
    if strategy == "min_communication":
        return min(hosts, key=lambda h: (-dest_hist[h], gpu_load[h], h))
    if strategy == "numa_aware_min_communication":
        same_numa_hosts = [h for h in hosts if same_numa(source_rank, h, world_size)]
        target_hosts = same_numa_hosts or hosts
        return min(target_hosts, key=lambda h: (-dest_hist[h], gpu_load[h], h))
    if strategy == "cv_aware":
        best_host = int(hosts[0])
        best_score = float("inf")
        for h in hosts:
            gpu_score = score_cv_after(gpu_load, h)
            phys_flat = [count for row in physical_counts for count in row]
            expert_index = h * len(physical_counts[0]) + expert_id
            expert_score = score_cv_after(phys_flat, expert_index)
            score = gpu_score + expert_score
            if score < best_score or (score == best_score and h < best_host):
                best_host = h
                best_score = score
        return best_host
    if strategy == "hybrid":
        total = max(float(sum(gpu_load)), 1.0)
        avg_gpu = total / max(float(len(gpu_load)), 1.0)
        positive_slots = sum(1 for row in physical_counts for count in row if count > 0)
        avg_expert = total / max(float(positive_slots), 1.0)
        best_host = int(hosts[0])
        best_score = float("inf")
        for h in hosts:
            gpu_term = float(gpu_load[h] + 1) / max(avg_gpu, 1.0)
            expert_term = float(physical_counts[h][expert_id] + 1) / max(avg_expert, 1.0)
            concentration_bonus = float(dest_hist[h]) / total
            score = 0.50 * gpu_term + 0.45 * expert_term - 0.05 * concentration_bonus
            if score < best_score or (score == best_score and h < best_host):
                best_host = h
                best_score = score
        return best_host
    raise ValueError(f"Unknown strategy: {strategy}")


def assign_traffic(
    *,
    traffic: list[list[int]],
    hosts_by_expert: dict[int, list[int]],
    strategy: str,
    world_size: int,
) -> dict[str, Any]:
    num_experts = len(traffic[0]) if traffic else 0
    rank_load = [0 for _ in range(world_size)]
    physical_counts = [[0 for _ in range(num_experts)] for _ in range(world_size)]
    per_src_outgoing = [0 for _ in range(world_size)]
    per_dst_incoming = [0 for _ in range(world_size)]
    local_hits = 0
    cross_total = 0
    cross_numa_total = 0
    total = 0
    local_first_opportunities = 0
    local_first_chosen = 0
    unique_remote_destinations_total = 0
    for src, row in enumerate(traffic):
        dest_hist = [0 for _ in range(world_size)]
        for expert_id, count in sorted(enumerate(row), key=lambda item: -int(item[1])):
            count = int(count)
            if count <= 0:
                continue
            hosts = [int(h) for h in hosts_by_expert[expert_id]]
            if src in hosts:
                dst = src
                local_first_opportunities += count
                local_first_chosen += count
            else:
                dst = choose_remote_host(
                    hosts=hosts,
                    source_rank=src,
                    expert_id=expert_id,
                    strategy=strategy,
                    world_size=world_size,
                    gpu_load=rank_load,
                    physical_counts=physical_counts,
                    dest_hist=dest_hist,
                )
                if dst != src:
                    unique_remote_destinations_total += 1
            rank_load[dst] += count
            physical_counts[dst][expert_id] += count
            per_src_outgoing[src] += count
            per_dst_incoming[dst] += count
            dest_hist[dst] += count
            total += count
            if dst == src:
                local_hits += count
            else:
                cross_total += count
            if not same_numa(src, dst, world_size):
                cross_numa_total += count
    return {
        "estimated_rank_compute_load": rank_load,
        "estimated_local_traffic": local_hits,
        "estimated_cross_traffic": cross_total,
        "estimated_cross_traffic_ratio": (cross_total / total) if total else 0.0,
        "estimated_cross_numa_traffic": cross_numa_total,
        "estimated_cross_numa_ratio": (cross_numa_total / total) if total else 0.0,
        "estimated_gpu_cv": cv([float(x) for x in rank_load]),
        "per_src_rank_outgoing_replicas": per_src_outgoing,
        "per_dst_rank_incoming_replicas": per_dst_incoming,
        "local_hit_ratio": (local_hits / total) if total else 0.0,
        "local_first_opportunities": local_first_opportunities,
        "local_first_chosen": local_first_chosen,
        "unique_remote_destinations_total": unique_remote_destinations_total,
        "physical_counts": physical_counts,
        "total_routed_replicas": total,
    }


def assign_layerwise(
    *,
    layer_traffic: list[list[list[int]]],
    hosts_by_expert: dict[int, list[int]],
    strategy: str,
    world_size: int,
) -> dict[str, Any]:
    per_layer = [
        assign_traffic(
            traffic=matrix,
            hosts_by_expert=hosts_by_expert,
            strategy=strategy,
            world_size=world_size,
        )
        for matrix in layer_traffic
    ]
    cross_values = [float(item["estimated_cross_traffic_ratio"]) for item in per_layer]
    gpu_values = [float(item["estimated_gpu_cv"]) for item in per_layer]
    local_values = [float(item["local_hit_ratio"]) for item in per_layer]
    xnuma_values = [float(item["estimated_cross_numa_ratio"]) for item in per_layer]
    return {
        "per_layer": per_layer,
        "estimated_cross_traffic_ratio_layer_mean": sum(cross_values) / len(cross_values) if cross_values else 0.0,
        "estimated_gpu_cv_layer_mean": sum(gpu_values) / len(gpu_values) if gpu_values else 0.0,
        "estimated_local_hit_ratio_layer_mean": sum(local_values) / len(local_values) if local_values else 0.0,
        "estimated_cross_numa_ratio_layer_mean": sum(xnuma_values) / len(xnuma_values) if xnuma_values else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--placement-json", type=Path, default=None)
    parser.add_argument("--base-placement", default="contiguous")
    parser.add_argument("--overlap", type=float, required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, default="hybrid")
    parser.add_argument(
        "--replica-placement-policy",
        choices=REPLICA_PLACEMENT_POLICIES,
        default="consecutive_from_primary",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--case-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_json(args.profile)
    aggregated = aggregate_profile_traffic(profile)
    layer_traffic = profile.get("traffic", [])
    world_size = int(profile["ep_size"])
    primary_owner, placement_payload = primary_owner_from_placement(
        profile=profile,
        placement_json_path=args.placement_json,
        base_placement=args.base_placement,
    )
    replicas = replicas_for_overlap(world_size, float(args.overlap))
    effective_policy = (
        "consecutive_from_primary"
        if replicas == 1
        else str(args.replica_placement_policy)
    )
    hosts_by_expert = build_hosts_by_expert(
        primary_owner=primary_owner,
        world_size=world_size,
        overlap=float(args.overlap),
        policy=effective_policy,
        aggregated_traffic=aggregated,
    )
    assigned = assign_traffic(
        traffic=aggregated,
        hosts_by_expert=hosts_by_expert,
        strategy=args.strategy,
        world_size=world_size,
    )
    layerwise = assign_layerwise(
        layer_traffic=layer_traffic,
        hosts_by_expert=hosts_by_expert,
        strategy=args.strategy,
        world_size=world_size,
    )
    base_hosts = build_hosts_by_expert(
        primary_owner=primary_owner,
        world_size=world_size,
        overlap=0.0,
        policy="consecutive_from_primary",
        aggregated_traffic=aggregated,
    )
    base_assigned = assign_traffic(
        traffic=aggregated,
        hosts_by_expert=base_hosts,
        strategy="disjoint",
        world_size=world_size,
    )
    placement_analysis = summarize_replica_placement(
        hosts_by_expert=hosts_by_expert,
        primary_owner=primary_owner,
        aggregated_traffic=aggregated,
        world_size=world_size,
    )
    payload = {
        "runtime_mode": "owner_local_ep",
        "case_id": args.case_id,
        "overlap": float(args.overlap),
        "effective_local_expert_fraction": float(replicas) / float(world_size),
        "replicas_per_expert": replicas,
        "routing_strategy": args.strategy,
        "replica_placement_policy": effective_policy,
        "world_size": world_size,
        "num_experts": int(profile["num_experts"]),
        "base_placement": str(placement_payload.get("policy") or args.base_placement),
        "base_placement_json_path": str(args.placement_json.resolve()) if args.placement_json else None,
        "profile_json_path": str(args.profile.resolve()),
        "primary_owner": primary_owner,
        "hosts_by_expert": {str(expert_id): hosts for expert_id, hosts in hosts_by_expert.items()},
        "estimated_metrics": {
            **assigned,
            **layerwise,
            **placement_analysis,
            "cross_numa_reduction": base_assigned["estimated_cross_numa_ratio"] - assigned["estimated_cross_numa_ratio"],
            "base_estimated_cross_traffic_ratio": base_assigned["estimated_cross_traffic_ratio"],
            "base_estimated_cross_numa_ratio": base_assigned["estimated_cross_numa_ratio"],
            "base_estimated_gpu_cv": base_assigned["estimated_gpu_cv"],
            "base_estimated_local_hit_ratio": base_assigned["local_hit_ratio"],
            "estimated_overlap_memory_multiplier": float(replicas),
        },
        "notes": [
            "Runtime consumes only hosts_by_expert plus routing_strategy; replica placement policy stays generator-side.",
            "Dispatch remains local-first when source rank hosts the expert; remote fallback follows the requested routing strategy.",
        ],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.case_id or f"{args.base_placement}_{effective_policy}_ov{args.overlap}_{args.strategy}"
    out_path = output_dir / f"moe_overlap_plan_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"overlap_plan_path": str(out_path), "estimated_metrics": payload["estimated_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
