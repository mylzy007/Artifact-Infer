"""Static expert replica placement helpers for overlap experiments."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


REPLICA_PLACEMENT_POLICIES = (
    "consecutive_from_primary",
    "numa_local_first",
    "traffic_aware_greedy",
    "traffic_plus_balance_greedy",
)


def rank_to_numa(rank: int, world_size: int) -> int:
    if world_size <= 1:
        return 0
    split = max(1, world_size // 2)
    return min(rank // split, 1)


def same_numa(rank_a: int, rank_b: int, world_size: int) -> bool:
    return rank_to_numa(rank_a, world_size) == rank_to_numa(rank_b, world_size)


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


def replicas_for_overlap(world_size: int, overlap: float) -> int:
    if overlap == 0.0:
        return 1
    return max(1, min(world_size, int(round(float(overlap) * float(world_size)))))


def _normalized_entropy(counts: Sequence[int]) -> float:
    total = float(sum(int(x) for x in counts))
    if total <= 0.0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts:
        p = float(count) / total
        if p > 0.0:
            entropy -= p * math.log(p)
    return entropy / math.log(float(len(counts)))


def _circular_distance(a: int, b: int, world_size: int) -> int:
    direct = abs(int(a) - int(b))
    return min(direct, max(0, int(world_size) - direct))


def _candidate_priority(
    *,
    rank: int,
    primary: int,
    expert_id: int,
    policy: str,
    world_size: int,
    aggregated_traffic: Sequence[Sequence[int]],
    extra_host_counts: Sequence[int],
    extra_host_load: Sequence[float],
) -> tuple[float, ...]:
    affinity = float(aggregated_traffic[rank][expert_id])
    same_numa_penalty = 0.0 if same_numa(rank, primary, world_size) else 1.0
    cross_numa_penalty = 0.0 if not same_numa(rank, primary, world_size) else 1.0
    distance = float(_circular_distance(rank, primary, world_size))
    if policy == "consecutive_from_primary":
        return (distance, float(rank))
    if policy == "numa_local_first":
        # Despite the legacy name, this policy now prioritizes cross-NUMA
        # coverage so the first extra replica lands on the other NUMA island
        # before filling additional same-NUMA slots.
        return (cross_numa_penalty, -affinity, distance, float(rank))
    if policy == "traffic_aware_greedy":
        return (-affinity, same_numa_penalty, float(extra_host_counts[rank]), distance, float(rank))
    if policy == "traffic_plus_balance_greedy":
        return (
            float(extra_host_load[rank]),
            float(extra_host_counts[rank]),
            -affinity,
            same_numa_penalty,
            distance,
            float(rank),
        )
    raise ValueError(f"Unknown replica placement policy: {policy!r}")


def _preferred_ranks(
    *,
    expert_id: int,
    primary: int,
    chosen: Sequence[int],
    policy: str,
    world_size: int,
    aggregated_traffic: Sequence[Sequence[int]],
    extra_host_counts: Sequence[int],
    extra_host_load: Sequence[float],
) -> list[int]:
    return sorted(
        [
            rank
            for rank in range(world_size)
            if rank not in chosen
        ],
        key=lambda rank: _candidate_priority(
            rank=rank,
            primary=primary,
            expert_id=expert_id,
            policy=policy,
            world_size=world_size,
            aggregated_traffic=aggregated_traffic,
            extra_host_counts=extra_host_counts,
            extra_host_load=extra_host_load,
        ),
    )


def _stage_quota(
    *,
    rank: int,
    remaining_quota: Sequence[int],
    remaining_stages: int,
) -> int:
    quota = int(remaining_quota[rank])
    stages = max(1, int(remaining_stages))
    return max(0, int(math.ceil(float(quota) / float(stages))))


def _assign_stage_hosts(
    *,
    expert_priority: Sequence[int],
    primary_owner: Sequence[int],
    chosen_hosts_by_expert: dict[int, list[int]],
    policy: str,
    world_size: int,
    aggregated_traffic: Sequence[Sequence[int]],
    extra_host_counts: Sequence[int],
    extra_host_load: Sequence[float],
    remaining_quota: Sequence[int],
    remaining_stages: int,
) -> dict[int, int]:
    stage_capacity = [
        _stage_quota(rank=rank, remaining_quota=remaining_quota, remaining_stages=remaining_stages)
        for rank in range(world_size)
    ]
    slot_ranks = [
        rank
        for rank, capacity in enumerate(stage_capacity)
        for _ in range(int(capacity))
    ]
    slots_by_rank = {
        rank: [slot_id for slot_id, slot_rank in enumerate(slot_ranks) if slot_rank == rank]
        for rank in range(world_size)
    }
    preferences = {
        int(expert_id): _preferred_ranks(
            expert_id=int(expert_id),
            primary=int(primary_owner[expert_id]),
            chosen=chosen_hosts_by_expert[int(expert_id)],
            policy=policy,
            world_size=world_size,
            aggregated_traffic=aggregated_traffic,
            extra_host_counts=extra_host_counts,
            extra_host_load=extra_host_load,
        )
        for expert_id in expert_priority
    }
    expert_to_slots = {
        expert_id: [
            slot_id
            for rank in preferences[expert_id]
            for slot_id in slots_by_rank[rank]
        ]
        for expert_id in expert_priority
    }
    slot_owner = [-1 for _ in slot_ranks]

    def try_assign(expert_id: int, seen_slots: set[int]) -> bool:
        for slot_id in expert_to_slots[expert_id]:
            if slot_id in seen_slots:
                continue
            seen_slots.add(slot_id)
            owner = slot_owner[slot_id]
            if owner < 0 or try_assign(owner, seen_slots):
                slot_owner[slot_id] = int(expert_id)
                return True
        return False

    assigned: dict[int, int] = {}
    for expert_id in expert_priority:
        if not try_assign(int(expert_id), set()):
            raise RuntimeError(
                f"could not satisfy balanced replica quota for expert {expert_id}"
            )
    for slot_id, expert_id in enumerate(slot_owner):
        if expert_id >= 0:
            assigned[int(expert_id)] = int(slot_ranks[slot_id])
    if len(assigned) != len(expert_priority):
        missing = [expert_id for expert_id in expert_priority if int(expert_id) not in assigned]
        raise RuntimeError(f"missing stage assignments for experts: {missing[:8]}")
    return assigned


def build_hosts_by_expert(
    *,
    primary_owner: Sequence[int],
    world_size: int,
    overlap: float,
    policy: str,
    aggregated_traffic: Sequence[Sequence[int]],
) -> dict[int, list[int]]:
    world_size = int(world_size)
    replicas = replicas_for_overlap(world_size, overlap)
    num_experts = len(primary_owner)
    if replicas == 1:
        return {expert_id: [int(primary_owner[expert_id])] for expert_id in range(num_experts)}

    if len(aggregated_traffic) != world_size:
        raise ValueError(
            f"aggregated_traffic has src_size={len(aggregated_traffic)} but world_size={world_size}"
        )

    hosts_by_expert: dict[int, list[int]] = {}
    extra_host_counts = [0 for _ in range(world_size)]
    extra_host_load = [0.0 for _ in range(world_size)]
    total_extra_hosts = num_experts * max(0, replicas - 1)
    base_quota = total_extra_hosts // world_size
    quota_remainder = total_extra_hosts % world_size
    remaining_quota = [
        base_quota + (1 if rank < quota_remainder else 0)
        for rank in range(world_size)
    ]
    expert_priority = sorted(
        range(num_experts),
        key=lambda expert_id: (
            -sum(int(aggregated_traffic[src][expert_id]) for src in range(world_size)),
            expert_id,
        ),
    )

    for expert_id in expert_priority:
        hosts_by_expert[expert_id] = [int(primary_owner[expert_id])]

    for stage_idx in range(replicas - 1):
        stage_assignment = _assign_stage_hosts(
            expert_priority=expert_priority,
            primary_owner=primary_owner,
            chosen_hosts_by_expert=hosts_by_expert,
            policy=policy,
            world_size=world_size,
            aggregated_traffic=aggregated_traffic,
            extra_host_counts=extra_host_counts,
            extra_host_load=extra_host_load,
            remaining_quota=remaining_quota,
            remaining_stages=(replicas - 1 - stage_idx),
        )
        for expert_id in expert_priority:
            best_rank = int(stage_assignment[int(expert_id)])
            hosts_by_expert[int(expert_id)].append(best_rank)
            extra_host_counts[best_rank] += 1
            extra_host_load[best_rank] += float(aggregated_traffic[best_rank][int(expert_id)])
            remaining_quota[best_rank] -= 1
    return {expert_id: hosts_by_expert[expert_id] for expert_id in range(num_experts)}


def summarize_replica_placement(
    *,
    hosts_by_expert: dict[int, list[int]],
    primary_owner: Sequence[int],
    aggregated_traffic: Sequence[Sequence[int]],
    world_size: int,
) -> dict[str, float | list[int]]:
    extra_host_counts = [0 for _ in range(world_size)]
    total_traffic = 0
    base_local_eligible = 0
    replica_local_eligible = 0
    for src, row in enumerate(aggregated_traffic):
        for expert_id, count in enumerate(row):
            count = int(count)
            total_traffic += count
            if int(primary_owner[expert_id]) == src:
                base_local_eligible += count
            if src in hosts_by_expert[expert_id]:
                replica_local_eligible += count
    for expert_id, hosts in hosts_by_expert.items():
        primary = int(primary_owner[expert_id])
        for host in hosts[1:]:
            if int(host) != primary:
                extra_host_counts[int(host)] += 1
    gain = max(0.0, float(replica_local_eligible - base_local_eligible))
    traffic_denom = float(total_traffic) if total_traffic > 0 else 1.0
    extra_memory = max(0.0, (sum(len(hosts) for hosts in hosts_by_expert.values()) / max(1.0, float(len(hosts_by_expert)))) - 1.0)
    affinity_gain = gain / traffic_denom
    return {
        "replica_host_entropy": _normalized_entropy(extra_host_counts),
        "source_to_host_affinity_gain": affinity_gain,
        "effective_gain_per_extra_memory": affinity_gain / extra_memory if extra_memory > 0.0 else 0.0,
        "extra_replica_hosts_per_rank": extra_host_counts,
    }
