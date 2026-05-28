"""Runtime helpers for static all-expert overlap in owner_local_ep + EP-HT."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from workshop.nanovllm_moe.services.utils.expert_placement import build_expert_placement
from workshop.nanovllm_moe.services.utils.expert_replica_placement import same_numa


ALL_STRATEGIES = (
    "disjoint",
    "greedy_balance",
    "min_communication",
    "cv_aware",
    "hybrid",
    "numa_aware_min_communication",
)


def _cv(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(float(v) for v in values) / float(len(values))
    if mean == 0.0:
        return 0.0
    return (
        sum((float(v) - mean) ** 2 for v in values) / float(len(values))
    ) ** 0.5 / mean


def _score_cv_after(values: list[int], idx: int) -> float:
    updated = list(values)
    updated[idx] += 1
    return _cv(updated)


def _choose_remote_host(
    *,
    hosts: Sequence[int],
    source_rank: int,
    world_size: int,
    expert_id: int,
    strategy: str,
    gpu_load: list[int],
    physical_counts: list[list[int]],
    dest_hist: list[int],
) -> int:
    if not hosts:
        raise ValueError("Expert has no candidate hosts")
    if len(hosts) == 1 or strategy == "disjoint":
        return int(hosts[0])

    if strategy == "greedy_balance":
        return min(
            (int(h) for h in hosts),
            key=lambda h: (gpu_load[h], physical_counts[h][expert_id], h),
        )

    if strategy == "min_communication":
        return min(
            (int(h) for h in hosts),
            key=lambda h: (-dest_hist[h], gpu_load[h], h),
        )

    if strategy == "numa_aware_min_communication":
        local_numa_hosts = [
            int(h) for h in hosts if same_numa(int(h), int(source_rank), int(world_size))
        ]
        candidate_hosts = local_numa_hosts or [int(h) for h in hosts]
        return min(
            candidate_hosts,
            key=lambda h: (-dest_hist[h], gpu_load[h], h),
        )

    if strategy == "cv_aware":
        best_host = int(hosts[0])
        best_score = float("inf")
        for h in hosts:
            h = int(h)
            gpu_score = _score_cv_after(gpu_load, h)
            phys_flat = [count for row in physical_counts for count in row]
            phys_idx = h * len(physical_counts[0]) + expert_id
            expert_score = _score_cv_after(phys_flat, phys_idx)
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
            h = int(h)
            gpu_term = float(gpu_load[h] + 1) / max(avg_gpu, 1.0)
            expert_term = float(physical_counts[h][expert_id] + 1) / max(avg_expert, 1.0)
            concentration_bonus = float(dest_hist[h]) / total
            score = 0.50 * gpu_term + 0.45 * expert_term - 0.05 * concentration_bonus
            if score < best_score or (score == best_score and h < best_host):
                best_host = h
                best_score = score
        return best_host

    raise ValueError(f"Unknown overlap strategy: {strategy!r}")


@dataclass(frozen=True)
class ExpertOverlapPlan:
    enabled: bool
    policy: str
    overlap: float
    effective_local_expert_fraction: float
    replicas_per_expert: int
    world_size: int
    num_experts: int
    base_placement: str
    plan_path: str | None
    hosts_by_expert: list[list[int]]
    experts_by_rank: list[list[int]]
    local_index_by_rank_expert: list[list[int]]

    @property
    def is_disjoint(self) -> bool:
        return self.replicas_per_expert == 1


def load_expert_overlap_plan(
    *,
    num_experts: int,
    world_size: int,
    expert_placement: str = "contiguous",
    expert_placement_seed: int = 0,
    expert_placement_path: str | None = None,
    overlap_path: str | None = None,
) -> ExpertOverlapPlan:
    num_experts = int(num_experts)
    world_size = int(world_size)
    if overlap_path:
        payload = json.loads(Path(overlap_path).read_text())
        if int(payload["num_experts"]) != num_experts:
            raise ValueError(
                f"overlap plan num_experts={payload['num_experts']} does not match {num_experts}"
            )
        if int(payload["world_size"]) != world_size:
            raise ValueError(
                f"overlap plan world_size={payload['world_size']} does not match {world_size}"
            )
        raw_hosts = payload["hosts_by_expert"]
        hosts_by_expert = []
        for expert_id in range(num_experts):
            hosts = raw_hosts.get(str(expert_id), raw_hosts.get(expert_id))
            if hosts is None:
                raise ValueError(f"overlap plan missing hosts for expert {expert_id}")
            hosts_by_expert.append([int(h) for h in hosts])
        overlap = float(payload.get("overlap", 0.0))
        effective_fraction = float(
            payload.get("effective_local_expert_fraction")
            or (len(hosts_by_expert[0]) / float(world_size) if hosts_by_expert else 0.0)
        )
        replicas = int(
            payload.get("replicas_per_expert")
            or (len(hosts_by_expert[0]) if hosts_by_expert else 1)
        )
        base_placement = str(payload.get("base_placement") or expert_placement)
        policy = "overlap_plan"
        plan_path = str(Path(overlap_path).resolve())
    else:
        placement = build_expert_placement(
            num_experts,
            world_size,
            expert_placement,
            seed=expert_placement_seed,
            placement_path=expert_placement_path,
        )
        hosts_by_expert = [[int(rank)] for rank in placement.expert_to_rank]
        overlap = 0.0
        effective_fraction = 1.0 / float(world_size)
        replicas = 1
        base_placement = str(placement.policy)
        policy = str(placement.policy)
        plan_path = None

    experts_by_rank: list[list[int]] = [[] for _ in range(world_size)]
    local_index_by_rank_expert = [[-1 for _ in range(num_experts)] for _ in range(world_size)]
    for expert_id, hosts in enumerate(hosts_by_expert):
        for host in hosts:
            host = int(host)
            if host < 0 or host >= world_size:
                raise ValueError(f"expert {expert_id} host rank {host} outside [0, {world_size})")
            local_id = len(experts_by_rank[host])
            experts_by_rank[host].append(expert_id)
            if local_index_by_rank_expert[host][expert_id] != -1:
                raise ValueError(f"expert {expert_id} appears twice on rank {host}")
            local_index_by_rank_expert[host][expert_id] = local_id

    return ExpertOverlapPlan(
        enabled=bool(overlap_path),
        policy=policy,
        overlap=overlap,
        effective_local_expert_fraction=effective_fraction,
        replicas_per_expert=replicas,
        world_size=world_size,
        num_experts=num_experts,
        base_placement=base_placement,
        plan_path=plan_path,
        hosts_by_expert=hosts_by_expert,
        experts_by_rank=experts_by_rank,
        local_index_by_rank_expert=local_index_by_rank_expert,
    )


@dataclass
class OverlapRoutingResult:
    target_rank: torch.Tensor
    local_expert_index: torch.Tensor


class ExpertOverlapRouter:
    def __init__(self, plan: ExpertOverlapPlan, strategy: str) -> None:
        self.plan = plan
        self.strategy = str(strategy)
        if self.strategy not in ALL_STRATEGIES:
            raise ValueError(
                f"Unknown overlap strategy {self.strategy!r}; expected one of {ALL_STRATEGIES}"
            )

    def route(self, flat_expert_ids: torch.Tensor, source_rank: int) -> OverlapRoutingResult:
        expert_ids = [int(x) for x in flat_expert_ids.detach().to(device="cpu", dtype=torch.int64).tolist()]
        gpu_load = [0 for _ in range(self.plan.world_size)]
        physical_counts = [
            [0 for _ in range(self.plan.num_experts)]
            for _ in range(self.plan.world_size)
        ]
        dest_hist = [0 for _ in range(self.plan.world_size)]
        target_ranks: list[int] = []
        local_eids: list[int] = []

        for expert_id in expert_ids:
            hosts = self.plan.hosts_by_expert[expert_id]
            if source_rank in hosts:
                dst = int(source_rank)
            else:
                dst = _choose_remote_host(
                    hosts=hosts,
                    source_rank=source_rank,
                    world_size=self.plan.world_size,
                    expert_id=expert_id,
                    strategy=self.strategy,
                    gpu_load=gpu_load,
                    physical_counts=physical_counts,
                    dest_hist=dest_hist,
                )
            local_id = self.plan.local_index_by_rank_expert[dst][expert_id]
            if local_id < 0:
                raise RuntimeError(
                    f"expert {expert_id} not hosted on chosen rank {dst}; hosts={hosts}"
                )
            target_ranks.append(dst)
            local_eids.append(local_id)
            gpu_load[dst] += 1
            physical_counts[dst][expert_id] += 1
            dest_hist[dst] += 1

        device = flat_expert_ids.device
        return OverlapRoutingResult(
            target_rank=torch.tensor(target_ranks, dtype=torch.int64, device=device),
            local_expert_index=torch.tensor(local_eids, dtype=torch.int32, device=device),
        )
