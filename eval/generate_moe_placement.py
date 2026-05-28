"""Generate static MoE expert placement JSON from a routing profile.

Examples:
  python -m eval.generate_moe_placement --profile eval_results/moe_routing_profile_x.json \
      --policy fixed_random_shuffle --seed 1234
  python -m eval.generate_moe_placement --profile eval_results/moe_routing_profile_x.json \
      --policy load_balanced_greedy_with_locality_tiebreak
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from workshop.nanovllm_moe.services.utils.expert_placement import (
    _local_to_global_for_policy,
    estimated_metrics,
    make_placement_section,
    write_placement_json,
)


def _aggregate_profile_traffic(profile: dict) -> list[list[int]]:
    traffic = profile["traffic"]
    if not traffic:
        raise ValueError("profile has no traffic")
    ep_size = int(profile["ep_size"])
    num_experts = int(profile["num_experts"])
    out = [[0 for _ in range(num_experts)] for _ in range(ep_size)]
    for matrix in traffic:
        for src in range(ep_size):
            for expert_id in range(num_experts):
                out[src][expert_id] += int(matrix[src][expert_id])
    return out


def _balanced_greedy(traffic: list[list[int]], ep_size: int, num_experts: int) -> list[list[int]]:
    e_local = num_experts // ep_size
    expert_load = [sum(row[e] for row in traffic) for e in range(num_experts)]
    local_to_global = [[] for _ in range(ep_size)]
    rank_load = [0 for _ in range(ep_size)]
    for expert_id in sorted(range(num_experts), key=lambda e: (-expert_load[e], e)):
        candidates = [r for r in range(ep_size) if len(local_to_global[r]) < e_local]
        rank = min(
            candidates,
            key=lambda r: (
                rank_load[r],
                -traffic[r][expert_id],
                len(local_to_global[r]),
                r,
            ),
        )
        local_to_global[rank].append(expert_id)
        rank_load[rank] += expert_load[expert_id]
    return local_to_global


def _comm_greedy(traffic: list[list[int]], ep_size: int, num_experts: int) -> list[list[int]]:
    e_local = num_experts // ep_size
    priorities = []
    for expert_id in range(num_experts):
        counts = [traffic[src][expert_id] for src in range(ep_size)]
        order = sorted(counts, reverse=True)
        best = order[0] if order else 0
        second = order[1] if len(order) > 1 else 0
        preferred = min(range(ep_size), key=lambda src: (-traffic[src][expert_id], src))
        total = sum(counts)
        priorities.append((best - second, total, expert_id, preferred))
    local_to_global = [[] for _ in range(ep_size)]
    for _priority, _total, expert_id, preferred in sorted(
        priorities, key=lambda x: (-x[0], -x[1], x[2])
    ):
        if len(local_to_global[preferred]) < e_local:
            rank = preferred
        else:
            candidates = [r for r in range(ep_size) if len(local_to_global[r]) < e_local]
            rank = min(candidates, key=lambda r: (-traffic[r][expert_id], len(local_to_global[r]), r))
        local_to_global[rank].append(expert_id)
    return local_to_global


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--policy",
        required=True,
        choices=[
            "contiguous",
            "round_robin",
            "fixed_random_shuffle",
            "load_balanced_greedy_with_locality_tiebreak",
            "communication_aware_greedy",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="eval_results")
    args = parser.parse_args()

    with open(args.profile) as f:
        profile = json.load(f)
    ep_size = int(profile["ep_size"])
    num_experts = int(profile["num_experts"])
    if num_experts % ep_size != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by ep_size={ep_size}")
    layer_traffic = profile["traffic"]
    traffic = _aggregate_profile_traffic(profile)

    if args.policy in {"contiguous", "round_robin", "fixed_random_shuffle"}:
        local_to_global = _local_to_global_for_policy(
            num_experts, ep_size, args.policy, seed=args.seed,
        )
    elif args.policy == "load_balanced_greedy_with_locality_tiebreak":
        local_to_global = _balanced_greedy(traffic, ep_size, num_experts)
    else:
        local_to_global = _comm_greedy(traffic, ep_size, num_experts)

    ts = time.strftime("%Y%m%d_%H%M%S")
    path = Path(args.output_dir) / f"moe_placement_{args.policy}_{ts}.json"
    out = write_placement_json(
        path,
        policy=args.policy,
        num_experts=num_experts,
        ep_size=ep_size,
        local_to_global=local_to_global,
        seed=args.seed if args.policy == "fixed_random_shuffle" else None,
        profile_path=args.profile,
        traffic=traffic,
        layer_traffic=layer_traffic,
    )

    section = make_placement_section(local_to_global, num_experts)
    metrics = estimated_metrics(
        traffic,
        section["expert_to_rank"],
        ep_size,
        layer_traffic=layer_traffic,
    )
    print(json.dumps({"placement_path": out, "estimated_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
