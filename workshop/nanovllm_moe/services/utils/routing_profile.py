"""Low-overhead MoE routing profile accumulator.

Enabled only when MOE_PROFILE_ROUTING=1. Each rank writes a rank-local JSON,
then rank 0 combines those files into eval_results/moe_routing_profile_<id>.json.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from workshop.nanovllm_moe.services.utils.parallel import (
    get_dp_rank,
    get_dp_world_size,
    get_ep_rank,
    get_ep_world_size,
    get_runtime_mode,
    is_dp_leader,
    is_owner_local_ep_mode,
)
from workshop.nanovllm_moe.services.utils.routing_profile_quality import compute_profile_quality

_TRAFFIC: dict[int, list[list[int]]] = {}
_META: dict[str, Any] = {}


def enabled() -> bool:
    return os.environ.get("MOE_PROFILE_ROUTING", "0") == "1"


def set_metadata(**kwargs) -> None:
    _META.update({k: v for k, v in kwargs.items() if v is not None})


def record_routing(layer_id: int, topk_ids: torch.Tensor, num_experts: int) -> None:
    if not enabled():
        return
    runtime_mode = get_runtime_mode()
    if runtime_mode == "vllm_dp_ep" and not is_dp_leader():
        return
    owner_local = runtime_mode in ("vllm_dp_ep", "owner_local_ep")
    src_size = get_dp_world_size() if owner_local else (get_ep_world_size() if dist.is_initialized() else 1)
    src_rank = get_dp_rank() if owner_local else (get_ep_rank() if dist.is_initialized() else 0)
    layer_id = int(layer_id)
    num_experts = int(num_experts)

    counts = torch.bincount(topk_ids.reshape(-1).to(torch.int64), minlength=num_experts)
    counts_cpu = [int(x) for x in counts[:num_experts].cpu().tolist()]
    traffic = _TRAFFIC.setdefault(
        layer_id, [[0 for _ in range(num_experts)] for _ in range(src_size)]
    )
    row = traffic[src_rank]
    for expert_id, count in enumerate(counts_cpu):
        row[expert_id] += count


def _rank_payload() -> dict:
    runtime_mode = get_runtime_mode()
    if _TRAFFIC:
        num_layers = max(_TRAFFIC) + 1
        src_size = len(next(iter(_TRAFFIC.values())))
        num_experts = len(next(iter(_TRAFFIC.values()))[0])
    else:
        num_layers = int(_META.get("num_layers", 0))
        src_size = int(_META.get("dp_size" if runtime_mode in ("vllm_dp_ep", "owner_local_ep") else "ep_size", 1))
        num_experts = int(_META.get("num_experts", 0))
    traffic = [
        _TRAFFIC.get(layer, [[0 for _ in range(num_experts)] for _ in range(src_size)])
        for layer in range(num_layers)
    ]
    return {
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        "ep_rank": get_ep_rank() if dist.is_initialized() else 0,
        "dp_rank": get_dp_rank() if dist.is_initialized() else 0,
        "is_dp_leader": is_dp_leader(),
        "metadata": dict(_META),
        "traffic": traffic,
    }


def save_profile(output_dir: str = "eval_results") -> str | None:
    if not enabled():
        return None
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("MOE_PROFILE_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank_path = Path(output_dir) / f"moe_routing_profile_{run_id}_rank{rank}.json"
    with open(rank_path, "w") as f:
        json.dump(_rank_payload(), f, indent=2)

    if dist.is_initialized():
        dist.barrier()
    if rank != 0:
        return None

    rank_payloads = []
    for r in range(world_size):
        with open(Path(output_dir) / f"moe_routing_profile_{run_id}_rank{r}.json") as f:
            rank_payloads.append(json.load(f))

    meta = dict(_META)
    if rank_payloads:
        meta.update(rank_payloads[0].get("metadata", {}))
    num_layers = max((len(p["traffic"]) for p in rank_payloads), default=0)
    runtime_mode = meta.get("runtime_mode", get_runtime_mode())
    src_size = int(meta.get("dp_size" if runtime_mode in ("vllm_dp_ep", "owner_local_ep") else "ep_size", 1))
    num_experts = int(meta.get("num_experts", 0))
    if num_experts == 0 and rank_payloads and rank_payloads[0]["traffic"]:
        num_experts = len(rank_payloads[0]["traffic"][0][0])

    traffic = [
        [[0 for _ in range(num_experts)] for _ in range(src_size)]
        for _ in range(num_layers)
    ]
    for payload in rank_payloads:
        for layer, matrix in enumerate(payload["traffic"]):
            for src, row in enumerate(matrix):
                for expert_id, count in enumerate(row):
                    traffic[layer][src][expert_id] += int(count)

    expert_load = [
        [sum(traffic[layer][src][e] for src in range(src_size)) for e in range(num_experts)]
        for layer in range(num_layers)
    ]
    per_rank_source_tokens = [
        sum(sum(traffic[layer][src]) for layer in range(num_layers))
        for src in range(src_size)
    ]
    total_routed = sum(per_rank_source_tokens)
    out_path = Path(output_dir) / f"moe_routing_profile_{run_id}.json"
    payload = {
        **meta,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile_run_id": run_id,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "ep_size": int(meta.get("ep_size", 1)),
        "dp_size": int(meta.get("dp_size", 1)),
        "source_definition": meta.get(
            "source_definition",
            "dp_leader" if runtime_mode == "vllm_dp_ep"
            else "owner_rank" if runtime_mode == "owner_local_ep"
            else "world_rank",
        ),
        "traffic": traffic,
        "expert_load": expert_load,
        "total_routed_replicas": total_routed,
        "per_rank_source_tokens": per_rank_source_tokens,
        "prefill_decode_split": "combined",
        "quality_summary": compute_profile_quality(traffic),
        "rank_profile_paths": [str(Path(output_dir) / f"moe_routing_profile_{run_id}_rank{r}.json") for r in range(world_size)],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[moe-profile] wrote {out_path}", flush=True)
    return str(out_path)


def reset() -> None:
    _TRAFFIC.clear()
    _META.clear()
