from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from workshop.nanovllm_moe.services.utils.parallel import get_ep_group


_STATE: dict[str, Any] = {}


def _group():
    return get_ep_group() if dist.is_initialized() else None


def _all_reduce_in_place(tensor: torch.Tensor, op: dist.ReduceOp) -> None:
    if not dist.is_initialized():
        return
    dist.all_reduce(tensor, op=op, group=_group())


def reset() -> None:
    _STATE.clear()


def _init_counter(name: str, device: torch.device) -> torch.Tensor:
    counter = _STATE.get(name)
    if counter is None:
        counter = torch.zeros((), dtype=torch.int64, device=device)
        _STATE[name] = counter
    return counter


def _init_layer_bucket(layer_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    per_layer = _STATE.setdefault("per_layer", {})
    bucket = per_layer.get(int(layer_id))
    if bucket is None:
        zero_counts = None
        bucket = {
            "calls": torch.zeros((), dtype=torch.int64, device=device),
            "routed_replicas": torch.zeros((), dtype=torch.int64, device=device),
            "overflowed_replicas": torch.zeros((), dtype=torch.int64, device=device),
            "overflowed_buckets": torch.zeros((), dtype=torch.int64, device=device),
            "observed_bucket_max": torch.zeros((), dtype=torch.int64, device=device),
            "bucket_sum_matrix": zero_counts,
            "bucket_max_matrix": zero_counts,
        }
        per_layer[int(layer_id)] = bucket
    return bucket


def set_static_metadata(**kwargs: Any) -> None:
    meta = _STATE.setdefault("meta", {})
    meta.update({k: v for k, v in kwargs.items() if v is not None})


def record_local_counts(
    *,
    layer_id: int,
    local_counts: torch.Tensor,
    m_max: int,
) -> None:
    if local_counts.numel() == 0:
        return
    device = local_counts.device
    local_counts_i64 = local_counts.to(torch.int64)
    excess = torch.clamp(local_counts_i64 - int(m_max), min=0)
    routed = local_counts_i64.sum()
    overflowed_replicas = excess.sum()
    overflowed_buckets = (local_counts_i64 > int(m_max)).sum().to(torch.int64)
    observed_bucket_max = local_counts_i64.max()

    _init_counter("calls", device).add_(1)
    _init_counter("routed_replicas", device).add_(routed)
    _init_counter("overflowed_replicas", device).add_(overflowed_replicas)
    _init_counter("overflowed_buckets", device).add_(overflowed_buckets)
    _init_counter("observed_bucket_max", device).copy_(
        torch.maximum(_STATE["observed_bucket_max"], observed_bucket_max)
    )

    bucket = _init_layer_bucket(layer_id, device)
    bucket["calls"].add_(1)
    bucket["routed_replicas"].add_(routed)
    bucket["overflowed_replicas"].add_(overflowed_replicas)
    bucket["overflowed_buckets"].add_(overflowed_buckets)
    bucket["observed_bucket_max"].copy_(
        torch.maximum(bucket["observed_bucket_max"], observed_bucket_max)
    )
    if bucket["bucket_sum_matrix"] is None:
        bucket["bucket_sum_matrix"] = local_counts_i64.clone()
        bucket["bucket_max_matrix"] = local_counts_i64.clone()
    else:
        bucket["bucket_sum_matrix"].add_(local_counts_i64)
        bucket["bucket_max_matrix"].copy_(
            torch.maximum(bucket["bucket_max_matrix"], local_counts_i64)
        )


def summarize() -> dict[str, Any]:
    calls = _STATE.get("calls")
    if calls is None:
        return {}

    calls = calls.clone()
    routed_replicas = _STATE["routed_replicas"].clone()
    overflowed_replicas = _STATE["overflowed_replicas"].clone()
    overflowed_buckets = _STATE["overflowed_buckets"].clone()
    observed_bucket_max = _STATE["observed_bucket_max"].clone()

    _all_reduce_in_place(calls, dist.ReduceOp.SUM)
    _all_reduce_in_place(routed_replicas, dist.ReduceOp.SUM)
    _all_reduce_in_place(overflowed_replicas, dist.ReduceOp.SUM)
    _all_reduce_in_place(overflowed_buckets, dist.ReduceOp.SUM)
    _all_reduce_in_place(observed_bucket_max, dist.ReduceOp.MAX)

    per_layer_summary = []
    for layer_id in sorted((_STATE.get("per_layer") or {}).keys()):
        bucket = _STATE["per_layer"][layer_id]
        layer_calls = bucket["calls"].clone()
        layer_routed = bucket["routed_replicas"].clone()
        layer_overflow_replicas = bucket["overflowed_replicas"].clone()
        layer_overflow_buckets = bucket["overflowed_buckets"].clone()
        layer_bucket_max = bucket["observed_bucket_max"].clone()
        _all_reduce_in_place(layer_calls, dist.ReduceOp.SUM)
        _all_reduce_in_place(layer_routed, dist.ReduceOp.SUM)
        _all_reduce_in_place(layer_overflow_replicas, dist.ReduceOp.SUM)
        _all_reduce_in_place(layer_overflow_buckets, dist.ReduceOp.SUM)
        _all_reduce_in_place(layer_bucket_max, dist.ReduceOp.MAX)
        routed_val = int(layer_routed.item())
        overflow_val = int(layer_overflow_replicas.item())
        per_layer_summary.append(
            {
                "layer_id": int(layer_id),
                "calls": int(layer_calls.item()),
                "routed_replicas": routed_val,
                "overflowed_replicas": overflow_val,
                "overflowed_buckets": int(layer_overflow_buckets.item()),
                "observed_bucket_max": int(layer_bucket_max.item()),
                "overflow_replica_fraction": (
                    float(overflow_val / routed_val) if routed_val > 0 else 0.0
                ),
                "bucket_sum_matrix": (
                    bucket["bucket_sum_matrix"].cpu().tolist()
                    if bucket["bucket_sum_matrix"] is not None else None
                ),
                "bucket_max_matrix": (
                    bucket["bucket_max_matrix"].cpu().tolist()
                    if bucket["bucket_max_matrix"] is not None else None
                ),
            }
        )

    meta = dict(_STATE.get("meta") or {})
    routed_val = int(routed_replicas.item())
    overflow_val = int(overflowed_replicas.item())
    return {
        **meta,
        "calls": int(calls.item()),
        "routed_replicas": routed_val,
        "overflowed_replicas": overflow_val,
        "overflowed_buckets": int(overflowed_buckets.item()),
        "observed_bucket_max": int(observed_bucket_max.item()),
        "overflow_replica_fraction": float(overflow_val / routed_val) if routed_val > 0 else 0.0,
        "per_layer": per_layer_summary,
    }


def summarize_local() -> dict[str, Any]:
    calls = _STATE.get("calls")
    if calls is None:
        return {}

    per_layer_summary = []
    for layer_id in sorted((_STATE.get("per_layer") or {}).keys()):
        bucket = _STATE["per_layer"][layer_id]
        routed_val = int(bucket["routed_replicas"].item())
        overflow_val = int(bucket["overflowed_replicas"].item())
        per_layer_summary.append(
            {
                "layer_id": int(layer_id),
                "calls": int(bucket["calls"].item()),
                "routed_replicas": routed_val,
                "overflowed_replicas": overflow_val,
                "overflowed_buckets": int(bucket["overflowed_buckets"].item()),
                "observed_bucket_max": int(bucket["observed_bucket_max"].item()),
                "overflow_replica_fraction": (
                    float(overflow_val / routed_val) if routed_val > 0 else 0.0
                ),
            }
        )

    meta = dict(_STATE.get("meta") or {})
    routed_val = int(_STATE["routed_replicas"].item())
    overflow_val = int(_STATE["overflowed_replicas"].item())
    return {
        **meta,
        "calls": int(_STATE["calls"].item()),
        "routed_replicas": routed_val,
        "overflowed_replicas": overflow_val,
        "overflowed_buckets": int(_STATE["overflowed_buckets"].item()),
        "observed_bucket_max": int(_STATE["observed_bucket_max"].item()),
        "overflow_replica_fraction": float(overflow_val / routed_val) if routed_val > 0 else 0.0,
        "per_layer": per_layer_summary,
    }
