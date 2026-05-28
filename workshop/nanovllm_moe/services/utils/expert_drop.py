"""Token-replica drop policies for owner_local_ep + EP-HT (Phase 4).

A "token-replica" is one (token, expert) routing branch — i.e., one of the top-K
expert assignments for a token. The drop unit is the replica, not the whole
expert: we may drop *some* tokens routed to expert e while keeping others.

Hard constraints honored by every policy:
  - never drop a replica whose target_rank == source_rank (preserves local-first).
  - every token must keep at least one replica. If a token's top-K is entirely
    remote, the highest-router-weight branch is protected from drop.

`drop_rate` is interpreted as a fraction of the total token-replicas (T*K),
but the actual drop budget is capped at the number of droppable (remote,
unprotected) replicas. The "effective" drop fraction is recorded in stats.

For per_expert_* policies the rate is interpreted per expert rather than as a
global budget — see comments inside `apply_drop`.
"""
from __future__ import annotations

import os
import random
from collections import Counter
from dataclasses import dataclass

import torch

from workshop.nanovllm_moe.services.utils.expert_replica_placement import same_numa


ALL_DROP_POLICIES = (
    "none",
    "tail_weight",
    "random",
    "cross_numa_first",
    "weighted_tail",          # Phase 4 v3: tail_weight + uniform noise
    "cross_numa_uniform",     # Phase 4 v3: cross-NUMA first, uniform within tier
    "per_expert_uniform",
    "hot_expert_relief",
    "per_expert_tailtoken",
    "hotspot_relief",
)

# Phase 4 P0/P2: GPU fast-path policies.
# - Score-based (P0): use `_select_drop_by_score` after computing per-replica score
# - Grouped quota (P2): use `_grouped_quota_drop` which enforces strict per-group quotas
GPU_SIMPLE_POLICIES = (
    "tail_weight",
    "random",
    "cross_numa_first",
    "weighted_tail",
    "cross_numa_uniform",
    "per_expert_uniform",     # Phase 4 P2: per-expert uniform drop_rate fraction
    "hot_expert_relief",      # Phase 4 P2: drop quota allocated to hot experts (excess load)
    "hotspot_relief",         # Phase 4 P2: drop quota allocated to hot dst ranks (excess load)
)


def _env_impl_default() -> str:
    return os.environ.get("MOE_DROP_IMPL", "auto").lower()


def _env_min_replicas_default() -> int:
    try:
        return int(os.environ.get("MOE_DROP_MIN_REPLICAS", "0"))
    except ValueError:
        return 0


def _env_gpu_stats_default() -> bool:
    return os.environ.get("MOE_DROP_GPU_STATS", "0") == "1"


@dataclass
class DropResult:
    keep_mask: torch.Tensor          # [T*K] bool, on device
    num_dropped: int
    num_dropped_remote: int
    num_dropped_cross_numa: int
    dropped_weight_mass: float
    total_replicas: int
    total_remote: int
    total_cross_numa: int
    total_weight_mass: float
    per_expert_total: dict[int, int]
    per_expert_dropped: dict[int, int]


def _empty_result(L: int, total_weight_mass: float, device: torch.device) -> DropResult:
    return DropResult(
        keep_mask=torch.ones(L, dtype=torch.bool, device=device),
        num_dropped=0,
        num_dropped_remote=0,
        num_dropped_cross_numa=0,
        dropped_weight_mass=0.0,
        total_replicas=L,
        total_remote=0,
        total_cross_numa=0,
        total_weight_mass=total_weight_mass,
        per_expert_total={},
        per_expert_dropped={},
    )


def apply_drop_cpu(
    *,
    flat_expert_ids: torch.Tensor,    # [T*K] int (on device)
    target_rank: torch.Tensor,        # [T*K] int (on device)
    flat_topk_w: torch.Tensor,        # [T*K] float
    source_rank: int,
    world_size: int,
    K: int,
    drop_policy: str,
    drop_rate: float,
    rng: random.Random,
) -> DropResult:
    if drop_policy not in ALL_DROP_POLICIES:
        raise ValueError(f"unknown drop policy {drop_policy!r}")
    device = flat_expert_ids.device
    L = int(flat_expert_ids.numel())
    total_weight_mass = float(flat_topk_w.sum().item()) if L else 0.0
    if L == 0 or drop_policy == "none" or drop_rate <= 0.0:
        return _empty_result(L, total_weight_mass, device)

    eid_list = flat_expert_ids.detach().to("cpu", torch.int64).tolist()
    tr_list = target_rank.detach().to("cpu", torch.int64).tolist()
    w_list = flat_topk_w.detach().to("cpu", torch.float32).tolist()
    T = L // K

    is_remote = [tr_list[i] != source_rank for i in range(L)]
    is_cross_numa = [
        is_remote[i] and not same_numa(int(tr_list[i]), int(source_rank), int(world_size))
        for i in range(L)
    ]
    total_remote = sum(is_remote)
    total_cross_numa = sum(is_cross_numa)

    # Per-token: track which positions in [T*K] belong to which token and which
    # of those are remote. A token whose top-K is fully remote needs its
    # highest-weight branch protected from drop.
    protected: set[int] = set()
    for t in range(T):
        positions = range(t * K, t * K + K)
        remote_positions = [p for p in positions if is_remote[p]]
        local_count = K - len(remote_positions)
        if local_count == 0 and remote_positions:
            protected.add(max(remote_positions, key=lambda p: (w_list[p], -p)))

    droppable: list[int] = [
        i for i in range(L) if is_remote[i] and i not in protected
    ]
    target_drop_global = int(round(drop_rate * L))
    target_drop_global = max(0, min(target_drop_global, len(droppable)))

    per_expert_total: dict[int, int] = dict(Counter(eid_list))

    if not droppable:
        result = _empty_result(L, total_weight_mass, device)
        result.total_remote = total_remote
        result.total_cross_numa = total_cross_numa
        result.per_expert_total = per_expert_total
        return result

    drop_positions: set[int] = set()

    if drop_policy == "tail_weight":
        sorted_idx = sorted(droppable, key=lambda i: (w_list[i], i))
        drop_positions = set(sorted_idx[:target_drop_global])

    elif drop_policy == "random":
        if target_drop_global > 0:
            sampled = rng.sample(droppable, target_drop_global)
            drop_positions = set(sampled)

    elif drop_policy == "cross_numa_first":
        cross = sorted(
            [i for i in droppable if is_cross_numa[i]],
            key=lambda i: (w_list[i], i),
        )
        same = sorted(
            [i for i in droppable if not is_cross_numa[i]],
            key=lambda i: (w_list[i], i),
        )
        take_cross = min(target_drop_global, len(cross))
        drop_positions.update(cross[:take_cross])
        remaining = target_drop_global - take_cross
        if remaining > 0:
            drop_positions.update(same[:remaining])

    elif drop_policy == "per_expert_uniform":
        # Per-expert quota: drop_rate fraction of each expert's droppable
        # (remote, unprotected) branches, in weight-ascending order. The
        # global drop count is implied, not enforced, so it may differ from
        # drop_rate * L when per-expert remote ratios vary.
        by_expert: dict[int, list[int]] = {}
        for i in droppable:
            by_expert.setdefault(eid_list[i], []).append(i)
        for positions in by_expert.values():
            quota = int(round(drop_rate * len(positions)))
            quota = max(0, min(quota, len(positions)))
            if quota:
                sorted_pos = sorted(positions, key=lambda i: (w_list[i], i))
                drop_positions.update(sorted_pos[:quota])

    elif drop_policy == "hot_expert_relief":
        # Quota proportional to how much each expert exceeds mean load.
        # Cold experts (load <= mean) are untouched.
        if per_expert_total:
            mean_load = sum(per_expert_total.values()) / len(per_expert_total)
            excess = {
                e: max(0.0, c - mean_load) for e, c in per_expert_total.items()
            }
            total_excess = sum(excess.values())
            by_expert: dict[int, list[int]] = {}
            for i in droppable:
                by_expert.setdefault(eid_list[i], []).append(i)
            if total_excess > 0:
                for eid, positions in by_expert.items():
                    share = excess.get(eid, 0.0) / total_excess
                    quota = int(round(target_drop_global * share))
                    quota = min(quota, len(positions))
                    if quota:
                        sorted_pos = sorted(positions, key=lambda i: (w_list[i], i))
                        drop_positions.update(sorted_pos[:quota])

    elif drop_policy == "per_expert_tailtoken":
        # For each expert, only the bottom-30% lowest-weight branches are
        # eligible; within that pool, drop drop_rate*total_branches_of_expert.
        by_expert: dict[int, list[int]] = {}
        for i in droppable:
            by_expert.setdefault(eid_list[i], []).append(i)
        for positions in by_expert.values():
            sorted_pos = sorted(positions, key=lambda i: (w_list[i], i))
            tail_size = max(1, int(0.3 * len(sorted_pos)))
            tail_pool = sorted_pos[:tail_size]
            quota = int(round(drop_rate * len(positions)))
            quota = min(quota, len(tail_pool))
            if quota:
                drop_positions.update(tail_pool[:quota])

    elif drop_policy == "hotspot_relief":
        # Per-destination-rank quota: dst ranks with above-mean incoming load
        # take a proportional share of the global budget. Within each dst,
        # pick lowest-weight droppable.
        dst_load = Counter(tr_list)
        if dst_load:
            mean_load = sum(dst_load.values()) / len(dst_load)
            excess = {d: max(0.0, c - mean_load) for d, c in dst_load.items()}
            total_excess = sum(excess.values())
            by_dst: dict[int, list[int]] = {}
            for i in droppable:
                by_dst.setdefault(tr_list[i], []).append(i)
            if total_excess > 0:
                for dst, positions in by_dst.items():
                    share = excess.get(dst, 0.0) / total_excess
                    quota = int(round(target_drop_global * share))
                    quota = min(quota, len(positions))
                    if quota:
                        sorted_pos = sorted(positions, key=lambda i: (w_list[i], i))
                        drop_positions.update(sorted_pos[:quota])
    else:
        raise ValueError(f"unknown drop policy {drop_policy!r}")

    # Guard: never drop protected or local (shouldn't happen since droppable
    # excluded them, but defensive).
    drop_positions -= protected
    drop_positions = {i for i in drop_positions if is_remote[i]}

    keep_mask_list = [True] * L
    per_expert_dropped: dict[int, int] = {}
    num_dropped_remote = 0
    num_dropped_cross_numa = 0
    dropped_weight_mass = 0.0
    for i in drop_positions:
        keep_mask_list[i] = False
        e = eid_list[i]
        per_expert_dropped[e] = per_expert_dropped.get(e, 0) + 1
        if is_remote[i]:
            num_dropped_remote += 1
        if is_cross_numa[i]:
            num_dropped_cross_numa += 1
        dropped_weight_mass += w_list[i]

    keep_mask = torch.tensor(keep_mask_list, dtype=torch.bool, device=device)
    return DropResult(
        keep_mask=keep_mask,
        num_dropped=len(drop_positions),
        num_dropped_remote=num_dropped_remote,
        num_dropped_cross_numa=num_dropped_cross_numa,
        dropped_weight_mass=float(dropped_weight_mass),
        total_replicas=L,
        total_remote=total_remote,
        total_cross_numa=total_cross_numa,
        total_weight_mass=total_weight_mass,
        per_expert_total=per_expert_total,
        per_expert_dropped=per_expert_dropped,
    )


# =============================================================================
# Phase 4 P0: GPU drop path (no host sync on the fast path)
# =============================================================================

def _empty_droppable_result(
    L: int, total_weight_mass: float, device: torch.device
) -> DropResult:
    return DropResult(
        keep_mask=torch.ones(L, dtype=torch.bool, device=device),
        num_dropped=0,
        num_dropped_remote=0,
        num_dropped_cross_numa=0,
        dropped_weight_mass=0.0,
        total_replicas=L,
        total_remote=0,
        total_cross_numa=0,
        total_weight_mass=total_weight_mass,
        per_expert_total={},
        per_expert_dropped={},
    )


def _select_drop_by_score(
    score: torch.Tensor,        # [L] float
    droppable: torch.Tensor,    # [L] bool
    budget: int,
) -> torch.Tensor:
    """Return a boolean keep_mask[L] (True = keep). At most `budget` positions
    flip to False; those positions are the lowest-`score` entries among
    `droppable`. Non-droppable positions are masked out with +inf so topk
    never picks them, and any +inf hits inside the topk results are filtered
    after the fact — no `.item()` / `.tolist()` calls needed.
    """
    L = score.numel()
    keep_mask = torch.ones(L, dtype=torch.bool, device=score.device)
    if budget <= 0 or L == 0:
        return keep_mask
    k = min(int(budget), L)
    masked = score.masked_fill(~droppable, float("inf"))
    vals, idx = torch.topk(masked, k=k, largest=False, sorted=False)
    # Defensive filter — if budget > droppable, some idx will have +inf score.
    finite = torch.isfinite(vals)
    idx = idx[finite]
    keep_mask[idx] = False
    return keep_mask


def _build_droppable_mask(
    target_rank: torch.Tensor,     # [L] int
    flat_topk_w: torch.Tensor,     # [L] float
    source_rank: int,
    T: int,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (is_remote, is_cross_numa-unset, droppable) — droppable means
    `is_remote AND not the per-token protected branch`. Cross-NUMA is handled
    separately by the cross_numa_first policy below."""
    is_remote = target_rank != int(source_rank)

    if K == 1 or T == 0:
        # K=1: dropping the only branch would zero the token. Never drop.
        # T=0: nothing to do.
        droppable = torch.zeros_like(is_remote)
        return is_remote, droppable, droppable

    weights_TK = flat_topk_w.view(T, K)
    remote_TK = is_remote.view(T, K)
    fully_remote_token = remote_TK.all(dim=1)            # [T]
    best_k = weights_TK.argmax(dim=1)                    # [T]
    arange_T = torch.arange(T, device=target_rank.device, dtype=torch.int64)
    protected_pos = arange_T * K + best_k.to(torch.int64)  # [T]

    protected = torch.zeros(T * K, dtype=torch.bool, device=target_rank.device)
    # Scatter True at positions for tokens that are fully remote.
    protected.scatter_(0, protected_pos, fully_remote_token)
    droppable = is_remote & ~protected
    return is_remote, protected, droppable


def _grouped_quota_drop(
    *,
    droppable: torch.Tensor,
    flat_topk_w: torch.Tensor,
    group_ids: torch.Tensor,
    num_groups: int,
    drop_rate: float,
    budget: int,
    quota_strategy: str,  # "uniform" or "excess"
) -> torch.Tensor:
    """Per-group quota drop with sorted-by-weight selection within group.

    Strategies:
      - "uniform": each group g loses round(drop_rate * count_g) of its droppable
                   replicas (the lowest-weight ones). per_expert_uniform uses this.
      - "excess":  groups with load above mean contribute drops proportional to
                   their excess. Cold groups contribute nothing. Total drops sum
                   to budget. hot_expert_relief / hotspot_relief use this.

    Pure GPU implementation, zero host sync. Returns keep_mask [L] bool.
    """
    device = droppable.device
    L = droppable.numel()
    group_ids_long = group_ids.long()

    # Step 1: count droppable replicas per group (scatter_add)
    counts = torch.zeros(num_groups + 1, device=device, dtype=torch.long)
    counts.scatter_add_(0, group_ids_long.clamp(0, num_groups), droppable.long())
    counts = counts[:num_groups]

    # Step 2: per-group drop quota
    if quota_strategy == "uniform":
        # Each group loses drop_rate fraction of its droppable replicas.
        quota = (counts.float() * drop_rate).round().long()
    elif quota_strategy == "excess":
        # Groups with excess load (count > mean) share `budget` proportional to excess.
        total_droppable = counts.sum().float()
        mean_load = total_droppable / max(num_groups, 1)
        excess = (counts.float() - mean_load).clamp(min=0)
        total_excess = excess.sum()
        # If perfectly balanced (no excess), no quota assigned → return identity.
        # Use float math to keep on device; division by 0 guarded.
        safe_excess = torch.where(total_excess > 0, total_excess, torch.ones_like(total_excess))
        quota = (excess / safe_excess * float(budget)).round().long()
        # If no excess, all quotas are 0 (mask will be all-keep).
    else:
        raise ValueError(f"unknown quota_strategy {quota_strategy!r}")

    # Step 3: sort by (group_id, weight ascending). Non-droppable pushed to end.
    # weight is softmax in [0,1], use BIG=2.0 to separate groups in sort order.
    BIG = 2.0
    sort_key = group_ids_long.float() * BIG + flat_topk_w.float()
    HUGE = float(num_groups + 2) * BIG
    sort_key = torch.where(droppable, sort_key, torch.full_like(sort_key, HUGE))
    sort_perm = torch.argsort(sort_key)

    # Step 4: rank within group (in sorted order). Use cummax-of-boundary trick.
    sorted_groups = group_ids_long[sort_perm]
    indices = torch.arange(L, device=device, dtype=torch.long)
    boundary = torch.cat([
        torch.ones(1, dtype=torch.bool, device=device),
        sorted_groups[1:] != sorted_groups[:-1],
    ])
    starts = torch.where(boundary, indices, torch.full_like(indices, -1))
    starts, _ = torch.cummax(starts, dim=0)  # propagate latest boundary index
    rank_in_group_sorted = indices - starts  # 0-indexed rank within group

    # Step 5: per-replica quota (in sorted order)
    sorted_quota = quota[sorted_groups]
    sorted_keep = rank_in_group_sorted >= sorted_quota  # keep if rank ≥ quota

    # Step 6: map back to original order
    keep_mask = torch.zeros(L, dtype=torch.bool, device=device)
    keep_mask[sort_perm] = sorted_keep
    # Non-droppable replicas (protected / local) are always kept.
    keep_mask = keep_mask | (~droppable)
    return keep_mask


def _drop_result_fast(
    keep_mask: torch.Tensor,
    L: int,
    device: torch.device,
) -> DropResult:
    """Build a DropResult without any host sync. Stats fields are zeroed; the
    only meaningful field is keep_mask. Use this on the performance path."""
    return DropResult(
        keep_mask=keep_mask,
        num_dropped=0,
        num_dropped_remote=0,
        num_dropped_cross_numa=0,
        dropped_weight_mass=0.0,
        total_replicas=L,
        total_remote=0,
        total_cross_numa=0,
        total_weight_mass=0.0,
        per_expert_total={},
        per_expert_dropped={},
    )


def _drop_result_with_stats(
    keep_mask: torch.Tensor,
    is_remote: torch.Tensor,
    is_cross_numa: torch.Tensor | None,
    flat_topk_w: torch.Tensor,
    flat_expert_ids: torch.Tensor,
) -> DropResult:
    """Build a DropResult with accurate per-layer stats. Triggers a handful of
    host syncs (.item() on reductions). Only invoke under the accounting/profile
    sweep, never on the performance path."""
    L = keep_mask.numel()
    dropped_mask = ~keep_mask
    dropped_and_remote = dropped_mask & is_remote
    total_remote = int(is_remote.sum().item())
    num_dropped = int(dropped_mask.sum().item())
    num_dropped_remote = int(dropped_and_remote.sum().item())
    if is_cross_numa is not None:
        num_dropped_cross_numa = int((dropped_mask & is_cross_numa).sum().item())
        total_cross_numa = int(is_cross_numa.sum().item())
    else:
        num_dropped_cross_numa = 0
        total_cross_numa = 0
    total_weight_mass = float(flat_topk_w.sum().item())
    dropped_weight_mass = float((flat_topk_w * dropped_mask.to(flat_topk_w.dtype)).sum().item())
    # per_expert counters — best-effort, may be empty when L=0.
    if L > 0:
        eids = flat_expert_ids.detach().to("cpu", torch.int64).tolist()
        drop_list = dropped_mask.detach().to("cpu").tolist()
        per_expert_total: dict[int, int] = {}
        per_expert_dropped: dict[int, int] = {}
        for e, dropped in zip(eids, drop_list):
            per_expert_total[e] = per_expert_total.get(e, 0) + 1
            if dropped:
                per_expert_dropped[e] = per_expert_dropped.get(e, 0) + 1
    else:
        per_expert_total = {}
        per_expert_dropped = {}
    return DropResult(
        keep_mask=keep_mask,
        num_dropped=num_dropped,
        num_dropped_remote=num_dropped_remote,
        num_dropped_cross_numa=num_dropped_cross_numa,
        dropped_weight_mass=dropped_weight_mass,
        total_replicas=L,
        total_remote=total_remote,
        total_cross_numa=total_cross_numa,
        total_weight_mass=total_weight_mass,
        per_expert_total=per_expert_total,
        per_expert_dropped=per_expert_dropped,
    )


def apply_drop_gpu_simple(
    *,
    flat_expert_ids: torch.Tensor,
    target_rank: torch.Tensor,
    flat_topk_w: torch.Tensor,
    source_rank: int,
    world_size: int,
    K: int,
    drop_policy: str,
    drop_rate: float,
    torch_generator: torch.Generator | None = None,
    collect_stats: bool = False,
    num_experts_global: int | None = None,
) -> DropResult:
    """Pure-GPU implementation of the three simple drop policies.

    Fast path (collect_stats=False): zero host sync between the device tensors
    and the returned keep_mask. Stats fields in DropResult are zeroed.
    """
    if drop_policy not in GPU_SIMPLE_POLICIES:
        raise ValueError(f"GPU drop only supports {GPU_SIMPLE_POLICIES}, got {drop_policy!r}")
    L = int(flat_expert_ids.numel())
    device = flat_expert_ids.device
    if L == 0 or drop_rate <= 0.0:
        return _empty_droppable_result(L, 0.0, device)
    T = L // K
    budget = int(round(drop_rate * L))
    if budget <= 0:
        keep_mask = torch.ones(L, dtype=torch.bool, device=device)
        return _drop_result_fast(keep_mask, L, device)

    is_remote, _protected, droppable = _build_droppable_mask(
        target_rank, flat_topk_w, source_rank, T, K
    )

    is_cross_numa: torch.Tensor | None = None
    if drop_policy == "tail_weight":
        score = flat_topk_w.float()
    elif drop_policy == "random":
        score = torch.rand(
            (L,), device=device, dtype=torch.float32, generator=torch_generator
        )
    elif drop_policy == "cross_numa_first":
        # Two-tier ordering: cross-NUMA remote replicas first (by ascending
        # weight), then same-NUMA remote replicas (by ascending weight).
        split = max(1, world_size // 2)
        src_numa = min(source_rank // split, 1)
        dst_numa = torch.div(target_rank, split, rounding_mode="floor").clamp_max(1)
        is_cross_numa = is_remote & (dst_numa != src_numa)
        # Same-NUMA remote replicas get a large positive bias so they're sorted
        # after every cross-NUMA replica. Use 2.0 (router weights are softmax
        # outputs in [0, 1]) to guarantee separation without numerical risk.
        BIAS = 2.0
        score = flat_topk_w.float() + (~is_cross_numa).to(torch.float32) * BIAS
    elif drop_policy == "weighted_tail":
        # Tail-by-weight, perturbed by uniform noise on the order of the typical
        # router weight spacing (~1/K). Softer than tail_weight: low-weight
        # branches still preferentially drop, but some chance to spare any one
        # of them. Hypothesis: better accuracy than tail_weight at same rate.
        noise = torch.rand(
            (L,), device=device, dtype=torch.float32, generator=torch_generator
        )
        noise_scale = 1.0 / max(K, 1)
        score = flat_topk_w.float() + noise * noise_scale
    elif drop_policy == "cross_numa_uniform":
        # Two-tier like cross_numa_first, but within each tier the order is
        # uniform random (not weight-ranked). Hypothesis: if a2a bytes are what
        # matters, random selection within cross-NUMA preserves accuracy as
        # well as weight-ranked.
        split = max(1, world_size // 2)
        src_numa = min(source_rank // split, 1)
        dst_numa = torch.div(target_rank, split, rounding_mode="floor").clamp_max(1)
        is_cross_numa = is_remote & (dst_numa != src_numa)
        noise = torch.rand(
            (L,), device=device, dtype=torch.float32, generator=torch_generator
        )
        BIAS = 2.0
        score = noise + (~is_cross_numa).to(torch.float32) * BIAS
    elif drop_policy in ("per_expert_uniform", "hot_expert_relief", "hotspot_relief"):
        # Grouped-quota policies: bypass the global score-based selector and
        # use per-group quotas via _grouped_quota_drop.
        if drop_policy in ("per_expert_uniform", "hot_expert_relief"):
            if num_experts_global is None:
                raise ValueError(
                    f"{drop_policy!r} requires `num_experts_global` arg (caller "
                    "must pass dispatch.E_global to keep the GPU fast path zero-sync)"
                )
            strategy = "uniform" if drop_policy == "per_expert_uniform" else "excess"
            keep_mask = _grouped_quota_drop(
                droppable=droppable, flat_topk_w=flat_topk_w,
                group_ids=flat_expert_ids, num_groups=int(num_experts_global),
                drop_rate=drop_rate, budget=budget,
                quota_strategy=strategy,
            )
        else:  # hotspot_relief
            # Drop quota allocated to hot dst ranks (load > mean); within hot
            # ranks, take lowest-weight replicas. world_size known statically.
            keep_mask = _grouped_quota_drop(
                droppable=droppable, flat_topk_w=flat_topk_w,
                group_ids=target_rank, num_groups=world_size,
                drop_rate=drop_rate, budget=budget,
                quota_strategy="excess",
            )
        if not collect_stats:
            return _drop_result_fast(keep_mask, L, device)
        return _drop_result_with_stats(
            keep_mask=keep_mask,
            is_remote=is_remote,
            is_cross_numa=None,
            flat_topk_w=flat_topk_w,
            flat_expert_ids=flat_expert_ids,
        )
    else:
        raise ValueError(f"unreachable drop_policy {drop_policy!r}")

    keep_mask = _select_drop_by_score(score, droppable, budget)

    if not collect_stats:
        return _drop_result_fast(keep_mask, L, device)
    return _drop_result_with_stats(
        keep_mask=keep_mask,
        is_remote=is_remote,
        is_cross_numa=is_cross_numa,
        flat_topk_w=flat_topk_w,
        flat_expert_ids=flat_expert_ids,
    )


# =============================================================================
# Top-level dispatcher
# =============================================================================

def apply_drop(
    *,
    flat_expert_ids: torch.Tensor,
    target_rank: torch.Tensor,
    flat_topk_w: torch.Tensor,
    source_rank: int,
    world_size: int,
    K: int,
    drop_policy: str,
    drop_rate: float,
    rng: random.Random,
    impl: str | None = None,
    min_replicas: int | None = None,
    torch_generator: torch.Generator | None = None,
    collect_stats: bool | None = None,
    num_experts_global: int | None = None,
) -> DropResult:
    """Routes to the GPU or CPU implementation depending on policy + impl.

    impl semantics:
      - "gpu": force GPU path. Errors for grouped policies.
      - "cpu": force the original CPU implementation.
      - "auto" (default): GPU for simple policies, CPU for grouped ones.
    `min_replicas`: skip drop entirely when flat_expert_ids.numel() <= this.
    `collect_stats`: only meaningful for the GPU path; defaults to env var
    MOE_DROP_GPU_STATS or False.
    """
    if drop_policy == "none" or drop_rate <= 0.0:
        L = int(flat_expert_ids.numel())
        return _empty_droppable_result(L, 0.0, flat_expert_ids.device)

    impl = (impl or _env_impl_default()).lower()
    if impl not in ("auto", "gpu", "cpu"):
        raise ValueError(f"unknown drop impl {impl!r}")
    if min_replicas is None:
        min_replicas = _env_min_replicas_default()
    if collect_stats is None:
        collect_stats = _env_gpu_stats_default()

    L = int(flat_expert_ids.numel())
    if min_replicas > 0 and L <= int(min_replicas):
        return _empty_droppable_result(L, 0.0, flat_expert_ids.device)

    use_gpu = (
        impl == "gpu"
        or (impl == "auto" and drop_policy in GPU_SIMPLE_POLICIES)
    )
    if use_gpu and drop_policy not in GPU_SIMPLE_POLICIES:
        raise ValueError(
            f"impl=gpu requested for {drop_policy!r}; GPU path covers only "
            f"{GPU_SIMPLE_POLICIES}. Use impl=cpu or impl=auto."
        )
    if use_gpu:
        return apply_drop_gpu_simple(
            flat_expert_ids=flat_expert_ids,
            target_rank=target_rank,
            flat_topk_w=flat_topk_w,
            source_rank=source_rank,
            world_size=world_size,
            K=K,
            drop_policy=drop_policy,
            drop_rate=drop_rate,
            torch_generator=torch_generator,
            collect_stats=bool(collect_stats),
            num_experts_global=num_experts_global,
        )
    return apply_drop_cpu(
        flat_expert_ids=flat_expert_ids,
        target_rank=target_rank,
        flat_topk_w=flat_topk_w,
        source_rank=source_rank,
        world_size=world_size,
        K=K,
        drop_policy=drop_policy,
        drop_rate=drop_rate,
        rng=rng,
    )
