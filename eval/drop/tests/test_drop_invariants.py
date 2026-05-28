"""Unit tests for two drop-kernel invariants used by the Tier 1 bench.

Run:
    /home/lzy/miniconda3/envs/vllm/bin/python -m eval.drop.tests.test_drop_invariants

Requires a single CUDA device (no torchrun needed).
"""
from __future__ import annotations

import sys

import torch

from workshop.nanovllm_moe.services.utils.expert_drop import apply_drop_gpu_simple


def _make_routing(T: int, K: int, world: int, source: int, seed: int = 0):
    """Build (flat_expert_ids, target_rank, flat_topk_w) on CPU then move to CUDA."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    eid = torch.randint(0, world * 16, (T * K,), generator=g, dtype=torch.int32)
    tr = (eid.long() % world).to(torch.int64)
    w = torch.softmax(torch.randn(T, K, generator=g), dim=-1).view(-1).contiguous()
    return eid, tr, w


def test_drop_rate_zero_keeps_all() -> None:
    """Invariant: drop_rate=0 must keep every replica (no token loses any expert)."""
    print("--- test_drop_rate_zero_keeps_all ---")
    device = torch.device("cuda:0")
    T, K, world = 64, 8, 8
    source = 0
    eid, tr, w = _make_routing(T, K, world, source, seed=42)
    eid, tr, w = eid.to(device), tr.to(device), w.to(device)
    gen = torch.Generator(device=device).manual_seed(7)
    res = apply_drop_gpu_simple(
        flat_expert_ids=eid,
        target_rank=tr,
        flat_topk_w=w,
        source_rank=source,
        world_size=world,
        K=K,
        drop_policy="tail_weight",
        drop_rate=0.0,
        torch_generator=gen,
        collect_stats=False,
    )
    kept = int(res.keep_mask.sum().item())
    assert kept == T * K, (
        f"drop_rate=0 must keep all replicas; got kept={kept} out of {T*K}"
    )
    print(f"  OK: kept={kept}/{T*K}")


def test_effective_drop_frac_matches_nominal() -> None:
    """Invariant: effective_drop_frac ≈ drop_rate (within 5% absolute) for large T
    when all routing is remote (so local-preserve constraint doesn't suppress drops).
    """
    print("--- test_effective_drop_frac_matches_nominal ---")
    device = torch.device("cuda:0")
    T, K, world = 4096, 8, 8
    source = 0
    g = torch.Generator(device="cpu").manual_seed(123)
    eid = torch.randint(0, world * 16, (T * K,), generator=g, dtype=torch.int32)
    # Force every target_rank != source: pick from {1..world-1}, then add source mod world.
    base_tr = torch.randint(1, world, (T * K,), generator=g, dtype=torch.int64)
    tr = (base_tr + source) % world
    w = torch.softmax(torch.randn(T, K, generator=g), dim=-1).view(-1).contiguous()
    eid, tr, w = eid.to(device), tr.to(device), w.to(device)

    for rate in (0.1, 0.3, 0.5):
        gen = torch.Generator(device=device).manual_seed(7)
        res = apply_drop_gpu_simple(
            flat_expert_ids=eid,
            target_rank=tr,
            flat_topk_w=w,
            source_rank=source,
            world_size=world,
            K=K,
            drop_policy="tail_weight",
            drop_rate=rate,
            torch_generator=gen,
            collect_stats=False,
        )
        kept = int(res.keep_mask.sum().item())
        effective_dropped = 1.0 - kept / (T * K)
        # Per-token min-keep=1 and per-token monotone tail_weight can deviate from rate;
        # 5% absolute tolerance is comfortable for these constraints at large T.
        err = abs(effective_dropped - rate)
        assert err < 0.05, (
            f"rate={rate}: effective_dropped={effective_dropped:.4f} "
            f"(err={err:.4f}, expected < 0.05)"
        )
        print(f"  OK rate={rate} effective={effective_dropped:.4f}")


def test_new_policies_register_and_run() -> None:
    """v3 sanity: weighted_tail + cross_numa_uniform run without error
    and produce keep_mask matching the rate."""
    print("--- test_new_policies_register_and_run ---")
    device = torch.device("cuda:0")
    T, K, world = 4096, 8, 8
    source = 0
    g = torch.Generator(device="cpu").manual_seed(321)
    eid = torch.randint(0, world * 16, (T * K,), generator=g, dtype=torch.int32)
    base_tr = torch.randint(1, world, (T * K,), generator=g, dtype=torch.int64)
    tr = (base_tr + source) % world
    w = torch.softmax(torch.randn(T, K, generator=g), dim=-1).view(-1).contiguous()
    eid, tr, w = eid.to(device), tr.to(device), w.to(device)

    for pol in ("weighted_tail", "cross_numa_uniform"):
        gen = torch.Generator(device=device).manual_seed(7)
        res = apply_drop_gpu_simple(
            flat_expert_ids=eid, target_rank=tr, flat_topk_w=w,
            source_rank=source, world_size=world, K=K,
            drop_policy=pol, drop_rate=0.3,
            torch_generator=gen, collect_stats=False,
        )
        kept = int(res.keep_mask.sum().item())
        eff = 1.0 - kept / (T * K)
        assert abs(eff - 0.3) < 0.05, f"{pol}: effective={eff:.3f}"
        print(f"  OK {pol}  effective={eff:.4f}")


def test_grouped_quota_policies() -> None:
    """v7 sanity: per_expert_uniform / hot_expert_relief / hotspot_relief run
    without error and produce keep_mask with effective drop ≈ drop_rate.

    These are GPU implementations of the originally-CPU grouped-quota policies.
    """
    print("--- test_grouped_quota_policies ---")
    device = torch.device("cuda:0")
    T, K, world = 4096, 8, 8
    source = 0
    E_global = world * 16  # = 128, like Qwen3
    g = torch.Generator(device="cpu").manual_seed(456)
    eid = torch.randint(0, E_global, (T * K,), generator=g, dtype=torch.int32)
    base_tr = torch.randint(1, world, (T * K,), generator=g, dtype=torch.int64)
    tr = (base_tr + source) % world
    w = torch.softmax(torch.randn(T, K, generator=g), dim=-1).view(-1).contiguous()
    eid, tr, w = eid.to(device), tr.to(device), w.to(device)

    # per_expert_uniform: each expert loses drop_rate fraction. Tolerance higher
    # because per-expert rounding accumulates and routing may have hot experts.
    for pol in ("per_expert_uniform", "hot_expert_relief", "hotspot_relief"):
        for rate in (0.1, 0.3, 0.5):
            gen = torch.Generator(device=device).manual_seed(7)
            res = apply_drop_gpu_simple(
                flat_expert_ids=eid, target_rank=tr, flat_topk_w=w,
                source_rank=source, world_size=world, K=K,
                drop_policy=pol, drop_rate=rate,
                torch_generator=gen, collect_stats=False,
                num_experts_global=E_global,
            )
            kept = int(res.keep_mask.sum().item())
            eff = 1.0 - kept / (T * K)
            # Tolerance:
            # - per_expert_uniform: bounded ±15% by per-token min-keep + per-expert rounding.
            # - excess-strategy (hot_expert_relief / hotspot_relief) is *inherently*
            #   load-dependent: only over-mean groups contribute. On nearly-balanced
            #   routing, effective drop < nominal rate (by design). Lower bound is
            #   relaxed accordingly.
            assert eff <= rate + 0.05, f"{pol}@{rate}: over-dropped ({eff:.3f})"
            if pol == "per_expert_uniform":
                assert eff > max(0, rate - 0.15), f"{pol}@{rate}: under-dropped {eff:.3f}"
            else:
                # excess-strategy: at least 30% of nominal rate (sanity).
                assert eff > rate * 0.30, f"{pol}@{rate}: under-dropped {eff:.3f}"
            print(f"  OK {pol:<22} rate={rate} effective={eff:.4f}")


def main() -> int:
    if not torch.cuda.is_available():
        print("[skip] CUDA not available")
        return 0
    test_drop_rate_zero_keeps_all()
    test_effective_drop_frac_matches_nominal()
    test_new_policies_register_and_run()
    test_grouped_quota_policies()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
