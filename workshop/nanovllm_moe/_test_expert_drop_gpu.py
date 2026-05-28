"""Unit tests for Phase 4 GPU drop path (P0).

Run:
    /home/lzy/miniconda3/envs/vllm/bin/python -m workshop.nanovllm_moe._test_expert_drop_gpu

Covers:
- per-token "keep at least 1" constraint.
- never-drop-local constraint.
- exact CPU/GPU parity for `tail_weight` and `cross_numa_first` (deterministic).
- `random`: CPU/GPU don't share RNG — only check constraints and reproducibility.
- `min_replicas` (small batch bypass) behavior.
- per-rank cross-NUMA grouping.
"""
from __future__ import annotations

import os
import random
import sys
import traceback

import torch

# Make sure env doesn't bias the dispatcher decisions during tests.
os.environ.pop("MOE_DROP_IMPL", None)
os.environ.pop("MOE_DROP_MIN_REPLICAS", None)
os.environ.pop("MOE_DROP_GPU_STATS", None)

from workshop.nanovllm_moe.services.utils.expert_drop import (  # noqa: E402
    apply_drop,
    apply_drop_cpu,
    apply_drop_gpu_simple,
)


def _make_routing(T: int, K: int, N: int, source: int, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    eid = torch.randint(0, 4 * N, (T * K,), generator=g, dtype=torch.int32)
    tr = (eid.long() % N).to(torch.int64)
    w = torch.softmax(torch.randn(T, K, generator=g), dim=-1).view(-1).contiguous()
    return eid, tr, w


def _check_constraints(keep_mask, target_rank, source_rank, T, K, label):
    L = keep_mask.numel()
    keep_list = keep_mask.tolist()
    tr_list = target_rank.tolist()
    # Local never dropped
    for i in range(L):
        if tr_list[i] == source_rank:
            assert keep_list[i], f"{label}: dropped local position {i}"
    # Every token keeps at least one
    for t in range(T):
        kept = sum(1 for k in range(K) if keep_list[t * K + k])
        assert kept >= 1, f"{label}: token {t} has 0 kept replicas"


def _make_device():
    if not torch.cuda.is_available():
        print("[skip] CUDA not available; GPU parity tests rely on it.")
        return None
    return torch.device("cuda:0")


def test_simple_constraints():
    print("--- test_simple_constraints ---")
    device = _make_device()
    if device is None:
        return
    T, K, N = 64, 8, 8
    source = 0
    eid, tr, w = _make_routing(T, K, N, source, seed=1)
    eid, tr, w = eid.to(device), tr.to(device), w.to(device)
    for pol in ("tail_weight", "random", "cross_numa_first"):
        for rate in (0.05, 0.10, 0.20):
            gen = torch.Generator(device=device).manual_seed(123)
            res = apply_drop_gpu_simple(
                flat_expert_ids=eid, target_rank=tr, flat_topk_w=w,
                source_rank=source, world_size=N, K=K,
                drop_policy=pol, drop_rate=rate,
                torch_generator=gen, collect_stats=False,
            )
            _check_constraints(res.keep_mask.cpu(), tr.cpu(), source, T, K,
                               label=f"{pol}@{rate}")
    print("  OK")


def test_all_local_no_drop():
    print("--- test_all_local_no_drop ---")
    device = _make_device()
    if device is None:
        return
    T, K, N = 16, 4, 4
    # Construct routing so every target_rank == source_rank
    source = 1
    eid = torch.full((T * K,), 17, dtype=torch.int32, device=device)
    tr = torch.full((T * K,), source, dtype=torch.int64, device=device)
    w = torch.full((T * K,), 0.25, dtype=torch.float32, device=device)
    for pol in ("tail_weight", "random", "cross_numa_first"):
        res = apply_drop_gpu_simple(
            flat_expert_ids=eid, target_rank=tr, flat_topk_w=w,
            source_rank=source, world_size=N, K=K,
            drop_policy=pol, drop_rate=0.5, collect_stats=False,
        )
        assert res.keep_mask.all().item(), f"{pol}: dropped a local replica"
    print("  OK")


def test_all_remote_protects_one_per_token():
    print("--- test_all_remote_protects_one_per_token ---")
    device = _make_device()
    if device is None:
        return
    T, K, N = 32, 4, 4
    source = 0
    eid = torch.randint(0, 100, (T * K,), dtype=torch.int32, device=device)
    # Map all to remote ranks (rotate around): rank in {1,2,3}
    tr = (torch.arange(T * K, device=device, dtype=torch.int64) % (N - 1)) + 1
    w = torch.softmax(torch.randn(T, K, device=device), dim=-1).view(-1)
    for pol in ("tail_weight", "random", "cross_numa_first"):
        for rate in (0.5, 0.95):
            gen = torch.Generator(device=device).manual_seed(7)
            res = apply_drop_gpu_simple(
                flat_expert_ids=eid, target_rank=tr, flat_topk_w=w,
                source_rank=source, world_size=N, K=K,
                drop_policy=pol, drop_rate=rate,
                torch_generator=gen, collect_stats=False,
            )
            km = res.keep_mask.view(T, K)
            assert (km.sum(dim=-1) >= 1).all().item(), \
                f"{pol}@{rate}: at least one token lost all replicas"
    print("  OK")


def test_tail_weight_parity():
    print("--- test_tail_weight_parity (CPU vs GPU) ---")
    device = _make_device()
    if device is None:
        return
    T, K, N = 128, 8, 8
    source = 3
    for seed in (0, 1, 2):
        eid, tr, w = _make_routing(T, K, N, source, seed=seed)
        eid_g, tr_g, w_g = eid.to(device), tr.to(device), w.to(device)
        for rate in (0.05, 0.10, 0.20):
            rng = random.Random(0)
            cpu = apply_drop_cpu(
                flat_expert_ids=eid, target_rank=tr, flat_topk_w=w,
                source_rank=source, world_size=N, K=K,
                drop_policy="tail_weight", drop_rate=rate, rng=rng,
            )
            gpu = apply_drop_gpu_simple(
                flat_expert_ids=eid_g, target_rank=tr_g, flat_topk_w=w_g,
                source_rank=source, world_size=N, K=K,
                drop_policy="tail_weight", drop_rate=rate, collect_stats=False,
            )
            cpu_mask = cpu.keep_mask.cpu()
            gpu_mask = gpu.keep_mask.cpu()
            # Sort-tiebreak between CPU lex (weight,i) and GPU topk may give
            # different choices when weights are equal. For random softmax
            # weights this is vanishingly rare; assert equal #drops and same
            # set when not tied.
            cpu_drop = int((~cpu_mask).sum())
            gpu_drop = int((~gpu_mask).sum())
            assert cpu_drop == gpu_drop, \
                f"seed={seed} rate={rate}: drop count CPU={cpu_drop} GPU={gpu_drop}"
            # Strict mask parity (acceptable because no tie expected on random softmax)
            assert torch.equal(cpu_mask, gpu_mask), \
                f"seed={seed} rate={rate}: mask differs (cpu_drop={cpu_drop})"
    print("  OK")


def test_cross_numa_first_parity():
    print("--- test_cross_numa_first_parity ---")
    device = _make_device()
    if device is None:
        return
    T, K, N = 96, 8, 8
    source = 1  # NUMA 0 (split=4)
    for seed in (0, 1, 2):
        eid, tr, w = _make_routing(T, K, N, source, seed=seed)
        eid_g, tr_g, w_g = eid.to(device), tr.to(device), w.to(device)
        for rate in (0.05, 0.10, 0.20):
            rng = random.Random(0)
            cpu = apply_drop_cpu(
                flat_expert_ids=eid, target_rank=tr, flat_topk_w=w,
                source_rank=source, world_size=N, K=K,
                drop_policy="cross_numa_first", drop_rate=rate, rng=rng,
            )
            gpu = apply_drop_gpu_simple(
                flat_expert_ids=eid_g, target_rank=tr_g, flat_topk_w=w_g,
                source_rank=source, world_size=N, K=K,
                drop_policy="cross_numa_first", drop_rate=rate, collect_stats=False,
            )
            cpu_mask = cpu.keep_mask.cpu()
            gpu_mask = gpu.keep_mask.cpu()
            cpu_drop = int((~cpu_mask).sum())
            gpu_drop = int((~gpu_mask).sum())
            assert cpu_drop == gpu_drop, \
                f"seed={seed} rate={rate}: drop count CPU={cpu_drop} GPU={gpu_drop}"
            assert torch.equal(cpu_mask, gpu_mask), \
                f"seed={seed} rate={rate}: mask differs (cpu_drop={cpu_drop})"
    print("  OK")


def test_random_reproducible():
    print("--- test_random_reproducible (GPU only) ---")
    device = _make_device()
    if device is None:
        return
    T, K, N = 64, 8, 8
    source = 2
    eid, tr, w = _make_routing(T, K, N, source, seed=42)
    eid, tr, w = eid.to(device), tr.to(device), w.to(device)
    rate = 0.15

    g1 = torch.Generator(device=device).manual_seed(987)
    r1 = apply_drop_gpu_simple(
        flat_expert_ids=eid, target_rank=tr, flat_topk_w=w,
        source_rank=source, world_size=N, K=K,
        drop_policy="random", drop_rate=rate,
        torch_generator=g1, collect_stats=False,
    )
    g2 = torch.Generator(device=device).manual_seed(987)
    r2 = apply_drop_gpu_simple(
        flat_expert_ids=eid, target_rank=tr, flat_topk_w=w,
        source_rank=source, world_size=N, K=K,
        drop_policy="random", drop_rate=rate,
        torch_generator=g2, collect_stats=False,
    )
    assert torch.equal(r1.keep_mask, r2.keep_mask), "random not reproducible under same seed"
    # Different seed should differ (overwhelmingly likely)
    g3 = torch.Generator(device=device).manual_seed(123)
    r3 = apply_drop_gpu_simple(
        flat_expert_ids=eid, target_rank=tr, flat_topk_w=w,
        source_rank=source, world_size=N, K=K,
        drop_policy="random", drop_rate=rate,
        torch_generator=g3, collect_stats=False,
    )
    assert not torch.equal(r1.keep_mask, r3.keep_mask), "random independent of seed"
    print("  OK")


def test_min_replicas_bypass():
    print("--- test_min_replicas_bypass ---")
    device = _make_device()
    if device is None:
        return
    T, K, N = 8, 8, 8  # L=64
    source = 0
    eid, tr, w = _make_routing(T, K, N, source, seed=5)
    eid, tr, w = eid.to(device), tr.to(device), w.to(device)
    # min_replicas > L → bypass: keep_mask should be all True
    res = apply_drop(
        flat_expert_ids=eid, target_rank=tr, flat_topk_w=w,
        source_rank=source, world_size=N, K=K,
        drop_policy="tail_weight", drop_rate=0.5,
        rng=random.Random(0),
        impl="gpu", min_replicas=128, collect_stats=False,
    )
    assert res.keep_mask.all().item(), "bypass should keep all"
    # min_replicas < L → drop should happen
    res2 = apply_drop(
        flat_expert_ids=eid, target_rank=tr, flat_topk_w=w,
        source_rank=source, world_size=N, K=K,
        drop_policy="tail_weight", drop_rate=0.5,
        rng=random.Random(0),
        impl="gpu", min_replicas=0, collect_stats=False,
    )
    assert int((~res2.keep_mask).sum()) > 0, "no bypass should drop"
    print("  OK")


def test_dispatcher_auto():
    print("--- test_dispatcher_auto (auto routes simple→GPU, grouped→CPU) ---")
    device = _make_device()
    if device is None:
        return
    T, K, N = 32, 4, 4
    source = 0
    eid, tr, w = _make_routing(T, K, N, source, seed=11)
    eid_g, tr_g, w_g = eid.to(device), tr.to(device), w.to(device)
    for pol in ("tail_weight", "random", "cross_numa_first"):
        res = apply_drop(
            flat_expert_ids=eid_g, target_rank=tr_g, flat_topk_w=w_g,
            source_rank=source, world_size=N, K=K,
            drop_policy=pol, drop_rate=0.1,
            rng=random.Random(0), impl="auto", min_replicas=0,
        )
        assert res.keep_mask.device.type == "cuda", f"{pol}: GPU mask expected"
    # Grouped policy with impl=auto -> CPU fallback (works on CPU tensors).
    res_cpu = apply_drop(
        flat_expert_ids=eid, target_rank=tr, flat_topk_w=w,
        source_rank=source, world_size=N, K=K,
        drop_policy="per_expert_uniform", drop_rate=0.2,
        rng=random.Random(0), impl="auto", min_replicas=0,
    )
    assert res_cpu.keep_mask.device.type == "cpu"
    # impl=gpu on grouped policy -> should error.
    try:
        apply_drop(
            flat_expert_ids=eid_g, target_rank=tr_g, flat_topk_w=w_g,
            source_rank=source, world_size=N, K=K,
            drop_policy="per_expert_uniform", drop_rate=0.2,
            rng=random.Random(0), impl="gpu", min_replicas=0,
        )
        raise AssertionError("expected ValueError for impl=gpu on grouped policy")
    except ValueError:
        pass
    print("  OK")


def main():
    tests = [
        test_simple_constraints,
        test_all_local_no_drop,
        test_all_remote_protects_one_per_token,
        test_tail_weight_parity,
        test_cross_numa_first_parity,
        test_random_reproducible,
        test_min_replicas_bypass,
        test_dispatcher_auto,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failures += 1
            print(f"  FAIL {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print()
    if failures:
        print(f"{failures} test(s) failed")
        sys.exit(1)
    print("All tests passed.")


if __name__ == "__main__":
    main()
