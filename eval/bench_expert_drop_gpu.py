"""Microbenchmark for Phase 4 drop implementations (P0).

Compares per-call latency of:
  - cpu_current  : the original CPU implementation (3× .tolist() + Python sort)
  - gpu_fast     : the new GPU path, no stats (no host sync)
  - gpu_stats    : the new GPU path, stats enabled (sync at the end)
  - gpu_bypass_*: GPU path with min_replicas bypass

Across (T, K, L) shapes and drop_rate combinations. Latency measured with CUDA
events; we also include a Python wall-clock measurement that includes any
host-side cost in the CPU path.

Run:
    /home/lzy/miniconda3/envs/vllm/bin/python -m eval.bench_expert_drop_gpu
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import time

import torch

# Make sure env doesn't pre-set knobs.
for k in ("MOE_DROP_IMPL", "MOE_DROP_MIN_REPLICAS", "MOE_DROP_GPU_STATS"):
    os.environ.pop(k, None)

from workshop.nanovllm_moe.services.utils.expert_drop import (  # noqa: E402
    apply_drop,
    apply_drop_cpu,
    apply_drop_gpu_simple,
)


def _make_routing(T, K, N, source, device, seed=0):
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)
    eid_cpu = torch.randint(0, 4 * N, (T * K,), generator=g_cpu, dtype=torch.int32)
    tr_cpu = (eid_cpu.long() % N).to(torch.int64)
    w_cpu = torch.softmax(torch.randn(T, K, generator=g_cpu), dim=-1).view(-1).contiguous()
    return (
        eid_cpu.to(device), tr_cpu.to(device), w_cpu.to(device),
        eid_cpu, tr_cpu, w_cpu,
    )


def bench_gpu(fn, *, iters: int, warmup: int):
    """Return median microseconds per call using CUDA events."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # CUDA events around a tight loop of `iters` invocations.
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)  # ms → µs
    return statistics.median(samples), max(samples), min(samples)


def bench_cpu(fn, *, iters: int, warmup: int):
    """Wall-clock latency for CPU-heavy path."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e6)  # s → µs
    return statistics.median(samples), max(samples), min(samples)


SHAPES = [
    (1, 8),     # decode step, single seq
    (8, 8),     # decode step, max_num_seqs=8
    (16, 8),
    (64, 8),
    (256, 8),   # prefill batch
    (1024, 8),
]
RATES = [0.05, 0.10, 0.20]
POLICIES = ["tail_weight", "random", "cross_numa_first"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; aborting.")
        return
    device = torch.device("cuda:0")

    header = f"{'shape (T,K,L)':<14} {'policy':<18} {'rate':>5} "
    header += f"{'cpu_us':>10} {'gpu_fast_us':>12} {'gpu_stats_us':>13} {'gpu_byp_us':>11} {'speedup':>9}"
    print(header)
    print("-" * len(header))

    for (T, K) in SHAPES:
        L = T * K
        eid_g, tr_g, w_g, eid_c, tr_c, w_c = _make_routing(
            T, K, args.world_size, args.source, device, seed=args.seed
        )
        for pol in POLICIES:
            for rate in RATES:
                # ---- CPU current ----
                def cpu_call():
                    rng = random.Random(0)
                    return apply_drop_cpu(
                        flat_expert_ids=eid_c, target_rank=tr_c, flat_topk_w=w_c,
                        source_rank=args.source, world_size=args.world_size, K=K,
                        drop_policy=pol, drop_rate=rate, rng=rng,
                    )
                cpu_med, _, _ = bench_cpu(cpu_call, iters=args.iters, warmup=args.warmup)

                # ---- GPU fast path ----
                gen = torch.Generator(device=device).manual_seed(0)
                def gpu_fast_call():
                    return apply_drop_gpu_simple(
                        flat_expert_ids=eid_g, target_rank=tr_g, flat_topk_w=w_g,
                        source_rank=args.source, world_size=args.world_size, K=K,
                        drop_policy=pol, drop_rate=rate,
                        torch_generator=gen, collect_stats=False,
                    )
                gpu_fast_med, _, _ = bench_gpu(gpu_fast_call, iters=args.iters, warmup=args.warmup)

                # ---- GPU with stats ----
                gen2 = torch.Generator(device=device).manual_seed(0)
                def gpu_stats_call():
                    return apply_drop_gpu_simple(
                        flat_expert_ids=eid_g, target_rank=tr_g, flat_topk_w=w_g,
                        source_rank=args.source, world_size=args.world_size, K=K,
                        drop_policy=pol, drop_rate=rate,
                        torch_generator=gen2, collect_stats=True,
                    )
                gpu_stats_med, _, _ = bench_gpu(gpu_stats_call, iters=args.iters, warmup=args.warmup)

                # ---- GPU bypass (min_replicas=128) ----
                gen3 = torch.Generator(device=device).manual_seed(0)
                def gpu_bypass_call():
                    return apply_drop(
                        flat_expert_ids=eid_g, target_rank=tr_g, flat_topk_w=w_g,
                        source_rank=args.source, world_size=args.world_size, K=K,
                        drop_policy=pol, drop_rate=rate,
                        rng=random.Random(0),
                        impl="auto", min_replicas=128,
                        torch_generator=gen3, collect_stats=False,
                    )
                gpu_byp_med, _, _ = bench_gpu(gpu_bypass_call, iters=args.iters, warmup=args.warmup)

                speedup = cpu_med / max(gpu_fast_med, 1e-3)
                print(
                    f"{(T,K,L)!s:<14} {pol:<18} {rate:>5} "
                    f"{cpu_med:>10.1f} {gpu_fast_med:>12.1f} "
                    f"{gpu_stats_med:>13.1f} {gpu_byp_med:>11.1f} {speedup:>8.1f}x"
                )


if __name__ == "__main__":
    main()
