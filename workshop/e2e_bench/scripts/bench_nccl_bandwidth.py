#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Quick NCCL collective bandwidth measurement on this 8x4090 box.

Measures **algorithm bandwidth** (bytes_moved_per_rank / mean_seconds) for
all_to_all_single, all_gather_into_tensor, all_reduce, and a single-pair
PCIe send/recv at the same payload sizes used by the FlashInfer EP step.

The output is a JSONL line per (collective, payload_size) pair, plus a
small summary table on stdout. Everything runs in a single process per
rank; no model code is loaded.

Usage:

    python bench_nccl_bandwidth.py \\
        --world-size 8 \\
        --sizes 32M,64M,128M,256M \\
        --iters 50 --warmup 10 \\
        --output results/qwen3_heavy/nccl_bw.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _parse_size(s: str) -> int:
    s = s.strip().upper()
    mult = 1
    for suf, m in (("KB", 1024), ("MB", 1024**2), ("GB", 1024**3),
                   ("K", 1024), ("M", 1024**2), ("G", 1024**3)):
        if s.endswith(suf):
            mult = m
            s = s[:-len(suf)]
            break
    return int(float(s) * mult)


def _time_collective(fn, iters: int, warmup: int, device) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        st = torch.cuda.Event(enable_timing=True)
        ed = torch.cuda.Event(enable_timing=True)
        st.record()
        fn()
        ed.record()
        ed.synchronize()
        times.append(st.elapsed_time(ed))  # ms
    return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0)


def _worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.port))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    sizes = [_parse_size(s) for s in args.sizes.split(",")]
    elem_bytes = 2  # bf16
    results = []

    for nbytes in sizes:
        nelem = nbytes // elem_bytes
        # Round nelem to multiple of world_size for clean splits
        nelem = (nelem // world_size) * world_size
        nbytes_aligned = nelem * elem_bytes

        if rank == 0:
            print(f"=== payload {nbytes_aligned/1024**2:.1f} MiB ({nelem} bf16 elems) ===")

        # ---- 1. all_to_all_single ----
        send = torch.empty(nelem, dtype=torch.bfloat16, device=device)
        recv = torch.empty_like(send)
        send.normal_()

        def _a2a():
            dist.all_to_all_single(recv, send)

        mean_ms, std_ms = _time_collective(_a2a, args.iters, args.warmup, device)
        # Algorithm bandwidth (busBW): bytes leaving each rank = nbytes * (N-1)/N
        # We report both per-rank "outbound" busBW and per-rank algorithm bw.
        out_bytes = nbytes_aligned * (world_size - 1) / world_size
        bw_alg = nbytes_aligned / (mean_ms / 1e3) / 1e9     # GB/s
        bw_bus = out_bytes / (mean_ms / 1e3) / 1e9         # GB/s
        results.append({
            "coll": "all_to_all_single",
            "world_size": world_size,
            "nbytes": nbytes_aligned,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "alg_bw_GBps": bw_alg,
            "bus_bw_GBps": bw_bus,
        })

        # ---- 2. all_gather_into_tensor ----
        per_rank = nelem // world_size
        send_g = torch.empty(per_rank, dtype=torch.bfloat16, device=device)
        recv_g = torch.empty(per_rank * world_size, dtype=torch.bfloat16,
                             device=device)
        send_g.normal_()

        def _ag():
            dist.all_gather_into_tensor(recv_g, send_g)

        mean_ms_ag, std_ms_ag = _time_collective(_ag, args.iters, args.warmup, device)
        out_bytes_ag = (per_rank * elem_bytes) * (world_size - 1)
        bw_alg_ag = (per_rank * elem_bytes * world_size) / (mean_ms_ag / 1e3) / 1e9
        bw_bus_ag = out_bytes_ag / (mean_ms_ag / 1e3) / 1e9
        results.append({
            "coll": "all_gather_into_tensor",
            "world_size": world_size,
            "nbytes": per_rank * elem_bytes * world_size,
            "mean_ms": mean_ms_ag,
            "std_ms": std_ms_ag,
            "alg_bw_GBps": bw_alg_ag,
            "bus_bw_GBps": bw_bus_ag,
        })

        # ---- 3. all_reduce ----
        ar_buf = torch.empty(nelem, dtype=torch.bfloat16, device=device)
        ar_buf.normal_()

        def _ar():
            dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM)

        mean_ms_ar, std_ms_ar = _time_collective(_ar, args.iters, args.warmup, device)
        # all_reduce algorithm bandwidth: each rank sends ~ 2*(N-1)/N * nbytes
        out_bytes_ar = 2 * (world_size - 1) / world_size * nbytes_aligned
        bw_alg_ar = nbytes_aligned / (mean_ms_ar / 1e3) / 1e9
        bw_bus_ar = out_bytes_ar / (mean_ms_ar / 1e3) / 1e9
        results.append({
            "coll": "all_reduce",
            "world_size": world_size,
            "nbytes": nbytes_aligned,
            "mean_ms": mean_ms_ar,
            "std_ms": std_ms_ar,
            "alg_bw_GBps": bw_alg_ar,
            "bus_bw_GBps": bw_bus_ar,
        })

        # ---- 4. Pairwise GPU(0)<->GPU(1) send/recv (for the per-link ceiling) ----
        if world_size >= 2:
            pair_buf = torch.empty(nelem, dtype=torch.bfloat16, device=device)
            pair_buf.normal_()

            def _pair():
                if rank == 0:
                    dist.send(pair_buf, dst=1)
                elif rank == 1:
                    dist.recv(pair_buf, src=0)
                else:
                    pass
                dist.barrier()

            mean_ms_pp, std_ms_pp = _time_collective(_pair, args.iters, args.warmup, device)
            bw_pp = nbytes_aligned / (mean_ms_pp / 1e3) / 1e9
            results.append({
                "coll": "p2p_send_recv_0to1",
                "world_size": world_size,
                "nbytes": nbytes_aligned,
                "mean_ms": mean_ms_pp,
                "std_ms": std_ms_pp,
                "alg_bw_GBps": bw_pp,
                "bus_bw_GBps": bw_pp,
            })

        del send, recv, send_g, recv_g, ar_buf
        torch.cuda.empty_cache()

    if rank == 0:
        if args.output:
            outp = Path(args.output)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with outp.open("a", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(f"[bw] appended {len(results)} rows to {outp}")
        print()
        print(f"# Summary (world={world_size})")
        print(f"{'coll':<22} {'payload':>10} {'mean (ms)':>10} {'alg BW (GB/s)':>14} {'bus BW (GB/s)':>14}")
        for r in results:
            print(f"{r['coll']:<22} {r['nbytes']/1024**2:>9.1f}M "
                  f"{r['mean_ms']:>9.3f} {r['alg_bw_GBps']:>13.2f} {r['bus_bw_GBps']:>13.2f}")

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--sizes", type=str,
                   default="4M,16M,64M,128M,256M",
                   help="Comma-separated list of payload sizes (e.g. 32M,128M)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--port", type=int, default=29555)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    mp.spawn(_worker, args=(args.world_size, args),
             nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
