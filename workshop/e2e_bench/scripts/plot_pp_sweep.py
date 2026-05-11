#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot pipeline-parallel MoE step time as a function of microbatch count.

Reads ``flashinfer_pp.jsonl`` (`benchmark_pp_moe.py` outputs) and renders
a comparison against the EP=8 and single-GPU references from
`flashinfer_native_ep.jsonl` and `flashinfer_qwen3_M32768.jsonl`.

Output: ``workshop/e2e_bench/figures/qwen3_pp_sweep.png``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_jsonl(p: Path) -> list[dict]:
    rows = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _ep_reference_step(rows: list[dict]) -> float | None:
    """Pick the world=8, oracle uniform, no-autotune EP run as the reference."""
    for r in rows:
        if (r.get("impl") == "flashinfer_native_ep_alltoall"
                and r.get("world_size") == 8
                and r.get("routing") == "oracle_uniform"
                and not r.get("autotune", False)):
            return r["max_rank_mean_step_ms"]
    return None


def _single_gpu_reference(rows: list[dict], M: int = 32768, autotune: bool = True) -> float | None:
    # Schema: impl="flashinfer_cutlass_fused_moe", step time field is
    # "total_step_mean_ms" (not "max_rank_mean_step_ms" — that lives in the EP
    # / PP files). routing="oracle_uniform".
    for r in rows:
        if (r.get("impl") == "flashinfer_cutlass_fused_moe"
                and r.get("tokens") == M
                and r.get("routing") == "oracle_uniform"
                and bool(r.get("autotune", False)) == autotune):
            return r.get("total_step_mean_ms") or r.get("max_rank_mean_step_ms")
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pp", type=Path,
                   default=Path("/home/yyx/personal/inference/vllm-bench/results/qwen3_heavy/flashinfer_pp.jsonl"))
    p.add_argument("--ep", type=Path,
                   default=Path("/home/yyx/personal/inference/vllm-bench/results/qwen3_heavy/flashinfer_native_ep.jsonl"))
    p.add_argument("--single", type=Path,
                   default=Path("/home/yyx/personal/inference/vllm-bench/results/qwen3_heavy/flashinfer_qwen3_M32768.jsonl"))
    p.add_argument("--hybrid", type=Path,
                   default=Path("/home/yyx/personal/inference/vllm-bench/results/qwen3_heavy/flashinfer_hybrid_ep_pp.jsonl"))
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "figures" / "qwen3_pp_sweep.png")
    args = p.parse_args()

    pp_rows = _read_jsonl(args.pp)
    ep_rows = _read_jsonl(args.ep)
    sg_rows = _read_jsonl(args.single)
    hyb_rows = _read_jsonl(args.hybrid)

    by_autotune: dict[bool, dict[int, float]] = defaultdict(dict)
    for r in pp_rows:
        if r.get("impl") != "flashinfer_pp":
            continue
        if r.get("tokens") != 32768 or r.get("world_size") != 8:
            continue
        B = int(r["microbatches"])
        t = float(r["max_rank_mean_step_ms"])
        prev = by_autotune[bool(r.get("autotune", False))].get(B)
        # Keep the last (most recent) measurement per (autotune, B) cell.
        by_autotune[bool(r.get("autotune", False))][B] = t

    ep_ref = _ep_reference_step(ep_rows)
    sg_ref_at = _single_gpu_reference(sg_rows, autotune=True)
    sg_ref_no = _single_gpu_reference(sg_rows, autotune=False)

    Bs = sorted(set(list(by_autotune[False].keys()) + list(by_autotune[True].keys())))
    if not Bs:
        raise SystemExit("No PP rows in the input file.")

    fig, ax = plt.subplots(figsize=(10, 5.6))

    # Theoretical lower bound: T_step = (B + S − 1) · T_stage_ideal,
    # where T_stage_ideal is the per-microbatch compute on one stage at the
    # autotuned single-GPU rate (128 TFLOP/s -> ~ 120 TFLOP/s effective in pipeline).
    H, I, K, L, M = 2048, 768, 8, 48, 32768
    S = 8
    L_per_stage = L // S
    Bs_dense = np.array(sorted(set(Bs + [1, 2, 4, 8, 16, 32, 64, 128])))
    M_micro_dense = M / Bs_dense
    flops_per_stage_per_mb = M_micro_dense * K * 6 * H * I * L_per_stage
    T_stage_ideal_ms = flops_per_stage_per_mb / (120e12) * 1e3
    T_step_ideal_ms = (Bs_dense + S - 1) * T_stage_ideal_ms
    ax.plot(Bs_dense, T_step_ideal_ms, ":", color="black", lw=1.4,
            label="theoretical lower bound\n(120 TFLOP/s, no comm overhead)")

    if False in by_autotune:
        xs = sorted(by_autotune[False].keys())
        ys = [by_autotune[False][b] for b in xs]
        ax.plot(xs, ys, "o-", color="tab:blue", lw=2, markersize=8,
                label="measured PP (no autotune)")
        for x, y in zip(xs, ys):
            ax.text(x, y + 25, f"{y:.0f}", ha="center", fontsize=8, color="tab:blue")

    if True in by_autotune:
        xs = sorted(by_autotune[True].keys())
        ys = [by_autotune[True][b] for b in xs]
        ax.plot(xs, ys, "s-", color="tab:purple", lw=2, markersize=8,
                label="measured PP (FlashInfer autotune)")
        for x, y in zip(xs, ys):
            ax.text(x, y - 35, f"{y:.0f}", ha="center", fontsize=8, color="tab:purple")

    # Hybrid EP × PP measurements (markers only, scatter style).
    hyb_groups: dict[tuple[int, int], dict[int, float]] = {}
    for r in hyb_rows:
        if r.get("impl") != "flashinfer_hybrid_ep_pp":
            continue
        if r.get("tokens") != 32768 or r.get("autotune", False):
            continue
        key = (int(r["ep_size"]), int(r["pp_size"]))
        hyb_groups.setdefault(key, {})[int(r["microbatches"])] = float(r["max_rank_mean_step_ms"])

    hyb_palette = {
        (2, 4): ("tab:brown",  "x"),
        (4, 2): ("tab:cyan",   "+"),
        (1, 8): ("tab:purple", "."),
        (8, 1): ("tab:red",    "."),
    }
    for (ep_s, pp_s), bs_to_ms in sorted(hyb_groups.items()):
        if not bs_to_ms:
            continue
        xs = sorted(bs_to_ms.keys())
        ys = [bs_to_ms[b] for b in xs]
        color, marker = hyb_palette.get((ep_s, pp_s), ("tab:gray", "x"))
        ax.plot(xs, ys, marker=marker, linestyle="none", color=color,
                markersize=10, mew=2,
                label=f"measured hybrid EP={ep_s}\u00d7PP={pp_s} (no autotune)")
        for x, y in zip(xs, ys):
            ax.text(x, y + 80, f"{y:.0f}", ha="center", fontsize=8, color=color)

    # References as horizontal lines.
    if ep_ref is not None:
        ax.axhline(ep_ref, color="tab:red", ls="--", lw=1.4,
                   label=f"FlashInfer EP=8 oracle ({ep_ref:.0f} ms)")
    if sg_ref_at is not None:
        ax.axhline(sg_ref_at, color="tab:green", ls="--", lw=1.4,
                   label=f"FlashInfer single-GPU autotune ({sg_ref_at:.0f} ms)")
    if sg_ref_no is not None and sg_ref_no != sg_ref_at:
        ax.axhline(sg_ref_no, color="tab:olive", ls=":", lw=1.0, alpha=0.6,
                   label=f"FlashInfer single-GPU ({sg_ref_no:.0f} ms)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(Bs)
    ax.set_xticklabels([str(b) for b in Bs])
    ax.set_xlabel("microbatch count B (M_total = 32 768, S = 8 stages)")
    ax.set_ylabel("step time (ms, max across ranks)")
    ax.set_title("Parallelism strategies on 8\u00d74090 — pure PP vs. EP\u00d7PP hybrids vs. EP=8 / single-GPU")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    if Bs:
        hyb_max = 0.0
        for d in hyb_groups.values():
            if d:
                hyb_max = max(hyb_max, max(d.values()))
        ymax = max(
            ep_ref if ep_ref is not None else 0,
            max(by_autotune.get(False, {0: 0}).values(), default=0),
            max(by_autotune.get(True, {0: 0}).values(), default=0),
            hyb_max,
        )
        ax.set_ylim(0, ymax * 1.15)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
