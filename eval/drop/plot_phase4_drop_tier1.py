"""Render Tier 1 plots from tier1_rows.jsonl + tier1_summary.json.

Usage:
    python -m eval.drop.plot_tier1 --output-dir eval_results/prefill_drop_l_sweep_tier1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    rows_path = os.path.join(args.output_dir, "tier1_rows.jsonl")
    summary_path = os.path.join(args.output_dir, "tier1_summary.json")
    with open(rows_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    with open(summary_path) as f:
        summary = json.load(f)

    # Group rows by (T_local, drop_rate).
    cells = defaultdict(list)
    for r in rows:
        cells[(r["T_local"], r["drop_rate"])].append(r)

    rates = sorted({rate for _, rate in cells.keys()})
    T_locals = sorted({T for T, _ in cells.keys()})

    def median(seg):
        out = {}
        for (T, rate), rs in cells.items():
            out[(T, rate)] = float(np.median([r[seg] for r in rs]))
        return out

    L_med = median("L_recv_max")
    total_med = median("total_us_rank_max")
    disp_med = median("dispatch_us_rank_max")
    exp_med = median("experts_us_rank_max")
    com_med = median("combine_us_rank_max")

    # --- Figure 1: total_us vs L_recv_max, baseline + drop two lines, log-x ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for rate in rates:
        xs = [L_med[(T, rate)] for T in T_locals]
        ys = [total_med[(T, rate)] for T in T_locals]
        label = f"baseline (rate=0)" if rate == 0.0 else f"drop rate={rate}"
        ax.plot(xs, ys, marker="o", label=label, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("L_recv_max (per-rank rows received, slowest)")
    ax.set_ylabel("MoE-block total_us (rank-max, median)")
    ax.set_title("Tier 1: total wall time vs L_recv (8 ranks, Qwen3-30B-A3B EP-HT)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    if summary.get("L_star"):
        ax.axvline(summary["L_star"], color="red", linestyle="--", alpha=0.5,
                   label=f"L* ≈ {summary['L_star']:.0f}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "tier1_total_vs_l.png"), dpi=140)
    plt.close(fig)

    # --- Figure 2: delta_pct vs L_recv ---
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [p["L_recv_max_p50"] for p in summary["delta_points"]]
    ys = [p["delta_pct"] * 100 for p in summary["delta_points"]]
    ax.plot(xs, ys, marker="o", color="C2", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(-5, color="red", linestyle="--", alpha=0.5, label="α = -5%")
    if summary.get("L_star"):
        ax.axvline(summary["L_star"], color="red", linestyle=":", alpha=0.7,
                   label=f"L* ≈ {summary['L_star']:.0f}")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:+.1f}%", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("L_recv_max")
    ax.set_ylabel("delta_pct (drop vs baseline) [%]")
    ax.set_title("Tier 1: drop vs baseline relative wall time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "tier1_delta_vs_l.png"), dpi=140)
    plt.close(fig)

    # --- Figure 3: segments breakdown for baseline & drop ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=False)
    for ax, rate in zip(axes, rates):
        xs = [L_med[(T, rate)] for T in T_locals]
        for seg, lbl, color in [
            (disp_med, "dispatch", "C0"),
            (exp_med, "experts", "C1"),
            (com_med, "combine", "C2"),
        ]:
            ys = [seg[(T, rate)] for T in T_locals]
            ax.plot(xs, ys, marker="o", label=lbl, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"drop_rate={rate}")
        ax.set_xlabel("L_recv_max")
        ax.set_ylabel("segment_us (rank-max median)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("Tier 1: per-segment wall time")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "tier1_segments.png"), dpi=140)
    plt.close(fig)

    print(f"[plot] wrote tier1_total_vs_l.png, tier1_delta_vs_l.png, tier1_segments.png in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
