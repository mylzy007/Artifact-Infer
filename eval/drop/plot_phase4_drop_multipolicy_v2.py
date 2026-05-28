"""Plots for v2 sweep: policy × rate + score (F1).

Generates:
  - v2_rate_lines_prefill.png   (rate -> prefill_speedup, one line per policy)
  - v2_rate_lines_e2e.png       (rate -> e2e_speedup, one line per policy)
  - v2_pareto_score_speedup.png (scatter: x=prefill_speedup, y=f1, color=policy)
  - v2_rate_lines_f1.png        (rate -> f1, one line per policy + horizontal baseline)
  - v2_table.md                 (markdown comprehensive table)

Usage:
    python -m eval.drop.plot_longbench_v2 --output-dir eval_results/.../v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    summary = json.load(open(os.path.join(args.output_dir, "v2_summary.json")))
    cells = summary["cells"]
    rel = summary["relative_to_baseline"]
    base = cells["baseline"]
    base_f1 = base["f1_mean"]["mean"] if base["f1_mean"] else None

    policies = sorted({c["policy"] for c in cells.values() if c["policy"] != "none"})
    rates = sorted({c["rate"] for c in cells.values() if c["rate"] > 0})

    GPU = {"tail_weight", "random", "cross_numa_first"}
    color_map = {}
    cmap = plt.get_cmap("tab10")
    for i, p_name in enumerate(policies):
        color_map[p_name] = cmap(i)

    def cell_for(pol, rate):
        return cells.get(f"{pol}_r{rate}")

    # ---------- 1) Prefill speedup vs rate, per policy ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    for pol in policies:
        xs, ys = [], []
        for r in rates:
            c = cell_for(pol, r)
            if c is None:
                continue
            label = f"{pol}_r{r}"
            sp = rel.get(label, {}).get("prefill_speedup")
            if sp is None:
                continue
            xs.append(r)
            ys.append(sp)
        marker = "o" if pol in GPU else "s"
        line = "-" if pol in GPU else "--"
        ax.plot(xs, ys, marker=marker, linestyle=line, color=color_map[pol], label=pol, linewidth=2)
    ax.axhline(1.0, color="black", linewidth=0.7)
    ax.set_xlabel("drop_rate")
    ax.set_ylabel("prefill_speedup vs baseline")
    ax.set_title("Phase A v2: prefill_speedup vs drop_rate per policy\n(solid=GPU path, dashed=CPU grouped path)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "v2_rate_lines_prefill.png"), dpi=140)
    plt.close(fig)

    # ---------- 2) e2e speedup vs rate ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    for pol in policies:
        xs, ys = [], []
        for r in rates:
            c = cell_for(pol, r)
            label = f"{pol}_r{r}"
            sp = rel.get(label, {}).get("e2e_speedup")
            if sp is None:
                continue
            xs.append(r)
            ys.append(sp)
        marker = "o" if pol in GPU else "s"
        line = "-" if pol in GPU else "--"
        ax.plot(xs, ys, marker=marker, linestyle=line, color=color_map[pol], label=pol, linewidth=2)
    ax.axhline(1.0, color="black", linewidth=0.7)
    ax.set_xlabel("drop_rate")
    ax.set_ylabel("e2e_speedup vs baseline")
    ax.set_title("Phase A v2: e2e_speedup vs drop_rate per policy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "v2_rate_lines_e2e.png"), dpi=140)
    plt.close(fig)

    # ---------- 3) F1 vs rate per policy ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    for pol in policies:
        xs, ys = [], []
        for r in rates:
            c = cell_for(pol, r)
            if c is None or not c["f1_mean"]:
                continue
            xs.append(r)
            ys.append(c["f1_mean"]["mean"])
        marker = "o" if pol in GPU else "s"
        line = "-" if pol in GPU else "--"
        ax.plot(xs, ys, marker=marker, linestyle=line, color=color_map[pol], label=pol, linewidth=2)
    if base_f1 is not None:
        ax.axhline(base_f1, color="black", linewidth=0.8, label=f"baseline F1={base_f1:.3f}")
    ax.set_xlabel("drop_rate")
    ax.set_ylabel("F1 (token-level) on LEval multidoc_qa")
    ax.set_title("Phase A v2: F1 vs drop_rate per policy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "v2_rate_lines_f1.png"), dpi=140)
    plt.close(fig)

    # ---------- 4) Pareto: x=prefill_speedup, y=F1 ----------
    fig, ax = plt.subplots(figsize=(9, 6))
    if base_f1 is not None:
        ax.scatter([1.0], [base_f1], marker="*", s=300, color="black", zorder=5, label="baseline")
    for pol in policies:
        xs, ys, sizes = [], [], []
        for r in rates:
            label = f"{pol}_r{r}"
            sp = rel.get(label, {}).get("prefill_speedup")
            c = cell_for(pol, r)
            if sp is None or c is None or not c["f1_mean"]:
                continue
            xs.append(sp)
            ys.append(c["f1_mean"]["mean"])
            sizes.append(100 + 200 * r)
        marker = "o" if pol in GPU else "s"
        ax.scatter(xs, ys, marker=marker, color=color_map[pol], label=pol, s=sizes, alpha=0.7, edgecolors="black")
        for r, x, y in zip(rates, xs, ys):
            ax.annotate(f"r={r}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)
    if base_f1 is not None:
        ax.axhline(base_f1, color="gray", linestyle=":", linewidth=0.6)
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=0.6)
    ax.set_xlabel("prefill_speedup vs baseline")
    ax.set_ylabel("F1 (higher = better accuracy)")
    ax.set_title("Pareto: F1 vs prefill_speedup. Top-right is best.\n"
                 "(point size scales with drop_rate; star = baseline at (1.0, baseline F1))")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "v2_pareto_score_speedup.png"), dpi=140)
    plt.close(fig)

    # ---------- 5) markdown comprehensive table ----------
    lines = [
        "# v2 sweep — comprehensive per-cell table",
        "",
        "| policy | path | rate | prefill (s) | prefill_sp | e2e_sp | F1 | F1 Δ (abs) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    if base_f1 is not None:
        lines.append(
            f"| baseline | - | 0.0 | "
            f"{base['prefill_time_s']['mean']:.3f} | 1.000 | 1.000 | "
            f"{base_f1:.3f} | — |"
        )
    for label in sorted(rel.keys()):
        s = cells[label]
        r = rel[label]
        path = "CPU" if s["is_cpu_path"] else "GPU"
        f1m = s["f1_mean"]["mean"] if s["f1_mean"] else None
        f1d = r["f1_delta_abs"]
        psp = r["prefill_speedup"]
        esp = r["e2e_speedup"]
        lines.append(
            f"| {s['policy']} | {path} | {s['rate']} | "
            f"{s['prefill_time_s']['mean']:.3f} | "
            f"{psp:.3f} | {esp:.3f} | "
            f"{'%.3f' % f1m if f1m is not None else '-'} | "
            f"{('%+.3f' % f1d) if f1d is not None else '-'} |"
        )
    open(os.path.join(args.output_dir, "v2_table.md"), "w").write("\n".join(lines))

    print(f"[plot_longbench_v2] wrote plots + v2_table.md to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
