"""Render plots for the longbench multi-axis sweep.

Reads `longbench_rows.jsonl` + `longbench_summary.json` and writes:
  - phase_a_policy_rate_heatmap.png   (policy x rate -> prefill_speedup)
  - phase_a_rate_lines.png            (one line per policy: rate -> speedup)
  - phase_b_bypass.png                (min_replicas -> speedup, log-x)
  - phase_c_tier.png                  (tier -> prefill_speedup, e2e_speedup)
  - per_cell_table.md                 (markdown table of every cell)

Usage:
    python -m eval.drop.plot_longbench --output-dir eval_results/owner_local_ep_phase4_drop_longbench
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

    rows = [
        json.loads(l)
        for l in open(os.path.join(args.output_dir, "longbench_rows.jsonl"))
        if l.strip()
    ]
    summary = json.load(open(os.path.join(args.output_dir, "longbench_summary.json")))
    cells = summary["cells"]
    speedups = summary["speedups"]

    # ---------- Phase A: heatmap policy x rate ----------
    a_cells = {k: v for k, v in cells.items() if v["phase"] == "A"}
    policies = sorted({v["policy"] for v in a_cells.values() if v["policy"] != "none"})
    rates = sorted({v["rate"] for v in a_cells.values() if v["rate"] > 0})
    grid_prefill = np.full((len(policies), len(rates)), np.nan)
    grid_e2e = np.full((len(policies), len(rates)), np.nan)
    for label, cell in a_cells.items():
        if cell["policy"] == "none":
            continue
        sp = speedups.get(label, {})
        if not sp:
            continue
        i = policies.index(cell["policy"])
        j = rates.index(cell["rate"])
        grid_prefill[i, j] = sp.get("prefill_speedup") or np.nan
        grid_e2e[i, j] = sp.get("e2e_speedup") or np.nan

    fig, axes = plt.subplots(1, 2, figsize=(13, max(3, len(policies) * 0.7 + 1)))
    for ax, grid, title in [
        (axes[0], grid_prefill, "prefill_speedup"),
        (axes[1], grid_e2e, "e2e_speedup"),
    ]:
        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=0.85, vmax=1.25)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{r:.2g}" for r in rates])
        ax.set_yticks(range(len(policies)))
        ax.set_yticklabels(policies)
        ax.set_xlabel("drop_rate")
        ax.set_ylabel("drop_policy")
        ax.set_title(f"Phase A: {title}")
        for i in range(len(policies)):
            for j in range(len(rates)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i,j]:.3f}",
                            ha="center", va="center", color="black", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "phase_a_policy_rate_heatmap.png"), dpi=140)
    plt.close(fig)

    # ---------- Phase A: line plot of rate -> speedup per policy ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, pol in enumerate(policies):
        ys = grid_prefill[i, :]
        ax.plot(rates, ys, marker="o", label=pol, linewidth=2)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xlabel("drop_rate")
    ax.set_ylabel("prefill speedup vs baseline")
    ax.set_title("Phase A: drop_rate vs prefill_speedup, per policy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "phase_a_rate_lines.png"), dpi=140)
    plt.close(fig)

    # ---------- Phase B: bypass sensitivity ----------
    b_points = []
    # Phase B cells are tail_weight@0.3 at varying min_replicas, plus the
    # corresponding A cell (which lives at the design's phase_ab_min_replicas).
    for label, cell in cells.items():
        if cell["policy"] != "tail_weight" or cell["rate"] != 0.3:
            continue
        if cell["phase"] not in ("A", "B"):
            continue
        sp = speedups.get(label, {})
        if not sp:
            continue
        b_points.append({
            "min_replicas": cell["min_replicas"],
            "prefill_speedup": sp.get("prefill_speedup"),
            "e2e_speedup": sp.get("e2e_speedup"),
        })
    b_points.sort(key=lambda x: x["min_replicas"])
    if b_points:
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = [p["min_replicas"] for p in b_points]
        ys_pref = [p["prefill_speedup"] for p in b_points]
        ys_e2e = [p["e2e_speedup"] for p in b_points]
        # Use symlog so we can show 0.
        ax.set_xscale("symlog", linthresh=1)
        ax.plot(xs, ys_pref, marker="o", label="prefill_speedup", linewidth=2)
        ax.plot(xs, ys_e2e, marker="s", label="e2e_speedup", linewidth=2)
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_xlabel("MOE_DROP_MIN_REPLICAS (bypass threshold)")
        ax.set_ylabel("speedup vs baseline")
        ax.set_title("Phase B: bypass threshold sensitivity (tail_weight@0.3, medium tier)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        for x, y in zip(xs, ys_pref):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=9, ha="center")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "phase_b_bypass.png"), dpi=140)
        plt.close(fig)

    # ---------- Phase C: tier scaling ----------
    c_cells = [v for v in cells.values() if v["phase"] == "C"]
    tier_order = ["short", "medium", "long"]
    tier_data = []
    for tier in tier_order:
        baseline_label = next((k for k, v in cells.items()
                               if v["phase"] == "C" and v["tier"] == tier and v["policy"] == "none"),
                              None)
        drop_label = next((k for k, v in cells.items()
                           if v["phase"] == "C" and v["tier"] == tier and v["policy"] != "none"),
                          None)
        if baseline_label is None or drop_label is None:
            continue
        sp = speedups.get(drop_label, {})
        tier_data.append({
            "tier": tier,
            "p50_tokens": cells[baseline_label]["total_prompt_tokens_mean"] / cells[baseline_label]["n_batches"] / 8,
            "prefill_speedup": sp.get("prefill_speedup"),
            "e2e_speedup": sp.get("e2e_speedup"),
            "baseline_prefill_s": cells[baseline_label]["prefill_time_s"]["mean"],
            "drop_prefill_s": cells[drop_label]["prefill_time_s"]["mean"],
        })
    if tier_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        x_labels = [d["tier"] for d in tier_data]
        x = np.arange(len(x_labels))
        w = 0.35
        ax.bar(x - w / 2, [d["prefill_speedup"] for d in tier_data], w, label="prefill_speedup")
        ax.bar(x + w / 2, [d["e2e_speedup"] for d in tier_data], w, label="e2e_speedup")
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d['tier']}\n(~{d['p50_tokens']:.0f} tok)" for d in tier_data])
        ax.set_ylabel("speedup vs baseline at same tier")
        ax.set_title("Phase C: prompt-length tier scaling (tail_weight@0.3, bypass=512)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        for i, d in enumerate(tier_data):
            ax.annotate(f"{d['prefill_speedup']:.3f}", (i - w / 2, d["prefill_speedup"]),
                        textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
            ax.annotate(f"{d['e2e_speedup']:.3f}", (i + w / 2, d["e2e_speedup"]),
                        textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "phase_c_tier.png"), dpi=140)
        plt.close(fig)

    # ---------- Markdown per-cell table ----------
    lines = ["# Per-cell summary\n"]
    lines.append("| phase | label | policy | rate | min_replicas | tier | n_batches | prefill_mean (s) | prefill_std | prefill_speedup | e2e_speedup |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for label, cell in sorted(cells.items(), key=lambda kv: (kv[1]["phase"], kv[0])):
        sp = speedups.get(label, {})
        pm = cell["prefill_time_s"]["mean"]
        ps = cell["prefill_time_s"]["std"]
        ps_pf = f"{sp.get('prefill_speedup'):.3f}" if sp.get("prefill_speedup") else "—"
        ps_e2 = f"{sp.get('e2e_speedup'):.3f}" if sp.get("e2e_speedup") else "—"
        lines.append(
            f"| {cell['phase']} | {label} | {cell['policy']} | {cell['rate']} | "
            f"{cell['min_replicas']} | {cell['tier']} | {cell['n_batches']} | "
            f"{pm:.3f} | {ps:.3f} | {ps_pf} | {ps_e2} |"
        )
    open(os.path.join(args.output_dir, "per_cell_table.md"), "w").write("\n".join(lines))

    print(f"[plot_longbench] wrote plots + per_cell_table.md to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
