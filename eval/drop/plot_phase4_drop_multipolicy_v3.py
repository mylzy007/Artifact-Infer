"""Plots for v3 sweep: 2 overlap plans × 5 GPU policies × 3 rates."""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(plan_dir: str) -> dict:
    return json.load(open(os.path.join(plan_dir, "v2_summary.json")))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--v3-root", required=True)
    args = p.parse_args()

    plan_dirs = {
        "RR / numa_local_first / min_communication": os.path.join(args.v3_root, "rr_mincomm"),
        "LBG / numa_local_first / greedy_balance":   os.path.join(args.v3_root, "lbg_greedybal"),
    }
    plans = {k: load(v) for k, v in plan_dirs.items()}

    # collect all (plan, policy, rate) tuples
    plan_names = list(plans.keys())
    policies = sorted({c["policy"]
                       for plan in plans.values()
                       for c in plan["cells"].values()
                       if c["policy"] != "none"})
    rates = sorted({c["rate"]
                    for plan in plans.values()
                    for c in plan["cells"].values()
                    if c["rate"] > 0})

    cmap = plt.get_cmap("tab10")
    color_map = {pol: cmap(i) for i, pol in enumerate(policies)}

    # ---------------- 1) Per-plan: e2e_speedup vs rate per policy --------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, plan_name in zip(axes, plan_names):
        plan = plans[plan_name]
        for pol in policies:
            xs, ys = [], []
            for r in rates:
                label = f"{pol}_r{r}"
                if label not in plan["relative_to_baseline"]:
                    continue
                sp = plan["relative_to_baseline"][label].get("e2e_speedup")
                if sp is None:
                    continue
                xs.append(r); ys.append(sp)
            ax.plot(xs, ys, marker="o", color=color_map[pol], label=pol, linewidth=2)
        ax.axhline(1.0, color="black", linewidth=0.7)
        ax.set_xlabel("drop_rate")
        ax.set_title(plan_name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("e2e_speedup vs baseline")
    fig.suptitle("v3 — e2e speedup per policy, across 2 Phase-3 overlap plans")
    fig.tight_layout()
    fig.savefig(os.path.join(args.v3_root, "v3_e2e_per_plan.png"), dpi=140)
    plt.close(fig)

    # ---------------- 2) Per-plan: prefill_speedup vs rate ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, plan_name in zip(axes, plan_names):
        plan = plans[plan_name]
        for pol in policies:
            xs, ys = [], []
            for r in rates:
                label = f"{pol}_r{r}"
                if label not in plan["relative_to_baseline"]:
                    continue
                sp = plan["relative_to_baseline"][label].get("prefill_speedup")
                if sp is None:
                    continue
                xs.append(r); ys.append(sp)
            ax.plot(xs, ys, marker="o", color=color_map[pol], label=pol, linewidth=2)
        ax.axhline(1.0, color="black", linewidth=0.7)
        ax.set_xlabel("drop_rate")
        ax.set_title(plan_name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("prefill_speedup")
    fig.suptitle("v3 — prefill speedup per policy")
    fig.tight_layout()
    fig.savefig(os.path.join(args.v3_root, "v3_prefill_per_plan.png"), dpi=140)
    plt.close(fig)

    # ---------------- 3) F1 vs rate per policy per plan -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, plan_name in zip(axes, plan_names):
        plan = plans[plan_name]
        base_f1 = plan["cells"]["baseline"]["f1_mean"]["mean"]
        for pol in policies:
            xs, ys = [], []
            for r in rates:
                label = f"{pol}_r{r}"
                if label not in plan["cells"]:
                    continue
                f1 = plan["cells"][label]["f1_mean"]["mean"]
                xs.append(r); ys.append(f1)
            ax.plot(xs, ys, marker="o", color=color_map[pol], label=pol, linewidth=2)
        ax.axhline(base_f1, color="black", linewidth=0.8, label=f"baseline F1={base_f1:.3f}")
        ax.set_xlabel("drop_rate")
        ax.set_title(plan_name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    axes[0].set_ylabel("F1 (token-level)")
    fig.suptitle("v3 — F1 per policy. Shaded band would be ±2·SEM ≈ ±0.04.")
    fig.tight_layout()
    fig.savefig(os.path.join(args.v3_root, "v3_f1_per_plan.png"), dpi=140)
    plt.close(fig)

    # ---------------- 4) Pareto across both plans ------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    markers = {plan_names[0]: "o", plan_names[1]: "s"}
    for plan_name in plan_names:
        plan = plans[plan_name]
        base_f1 = plan["cells"]["baseline"]["f1_mean"]["mean"]
        ax.scatter([1.0], [base_f1], marker=markers[plan_name], s=300, color="black",
                   zorder=5, label=f"baseline {plan_name[:18]}")
        for pol in policies:
            for r in rates:
                label = f"{pol}_r{r}"
                if label not in plan["cells"]:
                    continue
                sp = plan["relative_to_baseline"][label]["e2e_speedup"]
                f1 = plan["cells"][label]["f1_mean"]["mean"]
                ax.scatter([sp], [f1], marker=markers[plan_name], s=80+200*r,
                           color=color_map[pol], alpha=0.7, edgecolors="black")
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=0.6)
    ax.set_xlabel("e2e_speedup vs baseline")
    ax.set_ylabel("F1 (token-level)")
    ax.set_title("v3 Pareto: F1 vs e2e_speedup. Top-right is best.\n"
                 "Markers: ●=RR/min_comm, ■=LBG/greedy_balance. Size scales with drop_rate.")
    # Custom legend
    handles = []
    for pol in policies:
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[pol],
                                  markersize=10, label=pol))
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.v3_root, "v3_pareto.png"), dpi=140)
    plt.close(fig)

    # ---------------- 5) Cross-plan delta (same policy, RR vs LBG) -------------
    fig, ax = plt.subplots(figsize=(10, 5))
    rows = []
    for pol in policies:
        for r in rates:
            label = f"{pol}_r{r}"
            rr_sp = plans[plan_names[0]]["relative_to_baseline"].get(label, {}).get("e2e_speedup")
            lbg_sp = plans[plan_names[1]]["relative_to_baseline"].get(label, {}).get("e2e_speedup")
            if rr_sp is None or lbg_sp is None: continue
            rows.append((pol, r, rr_sp, lbg_sp, lbg_sp - rr_sp))
    rows.sort(key=lambda t: (t[0], t[1]))
    xlabels = [f"{p}\nr={r}" for (p, r, *_) in rows]
    x = np.arange(len(rows))
    width = 0.4
    ax.bar(x - width/2, [t[2] for t in rows], width, label="RR/min_comm")
    ax.bar(x + width/2, [t[3] for t in rows], width, label="LBG/greedy_balance")
    ax.axhline(1.0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=70, fontsize=7)
    ax.set_ylabel("e2e_speedup")
    ax.set_title("v3 — e2e_speedup per (policy, rate) under each Phase 3 overlap plan")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.v3_root, "v3_cross_plan_bars.png"), dpi=140)
    plt.close(fig)

    # ---------------- table ----------------------------------------------------
    lines = ["# v3 — comprehensive table\n"]
    lines.append("| plan | policy | rate | prefill_sp | e2e_sp | per-rank tok/s | F1 | F1 Δ |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for plan_name in plan_names:
        plan = plans[plan_name]
        b = plan["cells"]["baseline"]
        lines.append(f"| {plan_name} | baseline | 0.0 | 1.000 | 1.000 | "
                     f"{b['prefill_tok_s']['mean']:.0f} | {b['f1_mean']['mean']:.3f} | — |")
        for label in sorted(plan["relative_to_baseline"]):
            s = plan["cells"][label]
            r = plan["relative_to_baseline"][label]
            f1m = s["f1_mean"]["mean"]
            f1d = r["f1_delta_abs"]
            lines.append(f"| {plan_name} | {s['policy']} | {s['rate']} | "
                         f"{r['prefill_speedup']:.3f} | {r['e2e_speedup']:.3f} | "
                         f"{s['prefill_tok_s']['mean']:.0f} | {f1m:.3f} | "
                         f"{'%+.3f' % f1d if f1d is not None else '-'} |")
    open(os.path.join(args.v3_root, "v3_table.md"), "w").write("\n".join(lines))

    print(f"[plot_v3] wrote 5 PNGs + v3_table.md to {args.v3_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
