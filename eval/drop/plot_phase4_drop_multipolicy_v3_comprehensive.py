"""Comprehensive plots for the v3 sweep — cleaner views of parameter effects.

Generates 7 plots:
  1. v3_grid_heatmap.png         — 2×2 grid of (prefill_sp, e2e_sp, F1, tok/s) heatmaps,
                                    one per plan, showing policy×rate.
  2. v3_speedup_bars_with_err.png — grouped bar with std error bars: per (policy, rate)
                                    showing prefill_sp + e2e_sp, baseline = 1.0 line.
  3. v3_f1_with_noise_band.png    — F1 vs rate per policy with shaded ±2σ noise band.
  4. v3_throughput_absolute.png   — absolute per-rank tok/s for baseline vs each drop cell.
  5. v3_pareto_annotated.png      — Pareto with best-point annotations.
  6. v3_sensitivity.png           — sensitivity ranking: which axis moves which metric most?
  7. v3_plan_diff.png             — RR vs LBG plan: how much do they differ per cell?

Usage:
  python -m eval.drop.plot_v3_comprehensive --v3-root eval_results/owner_local_ep_phase4_drop_longbench_v3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(plan_dir: str) -> dict:
    return json.load(open(os.path.join(plan_dir, "v2_summary.json")))


def f1_sem(cell: dict) -> float:
    arr = cell.get("f1_per_prompt_flat") or []
    if len(arr) < 2:
        return 0.0
    return statistics.stdev(arr) / math.sqrt(len(arr))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--v3-root", required=True)
    args = p.parse_args()

    plan_dirs = {
        "RR/min_comm": os.path.join(args.v3_root, "rr_mincomm"),
        "LBG/greedy_balance": os.path.join(args.v3_root, "lbg_greedybal"),
    }
    plans = {k: load(v) for k, v in plan_dirs.items()}
    plan_names = list(plans.keys())

    policies = sorted({
        c["policy"]
        for plan in plans.values() for c in plan["cells"].values()
        if c["policy"] != "none"
    })
    rates = sorted({
        c["rate"]
        for plan in plans.values() for c in plan["cells"].values()
        if c["rate"] > 0
    })
    cmap = plt.get_cmap("tab10")
    policy_color = {pol: cmap(i) for i, pol in enumerate(policies)}

    # ===================================================================
    # 1) 2x2 grid of heatmaps per metric, per plan
    # ===================================================================
    # rows: metrics, cols: plans
    metric_specs = [
        ("prefill_speedup", "Prefill speedup", "RdYlGn", 0.85, 1.55),
        ("e2e_speedup",     "E2E speedup",      "RdYlGn", 0.90, 1.15),
        ("f1_mean",         "F1 (token-level)", "RdBu_r", 0.18, 0.24),
        ("prefill_tok_s",   "Per-rank tok/s",   "viridis", 700, 1100),
    ]
    fig, axes = plt.subplots(len(metric_specs), len(plan_names),
                             figsize=(5 * len(plan_names), 3.4 * len(metric_specs)))
    if len(plan_names) == 1:
        axes = axes[:, None]
    for ri, (key, title, cm, vmin, vmax) in enumerate(metric_specs):
        for ci, plan_name in enumerate(plan_names):
            plan = plans[plan_name]
            grid = np.full((len(policies), len(rates)), np.nan)
            for i, pol in enumerate(policies):
                for j, r in enumerate(rates):
                    label = f"{pol}_r{r}"
                    if label not in plan["cells"]:
                        continue
                    s = plan["cells"][label]
                    rel = plan["relative_to_baseline"].get(label, {})
                    if key == "prefill_speedup":
                        v = rel.get("prefill_speedup")
                    elif key == "e2e_speedup":
                        v = rel.get("e2e_speedup")
                    elif key == "f1_mean":
                        v = s["f1_mean"]["mean"] if s["f1_mean"] else None
                    elif key == "prefill_tok_s":
                        v = s["prefill_tok_s"]["mean"]
                    if v is not None:
                        grid[i, j] = v
            ax = axes[ri, ci]
            im = ax.imshow(grid, cmap=cm, aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(rates)))
            ax.set_xticklabels([f"{r:.1f}" for r in rates])
            ax.set_yticks(range(len(policies)))
            ax.set_yticklabels(policies, fontsize=8)
            for i in range(len(policies)):
                for j in range(len(rates)):
                    if np.isnan(grid[i, j]):
                        continue
                    txt = (f"{grid[i,j]:.3f}" if key not in ("prefill_tok_s",)
                           else f"{grid[i,j]:.0f}")
                    ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=8)
            if ri == 0:
                ax.set_title(plan_name)
            if ri == len(metric_specs) - 1:
                ax.set_xlabel("drop_rate")
            if ci == 0:
                ax.set_ylabel(title, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("v3 — Performance × Accuracy heatmaps (rows=metrics, cols=Phase-3 overlap plans)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.v3_root, "v3_grid_heatmap.png"), dpi=140)
    plt.close(fig)

    # ===================================================================
    # 2) Grouped bar with error: prefill_sp + e2e_sp per (policy, rate), per plan
    # ===================================================================
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    cells_order = [(pol, r) for pol in policies for r in rates]
    xlabels = [f"{p}\nr={r}" for (p, r) in cells_order]
    x = np.arange(len(cells_order))
    width = 0.4
    for ax_idx, plan_name in enumerate(plan_names):
        ax = axes[ax_idx]
        plan = plans[plan_name]
        base = plan["cells"]["baseline"]
        base_pt_std = base["prefill_time_s"]["std"]
        base_pt = base["prefill_time_s"]["mean"]
        prefill_sps, prefill_errs = [], []
        e2e_sps, e2e_errs = [], []
        for pol, r in cells_order:
            label = f"{pol}_r{r}"
            if label not in plan["cells"]:
                prefill_sps.append(np.nan); prefill_errs.append(0)
                e2e_sps.append(np.nan); e2e_errs.append(0)
                continue
            s = plan["cells"][label]; rel = plan["relative_to_baseline"][label]
            prefill_sps.append(rel["prefill_speedup"])
            # error propagation: speedup = base/drop; SE(speedup) ≈ speedup * sqrt((std_base/base)^2 + (std_drop/drop)^2)
            std_base_pt = base["prefill_time_s"]["std"]
            std_drop_pt = s["prefill_time_s"]["std"]
            mean_drop_pt = s["prefill_time_s"]["mean"]
            relerr = math.sqrt((std_base_pt/base_pt)**2 + (std_drop_pt/mean_drop_pt)**2)
            prefill_errs.append(rel["prefill_speedup"] * relerr / math.sqrt(s["n_batches"]))
            e2e_sps.append(rel["e2e_speedup"])
            std_base_e = base["e2e_total_time_s"]["std"]
            std_drop_e = s["e2e_total_time_s"]["std"]
            mean_drop_e = s["e2e_total_time_s"]["mean"]
            relerr_e = math.sqrt((std_base_e/base["e2e_total_time_s"]["mean"])**2 + (std_drop_e/mean_drop_e)**2)
            e2e_errs.append(rel["e2e_speedup"] * relerr_e / math.sqrt(s["n_batches"]))
        ax.bar(x - width/2, prefill_sps, width, yerr=prefill_errs, capsize=2,
               label="prefill_speedup", color="C0", alpha=0.85)
        ax.bar(x + width/2, e2e_sps, width, yerr=e2e_errs, capsize=2,
               label="e2e_speedup", color="C3", alpha=0.85)
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=70, fontsize=7)
        ax.set_ylabel("speedup")
        ax.set_title(f"{plan_name}  (baseline prefill={base_pt:.2f}s ± {base_pt_std:.2f})")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("v3 — Prefill / E2E speedup per (policy, rate) with batch-std error bars",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.v3_root, "v3_speedup_bars_with_err.png"), dpi=140)
    plt.close(fig)

    # ===================================================================
    # 3) F1 vs rate per policy, with ±2σ noise band shaded
    # ===================================================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax_idx, plan_name in enumerate(plan_names):
        ax = axes[ax_idx]
        plan = plans[plan_name]
        b = plan["cells"]["baseline"]
        base_f1 = b["f1_mean"]["mean"]
        base_sem = f1_sem(b)
        # noise band
        ax.axhspan(base_f1 - 2*base_sem, base_f1 + 2*base_sem,
                   color="gray", alpha=0.15, label=f"baseline ±2·SEM ({2*base_sem:.3f})")
        ax.axhline(base_f1, color="black", linewidth=1.0,
                   label=f"baseline F1={base_f1:.3f}")
        for pol in policies:
            xs, ys, errs = [], [], []
            for r in rates:
                label = f"{pol}_r{r}"
                if label not in plan["cells"]:
                    continue
                s = plan["cells"][label]
                xs.append(r)
                ys.append(s["f1_mean"]["mean"])
                errs.append(f1_sem(s))
            ax.errorbar(xs, ys, yerr=errs, marker="o", color=policy_color[pol],
                        label=pol, linewidth=2, capsize=4)
        ax.set_xlabel("drop_rate")
        if ax_idx == 0:
            ax.set_ylabel("F1 (token-level)")
        ax.set_title(plan_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    fig.suptitle("v3 — F1 with ±2·SEM noise band. Points inside the band are not significant.",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.v3_root, "v3_f1_with_noise_band.png"), dpi=140)
    plt.close(fig)

    # ===================================================================
    # 4) Absolute per-rank tok/s: baseline vs each cell
    # ===================================================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax_idx, plan_name in enumerate(plan_names):
        ax = axes[ax_idx]
        plan = plans[plan_name]
        base = plan["cells"]["baseline"]
        base_tps = base["prefill_tok_s"]["mean"]
        x = np.arange(len(rates))
        width = 0.16
        ax.axhline(base_tps, color="black", linestyle="--", linewidth=0.8,
                   label=f"baseline ({base_tps:.0f}/rank, ≈{8*base_tps:.0f} system)")
        for pi, pol in enumerate(policies):
            ys = []
            for r in rates:
                label = f"{pol}_r{r}"
                if label not in plan["cells"]:
                    ys.append(0); continue
                ys.append(plan["cells"][label]["prefill_tok_s"]["mean"])
            ax.bar(x + (pi - 2) * width, ys, width, label=pol, color=policy_color[pol])
        ax.set_xticks(x); ax.set_xticklabels([f"r={r}" for r in rates])
        if ax_idx == 0:
            ax.set_ylabel("per-rank prefill tok/s\n(× 8 for system total)")
        ax.set_title(plan_name)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("v3 — Absolute prefill throughput per rank (× 8 for total system)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.v3_root, "v3_throughput_absolute.png"), dpi=140)
    plt.close(fig)

    # ===================================================================
    # 5) Annotated Pareto
    # ===================================================================
    fig, ax = plt.subplots(figsize=(11, 7))
    markers = {plan_names[0]: "o", plan_names[1]: "s"}
    best_point = None  # (e2e_sp, f1, label)
    for plan_name in plan_names:
        plan = plans[plan_name]
        b = plan["cells"]["baseline"]
        base_f1 = b["f1_mean"]["mean"]
        ax.scatter([1.0], [base_f1], marker=markers[plan_name], s=400, color="black",
                   zorder=5, label=f"baseline ({plan_name})")
        for pol in policies:
            for r in rates:
                label = f"{pol}_r{r}"
                if label not in plan["cells"]:
                    continue
                sp = plan["relative_to_baseline"][label]["e2e_speedup"]
                f1 = plan["cells"][label]["f1_mean"]["mean"]
                size = 100 + 250 * r
                ax.scatter([sp], [f1], marker=markers[plan_name], s=size,
                           color=policy_color[pol], alpha=0.75, edgecolors="black", linewidth=0.5)
                # Track best: prioritize e2e_sp, then F1.
                if best_point is None or sp > best_point[0]:
                    best_point = (sp, f1, f"{plan_name}\n{pol}@r{r}")
    if best_point is not None:
        ax.annotate(
            f"BEST E2E\n{best_point[2]}\nspeedup={best_point[0]:.3f}\nF1={best_point[1]:.3f}",
            xy=(best_point[0], best_point[1]),
            xytext=(best_point[0] - 0.04, best_point[1] - 0.025),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            fontsize=9, color="red",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="red"),
        )
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=0.6)
    ax.set_xlabel("e2e_speedup vs baseline (higher is better)")
    ax.set_ylabel("F1 (token-level on LEval multidoc_qa, higher is better)")
    ax.set_title("v3 Pareto — circle=RR/min_comm, square=LBG/greedy_balance.\n"
                 "Point size scales with drop_rate (0.1, 0.3, 0.5).")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=policy_color[p],
                          markersize=12, label=p) for p in policies]
    handles.append(plt.Line2D([0], [0], marker="o", color="black", markersize=14, label="baseline (●○)"))
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.v3_root, "v3_pareto_annotated.png"), dpi=140)
    plt.close(fig)

    # ===================================================================
    # 6) Sensitivity: which axis moves which metric most?
    # ===================================================================
    # Compute std of each metric across each axis, marginalized over others.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax_idx, metric_key in enumerate(["prefill_speedup", "e2e_speedup", "f1_mean"]):
        ax = axes[ax_idx]
        # Collect all values
        rows = []
        for plan_name in plan_names:
            plan = plans[plan_name]
            for label, cell in plan["cells"].items():
                if cell["policy"] == "none":
                    continue
                if metric_key == "f1_mean":
                    v = cell["f1_mean"]["mean"] if cell["f1_mean"] else None
                else:
                    v = plan["relative_to_baseline"][label][metric_key]
                if v is None: continue
                rows.append((plan_name, cell["policy"], cell["rate"], v))
        # For each axis, compute std of (metric values within that axis-level)
        axes_to_test = [("plan", lambda r: r[0]),
                        ("policy", lambda r: r[1]),
                        ("rate", lambda r: r[2])]
        from collections import defaultdict
        sensitivities = {}
        for name, fn in axes_to_test:
            groups = defaultdict(list)
            for r in rows:
                groups[fn(r)].append(r[3])
            # Compute spread of means across groups (group-mean range)
            means = [statistics.mean(v) for v in groups.values() if v]
            if not means:
                sensitivities[name] = 0
                continue
            sensitivities[name] = max(means) - min(means)
        labels = list(sensitivities.keys())
        values = [sensitivities[k] for k in labels]
        bars = ax.bar(labels, values, color=["C2", "C1", "C0"])
        ax.set_title(metric_key)
        ax.set_ylabel("range of group means")
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("v3 sensitivity — which axis (plan / policy / rate) moves each metric most?\n"
                 "(higher bar = more impact)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.v3_root, "v3_sensitivity.png"), dpi=140)
    plt.close(fig)

    # ===================================================================
    # 7) Cross-plan delta: same (policy, rate) on RR vs LBG
    # ===================================================================
    fig, ax = plt.subplots(figsize=(13, 4.5))
    rows = []
    for pol in policies:
        for r in rates:
            label = f"{pol}_r{r}"
            rr = plans["RR/min_comm"]["relative_to_baseline"].get(label, {}).get("e2e_speedup")
            lbg = plans["LBG/greedy_balance"]["relative_to_baseline"].get(label, {}).get("e2e_speedup")
            if rr is None or lbg is None:
                continue
            rows.append((pol, r, lbg - rr))
    rows.sort(key=lambda t: (t[0], t[1]))
    x = np.arange(len(rows))
    ys = [t[2] for t in rows]
    colors = ["C2" if y > 0 else "C3" for y in ys]
    ax.bar(x, ys, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{p}\nr={r}" for (p, r, _) in rows], rotation=70, fontsize=7)
    ax.set_ylabel("e2e_speedup(LBG) − e2e_speedup(RR)")
    ax.set_title("v3 — plan choice impact: LBG − RR e2e_speedup per (policy, rate). \n"
                 "Most bars are near zero → plan choice rarely moves e2e by > 5%.")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.v3_root, "v3_plan_diff.png"), dpi=140)
    plt.close(fig)

    print(f"[plot_v3_comprehensive] wrote 7 PNGs to {args.v3_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
