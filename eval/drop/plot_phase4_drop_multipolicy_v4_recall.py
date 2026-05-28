"""v4-specific plots using recall (NOT F1) as the main accuracy metric.

In v4 we stored the full generation text (vs v3's gen_text_head[:200]). On this
benchmark the model produces verbose CoT preambles followed by the actual
answer, while the reference is concise — so F1 is dragged down by precision
mismatch. recall is the meaningful metric: "did the model find the right
tokens somewhere in its output?"

Generates:
  v4_recall_vs_speedup_pareto.png   — main: x=prefill_sp, y=recall, per (policy, rate, plan)
  v4_recall_rate_per_plan.png       — recall vs rate per policy, with ±2σ band
  v4_pareto_e2e_vs_recall.png       — x=e2e_sp, y=recall
  v4_grid_perf_recall.png           — 3x2 grid: prefill_sp / e2e_sp / recall × 2 plans
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--v4-root", required=True)
    args = p.parse_args()

    plan_dirs = {
        "RR/min_comm": os.path.join(args.v4_root, "rr_mincomm"),
        "LBG/greedy_balance": os.path.join(args.v4_root, "lbg_greedybal"),
    }
    plans_perf = {k: json.load(open(os.path.join(v, "v2_summary.json"))) for k, v in plan_dirs.items()}
    rescore = json.load(open(os.path.join(args.v4_root, "v3_rescore.json")))

    plan_names = list(plans_perf.keys())
    policies = sorted({
        c["policy"] for plan in plans_perf.values() for c in plan["cells"].values()
        if c["policy"] != "none"
    })
    rates = sorted({
        c["rate"] for plan in plans_perf.values() for c in plan["cells"].values()
        if c["rate"] > 0
    })
    cmap = plt.get_cmap("tab10")
    policy_color = {pol: cmap(i) for i, pol in enumerate(policies)}

    def recall(plan_name, label):
        return rescore[plan_name].get(label, {}).get("recall", {}).get("mean")

    def recall_sem(plan_name, label):
        return rescore[plan_name].get(label, {}).get("recall", {}).get("sem", 0)

    # ==================================================================
    # 1) prefill_speedup vs recall — single chart, all data
    # ==================================================================
    fig, ax = plt.subplots(figsize=(11, 7))
    markers = {plan_names[0]: "o", plan_names[1]: "s"}
    for plan_name in plan_names:
        plan = plans_perf[plan_name]
        base_recall = recall(plan_name, "baseline")
        if base_recall is None: continue
        ax.scatter([1.0], [base_recall], marker=markers[plan_name], s=400, color="black",
                   zorder=5, label=f"baseline ({plan_name})")
        for pol in policies:
            for r in rates:
                label = f"{pol}_r{r}"
                if label not in plan["relative_to_baseline"]: continue
                sp = plan["relative_to_baseline"][label]["prefill_speedup"]
                rec = recall(plan_name, label)
                if rec is None: continue
                size = 100 + 250 * r
                ax.scatter([sp], [rec], marker=markers[plan_name], s=size,
                           color=policy_color[pol], alpha=0.75,
                           edgecolors="black", linewidth=0.6)
                if r == 0.5:
                    ax.annotate(f"{pol[:8]}", (sp, rec),
                                textcoords="offset points", xytext=(7, -2),
                                fontsize=7, alpha=0.8)
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=0.6)
    base_avg = np.mean([recall(p, "baseline") for p in plan_names if recall(p, "baseline") is not None])
    ax.axhline(base_avg, color="gray", linestyle=":", linewidth=0.6)
    ax.set_xlabel("prefill_speedup vs baseline (higher is better)")
    ax.set_ylabel("recall against reference (higher is better)")
    ax.set_title("v4 Pareto: recall vs prefill_speedup. Top-right is best.\n"
                 "● = RR/min_comm, ■ = LBG/greedy_balance. Point size scales with drop_rate.")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=policy_color[p],
                          markersize=12, label=p) for p in policies]
    handles.append(plt.Line2D([0], [0], marker="o", color="black", markersize=14, label="baseline"))
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.v4_root, "v4_recall_vs_speedup_pareto.png"), dpi=140)
    plt.close(fig)

    # ==================================================================
    # 2) recall vs rate per policy per plan — with ±2σ noise band
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax_idx, plan_name in enumerate(plan_names):
        ax = axes[ax_idx]
        base_r = recall(plan_name, "baseline")
        base_sem = recall_sem(plan_name, "baseline")
        ax.axhspan(base_r - 2*base_sem, base_r + 2*base_sem,
                   color="gray", alpha=0.15, label=f"baseline ±2·SEM ({2*base_sem:.3f})")
        ax.axhline(base_r, color="black", linewidth=1.0, label=f"baseline recall={base_r:.3f}")
        for pol in policies:
            xs, ys, errs = [], [], []
            for r in rates:
                label = f"{pol}_r{r}"
                v = recall(plan_name, label)
                if v is None: continue
                xs.append(r); ys.append(v); errs.append(recall_sem(plan_name, label))
            ax.errorbar(xs, ys, yerr=errs, marker="o", color=policy_color[pol],
                        label=pol, linewidth=2, capsize=4)
        ax.set_xlabel("drop_rate")
        ax.set_title(plan_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    axes[0].set_ylabel("recall (token-level)")
    fig.suptitle("v4 — recall vs drop_rate per policy, with ±2·SEM noise band\n"
                 "tail_weight @ r=0.5 is the only policy that rises above noise band in BOTH plans.",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.v4_root, "v4_recall_rate_per_plan.png"), dpi=140)
    plt.close(fig)

    # ==================================================================
    # 3) e2e_speedup vs recall — Pareto (best at top-right)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(11, 7))
    best_recall_point = None
    for plan_name in plan_names:
        plan = plans_perf[plan_name]
        base_r = recall(plan_name, "baseline")
        ax.scatter([1.0], [base_r], marker=markers[plan_name], s=400, color="black", zorder=5)
        for pol in policies:
            for r in rates:
                label = f"{pol}_r{r}"
                if label not in plan["relative_to_baseline"]: continue
                esp = plan["relative_to_baseline"][label]["e2e_speedup"]
                rec = recall(plan_name, label)
                if rec is None: continue
                size = 100 + 250 * r
                ax.scatter([esp], [rec], marker=markers[plan_name], s=size,
                           color=policy_color[pol], alpha=0.75,
                           edgecolors="black", linewidth=0.6)
                # Best = product of e2e_sp gain and recall gain
                score = (esp - 1.0) + 2 * (rec - base_r)
                if best_recall_point is None or score > best_recall_point[3]:
                    best_recall_point = (esp, rec, f"{plan_name}\n{pol}@r{r}", score)
    if best_recall_point is not None:
        ax.annotate(
            f"BEST PARETO\n{best_recall_point[2]}\n"
            f"e2e_sp={best_recall_point[0]:.3f}, recall={best_recall_point[1]:.3f}",
            xy=(best_recall_point[0], best_recall_point[1]),
            xytext=(best_recall_point[0] - 0.04, best_recall_point[1] - 0.04),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            fontsize=9, color="red",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="red"),
        )
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=0.6)
    ax.set_xlabel("e2e_speedup (higher is better)")
    ax.set_ylabel("recall (higher is better)")
    ax.set_title("v4 Pareto: e2e_speedup vs recall.  ● = RR, ■ = LBG.  Size ∝ drop_rate.")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=policy_color[p],
                          markersize=12, label=p) for p in policies]
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.v4_root, "v4_pareto_e2e_vs_recall.png"), dpi=140)
    plt.close(fig)

    # ==================================================================
    # 4) 3x2 grid: prefill_sp / e2e_sp / recall across policy×rate, per plan
    # ==================================================================
    metric_specs = [
        ("prefill_sp", "Prefill speedup", "RdYlGn", 0.95, 1.55, lambda p, l: plans_perf[p]["relative_to_baseline"].get(l, {}).get("prefill_speedup")),
        ("e2e_sp", "E2E speedup", "RdYlGn", 0.95, 1.20, lambda p, l: plans_perf[p]["relative_to_baseline"].get(l, {}).get("e2e_speedup")),
        ("recall", "Recall", "RdBu_r", 0.48, 0.64, lambda p, l: recall(p, l)),
    ]
    fig, axes = plt.subplots(len(metric_specs), len(plan_names),
                             figsize=(5 * len(plan_names), 3.5 * len(metric_specs)))
    for ri, (key, title, cm, vmin, vmax, getter) in enumerate(metric_specs):
        for ci, plan_name in enumerate(plan_names):
            grid = np.full((len(policies), len(rates)), np.nan)
            for i, pol in enumerate(policies):
                for j, r in enumerate(rates):
                    label = f"{pol}_r{r}"
                    v = getter(plan_name, label)
                    if v is not None: grid[i, j] = v
            ax = axes[ri, ci]
            im = ax.imshow(grid, cmap=cm, aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(rates)))
            ax.set_xticklabels([f"{r:.1f}" for r in rates])
            ax.set_yticks(range(len(policies)))
            ax.set_yticklabels(policies, fontsize=8)
            for i in range(len(policies)):
                for j in range(len(rates)):
                    if np.isnan(grid[i, j]): continue
                    ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center",
                            color="black", fontsize=8)
            if ri == 0:
                ax.set_title(plan_name)
            if ri == len(metric_specs) - 1:
                ax.set_xlabel("drop_rate")
            if ci == 0:
                ax.set_ylabel(title)
            fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("v4 — speedup + recall heatmaps (rows=metrics, cols=plans)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.v4_root, "v4_grid_perf_recall.png"), dpi=140)
    plt.close(fig)

    # ==================================================================
    # 5) Markdown summary table — combined perf + accuracy
    # ==================================================================
    lines = ["# v4 — final per-cell table (perf + recall + sub-metrics)\n"]
    lines.append("| plan | policy | rate | prefill_sp | e2e_sp | tok/s | recall | rec Δ | F1 | substr | ROUGE-L |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for plan_name in plan_names:
        plan = plans_perf[plan_name]
        b = plan["cells"]["baseline"]
        rb = rescore[plan_name]["baseline"]
        lines.append(f"| {plan_name} | baseline | 0.0 | 1.000 | 1.000 | "
                     f"{b['prefill_tok_s']['mean']:.0f} | "
                     f"{rb['recall']['mean']:.3f} | — | "
                     f"{rb['f1']['mean']:.3f} | {rb['substring']['mean']:.3f} | "
                     f"{rb['rouge_l']['mean']:.3f} |")
        for label in sorted(plan["relative_to_baseline"]):
            s = plan["cells"][label]
            r = plan["relative_to_baseline"][label]
            rs = rescore[plan_name].get(label, {})
            rec_v = rs.get("recall", {}).get("mean")
            rec_d = (rec_v - rb["recall"]["mean"]) if rec_v is not None else None
            lines.append(
                f"| {plan_name} | {s['policy']} | {s['rate']} | "
                f"{r['prefill_speedup']:.3f} | {r['e2e_speedup']:.3f} | "
                f"{s['prefill_tok_s']['mean']:.0f} | "
                f"{rec_v:.3f} | {'%+.3f' % rec_d if rec_d is not None else '-'} | "
                f"{rs.get('f1', {}).get('mean', float('nan')):.3f} | "
                f"{rs.get('substring', {}).get('mean', float('nan')):.3f} | "
                f"{rs.get('rouge_l', {}).get('mean', float('nan')):.3f} |"
            )
    open(os.path.join(args.v4_root, "v4_final_table.md"), "w").write("\n".join(lines))

    print(f"[plot_v4_final] wrote 4 PNGs + v4_final_table.md to {args.v4_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
