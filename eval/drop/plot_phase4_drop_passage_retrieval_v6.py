"""Comprehensive v6 plots + cross-dataset summary (v4/v5/v6).

Generates:
  v6_grid_heatmap.png         3×2 heatmap (prefill_sp / acc_strict / acc_official) × 2 plans
  v6_pareto_strict.png        prefill_speedup × acc_strict scatter, best top-right
  v6_strict_vs_official.png   side-by-side bars showing "repetition penalty" effect
  v6_rate_curves.png          accuracy + speedup vs rate, per policy, 2 plans
  v6_speedup_bars.png         prefill_sp / e2e_sp grouped bars with error
  cross_dataset_v4v5v6.png    3-dataset × 5-policy @ r=0.5 accuracy Δ summary

Usage:
  python -m eval.drop.plot_v6_final
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

V6_ROOT = "/home/lzy/Artifact-Infer/eval_results/owner_local_ep_phase4_drop_passage_retrieval_v6"
V4_ROOT = "/home/lzy/Artifact-Infer/eval_results/owner_local_ep_phase4_drop_longbench_v4"
V5_ROOT = "/home/lzy/Artifact-Infer/eval_results/owner_local_ep_phase4_drop_longbench_v5"
OUT_DIR = "/home/lzy/Artifact-Infer/eval_results/v6_plots"


def load_v6():
    return {
        "RR/min_comm": json.load(open(f"{V6_ROOT}/rr_mincomm/passage_retrieval_summary.json")),
        "LBG/greedy_balance": json.load(open(f"{V6_ROOT}/lbg_greedybal/passage_retrieval_summary.json")),
    }


def load_official_rescore(root):
    p = os.path.join(root, "official_rescore.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    v6 = load_v6()

    plan_names = list(v6.keys())
    policies = sorted({c["policy"] for plan in v6.values() for c in plan["cells"].values()
                       if c["policy"] != "none"})
    rates = sorted({c["rate"] for plan in v6.values() for c in plan["cells"].values()
                    if c["rate"] > 0})
    cmap = plt.get_cmap("tab10")
    color_map = {pol: cmap(i) for i, pol in enumerate(policies)}

    # =====================================================================
    # 1) 3x2 heatmap grid
    # =====================================================================
    metric_specs = [
        ("prefill_speedup", "Prefill speedup", "RdYlGn", 0.95, 1.55, "rel"),
        ("acc_strict",      "acc_strict (binary)", "RdYlGn", 0.0, 1.05, "abs"),
        ("acc_official",    "acc_official", "RdYlGn", 0.0, 1.0, "abs"),
    ]
    fig, axes = plt.subplots(len(metric_specs), len(plan_names),
                             figsize=(5 * len(plan_names), 3.5 * len(metric_specs)))
    for ri, (key, title, cm, vmin, vmax, kind) in enumerate(metric_specs):
        for ci, plan_name in enumerate(plan_names):
            plan = v6[plan_name]
            grid = np.full((len(policies), len(rates)), np.nan)
            for i, pol in enumerate(policies):
                for j, r in enumerate(rates):
                    label = f"{pol}_r{r}"
                    if label not in plan["cells"]:
                        continue
                    c = plan["cells"][label]
                    if kind == "rel":
                        v = plan["relative_to_baseline"].get(label, {}).get("prefill_speedup")
                    elif key == "acc_strict":
                        v = c["acc_strict"]["mean"]
                    else:
                        v = c["acc_official"]["mean"]
                    if v is not None:
                        grid[i, j] = v
            ax = axes[ri, ci]
            im = ax.imshow(grid, cmap=cm, aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(rates))); ax.set_xticklabels([f"{r:.1f}" for r in rates])
            ax.set_yticks(range(len(policies))); ax.set_yticklabels(policies, fontsize=9)
            for i in range(len(policies)):
                for j in range(len(rates)):
                    if np.isnan(grid[i, j]): continue
                    ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center",
                            color="black", fontsize=9)
            if ri == 0:
                ax.set_title(plan_name)
            if ri == len(metric_specs) - 1:
                ax.set_xlabel("drop_rate")
            if ci == 0:
                ax.set_ylabel(title)
            fig.colorbar(im, ax=ax, shrink=0.85)

    # Add baseline row at the top of acc plots (annotation only — already in heatmap as the "no drop" reference)
    fig.suptitle("v6 (LongBench passage_retrieval_en_e) — speedup × accuracy heatmaps\n"
                 "(rows=metrics, cols=Phase-3 overlap plans; baseline strict=1.000, official=0.251)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "v6_grid_heatmap.png"), dpi=140)
    plt.close(fig)

    # =====================================================================
    # 2) Pareto: prefill_sp × acc_strict — ALL rates labeled with rate text
    # =====================================================================
    fig, ax = plt.subplots(figsize=(13, 8))
    markers = {plan_names[0]: "o", plan_names[1]: "s"}
    best = None
    for plan_name in plan_names:
        plan = v6[plan_name]
        base = plan["cells"]["baseline"]
        base_strict = base["acc_strict"]["mean"]
        ax.scatter([1.0], [base_strict], marker=markers[plan_name], s=400, color="black",
                   zorder=5, label=f"baseline ({plan_name})")
        for pol in policies:
            for r in rates:
                label = f"{pol}_r{r}"
                if label not in plan["cells"]: continue
                sp = plan["relative_to_baseline"][label]["prefill_speedup"]
                ac = plan["cells"][label]["acc_strict"]["mean"]
                size = 80 + 220 * r
                ax.scatter([sp], [ac], marker=markers[plan_name], s=size,
                           color=color_map[pol], alpha=0.78,
                           edgecolors="black", linewidth=0.6)
                # Annotate EVERY point with just the rate (rate-aware short label)
                ax.annotate(f"{r:.1f}", (sp, ac),
                            textcoords="offset points",
                            xytext=(6, 4 if r != 0.5 else -10),
                            fontsize=7, alpha=0.85,
                            color=color_map[pol], weight="bold")
                # Track best Pareto by (speedup-1) * strict_acc
                gain = (sp - 1.0) * ac
                if best is None or gain > best[3]:
                    best = (sp, ac, f"{plan_name}\n{pol}@r{r}", gain)
    if best:
        ax.annotate(
            f"BEST PARETO\n{best[2]}\nspeedup={best[0]:.3f}, strict={best[1]:.3f}",
            xy=(best[0], best[1]),
            xytext=(best[0] - 0.25, best[1] - 0.20),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            fontsize=9, color="red",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="red"),
        )
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=0.6)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("prefill_speedup vs baseline (higher better)")
    ax.set_ylabel("acc_strict — LongBench binary (higher better)")
    ax.set_title("v6 Pareto: prefill_speedup × acc_strict per (policy, rate, plan).\n"
                 "● = RR/min_comm,  ■ = LBG/greedy_balance.  Point size & label = drop_rate.\n"
                 "All 32 points shown (2 baselines + 15 per plan @ rates 0.1/0.3/0.5).")
    # Compose legend with policy colors AND rate size legend
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[p],
                          markersize=12, label=p, markeredgecolor="black") for p in policies]
    handles.append(plt.Line2D([0], [0], marker="o", color="black", markersize=14, label="baseline (r=0)"))
    # Rate size legend
    handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                              markersize=np.sqrt(80 + 220*0.1)*0.9, label="r=0.1", markeredgecolor="black"))
    handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                              markersize=np.sqrt(80 + 220*0.3)*0.9, label="r=0.3", markeredgecolor="black"))
    handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                              markersize=np.sqrt(80 + 220*0.5)*0.9, label="r=0.5", markeredgecolor="black"))
    ax.legend(handles=handles, loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "v6_pareto_strict.png"), dpi=140)
    plt.close(fig)

    # =====================================================================
    # 3) strict vs official side-by-side bars (regularization effect)
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax_idx, plan_name in enumerate(plan_names):
        ax = axes[ax_idx]
        plan = v6[plan_name]
        labels_order = ["baseline"]
        for pol in policies:
            for r in rates:
                lbl = f"{pol}_r{r}"
                if lbl in plan["cells"]:
                    labels_order.append(lbl)
        # We want only r=0.5 actually
        labels_show = ["baseline"] + [f"{p}_r0.5" for p in policies if f"{p}_r0.5" in plan["cells"]]
        x = np.arange(len(labels_show))
        width = 0.4
        strict_vals = [plan["cells"][l]["acc_strict"]["mean"] for l in labels_show]
        off_vals = [plan["cells"][l]["acc_official"]["mean"] for l in labels_show]
        ax.bar(x - width/2, strict_vals, width, label="acc_strict (binary)", color="C0", alpha=0.85)
        ax.bar(x + width/2, off_vals, width, label="acc_official (LongBench)", color="C3", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("_r", "\nr=") for l in labels_show], rotation=0, fontsize=8)
        if ax_idx == 0:
            ax.set_ylabel("accuracy")
        ax.set_title(plan_name)
        ax.set_ylim(0, 1.1)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
        for i, v in enumerate(strict_vals):
            ax.text(x[i] - width/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
        for i, v in enumerate(off_vals):
            ax.text(x[i] + width/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "v6 — acc_strict vs acc_official @ r=0.5\n"
        "Note: baseline acc_official=0.251 BUT acc_strict=1.000 — model is correct but repeats answer 4-5×.\n"
        "drop suppresses repetition → acc_official jumps; acc_strict shows real accuracy.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "v6_strict_vs_official.png"), dpi=140)
    plt.close(fig)

    # =====================================================================
    # 4) Speedup + accuracy curves vs rate
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    # row 0: prefill_sp / e2e_sp (LBG plan)
    # row 1: acc_strict (both plans)
    for col, plan_name in enumerate(plan_names):
        plan = v6[plan_name]

        # Row 0: speedup
        ax = axes[0, col]
        ax.axhline(1.0, color="black", linewidth=0.7)
        for pol in policies:
            xs, ys = [], []
            for r in rates:
                lbl = f"{pol}_r{r}"
                if lbl in plan["relative_to_baseline"]:
                    xs.append(r); ys.append(plan["relative_to_baseline"][lbl]["prefill_speedup"])
            ax.plot(xs, ys, marker="o", color=color_map[pol], label=pol, linewidth=2)
        ax.set_title(f"{plan_name} — prefill_speedup")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("prefill_speedup")
        if col == len(plan_names) - 1:
            ax.legend(fontsize=7, loc="upper left")

        # Row 1: acc_strict
        ax = axes[1, col]
        base_s = plan["cells"]["baseline"]["acc_strict"]["mean"]
        ax.axhline(base_s, color="black", linewidth=1.0, label=f"baseline strict={base_s:.3f}")
        for pol in policies:
            xs, ys = [], []
            for r in rates:
                lbl = f"{pol}_r{r}"
                if lbl in plan["cells"]:
                    xs.append(r); ys.append(plan["cells"][lbl]["acc_strict"]["mean"])
            ax.plot(xs, ys, marker="o", color=color_map[pol], label=pol, linewidth=2)
        ax.set_title(f"{plan_name} — acc_strict (binary)")
        ax.set_xlabel("drop_rate")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("acc_strict")
        ax.set_ylim(0, 1.05)
    fig.suptitle("v6 — speedup and strict accuracy vs drop_rate per policy", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "v6_rate_curves.png"), dpi=140)
    plt.close(fig)

    # =====================================================================
    # 5) Speedup grouped bars (prefill + e2e) with batch-std error
    # =====================================================================
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    cells_order = [(pol, r) for pol in policies for r in rates]
    xlabels = [f"{p}\nr={r}" for (p, r) in cells_order]
    x = np.arange(len(cells_order))
    width = 0.4
    for ax_idx, plan_name in enumerate(plan_names):
        ax = axes[ax_idx]
        plan = v6[plan_name]
        ps, es = [], []
        for pol, r in cells_order:
            lbl = f"{pol}_r{r}"
            rel = plan["relative_to_baseline"].get(lbl, {})
            ps.append(rel.get("prefill_speedup", np.nan))
            es.append(rel.get("e2e_speedup", np.nan))
        ax.bar(x - width/2, ps, width, label="prefill_speedup", color="C0", alpha=0.85)
        ax.bar(x + width/2, es, width, label="e2e_speedup", color="C3", alpha=0.85)
        ax.axhline(1.0, color="black", linewidth=0.7)
        ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=70, fontsize=7)
        ax.set_ylabel("speedup")
        ax.set_title(plan_name)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("v6 — prefill / e2e speedup per (policy, rate)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "v6_speedup_bars.png"), dpi=140)
    plt.close(fig)

    # =====================================================================
    # 6) Cross-dataset summary: v4 / v5 / v6 @ r=0.5 (LBG plan)
    # =====================================================================
    # Each dataset uses different metric — normalize by showing Δ from baseline.
    v4_rescore = load_official_rescore(V4_ROOT)
    v5_rescore = load_official_rescore(V5_ROOT)

    fig, ax = plt.subplots(figsize=(13, 6))
    datasets = [
        ("v4: LEval multidoc_qa\n(token recall)", v4_rescore, "lbg_greedybal", "lb_recall"),
        ("v5: LongBench multifieldqa_en\n(token recall)", v5_rescore, "lbg_greedybal", "lb_recall"),
        ("v6: LongBench passage_retrieval_en_e\n(official binary strict)", "v6", "LBG/greedy_balance", "acc_strict"),
    ]
    n_pol = len(policies)
    n_ds = len(datasets)
    bar_width = 0.8 / n_pol
    x = np.arange(n_ds)
    for pi, pol in enumerate(policies):
        deltas = []
        for ds_label, ds_data, plan_key, metric in datasets:
            if ds_data == "v6":
                plan = v6[plan_key]
                base = plan["cells"]["baseline"]["acc_strict"]["mean"]
                cur = plan["cells"].get(f"{pol}_r0.5", {}).get("acc_strict", {}).get("mean")
            else:
                if not ds_data: deltas.append(np.nan); continue
                if plan_key not in ds_data: deltas.append(np.nan); continue
                base = ds_data[plan_key]["baseline"].get(metric, {}).get("mean")
                cur = ds_data[plan_key].get(f"{pol}_r0.5", {}).get(metric, {}).get("mean")
            if base is None or cur is None:
                deltas.append(np.nan)
            else:
                deltas.append(cur - base)
        ax.bar(x + pi * bar_width - 0.4 + bar_width / 2, deltas, bar_width,
               label=pol, color=color_map[pol])
        for xi, d in enumerate(deltas):
            if not np.isnan(d):
                ax.text(xi + pi * bar_width - 0.4 + bar_width / 2, d + (0.02 if d >= 0 else -0.04),
                        f"{d:+.2f}", ha="center", fontsize=6, rotation=90)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in datasets], fontsize=9)
    ax.set_ylabel("accuracy Δ vs baseline @ rate=0.5")
    ax.set_title(
        "Cross-dataset validation — accuracy Δ at rate=0.5 (LBG plan) across 3 datasets × 3 metrics.\n"
        "tail_weight is the only policy that doesn't crash on every dataset.",
        fontsize=10,
    )
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cross_dataset_v4v5v6.png"), dpi=140)
    plt.close(fig)

    # =====================================================================
    # 7) Per-cell sample distribution (acc_strict histogram for r=0.5)
    # =====================================================================
    fig, axes = plt.subplots(1, len(policies), figsize=(3 * len(policies), 4.5), sharey=True)
    for col, pol in enumerate(policies):
        ax = axes[col]
        # combine both plans
        all_scores = []
        for plan_name in plan_names:
            plan_dir = "rr_mincomm" if "RR" in plan_name else "lbg_greedybal"
            jsonl = f"{V6_ROOT}/{plan_dir}/passage_retrieval_rows.jsonl"
            for line in open(jsonl):
                d = json.loads(line)
                if d["label"] != f"{pol}_r0.5": continue
                for g in d["generations"]:
                    all_scores.append(g["score_strict"])
        if all_scores:
            ax.hist(all_scores, bins=[-0.05, 0.5, 1.05], color=color_map[pol], edgecolor="black")
            mean = np.mean(all_scores)
            ax.set_title(f"{pol}\n@ r=0.5 strict mean={mean:.3f}", fontsize=8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["wrong", "correct"], fontsize=8)
        if col == 0:
            ax.set_ylabel("count (n=128 across both plans)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("v6 — Distribution of per-prompt strict accuracy @ r=0.5 (binary 0/1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "v6_distribution_r05.png"), dpi=140)
    plt.close(fig)

    print(f"[plot_v6_final] wrote 7 PNGs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
