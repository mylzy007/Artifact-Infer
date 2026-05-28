"""Cross-dataset comparison: v4 (LEval multidoc_qa) vs v5 (LongBench 2wikimqa).

Validates the v4 finding "tail_weight @ r=0.5 is best on accuracy" across a
second dataset (2wikimqa with very short factoid references).

Generates:
  - cross_dataset_recall_per_policy.png   : recall Δ from baseline per (policy, rate)
                                            with both datasets overlaid
  - cross_dataset_f1_per_policy.png       : same but LongBench official F1
  - cross_dataset_perf_per_policy.png     : prefill_speedup + e2e_speedup per cell
  - cross_dataset_table.md                : side-by-side complete table

Usage:
  python -m eval.drop.plot_v4_vs_v5 \\
      --v4-root eval_results/owner_local_ep_phase4_drop_longbench_v4 \\
      --v5-root eval_results/owner_local_ep_phase4_drop_longbench_v5 \\
      --out-dir eval_results/cross_dataset_v4v5
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


def load_summary(root: str):
    plans = {}
    for plan_dir in ("rr_mincomm", "lbg_greedybal"):
        p = os.path.join(root, plan_dir, "v2_summary.json")
        if os.path.exists(p):
            plans[plan_dir] = json.load(open(p))
    rescore_path = os.path.join(root, "official_rescore.json")
    rescore = json.load(open(rescore_path)) if os.path.exists(rescore_path) else {}
    return plans, rescore


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--v4-root", required=True)
    p.add_argument("--v5-root", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    v4_perf, v4_score = load_summary(args.v4_root)
    v5_perf, v5_score = load_summary(args.v5_root)

    if not v5_perf or not v5_score:
        print(f"[WARNING] v5 data incomplete at {args.v5_root}")

    datasets = {
        "v4: LEval multidoc_qa": (v4_perf, v4_score),
        "v5: LongBench 2wikimqa": (v5_perf, v5_score),
    }

    # collect all cells
    policies = set()
    rates = set()
    for ds_perf, _ in datasets.values():
        for plan in ds_perf.values():
            for label in plan["cells"]:
                if label == "baseline": continue
                if "_r" in label:
                    pol, rate = label.rsplit("_r", 1)
                    policies.add(pol); rates.add(float(rate))
    policies = sorted(policies)
    rates = sorted(rates)
    cmap = plt.get_cmap("tab10")
    policy_color = {pol: cmap(i) for i, pol in enumerate(policies)}

    # ---- 1) recall Δ per (policy, rate), per dataset (LBG plan) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax_idx, (ds_name, (ds_perf, ds_score)) in enumerate(datasets.items()):
        ax = axes[ax_idx]
        if "lbg_greedybal" not in ds_score or "baseline" not in ds_score["lbg_greedybal"]:
            ax.set_title(f"{ds_name}\n(NO DATA)")
            continue
        base_r = ds_score["lbg_greedybal"]["baseline"]["lb_recall"]["mean"]
        ax.axhline(0, color="black", linewidth=1.0)
        for pol in policies:
            xs, ys = [], []
            for r in rates:
                label = f"{pol}_r{r}"
                if label not in ds_score["lbg_greedybal"]: continue
                v = ds_score["lbg_greedybal"][label].get("lb_recall", {}).get("mean")
                if v is None: continue
                xs.append(r); ys.append(v - base_r)
            ax.plot(xs, ys, marker="o", color=policy_color[pol], label=pol, linewidth=2)
        ax.set_title(f"{ds_name}\n(baseline lb_recall={base_r:.3f})")
        ax.set_xlabel("drop_rate")
        if ax_idx == 0:
            ax.set_ylabel("lb_recall Δ vs baseline")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Cross-dataset validation — LongBench official recall Δ per policy (LBG plan)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cross_dataset_recall_per_policy.png"), dpi=140)
    plt.close(fig)

    # ---- 2) F1 Δ per (policy, rate), per dataset (LBG plan) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax_idx, (ds_name, (ds_perf, ds_score)) in enumerate(datasets.items()):
        ax = axes[ax_idx]
        if "lbg_greedybal" not in ds_score or "baseline" not in ds_score["lbg_greedybal"]:
            ax.set_title(f"{ds_name}\n(NO DATA)")
            continue
        base_f = ds_score["lbg_greedybal"]["baseline"]["lb_f1"]["mean"]
        ax.axhline(0, color="black", linewidth=1.0)
        for pol in policies:
            xs, ys = [], []
            for r in rates:
                label = f"{pol}_r{r}"
                if label not in ds_score["lbg_greedybal"]: continue
                v = ds_score["lbg_greedybal"][label].get("lb_f1", {}).get("mean")
                if v is None: continue
                xs.append(r); ys.append(v - base_f)
            ax.plot(xs, ys, marker="o", color=policy_color[pol], label=pol, linewidth=2)
        ax.set_title(f"{ds_name}\n(baseline lb_f1={base_f:.3f})")
        ax.set_xlabel("drop_rate")
        if ax_idx == 0:
            ax.set_ylabel("lb_f1 Δ vs baseline")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Cross-dataset validation — LongBench official F1 Δ per policy (LBG plan)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cross_dataset_f1_per_policy.png"), dpi=140)
    plt.close(fig)

    # ---- 3) prefill_sp + e2e_sp per dataset ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for col, (ds_name, (ds_perf, _)) in enumerate(datasets.items()):
        if "lbg_greedybal" not in ds_perf:
            continue
        plan = ds_perf["lbg_greedybal"]
        for row, key in enumerate(("prefill_speedup", "e2e_speedup")):
            ax = axes[row, col]
            ax.axhline(1.0, color="black", linewidth=1.0)
            for pol in policies:
                xs, ys = [], []
                for r in rates:
                    label = f"{pol}_r{r}"
                    if label not in plan["relative_to_baseline"]: continue
                    v = plan["relative_to_baseline"][label].get(key)
                    if v is None: continue
                    xs.append(r); ys.append(v)
                ax.plot(xs, ys, marker="o", color=policy_color[pol], label=pol, linewidth=2)
            if row == 0:
                ax.set_title(ds_name)
            ax.set_xlabel("drop_rate")
            if col == 0:
                ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 1:
                ax.legend(loc="best", fontsize=7)
    fig.suptitle("Cross-dataset — prefill & e2e speedup per policy (LBG plan)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cross_dataset_perf_per_policy.png"), dpi=140)
    plt.close(fig)

    # ---- 4) Side-by-side combined markdown ----
    lines = ["# v4 (LEval) vs v5 (LongBench 2wikimqa) cross-dataset comparison\n"]
    lines.append("Metric: LongBench official QA F1 + recall, on **LBG/greedy_balance plan**.\n")
    lines.append("| policy | rate | v4 prefill_sp | v4 e2e_sp | v4 recall Δ | v4 F1 Δ | v5 prefill_sp | v5 e2e_sp | v5 recall Δ | v5 F1 Δ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    v4p, v4s = datasets["v4: LEval multidoc_qa"]
    v5p, v5s = datasets["v5: LongBench 2wikimqa"]

    def get_cell(perf, score, label):
        out = {}
        if "lbg_greedybal" in perf and label in perf["lbg_greedybal"].get("relative_to_baseline", {}):
            r = perf["lbg_greedybal"]["relative_to_baseline"][label]
            out["prefill_sp"] = r.get("prefill_speedup")
            out["e2e_sp"] = r.get("e2e_speedup")
        if "lbg_greedybal" in score and label in score["lbg_greedybal"]:
            base = score["lbg_greedybal"].get("baseline", {})
            cur = score["lbg_greedybal"][label]
            for m in ("lb_recall", "lb_f1"):
                base_v = base.get(m, {}).get("mean")
                cur_v = cur.get(m, {}).get("mean")
                if base_v is not None and cur_v is not None:
                    out[m] = cur_v - base_v
        return out

    for pol in policies:
        for r in rates:
            label = f"{pol}_r{r}"
            v4c = get_cell(v4p, v4s, label)
            v5c = get_cell(v5p, v5s, label)
            cells = [pol, str(r)]
            for src in (v4c, v5c):
                for key in ("prefill_sp", "e2e_sp", "lb_recall", "lb_f1"):
                    v = src.get(key)
                    if v is None:
                        cells.append("—")
                    elif key in ("lb_recall", "lb_f1"):
                        cells.append(f"{v:+.3f}")
                    else:
                        cells.append(f"{v:.3f}")
            lines.append("| " + " | ".join(cells) + " |")
    open(os.path.join(args.out_dir, "cross_dataset_table.md"), "w").write("\n".join(lines))

    print(f"[plot_v4_vs_v5] wrote 3 PNGs + cross_dataset_table.md to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
