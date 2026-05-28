"""Re-score v3 generations with multiple accuracy metrics.

The bench saved (request_id, gen_text_head[:200], f1, ref_len) per prompt but
did NOT save the full reference_answer. We re-link by request_id against the
original dataset to recover reference_answer text, then compute:
  - F1            (already in bench; recomputed for sanity)
  - recall        (|gen∩ref| / |ref|): friendlier to short generations
  - substring     (1 if every ref token appears in gen, else 0)
  - exact_match_norm (1 if normalized gen == normalized ref)
  - rouge_l_f1    (LCS-based F1; standard for QA)

Writes:
  v3_rescore.json             — per-cell aggregated metrics
  v3_rescore_table.md         — markdown table
  v3_accuracy_metrics.png     — multi-metric per-policy plot
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import string
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")


def norm_tokens(s: str) -> list[str]:
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    return [t for t in s.split() if t]


def f1(pred: str, ref: str) -> float:
    p, r = norm_tokens(pred), norm_tokens(ref)
    if not p or not r: return 0.0
    pc, rc = defaultdict(int), defaultdict(int)
    for t in p: pc[t] += 1
    for t in r: rc[t] += 1
    common = sum(min(c, rc[t]) for t, c in pc.items())
    if not common: return 0.0
    P = common / sum(pc.values())
    R = common / sum(rc.values())
    return 2 * P * R / (P + R)


def recall(pred: str, ref: str) -> float:
    p, r = set(norm_tokens(pred)), set(norm_tokens(ref))
    if not r: return 0.0
    return len(p & r) / len(r)


def substring(pred: str, ref: str) -> float:
    p, r = set(norm_tokens(pred)), set(norm_tokens(ref))
    if not r: return 0.0
    return 1.0 if r.issubset(p) else 0.0


def exact_match_norm(pred: str, ref: str) -> float:
    return 1.0 if norm_tokens(pred) == norm_tokens(ref) else 0.0


def rouge_l(pred: str, ref: str) -> float:
    p, r = norm_tokens(pred), norm_tokens(ref)
    if not p or not r: return 0.0
    # LCS length via DP
    m, n = len(p), len(r)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if p[i] == r[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    if not lcs: return 0.0
    P = lcs / m; R = lcs / n
    return 2 * P * R / (P + R)


METRICS = {"f1": f1, "recall": recall, "substring": substring,
           "exact_match": exact_match_norm, "rouge_l": rouge_l}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--v3-root", required=True)
    p.add_argument("--dataset", default="/home/lzy/datasets/moe_benchmarks/prepared/leval.Generation_multidoc_qa.custom.jsonl")
    args = p.parse_args()

    # Build {request_id: reference_answer} as fallback
    ref_map = {}
    for line in open(args.dataset):
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        rid = d.get("request_id")
        ref_map[rid] = str(d.get("reference_answer", ""))
    print(f"[rescore] {len(ref_map)} reference answers loaded (fallback)")

    plans = {
        "RR/min_comm": "rr_mincomm",
        "LBG/greedy_balance": "lbg_greedybal",
    }

    all_results = {}
    for plan_name, plan_dir in plans.items():
        rows_path = os.path.join(args.v3_root, plan_dir, "v2_rows.jsonl")
        cell_scores = defaultdict(lambda: defaultdict(list))  # cell_scores[label][metric_name] = [...]
        for line in open(rows_path):
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            label = d["label"]
            for g in d.get("generations", []):
                # v4 stores full text under gen_text / ref_text;
                # v3 stored only gen_text_head[:200] and needed ref_map fallback.
                pred = g.get("gen_text") or g.get("gen_text_head", "")
                rid = g.get("request_id")
                ref = g.get("ref_text") or ref_map.get(rid, "")
                if not ref: continue
                for mname, fn in METRICS.items():
                    cell_scores[label][mname].append(fn(pred, ref))

        # aggregate
        out = {}
        for label, ms in cell_scores.items():
            out[label] = {}
            for mname, vals in ms.items():
                out[label][mname] = {
                    "mean": statistics.mean(vals) if vals else None,
                    "sem": statistics.stdev(vals)/(len(vals)**0.5) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
        all_results[plan_name] = out

    # Save JSON
    with open(os.path.join(args.v3_root, "v3_rescore.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Markdown table
    lines = ["# v3 — multi-metric re-score\n"]
    lines.append("Generation truncated to 200 chars (`gen_text_head[:200]`). "
                 "Reference is full LEval `reference_answer`. n=80 per cell.\n")
    lines.append("| plan | policy | rate | F1 | recall | substring | exact | ROUGE-L |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for plan_name, cells in all_results.items():
        for label in sorted(cells):
            c = cells[label]
            row = [plan_name, label.rsplit("_", 1)[0] if "_r" in label else label,
                   label.rsplit("_r", 1)[1] if "_r" in label else "—"]
            row += [f"{c[m]['mean']:.3f}" if c[m]['mean'] is not None else "—"
                    for m in ("f1", "recall", "substring", "exact_match", "rouge_l")]
            lines.append("| " + " | ".join(row) + " |")
    open(os.path.join(args.v3_root, "v3_rescore_table.md"), "w").write("\n".join(lines))

    # Plot: per plan, per metric, lines per policy
    metric_order = ["f1", "recall", "substring", "rouge_l"]
    fig, axes = plt.subplots(2, len(metric_order), figsize=(4 * len(metric_order), 8), sharex=True)
    rates = [0.1, 0.3, 0.5]
    policies_set = set()
    for cells in all_results.values():
        for label in cells:
            if "_r" not in label: continue
            policies_set.add(label.rsplit("_r", 1)[0])
    policies = sorted(policies_set)
    cmap = plt.get_cmap("tab10")
    policy_color = {pol: cmap(i) for i, pol in enumerate(policies)}

    for col, mname in enumerate(metric_order):
        for row, (plan_name, cells) in enumerate(all_results.items()):
            ax = axes[row, col]
            base = cells.get("baseline", {}).get(mname, {}).get("mean")
            base_sem = cells.get("baseline", {}).get(mname, {}).get("sem", 0)
            if base is not None:
                ax.axhspan(base - 2 * base_sem, base + 2 * base_sem, color="gray", alpha=0.15)
                ax.axhline(base, color="black", linewidth=1.0)
            for pol in policies:
                xs, ys, errs = [], [], []
                for r in rates:
                    label = f"{pol}_r{r}"
                    if label not in cells: continue
                    m = cells[label].get(mname, {})
                    if m.get("mean") is None: continue
                    xs.append(r)
                    ys.append(m["mean"])
                    errs.append(m["sem"])
                ax.errorbar(xs, ys, yerr=errs, marker="o", color=policy_color[pol],
                            label=pol, linewidth=1.8, capsize=3)
            if row == 0:
                ax.set_title(mname)
            if col == 0:
                ax.set_ylabel(plan_name)
            if row == 1:
                ax.set_xlabel("drop_rate")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == len(metric_order) - 1:
                ax.legend(loc="lower left", fontsize=6, framealpha=0.85)
    fig.suptitle("v3 multi-metric accuracy — 4 metrics × 2 plans × 5 policies. "
                 "Shaded band = baseline ±2·SEM (95% noise).",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.v3_root, "v3_accuracy_metrics.png"), dpi=140)
    plt.close(fig)

    # Print compact summary
    print("\n=== Multi-metric baseline values ===")
    print(f"{'plan':<22}{'F1':>9}{'recall':>9}{'substr':>9}{'exact':>9}{'ROUGE-L':>9}")
    for plan_name, cells in all_results.items():
        b = cells.get("baseline", {})
        line = f"{plan_name:<22}"
        for m in ("f1", "recall", "substring", "exact_match", "rouge_l"):
            v = b.get(m, {}).get("mean")
            line += f"{v:>9.3f}" if v is not None else f"{'—':>9}"
        print(line)

    print(f"\n[rescore] wrote v3_rescore.json + v3_rescore_table.md + v3_accuracy_metrics.png "
          f"to {args.v3_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
