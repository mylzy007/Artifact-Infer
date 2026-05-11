#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Bar-chart comparison with 95% CIs for the four-scenario heavy run
(oracle / AIME real / synthetic CV=0.05 / synthetic CV=0.40), plus a
per-layer step-time plot showing how tightly clustered each scenario's
48-layer profile is around its mean.

Usage:
    python plot_heavy_comparison.py \\
      --label "Oracle uniform (CV=0)" --jsonl results/qwen3_heavy/oracle.jsonl \\
      --label "AIME real trace"       --jsonl results/qwen3_heavy/aime.jsonl \\
      --label "Synth CV=0.05"         --jsonl results/qwen3_heavy/cv_0p05.jsonl \\
      --label "Synth CV=0.40"         --jsonl results/qwen3_heavy/cv_0p40.jsonl \\
      --bar-output    figures/qwen3_heavy_bar.png \\
      --layers-output figures/qwen3_heavy_perlayer.png \\
      --title "Qwen3-30B-A3B, all 48 layers, 32 768 tokens, 5 trials x 10 iters, 8x4090"
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_trials(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    return [r for r in rows if "per_layer_mean_step_ms" in r]


def _ci95(vals):
    n = len(vals)
    if n < 2:
        return statistics.mean(vals) if vals else 0.0, 0.0
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals)
    return mu, 1.96 * sd / math.sqrt(n)


def _bar_figure(sources, output, title):
    n = len(sources)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    fig.suptitle(title, fontsize=12)

    means, errs, labels, log_cvs, phy_cvs = [], [], [], [], []
    for label, path in sources:
        trials = _load_trials(path)
        vals = [r["max_rank_mean_step_ms"] for r in trials]
        per_log = [statistics.mean(r["per_layer_logical_cv"])  for r in trials]
        per_phy = [statistics.mean(r["per_layer_physical_cv"]) for r in trials]
        mu, ci = _ci95(vals)
        means.append(mu); errs.append(ci); labels.append(label)
        log_cvs.append(statistics.mean(per_log)); phy_cvs.append(statistics.mean(per_phy))

    colors = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e", "#9467bd"]
    x = np.arange(n)
    bars = ax.bar(
        x, means, yerr=errs, capsize=8,
        color=[colors[i % len(colors)] for i in range(n)],
        edgecolor="black", linewidth=0.8, alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=9)
    ax.set_ylabel("total step time across all 48 layers (ms), 95% CI")
    ymin = min(m - e for m, e in zip(means, errs))
    ymax = max(m + e for m, e in zip(means, errs))
    pad = (ymax - ymin) * 0.4 + 5
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(True, axis="y", alpha=0.3)

    for bi, (m, e, lc, pc) in enumerate(zip(means, errs, log_cvs, phy_cvs)):
        ax.text(
            bi, m + e + (ymax - ymin) * 0.05,
            f"{m:.1f}\n±{e:.1f}\nlog_cv={lc:.2f}\nphy_cv={pc:.2f}",
            ha="center", va="bottom", fontsize=8.5,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    print(f"wrote {output}")


def _layers_figure(sources, output, title):
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(title, fontsize=12)
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e", "#9467bd"]

    for idx, (label, path) in enumerate(sources):
        trials = _load_trials(path)
        if not trials:
            continue
        # Average per-layer ms across all trials (each trial has its own
        # avg-over-iters per layer).
        arr = np.asarray([t["per_layer_mean_step_ms"] for t in trials])
        mean_per_layer = arr.mean(axis=0)
        std_per_layer = arr.std(axis=0)
        x = np.arange(arr.shape[1])
        c = colors[idx % len(colors)]
        ax.plot(x, mean_per_layer, color=c, label=label, linewidth=1.6)
        ax.fill_between(
            x,
            mean_per_layer - std_per_layer,
            mean_per_layer + std_per_layer,
            color=c, alpha=0.15, linewidth=0,
        )

    ax.set_xlabel("layer")
    ax.set_ylabel("per-layer step time (ms), shaded = ±1 std across trials")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    print(f"wrote {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", action="append", default=[])
    p.add_argument("--jsonl", action="append", type=Path, default=[])
    p.add_argument("--bar-output", type=Path, required=True)
    p.add_argument("--layers-output", type=Path, required=True)
    p.add_argument("--title", type=str, required=True)
    args = p.parse_args()
    if len(args.label) != len(args.jsonl):
        raise ValueError("--label / --jsonl must come in pairs")
    sources = list(zip(args.label, args.jsonl))
    _bar_figure(sources, args.bar_output, args.title)
    _layers_figure(sources, args.layers_output, args.title)


if __name__ == "__main__":
    main()
