#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Visualise per-trial JSONL output of
``tests/kernels/moe/modular_kernel_tools/benchmark_eplb_multigpu.py``.

Two figures are produced:

1. **Per-layer plot** (``--per-layer-output``): one bar/line panel per
   input file showing per-layer step time, per-layer logical CV, and
   per-layer physical CV. Useful to compare a real-trace replay against
   one or more synthetic baselines side by side at the same model
   dimensions.
2. **CV-vs-time scatter** (``--scatter-output``): aggregated per-trial
   measurements ``(realized_physical_cv, max_rank_mean_step_ms)`` from
   every input file, with a separate marker per file. The realised
   physical CV is what the bench actually obtained on hardware; the time
   is the multi-GPU max over the iters' mean step times.

Usage:

    python plot_bench_results.py \\
        --label "real Qwen3.5 trace"   --jsonl results/qwen3p5_alllayers_v2/real_trace.jsonl \\
        --label "synthetic CV=0.05"    --jsonl results/qwen3p5_alllayers_v2/cv_sweep/kernel_cv_0p0500.jsonl \\
        --label "synthetic CV=0.15"    --jsonl results/qwen3p5_alllayers_v2/cv_sweep/kernel_cv_0p1500.jsonl \\
        --label "synthetic CV=0.40"    --jsonl results/qwen3p5_alllayers_v2/cv_sweep/kernel_cv_0p4000.jsonl \\
        --per-layer-output qwen3p5_per_layer.png \\
        --scatter-output   qwen3p5_scatter.png \\
        --title "Qwen3.5-35B-A3B, all 41 MoE layers, 16384 tokens, 8x4090"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # The bench appends a few aggregate / compare rows alongside
            # per-trial rows. We want only the per-trial rows for this
            # plot (they have the per-layer arrays).
            if "per_layer_mean_step_ms" not in row:
                continue
            out.append(row)
    return out


def _avg_per_layer(rows: list[dict], key: str) -> np.ndarray:
    arr = np.asarray([row[key] for row in rows], dtype=np.float64)
    return arr.mean(axis=0)


def _build_per_layer_figure(
    sources: list[tuple[str, Path]],
    output: Path,
    title: str,
) -> None:
    n_panels = len(sources)
    fig, axes = plt.subplots(
        n_panels, 3, figsize=(15, 2.6 * n_panels + 0.6), squeeze=False
    )
    fig.suptitle(title, fontsize=13)

    for row_i, (label, path) in enumerate(sources):
        rows = _read_jsonl(path)
        if not rows:
            for c in range(3):
                axes[row_i, c].set_title(f"{label}: no per-trial rows in {path.name}")
            continue
        per_layer_ms = _avg_per_layer(rows, "per_layer_mean_step_ms")
        per_layer_log = _avg_per_layer(rows, "per_layer_logical_cv")
        per_layer_phy = _avg_per_layer(rows, "per_layer_physical_cv")
        num_layers = len(per_layer_ms)
        x = np.arange(num_layers)

        ax = axes[row_i, 0]
        ax.bar(x, per_layer_ms, color="C0", alpha=0.85)
        ax.set_title(f"{label}: per-layer step time")
        ax.set_xlabel("layer")
        ax.set_ylabel("ms")
        ax.set_ylim(0, max(per_layer_ms) * 1.15)
        ax.grid(True, axis="y", alpha=0.3)
        # Annotate sum
        total = per_layer_ms.sum()
        ax.text(
            0.99, 0.97,
            f"sum = {total:.1f} ms\nmean = {per_layer_ms.mean():.2f} ms",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax = axes[row_i, 1]
        ax.plot(x, per_layer_log, color="C3", linewidth=1.8)
        ax.set_title(f"{label}: per-layer logical CV")
        ax.set_xlabel("layer")
        ax.set_ylabel("CV")
        ax.set_ylim(0, max(per_layer_log) * 1.2 + 0.05)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.99, 0.97,
            f"mean = {per_layer_log.mean():.3f}\nmax  = {per_layer_log.max():.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax = axes[row_i, 2]
        ax.plot(x, per_layer_phy, color="C0", linewidth=1.8)
        ax.set_title(f"{label}: per-layer physical CV")
        ax.set_xlabel("layer")
        ax.set_ylabel("CV")
        ymax = max(0.05, max(per_layer_phy) * 1.2 + 0.02)
        ax.set_ylim(0, ymax)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.99, 0.97,
            f"mean = {per_layer_phy.mean():.3f}\nmax  = {per_layer_phy.max():.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    print(f"wrote {output}")


def _build_scatter_figure(
    sources: list[tuple[str, Path]],
    output: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.suptitle(title, fontsize=13)

    cmap = plt.get_cmap("tab10")
    for idx, (label, path) in enumerate(sources):
        rows = _read_jsonl(path)
        if not rows:
            continue
        cvs = np.array([r["realized_physical_cv"] for r in rows])
        mss = np.array([r["max_rank_mean_step_ms"] for r in rows])
        ax.scatter(
            cvs, mss,
            label=label, color=cmap(idx % 10),
            s=80, edgecolors="black", linewidths=0.7, alpha=0.9,
        )

    ax.set_xlabel("realized physical CV (per-GPU token share)")
    ax.set_ylabel("max-rank mean step time across all layers (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label", action="append", default=[],
        help="Display label for the next --jsonl. Specify in pairs.",
    )
    parser.add_argument(
        "--jsonl", action="append", type=Path, default=[],
        help="Path to a per-trial bench JSONL output.",
    )
    parser.add_argument("--per-layer-output", type=Path, required=True)
    parser.add_argument("--scatter-output", type=Path, required=True)
    parser.add_argument("--title", type=str, default="MoE kernel benchmark")
    args = parser.parse_args()

    if len(args.label) != len(args.jsonl):
        raise ValueError(
            f"--label ({len(args.label)}) and --jsonl ({len(args.jsonl)}) "
            "must come in matched pairs"
        )
    if not args.jsonl:
        raise ValueError("at least one --label/--jsonl pair is required")

    sources = list(zip(args.label, args.jsonl))
    _build_per_layer_figure(sources, args.per_layer_output, args.title)
    _build_scatter_figure(sources, args.scatter_output, args.title)


if __name__ == "__main__":
    main()
