#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Visualise a routing trace produced by ``capture_routing_trace.py``.

Produces a single PNG with four subplots:

1. **Per-layer expert load heatmap.** Each row is a MoE layer; each column
   is a logical expert. Cell colour is `routed_tokens / max(routed_tokens)`
   so layers are comparable even when token counts differ. Dark cells are
   "cold" experts; bright cells are hot experts.
2. **Per-layer CV (logical and physical).**
   - "logical" = CV of the per-expert token count within a layer.
   - "physical" = CV of the token count summed within each EP rank's
     contiguous expert slice (default world_size = 8).
3. **Per-layer top-K hot share.** Fraction of routed tokens captured by
   the top-1, top-5 and top-10 most-loaded experts in each layer. Higher
   = more concentrated routing.
4. **Per-layer normalised entropy.** Closer to 1 means routing is closer
   to uniform across experts.

Usage:
    python plot_routing_trace.py --trace path/to/trace.pt \\
        --output path/to/trace.png [--world-size 8]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _layer_histogram(ids: torch.Tensor, num_experts: int) -> np.ndarray:
    return (
        torch.bincount(ids.view(-1), minlength=num_experts)
        .to(dtype=torch.int64)
        .cpu()
        .numpy()
    )


def _cv(probs: np.ndarray) -> float:
    mean = probs.mean()
    if mean <= 0:
        return 0.0
    return float(np.sqrt(((probs - mean) ** 2).mean()) / mean)


def _entropy_normalised(probs: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(probs, eps, None)
    H = float(-(p * np.log(p)).sum())
    Hmax = math.log(len(probs))
    return H / Hmax if Hmax > 0 else 0.0


def _topk_share(counts: np.ndarray, k: int) -> float:
    tot = int(counts.sum())
    if tot <= 0:
        return 0.0
    sorted_desc = np.sort(counts)[::-1]
    return float(sorted_desc[:k].sum()) / float(tot)


def _physical_cv(counts: np.ndarray, world_size: int) -> float:
    num_experts = len(counts)
    assert num_experts % world_size == 0, (
        f"num_experts ({num_experts}) must be divisible by world_size "
        f"({world_size})"
    )
    block = num_experts // world_size
    by_rank = counts.reshape(world_size, block).sum(axis=1).astype(np.float64)
    return _cv(by_rank / max(by_rank.sum(), 1.0))


def _build_figure(
    trace_path: Path,
    output_path: Path,
    world_size: int,
) -> None:
    blob = torch.load(trace_path, map_location="cpu", weights_only=False)
    md = blob["metadata"]
    layers = blob["layers"]
    num_layers = len(layers)
    num_experts = int(md["num_experts"])
    topk = int(md["topk"])
    title_model = md.get("model_name", trace_path.stem)
    capture_mode = md.get("capture_mode", "?")
    global_n = int(md.get("global_num_tokens", layers[0]["topk_ids"].shape[0]))

    histos = np.zeros((num_layers, num_experts), dtype=np.int64)
    for i, item in enumerate(layers):
        histos[i] = _layer_histogram(item["topk_ids"], num_experts)

    layer_total = histos.sum(axis=1)
    safe_total = np.where(layer_total > 0, layer_total, 1)
    layer_probs = histos / safe_total[:, None]

    logical_cv = np.array([_cv(p) for p in layer_probs])
    logical_entropy = np.array([_entropy_normalised(p) for p in layer_probs])
    physical_cv = np.array([_physical_cv(h, world_size) for h in histos])
    top1 = np.array([_topk_share(h, 1) for h in histos])
    top5 = np.array([_topk_share(h, 5) for h in histos])
    top10 = np.array([_topk_share(h, 10) for h in histos])

    layer_max = histos.max(axis=1, keepdims=True)
    safe_max = np.where(layer_max > 0, layer_max, 1)
    heat = histos / safe_max  # in [0, 1] per row

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 9), gridspec_kw={"hspace": 0.35, "wspace": 0.30}
    )
    fig.suptitle(
        f"{title_model}  ({capture_mode}, {num_layers} layers, "
        f"{num_experts} experts, top-{topk}, {global_n} tokens)",
        fontsize=13,
    )

    # 1. Heatmap
    ax = axes[0, 0]
    im = ax.imshow(
        heat,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        origin="lower",
    )
    ax.set_xlabel("expert id")
    ax.set_ylabel("layer")
    ax.set_title("per-layer expert load (per-row max-normalised)")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("routed_tokens / max in row")
    # Mark EP rank boundaries.
    block = num_experts // world_size
    for r in range(1, world_size):
        ax.axvline(r * block - 0.5, color="white", linewidth=0.5, alpha=0.4)

    # 2. CV
    ax = axes[0, 1]
    ax.plot(np.arange(num_layers), logical_cv, label="logical CV (per-expert)",
            color="C3", linewidth=2)
    ax.plot(np.arange(num_layers), physical_cv, label=f"physical CV (per-rank, world={world_size})",
            color="C0", linewidth=2)
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("coefficient of variation")
    ax.set_title("per-layer load balance (lower = more uniform)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # 3. Top-K hot share
    ax = axes[1, 0]
    ax.plot(np.arange(num_layers), top1, label="top-1 share", linewidth=2)
    ax.plot(np.arange(num_layers), top5, label="top-5 share", linewidth=2)
    ax.plot(np.arange(num_layers), top10, label="top-10 share", linewidth=2)
    uniform_topk = topk / num_experts
    ax.axhline(uniform_topk, color="gray", linewidth=0.6, linestyle="--",
               label=f"uniform top-1 share (= topk/E = {uniform_topk:.3f})")
    ax.set_xlabel("layer")
    ax.set_ylabel("share of routed tokens")
    ax.set_title("hot-expert share per layer (concentration)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # 4. Entropy
    ax = axes[1, 1]
    ax.plot(np.arange(num_layers), logical_entropy, color="C2", linewidth=2,
            label="normalised entropy")
    ax.set_xlabel("layer")
    ax.set_ylabel("H / log(num_experts)")
    ax.set_title("per-layer normalised entropy (1.0 = uniform routing)")
    ax.set_ylim(min(0.5, float(logical_entropy.min()) - 0.02), 1.005)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    print(f"wrote {output_path}")

    summary = {
        "model": title_model,
        "capture_mode": capture_mode,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "topk": topk,
        "global_tokens": global_n,
        "logical_cv": {
            "min": float(logical_cv.min()),
            "max": float(logical_cv.max()),
            "mean": float(logical_cv.mean()),
        },
        "physical_cv": {
            "min": float(physical_cv.min()),
            "max": float(physical_cv.max()),
            "mean": float(physical_cv.mean()),
        },
        "top1_share": {
            "min": float(top1.min()),
            "max": float(top1.max()),
            "mean": float(top1.mean()),
        },
        "uniform_top1_share": uniform_topk,
        "entropy": {
            "min": float(logical_entropy.min()),
            "max": float(logical_entropy.max()),
            "mean": float(logical_entropy.mean()),
        },
    }
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--world-size",
        type=int,
        default=8,
        help="EP world size used to compute per-rank physical CV.",
    )
    args = parser.parse_args()
    _build_figure(args.trace, args.output, args.world_size)


if __name__ == "__main__":
    main()
