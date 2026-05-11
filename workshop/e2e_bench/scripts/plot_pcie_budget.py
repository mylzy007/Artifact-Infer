#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot two diagnostics for the PCIe / EP cost analysis:

  1. Measured NCCL collective bandwidth as a function of payload size on
     this 8x4090 box, alongside theoretical PCIe Gen3 x8 unidirectional
     ceiling and the per-link p2p ceiling.

  2. Predicted vs measured per-layer step time for FlashInfer's all-to-all
     EP (world=8, M=4096/rank), broken down into "two all_to_all_singles"
     plus "compute + permute + overhead".

Outputs go to ``workshop/e2e_bench/figures/qwen3_pcie_budget.png``.

Reads:
    /home/yyx/personal/inference/vllm-bench/results/qwen3_heavy/nccl_bw.jsonl
    /home/yyx/personal/inference/vllm-bench/results/qwen3_heavy/flashinfer_native_ep.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PCIE_GEN3_X8_PEAK_GBPS = 8.0  # 7.88 GB/s effective; we use the round number


def _read_jsonl(p: Path) -> list[dict]:
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _bw_panel(ax, bw_rows: list[dict]) -> None:
    by_coll = defaultdict(list)
    for r in bw_rows:
        by_coll[r["coll"]].append(r)

    color_map = {
        "p2p_send_recv_0to1":      ("tab:blue",   "p2p send/recv (single pair, GPU0\u2194GPU1)"),
        "all_to_all_single":       ("tab:red",    "all_to_all_single (8 ranks)"),
        "all_gather_into_tensor":  ("tab:green",  "all_gather_into_tensor (8 ranks)"),
        "all_reduce":              ("tab:orange", "all_reduce (8 ranks)"),
    }
    for coll, rows in by_coll.items():
        if coll not in color_map:
            continue
        rows.sort(key=lambda r: r["nbytes"])
        x = [r["nbytes"] / 1024**2 for r in rows]
        y = [r["alg_bw_GBps"] for r in rows]
        c, lbl = color_map[coll]
        ax.plot(x, y, "o-", color=c, label=lbl, lw=1.6)

    ax.axhline(PCIE_GEN3_X8_PEAK_GBPS, color="black", ls="--", lw=1.0,
               label=f"PCIe Gen3 x8 peak ({PCIE_GEN3_X8_PEAK_GBPS:.1f} GB/s)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("payload size per rank (MiB)")
    ax.set_ylabel("achieved bandwidth (GB/s)")
    ax.set_title("Measured NCCL bandwidth on 8\u00d74090 (PCIe Gen3 x8, no NVLink, no P2P)")
    ax.set_ylim(0, max(PCIE_GEN3_X8_PEAK_GBPS + 1.0, max(r["alg_bw_GBps"] for r in bw_rows) + 1))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)


def _budget_panel(ax, bw_rows: list[dict], ep_rows: list[dict],
                  topk: int = 8, hidden: int = 2048,
                  inter: int = 768, num_layers: int = 48,
                  ) -> None:
    a2a = sorted(
        [r for r in bw_rows if r["coll"] == "all_to_all_single"],
        key=lambda r: r["nbytes"],
    )
    a2a_payloads = np.array([r["nbytes"] for r in a2a])
    a2a_means_ms = np.array([r["mean_ms"] for r in a2a])

    def predict_a2a_ms(nbytes: int) -> float:
        return float(np.interp(nbytes, a2a_payloads, a2a_means_ms))

    rows_to_plot = []
    for r in ep_rows:
        if r.get("impl") != "flashinfer_native_ep_alltoall":
            continue
        if r.get("autotune"):
            tag = f"world={r['world_size']}, M={r['tokens_per_rank']}, autotune"
        else:
            tag = f"world={r['world_size']}, M={r['tokens_per_rank']}"
        # Per-layer payload:
        # send buffer = M * K * H * 2 bytes (bf16)
        nb = r["tokens_per_rank"] * r["topk"] * r["hidden_size"] * 2
        a2a_ms = predict_a2a_ms(nb)
        # Per-layer compute:
        # FLOPs / slot = K * (gate+up+down) = 1*(2*H*I*2 + 2*I*H) = 6*H*I
        flops_per_layer = (
            r["tokens_per_rank"] * r["topk"] * 6 * r["hidden_size"] * r["moe_intermediate_size"]
        )
        compute_ms = flops_per_layer / (120e12) * 1e3  # 120 TFLOP/s autotuned single GPU
        measured_ms = r["max_rank_mean_step_ms"] / r["num_layers"]
        rows_to_plot.append({
            "tag": tag,
            "predicted_two_a2a_ms": 2 * a2a_ms,
            "predicted_compute_ms": compute_ms,
            "predicted_total_ms": 2 * a2a_ms + compute_ms,
            "measured_ms": measured_ms,
            "payload_MiB": nb / 1024**2,
        })

    if not rows_to_plot:
        return

    rows_to_plot.sort(key=lambda d: (d["payload_MiB"], "autotune" in d["tag"]))
    tags = [r["tag"] for r in rows_to_plot]
    pred_a2a = np.array([r["predicted_two_a2a_ms"] for r in rows_to_plot])
    pred_cmp = np.array([r["predicted_compute_ms"] for r in rows_to_plot])
    measured = np.array([r["measured_ms"] for r in rows_to_plot])
    overhead = measured - pred_a2a - pred_cmp

    x = np.arange(len(rows_to_plot))
    width = 0.4

    ax.bar(x - width/2, pred_a2a, width, label="predicted: 2 \u00d7 all_to_all_single",
           color="tab:red")
    ax.bar(x - width/2, pred_cmp, width, bottom=pred_a2a,
           label="predicted: GEMM (~120 TFLOP/s)", color="tab:gray")
    ax.bar(x + width/2, measured, width, label="measured per-layer step",
           color="tab:purple", alpha=0.85)

    for i, r in enumerate(rows_to_plot):
        ax.text(x[i] - width/2, r["predicted_total_ms"] + 1,
                f"{r['predicted_total_ms']:.1f}", ha="center", fontsize=8)
        ax.text(x[i] + width/2, r["measured_ms"] + 1,
                f"{r['measured_ms']:.1f}", ha="center", fontsize=8)
        gap_pct = (r["measured_ms"] - r["predicted_total_ms"]) / r["predicted_total_ms"] * 100
        ax.text(x[i], -3.5,
                f"\u0394 = {r['measured_ms'] - r['predicted_total_ms']:+.1f} ms "
                f"({gap_pct:+.0f}%)",
                ha="center", fontsize=7, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("per-layer step time (ms)")
    ax.set_title("FlashInfer all-to-all EP: predicted from PCIe budget vs measured")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bw", type=Path,
                   default=Path("/home/yyx/personal/inference/vllm-bench/results/qwen3_heavy/nccl_bw.jsonl"))
    p.add_argument("--ep", type=Path,
                   default=Path("/home/yyx/personal/inference/vllm-bench/results/qwen3_heavy/flashinfer_native_ep.jsonl"))
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "figures" / "qwen3_pcie_budget.png")
    args = p.parse_args()

    bw_rows = _read_jsonl(args.bw)
    ep_rows = _read_jsonl(args.ep)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    _bw_panel(axes[0], bw_rows)
    _budget_panel(axes[1], bw_rows, ep_rows)
    fig.suptitle("PCIe-only EP cost analysis (8\u00d74090, PCIe Gen3 x8, no NVLink/P2P)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
