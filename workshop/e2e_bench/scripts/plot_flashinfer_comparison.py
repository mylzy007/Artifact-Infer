#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare FlashInfer ``cutlass_fused_moe`` against the other single-GPU
MoE kernels we've measured on Qwen3-30B-A3B (HF v4 / v5 native, vLLM
Triton fused @ world=1) at two token-counts (M=4096 and M=32768).

Reads:
    - flashinfer_qwen3_M{4096,32768}.jsonl
    - hf{,5}_moe_qwen3_M{4096,32768}.jsonl
    - world1_oracle_perRankMatched.jsonl  (vLLM Triton @ M=4096)
    - world1_oracle_fullWorkload.jsonl    (vLLM Triton @ M=32768)

Writes:
    - figures/qwen3_flashinfer_vs_others.png
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _parse_hf_records(records: list[dict], fallback_version: str = "4") -> dict[tuple[str, int], dict]:
    """Returns {(impl_label, M): {mean, std}} from benchmark_hf_moe.py output."""
    out: dict[tuple[str, int], dict] = {}
    for r in records:
        ver_full = r.get("transformers_version") or fallback_version
        ver = str(ver_full).split(".")[0]
        impl = r.get("experts_implementation") or "eager"
        label = f"HF v{ver} {impl}"
        key = (label, int(r["tokens"]))
        out[key] = {
            "mean": float(r["total_step_mean_ms"]),
            "std": float(r["total_step_std_ms"]),
            "tflops": float(r["achieved_tflops_per_s"]),
        }
    return out


def _parse_flashinfer(records: list[dict]) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for r in records:
        suffix = "+autotune" if r.get("autotune") else ""
        label = f"FlashInfer CUTLASS ({r['routing']}{suffix})"
        key = (label, int(r["tokens"]))
        out[key] = {
            "mean": float(r["total_step_mean_ms"]),
            "std": float(r["total_step_std_ms"]),
            "tflops": float(r["achieved_tflops_per_s"]),
        }
    return out


def _parse_vllm_world1(records: list[dict]) -> dict[tuple[str, int], dict]:
    """vLLM-style record: max_rank_mean_step_ms, tokens_per_rank=[M]."""
    bucket: dict[int, list[float]] = {}
    for r in records:
        if r.get("world_size") != 1:
            continue
        tpr = r.get("tokens_per_rank", [])
        if len(tpr) != 1:
            continue
        bucket.setdefault(int(tpr[0]), []).append(float(r["max_rank_mean_step_ms"]))
    out: dict[tuple[str, int], dict] = {}
    label = "vLLM Triton (world=1)"
    for M, vals in bucket.items():
        out[(label, M)] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "tflops": None,
        }
    return out


def _parse_vllm_world8_oracle(records: list[dict]) -> dict[tuple[str, int], dict]:
    """vLLM oracle world=8: same JSON schema as world=1 but len(tpr)==world_size."""
    out: dict[tuple[str, int], dict] = {}
    by_M: dict[int, list[float]] = {}
    for r in records:
        ws = r.get("world_size")
        if ws is None or ws < 2:
            continue
        if r.get("routing_pattern") != "oracle_uniform":
            continue
        tpr = r.get("tokens_per_rank", [])
        if not tpr:
            continue
        M_total = sum(int(t) for t in tpr)
        by_M.setdefault(M_total, []).append(float(r["max_rank_mean_step_ms"]))
    label = "vLLM all2all+Triton (world=8)"
    for M, vals in by_M.items():
        out[(label, M)] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "tflops": None,
        }
    return out


def _parse_native_ep(records: list[dict]) -> dict[tuple[str, int], dict]:
    """HF / FlashInfer native-EP records use total_tokens for the M key."""
    out: dict[tuple[str, int], dict] = {}
    for r in records:
        impl = r.get("impl", "")
        ws = int(r.get("world_size", 0))
        autotune = bool(r.get("autotune", False))
        M = int(r.get("total_tokens", 0))
        if not M or not ws:
            continue
        if "flashinfer" in impl:
            label = f"FlashInfer all2all (world={ws})"
            if autotune:
                label += "+autotune"
        elif "hf_v5_native_ep" in impl:
            label = f"HF v5 native EP allgather (world={ws})"
        else:
            label = f"{impl} (world={ws})"
        out[(label, M)] = {
            "mean": float(r["max_rank_mean_step_ms"]),
            "std": float(r.get("max_rank_std_step_ms", 0.0)),
            "tflops": float(r.get("achieved_tflops_per_s", 0.0)) or None,
        }
    return out


COLORS = {
    "vLLM Triton (world=1)": "#2c7fb8",
    "HF v4 eager": "#fee08b",
    "HF v5 eager": "#f46d43",
    "HF v5 grouped_mm": "#d73027",
    "FlashInfer CUTLASS (oracle_uniform)": "#1a9850",
    "FlashInfer CUTLASS (softmax)": "#66bd63",
    "FlashInfer CUTLASS (oracle_uniform+autotune)": "#006837",
    # Multi-rank EP variants
    "vLLM all2all+Triton (world=8)": "#08519c",
    "FlashInfer all2all (world=8)": "#a50f15",
    "FlashInfer all2all (world=8)+autotune": "#67000d",
    "FlashInfer all2all (world=4)": "#cb181d",
    "HF v5 native EP allgather (world=8)": "#969696",
    "HF v5 native EP allgather (world=4)": "#bdbdbd",
}


# Order from slowest to fastest (visual: bottom to top in bar chart).
PREFERRED_ORDER = [
    "HF v5 eager",
    "HF v4 eager",
    "HF v5 grouped_mm",
    "FlashInfer CUTLASS (softmax)",
    "FlashInfer CUTLASS (oracle_uniform)",
    "vLLM Triton (world=1)",
    "FlashInfer CUTLASS (oracle_uniform+autotune)",
]

# Order for the multi-rank EP comparison (world=8, M_total=32768).
EP_ORDER = [
    "HF v5 native EP allgather (world=8)",
    "HF v5 native EP allgather (world=4)",
    "FlashInfer all2all (world=8)+autotune",
    "FlashInfer all2all (world=4)",
    "FlashInfer all2all (world=8)",
    "vLLM all2all+Triton (world=8)",
    "FlashInfer CUTLASS (oracle_uniform+autotune)",  # single-GPU reference
    "vLLM Triton (world=1)",                          # single-GPU reference
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path,
                   default=Path("/home/yyx/personal/inference/vllm-bench/results/qwen3_heavy"))
    p.add_argument("--out", type=Path,
                   default=Path("workshop/e2e_bench/figures/qwen3_flashinfer_vs_others.png"))
    args = p.parse_args()

    rd = args.results_dir
    data: dict[tuple[str, int], dict] = {}
    data.update(_parse_hf_records(_read_jsonl(rd / "hf_moe_qwen3_M4096.jsonl"), "4"))
    data.update(_parse_hf_records(_read_jsonl(rd / "hf_moe_qwen3_M32768.jsonl"), "4"))
    data.update(_parse_hf_records(_read_jsonl(rd / "hf5_moe_qwen3_M4096.jsonl"), "5"))
    data.update(_parse_hf_records(_read_jsonl(rd / "hf5_moe_qwen3_M32768.jsonl"), "5"))
    data.update(_parse_flashinfer(_read_jsonl(rd / "flashinfer_qwen3_M4096.jsonl")))
    data.update(_parse_flashinfer(_read_jsonl(rd / "flashinfer_qwen3_M32768.jsonl")))
    data.update(_parse_vllm_world1(_read_jsonl(rd / "world1_oracle_perRankMatched.jsonl")))
    data.update(_parse_vllm_world1(_read_jsonl(rd / "world1_oracle_fullWorkload.jsonl")))
    data.update(_parse_vllm_world8_oracle(_read_jsonl(rd / "oracle.jsonl")))
    data.update(_parse_native_ep(_read_jsonl(rd / "hf_native_ep.jsonl")))
    data.update(_parse_native_ep(_read_jsonl(rd / "flashinfer_native_ep.jsonl")))

    Ms = sorted({k[1] for k in data.keys()})
    labels = [lab for lab in PREFERRED_ORDER
              if any(k[0] == lab for k in data.keys())]

    fig, axes = plt.subplots(1, len(Ms), figsize=(4 + 4 * len(Ms), 5),
                             sharey=False)
    if len(Ms) == 1:
        axes = [axes]

    for ax, M in zip(axes, Ms):
        means, stds, colors, names = [], [], [], []
        for lab in labels:
            if (lab, M) not in data:
                continue
            d = data[(lab, M)]
            means.append(d["mean"])
            stds.append(d["std"])
            colors.append(COLORS.get(lab, "#888888"))
            names.append(lab)

        y = np.arange(len(names))
        ax.barh(y, means, xerr=stds, color=colors, edgecolor="black",
                linewidth=0.5, error_kw={"elinewidth": 1.0})
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("48-layer step time (ms)")
        ax.set_title(f"Qwen3-30B-A3B  M={M}  (single-GPU, bf16)")
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        # Annotate ms on top of each bar
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + max(means) * 0.01, i, f" {m:.0f} ± {s:.1f} ms",
                    va="center", fontsize=8)

    fig.suptitle("Single-GPU MoE kernel comparison — FlashInfer vs. HF native vs. vLLM Triton",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"[plot] wrote {args.out}")
    print()
    print("# Summary (mean ± std step time, ms):")
    for M in Ms:
        print(f"  M = {M}")
        for lab in labels:
            if (lab, M) in data:
                d = data[(lab, M)]
                tfl = d.get("tflops")
                tfl_s = f"  ({tfl:.1f} TFLOP/s)" if tfl else ""
                print(f"    {lab:48s}  {d['mean']:8.1f} ± {d['std']:5.1f}{tfl_s}")

    # ============================================================
    # Second figure: multi-rank EP comparison @ M_total = 32 768
    # ============================================================
    M_ep = 32768
    ep_means, ep_stds, ep_colors, ep_names = [], [], [], []
    for lab in EP_ORDER:
        if (lab, M_ep) not in data:
            continue
        d = data[(lab, M_ep)]
        ep_means.append(d["mean"])
        ep_stds.append(d["std"])
        ep_colors.append(COLORS.get(lab, "#888888"))
        ep_names.append(lab)

    if ep_names:
        fig2, ax2 = plt.subplots(figsize=(10, 5.5))
        y = np.arange(len(ep_names))
        ax2.barh(y, ep_means, xerr=ep_stds, color=ep_colors, edgecolor="black",
                 linewidth=0.5, error_kw={"elinewidth": 1.0})
        ax2.set_yticks(y)
        ax2.set_yticklabels(ep_names, fontsize=10)
        ax2.set_xlabel("48-layer step time (ms)")
        ax2.set_title(
            f"Qwen3-30B-A3B  total tokens = {M_ep}  oracle uniform routing  bf16  RTX 4090\n"
            "Multi-rank EP variants vs. single-GPU references"
        )
        ax2.invert_yaxis()
        ax2.grid(axis="x", linestyle=":", alpha=0.5)
        for i, (m, s) in enumerate(zip(ep_means, ep_stds)):
            ax2.text(m + max(ep_means) * 0.01, i, f" {m:.0f} ± {s:.1f} ms",
                     va="center", fontsize=9)
        fig2.tight_layout()
        out_ep = args.out.with_name("qwen3_flashinfer_ep_comparison.png")
        fig2.savefig(out_ep, dpi=140)
        print(f"[plot] wrote {out_ep}")
        print()
        print(f"# EP comparison @ M_total = {M_ep}:")
        for lab, m, s in zip(ep_names, ep_means, ep_stds):
            print(f"  {lab:48s}  {m:8.1f} ± {s:5.1f}")


if __name__ == "__main__":
    main()
