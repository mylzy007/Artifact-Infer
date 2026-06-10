"""Generate report-ready plots for the Tier 1 + Tier 2 microbenchmark.

Reads:
  eval_results/prefill_drop_l_sweep_tier1/tier1_summary.json
  eval_results/prefill_drop_l_sweep_tier2/tier2_summary.json
Writes:
  eval_results/tier_plots/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/lzy/Artifact-Infer/eval_results")
T1_PATH = ROOT / "prefill_drop_l_sweep_tier1" / "tier1_summary.json"
T2_PATH = ROOT / "prefill_drop_l_sweep_tier2" / "tier2_summary.json"
OUT = ROOT / "tier_plots"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 140,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load():
    with open(T1_PATH) as f:
        t1 = json.load(f)
    with open(T2_PATH) as f:
        t2 = json.load(f)
    return t1, t2


# ---------------------------------------------------------------------------
# Plot 1: Tier 1 delta_pct vs L_recv (the headline break-even plot)
# ---------------------------------------------------------------------------
def plot_tier1_delta(t1):
    pts = t1["delta_points"]
    L = np.array([p["L_recv_max_p50"] for p in pts], dtype=float)
    d = np.array([p["delta_pct"] * 100 for p in pts])
    L_star = t1["L_star"]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.axhspan(0, 50, alpha=0.08, color="red", label="slowdown")
    ax.axhspan(-30, 0, alpha=0.08, color="green", label="speedup")
    ax.axhline(0, color="gray", lw=1)
    ax.axhline(-5, color="darkgreen", lw=1, ls="--", alpha=0.7,
               label="-5% threshold (L*)")

    ax.plot(L, d, "-o", lw=2.5, ms=10, color="#1f4e9c", zorder=5)
    for x, y in zip(L, d):
        ax.annotate(f"{y:+.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 12 if y > 0 else -18), ha="center",
                    fontsize=10, fontweight="bold", color="#1f4e9c")

    ax.axvline(L_star, color="darkorange", lw=2, ls=":",
               label=f"L* = {L_star:.0f}")
    ax.annotate(f"L* = {L_star:.0f}\n(break-even)", (L_star, -22),
                textcoords="offset points", xytext=(10, 0),
                ha="left", fontsize=10, color="darkorange", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("L_recv_max  (rows per rank, log scale)")
    ax.set_ylabel("MoE-block Delta %  (drop vs baseline)")
    ax.set_title("Tier 1 — Drop break-even: L* = 3,271 rows/rank\n"
                 "(8x4090, Qwen3-30B-A3B EP-HT, tail_weight @ 0.3)",
                 fontsize=12, pad=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(-25, 45)

    plt.tight_layout()
    path = OUT / "tier1_delta_vs_L.png"
    plt.savefig(path)
    plt.close()
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Plot 2: Tier 1 segment breakdown (H2 refuted: savings come from comm)
# ---------------------------------------------------------------------------
def plot_tier1_segments(t1):
    base = t1["cell_agg"]["T2048_r0.0"]
    drop = t1["cell_agg"]["T2048_r0.3"]
    segs = ["dispatch", "experts", "combine"]
    keys = ["dispatch_us", "experts_us", "combine_us"]
    base_us = np.array([base[k]["mean"] for k in keys])
    drop_us = np.array([drop[k]["mean"] for k in keys])
    saving = base_us - drop_us
    total_saving = saving.sum()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # Left: side-by-side bars
    x = np.arange(len(segs))
    w = 0.36
    ax = axes[0]
    b1 = ax.bar(x - w / 2, base_us / 1000, w, color="#c62828",
                label="baseline (r=0)")
    b2 = ax.bar(x + w / 2, drop_us / 1000, w, color="#2e7d32",
                label="drop (tail_weight @ 0.3)")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}",
                    ha="center", fontsize=10)
    for i, s in enumerate(saving):
        if s > 100:
            ax.annotate(f"-{s/1000:.1f}ms",
                        (i + w / 2, drop_us[i] / 1000),
                        textcoords="offset points", xytext=(22, 10),
                        color="darkblue", fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(segs, fontsize=11)
    ax.set_ylabel("time (ms)")
    ax.set_title(f"Tier 1 — Segment breakdown @ T_local=2048\n"
                 f"baseline {base_us.sum()/1000:.1f} ms -> drop "
                 f"{drop_us.sum()/1000:.1f} ms "
                 f"(total -{total_saving/1000:.1f} ms)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)

    # Right: savings attribution donut
    ax = axes[1]
    colors = ["#1565c0", "#fbc02d", "#43a047"]
    wedges, texts, autotexts = ax.pie(
        saving,
        labels=[f"{s}\n-{us/1000:.2f} ms" for s, us in zip(segs, saving)],
        colors=colors, autopct=lambda p: f"{p:.1f}%",
        startangle=90, pctdistance=0.72,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("Source of drop savings — 97% from comm sides\n"
                 "(H2 refuted: experts GEMM only 2.8%)")

    plt.tight_layout()
    path = OUT / "tier1_segment_attribution.png"
    plt.savefig(path)
    plt.close()
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Plot 3: Tier 1 total_us vs L (baseline vs drop trajectories)
# ---------------------------------------------------------------------------
def plot_tier1_total(t1):
    pts = t1["delta_points"]
    L = np.array([p["L_recv_max_p50"] for p in pts], dtype=float)
    base = np.array([p["baseline_total_us_mean"] for p in pts]) / 1000
    drop = np.array([p["drop_total_us_mean"] for p in pts]) / 1000
    L_star = t1["L_star"]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(L, base, "-o", lw=2.4, ms=10, color="#c62828",
            label="baseline (r=0)")
    ax.plot(L, drop, "-s", lw=2.4, ms=10, color="#2e7d32",
            label="drop (tail_weight @ 0.3)")
    ax.axvline(L_star, color="darkorange", lw=2, ls=":",
               label=f"L* = {L_star:.0f}")
    ax.fill_between(L, base, drop, where=(drop > base), color="red", alpha=0.10)
    ax.fill_between(L, base, drop, where=(drop <= base), color="green",
                    alpha=0.10)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("L_recv_max  (rows per rank, log scale)")
    ax.set_ylabel("MoE-block total time  (ms, log scale)")
    ax.set_title("Tier 1 — Baseline vs Drop trajectories\n"
                 "drop adds overhead at L < L*; flips to net savings at L > L*")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)

    plt.tight_layout()
    path = OUT / "tier1_total_trajectories.png"
    plt.savefig(path)
    plt.close()
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Plot 4: Tier 2 prefill speedup
# ---------------------------------------------------------------------------
def plot_tier2_prefill(t2):
    sp = t2["speedups"]
    T = [s["T_local"] for s in sp]
    tot = [s["target_total_tokens"] for s in sp]
    base = np.array([s["baseline_prefill_s"] for s in sp])
    drop = np.array([s["drop_prefill_s"] for s in sp])
    speedup = np.array([s["prefill_speedup"] for s in sp])
    delta = np.array([-s["prefill_delta_pct"] * 100 for s in sp])

    base_std = np.array([t2["groups"][f"T{t}_r0.0"]["prefill_time_s"]["std"]
                         for t in T])
    drop_std = np.array([t2["groups"][f"T{t}_r0.3"]["prefill_time_s"]["std"]
                         for t in T])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: time bars
    ax = axes[0]
    x = np.arange(len(T))
    w = 0.36
    b1 = ax.bar(x - w / 2, base, w, yerr=base_std, capsize=4, color="#c62828",
                label="baseline")
    b2 = ax.bar(x + w / 2, drop, w, yerr=drop_std, capsize=4, color="#2e7d32",
                label="drop @ 0.3")
    for bars, vals in ((b1, base), (b2, drop)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                    f"{v:.2f}s", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T={t}\n({tt} tokens)" for t, tt in zip(T, tot)])
    ax.set_ylabel("prefill time (s)")
    ax.set_title("Tier 2 — Full-model prefill time\n"
                 "(Qwen3-30B-A3B, 48 layers, 3 reps)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    # Right: speedup
    ax = axes[1]
    bars = ax.bar(x, delta, color="#1565c0", width=0.5)
    ax.axhline(0, color="gray", lw=1)
    for bar, sp_v, d_v in zip(bars, speedup, delta):
        ax.text(bar.get_x() + bar.get_width() / 2, d_v + 0.3,
                f"+{d_v:.1f}%\n({sp_v:.3f}x)",
                ha="center", fontsize=11, fontweight="bold", color="#1565c0")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T={t}" for t in T])
    ax.set_ylabel("prefill speedup (%)")
    ax.set_title("Tier 2 — Prefill speedup\n"
                 "Sweet spot at mid prefill: T=512 (8.8%) > T=2048 (4.5%)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, max(delta) * 1.45)

    plt.tight_layout()
    path = OUT / "tier2_prefill_speedup.png"
    plt.savefig(path)
    plt.close()
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Plot 5: Tier 1 MoE-block vs Tier 2 full prefill (attention dilution)
# ---------------------------------------------------------------------------
def plot_combined(t1, t2):
    pts = t1["delta_points"]
    T1_T = {p["T_local"]: -p["delta_pct"] * 100 for p in pts}
    T2_T = {s["T_local"]: -s["prefill_delta_pct"] * 100 for s in t2["speedups"]}

    common_T = sorted(set(T1_T) & set(T2_T))
    moe_block = [T1_T[t] for t in common_T]
    full_prefill = [T2_T[t] for t in common_T]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(common_T))
    w = 0.36
    b1 = ax.bar(x - w / 2, moe_block, w, color="#7e57c2",
                label="Tier 1: MoE-block (isolated)")
    b2 = ax.bar(x + w / 2, full_prefill, w, color="#26a69a",
                label="Tier 2: full prefill (with attention)")
    for bars, vals in ((b1, moe_block), (b2, full_prefill)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                    f"+{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    for i, (m, f) in enumerate(zip(moe_block, full_prefill)):
        dilution = m - f
        ax.annotate(f"attention dilution\n-{dilution:.1f} pp",
                    (i, max(m, f) + 1.8), ha="center", fontsize=9,
                    color="gray", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([f"T_local={t}\n(total {t*8} tokens)" for t in common_T])
    ax.set_ylabel("Drop speedup (%)")
    ax.set_title("Tier 1 (isolated MoE-block) vs Tier 2 (full prefill)\n"
                 "Attention O(N^2) dilutes drop savings as N grows")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, max(moe_block) * 1.4)

    plt.tight_layout()
    path = OUT / "tier1_vs_tier2_dilution.png"
    plt.savefig(path)
    plt.close()
    print(f"  wrote {path}")


def main():
    t1, t2 = load()
    print(f"Generating Tier 1/2 plots -> {OUT}/")
    plot_tier1_delta(t1)
    plot_tier1_segments(t1)
    plot_tier1_total(t1)
    plot_tier2_prefill(t2)
    plot_combined(t1, t2)
    print(f"\nDONE — {len(list(OUT.glob('*.png')))} plots in {OUT}/")


if __name__ == "__main__":
    main()
