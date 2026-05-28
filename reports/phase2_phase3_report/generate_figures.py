"""Generate comparison figures for Phase 2 (static placement) and Phase 3 (replica + dispatch).

All numbers come directly from the user's reported summary tables.
Figures are saved to ./figures/ next to this script.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})

# ----------------------------------------------------------------------------
# Phase 2 — static placement (5 strategies)
# ----------------------------------------------------------------------------
PH2 = [
    # name, e2e_s, prefill, decode, gpu_cv, cross_traffic
    ("round_robin",          100.00, 61.95, 42.20, 0.0526, 0.8748),
    ("contiguous",           100.32, 61.02, 42.22, 0.0720, 0.8750),
    ("communication_aware",  103.41, 57.35, 42.62, 0.1435, 0.8707),
    ("load_balanced",        104.68, 59.43, 40.60, 0.0009, 0.8750),
    ("fixed_random_shuffle", 107.86, 54.38, 41.21, 0.1080, 0.8747),
]
ph2_names    = [r[0] for r in PH2]
ph2_e2e      = np.array([r[1] for r in PH2])
ph2_prefill  = np.array([r[2] for r in PH2])
ph2_decode   = np.array([r[3] for r in PH2])
ph2_gpu_cv   = np.array([r[4] for r in PH2])
ph2_cross    = np.array([r[5] for r in PH2])
baseline_e2e = ph2_e2e[ph2_names.index("contiguous")]

# Fig 1: e2e_s + relative to contiguous
fig, ax = plt.subplots(figsize=(8.2, 4.4))
colors = ["#4C9F70" if v <= baseline_e2e else "#D9534F" for v in ph2_e2e]
bars = ax.bar(ph2_names, ph2_e2e, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(baseline_e2e, color="gray", linestyle="--", linewidth=1,
           label=f"contiguous baseline = {baseline_e2e:.2f}s")
for b, v in zip(bars, ph2_e2e):
    delta = (v - baseline_e2e) / baseline_e2e * 100
    ax.text(b.get_x() + b.get_width()/2, v + 0.3,
            f"{v:.2f}\n({delta:+.2f}%)", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("End-to-end latency (s)")
ax.set_title("Phase 2: End-to-end latency by static expert placement (GSM8K-256, ep_ht, ws=8)")
ax.set_ylim(95, max(ph2_e2e) * 1.05)
ax.legend(loc="upper left")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT / "phase2_01_e2e.png")
plt.close()

# Fig 2: throughput (prefill / decode) grouped bar
fig, ax = plt.subplots(figsize=(8.2, 4.4))
x = np.arange(len(ph2_names))
w = 0.38
b1 = ax.bar(x - w/2, ph2_prefill, w, label="prefill (tok/s)", color="#4F81BD", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + w/2, ph2_decode,  w, label="decode (tok/s)",  color="#F79646", edgecolor="black", linewidth=0.5)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(ph2_names, rotation=20, ha="right")
ax.set_ylabel("Throughput (tok/s)")
ax.set_title("Phase 2: Prefill / Decode throughput by static placement")
ax.legend()
ax.set_ylim(0, max(ph2_prefill.max(), ph2_decode.max()) * 1.18)
plt.tight_layout()
plt.savefig(OUT / "phase2_02_throughput.png")
plt.close()

# Fig 3: gpu_cv (log scale) + cross_traffic_ratio in two panels
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

ax = axes[0]
bars = ax.bar(ph2_names, ph2_gpu_cv, color="#9C5FB5", edgecolor="black", linewidth=0.5)
ax.set_yscale("log")
ax.set_ylabel("GPU CV (lower = better balance, log scale)")
ax.set_title("Phase 2: Load imbalance across GPUs (gpu_cv)")
for b, v in zip(bars, ph2_gpu_cv):
    ax.text(b.get_x() + b.get_width()/2, v * 1.15,
            f"{v:.4f}", ha="center", va="bottom", fontsize=9)
ax.tick_params(axis="x", rotation=20)
for lbl in ax.get_xticklabels():
    lbl.set_horizontalalignment("right")

ax = axes[1]
bars = ax.bar(ph2_names, ph2_cross, color="#5DADE2", edgecolor="black", linewidth=0.5)
ax.set_ylabel("Cross-GPU traffic ratio")
ax.set_title("Phase 2: Cross-GPU traffic ratio")
ax.set_ylim(0.86, 0.88)
for b, v in zip(bars, ph2_cross):
    ax.text(b.get_x() + b.get_width()/2, v + 0.0005,
            f"{v:.4f}", ha="center", va="bottom", fontsize=9)
ax.tick_params(axis="x", rotation=20)
for lbl in ax.get_xticklabels():
    lbl.set_horizontalalignment("right")

plt.tight_layout()
plt.savefig(OUT / "phase2_03_cv_and_traffic.png")
plt.close()

# ----------------------------------------------------------------------------
# Phase 3 — replica × dispatch heatmaps for two base placements
# ----------------------------------------------------------------------------
REPLICAS  = ["consecutive", "numa_first", "traffic_aware", "traffic_balance"]
DISPATCH  = ["balance", "min-comm", "numa-aware", "hybrid"]

# round_robin base, ov=0.25, cell = (e2e, prefill, decode, xnuma)
RR = {
    "consecutive":     [(72.55, 185.79, 36.58, 0.4299),
                        (80.70, 174.77, 36.15, 0.4471),
                        (73.96, 163.58, 37.20, 0.2397),
                        (100.54, 82.07, 32.78, 0.4319)],
    "numa_first":      [(71.74, 178.65, 37.65, 0.4188),
                        (69.88, 172.38, 39.49, 0.3791),
                        (71.33, 164.45, 38.93, 0.0000),
                        (97.78,  88.79, 32.60, 0.4006)],
    "traffic_aware":   [(71.77, 179.89, 37.67, 0.4299),
                        (75.54, 174.05, 35.36, 0.4471),
                        (74.12, 163.65, 37.02, 0.2397),
                        (107.65, 81.80, 30.18, 0.4320)],
    "traffic_balance": [(74.60, 210.72, 34.23, 0.4299),
                        (70.51, 173.75, 39.11, 0.4473),
                        (75.57, 161.47, 37.81, 0.2397),
                        (102.86, 80.68, 32.35, 0.4320)],
}
RR_BASE_E2E = 104.40  # overlap=0 baseline

# load_balanced base
LB = {
    "consecutive":     [(72.56, 182.99, 37.77, 0.4280),
                        (72.25, 176.06, 37.95, 0.4342),
                        (78.18, 180.68, 33.87, 0.2032),
                        (94.78,  93.10, 33.26, 0.4265)],
    "numa_first":      [(69.91, 177.88, 39.11, 0.4197),
                        (70.74, 174.53, 38.77, 0.3792),
                        (72.01, 164.36, 38.87, 0.0000),
                        (99.60,  84.73, 32.89, 0.4017)],
    "traffic_aware":   [(72.76, 180.63, 37.51, 0.4280),
                        (75.00, 200.72, 34.28, 0.4339),
                        (76.64, 162.76, 35.67, 0.2032),
                        (104.74, 83.89, 30.16, 0.4264)],
    "traffic_balance": [(73.07, 173.06, 37.32, 0.4280),
                        (71.98, 191.61, 42.28, 0.4342),
                        (74.40, 163.46, 38.21, 0.2032),
                        (101.88, 79.38, 32.60, 0.4264)],
}
LB_BASE_E2E = 71.77

def mat(table: dict, idx: int) -> np.ndarray:
    return np.array([[table[r][d][idx] for d in range(len(DISPATCH))] for r in REPLICAS])

def heatmap(ax, M, title, fmt="{:.2f}", cmap="RdYlGn_r", vmin=None, vmax=None, highlight_min=True):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(DISPATCH))); ax.set_xticklabels(DISPATCH, rotation=15)
    ax.set_yticks(range(len(REPLICAS))); ax.set_yticklabels(REPLICAS)
    ax.set_xlabel("Dispatch policy"); ax.set_ylabel("Replica placement policy")
    ax.set_title(title)
    if highlight_min:
        i, j = np.unravel_index(np.argmin(M), M.shape)
    else:
        i, j = -1, -1
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            val = M[r, c]
            txt = fmt.format(val)
            color = "white" if (cmap == "RdYlGn_r" and val > (vmin + vmax) / 2 if (vmin is not None and vmax is not None) else False) else "black"
            ax.text(c, r, txt, ha="center", va="center", color=color, fontsize=9,
                    fontweight=("bold" if (r == i and c == j) else "normal"))
            if r == i and c == j:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                            edgecolor="blue", linewidth=2.5))
    return im

# Fig 4: e2e heatmaps (two bases side by side)
rr_e2e = mat(RR, 0); lb_e2e = mat(LB, 0)
vmin = min(rr_e2e.min(), lb_e2e.min()); vmax = max(rr_e2e.max(), lb_e2e.max())
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
im = heatmap(axes[0], rr_e2e, f"Base: round_robin   (overlap=0 baseline e2e={RR_BASE_E2E}s)",
             vmin=vmin, vmax=vmax)
heatmap(axes[1], lb_e2e, f"Base: load_balanced   (overlap=0 baseline e2e={LB_BASE_E2E}s)",
        vmin=vmin, vmax=vmax)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="e2e (s)  — lower is better")
fig.suptitle("Phase 3: e2e latency under Replica×Dispatch (overlap=0.25, ep_ht, ws=8)", y=1.02)
plt.savefig(OUT / "phase3_01_e2e_heatmap.png", bbox_inches="tight")
plt.close()

# Fig 5: cross_numa heatmaps
rr_xn = mat(RR, 3); lb_xn = mat(LB, 3)
vmin = min(rr_xn.min(), lb_xn.min()); vmax = max(rr_xn.max(), lb_xn.max())
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
im = heatmap(axes[0], rr_xn, "Base: round_robin", cmap="RdYlGn_r",
             vmin=vmin, vmax=vmax, fmt="{:.4f}")
heatmap(axes[1], lb_xn, "Base: load_balanced", cmap="RdYlGn_r",
        vmin=vmin, vmax=vmax, fmt="{:.4f}")
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="cross_numa ratio — lower is better")
fig.suptitle("Phase 3: cross-NUMA traffic ratio under Replica×Dispatch (overlap=0.25)", y=1.02)
plt.savefig(OUT / "phase3_02_xnuma_heatmap.png", bbox_inches="tight")
plt.close()

# Fig 6: prefill throughput heatmaps
rr_pf = mat(RR, 1); lb_pf = mat(LB, 1)
vmin = min(rr_pf.min(), lb_pf.min()); vmax = max(rr_pf.max(), lb_pf.max())
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
im = heatmap(axes[0], rr_pf, "Base: round_robin", cmap="RdYlGn",
             vmin=vmin, vmax=vmax, fmt="{:.1f}", highlight_min=False)
heatmap(axes[1], lb_pf, "Base: load_balanced", cmap="RdYlGn",
        vmin=vmin, vmax=vmax, fmt="{:.1f}", highlight_min=False)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="prefill tok/s — higher is better")
fig.suptitle("Phase 3: Prefill throughput under Replica×Dispatch (overlap=0.25)", y=1.02)
plt.savefig(OUT / "phase3_03_prefill_heatmap.png", bbox_inches="tight")
plt.close()

# Fig 7: best per replica policy, both bases, grouped bar
best_rr = {r: min(RR[r], key=lambda c: c[0]) for r in REPLICAS}
best_lb = {r: min(LB[r], key=lambda c: c[0]) for r in REPLICAS}
rr_vals = [best_rr[r][0] for r in REPLICAS]
lb_vals = [best_lb[r][0] for r in REPLICAS]
rr_disp = [DISPATCH[[c[0] for c in RR[r]].index(best_rr[r][0])] for r in REPLICAS]
lb_disp = [DISPATCH[[c[0] for c in LB[r]].index(best_lb[r][0])] for r in REPLICAS]

fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(len(REPLICAS)); w = 0.36
b1 = ax.bar(x - w/2, rr_vals, w, label="base = round_robin", color="#4F81BD", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + w/2, lb_vals, w, label="base = load_balanced", color="#9BBB59", edgecolor="black", linewidth=0.5)
for b, disp in zip(b1, rr_disp):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.4,
            f"{b.get_height():.2f}\n[{disp}]", ha="center", va="bottom", fontsize=8)
for b, disp in zip(b2, lb_disp):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.4,
            f"{b.get_height():.2f}\n[{disp}]", ha="center", va="bottom", fontsize=8)
ax.axhline(RR_BASE_E2E, color="#4F81BD", linestyle=":", linewidth=1.2,
           label=f"round_robin ov=0 baseline ({RR_BASE_E2E})")
ax.axhline(LB_BASE_E2E, color="#9BBB59", linestyle=":", linewidth=1.2,
           label=f"load_balanced ov=0 baseline ({LB_BASE_E2E})")
ax.set_xticks(x); ax.set_xticklabels(REPLICAS)
ax.set_ylabel("Best e2e latency (s)")
ax.set_title("Phase 3: Best e2e per Replica policy (annotated with the winning dispatch)")
ax.set_ylim(0, max(rr_vals + lb_vals) * 1.25)
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "phase3_04_best_per_replica.png")
plt.close()

# Fig 8: speedup baseline ov=0 vs best ov=0.25
rr_best = min((c[0] for r in REPLICAS for c in RR[r]))
lb_best = min((c[0] for r in REPLICAS for c in LB[r]))

fig, ax = plt.subplots(figsize=(7.6, 4.4))
groups = ["round_robin", "load_balanced"]
bv = [RR_BASE_E2E, LB_BASE_E2E]
ov = [rr_best, lb_best]
x = np.arange(len(groups)); w = 0.36
b1 = ax.bar(x - w/2, bv, w, label="overlap=0 baseline", color="#BFBFBF", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + w/2, ov, w, label="overlap=0.25 best", color="#4C9F70", edgecolor="black", linewidth=0.5)
for b, v in zip(b1, bv):
    ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
for b, v, bv_v in zip(b2, ov, bv):
    spd = bv_v / v
    ax.text(b.get_x() + b.get_width()/2, v + 0.6,
            f"{v:.2f}\n({spd:.2f}x)", ha="center", va="bottom", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("End-to-end latency (s)")
ax.set_title("Phase 3: Speedup from adding overlap=0.25 + best replica/dispatch")
ax.set_ylim(0, max(bv) * 1.25)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "phase3_05_speedup.png")
plt.close()

# Fig 9: phase2 vs phase3-best e2e summary
fig, ax = plt.subplots(figsize=(9, 4.4))
labels = ["Ph2 contiguous", "Ph2 round_robin", "Ph2 load_balanced",
          "Ph3 RR ov=0.25 best", "Ph3 LB ov=0.25 best"]
vals = [100.32, 100.00, 104.68, rr_best, lb_best]
colors = ["#BFBFBF", "#BFBFBF", "#BFBFBF", "#4C9F70", "#4C9F70"]
bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.8, f"{v:.2f}",
            ha="center", va="bottom", fontsize=10)
ax.set_ylabel("End-to-end latency (s)")
ax.set_title("Phase 2 vs Phase 3 best — overall e2e latency comparison")
ax.set_ylim(0, max(vals) * 1.18)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUT / "summary_phase2_vs_phase3.png")
plt.close()

print("Wrote figures to:", OUT)
for p in sorted(OUT.glob("*.png")):
    print(" -", p.name)
