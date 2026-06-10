"""Clean summary plot for the Stage 0 writeup."""
from __future__ import annotations

from pathlib import Path
import numpy as np

IN_DIR = Path("eval_results/compression_lowrank_stage0_v2")

LAYERS = [2, 6, 12, 24, 36, 46]
HIDDEN = 2048


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(LAYERS)))

    # Left: full curve.
    for c, L in zip(colors, LAYERS):
        p = IN_DIR / f"cum_var_centered_layer{L}.npy"
        if not p.exists():
            continue
        cum = np.load(p)
        xs = np.arange(1, len(cum) + 1) / HIDDEN
        ax1.plot(xs, cum, color=c, label=f"layer {L}")

    for thr in (0.9, 0.95, 0.99):
        ax1.axhline(thr, color="red", linestyle=":", linewidth=0.7, alpha=0.6)
        ax1.text(0.02, thr + 0.005, f"{thr:.2f}", fontsize=8, color="red")
    for frac, lbl in ((1 / 8, "d/8"), (1 / 4, "d/4"), (1 / 2, "d/2")):
        ax1.axvline(frac, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax1.text(frac + 0.005, 0.02, lbl, fontsize=8, color="gray")
    ax1.set_xlabel("l / d (kept rank fraction)")
    ax1.set_ylabel("cumulative explained variance (centered)")
    ax1.set_title("Global dispatch activation spectrum (Qwen3-30B-A3B)")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)

    # Right: zoom into 0-d/2 to read off the regime relevant to compression.
    for c, L in zip(colors, LAYERS):
        p = IN_DIR / f"cum_var_centered_layer{L}.npy"
        if not p.exists():
            continue
        cum = np.load(p)
        xs = np.arange(1, len(cum) + 1) / HIDDEN
        m = xs <= 0.5
        ax2.plot(xs[m], cum[m], color=c, label=f"layer {L}")

    ax2.axhline(0.90, color="red", linestyle=":", linewidth=0.8)
    ax2.text(0.005, 0.905, "0.90 (Go threshold)", fontsize=8, color="red")
    ax2.axvline(0.25, color="gray", linestyle="--", linewidth=0.8)
    ax2.text(0.255, 0.05, "d/4 (4x compression target)", fontsize=8, color="gray")
    ax2.set_xlabel("l / d")
    ax2.set_ylabel("cumulative explained variance")
    ax2.set_title("Zoom: l/d in [0, 1/2]")
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    out = IN_DIR / "svd_curve_summary.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
