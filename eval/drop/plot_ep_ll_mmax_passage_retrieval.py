from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_mib(num_bytes: float | int | None) -> float:
    if not num_bytes:
        return 0.0
    return float(num_bytes) / (1024.0 ** 2)


def as_gib(num_bytes: float | int | None) -> float:
    if not num_bytes:
        return 0.0
    return float(num_bytes) / (1024.0 ** 3)


def pick_rank_aggregate(rank_entry: dict, key: str) -> float:
    return float(rank_entry.get(key) or 0.0)


def render_overview(summary: dict, out_path: Path) -> None:
    cells = list(summary["cells"].values())
    cells.sort(key=lambda x: float(x.get("m_max_effective") or 0))

    labels = [
        "auto" if int(cell.get("m_max_cfg") or -1) <= 0 else str(cell.get("m_max_effective"))
        for cell in cells
    ]
    x = np.arange(len(cells))
    acc_off = [float(cell["acc_official"]["mean"] or 0.0) for cell in cells]
    acc_str = [float(cell["acc_strict"]["mean"] or 0.0) for cell in cells]
    e2e = [float(cell["e2e_total_time_s"]["mean"] or 0.0) for cell in cells]
    prefill_tok = [float(cell["prefill_tok_s"]["mean"] or 0.0) for cell in cells]
    overflow = [float(cell["ep_ll_overflow_replica_fraction"]["mean"] or 0.0) for cell in cells]
    bucket_max = [float(cell["ep_ll_observed_bucket_max"]["mean"] or 0.0) for cell in cells]
    workspace_gib = [as_gib(cell.get("ep_ll_workspace_bytes_total")) for cell in cells]
    mmax_effective = [int(cell.get("m_max_effective") or 0) for cell in cells]
    ep_size = int(summary.get("config", {}).get("world_size") or 8)
    system_prefill = [v * ep_size for v in prefill_tok]

    base_e2e = e2e[0] if e2e else 0.0
    base_acc = acc_off[0] if acc_off else 0.0
    e2e_speedup = [(base_e2e / v) if v > 0 else 0.0 for v in e2e]
    acc_delta = [v - base_acc for v in acc_off]

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    ax.plot(x, acc_off, marker="o", label="official")
    ax.plot(x, acc_str, marker="s", label="strict")
    ax.set_title("Score vs M_max")
    ax.set_xticks(x, labels)
    ax.set_ylabel("score")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(x, e2e, marker="o", color="#1f77b4", label="e2e_s")
    ax2 = ax.twinx()
    ax2.plot(x, e2e_speedup, marker="s", color="#d62728", label="speedup vs auto")
    ax.set_title("E2E Tradeoff")
    ax.set_xticks(x, labels)
    ax.set_ylabel("e2e_s")
    ax2.set_ylabel("speedup")
    ax.grid(alpha=0.25)

    ax = axes[0, 2]
    ax.plot(x, prefill_tok, marker="o", label="per-rank tok/s")
    ax.plot(x, system_prefill, marker="s", label=f"system tok/s (x{ep_size})")
    ax.set_title("Prefill Throughput")
    ax.set_xticks(x, labels)
    ax.set_ylabel("tok/s")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.bar(x, overflow, color="#ff7f0e", alpha=0.85, label="overflow fraction")
    ax2 = ax.twinx()
    ax2.plot(x, bucket_max, marker="o", color="#2ca02c", label="observed bucket max")
    ax.set_title("Overflow vs Bucket Pressure")
    ax.set_xticks(x, labels)
    ax.set_ylabel("overflow fraction")
    ax2.set_ylabel("observed bucket max")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.bar(x, workspace_gib, color="#9467bd")
    ax.set_title("Workspace Footprint")
    ax.set_xticks(x, labels)
    ax.set_ylabel("workspace GiB / rank")
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    sizes = [max(40.0, 1200.0 * v + 40.0) for v in overflow]
    ax.scatter(e2e_speedup, acc_delta, s=sizes, c=mmax_effective, cmap="viridis")
    for idx, label in enumerate(labels):
        ax.annotate(label, (e2e_speedup[idx], acc_delta[idx]), xytext=(5, 5), textcoords="offset points")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.35)
    ax.axvline(1.0, color="black", linewidth=1, alpha=0.35)
    ax.set_title("Pareto: Speedup vs Accuracy Delta")
    ax.set_xlabel("e2e speedup vs auto")
    ax.set_ylabel("official score delta vs auto")
    ax.grid(alpha=0.25)

    fig.suptitle("EP-LL M_max Experiment Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _extract_heatmap(rows: list[dict], target_label: str, field: str, reduce_mode: str) -> np.ndarray | None:
    target_rows = [row for row in rows if row.get("label") == target_label]
    if not target_rows:
        return None
    rank_entries = target_rows[0].get("ep_ll_stats_by_rank") or []
    if not rank_entries:
        return None
    rank_entries = sorted(rank_entries, key=lambda x: int(x.get("global_rank", 0)))
    num_ranks = len(rank_entries)
    max_layer = max(len(entry.get("per_layer") or []) for entry in rank_entries)
    mat = np.zeros((max_layer, num_ranks), dtype=np.float64)
    for col, rank_entry in enumerate(rank_entries):
        for layer_item in rank_entry.get("per_layer") or []:
            layer_id = int(layer_item["layer_id"])
            matrix = layer_item.get(field)
            if matrix is None:
                continue
            arr = np.asarray(matrix, dtype=np.float64)
            if reduce_mode == "max":
                mat[layer_id, col] = arr.max()
            elif reduce_mode == "overflow_count":
                m_max = float(rank_entry.get("m_max") or 0)
                mat[layer_id, col] = float((arr > m_max).sum())
            elif reduce_mode == "mean":
                mat[layer_id, col] = arr.mean()
            else:
                raise ValueError(reduce_mode)
    return mat


def render_heatmaps(summary: dict, rows: list[dict], out_path: Path) -> None:
    cells = list(summary["cells"].items())
    cells.sort(key=lambda item: float(item[1].get("m_max_effective") or 0))
    selected = []
    if cells:
        selected.append(cells[0][0])
    if len(cells) > 1:
        selected.append(cells[-1][0])

    if not selected:
        return

    fig, axes = plt.subplots(len(selected), 2, figsize=(12, 4 * len(selected)))
    if len(selected) == 1:
        axes = np.asarray([axes])

    for row_idx, label in enumerate(selected):
        cell = summary["cells"][label]
        max_heat = _extract_heatmap(rows, label, "bucket_max_matrix", "max")
        overflow_heat = _extract_heatmap(rows, label, "bucket_max_matrix", "overflow_count")
        for col_idx, (heat, title_suffix) in enumerate(
            [
                (max_heat, "bucket max"),
                (overflow_heat, "overflow bucket count"),
            ]
        ):
            ax = axes[row_idx, col_idx]
            if heat is None:
                ax.axis("off")
                continue
            im = ax.imshow(heat, aspect="auto", cmap="magma")
            ax.set_title(f"{label} ({cell['m_max_effective']}) - {title_suffix}")
            ax.set_xlabel("global rank")
            ax.set_ylabel("layer")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("EP-LL Per-Rank / Per-Layer Bucket Pressure", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_table(summary: dict, out_path: Path) -> None:
    rows = []
    for label, cell in summary["cells"].items():
        rows.append(
            {
                "label": label,
                "m_max_cfg": cell.get("m_max_cfg"),
                "m_max_effective": cell.get("m_max_effective"),
                "official_score": cell["acc_official"]["mean"],
                "strict_score": cell["acc_strict"]["mean"],
                "e2e_total_time_s": cell["e2e_total_time_s"]["mean"],
                "prefill_tok_s_per_rank": cell["prefill_tok_s"]["mean"],
                "decode_tok_s_per_rank": cell["decode_tok_s"]["mean"],
                "overflow_fraction": cell["ep_ll_overflow_replica_fraction"]["mean"],
                "observed_bucket_max": cell["ep_ll_observed_bucket_max"]["mean"],
                "workspace_gib_per_rank": as_gib(cell.get("ep_ll_workspace_bytes_total")),
            }
        )
    rows.sort(key=lambda x: float(x.get("m_max_effective") or 0))
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary = load_json(run_dir / "ep_ll_mmax_summary.json")
    rows = load_jsonl(run_dir / "ep_ll_mmax_rows.jsonl")
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    render_overview(summary, out_dir / "ep_ll_mmax_overview.png")
    render_heatmaps(summary, rows, out_dir / "ep_ll_mmax_heatmaps.png")
    write_table(summary, out_dir / "ep_ll_mmax_table.csv")


if __name__ == "__main__":
    main()
