#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


def _prob_std(probs: list[float]) -> float:
    mean = sum(probs) / len(probs)
    return math.sqrt(sum((x - mean) ** 2 for x in probs) / len(probs))


def _prob_cv(probs: list[float]) -> float:
    mean = sum(probs) / len(probs)
    if mean <= 0:
        return 0.0
    return _prob_std(probs) / mean


def _prob_entropy_nats(probs: list[float]) -> float:
    eps = 1e-12
    return -sum(x * math.log(max(x, eps)) for x in probs if x > 0.0)


def _weighted_mean(values: list[float], weights: list[float]) -> float | None:
    if not values:
        return None
    total_w = sum(weights)
    if total_w <= 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / total_w


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _mode_name(mode: int) -> str:
    if int(mode) == 1:
        return "prefill"
    if int(mode) == 2:
        return "decode"
    return "other"


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _fmt_float(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.6f}"


def _compute_case(case_dir: Path) -> Path:
    meta = _load_json(case_dir / "record_meta.json")
    pt_path = Path(meta["expert_record_pt"])
    data = torch.load(pt_path, map_location="cpu")

    rank_count = data["rank_count"]
    forward_modes = [int(x) for x in data["forward_modes"]]
    moe_layer_ids_in_model = list(meta["moe_layer_ids_in_model"])
    ep_size = int(meta["ep_size"])

    if rank_count.ndim != 3:
        raise ValueError(f"Unexpected rank_count shape: {tuple(rank_count.shape)}")

    if rank_count.shape[0] != len(forward_modes):
        raise ValueError(
            f"rank_count time dimension ({rank_count.shape[0]}) does not match "
            f"forward_modes length ({len(forward_modes)}) for {case_dir}"
        )

    rank_count = rank_count[:, moe_layer_ids_in_model, :].to(torch.float64)

    out_path = case_dir / "layer_gpu_share_stats.csv"
    fieldnames = [
        "moe_layer",
        "model_layer",
        "num_gpus",
        "steps_all",
        "mean_routed_tokens_per_step_all",
        "weighted_mean_cv_all",
        "weighted_mean_std_all",
        "weighted_mean_entropy_nats_all",
        "p50_cv_all",
        "p90_cv_all",
        "steps_prefill",
        "mean_routed_tokens_per_step_prefill",
        "weighted_mean_cv_prefill",
        "weighted_mean_std_prefill",
        "weighted_mean_entropy_nats_prefill",
        "p50_cv_prefill",
        "p90_cv_prefill",
        "steps_decode",
        "mean_routed_tokens_per_step_decode",
        "weighted_mean_cv_decode",
        "weighted_mean_std_decode",
        "weighted_mean_entropy_nats_decode",
        "p50_cv_decode",
        "p90_cv_decode",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for moe_layer_idx, model_layer_idx in enumerate(moe_layer_ids_in_model):
            by_bucket: dict[str, dict[str, list[float]]] = {
                "all": {"cv": [], "std": [], "entropy": [], "weights": []},
                "prefill": {"cv": [], "std": [], "entropy": [], "weights": []},
                "decode": {"cv": [], "std": [], "entropy": [], "weights": []},
            }

            layer_rank_count = rank_count[:, moe_layer_idx, :]
            for step_idx in range(layer_rank_count.shape[0]):
                counts = layer_rank_count[step_idx].tolist()
                total = float(sum(counts))
                if total <= 0:
                    continue
                probs = [x / total for x in counts]
                cur_std = _prob_std(probs)
                cur_cv = _prob_cv(probs)
                cur_entropy = _prob_entropy_nats(probs)
                mode_bucket = _mode_name(forward_modes[step_idx])

                for bucket_name in ("all", mode_bucket):
                    if bucket_name not in by_bucket:
                        continue
                    by_bucket[bucket_name]["cv"].append(cur_cv)
                    by_bucket[bucket_name]["std"].append(cur_std)
                    by_bucket[bucket_name]["entropy"].append(cur_entropy)
                    by_bucket[bucket_name]["weights"].append(total)

            row: dict[str, str | int] = {
                "moe_layer": moe_layer_idx,
                "model_layer": model_layer_idx,
                "num_gpus": ep_size,
            }

            for bucket_name in ("all", "prefill", "decode"):
                bucket = by_bucket[bucket_name]
                weights = bucket["weights"]
                steps = len(weights)
                prefix = bucket_name
                row[f"steps_{prefix}"] = steps
                row[f"mean_routed_tokens_per_step_{prefix}"] = _fmt_float(
                    (sum(weights) / steps) if steps > 0 else None
                )
                row[f"weighted_mean_cv_{prefix}"] = _fmt_float(
                    _weighted_mean(bucket["cv"], weights)
                )
                row[f"weighted_mean_std_{prefix}"] = _fmt_float(
                    _weighted_mean(bucket["std"], weights)
                )
                row[f"weighted_mean_entropy_nats_{prefix}"] = _fmt_float(
                    _weighted_mean(bucket["entropy"], weights)
                )
                row[f"p50_cv_{prefix}"] = _fmt_float(_percentile(bucket["cv"], 0.50))
                row[f"p90_cv_{prefix}"] = _fmt_float(_percentile(bucket["cv"], 0.90))

            writer.writerow(row)

    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-layer GPU-share statistics from vLLM expert recorder output, "
            "using token-weighted per-step CV/std/entropy."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing case subdirectories with record_meta.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root
    case_dirs = sorted(
        p.parent for p in root.glob("*/record_meta.json") if p.parent.is_dir()
    )
    if not case_dirs:
        raise SystemExit(f"No case directories with record_meta.json found under {root}")

    for case_dir in case_dirs:
        out_path = _compute_case(case_dir)
        print(out_path)


if __name__ == "__main__":
    main()
