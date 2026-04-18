#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


FIELDNAMES = [
    "engine",
    "dataset_id",
    "case_name",
    "chunked_prefill",
    "eplb",
    "num_steps_all",
    "num_steps_prefill",
    "num_steps_decode",
    "num_layers",
    "whole_model_cv_all",
    "whole_model_cv_prefill",
    "whole_model_cv_decode",
    "layer_cv_all",
    "whole_model_entropy_all",
    "whole_model_entropy_decode",
    "mean_routed_tokens_per_step_all",
    "mean_routed_tokens_per_step_decode",
    "record_meta_path",
]


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


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _mode_bucket(mode: int | None) -> str:
    if int(mode or 0) == 1:
        return "prefill"
    if int(mode or 0) == 2:
        return "decode"
    return "other"


def _case_flags(case_name: str) -> tuple[str, str]:
    chunked_prefill = "1" if case_name.startswith("chunk_") else "0"
    eplb = "1" if "_eplb" in case_name and "noeplb" not in case_name else "0"
    return chunked_prefill, eplb


def _scan_case_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("*/*/record_meta.json"))


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _empty_row(engine: str, dataset_id: str, case_name: str, record_meta_path: Path) -> dict[str, str]:
    chunked_prefill, eplb = _case_flags(case_name)
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "engine": engine,
            "dataset_id": dataset_id,
            "case_name": case_name,
            "chunked_prefill": chunked_prefill,
            "eplb": eplb,
            "record_meta_path": str(record_meta_path),
        }
    )
    return row


def _compute_row(engine: str, case_dir: Path) -> dict[str, str]:
    dataset_id = case_dir.parent.name
    case_name = case_dir.name
    record_meta_path = case_dir / "record_meta.json"
    row = _empty_row(engine, dataset_id, case_name, record_meta_path)

    meta = _load_json(record_meta_path)
    pt_path = Path(meta["expert_record_pt"])
    data = torch.load(pt_path, map_location="cpu")

    rank_count = data["rank_count"]
    if rank_count.ndim != 3:
        raise ValueError(f"Unexpected rank_count shape {tuple(rank_count.shape)}")

    forward_modes = [int(x or 0) for x in data.get("forward_modes", [])]
    if len(forward_modes) != rank_count.shape[0]:
        raise ValueError(
            f"forward_modes length {len(forward_modes)} != rank_count steps {rank_count.shape[0]}"
        )

    moe_layers = meta.get("moe_layer_ids_in_model")
    if moe_layers is None:
        moe_layers = list(range(rank_count.shape[1]))
    rank_count = rank_count[:, moe_layers, :].to(torch.float64)

    whole_values: dict[str, list[float]] = {"all": [], "prefill": [], "decode": []}
    whole_entropy: dict[str, list[float]] = {"all": [], "prefill": [], "decode": []}
    whole_weights: dict[str, list[float]] = {"all": [], "prefill": [], "decode": []}
    layer_cvs: list[float] = []
    layer_weights: list[float] = []

    for step_idx in range(rank_count.shape[0]):
        step_rank_count = rank_count[step_idx].sum(dim=0)
        total = float(step_rank_count.sum().item())
        if total <= 0:
            continue

        probs = [float(x / total) for x in step_rank_count.tolist()]
        mode_bucket = _mode_bucket(forward_modes[step_idx])
        cur_cv = _prob_cv(probs)
        cur_entropy = _prob_entropy_nats(probs)

        for bucket_name in ("all", mode_bucket):
            if bucket_name in whole_values:
                whole_values[bucket_name].append(cur_cv)
                whole_entropy[bucket_name].append(cur_entropy)
                whole_weights[bucket_name].append(total)

        for layer_idx in range(rank_count.shape[1]):
            layer_rank_count = rank_count[step_idx, layer_idx]
            layer_total = float(layer_rank_count.sum().item())
            if layer_total <= 0:
                continue
            layer_probs = [float(x / layer_total) for x in layer_rank_count.tolist()]
            layer_cvs.append(_prob_cv(layer_probs))
            layer_weights.append(layer_total)

    row.update(
        {
            "num_steps_all": str(len(whole_weights["all"])),
            "num_steps_prefill": str(len(whole_weights["prefill"])),
            "num_steps_decode": str(len(whole_weights["decode"])),
            "num_layers": str(rank_count.shape[1]),
            "whole_model_cv_all": _fmt_float(
                _weighted_mean(whole_values["all"], whole_weights["all"])
            ),
            "whole_model_cv_prefill": _fmt_float(
                _weighted_mean(whole_values["prefill"], whole_weights["prefill"])
            ),
            "whole_model_cv_decode": _fmt_float(
                _weighted_mean(whole_values["decode"], whole_weights["decode"])
            ),
            "layer_cv_all": _fmt_float(_weighted_mean(layer_cvs, layer_weights)),
            "whole_model_entropy_all": _fmt_float(
                _weighted_mean(whole_entropy["all"], whole_weights["all"])
            ),
            "whole_model_entropy_decode": _fmt_float(
                _weighted_mean(whole_entropy["decode"], whole_weights["decode"])
            ),
            "mean_routed_tokens_per_step_all": _fmt_float(
                (sum(whole_weights["all"]) / len(whole_weights["all"])) if whole_weights["all"] else None
            ),
            "mean_routed_tokens_per_step_decode": _fmt_float(
                (sum(whole_weights["decode"]) / len(whole_weights["decode"]))
                if whole_weights["decode"]
                else None
            ),
        }
    )
    return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute unified whole-model and layer-level CV metrics for existing vLLM/SGLang runs."
    )
    parser.add_argument(
        "--vllm-root",
        type=Path,
        default=Path("/home/lzy/eval/vllm_deepseek_v2_lite_matrix4"),
        help="Evaluation root for vLLM runs.",
    )
    parser.add_argument(
        "--sglang-root",
        type=Path,
        default=Path("/home/lzy/eval/sglang_deepseek_v2_lite_matrix4"),
        help="Evaluation root for SGLang runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/lzy/vllm/refine-logs/q2_unified_cv_existing_runs.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows: list[dict[str, str]] = []

    for engine, root in (("vllm", args.vllm_root), ("sglang", args.sglang_root)):
        if not root.exists():
            continue
        for case_dir in _scan_case_dirs(root):
            try:
                rows.append(_compute_row(engine, case_dir))
            except Exception as exc:
                dataset_id = case_dir.parent.name
                case_name = case_dir.name
                row = _empty_row(engine, dataset_id, case_name, case_dir / "record_meta.json")
                print(f"[warn] schema_blocked {engine} {dataset_id}/{case_name}: {exc}")
                rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[done] wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
