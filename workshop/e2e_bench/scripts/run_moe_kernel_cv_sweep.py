#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import math
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _parse_float_list(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated float list")
    return values


def _slugify_cv(cv: float) -> str:
    return f"{cv:.4f}".replace(".", "p")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def _extract_summary(path: Path) -> dict:
    rows = _read_jsonl(path)
    trials = [row for row in rows if row.get("case") == "eplb_off"]
    if not trials:
        raise RuntimeError(f"No eplb_off trial rows found in {path}")

    tps = [float(row["global_tokens_per_s"]) for row in trials]
    ms = [float(row["max_rank_mean_step_ms"]) for row in trials]
    realized_cv = [float(row["realized_physical_cv"]) for row in trials]
    realized_entropy = [float(row["realized_physical_entropy"]) for row in trials]

    return {
        "num_trials": len(trials),
        "mean_realized_physical_cv": _mean(realized_cv),
        "std_realized_physical_cv": _std(realized_cv),
        "mean_realized_physical_entropy": _mean(realized_entropy),
        "std_realized_physical_entropy": _std(realized_entropy),
        "mean_tokens_per_s": _mean(tps),
        "std_tokens_per_s": _std(tps),
        "mean_max_rank_mean_step_ms": _mean(ms),
        "std_max_rank_mean_step_ms": _std(ms),
    }


def _format_float(x: float) -> str:
    return f"{x:.6f}"


def _build_command(args: argparse.Namespace, cv: float, out_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "tests.kernels.moe.modular_kernel_tools.benchmark_eplb_multigpu",
        "--world-size",
        str(args.world_size),
        "--routing-space",
        "physical_gpu",
        "--backend",
        args.backend,
        "--num-experts",
        str(args.num_experts),
        "--topk",
        str(args.topk),
        "-k",
        str(args.hidden_size),
        "-n",
        str(args.intermediate_size),
        "--total-tokens",
        str(args.total_tokens),
        "--target-dest-cv",
        _format_float(cv),
        "--num-random-trials",
        str(args.num_random_trials),
        "--sample-attempts",
        str(args.sample_attempts),
        "--hot-token-frac",
        _format_float(args.hot_token_frac),
        "--hot-expert-frac",
        _format_float(args.hot_expert_frac),
        "--warmup-iters",
        str(args.warmup_iters),
        "--iters",
        str(args.iters),
        "--seed",
        str(args.seed),
        "--mode",
        "off",
        "--append-output",
        "--output-json",
        str(out_path),
    ]
    if args.allow_non_modular:
        cmd.append("--allow-non-modular")
    return cmd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep physical-GPU CV values for the single-layer MoE kernel "
            "benchmark and export a compact summary table."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store per-CV jsonl outputs and the final summary",
    )
    parser.add_argument(
        "--cv-values",
        type=str,
        default="0.05,0.075,0.10,0.125,0.15,0.20,0.30,0.40,0.70,1.00,1.50",
        help="Comma-separated CV values to sweep",
    )
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument(
        "--backend",
        type=str,
        default="allgather_reducescatter",
    )
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument(
        "--topk",
        type=int,
        default=6,
        help="DeepSeek-V2-Lite uses num_experts_per_tok=6",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=2048,
        help="DeepSeek-V2-Lite hidden size",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=1408,
        help="DeepSeek-V2-Lite moe_intermediate_size",
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=131072,
        help="Total source tokens for each synthetic MoE step",
    )
    parser.add_argument(
        "--num-random-trials",
        type=int,
        default=60,
        help="Monte Carlo trials per CV. Larger is more stable but slower.",
    )
    parser.add_argument("--sample-attempts", type=int, default=10000)
    parser.add_argument("--warmup-iters", type=int, default=15)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hot-token-frac", type=float, default=0.8)
    parser.add_argument("--hot-expert-frac", type=float, default=0.375)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="vLLM repo root; benchmark module runs from here",
    )
    parser.add_argument(
        "--allow-non-modular",
        action="store_true",
        help="Pass through to the kernel benchmark for debugging only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print commands; do not execute them",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cv_values = _parse_float_list(args.cv_values)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = args.output_dir / f"cv_sweep_summary_{run_tag}.csv"
    summary_jsonl = args.output_dir / f"cv_sweep_summary_{run_tag}.jsonl"
    fieldnames = [
        "target_cv",
        "num_trials",
        "mean_realized_physical_cv",
        "std_realized_physical_cv",
        "mean_realized_physical_entropy",
        "std_realized_physical_entropy",
        "mean_tokens_per_s",
        "std_tokens_per_s",
        "mean_max_rank_mean_step_ms",
        "std_max_rank_mean_step_ms",
        "raw_jsonl",
    ]

    if not args.dry_run:
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        with summary_jsonl.open("w", encoding="utf-8"):
            pass

    summary_rows: list[dict] = []
    for cv in cv_values:
        out_path = args.output_dir / f"kernel_cv_{_slugify_cv(cv)}.jsonl"
        cmd = _build_command(args, cv, out_path)
        print("+", " ".join(cmd))
        if args.dry_run:
            continue

        subprocess.run(cmd, cwd=args.repo_root, check=True)
        aggregate = _extract_summary(out_path)
        row = {
            "target_cv": cv,
            "num_trials": aggregate["num_trials"],
            "mean_realized_physical_cv": aggregate["mean_realized_physical_cv"],
            "std_realized_physical_cv": aggregate["std_realized_physical_cv"],
            "mean_realized_physical_entropy": aggregate[
                "mean_realized_physical_entropy"
            ],
            "std_realized_physical_entropy": aggregate[
                "std_realized_physical_entropy"
            ],
            "mean_tokens_per_s": aggregate["mean_tokens_per_s"],
            "std_tokens_per_s": aggregate["std_tokens_per_s"],
            "mean_max_rank_mean_step_ms": aggregate[
                "mean_max_rank_mean_step_ms"
            ],
            "std_max_rank_mean_step_ms": aggregate["std_max_rank_mean_step_ms"],
            "raw_jsonl": str(out_path),
        }
        summary_rows.append(row)

        with summary_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
        with summary_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.dry_run:
        return

    print("\nSummary:")
    print(
        "| target_cv | realized_cv | entropy | throughput(tokens/s) | max_step_ms | trials |"
    )
    print("|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        print(
            "| "
            f"{row['target_cv']:.4f} | "
            f"{row['mean_realized_physical_cv']:.4f} | "
            f"{row['mean_realized_physical_entropy']:.4f} | "
            f"{row['mean_tokens_per_s']:.2f} | "
            f"{row['mean_max_rank_mean_step_ms']:.4f} | "
            f"{row['num_trials']} |"
        )

    print(f"\nWrote {summary_csv}")
    print(f"Wrote {summary_jsonl}")


if __name__ == "__main__":
    main()
