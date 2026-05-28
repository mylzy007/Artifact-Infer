"""Phase 1/2 driver for the pre-token-owner MoE exploration plan."""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

from eval.analyze_routing_profile_quality import build_report, load_profile, markdown_report
from eval.generate_moe_placement import (
    _aggregate_profile_traffic,
    _balanced_greedy,
    _comm_greedy,
)
from workshop.nanovllm_moe.services.utils.expert_placement import (
    _local_to_global_for_policy,
    estimated_metrics,
    make_placement_section,
)

DEFAULT_MODEL = Path("/home/lzy/models/Qwen3-30B-A3B")
DEFAULT_DATASET = Path("/home/lzy/datasets/moe_benchmarks/gsm8k/main/test-00000-of-00001.parquet")
DEFAULT_OUTPUT_ROOT = REPO / "eval_results"
DEFAULT_PLACEMENTS = [
    "contiguous",
    "round_robin",
    "fixed_random_shuffle",
    "load_balanced_greedy_with_locality_tiebreak",
    "communication_aware_greedy",
]


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "run"


def shlex_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def parse_list(value: str) -> list[int]:
    items = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not items:
        raise SystemExit("expected at least one integer value")
    return items


def resolve_existing_path(path_value: str | None, *, search_root: Path) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value)
    if candidate.exists():
        return candidate.resolve()
    repo_candidate = (REPO / path_value).resolve()
    if repo_candidate.exists():
        return repo_candidate
    basename = candidate.name
    if not basename:
        return None
    matches = sorted(search_root.rglob(basename))
    if matches:
        return matches[0].resolve()
    return None


def prepare_dataset_if_needed(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.dry_run or args.prepare_only:
        return args.dataset_path

    import datasets

    rows = datasets.load_dataset("parquet", data_files=str(args.dataset_path), split="train")
    if len(rows) > 0 and "prompt" in rows.column_names:
        return args.dataset_path

    from eval.run_moe_stage01_gsm import normalize_gsm_dataset

    prepared_dir = run_dir / "prepared"
    prepared_path = prepared_dir / f"{args.dataset_label}_prepared_{sanitize(args.run_id)}.parquet"
    normalize_gsm_dataset(args.dataset_path, prepared_path, limit=args.num_problems)
    return prepared_path


def aggregate_geometry_profile(profile_report: dict[str, Any]) -> dict[str, Any]:
    summary = profile_report["summary"]
    return {
        "profile_identical_fraction_mean": summary["profile_identical_fraction_mean"],
        "profile_pairwise_l1_mean": summary["profile_pairwise_l1_mean"],
        "profile_pairwise_cosine_mean": summary["profile_pairwise_cosine_mean"],
        "profile_all_layers_identical": summary["profile_all_layers_identical"],
    }


def placement_metrics_for_policy(
    profile: dict[str, Any],
    *,
    policy: str,
    seed: int,
) -> dict[str, Any]:
    ep_size = int(profile["ep_size"])
    num_experts = int(profile["num_experts"])
    traffic = _aggregate_profile_traffic(profile)
    layer_traffic = profile["traffic"]

    if policy in {"contiguous", "round_robin", "fixed_random_shuffle"}:
        local_to_global = _local_to_global_for_policy(num_experts, ep_size, policy, seed=seed)
    elif policy == "load_balanced_greedy_with_locality_tiebreak":
        local_to_global = _balanced_greedy(traffic, ep_size, num_experts)
    elif policy == "communication_aware_greedy":
        local_to_global = _comm_greedy(traffic, ep_size, num_experts)
    else:
        raise ValueError(f"unsupported placement policy: {policy}")

    section = make_placement_section(local_to_global, num_experts)
    return estimated_metrics(traffic, section["expert_to_rank"], ep_size, layer_traffic=layer_traffic)


def run_cmd(cmd: list[str], *, cwd: Path) -> int:
    proc = subprocess.run(cmd, cwd=cwd, text=True)
    return int(proc.returncode)


def build_geometry_cmd(args: argparse.Namespace, *, tp_size: int, run_dir: Path, run_id: str) -> list[str]:
    placements = ",".join(DEFAULT_PLACEMENTS)
    cmd = [
        args.python,
        "-m",
        "eval.run_moe_placement_experiments",
        "--model-path",
        str(args.model_path),
        "--dataset-path",
        str(args.prepared_dataset_path),
        "--dataset-label",
        args.dataset_label,
        "--world-size",
        str(args.world_size),
        "--gpus-per-job",
        str(args.world_size),
        "--parallel-jobs",
        "1",
        "--tp-size",
        str(tp_size),
        "--impls",
        "ep_ht",
        "--profile-impl",
        "ep_ht",
        "--placements",
        placements,
        "--num-layers",
        str(args.num_layers),
        "--num-problems",
        str(args.num_problems),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enforce-eager",
        "1",
        "--moe-ll-overflow-policy",
        "error",
        "--gpu-ids",
        args.gpu_ids,
        "--seed",
        str(args.seed),
        "--base-master-port",
        str(args.base_master_port + tp_size * 100),
        "--port-retries",
        str(args.port_retries),
        "--timeout-s",
        str(args.timeout_s),
        "--experiment-name",
        f"pre_token_owner_tp{tp_size}_ep{args.world_size // tp_size}",
        "--run-id",
        run_id,
        "--output-root",
        str(args.output_root),
        "--run-dir",
        str(run_dir),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def summarize_eval_row(
    *,
    eval_result: dict[str, Any],
    profile_metrics: dict[str, Any],
    placement_metrics: dict[str, Any],
    tp_size: int,
    ep_size: int,
    world_size: int,
) -> dict[str, Any]:
    metrics = eval_result.get("metrics", {}) or {}
    result_json_path = resolve_existing_path(eval_result.get("result_json_path"), search_root=REPO)
    score = metrics.get("average_score")
    if result_json_path is not None:
        try:
            result_payload = json.loads(result_json_path.read_text())
            result_summary = result_payload.get("summary", {}) or {}
            score = result_summary.get("average_score", score)
        except Exception:
            pass
    return {
        "tp_size": tp_size,
        "ep_size": ep_size,
        "world_size": world_size,
        "impl": eval_result.get("impl", "ep_ht"),
        "placement": eval_result.get("placement"),
        "score": score,
        "e2e_s": metrics.get("e2e_total_time_s"),
        "prefill_tok_s": metrics.get("prefill_throughput_tok_s"),
        "decode_tok_s": metrics.get("decode_throughput_tok_s"),
        "total_generated_tokens": eval_result.get("total_generated_tokens"),
        "gpu_cv": placement_metrics.get("estimated_gpu_cv"),
        "gpu_cv_layer_mean": placement_metrics.get("estimated_gpu_cv_layer_mean"),
        "cross_traffic_ratio": placement_metrics.get("estimated_cross_traffic_ratio"),
        "cross_traffic_ratio_layer_mean": placement_metrics.get("estimated_cross_traffic_ratio_layer_mean"),
        "profile_identical_fraction_mean": profile_metrics.get("profile_identical_fraction_mean"),
        "profile_pairwise_l1_mean": profile_metrics.get("profile_pairwise_l1_mean"),
        "profile_pairwise_cosine_mean": profile_metrics.get("profile_pairwise_cosine_mean"),
        "profile_all_layers_identical": profile_metrics.get("profile_all_layers_identical"),
        "pass": eval_result.get("pass"),
        "exit_code": eval_result.get("exit_code"),
        "log_path": eval_result.get("log_path"),
        "result_json_path": str(result_json_path) if result_json_path is not None else eval_result.get("result_json_path"),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tp_size",
        "ep_size",
        "world_size",
        "impl",
        "placement",
        "score",
        "e2e_s",
        "prefill_tok_s",
        "decode_tok_s",
        "total_generated_tokens",
        "gpu_cv",
        "gpu_cv_layer_mean",
        "cross_traffic_ratio",
        "cross_traffic_ratio_layer_mean",
        "profile_identical_fraction_mean",
        "profile_pairwise_l1_mean",
        "profile_pairwise_cosine_mean",
        "profile_all_layers_identical",
        "pass",
        "exit_code",
        "log_path",
        "result_json_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def geometry_summary(rows: list[dict[str, Any]], tp_size: int, ep_size: int) -> dict[str, Any]:
    e2e_values = [float(r["e2e_s"]) for r in rows if r.get("e2e_s") is not None]
    contiguous = next((r for r in rows if r["placement"] == "contiguous"), None)
    cross_values = [float(r["cross_traffic_ratio"]) for r in rows if r.get("cross_traffic_ratio") is not None]
    return {
        "tp_size": tp_size,
        "ep_size": ep_size,
        "num_rows": len(rows),
        "mean_e2e_s": statistics.mean(e2e_values) if e2e_values else None,
        "contiguous_e2e_s": contiguous.get("e2e_s") if contiguous else None,
        "contiguous_decode_tok_s": contiguous.get("decode_tok_s") if contiguous else None,
        "profile_identical_fraction_mean": rows[0]["profile_identical_fraction_mean"] if rows else None,
        "profile_pairwise_cosine_mean": rows[0]["profile_pairwise_cosine_mean"] if rows else None,
        "placement_cross_traffic_span": (max(cross_values) - min(cross_values)) if len(cross_values) >= 2 else 0.0,
        "all_pass": all(bool(r.get("pass")) for r in rows),
    }


def choose_best_runtime(geometry_rows: dict[tuple[int, int], list[dict[str, Any]]]) -> tuple[int, int] | None:
    candidates = []
    for key, rows in geometry_rows.items():
        summary = geometry_summary(rows, key[0], key[1])
        e2e_s = summary["contiguous_e2e_s"] if summary["contiguous_e2e_s"] is not None else summary["mean_e2e_s"]
        if e2e_s is None:
            continue
        candidates.append((float(e2e_s), key))
    if not candidates:
        return None
    return min(candidates)[1]


def decide_tp2_baseline(geometry_summaries: dict[tuple[int, int], dict[str, Any]]) -> str:
    tp1 = geometry_summaries.get((1, 8))
    tp2 = geometry_summaries.get((2, 4))
    tp4 = geometry_summaries.get((4, 2))
    if not tp2:
        return "insufficient data"
    if tp1 and not tp4:
        better_than_tp1 = (
            tp2["profile_identical_fraction_mean"] is not None
            and tp1["profile_identical_fraction_mean"] is not None
            and float(tp2["profile_identical_fraction_mean"]) < float(tp1["profile_identical_fraction_mean"])
        )
        return "yes_on_available_data" if better_than_tp1 else "partial"
    if tp1 and tp4:
        better_than_tp1 = (
            tp2["profile_identical_fraction_mean"] is not None
            and tp1["profile_identical_fraction_mean"] is not None
            and float(tp2["profile_identical_fraction_mean"]) < float(tp1["profile_identical_fraction_mean"])
        )
        preserves_more_headroom_than_tp4 = float(tp2["placement_cross_traffic_span"]) >= float(tp4["placement_cross_traffic_span"])
        if better_than_tp1 and preserves_more_headroom_than_tp4:
            return "yes"
        if better_than_tp1:
            return "maybe"
        return "no"
    return "partial"


def generate_findings(
    *,
    geometry_rows: dict[tuple[int, int], list[dict[str, Any]]],
    top_level: dict[str, Any],
) -> str:
    geometry_summaries = {
        key: geometry_summary(rows, key[0], key[1]) for key, rows in geometry_rows.items()
    }
    worst_profile = None
    ranked_profile = sorted(
        (
            (summary["profile_identical_fraction_mean"], key)
            for key, summary in geometry_summaries.items()
            if summary["profile_identical_fraction_mean"] is not None
        ),
        reverse=True,
    )
    if ranked_profile:
        worst_profile = ranked_profile[0][1]
    best_runtime = choose_best_runtime(geometry_rows)
    tp2_decision = decide_tp2_baseline(geometry_summaries)

    lines = [
        f"# pre_token_owner_findings_{top_level['run_id']}",
        "",
        "## Scope",
        "",
        f"- Run id: `{top_level['run_id']}`",
        f"- Output root: `{top_level['run_dir']}`",
        f"- Mode: `{'dry_run' if top_level['dry_run'] else 'executed'}`",
        f"- Workload: `num_problems={top_level['args']['num_problems']}, max_tokens={top_level['args']['max_tokens']}, max_model_len={top_level['args']['max_model_len']}, max_num_batched_tokens={top_level['args']['max_num_batched_tokens']}, max_num_seqs={top_level['args']['max_num_seqs']}`",
        "",
        "## Geometry Summary",
        "",
        "| geometry | identical_fraction_mean | pairwise_cosine_mean | contiguous_e2e_s | placement_cross_traffic_span | all_pass |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in sorted(geometry_summaries):
        summary = geometry_summaries[key]
        lines.append(
            f"| TP={key[0]},EP={key[1]} | {summary['profile_identical_fraction_mean']} | "
            f"{summary['profile_pairwise_cosine_mean']} | {summary['contiguous_e2e_s']} | "
            f"{summary['placement_cross_traffic_span']} | {summary['all_pass']} |"
        )

    lines.extend(
        [
            "",
            "## Findings",
            "",
            (
                f"- Profile degeneracy is worst under `TP={worst_profile[0]}, EP={worst_profile[1]}` "
                f"based on the highest `profile_identical_fraction_mean`."
                if worst_profile
                else "- Profile degeneracy could not be ranked because no completed profile analysis was available."
            ),
            (
                f"- The most reasonable runtime baseline in this run is `TP={best_runtime[0]}, EP={best_runtime[1]}` "
                f"based on the lowest contiguous/mean `e2e_s` among completed geometries."
                if best_runtime
                else "- Runtime comparison is incomplete because no geometry produced comparable timing metrics."
            ),
            f"- `TP=2, EP=4` as the next main baseline: `{tp2_decision}`.",
            "- Trustworthy now: routing-profile row diagnostics, end-to-end runtime on the same workload, and relative placement estimates from the same profile source.",
            "- Proxy-only now: placement metrics (`gpu_cv`, `cross_traffic_ratio`) remain offline estimates, and answer score is still confounded by short-generation / formatting effects from the current runtime.",
        ]
    )
    failed_geometries = []
    for g in top_level.get("geometries", []):
        if g.get("completed_successfully", False):
            continue
        label = f"TP={g['tp_size']},EP={g['ep_size']}"
        if g.get("error"):
            label += f" ({g['error']})"
        failed_geometries.append(label)
    if failed_geometries:
        lines.append(f"- Incomplete geometries in this run: `{failed_geometries}`.")

    if worst_profile and best_runtime:
        recommend_refactor = "yes" if worst_profile != best_runtime or tp2_decision in {"yes", "maybe"} else "maybe"
    else:
        recommend_refactor = "partial"
    lines.append(f"- Worth entering token-owner refactor: `{recommend_refactor}`.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/home/lzy/miniconda3/envs/vllm/bin/python")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-label", default="gsm8k")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--tp-sizes", default="1,2,4")
    parser.add_argument("--num-problems", type=int, default=112)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--base-master-port", type=int, default=29600)
    parser.add_argument("--port-retries", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=7200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--num-layers", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tp_sizes = parse_list(args.tp_sizes)
    if args.world_size != 8:
        raise SystemExit("this Phase 2 sweep is currently defined for world_size=8")
    for tp_size in tp_sizes:
        if args.world_size % tp_size != 0:
            raise SystemExit(f"world_size={args.world_size} must be divisible by tp_size={tp_size}")

    run_id = sanitize(args.run_id)
    run_dir = args.output_root / f"pre_token_owner_exploration_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prepared_dataset_path = prepare_dataset_if_needed(args, run_dir)
    args.prepared_dataset_path = prepared_dataset_path

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "dry_run": bool(args.dry_run),
        "prepare_only": bool(args.prepare_only),
        "args": {
            "model_path": str(args.model_path),
            "dataset_path": str(args.dataset_path),
            "prepared_dataset_path": str(prepared_dataset_path),
            "dataset_label": args.dataset_label,
            "world_size": args.world_size,
            "tp_sizes": tp_sizes,
            "num_problems": args.num_problems,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "num_layers": args.num_layers,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "gpu_ids": args.gpu_ids,
        },
        "geometries": [],
    }

    summary_rows: list[dict[str, Any]] = []
    geometry_rows: dict[tuple[int, int], list[dict[str, Any]]] = {}

    for tp_size in tp_sizes:
        ep_size = args.world_size // tp_size
        geometry_dir = run_dir / f"tp{tp_size}_ep{ep_size}"
        geometry_run_id = f"{run_id}_tp{tp_size}_ep{ep_size}"
        cmd = build_geometry_cmd(args, tp_size=tp_size, run_dir=geometry_dir, run_id=geometry_run_id)
        geometry_record: dict[str, Any] = {
            "tp_size": tp_size,
            "ep_size": ep_size,
            "run_dir": str(geometry_dir.resolve()),
            "command": shlex_join(cmd),
            "completed_successfully": False,
        }

        if not args.prepare_only:
            summary_path = geometry_dir / "summary.json"
            exit_code = 0
            if not (args.reuse_existing and summary_path.exists()):
                exit_code = run_cmd(cmd, cwd=REPO)
            if args.dry_run:
                if summary_path.exists():
                    geometry_record["summary_path"] = str(summary_path.resolve())
                geometry_record["exit_code"] = exit_code
                manifest["geometries"].append(geometry_record)
                continue
            if not summary_path.exists():
                geometry_record["exit_code"] = exit_code
                geometry_record["error"] = f"missing summary after geometry run: {summary_path}"
                manifest["geometries"].append(geometry_record)
                if not args.continue_on_failure:
                    raise SystemExit(1)
                continue
            geometry_summary_payload = json.loads(summary_path.read_text())
            geometry_record["summary_path"] = str(summary_path.resolve())
            geometry_record["summary"] = geometry_summary_payload
            geometry_record["exit_code"] = exit_code

            profile_entry = geometry_summary_payload.get("profile") or {}
            profile_path = resolve_existing_path(profile_entry.get("profile_path"), search_root=geometry_dir)
            if profile_path is None:
                detail = profile_entry.get("exception_summary") or []
                detail_text = "; ".join(str(x) for x in detail[-3:]) if detail else ""
                geometry_record["error"] = (
                    f"could not locate profile json for TP={tp_size}, EP={ep_size}"
                    + (f": {detail_text}" if detail_text else "")
                )
                manifest["geometries"].append(geometry_record)
                if not args.continue_on_failure:
                    raise SystemExit(1)
                continue
            profile = load_profile(profile_path)
            profile_report = build_report(profile, profile_path)
            profile_json_path = geometry_dir / "routing_profile_quality.json"
            profile_md_path = geometry_dir / "routing_profile_quality.md"
            write_json(profile_json_path, profile_report)
            profile_md_path.write_text(markdown_report(profile_report))
            geometry_record["profile_analysis_json"] = str(profile_json_path.resolve())
            geometry_record["profile_analysis_md"] = str(profile_md_path.resolve())

            profile_metrics = aggregate_geometry_profile(profile_report)
            placement_metric_cache = {
                policy: placement_metrics_for_policy(profile, policy=policy, seed=args.seed)
                for policy in DEFAULT_PLACEMENTS
            }
            rows = []
            for eval_result in geometry_summary_payload.get("eval_results", []):
                placement = eval_result.get("placement")
                if placement not in placement_metric_cache:
                    continue
                row = summarize_eval_row(
                    eval_result=eval_result,
                    profile_metrics=profile_metrics,
                    placement_metrics=placement_metric_cache[placement],
                    tp_size=tp_size,
                    ep_size=ep_size,
                    world_size=args.world_size,
                )
                rows.append(row)
                summary_rows.append(row)
            geometry_rows[(tp_size, ep_size)] = rows
            geometry_record["completed_successfully"] = bool(
                exit_code == 0 and rows and all(r.get("pass") for r in rows)
            )
        manifest["geometries"].append(geometry_record)

    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)

    if args.prepare_only or args.dry_run:
        mode = "dry-run" if args.dry_run else "prepare-only"
        print(f"[{mode}] wrote {manifest_path}")
        return

    summary_json_path = run_dir / "pre_token_owner_summary.json"
    summary_csv_path = run_dir / "pre_token_owner_summary.csv"
    findings_path = run_dir / f"pre_token_owner_findings_{run_id}.md"
    write_json(summary_json_path, {"run_id": run_id, "rows": summary_rows})
    write_csv(summary_csv_path, summary_rows)
    findings_path.write_text(generate_findings(geometry_rows=geometry_rows, top_level=manifest))
    print(f"[exploration] wrote {summary_json_path}")
    print(f"[exploration] wrote {summary_csv_path}")
    print(f"[exploration] wrote {findings_path}")


if __name__ == "__main__":
    main()
