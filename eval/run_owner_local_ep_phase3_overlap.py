from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from eval.utils_owner_local_ep import (
    add_common_owner_local_args,
    analyze_profile_if_present,
    build_test_bazaar_moe_cmd,
    command_env,
    copy_if_present,
    finalize_owner_local_args,
    gather_summary_rows,
    make_run_dir,
    parse_test_bazaar_moe_outputs,
    prepare_gsm_dataset_if_needed,
    render_findings_markdown,
    resolve_existing_path,
    run_subprocess_job,
    selected_gpu_group,
    write_csv,
    write_json,
    write_markdown,
)


STAGE_NAME = "owner_local_ep_phase3_overlap"
DEFAULT_BASE_PLACEMENTS = [
    "round_robin",
    "load_balanced_greedy_with_locality_tiebreak",
]
DEFAULT_OVERLAPS = [0.0, 0.25]
DEFAULT_REPLICA_PLACEMENT_POLICIES = [
    "consecutive_from_primary",
    "numa_local_first",
    "traffic_aware_greedy",
    "traffic_plus_balance_greedy",
]
DEFAULT_DISPATCH_POLICIES = [
    "greedy_balance",
    "min_communication",
    "hybrid",
    "numa_aware_min_communication",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_owner_local_args(parser)
    parser.add_argument("--phase2-summary", type=Path, default=None)
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--impl", default=None)
    parser.add_argument("--base-placements", default=",".join(DEFAULT_BASE_PLACEMENTS))
    parser.add_argument("--overlaps", default="0,0.25")
    parser.add_argument(
        "--replica-placement-policies",
        default=",".join(DEFAULT_REPLICA_PLACEMENT_POLICIES),
    )
    parser.add_argument("--dispatch-policies", default=",".join(DEFAULT_DISPATCH_POLICIES))
    parser.set_defaults(num_layers=-1, num_problems=256, max_tokens=64, max_num_seqs=8, enforce_eager=1)
    return finalize_owner_local_args(parser.parse_args())


def stage_dirs(run_dir: Path) -> dict[str, Path]:
    dirs = {
        "run_dir": run_dir,
        "logs": run_dir / "logs",
        "profiles": run_dir / "profiles",
        "profile_quality": run_dir / "profile_quality",
        "overlap_plans": run_dir / "overlap_plans",
        "overlap_stats": run_dir / "overlap_stats",
        "results": run_dir / "results",
        "prepared": run_dir / "prepared",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def parse_csv_str(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_csv_float(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def latest_phase2_summary(output_root: Path) -> Path | None:
    candidates = sorted(output_root.glob("owner_local_ep_phase2_*/summary.json"), reverse=True)
    return candidates[0] if candidates else None


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def select_base_rows(summary: dict[str, Any], requested: list[str]) -> list[dict[str, Any]]:
    rows = [row for row in summary.get("rows", []) if row.get("pass")]
    out = []
    for placement in requested:
        match = next((row for row in rows if row.get("placement") == placement), None)
        if match is not None:
            out.append(match)
    return out


def run_generator(cmd: list[str], *, log_path: Path, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("$ " + " ".join(cmd) + "\n")
        return {"command": " ".join(cmd), "pass": True, "exit_code": None, "dry_run": True}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    text = log_path.read_text(errors="replace")
    start = text.find("{")
    end = text.rfind("}")
    payload: dict[str, Any] = {}
    if start >= 0 and end >= start:
        payload = json.loads(text[start : end + 1])
    return {"command": " ".join(cmd), "pass": proc.returncode == 0, "exit_code": proc.returncode, **payload}


def build_overlap_plan_cmd(
    args: argparse.Namespace,
    *,
    profile_path: Path,
    placement_json_path: Path | None,
    base_placement: str,
    overlap: float,
    replica_placement_policy: str,
    dispatch_policy: str,
    output_dir: Path,
    case_id: str,
) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "eval.generate_moe_overlap_plan",
        "--profile",
        str(profile_path),
        "--base-placement",
        base_placement,
        "--overlap",
        str(overlap),
        "--strategy",
        dispatch_policy,
        "--replica-placement-policy",
        replica_placement_policy,
        "--output-dir",
        str(output_dir),
        "--case-id",
        case_id,
    ]
    if placement_json_path is not None:
        cmd += ["--placement-json", str(placement_json_path)]
    return cmd


def best_runtime_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [row for row in rows if row.get("pass")]
    if not passing:
        return None
    return min(
        passing,
        key=lambda row: (
            float(row.get("e2e_s") or 1e18),
            -float(row.get("score") or 0.0),
            float(row.get("cross_traffic_ratio") or 1e18),
        ),
    )


def build_row(
    *,
    args: argparse.Namespace,
    case_id: str,
    base_placement: str,
    plan_payload: dict[str, Any],
    plan_path: Path | None,
    runtime_parsed: dict[str, Any] | None,
    overlap_stats_payload: dict[str, Any] | None,
    profile_quality: dict[str, Any],
) -> dict[str, Any]:
    metrics = ((runtime_parsed or {}).get("metrics") or {})
    estimated_metrics = dict(plan_payload.get("estimated_metrics") or {})
    return {
        "case_id": case_id,
        "impl": args.impl,
        "runtime_mode": "owner_local_ep",
        "base_placement": base_placement,
        "replica_placement_policy": plan_payload.get("replica_placement_policy"),
        "overlap": float(plan_payload.get("overlap", 0.0)),
        "effective_local_expert_fraction": plan_payload.get("effective_local_expert_fraction"),
        "replicas_per_expert": plan_payload.get("replicas_per_expert"),
        "routing_strategy": plan_payload.get("routing_strategy"),
        "overlap_plan_json_path": str(plan_path) if plan_path else None,
        "score": metrics.get("average_score"),
        "e2e_s": metrics.get("e2e_total_time_s"),
        "prefill_tok_s": metrics.get("prefill_throughput_tok_s"),
        "decode_tok_s": metrics.get("decode_throughput_tok_s"),
        "total_generated_tokens": metrics.get("total_generated_tokens"),
        "cross_traffic_ratio": (overlap_stats_payload or {}).get("cross_traffic_ratio"),
        "local_hit_ratio": (overlap_stats_payload or {}).get("local_hit_ratio"),
        "cross_numa_ratio": (overlap_stats_payload or {}).get("cross_numa_ratio"),
        "gpu_cv": (overlap_stats_payload or {}).get("gpu_cv"),
        "per_rank_compute_load": (overlap_stats_payload or {}).get("per_rank_compute_load"),
        "per_src_outgoing": (overlap_stats_payload or {}).get("per_src_outgoing"),
        "per_dst_incoming": (overlap_stats_payload or {}).get("per_dst_incoming"),
        "routing_profile_path": metrics.get("routing_profile_path"),
        "overlap_runtime_stats_path": metrics.get("overlap_runtime_stats_path"),
        "estimated_cross_traffic_ratio": estimated_metrics.get("estimated_cross_traffic_ratio"),
        "estimated_local_hit_ratio": estimated_metrics.get("local_hit_ratio"),
        "estimated_cross_numa_ratio": estimated_metrics.get("estimated_cross_numa_ratio"),
        "estimated_gpu_cv": estimated_metrics.get("estimated_gpu_cv"),
        "replica_host_entropy": estimated_metrics.get("replica_host_entropy"),
        "source_to_host_affinity_gain": estimated_metrics.get("source_to_host_affinity_gain"),
        "effective_gain_per_extra_memory": estimated_metrics.get("effective_gain_per_extra_memory"),
        "profile_identical_fraction_mean": profile_quality.get("profile_identical_fraction_mean"),
        "profile_pairwise_l1_mean": profile_quality.get("profile_pairwise_l1_mean"),
        "profile_pairwise_cosine_mean": profile_quality.get("profile_pairwise_cosine_mean"),
        "pass": bool((runtime_parsed or {}).get("pass")),
        "exit_code": (runtime_parsed or {}).get("exit_code"),
        "result_json_path": (runtime_parsed or {}).get("result_json_path"),
        "log_path": (runtime_parsed or {}).get("log_path"),
    }


def main() -> None:
    args = parse_args()
    phase2_summary_path = args.phase2_summary or latest_phase2_summary(args.output_root)
    if phase2_summary_path is None or not phase2_summary_path.exists():
        raise SystemExit("need --phase2-summary or an existing owner_local_ep_phase2 summary.json")
    phase2_summary = load_json(phase2_summary_path)
    base_rows = select_base_rows(phase2_summary, parse_csv_str(args.base_placements))
    if not base_rows:
        raise SystemExit("no matching Phase 2 base placements found")

    lead = base_rows[0]
    args.impl = args.impl or str(phase2_summary.get("impl") or lead.get("impl") or "ep_ht")
    args.world_size = int(lead.get("world_size") or args.world_size)
    args.tp_size = int(lead.get("tp_size") or args.tp_size)
    args.data_parallel_size = int(lead.get("dp_size") or args.data_parallel_size)
    args.ep_size = int(lead.get("ep_size") or args.ep_size)

    run_dir = make_run_dir(args.output_root, STAGE_NAME, args.run_id)
    dirs = stage_dirs(run_dir)
    prepared_path = dirs["prepared"] / f"{args.dataset_label}_owner_local_ep.parquet"
    args.prepared_dataset_path = prepared_path

    if args.prepare_only or not args.dry_run:
        dataset_info = prepare_gsm_dataset_if_needed(args.dataset_path, prepared_path, limit=args.num_problems)
    else:
        dataset_info = {
            "source": str(args.dataset_path),
            "prepared_path": str(prepared_path),
            "num_rows": args.num_problems,
            "prompt_format": "answer_only_plain_number_or_short_expression",
        }

    commands: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    overlaps = parse_csv_float(args.overlaps)
    dispatch_policies = parse_csv_str(args.dispatch_policies)
    replica_placement_policies = parse_csv_str(args.replica_placement_policies)
    gpu_group = selected_gpu_group(args)
    port_offset = 0

    for base_row in base_rows:
        base_placement = str(base_row.get("placement"))
        profile_path = args.profile or resolve_existing_path(
            base_row.get("profile_path"),
            search_roots=[phase2_summary_path.parent, args.output_root],
        )
        if profile_path is None:
            raise SystemExit(f"could not resolve base profile path for placement {base_placement}")
        placement_json_path = resolve_existing_path(
            base_row.get("placement_json_path"),
            search_roots=[phase2_summary_path.parent, args.output_root],
        )
        profile_copy = copy_if_present(profile_path, dirs["profiles"]) or profile_path
        profile_quality = analyze_profile_if_present(profile_copy, output_dir=dirs["profile_quality"])

        for overlap in overlaps:
            overlap_str = str(overlap).replace(".", "p")
            case_replica_policies = ["consecutive_from_primary"] if overlap == 0.0 else replica_placement_policies
            case_dispatch_policies = ["disjoint"] if overlap == 0.0 else dispatch_policies
            for replica_placement_policy in case_replica_policies:
                for dispatch_policy in case_dispatch_policies:
                    case_id = f"{base_placement}__{replica_placement_policy}__ov{overlap_str}_{dispatch_policy}"
                    plan_cmd = build_overlap_plan_cmd(
                        args,
                        profile_path=profile_copy,
                        placement_json_path=placement_json_path,
                        base_placement=base_placement,
                        overlap=overlap,
                        replica_placement_policy=replica_placement_policy,
                        dispatch_policy=dispatch_policy,
                        output_dir=dirs["overlap_plans"],
                        case_id=case_id,
                    )
                    commands.append({"name": f"plan_{case_id}", "command": " ".join(plan_cmd)})
                    plan_result = None if args.prepare_only else run_generator(
                        plan_cmd,
                        log_path=dirs["logs"] / f"plan_{case_id}.log",
                        dry_run=args.dry_run,
                    )
                    plan_path = Path(plan_result["overlap_plan_path"]) if plan_result and plan_result.get("overlap_plan_path") else None
                    plan_payload = load_json(plan_path)

                    runtime_output_dir = dirs["results"] / case_id
                    runtime_cmd = build_test_bazaar_moe_cmd(
                        args,
                        impl=args.impl,
                        output_dir=runtime_output_dir,
                        master_port=args.master_port + port_offset,
                        profile_enabled=True,
                        overlap_runtime_enabled=True,
                        placement=base_placement,
                        placement_path=placement_json_path,
                        overlap_path=plan_path,
                        overlap_value=overlap,
                        overlap_strategy=dispatch_policy,
                    )
                    port_offset += 1
                    commands.append({"name": f"runtime_{case_id}", "command": " ".join(runtime_cmd)})

                    runtime_parsed: dict[str, Any] | None = None
                    overlap_stats_payload: dict[str, Any] | None = None
                    if not args.prepare_only:
                        runtime_job = run_subprocess_job(
                            cmd=runtime_cmd,
                            env=command_env(
                                args,
                                cuda_visible_devices=gpu_group,
                                run_id=f"{args.run_id}_{case_id}",
                                extra_env={"FLASHINFER_WORKSPACE_BASE": str(run_dir / "flashinfer_workspace")},
                            ),
                            log_path=dirs["logs"] / f"runtime_{case_id}.log",
                            timeout_s=args.timeout_s,
                            dry_run=args.dry_run,
                        )
                        runtime_parsed = parse_test_bazaar_moe_outputs(
                            runtime_job,
                            output_dir=runtime_output_dir,
                            search_roots=[run_dir, runtime_output_dir],
                        )
                        metrics = (runtime_parsed.get("metrics") or {})
                        overlap_stats_path = resolve_existing_path(
                            metrics.get("overlap_runtime_stats_path"),
                            search_roots=[run_dir, runtime_output_dir],
                        )
                        copied_overlap_stats = copy_if_present(overlap_stats_path, dirs["overlap_stats"])
                        overlap_stats_payload = load_json(copied_overlap_stats or overlap_stats_path)
                        copied_profile = copy_if_present(
                            resolve_existing_path(metrics.get("routing_profile_path"), search_roots=[run_dir, runtime_output_dir]),
                            dirs["profiles"],
                        )
                        if copied_profile is not None:
                            runtime_parsed.setdefault("metrics", {})
                            runtime_parsed["metrics"]["routing_profile_path"] = str(copied_profile)
                        if copied_overlap_stats is not None:
                            runtime_parsed.setdefault("metrics", {})
                            runtime_parsed["metrics"]["overlap_runtime_stats_path"] = str(copied_overlap_stats)

                    rows.append(
                        build_row(
                            args=args,
                            case_id=case_id,
                            base_placement=base_placement,
                            plan_payload=plan_payload,
                            plan_path=plan_path,
                            runtime_parsed=runtime_parsed,
                            overlap_stats_payload=overlap_stats_payload,
                            profile_quality=profile_quality,
                        )
                    )

    summary_rows = gather_summary_rows(rows)
    best = best_runtime_row(summary_rows)
    findings_bullets = [
        f"Phase 3 overlap runtime impl: {args.impl}.",
        "This sweep jointly varies base placement, replica placement policy, and dispatch policy while reusing the existing owner_local_ep + ep_ht static overlap runtime.",
        "overlap=0 rows are disjoint baselines; overlap=0.25 rows use the requested replica placement policies and dispatch policies.",
        (
            "Best measured runtime case: "
            + (
                f"{best['base_placement']} / {best['replica_placement_policy']} / {best['routing_strategy']}"
                if best
                else "none"
            )
            + "."
        ),
    ]
    findings_md = render_findings_markdown(
        title="Owner Local EP Phase 3 Overlap Findings",
        bullets=findings_bullets,
        rows=summary_rows,
        columns=[
            "base_placement",
            "replica_placement_policy",
            "overlap",
            "routing_strategy",
            "score",
            "e2e_s",
            "prefill_tok_s",
            "decode_tok_s",
            "cross_traffic_ratio",
            "local_hit_ratio",
            "cross_numa_ratio",
            "gpu_cv",
            "pass",
        ],
    )
    manifest = {
        "stage": STAGE_NAME,
        "run_dir": str(run_dir),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "phase2_summary_path": str(phase2_summary_path),
        "dataset": dataset_info,
        "commands": commands,
    }
    summary = {
        "stage": STAGE_NAME,
        "run_id": args.run_id,
        "runtime_mode": "owner_local_ep",
        "impl": args.impl,
        "best_runtime_case": best,
        "rows": summary_rows,
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "summary.csv", summary_rows)
    write_markdown(run_dir / "findings.md", findings_md)


if __name__ == "__main__":
    main()
