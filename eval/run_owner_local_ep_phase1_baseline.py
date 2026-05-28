from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    run_subprocess_job,
    selected_gpu_group,
    write_csv,
    write_json,
    write_markdown,
)


STAGE_NAME = "owner_local_ep_phase1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_owner_local_args(parser)
    parser.add_argument("--impls", default="ep_ll_triton,ep_ht")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.set_defaults(num_layers=2, num_problems=8, max_tokens=16)
    args = finalize_owner_local_args(parser.parse_args())
    if args.smoke and not args.full:
        args.num_layers = 2
        args.num_problems = 8
        args.max_tokens = 16
    return args


def stage_dirs(run_dir: Path) -> dict[str, Path]:
    dirs = {
        "run_dir": run_dir,
        "logs": run_dir / "logs",
        "profiles": run_dir / "profiles",
        "profile_quality": run_dir / "profile_quality",
        "prepared": run_dir / "prepared",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def recommend_impl(rows: list[dict[str, object]]) -> str | None:
    passing = [row for row in rows if row.get("pass")]
    if not passing:
        return None
    best = min(
        passing,
        key=lambda row: (
            float(row.get("e2e_total_time_s") or 1e18),
            -float(row.get("average_score") or 0.0),
            -float(row.get("profile_pairwise_l1_mean") or 0.0),
        ),
    )
    return str(best.get("impl"))


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_root, STAGE_NAME, args.run_id)
    dirs = stage_dirs(run_dir)
    prepared_path = dirs["prepared"] / f"{args.dataset_label}_owner_local_ep.parquet"
    args.prepared_dataset_path = prepared_path

    dataset_info = None
    if args.prepare_only or not args.dry_run:
        dataset_info = prepare_gsm_dataset_if_needed(args.dataset_path, prepared_path, limit=args.num_problems)
    else:
        dataset_info = {
            "source": str(args.dataset_path),
            "prepared_path": str(prepared_path),
            "num_rows": args.num_problems,
            "prompt_format": "answer_only_plain_number_or_short_expression",
        }

    impls = [item.strip() for item in args.impls.split(",") if item.strip()]
    gpu_group = selected_gpu_group(args)
    commands: list[dict[str, str]] = []
    rows: list[dict[str, object]] = []

    for offset, impl in enumerate(impls):
        job_name = f"{impl}_baseline"
        job_output_dir = dirs["profiles"] / job_name
        cmd = build_test_bazaar_moe_cmd(
            args,
            impl=impl,
            output_dir=job_output_dir,
            master_port=args.master_port + offset,
            profile_enabled=True,
            placement="contiguous",
        )
        commands.append({"name": job_name, "command": " ".join(cmd), "output_dir": str(job_output_dir)})
        if args.prepare_only:
            continue
        env = command_env(
            args,
            cuda_visible_devices=gpu_group,
            run_id=f"{args.run_id}_{job_name}",
            extra_env={"FLASHINFER_WORKSPACE_BASE": str(run_dir / "flashinfer_workspace" / job_name)},
        )
        job_result = run_subprocess_job(
            cmd=cmd,
            env=env,
            log_path=dirs["logs"] / f"{job_name}.log",
            timeout_s=args.timeout_s,
            dry_run=args.dry_run,
        )
        parsed = parse_test_bazaar_moe_outputs(job_result, output_dir=job_output_dir, search_roots=[run_dir, job_output_dir])
        result_json_path = Path(parsed["result_json_path"]) if parsed.get("result_json_path") else None
        profile_path = Path(parsed["profile_path"]) if parsed.get("profile_path") else None
        profile_copy = copy_if_present(profile_path, dirs["profiles"])
        quality = analyze_profile_if_present(profile_copy or profile_path, output_dir=dirs["profile_quality"])
        metrics = parsed.get("metrics", {}) or {}
        profile_payload = json.loads((profile_copy or profile_path).read_text()) if (profile_copy or profile_path) and (profile_copy or profile_path).exists() else {}
        row = {
            "impl": impl,
            "runtime_mode": "owner_local_ep",
            "world_size": args.world_size,
            "tp_size": args.tp_size,
            "dp_size": args.data_parallel_size,
            "ep_size": args.ep_size,
            "e2e_total_time_s": metrics.get("e2e_total_time_s"),
            "prefill_throughput_tok_s": metrics.get("prefill_throughput_tok_s"),
            "decode_throughput_tok_s": metrics.get("decode_throughput_tok_s"),
            "total_generated_tokens": metrics.get("total_generated_tokens"),
            "average_score": metrics.get("average_score"),
            "prefill_tokens": metrics.get("prefill_tokens"),
            "decode_tokens": metrics.get("decode_tokens"),
            "prefill_time_s": metrics.get("prefill_time_s"),
            "decode_time_s": metrics.get("decode_time_s"),
            "steps": len(metrics.get("steps", []) or []),
            "profile_path": str(profile_copy or profile_path) if (profile_copy or profile_path) else None,
            "profile_source_definition": profile_payload.get("source_definition"),
            "profile_src_dim": len(profile_payload.get("traffic", [[[]]])[0]) if profile_payload.get("traffic") else None,
            "result_json_path": parsed.get("result_json_path"),
            "pass": bool(parsed.get("pass")),
            "exit_code": parsed.get("exit_code"),
        }
        rows.append({**row, **quality})

    summary_rows = gather_summary_rows(rows)
    baseline_impl = recommend_impl(summary_rows)
    best_row = next((row for row in summary_rows if row.get("impl") == baseline_impl), None)
    bottleneck = None
    if best_row:
        prefill_time = float(best_row.get("prefill_time_s") or 0.0)
        decode_time = float(best_row.get("decode_time_s") or 0.0)
        bottleneck = "decode" if decode_time >= prefill_time else "prefill"
    findings_bullets = [
        f"Baseline impl candidates: {', '.join(impls)}.",
        (
            "Recommended baseline impl for later placement/replica work: "
            + (baseline_impl or "undetermined")
            + "."
        ),
        (
            "Routing profile source semantics should now be `owner_rank`, avoiding the old repeated-source dp_leader interpretation."
        ),
        (
            "Dominant time on the current best run: "
            + (bottleneck or "unknown")
            + "."
        ),
        (
            "Phase 2 is ready if at least one impl passed and emitted routing_profile_quality.json."
        ),
    ]
    findings_md = render_findings_markdown(
        title="Owner Local EP Phase 1 Findings",
        bullets=findings_bullets,
        rows=summary_rows,
        columns=[
            "impl",
            "pass",
            "average_score",
            "e2e_total_time_s",
            "prefill_throughput_tok_s",
            "decode_throughput_tok_s",
            "profile_identical_fraction_mean",
            "profile_pairwise_l1_mean",
            "profile_pairwise_cosine_mean",
        ],
    )
    manifest = {
        "stage": STAGE_NAME,
        "run_dir": str(run_dir),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "dataset": dataset_info,
        "commands": commands,
    }
    summary = {
        "stage": STAGE_NAME,
        "run_id": args.run_id,
        "runtime_mode": "owner_local_ep",
        "recommended_impl": baseline_impl,
        "bottleneck": bottleneck,
        "rows": summary_rows,
        "can_enter_phase2": any(row.get("pass") and row.get("profile_quality_json_path") for row in summary_rows),
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "summary.csv", summary_rows)
    write_markdown(run_dir / "findings.md", findings_md)


if __name__ == "__main__":
    main()
