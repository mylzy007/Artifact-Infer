from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from eval.utils_owner_local_ep import (
    DEFAULT_PLACEMENTS,
    add_common_owner_local_args,
    analyze_profile_if_present,
    build_generate_placement_cmd,
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


STAGE_NAME = "owner_local_ep_phase2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_owner_local_args(parser)
    parser.add_argument("--impl", default=None)
    parser.add_argument("--placements", default=",".join(DEFAULT_PLACEMENTS))
    parser.add_argument("--seed", type=int, default=1234)
    parser.set_defaults(num_layers=2, num_problems=8, max_tokens=16)
    args = finalize_owner_local_args(parser.parse_args())
    args.enforce_eager = 1
    return args


def stage_dirs(run_dir: Path) -> dict[str, Path]:
    dirs = {
        "run_dir": run_dir,
        "logs": run_dir / "logs",
        "profiles": run_dir / "profiles",
        "profile_quality": run_dir / "profile_quality",
        "placements": run_dir / "placements",
        "prepared": run_dir / "prepared",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def resolve_default_impl(output_root: Path) -> str:
    candidates = sorted(output_root.glob("owner_local_ep_phase1_*/summary.json"), reverse=True)
    for path in candidates:
        try:
            payload = json.loads(path.read_text())
            impl = payload.get("recommended_impl")
            if impl:
                return str(impl)
        except Exception:
            continue
    return "ep_ht"


def run_generator(cmd: list[str], *, log_path: Path, dry_run: bool) -> dict[str, object]:
    if dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("$ " + " ".join(cmd) + "\n")
        return {"command": " ".join(cmd), "pass": True, "exit_code": None, "dry_run": True}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=subprocess.STDOUT, text=True)
    text = log_path.read_text(errors="replace")
    start = text.find("{")
    end = text.rfind("}")
    payload = {}
    if start >= 0 and end >= start:
        payload = json.loads(text[start : end + 1])
    return {
        "command": " ".join(cmd),
        "pass": proc.returncode == 0,
        "exit_code": proc.returncode,
        **payload,
    }


def main() -> None:
    args = parse_args()
    if args.impl is None:
        args.impl = resolve_default_impl(args.output_root)
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

    placements = [item.strip() for item in args.placements.split(",") if item.strip()]
    gpu_group = selected_gpu_group(args)
    commands: list[dict[str, str]] = []
    rows: list[dict[str, object]] = []

    base_output_dir = dirs["profiles"] / "baseline_contiguous"
    base_cmd = build_test_bazaar_moe_cmd(
        args,
        impl=args.impl,
        output_dir=base_output_dir,
        master_port=args.master_port,
        profile_enabled=True,
        placement="contiguous",
    )
    commands.append({"name": "baseline_contiguous", "command": " ".join(base_cmd), "output_dir": str(base_output_dir)})
    base_profile_path: Path | None = None
    if not args.prepare_only:
        base_job = run_subprocess_job(
            cmd=base_cmd,
            env=command_env(
                args,
                cuda_visible_devices=gpu_group,
                run_id=f"{args.run_id}_baseline_contiguous",
                extra_env={"FLASHINFER_WORKSPACE_BASE": str(run_dir / "flashinfer_workspace" / "baseline_contiguous")},
            ),
            log_path=dirs["logs"] / "baseline_contiguous.log",
            timeout_s=args.timeout_s,
            dry_run=args.dry_run,
        )
        base_parsed = parse_test_bazaar_moe_outputs(base_job, output_dir=base_output_dir, search_roots=[run_dir, base_output_dir])
        base_profile_path = Path(base_parsed["profile_path"]) if base_parsed.get("profile_path") else None
        if base_profile_path is not None:
            base_profile_path = copy_if_present(base_profile_path, dirs["profiles"]) or base_profile_path

    placement_paths: dict[str, Path | None] = {}
    placement_generation: dict[str, dict[str, object]] = {}
    for placement in placements:
        cmd = None
        if base_profile_path is not None or args.dry_run:
            profile_for_gen = base_profile_path or (dirs["profiles"] / "baseline_contiguous" / "moe_routing_profile_DRY_RUN.json")
            cmd = build_generate_placement_cmd(args, profile_path=profile_for_gen, placement=placement, output_dir=dirs["placements"])
            commands.append({"name": f"generate_{placement}", "command": " ".join(cmd), "output_dir": str(dirs["placements"])})
        if args.prepare_only:
            continue
        if cmd is None:
            raise SystemExit("baseline contiguous run did not produce a profile for placement generation")
        result = run_generator(cmd, log_path=dirs["logs"] / f"generate_{placement}.log", dry_run=args.dry_run)
        placement_generation[placement] = result
        placement_path = Path(result["placement_path"]) if result.get("placement_path") else None
        placement_paths[placement] = placement_path

    for offset, placement in enumerate(placements, start=1):
        job_name = f"{args.impl}_{placement}"
        job_output_dir = dirs["profiles"] / job_name
        placement_path = placement_paths.get(placement)
        cmd = build_test_bazaar_moe_cmd(
            args,
            impl=args.impl,
            output_dir=job_output_dir,
            master_port=args.master_port + offset,
            profile_enabled=True,
            placement=placement,
            placement_path=placement_path,
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
        profile_path = Path(parsed["profile_path"]) if parsed.get("profile_path") else None
        profile_copy = copy_if_present(profile_path, dirs["profiles"])
        quality = analyze_profile_if_present(profile_copy or profile_path, output_dir=dirs["profile_quality"])
        metrics = parsed.get("metrics", {}) or {}
        profile_payload = json.loads((profile_copy or profile_path).read_text()) if (profile_copy or profile_path) and (profile_copy or profile_path).exists() else {}
        estimated = dict((placement_generation.get(placement) or {}).get("estimated_metrics", {}) or {})
        row = {
            "placement": placement,
            "impl": args.impl,
            "runtime_mode": "owner_local_ep",
            "world_size": args.world_size,
            "tp_size": args.tp_size,
            "dp_size": args.data_parallel_size,
            "ep_size": args.ep_size,
            "score": metrics.get("average_score"),
            "e2e_s": metrics.get("e2e_total_time_s"),
            "prefill_tok_s": metrics.get("prefill_throughput_tok_s"),
            "decode_tok_s": metrics.get("decode_throughput_tok_s"),
            "total_generated_tokens": metrics.get("total_generated_tokens"),
            "gpu_cv": estimated.get("estimated_gpu_cv"),
            "gpu_cv_layer_mean": estimated.get("estimated_gpu_cv_layer_mean"),
            "cross_traffic_ratio": estimated.get("estimated_cross_traffic_ratio"),
            "cross_traffic_ratio_layer_mean": estimated.get("estimated_cross_traffic_ratio_layer_mean"),
            "placement_json_path": str(placement_path) if placement_path else None,
            "profile_path": str(profile_copy or profile_path) if (profile_copy or profile_path) else None,
            "result_json_path": parsed.get("result_json_path"),
            "pass": bool(parsed.get("pass")),
            "exit_code": parsed.get("exit_code"),
            "profile_source_definition": profile_payload.get("source_definition"),
            "profile_src_dim": len(profile_payload.get("traffic", [[[]]])[0]) if profile_payload.get("traffic") else None,
        }
        rows.append({**row, **quality})

    summary_rows = gather_summary_rows(rows)
    passing = [row for row in summary_rows if row.get("pass")]
    best = min(passing, key=lambda row: float(row.get("e2e_s") or 1e18)) if passing else None
    best_placement = str(best.get("placement")) if best else None
    findings_bullets = [
        f"Placement sweep impl: {args.impl}.",
        f"Static placements evaluated: {', '.join(placements)}.",
        "Placement protocol and owner_local_ep runtime semantics stayed unchanged; only script-layer sweep/reporting was added.",
        (
            "Best current static placement by end-to-end time: "
            + (best_placement or "undetermined")
            + "."
        ),
        (
            "Replica work should inherit the best passing placement from this summary once quality/runtime tradeoffs look stable."
        ),
    ]
    findings_md = render_findings_markdown(
        title="Owner Local EP Phase 2 Findings",
        bullets=findings_bullets,
        rows=summary_rows,
        columns=[
            "placement",
            "pass",
            "score",
            "e2e_s",
            "prefill_tok_s",
            "decode_tok_s",
            "cross_traffic_ratio",
            "gpu_cv",
            "profile_pairwise_l1_mean",
        ],
    )
    manifest = {
        "stage": STAGE_NAME,
        "run_dir": str(run_dir),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "dataset": dataset_info,
        "commands": commands,
        "baseline_impl_source": "phase1_summary" if args.impl != "ep_ht" else "fallback_default",
    }
    summary = {
        "stage": STAGE_NAME,
        "run_id": args.run_id,
        "runtime_mode": "owner_local_ep",
        "impl": args.impl,
        "recommended_next_placement": best_placement,
        "rows": summary_rows,
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "summary.csv", summary_rows)
    write_markdown(run_dir / "findings.md", findings_md)


if __name__ == "__main__":
    main()
