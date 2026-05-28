from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.utils_owner_local_ep import (
    add_common_owner_local_args,
    analyze_profile_if_present,
    build_test_bazaar_moe_cmd,
    check_output_order,
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


STAGE_NAME = "owner_local_ep_phase0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_owner_local_args(parser)
    parser.add_argument("--impls", default="ep_ll_torch,ep_ll_triton,ep_ht")
    parser.set_defaults(num_layers=1, num_problems=2, max_tokens=8)
    return finalize_owner_local_args(parser.parse_args())


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
    expected_ids = []
    if args.prepare_only or not args.dry_run:
        import datasets

        prepared_rows = datasets.load_dataset("parquet", data_files=str(prepared_path), split="train")
        expected_ids = [str(prepared_rows[idx]["id"]) for idx in range(min(args.num_problems, len(prepared_rows)))]

    for offset, impl in enumerate(impls):
        job_name = f"{impl}_contiguous"
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
        order = check_output_order(result_json_path, expected_ids) if expected_ids else {"output_order_ok": None, "output_order_note": "not_checked"}
        profile_payload = json.loads((profile_copy or profile_path).read_text()) if (profile_copy or profile_path) and (profile_copy or profile_path).exists() else {}
        profile_src_dim = len(profile_payload.get("traffic", [[[]]])[0]) if profile_payload.get("traffic") else None
        notes = []
        if profile_src_dim == args.world_size:
            notes.append("profile src_dim matches world_size")
        if profile_payload.get("source_definition") == "owner_rank":
            notes.append("profile source_definition is owner_rank, distinct from vllm_dp_ep dp_leader")
        if impl == "ep_ht":
            notes.append("owner-local HT routing semantics rely on local top-k / owner-local combine runtime path")
        row = {
            "impl": impl,
            "runtime_mode": "owner_local_ep",
            "world_size": args.world_size,
            "tp_size": args.tp_size,
            "dp_size": args.data_parallel_size,
            "ep_size": args.ep_size,
            "num_layers": args.num_layers,
            "num_problems": args.num_problems,
            "pass": bool(parsed.get("pass")) and bool(order.get("output_order_ok")) and profile_src_dim == args.world_size,
            "exit_code": parsed.get("exit_code"),
            "profile_path": str(profile_copy or profile_path) if (profile_copy or profile_path) else None,
            "result_json_path": parsed.get("result_json_path"),
            "rollout_txt_path": parsed.get("rollout_txt_path"),
            "output_order_ok": order.get("output_order_ok"),
            "profile_src_dim": profile_src_dim,
            "profile_source_definition": profile_payload.get("source_definition"),
            "notes": "; ".join(notes + ([order.get("output_order_note")] if order.get("output_order_note") else [])),
        }
        rows.append({**row, **quality})

    summary_rows = gather_summary_rows(rows)
    findings_bullets = [
        f"Bring-up impls requested: {', '.join(impls)}.",
        f"Owner-local geometry fixed to tp=1, dp=world={args.world_size}, ep=world={args.ep_size}.",
        (
            "Impls that passed end-to-end bring-up: "
            + (", ".join(row["impl"] for row in summary_rows if row.get("pass")) or "none")
            + "."
        ),
        (
            "Current source semantics: owner_local_ep profiles should report `owner_rank`; "
            "vllm_dp_ep remains `dp_leader`."
        ),
        (
            "Phase 1 is ready if at least one fast impl has pass=true and a real routing profile."
        ),
    ]
    findings_md = render_findings_markdown(
        title="Owner Local EP Phase 0 Findings",
        bullets=findings_bullets,
        rows=summary_rows,
        columns=[
            "impl",
            "pass",
            "exit_code",
            "output_order_ok",
            "profile_src_dim",
            "profile_source_definition",
            "profile_path",
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
        "rows": summary_rows,
        "can_enter_phase1": any(row.get("pass") for row in summary_rows),
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "summary.csv", summary_rows)
    write_markdown(run_dir / "findings.md", findings_md)


if __name__ == "__main__":
    main()
