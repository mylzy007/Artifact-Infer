from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from eval.utils_owner_local_ep import (
    add_common_owner_local_args,
    analyze_profile_if_present,
    finalize_owner_local_args,
    gather_summary_rows,
    make_run_dir,
    prepare_gsm_dataset_if_needed,
    render_findings_markdown,
    resolve_existing_path,
    write_csv,
    write_json,
    write_markdown,
)


STAGE_NAME = "owner_local_ep_phase3"
DEFAULT_REPLICA_CASES = ["R0", "R1", "R2", "R4", "R5"]
DEFAULT_ROUTING_CASES = ["S1", "S2", "S3", "S4"]
DEFAULT_JOINT_CASES = ["J0", "J1", "J2", "J3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_owner_local_args(parser)
    parser.add_argument("--phase2-summary", type=Path, default=None)
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--impl", default=None)
    parser.add_argument("--base-placement", default=None)
    parser.add_argument("--base-placements", default=None)
    parser.add_argument("--replica-policies", default=",".join(DEFAULT_REPLICA_CASES))
    parser.add_argument("--replica-routing-policies", default=",".join(DEFAULT_ROUTING_CASES))
    parser.add_argument("--hot-fraction-list", default="0.05,0.10,0.20")
    parser.add_argument("--replica-factor-list", default="2,4")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--joint-policies", default=",".join(DEFAULT_JOINT_CASES))
    parser.set_defaults(num_layers=-1, num_problems=256, max_tokens=64, max_num_seqs=8, enforce_eager=1)
    return finalize_owner_local_args(parser.parse_args())


def stage_dirs(run_dir: Path) -> dict[str, Path]:
    dirs = {
        "run_dir": run_dir,
        "logs": run_dir / "logs",
        "profiles": run_dir / "profiles",
        "profile_quality": run_dir / "profile_quality",
        "replica_plans": run_dir / "replica_plans",
        "prepared": run_dir / "prepared",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def parse_csv_str(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def latest_phase2_summary(output_root: Path) -> Path | None:
    candidates = sorted(output_root.glob("owner_local_ep_phase2_*/summary.json"), reverse=True)
    return candidates[0] if candidates else None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def best_phase2_row(summary: dict[str, Any]) -> dict[str, Any] | None:
    passing = [row for row in summary.get("rows", []) if row.get("pass")]
    if not passing:
        return None
    return min(
        passing,
        key=lambda row: (
            float(row.get("e2e_s") or 1e18),
            -float(row.get("score") or 0.0),
        ),
    )


def select_base_rows(summary: dict[str, Any], requested: list[str] | None) -> list[dict[str, Any]]:
    rows = [row for row in summary.get("rows", []) if row.get("pass")]
    if not rows:
        return []
    if requested:
        out = []
        for placement in requested:
            match = next((row for row in rows if row.get("placement") == placement), None)
            if match is not None:
                out.append(match)
        return out
    preferred = [
        "round_robin",
        "load_balanced_greedy_with_locality_tiebreak",
    ]
    out = []
    for placement in preferred:
        match = next((row for row in rows if row.get("placement") == placement), None)
        if match is not None:
            out.append(match)
    if len(out) >= 2:
        return out[:2]
    sorted_rows = sorted(rows, key=lambda row: (float(row.get("e2e_s") or 1e18), -float(row.get("score") or 0.0)))
    for row in sorted_rows:
        if row not in out:
            out.append(row)
        if len(out) >= 2:
            break
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


def build_replica_plan_cmd(
    args: argparse.Namespace,
    *,
    profile_path: Path,
    placement_json_path: Path | None,
    base_placement: str,
    replica_policy: str,
    routing_policy: str,
    replica_placement_policy: str,
    hot_fraction: float,
    replica_factor: int,
    output_dir: Path,
    case_id: str,
) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "eval.generate_moe_replica_plan",
        "--profile",
        str(profile_path),
        "--base-placement",
        base_placement,
        "--replica-policy",
        replica_policy,
        "--routing-policy",
        routing_policy,
        "--replica-placement-policy",
        replica_placement_policy,
        "--hot-fraction",
        str(hot_fraction),
        "--replica-factor",
        str(replica_factor),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
        "--case-id",
        case_id,
    ]
    if placement_json_path is not None:
        cmd += ["--placement-json", str(placement_json_path)]
    return cmd


def replica_case_configs() -> dict[str, dict[str, Any]]:
    return {
        "R0": {"replica_policy": "none", "routing_policy": "fixed_primary", "hot_fraction": 0.0, "replica_factor": 1, "replica_placement_policy": "naive_primary_plus_offsets"},
        "R1": {"replica_policy": "topk_hot", "routing_policy": "local_first_then_primary", "hot_fraction": 0.05, "replica_factor": 2, "replica_placement_policy": "naive_primary_plus_offsets"},
        "R2": {"replica_policy": "topk_hot", "routing_policy": "local_first_then_primary", "hot_fraction": 0.10, "replica_factor": 2, "replica_placement_policy": "naive_primary_plus_offsets"},
        "R3": {"replica_policy": "topk_hot", "routing_policy": "local_first_then_primary", "hot_fraction": 0.20, "replica_factor": 2, "replica_placement_policy": "naive_primary_plus_offsets"},
        "R4": {"replica_policy": "cross_numa_hot", "routing_policy": "local_first_then_numa_first", "hot_fraction": 0.10, "replica_factor": 2, "replica_placement_policy": "numa_symmetric"},
        "R5": {"replica_policy": "topk_hot", "routing_policy": "local_first_then_primary", "hot_fraction": 0.05, "replica_factor": 4, "replica_placement_policy": "naive_primary_plus_offsets"},
        "R6": {"replica_policy": "topk_hot", "routing_policy": "local_first_then_primary", "hot_fraction": 0.10, "replica_factor": 4, "replica_placement_policy": "naive_primary_plus_offsets"},
    }


def routing_case_configs(anchor: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "S0": {**anchor, "routing_policy": "fixed_primary"},
        "S1": {**anchor, "routing_policy": "strict_local_first"},
        "S2": {**anchor, "routing_policy": "local_first_then_primary"},
        "S3": {**anchor, "routing_policy": "local_first_then_numa_first"},
        "S4": {**anchor, "routing_policy": "local_first_then_least_loaded"},
        "S5": {**anchor, "routing_policy": "score_margin_local_first"},
    }


def joint_case_configs(anchor: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "J0": {**anchor, "routing_policy": "local_first_then_primary", "replica_placement_policy": "naive_primary_plus_offsets"},
        "J1": {**anchor, "routing_policy": "local_first_then_numa_first", "replica_placement_policy": "numa_symmetric"},
        "J2": {**anchor, "routing_policy": "local_first_then_primary", "replica_placement_policy": "source_traffic_aware"},
        "J3": {**anchor, "routing_policy": "local_first_then_least_loaded", "replica_placement_policy": "peak_load_aware"},
    }


def build_row(
    *,
    case_id: str,
    phase: str,
    args: argparse.Namespace,
    base_row: dict[str, Any],
    base_placement: str,
    plan_payload: dict[str, Any] | None,
    plan_path: Path | None,
    profile_quality: dict[str, Any],
    pass_value: bool,
    exit_code: int | None,
) -> dict[str, Any]:
    estimated = dict((plan_payload or {}).get("estimated_metrics", {}) or {})
    replicated_ids = list((plan_payload or {}).get("replicated_expert_ids", []) or [])
    return {
        "case_id": case_id,
        "placement_case_id": f"{base_placement}:{case_id}",
        "case_group": phase,
        "impl": args.impl,
        "runtime_mode": "owner_local_ep",
        "world_size": args.world_size,
        "tp_size": args.tp_size,
        "dp_size": args.data_parallel_size,
        "ep_size": args.ep_size,
        "base_placement": base_placement,
        "replica_policy": (plan_payload or {}).get("policy"),
        "replica_routing_policy": (plan_payload or {}).get("routing_policy"),
        "hot_fraction": (plan_payload or {}).get("hot_fraction"),
        "replica_factor": (plan_payload or {}).get("replica_factor"),
        "replica_budget_experts": (plan_payload or {}).get("replica_budget_experts"),
        "replicated_expert_ids": ",".join(str(item) for item in replicated_ids[:32]) + ("..." if len(replicated_ids) > 32 else ""),
        "replica_plan_json_path": str(plan_path) if plan_path else None,
        "score": base_row.get("score") if (plan_payload or {}).get("policy") == "none" else None,
        "e2e_s": base_row.get("e2e_s") if (plan_payload or {}).get("policy") == "none" else None,
        "prefill_tok_s": base_row.get("prefill_tok_s") if (plan_payload or {}).get("policy") == "none" else None,
        "decode_tok_s": base_row.get("decode_tok_s") if (plan_payload or {}).get("policy") == "none" else None,
        "total_generated_tokens": base_row.get("total_generated_tokens") if (plan_payload or {}).get("policy") == "none" else None,
        "gpu_cv": estimated.get("estimated_gpu_cv"),
        "gpu_cv_layer_mean": estimated.get("estimated_gpu_cv_layer_mean"),
        "cross_traffic_ratio": estimated.get("estimated_cross_traffic_ratio"),
        "cross_traffic_ratio_layer_mean": estimated.get("estimated_cross_traffic_ratio_layer_mean"),
        "profile_identical_fraction_mean": profile_quality.get("profile_identical_fraction_mean"),
        "profile_pairwise_l1_mean": profile_quality.get("profile_pairwise_l1_mean"),
        "profile_pairwise_cosine_mean": profile_quality.get("profile_pairwise_cosine_mean"),
        "pass": pass_value,
        "exit_code": exit_code,
        "local_hit_ratio": estimated.get("local_hit_ratio"),
        "replica_hit_ratio": estimated.get("replica_hit_ratio"),
        "cross_numa_reduction": estimated.get("cross_numa_reduction"),
        "estimated_replica_memory_cost": estimated.get("estimated_replica_memory_cost"),
        "runtime_support_status": "reused_phase2_runtime" if (plan_payload or {}).get("policy") == "none" else "offline_replica_proxy_only",
        "notes": "Phase 3 currently reuses real Phase 2 runtime only for R0; replica cases are offline owner-selection estimates until runtime dispatch supports replica plans.",
    }


def best_offline_replica_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row for row in rows
        if row.get("pass") and row.get("replica_policy") not in {None, "none"}
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            float(row.get("cross_traffic_ratio") or 1e18),
            float(row.get("gpu_cv") or 1e18),
            float(-(row.get("local_hit_ratio") or 0.0)),
        ),
    )


def choose_anchor_replica_set(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = best_offline_replica_row(rows)
    if best is None:
        return {
            "replica_policy": "topk_hot",
            "hot_fraction": 0.10,
            "replica_factor": 2,
            "replica_placement_policy": "naive_primary_plus_offsets",
        }
    return {
        "replica_policy": best["replica_policy"],
        "hot_fraction": float(best["hot_fraction"]),
        "replica_factor": int(best["replica_factor"]),
        "replica_placement_policy": "naive_primary_plus_offsets",
    }


def main() -> None:
    args = parse_args()
    phase2_summary_path = args.phase2_summary or latest_phase2_summary(args.output_root)
    if phase2_summary_path is None or not phase2_summary_path.exists():
        raise SystemExit("need --phase2-summary or an existing owner_local_ep_phase2 summary.json")
    phase2_summary = load_json(phase2_summary_path)
    requested_base_placements = parse_csv_str(args.base_placements) if args.base_placements else ([args.base_placement] if args.base_placement else None)
    base_rows = select_base_rows(phase2_summary, requested_base_placements)
    if not base_rows:
        raise SystemExit(f"no passing Phase 2 row found in {phase2_summary_path}")

    lead_base_row = base_rows[0]
    args.impl = args.impl or str(phase2_summary.get("impl") or lead_base_row.get("impl") or "ep_ht")
    args.world_size = int(lead_base_row.get("world_size") or args.world_size)
    args.tp_size = int(lead_base_row.get("tp_size") or args.tp_size)
    args.data_parallel_size = int(lead_base_row.get("dp_size") or args.data_parallel_size)
    args.ep_size = int(lead_base_row.get("ep_size") or args.ep_size)

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
    selected_base_placements: list[str] = []

    for base_row in base_rows:
        base_placement = str(base_row.get("placement"))
        selected_base_placements.append(base_placement)
        profile_path = args.profile or resolve_existing_path(base_row.get("profile_path"), search_roots=[phase2_summary_path.parent, args.output_root])
        if profile_path is None:
            raise SystemExit(f"could not resolve base profile path for placement {base_placement}")
        placement_json_path = resolve_existing_path(base_row.get("placement_json_path"), search_roots=[phase2_summary_path.parent, args.output_root])
        base_profile_copy = profile_path
        if profile_path.parent != dirs["profiles"]:
            target = dirs["profiles"] / f"{base_placement}_{profile_path.name}"
            target.write_text(profile_path.read_text())
            base_profile_copy = target
        profile_quality = analyze_profile_if_present(base_profile_copy, output_dir=dirs["profile_quality"])

        placement_rows: list[dict[str, Any]] = []
        replica_cases = replica_case_configs()
        for case_id in parse_csv_str(args.replica_policies):
            cfg = replica_cases.get(case_id)
            if cfg is None:
                raise SystemExit(f"unknown replica case id {case_id}")
            full_case_id = f"{base_placement}__{case_id}"
            cmd = build_replica_plan_cmd(
                args,
                profile_path=base_profile_copy,
                placement_json_path=placement_json_path,
                base_placement=base_placement,
                replica_policy=cfg["replica_policy"],
                routing_policy=cfg["routing_policy"],
                replica_placement_policy=cfg["replica_placement_policy"],
                hot_fraction=float(cfg["hot_fraction"]),
                replica_factor=int(cfg["replica_factor"]),
                output_dir=dirs["replica_plans"],
                case_id=full_case_id,
            )
            commands.append({"name": full_case_id, "command": " ".join(cmd), "group": "replica_set_sweep", "base_placement": base_placement})
            result = None if args.prepare_only else run_generator(cmd, log_path=dirs["logs"] / f"{full_case_id}.log", dry_run=args.dry_run)
            plan_path = Path(result["replica_plan_path"]) if result and result.get("replica_plan_path") else None
            plan_payload = load_json(plan_path) if plan_path and plan_path.exists() else None
            row = build_row(
                case_id=case_id,
                phase="replica_set_sweep",
                args=args,
                base_row=base_row,
                base_placement=base_placement,
                plan_payload=plan_payload,
                plan_path=plan_path,
                profile_quality=profile_quality,
                pass_value=True if args.prepare_only else bool(result and result.get("pass")),
                exit_code=None if args.prepare_only else (result or {}).get("exit_code"),
            )
            rows.append(row)
            placement_rows.append(row)

        anchor = choose_anchor_replica_set(placement_rows)
        routing_cases = routing_case_configs(anchor)
        for case_id in parse_csv_str(args.replica_routing_policies):
            cfg = routing_cases.get(case_id)
            if cfg is None:
                raise SystemExit(f"unknown routing case id {case_id}")
            full_case_id = f"{base_placement}__{case_id}"
            cmd = build_replica_plan_cmd(
                args,
                profile_path=base_profile_copy,
                placement_json_path=placement_json_path,
                base_placement=base_placement,
                replica_policy=cfg["replica_policy"],
                routing_policy=cfg["routing_policy"],
                replica_placement_policy=cfg["replica_placement_policy"],
                hot_fraction=float(cfg["hot_fraction"]),
                replica_factor=int(cfg["replica_factor"]),
                output_dir=dirs["replica_plans"],
                case_id=full_case_id,
            )
            commands.append({"name": full_case_id, "command": " ".join(cmd), "group": "routing_sweep", "base_placement": base_placement})
            result = None if args.prepare_only else run_generator(cmd, log_path=dirs["logs"] / f"{full_case_id}.log", dry_run=args.dry_run)
            plan_path = Path(result["replica_plan_path"]) if result and result.get("replica_plan_path") else None
            plan_payload = load_json(plan_path) if plan_path and plan_path.exists() else None
            rows.append(
                build_row(
                    case_id=case_id,
                    phase="routing_sweep",
                    args=args,
                    base_row=base_row,
                    base_placement=base_placement,
                    plan_payload=plan_payload,
                    plan_path=plan_path,
                    profile_quality=profile_quality,
                    pass_value=True if args.prepare_only else bool(result and result.get("pass")),
                    exit_code=None if args.prepare_only else (result or {}).get("exit_code"),
                )
            )

        joint_cases = joint_case_configs(anchor)
        for case_id in parse_csv_str(args.joint_policies):
            cfg = joint_cases.get(case_id)
            if cfg is None:
                raise SystemExit(f"unknown joint case id {case_id}")
            full_case_id = f"{base_placement}__{case_id}"
            cmd = build_replica_plan_cmd(
                args,
                profile_path=base_profile_copy,
                placement_json_path=placement_json_path,
                base_placement=base_placement,
                replica_policy=cfg["replica_policy"],
                routing_policy=cfg["routing_policy"],
                replica_placement_policy=cfg["replica_placement_policy"],
                hot_fraction=float(cfg["hot_fraction"]),
                replica_factor=int(cfg["replica_factor"]),
                output_dir=dirs["replica_plans"],
                case_id=full_case_id,
            )
            commands.append({"name": full_case_id, "command": " ".join(cmd), "group": "joint_placement_replica", "base_placement": base_placement})
            result = None if args.prepare_only else run_generator(cmd, log_path=dirs["logs"] / f"{full_case_id}.log", dry_run=args.dry_run)
            plan_path = Path(result["replica_plan_path"]) if result and result.get("replica_plan_path") else None
            plan_payload = load_json(plan_path) if plan_path and plan_path.exists() else None
            rows.append(
                build_row(
                    case_id=case_id,
                    phase="joint_placement_replica",
                    args=args,
                    base_row=base_row,
                    base_placement=base_placement,
                    plan_payload=plan_payload,
                    plan_path=plan_path,
                    profile_quality=profile_quality,
                    pass_value=True if args.prepare_only else bool(result and result.get("pass")),
                    exit_code=None if args.prepare_only else (result or {}).get("exit_code"),
                )
            )

    summary_rows = gather_summary_rows(rows)
    best_replica = best_offline_replica_row(summary_rows)
    findings_bullets = [
        f"Phase 3 base impl: {args.impl}; base placements: {', '.join(selected_base_placements)}.",
        "Runtime integration status: R0 reuses real Phase 2 runtime metrics; replica rows are offline owner-selection estimates pending dispatch/runtime support.",
        (
            "Static placement plus replica appears most promising on offline proxy for: "
            + (
                f"{best_replica['base_placement']} / {best_replica['case_id']} ({best_replica['replica_policy']}, routing={best_replica['replica_routing_policy']})"
                if best_replica
                else "no replica case"
            )
            + "."
        ),
        "This version explicitly compares two strong Phase 2 placements before making replica conclusions.",
        "Drop or dynamic transfer should wait until at least one replica routing policy is implemented in runtime and validated against these offline plans.",
    ]
    findings_md = render_findings_markdown(
        title="Owner Local EP Phase 3 Findings",
        bullets=findings_bullets,
        rows=summary_rows,
        columns=[
            "placement_case_id",
            "case_group",
            "replica_policy",
            "replica_routing_policy",
            "hot_fraction",
            "replica_factor",
            "cross_traffic_ratio",
            "gpu_cv",
            "local_hit_ratio",
            "cross_numa_reduction",
        ],
    )
    manifest = {
        "stage": STAGE_NAME,
        "run_dir": str(run_dir),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "phase2_summary_path": str(phase2_summary_path),
        "base_rows": base_rows,
        "dataset": dataset_info,
        "commands": commands,
        "runtime_todo": [
            "teach dispatch to consult replica_plan_json_path",
            "teach replica routing policies to choose expert owner before EP all_to_all",
            "add runtime counters for local_hit_ratio / replica_hit_ratio / cross_numa_reduction",
        ],
    }
    summary = {
        "stage": STAGE_NAME,
        "run_id": args.run_id,
        "runtime_mode": "owner_local_ep",
        "impl": args.impl,
        "base_placements": selected_base_placements,
        "best_offline_replica_case": best_replica["placement_case_id"] if best_replica else None,
        "best_offline_replica_plan_json_path": best_replica["replica_plan_json_path"] if best_replica else None,
        "rows": summary_rows,
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "summary.csv", summary_rows)
    write_markdown(run_dir / "findings.md", findings_md)


if __name__ == "__main__":
    main()
