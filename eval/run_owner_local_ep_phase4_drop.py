"""Phase 4 — token-replica drop sweep on owner_local_ep + EP-HT.

Reuses the better Phase 3 overlap baselines (overlap=0.25, local-first dispatch)
and varies drop_policy x drop_rate. Reuses the overlap plan JSONs produced by
Phase 3 instead of regenerating them.

Defaults run on the answer-first GSM prompt with score_mode=answer_first so the
score reflects whether the model's answer survives partial-topk combine.
"""
from __future__ import annotations

import argparse
import datetime
import json
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


STAGE_NAME = "owner_local_ep_phase4_drop"

# Each baseline triple = (base_placement, replica_placement_policy, dispatch_policy)
# overlap is fixed at 0.25 (the only working ov on 24GB cards).
DEFAULT_BASELINES = [
    "round_robin/numa_local_first/min_communication",
    "load_balanced_greedy_with_locality_tiebreak/numa_local_first/greedy_balance",
]

DEFAULT_DROP_POLICIES = [
    "tail_weight",
    "per_expert_uniform",
    "hot_expert_relief",
    "per_expert_tailtoken",
    "cross_numa_first",
    "random",
    "hotspot_relief",
]

DEFAULT_DROP_RATES = "0,0.05,0.10,0.20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_owner_local_args(parser)
    parser.add_argument("--phase3-summary", type=Path, default=None,
                        help="Phase 3 summary.json; defaults to latest under output_root")
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES),
                        help="comma-separated base/replica/dispatch triples (overlap fixed at 0.25)")
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--drop-policies", default=",".join(DEFAULT_DROP_POLICIES))
    parser.add_argument("--drop-rates", default=DEFAULT_DROP_RATES)
    parser.add_argument("--drop-seed", type=int, default=0)
    parser.add_argument("--num-repeats", type=int, default=1,
                        help="repeat each case N times; per-rep rows are stored individually and mean/stdev aggregated in summary_agg.json")
    parser.add_argument("--warmup-cases", type=int, default=0,
                        help="run this many throwaway cases at the start to warm fs cache / NCCL state before the real sweep")
    parser.add_argument("--prompt-style", default="reasoning_brief",
                        choices=["answer_only", "answer_first", "reasoning_brief"])
    parser.add_argument("--score-mode", default="answer_first",
                        choices=["default", "answer_first"])
    parser.add_argument("--impl", default="ep_ht")
    parser.add_argument("--drop-impl", default="auto",
                        choices=["auto", "gpu", "cpu"],
                        help="drop implementation: auto picks GPU for simple policies, CPU for grouped")
    parser.add_argument("--drop-min-replicas", type=int, default=0,
                        help="bypass drop when T*K <= this value (small batch bypass)")
    parser.add_argument("--profile-drop-runtime-stats", type=int, default=1,
                        choices=[0, 1],
                        help="0 = pure performance sweep (no host sync from stats), 1 = accounting sweep")
    parser.add_argument("--router-keff", type=int, default=0,
                        help="Phase 4 P1: if >0 and < model top_k, dispatch only top-K_eff branches per token")
    parser.set_defaults(
        num_layers=-1,
        num_problems=256,
        max_tokens=96,
        max_num_seqs=8,
        enforce_eager=1,
    )
    return finalize_owner_local_args(parser.parse_args())


def stage_dirs(run_dir: Path) -> dict[str, Path]:
    dirs = {
        "run_dir": run_dir,
        "logs": run_dir / "logs",
        "profiles": run_dir / "profiles",
        "profile_quality": run_dir / "profile_quality",
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


def latest_phase3_summary(output_root: Path) -> Path | None:
    candidates = sorted(output_root.glob("owner_local_ep_phase3_overlap_*/summary.json"), reverse=True)
    return candidates[0] if candidates else None


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def find_phase3_row(
    summary: dict[str, Any], *, base: str, replica: str, dispatch: str, overlap: float,
) -> dict[str, Any] | None:
    for row in summary.get("rows", []):
        if (
            str(row.get("base_placement")) == base
            and str(row.get("replica_placement_policy")) == replica
            and str(row.get("routing_strategy")) == dispatch
            and float(row.get("overlap") or 0.0) == float(overlap)
        ):
            return row
    return None


def build_row(
    *,
    args: argparse.Namespace,
    case_id: str,
    baseline_key: str,
    drop_policy: str,
    drop_rate: float,
    runtime_parsed: dict[str, Any] | None,
    overlap_stats_payload: dict[str, Any] | None,
    plan_payload: dict[str, Any],
) -> dict[str, Any]:
    metrics = ((runtime_parsed or {}).get("metrics") or {})
    return {
        "case_id": case_id,
        "baseline_key": baseline_key,
        "impl": args.impl,
        "runtime_mode": "owner_local_ep",
        "base_placement": plan_payload.get("base_placement"),
        "replica_placement_policy": plan_payload.get("replica_placement_policy"),
        "routing_strategy": plan_payload.get("routing_strategy"),
        "overlap": float(plan_payload.get("overlap", 0.0)),
        "drop_policy": drop_policy,
        "drop_rate": drop_rate,
        "score": metrics.get("average_score"),
        "e2e_s": metrics.get("e2e_total_time_s"),
        "prefill_tok_s": metrics.get("prefill_throughput_tok_s"),
        "decode_tok_s": metrics.get("decode_throughput_tok_s"),
        "total_generated_tokens": metrics.get("total_generated_tokens"),
        "average_generated_tokens": metrics.get("average_generated_tokens"),
        "cross_traffic_ratio": (overlap_stats_payload or {}).get("cross_traffic_ratio"),
        "local_hit_ratio": (overlap_stats_payload or {}).get("local_hit_ratio"),
        "cross_numa_ratio": (overlap_stats_payload or {}).get("cross_numa_ratio"),
        "gpu_cv": (overlap_stats_payload or {}).get("gpu_cv"),
        "drop_replicas_total": (overlap_stats_payload or {}).get("drop_replicas_total"),
        "drop_fraction_total": (overlap_stats_payload or {}).get("drop_fraction_total"),
        "drop_fraction_remote": (overlap_stats_payload or {}).get("drop_fraction_remote"),
        "drop_fraction_cross_numa": (overlap_stats_payload or {}).get("drop_fraction_cross_numa"),
        "dropped_weight_mass_fraction": (overlap_stats_payload or {}).get("dropped_weight_mass_fraction"),
        "per_layer_drop_fraction_mean": (overlap_stats_payload or {}).get("per_layer_drop_fraction_mean"),
        "effective_dropped_expert_load_cv": (overlap_stats_payload or {}).get("effective_dropped_expert_load_cv"),
        "overlap_runtime_stats_path": metrics.get("overlap_runtime_stats_path"),
        "result_json_path": (runtime_parsed or {}).get("result_json_path"),
        "log_path": (runtime_parsed or {}).get("log_path"),
        "pass": bool((runtime_parsed or {}).get("pass")),
        "exit_code": (runtime_parsed or {}).get("exit_code"),
    }


def aggregate_repeats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group rows by case_key (baseline + policy + rate) and aggregate metrics
    across reps. Returns one row per case with mean/stdev/n_reps populated."""
    import statistics as _st
    from collections import defaultdict
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        ck = r.get("case_key") or r.get("case_id")
        grouped[ck].append(r)
    out: list[dict[str, Any]] = []
    numeric_keys = [
        "score", "e2e_s", "prefill_tok_s", "decode_tok_s",
        "drop_fraction_total", "drop_fraction_remote", "drop_fraction_cross_numa",
        "dropped_weight_mass_fraction", "cross_traffic_ratio", "cross_numa_ratio",
        "gpu_cv", "total_generated_tokens", "per_layer_drop_fraction_mean",
    ]
    for case_key, reps in grouped.items():
        passing = [r for r in reps if r.get("pass")]
        lead = passing[0] if passing else reps[0]
        agg_row = {
            "case_key": case_key,
            "baseline_key": lead.get("baseline_key"),
            "drop_policy": lead.get("drop_policy"),
            "drop_rate": lead.get("drop_rate"),
            "n_reps": len(reps),
            "n_pass": len(passing),
            "pass": len(passing) > 0,
        }
        for k in numeric_keys:
            vals = [float(r.get(k)) for r in passing if r.get(k) is not None]
            if vals:
                agg_row[f"{k}_mean"] = sum(vals) / len(vals)
                agg_row[f"{k}_std"] = _st.stdev(vals) if len(vals) > 1 else 0.0
                agg_row[f"{k}_min"] = min(vals)
                agg_row[f"{k}_max"] = max(vals)
            else:
                agg_row[f"{k}_mean"] = None
                agg_row[f"{k}_std"] = None
                agg_row[f"{k}_min"] = None
                agg_row[f"{k}_max"] = None
        # Convenience aliases for downstream readers / sort keys.
        agg_row["score"] = agg_row.get("score_mean")
        agg_row["e2e_s"] = agg_row.get("e2e_s_mean")
        out.append(agg_row)
    return out


def best_runtime_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [r for r in rows if r.get("pass")]
    if not passing:
        return None
    # Aggregated rows expose `score_mean` / `e2e_s_mean`. For per-rep rows we
    # fall back to score / e2e_s.
    def keyf(r):
        s = r.get("score_mean") if "score_mean" in r else r.get("score")
        e = r.get("e2e_s_mean") if "e2e_s_mean" in r else r.get("e2e_s")
        return (-float(s or 0.0), float(e or 1e18))
    return min(passing, key=keyf)


def main() -> None:
    args = parse_args()
    phase3_path = args.phase3_summary or latest_phase3_summary(args.output_root)
    if phase3_path is None or not phase3_path.exists():
        raise SystemExit("need --phase3-summary or an existing owner_local_ep_phase3_overlap summary.json")
    phase3 = load_json(phase3_path)

    baselines: list[tuple[str, str, str, str]] = []
    for spec in parse_csv_str(args.baselines):
        try:
            base, replica, dispatch = spec.split("/")
        except ValueError:
            raise SystemExit(f"--baselines triple must be base/replica/dispatch, got {spec!r}")
        row = find_phase3_row(phase3, base=base, replica=replica, dispatch=dispatch, overlap=args.overlap)
        if row is None:
            raise SystemExit(
                f"Phase 3 summary missing row for base={base} replica={replica} "
                f"dispatch={dispatch} overlap={args.overlap}"
            )
        baseline_key = f"{base}__{replica}__{dispatch}"
        baselines.append((baseline_key, base, replica, dispatch))

    # Use the first baseline to fix world/impl knobs (they're identical across baselines).
    lead = find_phase3_row(
        phase3,
        base=baselines[0][1], replica=baselines[0][2], dispatch=baselines[0][3], overlap=args.overlap,
    )
    args.impl = args.impl or str(phase3.get("impl") or lead.get("impl") or "ep_ht")
    # Phase 3 rows don't always carry world/tp/dp — pull from the overlap plan
    # JSON, which records world_size authoritatively.
    lead_plan_path = resolve_existing_path(
        lead.get("overlap_plan_json_path"),
        search_roots=[phase3_path.parent, args.output_root],
    )
    lead_plan = load_json(lead_plan_path) if lead_plan_path else {}
    plan_world_size = int(lead_plan.get("world_size") or 0)
    if plan_world_size > 0:
        args.world_size = plan_world_size
        args.data_parallel_size = plan_world_size
        args.tp_size = 1
        args.ep_size = plan_world_size

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

    drop_policies = parse_csv_str(args.drop_policies)
    drop_rates = parse_csv_float(args.drop_rates)

    commands: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    gpu_group = selected_gpu_group(args)
    port_offset = 0
    num_repeats = max(1, int(args.num_repeats))
    warmup_cases = max(0, int(args.warmup_cases))

    # ---------- warmup (throwaway) ----------
    if warmup_cases > 0 and not args.prepare_only:
        warm_dir = dirs["results"] / "_warmup"
        warm_baseline_key, warm_base, warm_replica, warm_dispatch = baselines[0]
        warm_phase3_row = find_phase3_row(
            phase3, base=warm_base, replica=warm_replica, dispatch=warm_dispatch, overlap=args.overlap,
        )
        warm_plan_path = resolve_existing_path(
            warm_phase3_row.get("overlap_plan_json_path"),
            search_roots=[phase3_path.parent, args.output_root],
        )
        warm_plan_payload = load_json(warm_plan_path)
        warm_base_placement_path = resolve_existing_path(
            warm_plan_payload.get("base_placement_json_path"),
            search_roots=[phase3_path.parent, args.output_root],
        )
        for w in range(warmup_cases):
            warm_out = warm_dir / f"warm{w}"
            warm_cmd = build_test_bazaar_moe_cmd(
                args,
                impl=args.impl,
                output_dir=warm_out,
                master_port=args.master_port + port_offset,
                profile_enabled=False,
                overlap_runtime_enabled=False,
                placement=warm_base,
                placement_path=warm_base_placement_path,
                overlap_path=warm_plan_path,
                overlap_value=args.overlap,
                overlap_strategy=warm_dispatch,
            ) + [
                "--moe-drop-policy", "none",
                "--moe-drop-rate", "0",
                "--moe-drop-seed", str(args.drop_seed),
                "--prompt-style", args.prompt_style,
                "--score-mode", args.score_mode,
            ]
            port_offset += 1
            run_subprocess_job(
                cmd=warm_cmd,
                env=command_env(
                    args,
                    cuda_visible_devices=gpu_group,
                    run_id=f"{args.run_id}_warm{w}",
                    extra_env={"FLASHINFER_WORKSPACE_BASE": str(run_dir / "flashinfer_workspace")},
                ),
                log_path=dirs["logs"] / f"warmup_{w}.log",
                timeout_s=args.timeout_s,
                dry_run=args.dry_run,
            )

    for baseline_key, base_placement, replica_policy, dispatch_policy in baselines:
        phase3_row = find_phase3_row(
            phase3, base=base_placement, replica=replica_policy, dispatch=dispatch_policy, overlap=args.overlap,
        )
        plan_path = resolve_existing_path(
            phase3_row.get("overlap_plan_json_path"),
            search_roots=[phase3_path.parent, args.output_root],
        )
        if plan_path is None:
            raise SystemExit(f"can't resolve overlap plan for {baseline_key}")
        plan_payload = load_json(plan_path)
        base_placement_path = resolve_existing_path(
            plan_payload.get("base_placement_json_path"),
            search_roots=[phase3_path.parent, args.output_root],
        )

        # rate=0 only needs to be run once per baseline (every policy gives the same result).
        rate0_done = False
        for drop_policy in drop_policies:
            for drop_rate in drop_rates:
                effective_policy = drop_policy if drop_rate > 0 else "none"
                if drop_rate == 0.0:
                    if rate0_done:
                        continue
                    rate0_done = True
                    effective_policy = "none"

                rate_str = ("%.3f" % drop_rate).rstrip("0").rstrip(".")
                if rate_str == "" or rate_str == "-":
                    rate_str = "0"
                case_key = f"{baseline_key}__{effective_policy}__r{rate_str}"

                for rep in range(num_repeats):
                    case_id = f"{case_key}__rep{rep}" if num_repeats > 1 else case_key

                    runtime_output_dir = dirs["results"] / case_id
                    overlap_stats_on = bool(args.profile_drop_runtime_stats)
                    runtime_cmd = build_test_bazaar_moe_cmd(
                        args,
                        impl=args.impl,
                        output_dir=runtime_output_dir,
                        master_port=args.master_port + port_offset,
                        profile_enabled=False,
                        overlap_runtime_enabled=overlap_stats_on,
                        placement=base_placement,
                        placement_path=base_placement_path,
                        overlap_path=plan_path,
                        overlap_value=args.overlap,
                        overlap_strategy=dispatch_policy,
                    )
                    runtime_cmd += [
                        "--moe-drop-policy", effective_policy,
                        "--moe-drop-rate", str(drop_rate),
                        "--moe-drop-seed", str(args.drop_seed + rep),
                        "--moe-router-keff", str(args.router_keff),
                        "--prompt-style", args.prompt_style,
                        "--score-mode", args.score_mode,
                    ]
                    port_offset += 1
                    commands.append({"name": f"runtime_{case_id}", "command": " ".join(str(p) for p in runtime_cmd)})

                    runtime_parsed: dict[str, Any] | None = None
                    overlap_stats_payload: dict[str, Any] | None = None
                    if not args.prepare_only:
                        extra_env = {
                            "FLASHINFER_WORKSPACE_BASE": str(run_dir / "flashinfer_workspace"),
                            "MOE_DROP_IMPL": args.drop_impl,
                            "MOE_DROP_MIN_REPLICAS": str(args.drop_min_replicas),
                            "MOE_DROP_GPU_STATS": "1" if overlap_stats_on else "0",
                        }
                        runtime_job = run_subprocess_job(
                            cmd=runtime_cmd,
                            env=command_env(
                                args,
                                cuda_visible_devices=gpu_group,
                                run_id=f"{args.run_id}_{case_id}",
                                extra_env=extra_env,
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
                        if copied_overlap_stats is not None:
                            runtime_parsed.setdefault("metrics", {})
                            runtime_parsed["metrics"]["overlap_runtime_stats_path"] = str(copied_overlap_stats)

                    row = build_row(
                        args=args,
                        case_id=case_id,
                        baseline_key=baseline_key,
                        drop_policy=effective_policy,
                        drop_rate=drop_rate,
                        runtime_parsed=runtime_parsed,
                        overlap_stats_payload=overlap_stats_payload,
                        plan_payload=plan_payload,
                    )
                    row["case_key"] = case_key
                    row["rep"] = rep
                    rows.append(row)

                    # Incremental dump after each rep so a crash mid-sweep still
                    # leaves usable partial output.
                    agg = aggregate_repeats(rows)
                    _summary = {
                        "stage": STAGE_NAME,
                        "run_id": args.run_id,
                        "runtime_mode": "owner_local_ep",
                        "impl": args.impl,
                        "num_repeats": num_repeats,
                        "best_runtime_case": best_runtime_row(agg),
                        "rows": rows,
                        "rows_aggregated": agg,
                    }
                    write_json(run_dir / "summary.json", _summary)
                    write_csv(run_dir / "summary.csv", rows)
                    write_csv(run_dir / "summary_agg.csv", agg)

    summary_rows = gather_summary_rows(rows)
    aggregated_rows = aggregate_repeats(rows)
    best = best_runtime_row(aggregated_rows)
    findings = [
        f"Phase 4 drop sweep on impl={args.impl}, overlap={args.overlap}, prompt={args.prompt_style}.",
        f"num_repeats={num_repeats}, warmup_cases={warmup_cases}. mean ± stdev across reps in summary_agg.csv.",
        (
            "Best (by mean score desc, mean e2e asc): "
            + (
                f"{best['baseline_key']} / {best['drop_policy']} @ rate={best['drop_rate']} "
                f"score_mean={best.get('score_mean')} e2e_mean={best.get('e2e_s_mean')}"
                if best
                else "none"
            )
        ),
    ]
    findings_md = render_findings_markdown(
        title="Owner Local EP Phase 4 Drop Findings",
        bullets=findings,
        rows=aggregated_rows,
        columns=[
            "baseline_key",
            "drop_policy",
            "drop_rate",
            "n_reps",
            "score_mean",
            "score_std",
            "e2e_s_mean",
            "e2e_s_std",
            "prefill_tok_s_mean",
            "decode_tok_s_mean",
            "drop_fraction_total_mean",
            "drop_fraction_remote_mean",
            "dropped_weight_mass_fraction_mean",
            "pass",
        ],
    )
    manifest = {
        "stage": STAGE_NAME,
        "run_dir": str(run_dir),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "phase3_summary_path": str(phase3_path),
        "dataset": dataset_info,
        "baselines": [b[0] for b in baselines],
        "drop_policies": drop_policies,
        "drop_rates": drop_rates,
        "commands": commands,
        "completed_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    final_summary = {
        "stage": STAGE_NAME,
        "run_id": args.run_id,
        "runtime_mode": "owner_local_ep",
        "impl": args.impl,
        "num_repeats": num_repeats,
        "best_runtime_case": best,
        "rows": summary_rows,
        "rows_aggregated": aggregated_rows,
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "summary.json", final_summary)
    write_csv(run_dir / "summary.csv", summary_rows)
    write_csv(run_dir / "summary_agg.csv", aggregated_rows)
    write_markdown(run_dir / "findings.md", findings_md)


if __name__ == "__main__":
    main()
