from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import datasets

from eval.analyze_routing_profile_quality import build_report, markdown_report


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/lzy/models/Qwen3-30B-A3B")
DEFAULT_DATASET = Path("/home/lzy/datasets/moe_benchmarks/gsm8k/main/test-00000-of-00001.parquet")
DEFAULT_OUTPUT_ROOT = REPO / "eval_results"
DEFAULT_PYTHON = "/home/lzy/miniconda3/envs/vllm/bin/python"
DEFAULT_GPU_IDS = "0,1,2,3,4,5,6,7"
DEFAULT_PLACEMENTS = [
    "contiguous",
    "round_robin",
    "fixed_random_shuffle",
    "load_balanced_greedy_with_locality_tiebreak",
    "communication_aware_greedy",
]


def sanitize_run_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value.strip("_") or "run"


def shlex_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def make_run_dir(output_root: Path, stage_name: str, run_id: str) -> Path:
    return output_root / f"{stage_name}_{sanitize_run_id(run_id)}"


def build_gsm_answer_only_prompt(question: str) -> str:
    return (
        "Solve the following math word problem.\n"
        "Return only the final answer as a plain number or short expression.\n"
        "Do not include reasoning, explanation, units, or boxed formatting.\n\n"
        f"Question:\n{question.strip()}"
    )


def normalize_final_answer(text: object) -> str:
    value = "" if text is None else str(text)
    value = value.strip().replace("$", "").replace(",", "")
    return value.strip()


def extract_final_answer(text: object) -> str:
    value = "" if text is None else str(text)
    match = re.search(r"####\s*(.+?)\s*$", value, flags=re.S)
    if match:
        return normalize_final_answer(match.group(1))
    return normalize_final_answer(value)


def load_dataset_rows(path: Path):
    if path.suffix == ".jsonl":
        return datasets.load_dataset("json", data_files=str(path), split="train")
    if path.suffix == ".json":
        return datasets.load_dataset("json", data_files=str(path), split="train")
    if path.suffix == ".parquet":
        return datasets.load_dataset("parquet", data_files=str(path), split="train")
    raise SystemExit(f"unsupported dataset format: {path}")


def prepare_gsm_dataset_if_needed(
    source: Path,
    prepared_path: Path,
    *,
    limit: int,
) -> dict[str, Any]:
    rows = load_dataset_rows(source)
    n = len(rows) if limit <= 0 else min(limit, len(rows))
    items: list[dict[str, Any]] = []
    preview: list[dict[str, str]] = []
    for idx in range(n):
        row = rows[idx]
        if "conversations" in row:
            conv = row["conversations"]
            question = str(conv[0]["value"])
            solution = str(conv[-1]["value"])
            row_id = f"gsm8k-{idx:05d}"
        else:
            question = str(row.get("question") or row.get("problem") or "")
            solution = str(row.get("solution") or row.get("answer") or "")
            row_id = str(row.get("id") or f"gsm8k-{idx:05d}")
        answer = extract_final_answer(row.get("answer") if "answer" in row else solution)
        if not answer or "####" in answer:
            answer = extract_final_answer(solution)
        prompt_text = build_gsm_answer_only_prompt(question)
        item = {
            "id": row_id,
            "question": question,
            "raw_question": question,
            "answer": answer,
            "solution": solution,
            "prompt_text": prompt_text,
            "prompt": [{"role": "user", "content": prompt_text}],
        }
        items.append(item)
        if idx < 5:
            preview.append({"id": row_id, "question": question, "prompt_text": prompt_text, "answer": answer})
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(items).to_parquet(str(prepared_path))
    preview_path = prepared_path.with_suffix(".preview.json")
    write_json(preview_path, {"source": str(source), "num_rows": len(items), "preview": preview})
    return {
        "source": str(source),
        "prepared_path": str(prepared_path),
        "preview_path": str(preview_path),
        "num_rows": len(items),
        "prompt_format": "answer_only_plain_number_or_short_expression",
    }


def add_common_owner_local_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-label", default="gsm8k")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=None)
    parser.add_argument("--master-port", type=int, default=29555)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-problems", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--gpu-ids", default=DEFAULT_GPU_IDS)
    parser.add_argument("--cuda-home", type=Path, default=Path("/usr/local/cuda-12.8") if Path("/usr/local/cuda-12.8/bin/nvcc").exists() else None)
    parser.add_argument("--enforce-eager", type=int, default=1)
    parser.add_argument("--moe-ll-overflow-policy", choices=["drop", "error"], default="error")
    parser.add_argument("--moe-ll-m-max", type=int, default=-1)


def finalize_owner_local_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.tp_size != 1:
        raise SystemExit("owner_local_ep requires --tp-size 1")
    if args.world_size <= 0:
        raise SystemExit("--world-size must be positive")
    if args.data_parallel_size is None:
        args.data_parallel_size = args.world_size
    if args.data_parallel_size != args.world_size:
        raise SystemExit(
            f"owner_local_ep requires --data-parallel-size == --world-size == {args.world_size}, "
            f"got {args.data_parallel_size}"
        )
    args.ep_size = args.world_size
    args.runtime_mode = "owner_local_ep"
    args.output_root = args.output_root.resolve()
    args.model_path = args.model_path.resolve()
    args.dataset_path = args.dataset_path.resolve()
    return args


def selected_gpu_group(args: argparse.Namespace) -> str:
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if len(gpu_ids) < args.world_size:
        raise SystemExit(
            f"need at least world_size={args.world_size} GPU ids, got {len(gpu_ids)} from --gpu-ids"
        )
    return ",".join(gpu_ids[: args.world_size])


def build_test_bazaar_moe_cmd(
    args: argparse.Namespace,
    *,
    impl: str,
    output_dir: Path,
    master_port: int,
    profile_enabled: bool,
    overlap_runtime_enabled: bool = False,
    placement: str = "contiguous",
    placement_path: Path | None = None,
    overlap_path: Path | None = None,
    overlap_value: float = 0.0,
    overlap_strategy: str = "hybrid",
) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "eval.test_bazaar_moe",
        "--model-path",
        str(args.model_path),
        "--dataset-path",
        str(args.prepared_dataset_path),
        "--dataset-label",
        args.dataset_label,
        "--world-size",
        str(args.world_size),
        "--tp-size",
        str(args.tp_size),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--master-port",
        str(master_port),
        "--moe-impl",
        impl,
        "--moe-runtime-mode",
        "owner_local_ep",
        "--moe-expert-placement",
        placement,
        "--moe-expert-placement-seed",
        str(getattr(args, "seed", 0)),
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
        str(args.enforce_eager),
        "--moe-ll-overflow-policy",
        args.moe_ll_overflow_policy,
        "--moe-profile-routing",
        "1" if profile_enabled else "0",
        "--moe-profile-overlap-runtime",
        "1" if overlap_runtime_enabled else "0",
        "--output-dir",
        str(output_dir),
        "--moe-expert-overlap-value",
        str(overlap_value),
        "--moe-expert-overlap-strategy",
        overlap_strategy,
    ]
    if args.moe_ll_m_max > 0:
        cmd += ["--moe-ll-m-max", str(args.moe_ll_m_max)]
    if placement_path is not None:
        cmd += ["--moe-expert-placement-path", str(placement_path)]
    if overlap_path is not None:
        cmd += ["--moe-expert-overlap-path", str(overlap_path)]
    return cmd


def build_generate_placement_cmd(
    args: argparse.Namespace,
    *,
    profile_path: Path,
    placement: str,
    output_dir: Path,
) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "eval.generate_moe_placement",
        "--profile",
        str(profile_path),
        "--policy",
        placement,
        "--output-dir",
        str(output_dir),
    ]
    if placement == "fixed_random_shuffle":
        cmd += ["--seed", str(getattr(args, "seed", 0))]
    return cmd


def command_env(args: argparse.Namespace, *, cuda_visible_devices: str, run_id: str, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["MOE_PROFILE_RUN_ID"] = run_id
    env.setdefault("PYTHONUNBUFFERED", "1")
    python_bin = str(Path(args.python).resolve().parent)
    path_parts = [python_bin]
    if getattr(args, "cuda_home", None):
        cuda_home = Path(args.cuda_home)
        env["CUDA_HOME"] = str(cuda_home)
        env["FLASHINFER_NVCC"] = str(cuda_home / "bin" / "nvcc")
        env["CUDACXX"] = str(cuda_home / "bin" / "nvcc")
        env["NVCC"] = str(cuda_home / "bin" / "nvcc")
        path_parts.append(str(cuda_home / "bin"))
        lib64 = str(cuda_home / "lib64")
        env["LD_LIBRARY_PATH"] = lib64 + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = os.pathsep.join(path_parts + [env.get("PATH", "")])
    if extra_env:
        env.update(extra_env)
    return env


def run_subprocess_job(
    *,
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    timeout_s: int,
    dry_run: bool,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "command": shlex_join(cmd),
        "log_path": str(log_path),
        "timeout_s": timeout_s,
        "dry_run": dry_run,
        "pass": False,
        "exit_code": None,
    }
    if dry_run:
        log_path.write_text("$ " + shlex_join(cmd) + "\n")
        summary["pass"] = True
        return summary

    t0 = time.time()
    with log_path.open("w") as log:
        log.write("$ " + shlex_join(cmd) + "\n")
        log.write(f"# CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
            )
            summary["exit_code"] = int(proc.returncode)
            summary["pass"] = proc.returncode == 0
        except subprocess.TimeoutExpired as exc:
            log.write(f"\n[TIMEOUT] exceeded {timeout_s}s\n{exc}\n")
            summary["exit_code"] = 124
            summary["timed_out"] = True
    summary["elapsed_s"] = time.time() - t0
    return summary


def resolve_existing_path(path_value: str | Path | None, *, search_roots: list[Path]) -> Path | None:
    if path_value is None:
        return None
    candidate = Path(path_value)
    direct_candidates = [candidate]
    if not candidate.is_absolute():
        direct_candidates.extend(root / candidate for root in search_roots)
    for item in direct_candidates:
        if item.exists():
            return item.resolve()
    name = candidate.name
    if not name:
        return None
    for root in search_roots:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0].resolve()
    return None


def load_result_json_metrics(result_json_path: Path | None) -> dict[str, Any]:
    if result_json_path is None or not result_json_path.exists():
        return {}
    payload = json.loads(result_json_path.read_text())
    summary = payload.get("summary", {}) or {}
    metrics = dict(summary.get("generation_metrics", {}) or {})
    metrics["average_score"] = summary.get("average_score")
    metrics["total_generated_tokens"] = summary.get("total_generated_tokens")
    metrics["num_problems"] = summary.get("num_problems")
    metrics["routing_profile_path"] = summary.get("routing_profile_path")
    metrics["overlap_runtime_stats_path"] = summary.get("overlap_runtime_stats_path")
    return metrics


def parse_test_bazaar_moe_outputs(
    job_result: dict[str, Any],
    *,
    output_dir: Path,
    search_roots: list[Path],
) -> dict[str, Any]:
    log_path = Path(job_result["log_path"])
    parsed: dict[str, Any] = dict(job_result)
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    profile_matches = re.findall(r"\[moe-profile\]\s+wrote\s+([^\s]+\.json)", text)
    result_matches = re.findall(r"results:\s+([^\s]+\.json)", text)
    rollout_matches = re.findall(r"rollouts:\s+([^\s]+\.txt)", text)

    profile_path = resolve_existing_path(profile_matches[-1], search_roots=search_roots) if profile_matches else None
    result_json_path = resolve_existing_path(result_matches[-1], search_roots=search_roots) if result_matches else None
    rollout_txt_path = resolve_existing_path(rollout_matches[-1], search_roots=search_roots) if rollout_matches else None

    if profile_path is None:
        profile_path = next(iter(sorted(output_dir.glob("moe_routing_profile_*.json"))), None)
    if result_json_path is None:
        result_json_path = next(iter(sorted(output_dir.glob("*.json"))), None)
        if result_json_path is not None and (
            result_json_path.name.startswith("moe_routing_profile_")
            or result_json_path.name.startswith("overlap_runtime_stats_")
        ):
            non_profile = [
                path for path in sorted(output_dir.glob("*.json"))
                if not path.name.startswith("moe_routing_profile_")
                and not path.name.startswith("overlap_runtime_stats_")
            ]
            result_json_path = non_profile[0] if non_profile else None
    if rollout_txt_path is None:
        rollout_txt_path = next(iter(sorted(output_dir.glob("*.txt"))), None)

    parsed["profile_path"] = str(profile_path) if profile_path else None
    parsed["result_json_path"] = str(result_json_path) if result_json_path else None
    parsed["rollout_txt_path"] = str(rollout_txt_path) if rollout_txt_path else None
    parsed["metrics"] = load_result_json_metrics(result_json_path)
    if not parsed.get("pass"):
        lines = text.splitlines()
        parsed["exception_summary"] = [
            line for line in lines
            if "Traceback" in line or "RuntimeError" in line or "NCCL" in line or "TIMEOUT" in line
        ][-12:]
    return parsed


def analyze_profile_if_present(profile_path: Path | None, *, output_dir: Path) -> dict[str, Any]:
    if profile_path is None or not profile_path.exists():
        return {}
    profile_payload = json.loads(profile_path.read_text())
    report = build_report(profile_payload, profile_path)
    base = profile_path.stem
    json_path = output_dir / f"{base}_quality.json"
    md_path = output_dir / f"{base}_quality.md"
    write_json(json_path, report)
    write_markdown(md_path, markdown_report(report))
    traffic = profile_payload.get("traffic") or []
    return {
        "profile_quality_json_path": str(json_path),
        "profile_quality_md_path": str(md_path),
        **report["summary"],
        "profile_src_dim": len(traffic[0]) if traffic else None,
        "profile_source_definition": profile_payload.get("source_definition"),
    }


def load_prepared_ids(dataset_path: Path, *, limit: int) -> list[str]:
    rows = datasets.load_dataset("parquet", data_files=str(dataset_path), split="train")
    n = len(rows) if limit <= 0 else min(limit, len(rows))
    return [str(rows[idx]["id"]) for idx in range(n)]


def check_output_order(result_json_path: Path | None, expected_ids: list[str]) -> dict[str, Any]:
    if result_json_path is None or not result_json_path.exists():
        return {"output_order_ok": False, "output_order_note": "missing result json"}
    payload = json.loads(result_json_path.read_text())
    per_item = payload.get("per_item", []) or []
    actual_ids = [str(item.get("id")) for item in per_item]
    actual_idx = [int(item.get("idx", -1)) for item in per_item]
    expected_idx = list(range(len(actual_ids)))
    mismatches = []
    for idx, (expected_id, actual_id) in enumerate(zip(expected_ids, actual_ids)):
        if expected_id != actual_id:
            mismatches.append({"idx": idx, "expected_id": expected_id, "actual_id": actual_id})
            if len(mismatches) >= 5:
                break
    return {
        "output_order_ok": actual_ids == expected_ids[: len(actual_ids)] and actual_idx == expected_idx,
        "output_order_note": "ok" if not mismatches and actual_idx == expected_idx else "mismatch",
        "output_order_mismatches": mismatches,
    }


def render_findings_markdown(
    *,
    title: str,
    bullets: list[str],
    rows: list[dict[str, Any]] | None = None,
    columns: list[str] | None = None,
) -> str:
    lines = [f"# {title}", ""]
    for bullet in bullets:
        lines.append(f"- {bullet}")
    if rows and columns:
        lines.extend(["", "|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"])
        for row in rows:
            lines.append("|" + "|".join(str(row.get(col, "")) for col in columns) + "|")
    lines.append("")
    return "\n".join(lines)


def gather_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def copy_if_present(path: Path | None, dst_dir: Path) -> Path | None:
    if path is None or not path.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / path.name
    if path.resolve() != dst.resolve():
        shutil.copy2(path, dst)
    return dst
