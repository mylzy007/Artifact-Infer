"""Run MoE placement experiments for EP-LL / EP-HT.

Default is intended for full-model placement runs on 8x RTX 4090 without NVLink:
two parallel 4-GPU TPxEP jobs, full layers, 1 GSM8K prompt, 8 decode tokens, and
512-token model/batch budget.

Flow:
  1. Profile contiguous once with the first requested implementation, unless
     --profile-impl overrides it.
  2. Generate fixed_random_shuffle, load-balanced, and communication-aware JSON.
  3. Eval every requested moe_impl x placement.
  4. Write structured run summary, logs, and append a markdown report.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import socket
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass, asdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO / "datasets" / "gsm8k_moe_smoke.parquet"
DEFAULT_MODEL = Path("/home/lzy/models/Qwen3-30B-A3B")
DEFAULT_OUTPUT_ROOT = REPO / "eval_results"


@dataclass
class Job:
    name: str
    cmd: list[str]
    env: dict[str, str]
    log_path: str
    timeout_s: int
    output_prefix: str | None = None
    impl: str | None = None
    placement: str | None = None
    world_size: int | None = None
    tp_size: int | None = None
    gpu_group: str | None = None
    port_retries: int = 0


def shlex_join(cmd: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(x) for x in cmd)


def find_free_port(start_port: int, reserved: set[int] | None = None) -> int:
    reserved = reserved or set()
    for port in range(int(start_port), 65535):
        if port in reserved:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"could not find a free port at or above {start_port}")


def reserve_master_port(args: argparse.Namespace, start_port: int, reserved: set[int]) -> int:
    if args.dry_run:
        port = int(start_port)
        while port in reserved:
            port += 1
        return port
    return find_free_port(start_port, reserved)


def get_master_port(cmd: list[str]) -> int | None:
    try:
        idx = cmd.index("--master-port")
    except ValueError:
        return None
    if idx + 1 >= len(cmd):
        return None
    return int(cmd[idx + 1])


def set_master_port(cmd: list[str], port: int) -> list[str]:
    updated = list(cmd)
    idx = updated.index("--master-port")
    updated[idx + 1] = str(port)
    return updated


def sanitize_label(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "run"


def resolve_output_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO / candidate


def output_root_path(args: argparse.Namespace) -> Path:
    root = Path(getattr(args, "output_root", DEFAULT_OUTPUT_ROOT))
    return root if root.is_absolute() else REPO / root


def make_run_dir(args: argparse.Namespace, default_name: str) -> Path:
    explicit = getattr(args, "run_dir", None)
    if explicit:
        return Path(explicit)
    output_root = output_root_path(args)
    name = sanitize_label(getattr(args, "experiment_name", None) or default_name)
    run_id = sanitize_label(args.run_id)
    return output_root / f"{name}_{run_id}"


def is_port_in_use_log(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(errors="replace").lower()
    return (
        "eaddrinuse" in text
        or "address already in use" in text
        or "failed to listen on any local network address" in text
    )


def run_job(job: Job) -> dict:
    t0 = time.time()
    log_path = Path(job.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = list(job.cmd)
    retries = 0
    timed_out = False
    exit_code = 1

    while True:
        mode = "w" if retries == 0 else "a"
        with log_path.open(mode) as log:
            if retries:
                log.write(f"\n[RETRY {retries}] retrying after master-port collision\n")
            log.write("$ " + shlex_join(cmd) + "\n")
            log.write(f"# CUDA_VISIBLE_DEVICES={job.env.get('CUDA_VISIBLE_DEVICES', '')}\n\n")
            log.flush()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=REPO,
                    env=job.env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=job.timeout_s,
                )
                exit_code = proc.returncode
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                log.write(f"\n[TIMEOUT] exceeded {job.timeout_s}s\n{exc}\n")
                exit_code = 124
                timed_out = True

        master_port = get_master_port(cmd)
        if (
            exit_code != 0
            and not timed_out
            and master_port is not None
            and retries < job.port_retries
            and is_port_in_use_log(log_path)
        ):
            retries += 1
            cmd = set_master_port(cmd, find_free_port(master_port + 1000 + retries * 100))
            continue
        break

    elapsed = time.time() - t0
    summary = {
        **public_job_dict(job),
        "command": shlex_join(cmd),
        "exit_code": exit_code,
        "pass": exit_code == 0,
        "timed_out": timed_out,
        "elapsed_s": elapsed,
        "port_retries_used": retries,
    }
    summary.update(extract_outputs(summary))
    return summary


def public_job_dict(job: Job) -> dict:
    data = asdict(job)
    env = data.pop("env", {})
    env_keys = (
        "CUDA_VISIBLE_DEVICES",
        "MOE_PROFILE_RUN_ID",
        "PYTHONUNBUFFERED",
        "CUDA_HOME",
        "FLASHINFER_NVCC",
        "FLASHINFER_WORKSPACE_BASE",
    )
    data["env"] = {k: env[k] for k in env_keys if k in env}
    return data


def extract_outputs(summary: dict) -> dict:
    out: dict[str, object] = {}
    log_path = Path(summary["log_path"])
    if not log_path.exists():
        out["log_missing"] = True
        out["exception_summary"] = [f"log file missing: {log_path}"]
        return out

    text = log_path.read_text(errors="replace")
    profile_matches = re.findall(r"\[moe-profile\]\s+wrote\s+([^\s]+\.json)", text)
    if not profile_matches:
        profile_matches = re.findall(r"([^\s]*moe_routing_profile_[A-Za-z0-9_.-]+\.json)", text)
    if profile_matches:
        out["profile_path"] = profile_matches[-1]
    result_matches = re.findall(r"results:\s+([^\s]+\.json)", text)
    rollout_matches = re.findall(r"rollouts:\s+([^\s]+\.txt)", text)
    if result_matches:
        out["result_json_path"] = result_matches[-1]
        try:
            payload = json.loads(resolve_output_path(result_matches[-1]).read_text())
            metrics = payload.get("summary", {}).get("generation_metrics", {})
            out["metrics"] = metrics
            out["total_generated_tokens"] = payload.get("summary", {}).get("total_generated_tokens")
        except Exception as exc:
            out["result_json_error"] = f"{type(exc).__name__}: {exc}"
    if rollout_matches:
        out["rollout_txt_path"] = rollout_matches[-1]
    if not summary["pass"]:
        lines = text.splitlines()
        interesting = [
            line for line in lines
            if "Traceback" in line or "RuntimeError" in line or "CUDA out of memory" in line
            or "NCCL" in line or "TIMEOUT" in line or "EXCEPTION" in line
        ]
        out["exception_summary"] = interesting[-12:]
        out["oom"] = any("out of memory" in line.lower() for line in lines)
    return out


def make_eval_job(
    *,
    args,
    impl: str,
    placement: str,
    placement_path: str | None,
    profile: bool,
    run_id: str,
    gpu_group: str,
    master_port: int,
    log_dir: Path,
) -> Job:
    ep_size = args.world_size // args.tp_size
    cmd = [
        args.python,
        "-m",
        "eval.test_bazaar_moe",
        "--model-path",
        str(args.model_path),
        "--dataset-path",
        str(args.dataset_path),
        "--dataset-label",
        str(args.dataset_label),
        "--world-size",
        str(args.world_size),
        "--tp-size",
        str(args.tp_size),
        "--master-port",
        str(master_port),
        "--moe-impl",
        impl,
        "--moe-expert-placement",
        placement,
        "--moe-expert-placement-seed",
        str(args.seed),
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
        "1" if profile else "0",
        "--output-dir",
        str(log_dir.parent),
    ]
    if args.moe_ll_m_max > 0:
        cmd += ["--moe-ll-m-max", str(args.moe_ll_m_max)]
    if placement_path:
        cmd += ["--moe-expert-placement-path", placement_path]
    env = os.environ.copy()
    python_bin = str(Path(args.python).resolve().parent)
    cuda_home = str(args.cuda_home) if args.cuda_home else ""
    path_parts = [python_bin]
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
        env["FLASHINFER_NVCC"] = str(Path(cuda_home) / "bin" / "nvcc")
        path_parts.append(str(Path(cuda_home) / "bin"))
        lib64 = str(Path(cuda_home) / "lib64")
        env["LD_LIBRARY_PATH"] = lib64 + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["FLASHINFER_WORKSPACE_BASE"] = str(log_dir.parent / "flashinfer_workspace")
    env["PATH"] = os.pathsep.join(path_parts + [env.get("PATH", "")])
    env["CUDA_VISIBLE_DEVICES"] = gpu_group
    env["MOE_PROFILE_RUN_ID"] = run_id
    env.setdefault("PYTHONUNBUFFERED", "1")
    name = f"{impl}_{placement}_ws{args.world_size}_tp{args.tp_size}"
    return Job(
        name=name,
        cmd=cmd,
        env=env,
        log_path=str(log_dir / f"{name}.log"),
        timeout_s=args.timeout_s,
        output_prefix=(
            f"{args.dataset_label}_{impl}_{placement}_tp{args.tp_size}_ep{ep_size}"
            f"_eager{int(bool(args.enforce_eager))}"
            f"{f'_layers{args.num_layers}' if args.num_layers > 0 else ''}"
        ),
        impl=impl,
        placement=placement,
        world_size=args.world_size,
        tp_size=args.tp_size,
        gpu_group=gpu_group,
        port_retries=args.port_retries,
    )


def run_placement_generator(args, profile_path: str, policy: str, run_dir: Path) -> dict:
    cmd = [
        args.python,
        "-m",
        "eval.generate_moe_placement",
        "--profile",
        profile_path,
        "--policy",
        policy,
        "--output-dir",
        str(run_dir),
    ]
    if policy == "fixed_random_shuffle":
        cmd += ["--seed", str(args.seed)]
    log_path = run_dir / "logs" / f"generate_{policy}.log"
    job = Job(
        name=f"generate_{policy}",
        cmd=cmd,
        env=os.environ.copy(),
        log_path=str(log_path),
        timeout_s=300,
        placement=policy,
    )
    result = run_job(job)
    if not log_path.exists():
        result["parse_error"] = f"log file missing: {log_path}"
        return result
    text = log_path.read_text(errors="replace")
    try:
        payload = json.loads(text[text.find("{") : text.rfind("}") + 1])
        result["placement_path"] = payload["placement_path"]
        result["estimated_metrics"] = payload.get("estimated_metrics", {})
    except Exception as exc:
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
    return result


def gpu_groups(args) -> list[str]:
    ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    groups = []
    for start in range(0, len(ids), args.gpus_per_job):
        group = ids[start : start + args.gpus_per_job]
        if len(group) == args.gpus_per_job:
            groups.append(",".join(group))
    if not groups:
        raise SystemExit("no complete GPU groups available")
    return groups[: args.parallel_jobs]


def run_eval_jobs(args, jobs: list[Job], run_dir: Path) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    results = []
    if args.dry_run:
        for job in jobs:
            results.append({**public_job_dict(job), "command": shlex_join(job.cmd), "dry_run": True})
        return results

    with ThreadPoolExecutor(max_workers=args.parallel_jobs) as pool:
        pending = set()
        future_to_job: dict[object, Job] = {}
        active_groups: set[str] = set()
        job_queue = deque(jobs)
        while True:
            scheduled_any = True
            while len(pending) < args.parallel_jobs and job_queue and scheduled_any:
                scheduled_any = False
                for _ in range(len(job_queue)):
                    job = job_queue.popleft()
                    group = job.gpu_group or ""
                    if group in active_groups:
                        job_queue.append(job)
                        continue
                    fut = pool.submit(run_job, job)
                    pending.add(fut)
                    future_to_job[fut] = job
                    active_groups.add(group)
                    scheduled_any = True
                    break
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                job = future_to_job.pop(fut)
                active_groups.discard(job.gpu_group or "")
                result = fut.result()
                results.append(result)
                append_report(run_dir, [result])
    return results


def append_report(run_dir: Path, rows: list[dict]) -> None:
    report = run_dir / "moe_placement_experiment_report.md"
    exists = report.exists()
    with report.open("a") as f:
        if not exists:
            f.write("# MoE Placement Experiment Run\n\n")
            f.write("| impl | placement | gpus | pass | exit | e2e_s | prefill_tok_s | decode_tok_s | log | output | notes |\n")
            f.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |\n")
        for r in rows:
            metrics = r.get("metrics", {}) or {}
            log = r.get("log_path", "")
            out = r.get("result_json_path", "")
            notes = ""
            if r.get("timed_out"):
                notes = "timeout"
            elif r.get("oom"):
                notes = "OOM"
            elif r.get("exception_summary"):
                notes = "failed; see log"
            f.write(
                f"| {r.get('impl','')} | {r.get('placement','')} | {r.get('gpu_group','')} | "
                f"{r.get('pass', False)} | {r.get('exit_code','')} | "
                f"{metrics.get('e2e_total_time_s','')} | "
                f"{metrics.get('prefill_throughput_tok_s','')} | "
                f"{metrics.get('decode_throughput_tok_s','')} | "
                f"`{log}` | `{out}` | {notes} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/home/lzy/miniconda3/envs/vllm/bin/python")
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=Path("/usr/local/cuda-12.8") if Path("/usr/local/cuda-12.8/bin/nvcc").exists() else None,
        help="CUDA toolkit for FlashInfer JIT; must support the GPU arch",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-label", default="gsm8k")
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--gpus-per-job", type=int, default=4)
    parser.add_argument("--parallel-jobs", type=int, default=2)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument(
        "--world-sizes",
        default=None,
        help=(
            "comma-separated sweep, e.g. 8,4. When set, runs each world size "
            "sequentially; gpus-per-job is set to world-size and parallel-jobs "
            "is derived from available GPUs."
        ),
    )
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--impls", default="ep_ll_triton,ep_ht")
    parser.add_argument(
        "--profile-impl",
        default=None,
        help="implementation used for the routing profile; defaults to the first --impls entry",
    )
    parser.add_argument(
        "--placements",
        default=(
            "contiguous,round_robin,fixed_random_shuffle,"
            "load_balanced_greedy_with_locality_tiebreak,communication_aware_greedy"
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-layers", type=int, default=-1)
    parser.add_argument("--num-problems", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--scale-steps",
        default=None,
        help=(
            "progressive problem/token sweep such as 1x8,4x32,8x64. "
            "Each step runs all requested world sizes and stops before larger "
            "steps if any run fails."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", type=int, default=1)
    parser.add_argument("--moe-ll-overflow-policy", default="drop", choices=["drop", "error"])
    parser.add_argument("--moe-ll-m-max", type=int, default=-1)
    parser.add_argument("--base-master-port", type=int, default=29600)
    parser.add_argument("--port-retries", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--experiment-name", default=None, help="name prefix for this top-level experiment directory")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="root directory for experiment outputs")
    parser.add_argument("--run-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-profile", type=str, default=None, help="reuse existing profile json")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise SystemExit("--world-sizes must contain at least one integer")
    return out


def parse_scale_steps(value: str) -> list[tuple[int, int]]:
    steps = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" in item:
            problems, tokens = item.split("x", 1)
        elif ":" in item:
            problems, tokens = item.split(":", 1)
        else:
            raise SystemExit("--scale-steps entries must look like 1x8 or 1:8")
        steps.append((int(problems), int(tokens)))
    if not steps:
        raise SystemExit("--scale-steps must contain at least one entry")
    return steps


def world_sizes_for_args(args: argparse.Namespace) -> list[int]:
    if args.world_sizes:
        return parse_int_list(args.world_sizes)
    return [args.world_size]


def run_progressive_scale_sweep(args: argparse.Namespace) -> None:
    if args.skip_profile:
        raise SystemExit("--skip-profile is ambiguous with --scale-steps; run one scale step at a time")

    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    base_run_id = args.run_id
    world_sizes = world_sizes_for_args(args)
    scale_steps = parse_scale_steps(args.scale_steps)
    sweep_dir = make_run_dir(args, "moe_placement_scale_sweep")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_summary: dict[str, object] = {
        "run_id": base_run_id,
        "world_sizes": world_sizes,
        "scale_steps": [
            {"num_problems": num_problems, "max_tokens": max_tokens}
            for num_problems, max_tokens in scale_steps
        ],
        "runs": [],
        "stopped": False,
        "stop_reason": None,
    }

    stop = False
    for step_idx, (num_problems, max_tokens) in enumerate(scale_steps):
        if stop:
            break
        for world_idx, world_size in enumerate(world_sizes):
            if world_size > len(gpu_ids):
                raise SystemExit(f"world_size={world_size} needs {world_size} GPUs, only got {len(gpu_ids)}")
            child = copy.copy(args)
            child.scale_steps = None
            child.world_sizes = None
            child.world_size = world_size
            child.gpus_per_job = world_size
            child.parallel_jobs = max(1, len(gpu_ids) // world_size)
            child.num_problems = num_problems
            child.max_tokens = max_tokens
            child.run_id = f"{base_run_id}_p{num_problems}_t{max_tokens}_ws{world_size}"
            child.run_dir = sweep_dir / f"p{num_problems}_t{max_tokens}_ws{world_size}"
            child.experiment_name = None
            child.output_root = sweep_dir
            child.base_master_port = args.base_master_port + step_idx * 1000 + world_idx * 100

            print(
                f"[scale] starting num_problems={num_problems}, max_tokens={max_tokens}, "
                f"world_size={world_size}, gpus_per_job={child.gpus_per_job}, "
                f"parallel_jobs={child.parallel_jobs}, run_id={child.run_id}",
                flush=True,
            )
            run_summary = run_single(child)
            run_record = {
                "num_problems": num_problems,
                "max_tokens": max_tokens,
                "world_size": world_size,
                "run_id": child.run_id,
                "summary_path": str(run_summary["summary_path"]),
                "report_path": str(run_summary["report_path"]),
                "all_pass": run_summary["all_pass"],
            }
            sweep_summary["runs"].append(run_record)
            if not run_summary["all_pass"]:
                sweep_summary["stopped"] = True
                sweep_summary["stop_reason"] = (
                    f"failed at num_problems={num_problems}, "
                    f"max_tokens={max_tokens}, world_size={world_size}"
                )
                stop = True
            (sweep_dir / "summary.json").write_text(json.dumps(sweep_summary, indent=2))
            if stop:
                break

    print(f"[scale] wrote {sweep_dir / 'summary.json'}")
    if sweep_summary["stopped"]:
        print(f"[scale] stopped: {sweep_summary['stop_reason']}")


def run_world_size_sweep(args: argparse.Namespace) -> None:
    if args.skip_profile:
        raise SystemExit("--skip-profile is ambiguous with --world-sizes; run one world size at a time")

    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    base_run_id = args.run_id
    world_sizes = parse_int_list(args.world_sizes)
    sweep_dir = make_run_dir(args, "moe_placement_sweep")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_summary: dict[str, object] = {
        "run_id": base_run_id,
        "world_sizes": world_sizes,
        "runs": [],
    }

    for offset, world_size in enumerate(world_sizes):
        if world_size > len(gpu_ids):
            raise SystemExit(f"world_size={world_size} needs {world_size} GPUs, only got {len(gpu_ids)}")
        child = copy.copy(args)
        child.world_sizes = None
        child.world_size = world_size
        child.gpus_per_job = world_size
        child.parallel_jobs = max(1, len(gpu_ids) // world_size)
        child.run_id = f"{base_run_id}_ws{world_size}"
        child.run_dir = sweep_dir / f"ws{world_size}"
        child.experiment_name = None
        child.output_root = sweep_dir
        child.base_master_port = args.base_master_port + offset * 100

        print(
            f"[sweep] starting world_size={world_size}, "
            f"gpus_per_job={child.gpus_per_job}, parallel_jobs={child.parallel_jobs}, "
            f"run_id={child.run_id}",
            flush=True,
        )
        run_summary = run_single(child)
        sweep_summary["runs"].append(
            {
                "world_size": world_size,
                "run_id": child.run_id,
                "summary_path": str(run_summary["summary_path"]),
                "report_path": str(run_summary["report_path"]),
                "all_pass": run_summary["all_pass"],
            }
        )
        (sweep_dir / "summary.json").write_text(json.dumps(sweep_summary, indent=2))

    print(f"[sweep] wrote {sweep_dir / 'summary.json'}")


def run_single(args: argparse.Namespace) -> dict:

    if args.world_size != args.gpus_per_job:
        raise SystemExit("--world-size should match --gpus-per-job for CUDA_VISIBLE_DEVICES groups")
    if args.world_size % args.tp_size != 0:
        raise SystemExit("--world-size must be divisible by --tp-size")
    if not args.dataset_path.exists():
        raise SystemExit(f"dataset not found: {args.dataset_path}")

    run_dir = make_run_dir(args, "moe_placement_run")
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    groups = gpu_groups(args)
    impls = [x.strip() for x in args.impls.split(",") if x.strip()]
    placements = [x.strip() for x in args.placements.split(",") if x.strip()]
    if not impls:
        raise SystemExit("--impls must contain at least one implementation")
    profile_impl = args.profile_impl or impls[0]
    reserved_ports: set[int] = set()
    profile_port = reserve_master_port(args, args.base_master_port, reserved_ports)
    reserved_ports.add(profile_port)

    summary: dict[str, object] = {
        "run_id": args.run_id,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "gpu_groups": groups,
        "profile_impl": profile_impl,
        "profile": None,
        "placement_generation": [],
        "eval_results": [],
    }

    profile_path = args.skip_profile
    if profile_path is None:
        profile_job = make_eval_job(
            args=args,
            impl=profile_impl,
            placement="contiguous",
            placement_path=None,
            profile=True,
            run_id=f"{args.run_id}_profile",
            gpu_group=groups[0],
            master_port=profile_port,
            log_dir=log_dir,
        )
        if args.dry_run:
            profile_result = {**public_job_dict(profile_job), "command": shlex_join(profile_job.cmd), "dry_run": True}
        else:
            profile_result = run_job(profile_job)
        summary["profile"] = profile_result
        profile_path = profile_result.get("profile_path")
        if not profile_path and not args.dry_run:
            summary_path = run_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            raise SystemExit("profile run did not produce a profile json; see logs")

    placement_paths: dict[str, str | None] = {
        "contiguous": None,
        "round_robin": None,
    }
    for policy in (
        "fixed_random_shuffle",
        "load_balanced_greedy_with_locality_tiebreak",
        "communication_aware_greedy",
    ):
        if policy not in placements:
            continue
        if args.dry_run:
            placement_paths[policy] = str(run_dir / f"moe_placement_{policy}_DRY_RUN.json")
            continue
        gen_result = run_placement_generator(args, str(profile_path), policy, run_dir)
        summary["placement_generation"].append(gen_result)
        placement_paths[policy] = gen_result.get("placement_path")
        if not placement_paths[policy]:
            summary_path = run_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            raise SystemExit(f"failed to generate placement for {policy}; see logs")

    jobs: list[Job] = []
    for idx, (impl, placement) in enumerate((i, p) for i in impls for p in placements):
        group = groups[idx % len(groups)]
        master_port = reserve_master_port(args, args.base_master_port + 1 + idx * 10, reserved_ports)
        reserved_ports.add(master_port)
        jobs.append(
            make_eval_job(
                args=args,
                impl=impl,
                placement=placement,
                placement_path=placement_paths.get(placement),
                profile=False,
                run_id=f"{args.run_id}_{impl}_{placement}",
                gpu_group=group,
                master_port=master_port,
                log_dir=log_dir,
            )
        )

    eval_results = run_eval_jobs(args, jobs, run_dir)
    summary["eval_results"] = eval_results
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    top_report = output_root_path(args) / "moe_placement_experiment_report.md"
    top_report.parent.mkdir(parents=True, exist_ok=True)
    with top_report.open("a") as f:
        f.write(f"\n\n## Run {args.run_id}\n\n")
        f.write(f"- Summary: `{run_dir / 'summary.json'}`\n")
        f.write(f"- Per-run report: `{run_dir / 'moe_placement_experiment_report.md'}`\n")
        f.write(f"- Dataset: `{args.dataset_path}`\n")
        f.write(f"- GPU groups: {groups}\n")
    report_path = run_dir / "moe_placement_experiment_report.md"
    print(f"[runner] wrote {summary_path}")
    print(f"[runner] report {report_path}")
    profile = summary.get("profile")
    profile_ok = True if args.skip_profile else bool(profile and profile.get("pass"))
    all_pass = bool(
        profile_ok
        and all(x.get("pass") for x in summary.get("placement_generation", []))
        and all(x.get("pass") for x in summary.get("eval_results", []))
    )
    if args.dry_run:
        all_pass = True
    return {
        "summary_path": summary_path,
        "report_path": report_path,
        "all_pass": all_pass,
    }


def main() -> None:
    args = parse_args()
    if args.scale_steps:
        run_progressive_scale_sweep(args)
    elif args.world_sizes:
        run_world_size_sweep(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
