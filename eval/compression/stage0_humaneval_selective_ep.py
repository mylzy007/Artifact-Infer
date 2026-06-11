"""HumanEval pass@1 on E1/E2 selective compression configs — REAL EP=8 path.

After modifying dispatch_ep_ht.py / combine_ep_ht.py to support
MOE_COMBINE_SELECT_MODE and MOE_COMBINE_SELECT_FRAC env vars (Phase 5
selective mode), we can run HumanEval on the production EP=8 NCCL path
~6x faster than the HF + device_map='auto' pipeline-parallel approach.

Configs:
  teacher                 — no compression
  baseline_uniform_4x     — uniform topk_l2+FP8 with keep_frac=0.25
  selective_row_weight_50 — compress bottom 50% of rows by routing weight
                            (approximates E2 T=0.5 from the HF sweep)
  selective_row_weight_875 — compress bottom 87.5% of rows by routing weight
                             (approximates E2 T=top1_only)
  selective_row_norm_50   — compress bottom 50% of rows by recv_hidden L2 norm
                            (approximates E1 hidden_norm cf=0.5)

Run:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
    CUDA_HOME=/usr/local/cuda-12.8 \\
    PATH="/usr/local/cuda-12.8/bin:/home/lzy/miniconda3/envs/vllm/bin:$PATH" \\
    /home/lzy/miniconda3/envs/vllm/bin/python -m eval.compression.stage0_humaneval_selective_ep
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "8"))
MOE_IMPL = os.environ.get("MOE_IMPL", "ep_ht")
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "1") == "1"
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/Qwen3-30B-A3B"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "384"))
N_PROBLEMS = int(os.environ.get("N_PROBLEMS", "-1"))
GRADER_WORKERS = int(os.environ.get("GRADER_WORKERS", "8"))
GRADER_TIMEOUT = float(os.environ.get("GRADER_TIMEOUT", "10"))

OUT_DIR = Path("eval_results/compression_lowrank_stage0_humaneval_selective_ep")

STOP_PATTERNS = [
    "\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n\n\n",
]


def truncate_completion(text: str) -> str:
    earliest = len(text)
    for pat in STOP_PATTERNS:
        i = text.find(pat)
        if 0 <= i < earliest:
            earliest = i
    return text[:earliest]


# label, keep_frac, select_mode, select_frac
CONFIGS = [
    ("teacher",                 None, "uniform",     1.0),
    ("baseline_uniform_4x",     0.25, "uniform",     1.0),
    ("selective_row_weight_50", 0.25, "row_weight",  0.5),
    ("selective_row_weight_875",0.25, "row_weight",  0.875),
    ("selective_row_norm_50",   0.25, "row_norm",    0.5),
]


def set_env_for_config(keep_frac, select_mode, select_frac):
    if keep_frac is None:
        os.environ.pop("MOE_COMBINE_COMPRESS_KEEP_FRAC", None)
        os.environ.pop("MOE_COMBINE_COMPRESS_FP8", None)
        os.environ.pop("MOE_COMBINE_COMPRESS_NO_L2", None)
        os.environ.pop("MOE_COMBINE_SELECT_MODE", None)
        os.environ.pop("MOE_COMBINE_SELECT_FRAC", None)
    else:
        os.environ["MOE_COMBINE_COMPRESS_KEEP_FRAC"] = str(keep_frac)
        os.environ["MOE_COMBINE_COMPRESS_FP8"] = "1"
        os.environ["MOE_COMBINE_COMPRESS_NO_L2"] = "0"
        os.environ["MOE_COMBINE_SELECT_MODE"] = select_mode
        os.environ["MOE_COMBINE_SELECT_FRAC"] = str(select_frac)


def worker(rank: int, world_size: int):
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29721")
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{rank}"),
        )
        torch.set_default_device(f"cuda:{rank}")
        if rank == 0:
            print(f"[hes] world={world_size} moe_impl={MOE_IMPL} max_new_tokens={MAX_NEW_TOKENS}", flush=True)

        from workshop.nanovllm_moe.services.engine.llm_engine import LLMEngine
        from workshop.nanovllm_moe.services.sampling_params import SamplingParams
        from human_eval.data import read_problems
        from human_eval.execution import check_correctness

        kwargs = dict(
            max_num_batched_tokens=16384,
            max_num_seqs=32,
            max_model_len=1024,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1,
            enforce_eager=ENFORCE_EAGER,
            moe_impl=MOE_IMPL,
            moe_combine_compress_keep_frac=0.0,
        )

        if rank == 0:
            print("[hes] constructing LLMEngine...", flush=True)
        t0 = time.perf_counter()
        engine = LLMEngine(model=MODEL_PATH, **kwargs)
        if rank == 0:
            print(f"[hes] LLMEngine ready in {time.perf_counter() - t0:.1f}s", flush=True)

        problems = read_problems()
        items = list(problems.items())
        if N_PROBLEMS > 0:
            items = items[:N_PROBLEMS]
        prompts = [p["prompt"] for _, p in items]
        task_ids = [tid for tid, _ in items]
        if rank == 0:
            print(f"[hes] {len(items)} problems", flush=True)

        sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

        if rank == 0:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "completions").mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "results").mkdir(parents=True, exist_ok=True)

        summary = {
            "model_path": MODEL_PATH,
            "n_problems": len(items),
            "max_new_tokens": MAX_NEW_TOKENS,
            "configs": {},
        }

        for label, keep_frac, select_mode, select_frac in CONFIGS:
            if rank == 0:
                print(f"\n[hes] === config: {label} (keep_frac={keep_frac}, mode={select_mode}, frac={select_frac}) ===", flush=True)
            set_env_for_config(keep_frac, select_mode, select_frac)
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = engine.generate(prompts, sp, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            if rank == 0:
                completions = {}
                for tid, o in zip(task_ids, outputs):
                    completions[tid] = truncate_completion(o.get("text", ""))
                with (OUT_DIR / "completions" / f"{label}.jsonl").open("w") as f:
                    for tid in task_ids:
                        f.write(json.dumps({"task_id": tid, "completion": completions[tid]}) + "\n")
                print(f"[hes]   generation took {elapsed:.1f}s ({elapsed/len(items):.2f}s/prob)", flush=True)

                results = []
                t1 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=GRADER_WORKERS) as pool:
                    futures = {}
                    for tid, problem in items:
                        fut = pool.submit(
                            check_correctness, problem, completions[tid],
                            GRADER_TIMEOUT, completion_id=0,
                        )
                        futures[fut] = tid
                    for fut in as_completed(futures):
                        r = fut.result()
                        results.append({
                            "task_id": futures[fut],
                            "passed": bool(r.get("passed", False)),
                            "result": str(r.get("result", "")),
                        })
                grade_elapsed = time.perf_counter() - t1
                results.sort(key=lambda x: int(x["task_id"].split("/")[-1]))
                with (OUT_DIR / "results" / f"{label}.jsonl").open("w") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")
                n_pass = sum(1 for r in results if r["passed"])
                pa1 = n_pass / max(len(results), 1)
                summary["configs"][label] = {
                    "keep_frac": keep_frac,
                    "select_mode": select_mode,
                    "select_frac": select_frac,
                    "n_pass": n_pass,
                    "n_total": len(results),
                    "pass_at_1": pa1,
                    "gen_time_s": elapsed,
                    "grade_time_s": grade_elapsed,
                }
                print(
                    f"[hes]   {label}: pass@1 = {pa1*100:.2f}% ({n_pass}/{len(results)}) "
                    f"gen={elapsed:.1f}s",
                    flush=True,
                )
            dist.barrier()

        if rank == 0:
            teacher_pa1 = summary["configs"]["teacher"]["pass_at_1"]
            base_pa1 = summary["configs"]["baseline_uniform_4x"]["pass_at_1"]
            for lbl, c in summary["configs"].items():
                c["pass_drop_vs_teacher_pp"] = (teacher_pa1 - c["pass_at_1"]) * 100
                c["pass_drop_vs_baseline_pp"] = (base_pa1 - c["pass_at_1"]) * 100
            (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
            print(f"\n[hes] wrote {OUT_DIR/'summary.json'}", flush=True)

            lines = []
            lines.append("# HumanEval pass@1 — E1/E2 selective compression (real EP=8)")
            lines.append("")
            lines.append(f"{len(items)} problems, greedy, max_new_tokens={MAX_NEW_TOKENS}.")
            lines.append("")
            lines.append("| config | mode | frac | pass@1 | drop vs teacher | drop vs 4x baseline |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for lbl, c in summary["configs"].items():
                lines.append(
                    f"| {lbl} | {c['select_mode']} | {c['select_frac']:.3f} | "
                    f"{c['pass_at_1']*100:.2f}% ({c['n_pass']}/{c['n_total']}) | "
                    f"{c['pass_drop_vs_teacher_pp']:+.2f}pp | "
                    f"{c['pass_drop_vs_baseline_pp']:+.2f}pp |"
                )
            (OUT_DIR / "report.md").write_text("\n".join(lines))
            print(f"[hes] wrote {OUT_DIR/'report.md'}", flush=True)

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import numpy as np
                labels = list(summary["configs"].keys())
                ys = [summary["configs"][l]["pass_at_1"] * 100 for l in labels]
                fig, ax = plt.subplots(figsize=(9, 4.5))
                colors = ["C3"] + ["C0"] + ["C2"] * (len(labels) - 2)
                bars = ax.bar(range(len(labels)), ys, color=colors)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=20, ha="right")
                ax.set_ylabel("HumanEval pass@1 (%)")
                ax.axhline(ys[0], color="red", linestyle="--", linewidth=0.8, label=f"teacher = {ys[0]:.2f}%")
                ax.axhline(ys[1], color="blue", linestyle=":", linewidth=0.8, label=f"baseline 4x = {ys[1]:.2f}%")
                for bar, y in zip(bars, ys):
                    ax.text(bar.get_x() + bar.get_width() / 2, y + 1, f"{y:.1f}%", ha="center", fontsize=9)
                ax.grid(True, axis="y", alpha=0.3)
                ax.legend(fontsize=8)
                ax.set_title(f"E1/E2 selective compression — HumanEval pass@1 ({len(items)} problems, real EP=8)")
                fig.tight_layout()
                fig.savefig(OUT_DIR / "pass_at_1.png", dpi=140)
                plt.close(fig)
                print(f"[hes] wrote {OUT_DIR/'pass_at_1.png'}", flush=True)
            except Exception as e:
                print(f"plot failed: {e}", flush=True)

        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"[rank {rank}] FAIL: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


def main() -> int:
    if WORLD_SIZE == 1:
        worker(0, 1)
    else:
        mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
