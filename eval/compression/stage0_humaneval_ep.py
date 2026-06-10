"""Phase B (HumanEval pass@1) — real EP=8 path via nanovllm_moe LLMEngine.

Faster than the HF device_map="auto" version because EP=8 runs all 8 GPUs
concurrently (each rank holds a slice of experts + replicated attention),
whereas device_map="auto" does pipeline parallel (one GPU active at a time).

Workflow:
  spawn 8 workers (EP=8)
  -> each constructs the same LLMEngine (full Qwen3-30B-A3B)
  -> for each of 4 configs (teacher + keep=0.5/0.25/0.125), set
     MOE_COMBINE_COMPRESS_KEEP_FRAC env var, run engine.generate(all 164 prompts)
  -> rank 0 saves completions, grades via human-eval check_correctness, writes summary

Run:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
    CUDA_HOME=/usr/local/cuda-12.8 \\
    PATH="/usr/local/cuda-12.8/bin:/home/lzy/miniconda3/envs/vllm/bin:$PATH" \\
    /home/lzy/miniconda3/envs/vllm/bin/python -m eval.compression.stage0_humaneval_ep
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
N_PROBLEMS = int(os.environ.get("N_PROBLEMS", "-1"))   # -1 = all 164
KEEP_FRACS_STR = os.environ.get("KEEP_FRACS", "0.5,0.25,0.125")
GRADER_WORKERS = int(os.environ.get("GRADER_WORKERS", "8"))
GRADER_TIMEOUT = float(os.environ.get("GRADER_TIMEOUT", "10"))

OUT_DIR = Path("eval_results/compression_lowrank_stage0_humaneval_ep")

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


def worker(rank: int, world_size: int):
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29711")
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{rank}"),
        )
        torch.set_default_device(f"cuda:{rank}")
        if rank == 0:
            print(
                f"[humaneval_ep] world={world_size} moe_impl={MOE_IMPL} "
                f"eager={ENFORCE_EAGER} max_new_tokens={MAX_NEW_TOKENS} "
                f"keep_fracs={KEEP_FRACS_STR}",
                flush=True,
            )

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
            # leave compression off in the Config; we toggle per-config via env vars
            moe_combine_compress_keep_frac=0.0,
        )

        if rank == 0:
            print("[humaneval_ep] constructing LLMEngine...", flush=True)
        t0 = time.perf_counter()
        engine = LLMEngine(model=MODEL_PATH, **kwargs)
        if rank == 0:
            print(f"[humaneval_ep] LLMEngine ready in {time.perf_counter() - t0:.1f}s", flush=True)

        problems_dict = read_problems()
        items = list(problems_dict.items())
        if N_PROBLEMS > 0:
            items = items[:N_PROBLEMS]
        prompts = [p["prompt"] for _, p in items]
        task_ids = [tid for tid, _ in items]
        if rank == 0:
            print(f"[humaneval_ep] {len(items)} problems", flush=True)

        sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

        configs = [("teacher", None)] + [
            (f"keep={kf}_fp8_l2", float(kf)) for kf in KEEP_FRACS_STR.split(",") if kf.strip()
        ]

        if rank == 0:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "completions").mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "results").mkdir(parents=True, exist_ok=True)

        summary = {
            "model_path": MODEL_PATH,
            "n_problems": len(items),
            "max_new_tokens": MAX_NEW_TOKENS,
            "moe_impl": MOE_IMPL,
            "configs": {},
        }

        for label, kf in configs:
            if rank == 0:
                print(f"\n[humaneval_ep] === config: {label} ===", flush=True)
            # Set env vars on ALL ranks (each worker is a separate process).
            if kf is None:
                os.environ.pop("MOE_COMBINE_COMPRESS_KEEP_FRAC", None)
                os.environ.pop("MOE_COMBINE_COMPRESS_FP8", None)
                os.environ.pop("MOE_COMBINE_COMPRESS_NO_L2", None)
            else:
                os.environ["MOE_COMBINE_COMPRESS_KEEP_FRAC"] = str(kf)
                os.environ["MOE_COMBINE_COMPRESS_FP8"] = "1"
                os.environ["MOE_COMBINE_COMPRESS_NO_L2"] = "0"

            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = engine.generate(prompts, sp, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            if rank == 0:
                # Build (task_id, completion) list. engine.generate returns
                # dicts with 'text' and 'token_ids'.
                completions: dict[str, str] = {}
                for tid, o in zip(task_ids, outputs):
                    txt = o.get("text", "")
                    completions[tid] = truncate_completion(txt)
                # Save raw completions for re-grading / inspection.
                with (OUT_DIR / "completions" / f"{label}.jsonl").open("w") as f:
                    for tid in task_ids:
                        f.write(json.dumps({"task_id": tid, "completion": completions[tid]}) + "\n")
                print(f"[humaneval_ep]   generation took {elapsed:.1f}s ({elapsed/len(items):.2f}s/prob)", flush=True)

                # Grade in parallel.
                t1 = time.perf_counter()
                results: list[dict] = []
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
                n_pass = sum(1 for r in results if r["passed"])
                with (OUT_DIR / "results" / f"{label}.jsonl").open("w") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")
                pa1 = n_pass / max(len(results), 1)
                summary["configs"][label] = {
                    "keep_frac": kf,
                    "n_pass": n_pass,
                    "n_total": len(results),
                    "pass_at_1": pa1,
                    "gen_time_s": elapsed,
                    "grade_time_s": grade_elapsed,
                }
                print(
                    f"[humaneval_ep]   {label}: pass@1 = {pa1*100:.2f}% "
                    f"({n_pass}/{len(results)})  gen={elapsed:.1f}s  grade={grade_elapsed:.1f}s",
                    flush=True,
                )
            dist.barrier()

        if rank == 0:
            # Compare to teacher.
            teacher_p1 = summary["configs"]["teacher"]["pass_at_1"]
            for lbl in summary["configs"]:
                c = summary["configs"][lbl]
                c["pass_drop_pp"] = (teacher_p1 - c["pass_at_1"]) * 100
                c["pass_drop_rel_pct"] = (1 - c["pass_at_1"] / max(teacher_p1, 1e-12)) * 100

            (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
            print(f"\n[humaneval_ep] wrote {OUT_DIR/'summary.json'}", flush=True)

            # Markdown.
            lines = []
            lines.append("# HumanEval pass@1 under combine-side compression (Qwen3-30B-A3B, real EP=8)")
            lines.append("")
            lines.append(f"{len(items)} problems, greedy decoding, max_new_tokens={MAX_NEW_TOKENS}.")
            lines.append("")
            lines.append("| config | value compr | pass@1 | drop (pp) | relative drop | gen time |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for lbl in summary["configs"]:
                c = summary["configs"][lbl]
                kf = c["keep_frac"]
                comp = f"{int(round(1/kf))}x" if kf else "-"
                lines.append(
                    f"| {lbl} | {comp} | {c['pass_at_1']*100:.2f}% ({c['n_pass']}/{c['n_total']}) | "
                    f"{c['pass_drop_pp']:+.2f}pp | {c['pass_drop_rel_pct']:+.1f}% | "
                    f"{c['gen_time_s']:.1f}s |"
                )
            lines.append("")
            (OUT_DIR / "report.md").write_text("\n".join(lines))
            print(f"[humaneval_ep] wrote {OUT_DIR/'report.md'}", flush=True)

            # Plot.
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import numpy as np
                labels = list(summary["configs"].keys())
                passes = [summary["configs"][l]["pass_at_1"] for l in labels]
                fig, ax = plt.subplots(figsize=(7, 4.5))
                xs = np.arange(len(labels))
                bars = ax.bar(xs, [p * 100 for p in passes], color=["C3"] + ["C0"] * (len(labels) - 1))
                ax.axhline(passes[0] * 100, color="red", linestyle="--", linewidth=0.8,
                           label=f"teacher pass@1 = {passes[0]*100:.2f}%")
                ax.set_xticks(xs)
                ax.set_xticklabels(labels, rotation=20)
                ax.set_ylabel("HumanEval pass@1 (%)")
                ax.set_title(f"HumanEval pass@1 — Qwen3-30B-A3B + combine compression (real EP=8)")
                ax.legend(fontsize=8)
                ax.grid(True, axis="y", alpha=0.3)
                for bar, p in zip(bars, passes):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f"{p*100:.1f}%", ha="center", fontsize=9)
                fig.tight_layout()
                fig.savefig(OUT_DIR / "pass_at_1.png", dpi=140)
                plt.close(fig)
                print(f"[humaneval_ep] wrote {OUT_DIR/'pass_at_1.png'}", flush=True)
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
