"""Full-model EP=8 evaluation: combine-side compression vs uncompressed teacher.

Loads the full 48-layer Qwen3-30B-A3B under EP-HT on 8x4090, generates greedy
output for a small batch of prompts twice in the same engine:
  1. no compression  (teacher)
  2. with MOE_COMBINE_COMPRESS_KEEP_FRAC set (student)

Reports:
  - teacher vs student token-level match rate
  - per-batch wall-clock time
  - first ~30 generated chars for human eyeball check

Run a single config:
  WORLD_SIZE=8 KEEP_FRAC=0.25 USE_FP8=1 \\
    /home/lzy/miniconda3/envs/vllm/bin/python -m eval.compression.test_ep_ht_compress_full

Multiple configs in one engine load by passing comma-separated:
  WORLD_SIZE=8 KEEP_FRACS=0.5,0.25,0.125 USE_FP8=1 \\
    /home/lzy/miniconda3/envs/vllm/bin/python -m eval.compression.test_ep_ht_compress_full
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "8"))
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "-1"))
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "1") == "1"
MOE_IMPL = os.environ.get("MOE_IMPL", "ep_ht")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/Qwen3-30B-A3B"))
KEEP_FRACS = os.environ.get(
    "KEEP_FRACS", os.environ.get("KEEP_FRAC", "0.25")
).split(",")
USE_FP8 = os.environ.get("USE_FP8", "1")
USE_L2 = os.environ.get("USE_L2", "1")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32"))

PROMPTS = [
    "The capital of France is",
    "Write a Python function to compute the factorial of n.\n\ndef factorial(n):",
    "Q: What is 17 * 23?\nA:",
    "The quick brown fox",
]


def worker(rank: int, world_size: int):
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29615")
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
                f"[rank 0] world={world_size} layers={NUM_LAYERS} "
                f"eager={ENFORCE_EAGER} moe_impl={MOE_IMPL} "
                f"KEEP_FRACS={KEEP_FRACS} USE_FP8={USE_FP8} USE_L2={USE_L2} "
                f"MAX_TOKENS={MAX_TOKENS}",
                flush=True,
            )

        from workshop.nanovllm_moe.services.engine.llm_engine import LLMEngine
        from workshop.nanovllm_moe.services.sampling_params import SamplingParams

        kwargs = dict(
            max_num_batched_tokens=4096,
            max_num_seqs=8,
            max_model_len=512,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1,
            enforce_eager=ENFORCE_EAGER,
            moe_impl=MOE_IMPL,
        )
        if NUM_LAYERS > 0:
            kwargs["num_hidden_layers_override"] = NUM_LAYERS

        if rank == 0:
            print("[rank 0] constructing LLMEngine...", flush=True)
        t0 = time.perf_counter()
        engine = LLMEngine(model=MODEL_PATH, **kwargs)
        if rank == 0:
            print(f"[rank 0] LLMEngine ready in {time.perf_counter() - t0:.1f}s", flush=True)

        sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

        # --- Warmup pass (don't time, just get JIT / capture done) ---
        os.environ.pop("MOE_COMBINE_COMPRESS_KEEP_FRAC", None)
        dist.barrier()
        if rank == 0:
            print("[rank 0] warmup...", flush=True)
        _ = engine.generate(PROMPTS, sp, use_tqdm=False)

        # --- Teacher pass ---
        os.environ.pop("MOE_COMBINE_COMPRESS_KEEP_FRAC", None)
        os.environ.pop("MOE_COMBINE_COMPRESS_FP8", None)
        os.environ.pop("MOE_COMBINE_COMPRESS_NO_L2", None)
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_teacher = engine.generate(PROMPTS, sp, use_tqdm=False)
        torch.cuda.synchronize()
        t_teacher = time.perf_counter() - t0
        if rank == 0:
            print(f"\n[rank 0] === teacher in {t_teacher:.2f}s ===", flush=True)
            for i, o in enumerate(out_teacher):
                txt = o["text"]
                print(f"[rank 0] p[{i}] text: {txt[:80]!r}")

        # --- Sweep student passes ---
        students: dict[str, dict] = {}
        for kf_str in KEEP_FRACS:
            kf_str = kf_str.strip()
            if not kf_str or float(kf_str) <= 0:
                continue
            os.environ["MOE_COMBINE_COMPRESS_KEEP_FRAC"] = kf_str
            os.environ["MOE_COMBINE_COMPRESS_FP8"] = USE_FP8
            os.environ["MOE_COMBINE_COMPRESS_NO_L2"] = "0" if USE_L2 == "1" else "1"
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_student = engine.generate(PROMPTS, sp, use_tqdm=False)
            torch.cuda.synchronize()
            t_student = time.perf_counter() - t0
            if rank == 0:
                print(
                    f"\n[rank 0] === student keep_frac={kf_str} fp8={USE_FP8} l2={USE_L2} "
                    f"in {t_student:.2f}s ===",
                    flush=True,
                )
                match_total = 0
                tok_total = 0
                for i, (te, st) in enumerate(zip(out_teacher, out_student)):
                    t_ids = te["token_ids"]
                    s_ids = st["token_ids"]
                    n = min(len(t_ids), len(s_ids))
                    match = sum(1 for j in range(n) if t_ids[j] == s_ids[j])
                    match_total += match
                    tok_total += n
                    s_txt = st["text"]
                    print(
                        f"[rank 0] p[{i}] text: {s_txt[:80]!r}  "
                        f"match {match}/{n} ({100*match/max(n,1):.0f}%)"
                    )
                students[kf_str] = {
                    "time": t_student,
                    "match_total": match_total,
                    "tok_total": tok_total,
                    "match_ratio": match_total / max(tok_total, 1),
                }

        if rank == 0:
            print("\n[rank 0] === summary ===")
            print(f"[rank 0] teacher time: {t_teacher:.2f}s")
            for kf, m in students.items():
                ratio_t = m["time"] / t_teacher
                print(
                    f"[rank 0] keep_frac={kf}: time={m['time']:.2f}s "
                    f"({ratio_t:.2f}x teacher)  "
                    f"match={m['match_total']}/{m['tok_total']} "
                    f"({100*m['match_ratio']:.1f}%)"
                )

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
