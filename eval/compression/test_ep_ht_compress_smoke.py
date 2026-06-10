"""Smoke test: EP-HT combine-side topk_l2 + FP8 compression on a real EP=2 run.

Loads a trimmed Qwen3-30B-A3B (NUM_LAYERS=2 by default, fits one rank) under
EP-HT, greedy-decodes a short prompt twice in the same engine:
  1. with no compression                 (teacher reference)
  2. with MOE_COMBINE_COMPRESS_KEEP_FRAC=0.25 + FP8  (student)

Verifies that the student generation does not crash AND prints the token-by-
token comparison so we can eyeball drift. With keep_frac=0.25 + FP8 the HF
simulation gave ~+17% PPL on lcc, so we EXPECT the greedy decoded text to
mostly match, especially on early tokens.

Run:
  WORLD_SIZE=2 NUM_LAYERS=2 MOE_IMPL=ep_ht ENFORCE_EAGER=1 \
    /home/lzy/miniconda3/envs/atom/bin/python -m eval.compression.test_ep_ht_compress_smoke
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "2"))
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "2"))
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "1") == "1"
MOE_IMPL = os.environ.get("MOE_IMPL", "ep_ht")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/Qwen3-30B-A3B"))
KEEP_FRAC = os.environ.get("KEEP_FRAC", "0.25")
USE_FP8 = os.environ.get("USE_FP8", "1")


def worker(rank: int, world_size: int):
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29611")
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
                f"KEEP_FRAC={KEEP_FRAC} USE_FP8={USE_FP8}",
                flush=True,
            )

        from workshop.nanovllm_moe.services.engine.llm_engine import LLMEngine
        from workshop.nanovllm_moe.services.sampling_params import SamplingParams

        kwargs = dict(
            max_num_batched_tokens=2048,
            max_num_seqs=4,
            max_model_len=256,
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

        sp = SamplingParams(temperature=0.0, max_tokens=16)
        prompts = ["The capital of France is", "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot ="]

        # --- Pass 1: no compression (teacher) ---
        os.environ.pop("MOE_COMBINE_COMPRESS_KEEP_FRAC", None)
        os.environ.pop("MOE_COMBINE_COMPRESS_FP8", None)
        dist.barrier()
        t0 = time.perf_counter()
        out_teacher = engine.generate(prompts, sp, use_tqdm=False)
        t_teacher = time.perf_counter() - t0
        if rank == 0:
            print(f"\n[rank 0] === teacher (no compression) in {t_teacher:.2f}s ===", flush=True)
            for i, o in enumerate(out_teacher):
                print(f"[rank 0] prompt[{i}] tokens: {o['token_ids']}")
                print(f"[rank 0] prompt[{i}] text:   {o['text']!r}")

        # --- Pass 2: compressed (student) ---
        os.environ["MOE_COMBINE_COMPRESS_KEEP_FRAC"] = KEEP_FRAC
        os.environ["MOE_COMBINE_COMPRESS_FP8"] = USE_FP8
        dist.barrier()
        t0 = time.perf_counter()
        out_student = engine.generate(prompts, sp, use_tqdm=False)
        t_student = time.perf_counter() - t0
        if rank == 0:
            print(
                f"\n[rank 0] === student (keep_frac={KEEP_FRAC} fp8={USE_FP8}) "
                f"in {t_student:.2f}s ===", flush=True,
            )
            for i, o in enumerate(out_student):
                print(f"[rank 0] prompt[{i}] tokens: {o['token_ids']}")
                print(f"[rank 0] prompt[{i}] text:   {o['text']!r}")
            print(f"\n[rank 0] teacher_time={t_teacher:.2f}s  student_time={t_student:.2f}s")
            # Compute per-prompt match ratio.
            for i, (te, st) in enumerate(zip(out_teacher, out_student)):
                t_ids = te["token_ids"]
                s_ids = st["token_ids"]
                n = min(len(t_ids), len(s_ids))
                match = sum(1 for j in range(n) if t_ids[j] == s_ids[j])
                print(
                    f"[rank 0] prompt[{i}]: matched {match}/{n} tokens "
                    f"({100*match/max(n,1):.1f}%)"
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
