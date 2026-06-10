"""Verify Config-time compression knob (no env var needed).

Two ways to enable combine-side compression after this commit:

(A) At engine construction (Config-time, the recommended way for benchmarks):
       engine = LLMEngine(
           model=...,
           moe_impl="ep_ht",
           moe_combine_compress_keep_frac=0.25,
           moe_combine_compress_fp8=True,
           moe_combine_compress_use_l2=True,
       )
       # ALL forwards through this engine are compressed.

(B) At runtime via env var (overrides whatever Config said, useful for quick
    A/B inside one engine instance):
       os.environ["MOE_COMBINE_COMPRESS_KEEP_FRAC"] = "0.25"
       # subsequent forwards see compression; unset to disable.

This smoke test exercises path (A): builds two engines back-to-back, one with
keep_frac=0.0 (baseline) and one with keep_frac=0.25 + FP8 + L2 rescale, runs
the same prompts under each, and prints whatever they generate.

Run (small smoke, 2 GPUs, trimmed model):
  WORLD_SIZE=2 NUM_LAYERS=2 \\
    /home/lzy/miniconda3/envs/vllm/bin/python -m eval.compression.test_ep_ht_compress_config
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
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/Qwen3-30B-A3B"))
KEEP_FRAC = float(os.environ.get("KEEP_FRAC", "0.25"))


def _engine_kwargs(keep_frac: float) -> dict:
    kw = dict(
        model=MODEL_PATH,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        max_model_len=256,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        enforce_eager=True,
        moe_impl="ep_ht",
        moe_combine_compress_keep_frac=float(keep_frac),
        moe_combine_compress_fp8=True,
        moe_combine_compress_use_l2=True,
    )
    if NUM_LAYERS > 0:
        kw["num_hidden_layers_override"] = NUM_LAYERS
    return kw


def worker(rank: int, world_size: int):
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29617")
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{rank}"),
        )
        torch.set_default_device(f"cuda:{rank}")

        # Clean any leftover env-var overrides so Config defaults take effect.
        for key in ("MOE_COMBINE_COMPRESS_KEEP_FRAC", "MOE_COMBINE_COMPRESS_FP8", "MOE_COMBINE_COMPRESS_NO_L2"):
            os.environ.pop(key, None)

        from workshop.nanovllm_moe.services.engine.llm_engine import LLMEngine
        from workshop.nanovllm_moe.services.sampling_params import SamplingParams

        prompts = ["The capital of France is", "1 + 1 ="]
        sp = SamplingParams(temperature=0.0, max_tokens=8)

        # --- Engine A: baseline (keep_frac=0 from Config default, no compression) ---
        if rank == 0:
            print("[rank 0] === engine A (Config moe_combine_compress_keep_frac=0.0) ===", flush=True)
        engine_a = LLMEngine(**_engine_kwargs(keep_frac=0.0))
        dist.barrier()
        out_a = engine_a.generate(prompts, sp, use_tqdm=False)
        if rank == 0:
            for i, o in enumerate(out_a):
                print(f"[rank 0] A p[{i}] text: {o['text']!r}")
        del engine_a
        torch.cuda.empty_cache()
        dist.barrier()

        # --- Engine B: with Config-time compression ---
        if rank == 0:
            print(
                f"\n[rank 0] === engine B (Config moe_combine_compress_keep_frac={KEEP_FRAC}) ===",
                flush=True,
            )
        engine_b = LLMEngine(**_engine_kwargs(keep_frac=KEEP_FRAC))
        dist.barrier()
        out_b = engine_b.generate(prompts, sp, use_tqdm=False)
        if rank == 0:
            for i, o in enumerate(out_b):
                print(f"[rank 0] B p[{i}] text: {o['text']!r}")

        if rank == 0:
            print("\n[rank 0] === PASS if both engines ran without crash ===")

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
