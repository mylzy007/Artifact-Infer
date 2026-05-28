"""Pure-EP stochastic sampling consistency smoke test.

This checks that all EP ranks update sequence state with the same sampled token
ids when temperature > 0. It is intentionally small and mirrors
_test_engine_ep_ll.py without changing that broader smoke test.
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
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "1"))
MOE_IMPL = os.environ.get("MOE_IMPL", "ep_ll_triton")
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "1") == "1"
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/Qwen3-30B-A3B"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "12"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.6"))


def worker(rank: int, world_size: int):
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29631")
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{rank}"),
        )
        torch.set_default_device(f"cuda:{rank}")

        from workshop.nanovllm_moe.services.engine.llm_engine import LLMEngine
        from workshop.nanovllm_moe.services.sampling_params import SamplingParams

        kwargs = dict(
            max_num_batched_tokens=512,
            max_num_seqs=1,
            max_model_len=512,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1,
            enforce_eager=ENFORCE_EAGER,
            moe_impl=MOE_IMPL,
        )
        if NUM_LAYERS > 0:
            kwargs["num_hidden_layers_override"] = NUM_LAYERS

        if rank == 0:
            print(
                f"[rank 0] stochastic consistency: world={world_size}, "
                f"layers={NUM_LAYERS}, impl={MOE_IMPL}, "
                f"temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}",
                flush=True,
            )
        engine = LLMEngine(model=MODEL_PATH, **kwargs)
        sp = SamplingParams(temperature=TEMPERATURE, top_k=20, top_p=0.95, max_tokens=MAX_TOKENS)
        t0 = time.perf_counter()
        outputs = engine.generate(["What is 2+2?"], sp, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        token_ids = outputs[0]["token_ids"]
        print(f"[rank {rank}] token_ids={token_ids} elapsed={elapsed:.2f}s", flush=True)

        gathered = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(token_ids, gathered, dst=0)
        if rank == 0:
            assert all(ids == gathered[0] for ids in gathered), (
                f"EP ranks diverged: {gathered}"
            )
            print(f"[rank 0] PASS: all ranks received identical token_ids {gathered[0]}", flush=True)
        sys.exit(0)
    except Exception as exc:
        print(f"[rank {rank}] EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)


def main():
    if not os.path.isdir(MODEL_PATH):
        print(f"SKIP: model path not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(2)
    mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
