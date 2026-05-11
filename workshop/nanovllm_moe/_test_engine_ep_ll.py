"""Multi-process EP engine smoke test (EP-LL or EP-HT).

Spawns WORLD_SIZE processes; each constructs a full LLMEngine for Qwen3-30B-A3B
(or trimmed to NUM_LAYERS). The model uses EP modules (Dispatch/Experts/Combine,
EP-LL or EP-HT variant) which communicate via NCCL all_to_all_single.

Run:
  WORLD_SIZE=2 NUM_LAYERS=2 MOE_IMPL=ep_ll_torch    python -m workshop.nanovllm_moe._test_engine_ep_ll
  WORLD_SIZE=8 NUM_LAYERS=2 MOE_IMPL=ep_ll_triton   python -m workshop.nanovllm_moe._test_engine_ep_ll
  WORLD_SIZE=8 NUM_LAYERS=2 MOE_IMPL=ep_ht          python -m workshop.nanovllm_moe._test_engine_ep_ll
  WORLD_SIZE=8 ENFORCE_EAGER=0 MOE_IMPL=ep_ll_triton python -m workshop.nanovllm_moe._test_engine_ep_ll  # cuda graph
"""
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "2"))
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "2"))   # -1 = full model
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "1") == "1"
MOE_IMPL = os.environ.get("MOE_IMPL", "ep_ll_torch")   # ep_ll_torch | ep_ll_triton
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/Qwen3-30B-A3B"))


def worker(rank: int, world_size: int):
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29503")
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{rank}"),
        )
        torch.set_default_device(f"cuda:{rank}")
        if rank == 0:
            print(f"[rank 0] world_size={world_size}, NUM_LAYERS={NUM_LAYERS}, "
                  f"ENFORCE_EAGER={ENFORCE_EAGER}, MOE_IMPL={MOE_IMPL}", flush=True)

        from workshop.nanovllm_moe.services.engine.llm_engine import LLMEngine
        from workshop.nanovllm_moe.services.sampling_params import SamplingParams

        kwargs = dict(
            max_num_batched_tokens=2048,
            max_num_seqs=4,
            max_model_len=512,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1,   # pure EP — no TP within attention
            enforce_eager=ENFORCE_EAGER,
            moe_impl=MOE_IMPL,
        )
        if NUM_LAYERS > 0:
            kwargs["num_hidden_layers_override"] = NUM_LAYERS

        if rank == 0:
            print(f"[rank 0] constructing LLMEngine...", flush=True)
        t0 = time.perf_counter()
        engine = LLMEngine(model=MODEL_PATH, **kwargs)
        if rank == 0:
            print(f"[rank 0] LLMEngine ready in {time.perf_counter() - t0:.1f}s", flush=True)

        sp = SamplingParams(temperature=0.0, max_tokens=8)
        prompts = ["The capital of France is"]

        t0 = time.perf_counter()
        outputs = engine.generate(prompts, sp, use_tqdm=False)
        elapsed = time.perf_counter() - t0

        if rank == 0:
            for i, o in enumerate(outputs):
                tids = o["token_ids"]
                text = o["text"]
                print(f"[rank 0] prompt[{i}] -> {len(tids)} tokens in {elapsed:.2f}s")
                print(f"[rank 0] tokens: {tids}")
                print(f"[rank 0] text: {text!r}")
                # Sanity: with only NUM_LAYERS layers we don't expect coherent text,
                # but we DO expect the pipeline to produce SOME tokens without crashing.
                assert len(tids) > 0, "engine produced 0 tokens"

        # Don't barrier — destroy_process_group sometimes hangs when sub-procs
        # have outstanding NCCL ops. The watchdog warning from PyTorch is benign.
        if rank == 0:
            print(f"[rank 0] PASS", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"[rank {rank}] EXCEPTION: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


def main():
    if not os.path.isdir(MODEL_PATH):
        print(f"SKIP: model path not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(2)

    print(f"Spawning {WORLD_SIZE} ranks for EP-LL engine smoke test", flush=True)
    mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
