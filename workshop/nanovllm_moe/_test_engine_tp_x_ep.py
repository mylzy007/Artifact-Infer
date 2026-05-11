"""Multi-process TP × EP composition smoke test.

Spawns WORLD_SIZE = TP_SIZE * EP_SIZE processes. Each constructs an LLMEngine
with `tensor_parallel_size=TP_SIZE` (the engine derives ep_size internally as
world_size / tp_size).

Layout convention (matches services/utils/parallel.py):
  rank = ep_rank * TP_SIZE + tp_rank

What this exercises that pure-EP does not:
  - Attention/qkv_proj/o_proj split across the TP subgroup (RowParallelLinear
    does an all-reduce inside the TP group).
  - Embedding/lm_head shard the vocab across the TP subgroup.
  - Each EP slot (TP subgroup) holds 1/EP_SIZE of the experts; the EP a2a runs
    within the EP subgroup (size EP_SIZE).
  - tp_rank==0 of each EP slot samples; result broadcast within TP subgroup.

Run:
  TP_SIZE=2 EP_SIZE=4 NUM_LAYERS=2 MOE_IMPL=ep_ll_triton ENFORCE_EAGER=0 \\
      python -m workshop.nanovllm_moe._test_engine_tp_x_ep
  TP_SIZE=4 EP_SIZE=2 NUM_LAYERS=2 MOE_IMPL=ep_ht        ENFORCE_EAGER=1 \\
      python -m workshop.nanovllm_moe._test_engine_tp_x_ep
"""
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


TP_SIZE = int(os.environ.get("TP_SIZE", "2"))
EP_SIZE = int(os.environ.get("EP_SIZE", "4"))
WORLD_SIZE = TP_SIZE * EP_SIZE
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "2"))
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "1") == "1"
MOE_IMPL = os.environ.get("MOE_IMPL", "ep_ll_triton")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/Qwen3-30B-A3B"))


def worker(rank: int, world_size: int):
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29504")
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{rank}"),
        )
        torch.set_default_device(f"cuda:{rank}")
        ep_rank = rank // TP_SIZE
        tp_rank = rank % TP_SIZE
        if rank == 0:
            print(f"[rank 0] world={world_size} TP={TP_SIZE} EP={EP_SIZE} "
                  f"NUM_LAYERS={NUM_LAYERS} EAGER={ENFORCE_EAGER} MOE_IMPL={MOE_IMPL}",
                  flush=True)

        from workshop.nanovllm_moe.services.engine.llm_engine import LLMEngine
        from workshop.nanovllm_moe.services.sampling_params import SamplingParams

        kwargs = dict(
            max_num_batched_tokens=2048,
            max_num_seqs=4,
            max_model_len=512,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=TP_SIZE,
            enforce_eager=ENFORCE_EAGER,
            moe_impl=MOE_IMPL,
        )
        if NUM_LAYERS > 0:
            kwargs["num_hidden_layers_override"] = NUM_LAYERS

        if rank == 0:
            print(f"[rank 0] constructing LLMEngine (tp={TP_SIZE}, ep={EP_SIZE})...",
                  flush=True)
        t0 = time.perf_counter()
        engine = LLMEngine(model=MODEL_PATH, **kwargs)
        if rank == 0:
            print(f"[rank 0] LLMEngine ready in {time.perf_counter() - t0:.1f}s",
                  flush=True)

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
                assert len(tids) > 0, "engine produced 0 tokens"
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

    print(f"Spawning {WORLD_SIZE} ranks (TP={TP_SIZE}, EP={EP_SIZE})", flush=True)
    mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
