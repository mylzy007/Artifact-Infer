"""Bazaar entry for the MoE engine (`workshop/nanovllm_moe`).

Two modes:

  1. **Single-process** — mirrors `bazaar/nanovllm_v2_5.py`:

         from bazaar.nanovllm_moe import combine
         engine, SamplingParams = combine(model="/path/to/Qwen3-30B-A3B",
                                          moe_impl="triton",  # single-rank
                                          enforce_eager=False)
         outputs = engine.generate(prompts, SamplingParams(...))

     Works for `moe_impl in {"torch", "triton"}` (single-rank). For Qwen3-30B-A3B
     this requires either a big GPU or `num_hidden_layers_override=N` for a
     trimmed smoke test.

  2. **Multi-rank SPMD** — for EP-LL / EP-HT / TP × EP:

         from bazaar.nanovllm_moe import spawn_eval

         def my_eval(engine, SP, rank, world_size):
             # called inside each worker process
             outputs = engine.generate(prompts, SP(...))
             if rank == 0:
                 # write results
                 ...

         spawn_eval(
             world_size=8,
             engine_kwargs=dict(model="/path/to/Qwen3-30B-A3B",
                                tensor_parallel_size=1,    # pure EP
                                moe_impl="ep_ll_triton",
                                enforce_eager=False),
             eval_fn=my_eval,
         )

The MoE `LLMEngine` already orchestrates Dispatch/Experts/Combine, MoeBackend,
ModelRunner, BlockManager, Scheduler internally via `RegistryOrchestrator` —
the bazaar wrapper is intentionally thin.
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Callable

# Make sure the repo root is on sys.path so `workshop.nanovllm_moe...` resolves
# whether the script is run from `eval/` or anywhere else.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workshop.nanovllm_moe.services.engine.llm_engine import LLMEngine
from workshop.nanovllm_moe.services.sampling_params import SamplingParams


def combine(**kwargs):
    """Construct a single-process `LLMEngine` and return `(engine, SamplingParams)`.

    Pass-through to `LLMEngine(model=..., **kwargs)`. All MoE-relevant config
    fields (see `workshop/nanovllm_moe/services/config.py`):

      moe_impl: "torch" | "triton" | "ep_ll_torch" | "ep_ll_triton" | "ep_ht"
      moe_block_size_m: int = 64
      moe_ll_m_max: int = -1   (auto-size)
      tensor_parallel_size: int = 1
      enforce_eager: bool = True
      num_hidden_layers_override: int = -1   (trim for testing big models)
      max_num_batched_tokens, max_num_seqs, max_model_len, gpu_memory_utilization

    Constraints (enforced by MoeBackend):
      - moe_impl="torch" / "ep_ll_torch" / "ep_ht"  → enforce_eager=True only
      - moe_impl="ep_ll_*" / "ep_ht"                → world_size > 1, use spawn_eval
    """
    engine = LLMEngine(**kwargs)
    return engine, SamplingParams


def _spmd_worker(
    rank: int,
    world_size: int,
    eval_fn: Callable,
    eval_fn_kwargs: dict,
    engine_kwargs: dict,
    master_addr: str,
    master_port: str,
):
    """Each spawn'd process runs this. `eval_fn` MUST be a top-level (importable)
    function so it can be pickled by `mp.spawn` — closures will not work."""
    try:
        import torch
        import torch.distributed as dist

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{rank}"),
        )
        torch.set_default_device(f"cuda:{rank}")

        engine, SP = combine(**engine_kwargs)
        eval_fn(engine, SP, rank, world_size, **eval_fn_kwargs)

        # Don't call dist.destroy_process_group() here — it occasionally hangs
        # when sub-procs have outstanding NCCL ops. The PyTorch shutdown warning
        # is benign.
        sys.exit(0)

    except Exception as e:
        print(f"[rank {rank}] EXCEPTION: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


def spawn_eval(
    world_size: int,
    engine_kwargs: dict,
    eval_fn: Callable,
    eval_fn_kwargs: dict | None = None,
    *,
    master_addr: str = "127.0.0.1",
    master_port: str = "29555",
):
    """Spawn `world_size` SPMD workers; each constructs an `LLMEngine` via
    `combine(**engine_kwargs)` and then calls
    `eval_fn(engine, SamplingParams, rank, world_size, **eval_fn_kwargs)`.

    `engine_kwargs` typically includes:
      - `tensor_parallel_size` (TP within each EP slot; 1 for pure EP)
      - `moe_impl` ("ep_ll_triton" recommended for cuda-graph EP)
      - `enforce_eager` (False = capture, only valid for "ep_ll_triton")

    World layout (matches `services/utils/parallel.py`):
      world_size = tensor_parallel_size * ep_size,
      rank = ep_rank * tensor_parallel_size + tp_rank.

    `eval_fn` is responsible for any rank-conditional output (e.g. only rank 0
    writes results to disk). All workers must call `engine.generate(...)`
    collectively — the engine is SPMD.

    IMPORTANT: `eval_fn` and every value in `eval_fn_kwargs` must be picklable
    (module-level functions, primitives, dicts, lists). Local closures will not
    survive `mp.spawn`.
    """
    import torch.multiprocessing as mp

    if eval_fn_kwargs is None:
        eval_fn_kwargs = {}

    print(
        f"[bazaar.nanovllm_moe] spawning {world_size} workers "
        f"(tp_size={engine_kwargs.get('tensor_parallel_size', 1)}, "
        f"moe_impl={engine_kwargs.get('moe_impl', 'triton')!r}, "
        f"enforce_eager={engine_kwargs.get('enforce_eager', True)})",
        flush=True,
    )
    mp.spawn(
        _spmd_worker,
        args=(world_size, eval_fn, eval_fn_kwargs, engine_kwargs, master_addr, master_port),
        nprocs=world_size,
        join=True,
    )
