"""Tier 2 microbenchmark for the Prefill L Sweep.

Verifies whether Tier-1 MoE-block speedup survives in a full-model prefill
once attention / sampling / scheduler are added. Loads Qwen3-30B-A3B via
LLMEngine and submits synthetic prompts (token ids repeated) at one or more
target T_local values; runs baseline (drop_rate=0) and drop (default 0.3)
back-to-back, captures engine metrics.

Usage (8-rank):

    torchrun --nproc_per_node=8 --master_port=29504 \\
        -m eval.drop.tier2_bench \\
        --output-dir eval_results/prefill_drop_l_sweep_tier2 \\
        --model /home/lzy/models/Qwen3-30B-A3B \\
        --t-local-values 512,2048 \\
        --drop-rate 0.3 \\
        --expert-overlap-path eval_results/.../moe_overlap_plan_smoke...json \\
        --repeats 3 --warmup 1

Constraints:
    target_total_tokens = T_local * world_size
    max_num_batched_tokens must be >= max(T_local)
    max_model_len must be >= max(T_local)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Drop env vars before any submodule import.
os.environ.setdefault("MOE_DROP_IMPL", "gpu")
os.environ.setdefault("MOE_DROP_MIN_REPLICAS", "0")
os.environ.setdefault("MOE_DROP_GPU_STATS", "0")
os.environ.setdefault("MOE_PROFILE_OVERLAP_RUNTIME", "0")
os.environ.setdefault("MOE_RECORD_TIMING", "1")  # engine needs perf_counter timing
os.environ.setdefault("MOE_PROFILE_ROUTING", "0")

import torch
import torch.distributed as dist

from eval.drop.shared import (
    WandbLogger,
    agg_stats,
    truncate_file,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-30B-A3B"))
    p.add_argument("--t-local-values", default="512,2048")
    p.add_argument("--drop-rate", type=float, default=0.3)
    p.add_argument("--drop-policy", default="tail_weight")
    p.add_argument("--expert-overlap-path", default="")
    p.add_argument("--expert-overlap-strategy", default="greedy_balance")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-num-batched-tokens", type=int, default=0,
                   help="0 -> auto from max(T_local)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--num-hidden-layers-override", type=int, default=-1,
                   help="-1 -> full model")
    p.add_argument("--wandb-project", default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    torch.set_default_device(f"cuda:{local_rank}")
    return rank, local_rank, world


def build_engine(args: argparse.Namespace, world: int, max_T_local: int):
    """Build LLMEngine. Distributed groups are initialized by the engine itself."""
    from workshop.nanovllm_moe.services.engine.llm_engine import LLMEngine

    max_batched = args.max_num_batched_tokens or max(max_T_local, 4096)
    kwargs = dict(
        max_num_batched_tokens=max_batched,
        max_num_seqs=64,
        max_model_len=max_batched,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        data_parallel_size=world,
        enforce_eager=True,
        moe_impl="ep_ht",
        moe_runtime_mode="owner_local_ep",
        moe_expert_overlap_enabled=bool(args.expert_overlap_path),
        moe_expert_overlap_path=args.expert_overlap_path or None,
        moe_expert_overlap_strategy=args.expert_overlap_strategy,
        moe_drop_policy="none",  # start as baseline; toggle per-cell
        moe_drop_rate=0.0,
    )
    if args.num_hidden_layers_override > 0:
        kwargs["num_hidden_layers_override"] = args.num_hidden_layers_override
    return LLMEngine(model=args.model, **kwargs)


def configure_drop(engine, policy: str, rate: float) -> None:
    """Reach into every DispatchEPHT layer and update its drop config."""
    engine.config.moe_drop_policy = policy
    engine.config.moe_drop_rate = rate
    enabled = policy != "none" and rate > 0.0
    model = engine.model_runner.model
    n_changed = 0
    for module in model.modules():
        if type(module).__name__ == "DispatchEPHT":
            module.drop_policy = policy
            module.drop_rate = rate
            module.drop_enabled = enabled
            n_changed += 1
    return n_changed


def make_prompts(T_local: int, world: int, tokenizer) -> list[list[int]]:
    """One prompt per rank, each of length T_local tokens (single token id repeated).

    In owner_local_ep the engine assigns prompt[i] to rank i%world; with `world`
    prompts each rank gets exactly one, so each rank processes T_local tokens.
    """
    # Pick a token that's safe (single-token, no special handling).
    base = tokenizer.encode(" the", add_special_tokens=False)
    if not base:
        base = [50256]  # arbitrary fallback
    base_id = base[0]
    return [[base_id] * T_local for _ in range(world)]


def run_cell(engine, prompts: list[list[int]], max_new_tokens: int = 1) -> dict:
    """Drive engine.generate() once and return its captured metrics."""
    from workshop.nanovllm_moe.services.sampling_params import SamplingParams

    engine.reset()
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    t0 = time.perf_counter()
    _outputs = engine.generate(prompts, sp, use_tqdm=False)
    wall_s = time.perf_counter() - t0
    metrics = getattr(engine, "last_generation_metrics", {}) or {}
    return {
        "wall_s": wall_s,
        "prefill_tokens": metrics.get("prefill_tokens"),
        "prefill_time_s": metrics.get("prefill_time_s"),
        "prefill_tok_s": metrics.get("prefill_throughput_tok_s"),
        "decode_tokens": metrics.get("decode_tokens"),
        "decode_time_s": metrics.get("decode_time_s"),
        "decode_tok_s": metrics.get("decode_throughput_tok_s"),
        "e2e_total_time_s": metrics.get("e2e_total_time_s"),
    }


def main() -> int:
    args = parse_args()
    rank, local_rank, world = setup_distributed()

    T_locals = sorted({int(x) for x in args.t_local_values.split(",")})
    max_T = max(T_locals)

    if rank == 0:
        print(f"[tier2] world={world} T_locals={T_locals} drop_rate={args.drop_rate} "
              f"repeats={args.repeats} warmup={args.warmup}")
        print(f"[tier2] loading engine: {args.model}")
    t_load = time.perf_counter()
    engine = build_engine(args, world, max_T)
    if rank == 0:
        print(f"[tier2] engine ready in {time.perf_counter() - t_load:.1f}s")

    tokenizer = engine.tokenizer
    jsonl_path = os.path.join(args.output_dir, "tier2_rows.jsonl")
    truncate_file(jsonl_path)
    dist.barrier()

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=f"tier2_w{world}_seed{args.seed}",
        config=vars(args),
        enabled=bool(args.wandb_project),
    )

    cells = []
    for T_local in T_locals:
        for rate in (0.0, args.drop_rate):
            cells.append((T_local, rate))

    for ci, (T_local, rate) in enumerate(cells):
        if rank == 0:
            print(f"[cell {ci+1}/{len(cells)}] T_local={T_local} drop_rate={rate}")
        policy = args.drop_policy if rate > 0 else "none"
        n_layers_changed = configure_drop(engine, policy, rate)
        if rank == 0:
            print(f"  applied drop policy={policy} rate={rate} to {n_layers_changed} layers")
        prompts = make_prompts(T_local, world, tokenizer)

        # Warmup.
        for w in range(args.warmup):
            _ = run_cell(engine, prompts)
        torch.cuda.synchronize()
        dist.barrier()

        for rep in range(args.repeats):
            m = run_cell(engine, prompts)
            torch.cuda.synchronize()
            row = {
                "cell_id": ci,
                "T_local": T_local,
                "drop_policy": policy,
                "drop_rate": rate,
                "repeat_id": rep,
                "world_size": world,
                "target_total_tokens": T_local * world,
                **m,
            }
            if rank == 0:
                write_jsonl(jsonl_path, row)
                wandb_logger.log(row)
                print(f"    rep={rep} prefill_time={row['prefill_time_s']:.3f}s "
                      f"prefill_tok_s={row['prefill_tok_s']:.1f} e2e={row['e2e_total_time_s']:.3f}s")
            dist.barrier()

    # Aggregate per (T_local, rate); write summary.
    if rank == 0:
        from collections import defaultdict
        rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
        groups = defaultdict(list)
        for r in rows:
            groups[(r["T_local"], r["drop_rate"])].append(r)
        summary_groups = {}
        for (T, rate), rs in groups.items():
            summary_groups[f"T{T}_r{rate}"] = {
                "T_local": T,
                "drop_rate": rate,
                "n": len(rs),
                "prefill_time_s": agg_stats(
                    [r["prefill_time_s"] for r in rs if r["prefill_time_s"] is not None]
                ),
                "prefill_tok_s": agg_stats(
                    [r["prefill_tok_s"] for r in rs if r["prefill_tok_s"] is not None]
                ),
                "e2e_total_time_s": agg_stats(
                    [r["e2e_total_time_s"] for r in rs if r["e2e_total_time_s"] is not None]
                ),
            }
        # speedups per T
        speedups = []
        for T in T_locals:
            base = summary_groups.get(f"T{T}_r0.0")
            drop = summary_groups.get(f"T{T}_r{args.drop_rate}")
            if base is None or drop is None:
                continue
            base_pt = base["prefill_time_s"]["mean"]
            drop_pt = drop["prefill_time_s"]["mean"]
            base_e2e = base["e2e_total_time_s"]["mean"]
            drop_e2e = drop["e2e_total_time_s"]["mean"]
            speedups.append({
                "T_local": T,
                "target_total_tokens": T * world,
                "baseline_prefill_s": base_pt,
                "drop_prefill_s": drop_pt,
                "prefill_speedup": base_pt / drop_pt if drop_pt > 0 else None,
                "prefill_delta_pct": (drop_pt - base_pt) / base_pt if base_pt > 0 else None,
                "baseline_e2e_s": base_e2e,
                "drop_e2e_s": drop_e2e,
                "e2e_speedup": base_e2e / drop_e2e if drop_e2e > 0 else None,
            })
        summary = {
            "world_size": world,
            "drop_policy": args.drop_policy,
            "drop_rate": args.drop_rate,
            "speedups": speedups,
            "groups": summary_groups,
        }
        out_path = os.path.join(args.output_dir, "tier2_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[tier2] wrote {out_path}")
        for s in speedups:
            print(f"  T_local={s['T_local']:5d}  base_prefill={s['baseline_prefill_s']:.3f}s "
                  f"drop_prefill={s['drop_prefill_s']:.3f}s  speedup={s['prefill_speedup']:.4f}x "
                  f"({s['prefill_delta_pct']*100:+.2f}%)")

    wandb_logger.finish()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
