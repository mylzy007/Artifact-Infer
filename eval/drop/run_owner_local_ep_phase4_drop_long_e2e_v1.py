"""End-to-end drop benchmark on real long-prompt datasets.

Reads a prepared `.custom.jsonl` (LongBench / lEval / etc.) and runs
LLMEngine.generate twice — once baseline, once with drop — back-to-back
on the same set of prompts. Records prefill / decode / e2e timing.

Key knobs based on Tier 1/2 findings:
- `--moe-drop-min-replicas`: set to ~512 so drop is bypassed during decode
  (L_recv≈8 → drop hurts) but active during prefill (L_recv≥512 → drop helps).
- Prompt-length filter via `--max-prompt-tokens` + `--min-prompt-tokens`
  keeps each rank's source-token count in the sweet spot.

Usage (8-rank):

    CUDA_HOME=/usr/local/cuda-12.8 \\
        PATH=/usr/local/cuda-12.8/bin:$PATH \\
        FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_long \\
        torchrun --nproc_per_node=8 --master_port=29508 \\
        -m eval.drop.long_e2e_bench \\
        --output-dir eval_results/drop_long_e2e \\
        --dataset /home/lzy/datasets/moe_benchmarks/prepared/longbench.2wikimqa.custom.jsonl \\
        --num-samples 48 \\
        --batch-size 8 \\
        --min-prompt-tokens 1024 --max-prompt-tokens 8192 \\
        --max-new-tokens 32 \\
        --drop-rate 0.3 \\
        --moe-drop-min-replicas 512 \\
        --expert-overlap-path eval_results/.../moe_overlap_plan_smoke_plan_20260521_222623.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Drop env vars before any submodule import. drop-min-replicas is set later
# from CLI (cannot use setdefault: we want to override whatever's inherited).
os.environ.setdefault("MOE_DROP_IMPL", "gpu")
os.environ.setdefault("MOE_DROP_GPU_STATS", "0")
os.environ.setdefault("MOE_PROFILE_OVERLAP_RUNTIME", "0")
os.environ.setdefault("MOE_RECORD_TIMING", "1")
os.environ.setdefault("MOE_PROFILE_ROUTING", "0")

import torch
import torch.distributed as dist

from eval.drop.shared import WandbLogger, agg_stats, truncate_file, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-30B-A3B"))
    p.add_argument("--dataset", required=True,
                   help="Path to a prepared .custom.jsonl with {request_id, prompt, output_tokens} per row")
    p.add_argument("--num-samples", type=int, default=48,
                   help="how many prompts to include after length filtering")
    p.add_argument("--batch-size", type=int, default=8,
                   help="prompts submitted together per LLMEngine.generate call")
    p.add_argument("--min-prompt-tokens", type=int, default=512)
    p.add_argument("--max-prompt-tokens", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--drop-policy", default="tail_weight")
    p.add_argument("--drop-rate", type=float, default=0.3)
    p.add_argument("--expert-overlap-path", default="")
    p.add_argument("--expert-overlap-strategy", default="greedy_balance")
    p.add_argument("--moe-drop-min-replicas", type=int, default=512,
                   help="bypass drop when target_rank.numel() <= this (Tier 1 finding: drop hurts below L_recv~3k)")
    p.add_argument("--max-num-batched-tokens", type=int, default=0,
                   help="0 -> auto from max(prompt_len*batch, 8192)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--num-hidden-layers-override", type=int, default=-1)
    p.add_argument("--repeats", type=int, default=1,
                   help="run the whole sweep N times (for noise estimation)")
    p.add_argument("--warmup-batches", type=int, default=2,
                   help="discard timing for this many initial batches (flashinfer JIT warmup, "
                        "owner_local_ep cold-path setup, etc.)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="")
    p.add_argument("--cases", default="baseline,drop",
                   help="comma list of cases to run: subset of {baseline, drop}")
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


# ---------------------------------------------------------------------------
# Dataset loading + length-filtered sampling
# ---------------------------------------------------------------------------

def load_prompts(
    dataset_path: str,
    tokenizer,
    min_tokens: int,
    max_tokens: int,
    num_samples: int,
    seed: int,
) -> list[dict]:
    """Read prompts, tokenize, filter by length, return up to num_samples."""
    rows = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = d.get("prompt") or ""
            if not prompt:
                continue
            rows.append({
                "request_id": d.get("request_id", f"row-{len(rows)}"),
                "prompt": prompt,
                "ref_output_tokens": int(d.get("output_tokens", 32)),
            })
    # Deterministic shuffle for reproducible sampling.
    import random
    rng = random.Random(seed)
    rng.shuffle(rows)

    out = []
    for row in rows:
        ids = tokenizer.encode(row["prompt"], add_special_tokens=True)
        n = len(ids)
        if n < min_tokens or n > max_tokens:
            continue
        row["prompt_token_ids"] = ids
        row["prompt_len"] = n
        out.append(row)
        if len(out) >= num_samples:
            break
    return out


def make_engine(args: argparse.Namespace, world: int, max_prompt_tokens: int):
    from workshop.nanovllm_moe.services.engine.llm_engine import LLMEngine

    max_batched = (
        args.max_num_batched_tokens
        or max(max_prompt_tokens * args.batch_size, 8192)
    )
    kwargs = dict(
        max_num_batched_tokens=max_batched,
        max_num_seqs=max(64, args.batch_size * world),
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
        moe_drop_policy="none",
        moe_drop_rate=0.0,
    )
    if args.num_hidden_layers_override > 0:
        kwargs["num_hidden_layers_override"] = args.num_hidden_layers_override
    return LLMEngine(model=args.model, **kwargs)


def configure_drop(engine, policy: str, rate: float) -> int:
    engine.config.moe_drop_policy = policy
    engine.config.moe_drop_rate = rate
    enabled = policy != "none" and rate > 0.0
    n = 0
    for m in engine.model_runner.model.modules():
        if type(m).__name__ == "DispatchEPHT":
            m.drop_policy = policy
            m.drop_rate = rate
            m.drop_enabled = enabled
            n += 1
    return n


def run_batch(engine, prompt_token_ids_list: list[list[int]], max_new_tokens: int):
    from workshop.nanovllm_moe.services.sampling_params import SamplingParams

    engine.reset()
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    t0 = time.perf_counter()
    _ = engine.generate(prompt_token_ids_list, sp, use_tqdm=False)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    rank, local_rank, world = setup_distributed()

    # Push the bypass threshold into the env BEFORE the engine is constructed —
    # DispatchEPHT reads MOE_DROP_MIN_REPLICAS at __init__ time. Overriding
    # whatever was inherited from the shell.
    os.environ["MOE_DROP_MIN_REPLICAS"] = str(args.moe_drop_min_replicas)

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    assert all(c in ("baseline", "drop") for c in cases), f"bad --cases: {cases}"

    if rank == 0:
        print(f"[long_e2e] world={world}  dataset={os.path.basename(args.dataset)}")
        print(f"[long_e2e] cases={cases}  drop_rate={args.drop_rate} "
              f"min_replicas_bypass={args.moe_drop_min_replicas}")

    # Build engine first to get tokenizer.
    t_load = time.perf_counter()
    # Use a conservative max_prompt_tokens guess for capacity; load_prompts() will
    # then filter to the actual envelope.
    engine = make_engine(args, world, max_prompt_tokens=args.max_prompt_tokens)
    if rank == 0:
        print(f"[long_e2e] engine ready in {time.perf_counter() - t_load:.1f}s")

    tokenizer = engine.tokenizer
    prompts = load_prompts(
        args.dataset, tokenizer,
        min_tokens=args.min_prompt_tokens,
        max_tokens=args.max_prompt_tokens,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    if rank == 0:
        lens = [p["prompt_len"] for p in prompts]
        if lens:
            print(f"[long_e2e] loaded {len(prompts)} prompts after length filter "
                  f"[{min(lens)}, {max(lens)}], p50={sorted(lens)[len(lens)//2]}")
        else:
            print(f"[long_e2e] WARNING: 0 prompts passed length filter "
                  f"[{args.min_prompt_tokens}, {args.max_prompt_tokens}]")
            return 1

    if rank != 0:
        # Other ranks don't iterate prompts list; they go through LLMEngine
        # collective ops via the engine's owner_local_ep dispatch.
        pass

    # Broadcast prompt set from rank 0 so every rank submits identical batches.
    # In owner_local_ep, LLMEngine.generate() expects the same prompts list on
    # every rank.
    obj = [prompts]
    dist.broadcast_object_list(obj, src=0)
    prompts = obj[0]

    jsonl_path = os.path.join(args.output_dir, "long_e2e_rows.jsonl")
    truncate_file(jsonl_path)
    dist.barrier()

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=f"long_e2e_w{world}_seed{args.seed}",
        config=vars(args),
        enabled=bool(args.wandb_project),
    )

    # Split prompts into batches.
    batches = [
        prompts[i:i + args.batch_size]
        for i in range(0, len(prompts), args.batch_size)
    ]

    # ---- JIT warmup: run a few baseline batches, discard timing. flashinfer
    # compiles per-shape kernels lazily; the FIRST few batches pay this cost.
    # We always warm up in 'baseline' mode so the drop case never overlaps
    # JIT compilation with measurement.
    if args.warmup_batches > 0:
        configure_drop(engine, "none", 0.0)
        if rank == 0:
            print(f"\n[warmup] running {args.warmup_batches} baseline batch(es); timing discarded")
        for w in range(args.warmup_batches):
            batch = batches[w % len(batches)]
            tokens_list = [p["prompt_token_ids"] for p in batch]
            wm = run_batch(engine, tokens_list, args.max_new_tokens)
            torch.cuda.synchronize()
            if rank == 0:
                print(f"  warmup batch {w+1}/{args.warmup_batches}  "
                      f"prefill={wm['prefill_time_s']:.3f}s (discarded)")
            dist.barrier()

    for rep in range(args.repeats):
        # Alternate case order across reps so any residual cache asymmetry
        # averages out.
        case_order = cases if rep % 2 == 0 else list(reversed(cases))
        for case in case_order:
            if case == "baseline":
                n_changed = configure_drop(engine, "none", 0.0)
                tag = "baseline"
            else:
                n_changed = configure_drop(engine, args.drop_policy, args.drop_rate)
                tag = f"drop_{args.drop_policy}_r{args.drop_rate}"
            if rank == 0:
                print(f"\n[rep {rep+1}/{args.repeats}] [case={case}] "
                      f"applied to {n_changed} layers")

            for bi, batch in enumerate(batches):
                tokens_list = [p["prompt_token_ids"] for p in batch]
                m = run_batch(engine, tokens_list, args.max_new_tokens)
                torch.cuda.synchronize()
                row = {
                    "rep": rep,
                    "case": case,
                    "case_tag": tag,
                    "batch_id": bi,
                    "batch_size": len(batch),
                    "world_size": world,
                    "prompt_lens": [p["prompt_len"] for p in batch],
                    "total_prompt_tokens": sum(p["prompt_len"] for p in batch),
                    "drop_rate": (args.drop_rate if case == "drop" else 0.0),
                    "min_replicas_bypass": args.moe_drop_min_replicas,
                    **m,
                }
                if rank == 0:
                    write_jsonl(jsonl_path, row)
                    wandb_logger.log(row)
                    pt = m["prefill_time_s"] or 0.0
                    et = m["e2e_total_time_s"] or 0.0
                    print(f"  batch={bi+1}/{len(batches)}  prompt_tok={row['total_prompt_tokens']:6d}  "
                          f"prefill={pt:.3f}s  e2e={et:.3f}s  prefill_tok_s={m['prefill_tok_s'] or 0:.1f}")
                dist.barrier()

    # Summary on rank 0.
    if rank == 0:
        from collections import defaultdict
        rows = [json.loads(line) for line in open(jsonl_path) if line.strip()]
        by_case = defaultdict(list)
        for r in rows:
            by_case[r["case"]].append(r)
        summary = {}
        for case, rs in by_case.items():
            summary[case] = {
                "n_batches": len(rs),
                "total_prompt_tokens_sum": sum(r["total_prompt_tokens"] for r in rs),
                "prefill_time_s": agg_stats(
                    [r["prefill_time_s"] for r in rs if r["prefill_time_s"] is not None]
                ),
                "prefill_tok_s": agg_stats(
                    [r["prefill_tok_s"] for r in rs if r["prefill_tok_s"] is not None]
                ),
                "decode_time_s": agg_stats(
                    [r["decode_time_s"] for r in rs if r["decode_time_s"] is not None]
                ),
                "e2e_total_time_s": agg_stats(
                    [r["e2e_total_time_s"] for r in rs if r["e2e_total_time_s"] is not None]
                ),
            }
        # Aggregate prefill / e2e speedup if both cases present.
        speedup = {}
        if "baseline" in summary and "drop" in summary:
            b_pt = summary["baseline"]["prefill_time_s"]["mean"]
            d_pt = summary["drop"]["prefill_time_s"]["mean"]
            b_e2e = summary["baseline"]["e2e_total_time_s"]["mean"]
            d_e2e = summary["drop"]["e2e_total_time_s"]["mean"]
            speedup = {
                "prefill_time_baseline_s": b_pt,
                "prefill_time_drop_s": d_pt,
                "prefill_speedup": b_pt / d_pt if d_pt > 0 else None,
                "prefill_delta_pct": (d_pt - b_pt) / b_pt if b_pt > 0 else None,
                "e2e_time_baseline_s": b_e2e,
                "e2e_time_drop_s": d_e2e,
                "e2e_speedup": b_e2e / d_e2e if d_e2e > 0 else None,
                "e2e_delta_pct": (d_e2e - b_e2e) / b_e2e if b_e2e > 0 else None,
            }
        out_path = os.path.join(args.output_dir, "long_e2e_summary.json")
        with open(out_path, "w") as f:
            json.dump({
                "config": {
                    "dataset": os.path.basename(args.dataset),
                    "num_samples": args.num_samples,
                    "batch_size": args.batch_size,
                    "min_prompt_tokens": args.min_prompt_tokens,
                    "max_prompt_tokens": args.max_prompt_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "drop_policy": args.drop_policy,
                    "drop_rate": args.drop_rate,
                    "min_replicas_bypass": args.moe_drop_min_replicas,
                },
                "per_case": summary,
                "speedup": speedup,
            }, f, indent=2)
        print(f"\n[long_e2e] wrote {out_path}")
        if speedup:
            print(f"  prefill_speedup={speedup['prefill_speedup']:.4f}x  "
                  f"({speedup['prefill_delta_pct']*100:+.2f}%)")
            print(f"  e2e_speedup    ={speedup['e2e_speedup']:.4f}x  "
                  f"({speedup['e2e_delta_pct']*100:+.2f}%)")

    wandb_logger.finish()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
