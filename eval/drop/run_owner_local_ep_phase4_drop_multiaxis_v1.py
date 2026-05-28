"""Multi-axis drop sweep on long-prompt benchmarks.

Designed to live inside a single LLMEngine load (model + KV cache loaded once,
each cell mutates only `dispatch.drop_policy / drop_rate / drop_min_replicas`).

Three experimental phases:

  Phase A — drop_policy x drop_rate matrix (at fixed prompt-length tier).
  Phase B — drop_min_replicas bypass sensitivity (at policy=tail_weight, rate=0.3).
  Phase C — prompt-length scaling (3 tiers x {baseline, drop@0.3}).

Each cell records {prefill_time, e2e_time, prefill_tok_s, ...} across N batches;
all rows go into one JSONL. The summary.json aggregates per cell and computes
speedup vs the matching baseline.

Usage:

    CUDA_HOME=/usr/local/cuda-12.8 \\
        PATH=/usr/local/cuda-12.8/bin:$PATH \\
        FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_longbench \\
        /home/lzy/miniconda3/envs/vllm/bin/torchrun --nproc_per_node=8 \\
        --master_port=29511 -m eval.drop.longbench_sweep \\
        --output-dir eval_results/owner_local_ep_phase4_drop_longbench \\
        --dataset /home/lzy/datasets/moe_benchmarks/prepared/leval.Generation_multidoc_qa.custom.jsonl \\
        --expert-overlap-path eval_results/.../moe_overlap_plan_smoke_plan_20260521_222623.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("MOE_DROP_IMPL", "gpu")
os.environ.setdefault("MOE_DROP_GPU_STATS", "0")
os.environ.setdefault("MOE_PROFILE_OVERLAP_RUNTIME", "0")
os.environ.setdefault("MOE_RECORD_TIMING", "1")
os.environ.setdefault("MOE_PROFILE_ROUTING", "0")

import torch
import torch.distributed as dist

from eval.drop.shared import WandbLogger, agg_stats, truncate_file, write_jsonl


# Length tiers (token range, applied via tokenizer.encode length).
# Upper bound on `long` is set to 6144 to stay under max_num_batched_tokens
# on 24GB cards.
LENGTH_TIERS = {
    "short":  (1024, 2048),
    "medium": (2048, 4096),
    "long":   (4096, 6144),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-30B-A3B"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--num-samples-per-tier", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--expert-overlap-path", default="")
    p.add_argument("--expert-overlap-strategy", default="greedy_balance")
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-batches", type=int, default=3)
    p.add_argument("--wandb-project", default="")
    # Sweep parameters (with defaults that match the design doc).
    p.add_argument("--drop-rates", default="0.0,0.1,0.2,0.3,0.5")
    p.add_argument("--drop-policies", default="tail_weight,random,cross_numa_first")
    p.add_argument("--bypass-list", default="0,128,512,2048",
                   help="Phase B: min_replicas values tested at policy=tail_weight rate=0.3")
    p.add_argument("--length-tiers", default="short,medium,long")
    p.add_argument("--phase-c-policy", default="tail_weight")
    p.add_argument("--phase-c-rate", type=float, default=0.3)
    p.add_argument("--phase-c-min-replicas", type=int, default=512)
    # Phase A baseline tier
    p.add_argument("--phase-ab-tier", default="medium")
    p.add_argument("--phase-ab-min-replicas", type=int, default=512)
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


def make_engine(args: argparse.Namespace, world: int):
    from workshop.nanovllm_moe.services.engine.llm_engine import LLMEngine
    return LLMEngine(
        model=args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=max(64, args.batch_size * world),
        max_model_len=args.max_num_batched_tokens,
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


def configure_drop(engine, policy: str, rate: float, min_replicas: int) -> int:
    """Mutate every DispatchEPHT layer's drop config + bypass threshold."""
    engine.config.moe_drop_policy = policy
    engine.config.moe_drop_rate = rate
    enabled = policy != "none" and rate > 0.0
    n = 0
    for m in engine.model_runner.model.modules():
        if type(m).__name__ == "DispatchEPHT":
            m.drop_policy = policy
            m.drop_rate = rate
            m.drop_enabled = enabled
            m.drop_min_replicas = int(min_replicas)
            n += 1
    return n


def load_prompts_at_tiers(
    dataset_path: str,
    tokenizer,
    tiers: list[str],
    num_per_tier: int,
    seed: int,
) -> dict[str, list[dict]]:
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
            })

    # Tokenize once, then bucket by length tier.
    tokenized: list[tuple[int, dict]] = []
    for r in rows:
        ids = tokenizer.encode(r["prompt"], add_special_tokens=True)
        r["prompt_token_ids"] = ids
        r["prompt_len"] = len(ids)
        tokenized.append((len(ids), r))

    rng = random.Random(seed)
    rng.shuffle(tokenized)

    out: dict[str, list[dict]] = {}
    for tier in tiers:
        lo, hi = LENGTH_TIERS[tier]
        bucket = [r for L, r in tokenized if lo <= L <= hi]
        out[tier] = bucket[:num_per_tier]
    return out


def run_batch(engine, prompt_token_ids: list[list[int]], max_new_tokens: int) -> dict:
    from workshop.nanovllm_moe.services.sampling_params import SamplingParams
    engine.reset()
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    t0 = time.perf_counter()
    _ = engine.generate(prompt_token_ids, sp, use_tqdm=False)
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


def build_cell_plan(args: argparse.Namespace) -> list[dict]:
    """Build the list of cells (dicts) we will run.

    Phase A: drop_policy x drop_rate matrix at medium tier, min_replicas=512.
             rate=0.0 baseline shared across policies (one cell with policy='none').
    Phase B: bypass min_replicas sensitivity at policy=tail_weight, rate=0.3, medium.
    Phase C: length tier x {baseline, drop@0.3} cross.
    """
    rates = sorted({float(x) for x in args.drop_rates.split(",")})
    policies = [s.strip() for s in args.drop_policies.split(",") if s.strip()]
    bypass_list = [int(x) for x in args.bypass_list.split(",")]
    tiers = [t.strip() for t in args.length_tiers.split(",") if t.strip()]

    cells = []
    # --- Phase A ---
    # Single baseline at medium tier with phase-ab min_replicas.
    cells.append({
        "phase": "A",
        "label": f"A_baseline",
        "policy": "none",
        "rate": 0.0,
        "min_replicas": args.phase_ab_min_replicas,
        "tier": args.phase_ab_tier,
    })
    for policy in policies:
        for rate in rates:
            if rate == 0.0:
                continue  # baseline handled above
            cells.append({
                "phase": "A",
                "label": f"A_{policy}_r{rate}",
                "policy": policy,
                "rate": float(rate),
                "min_replicas": args.phase_ab_min_replicas,
                "tier": args.phase_ab_tier,
            })

    # --- Phase B --- bypass sensitivity (tail_weight, rate=0.3)
    for mr in bypass_list:
        if mr == args.phase_ab_min_replicas:
            continue  # already in Phase A as tail_weight@0.3
        cells.append({
            "phase": "B",
            "label": f"B_bypass{mr}",
            "policy": "tail_weight",
            "rate": 0.3,
            "min_replicas": mr,
            "tier": args.phase_ab_tier,
        })

    # --- Phase C --- length-tier x {baseline, drop@0.3}
    for tier in tiers:
        cells.append({
            "phase": "C",
            "label": f"C_{tier}_baseline",
            "policy": "none",
            "rate": 0.0,
            "min_replicas": args.phase_c_min_replicas,
            "tier": tier,
        })
        cells.append({
            "phase": "C",
            "label": f"C_{tier}_drop",
            "policy": args.phase_c_policy,
            "rate": float(args.phase_c_rate),
            "min_replicas": args.phase_c_min_replicas,
            "tier": tier,
        })

    return cells


def main() -> int:
    args = parse_args()
    rank, local_rank, world = setup_distributed()

    if rank == 0:
        print(f"[longbench_sweep] world={world}  dataset={os.path.basename(args.dataset)}")

    t_load = time.perf_counter()
    engine = make_engine(args, world)
    if rank == 0:
        print(f"[longbench_sweep] engine ready in {time.perf_counter() - t_load:.1f}s")

    tokenizer = engine.tokenizer
    tiers_needed = sorted({c["tier"] for c in build_cell_plan(args)})
    prompts_by_tier = load_prompts_at_tiers(
        args.dataset, tokenizer, tiers_needed,
        num_per_tier=args.num_samples_per_tier,
        seed=args.seed,
    )

    # rank0 reports counts; broadcast prompts to all ranks for identical batches.
    if rank == 0:
        for tier, ps in prompts_by_tier.items():
            lens = [p["prompt_len"] for p in ps]
            if lens:
                p50 = sorted(lens)[len(lens) // 2]
                print(f"[tier={tier}] n={len(ps)} p50={p50} range=[{min(lens)}, {max(lens)}]")
            else:
                print(f"[tier={tier}] n=0 (no prompts matched length filter!)")
    obj = [prompts_by_tier]
    dist.broadcast_object_list(obj, src=0)
    prompts_by_tier = obj[0]

    cells = build_cell_plan(args)
    # Shuffle cell order with a fixed seed for reproducibility.
    rng = random.Random(args.seed)
    rng.shuffle(cells)
    if rank == 0:
        print(f"[longbench_sweep] running {len(cells)} cells (shuffled).")

    jsonl_path = os.path.join(args.output_dir, "longbench_rows.jsonl")
    truncate_file(jsonl_path)
    dist.barrier()

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=f"longbench_sweep_w{world}_seed{args.seed}",
        config=vars(args),
        enabled=bool(args.wandb_project),
    )

    # ---- One-time warmup at medium tier in baseline mode (JIT + caches) ----
    if args.warmup_batches > 0:
        configure_drop(engine, "none", 0.0, args.phase_ab_min_replicas)
        warm_set = prompts_by_tier.get(args.phase_ab_tier) or next(iter(prompts_by_tier.values()))
        warm_batches = [warm_set[i:i + args.batch_size]
                        for i in range(0, len(warm_set), args.batch_size)]
        if rank == 0:
            print(f"\n[warmup] {args.warmup_batches} baseline batches at tier={args.phase_ab_tier}; "
                  f"timing discarded")
        for w in range(args.warmup_batches):
            batch = warm_batches[w % len(warm_batches)]
            tokens_list = [p["prompt_token_ids"] for p in batch]
            wm = run_batch(engine, tokens_list, args.max_new_tokens)
            torch.cuda.synchronize()
            if rank == 0:
                print(f"  warmup {w+1}/{args.warmup_batches}  "
                      f"prefill={wm['prefill_time_s']:.3f}s (discarded)")
            dist.barrier()

    # ---- Sweep ----
    for ci, cell in enumerate(cells):
        configure_drop(engine, cell["policy"], cell["rate"], cell["min_replicas"])
        prompt_set = prompts_by_tier[cell["tier"]]
        batches = [prompt_set[i:i + args.batch_size]
                   for i in range(0, len(prompt_set), args.batch_size)]
        if rank == 0:
            print(f"\n[cell {ci+1}/{len(cells)}] phase={cell['phase']} label={cell['label']} "
                  f"policy={cell['policy']} rate={cell['rate']} "
                  f"min_replicas={cell['min_replicas']} tier={cell['tier']} batches={len(batches)}")
        for bi, batch in enumerate(batches):
            tokens_list = [p["prompt_token_ids"] for p in batch]
            m = run_batch(engine, tokens_list, args.max_new_tokens)
            torch.cuda.synchronize()
            row = {
                "cell_id": ci,
                "phase": cell["phase"],
                "label": cell["label"],
                "policy": cell["policy"],
                "rate": cell["rate"],
                "min_replicas": cell["min_replicas"],
                "tier": cell["tier"],
                "batch_id": bi,
                "batch_size": len(batch),
                "world_size": world,
                "prompt_lens": [p["prompt_len"] for p in batch],
                "total_prompt_tokens": sum(p["prompt_len"] for p in batch),
                **m,
            }
            if rank == 0:
                write_jsonl(jsonl_path, row)
                wandb_logger.log(row)
                print(f"    batch={bi+1}/{len(batches)}  "
                      f"prompt_tok={row['total_prompt_tokens']:6d}  "
                      f"prefill={m['prefill_time_s']:.3f}s  "
                      f"e2e={m['e2e_total_time_s']:.3f}s  "
                      f"tok_s={m['prefill_tok_s']:.0f}")
            dist.barrier()

    # ---- Aggregate ----
    if rank == 0:
        from collections import defaultdict
        rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
        per_cell: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            per_cell[r["label"]].append(r)
        cell_summary: dict[str, dict] = {}
        for label, rs in per_cell.items():
            cell_summary[label] = {
                "phase": rs[0]["phase"],
                "policy": rs[0]["policy"],
                "rate": rs[0]["rate"],
                "min_replicas": rs[0]["min_replicas"],
                "tier": rs[0]["tier"],
                "n_batches": len(rs),
                "prefill_time_s": agg_stats([r["prefill_time_s"] for r in rs if r["prefill_time_s"] is not None]),
                "prefill_tok_s": agg_stats([r["prefill_tok_s"] for r in rs if r["prefill_tok_s"] is not None]),
                "e2e_total_time_s": agg_stats([r["e2e_total_time_s"] for r in rs if r["e2e_total_time_s"] is not None]),
                "decode_time_s": agg_stats([r["decode_time_s"] for r in rs if r["decode_time_s"] is not None]),
                "total_prompt_tokens_mean": sum(r["total_prompt_tokens"] for r in rs) / max(len(rs), 1),
            }

        # Compute speedups: each non-baseline cell vs the matching baseline (same tier, same min_replicas
        # for Phase A; same tier for Phase C; tail_weight@0.3 across bypass for Phase B uses phase A baseline).
        # We pick the baseline by matching (tier, phase-specific rule).
        # Simple rule: baseline = first cell with policy=='none' and same tier; min_replicas matches the
        # phase's bypass for A; matches phase_c_min_replicas for C; for B we always compare against
        # the Phase A baseline since the same prompt set is used.
        a_baseline = next(
            (s for s in cell_summary.values()
             if s["policy"] == "none" and s["tier"] == args.phase_ab_tier),
            None,
        )
        speedups = {}
        for label, s in cell_summary.items():
            if s["policy"] == "none":
                continue
            if s["phase"] in ("A", "B"):
                ref = a_baseline
            else:  # phase C: baseline at same tier
                ref = next(
                    (x for x in cell_summary.values()
                     if x["policy"] == "none" and x["tier"] == s["tier"]),
                    a_baseline,
                )
            if ref is None:
                continue
            b_pt = ref["prefill_time_s"]["mean"]
            d_pt = s["prefill_time_s"]["mean"]
            b_e2e = ref["e2e_total_time_s"]["mean"]
            d_e2e = s["e2e_total_time_s"]["mean"]
            speedups[label] = {
                "ref_label": next(k for k, v in cell_summary.items() if v is ref),
                "prefill_speedup": b_pt / d_pt if d_pt > 0 else None,
                "prefill_delta_pct": (d_pt - b_pt) / b_pt if b_pt > 0 else None,
                "e2e_speedup": b_e2e / d_e2e if d_e2e > 0 else None,
                "e2e_delta_pct": (d_e2e - b_e2e) / b_e2e if b_e2e > 0 else None,
            }

        out = {
            "config": vars(args),
            "cells": cell_summary,
            "speedups": speedups,
        }
        out_path = os.path.join(args.output_dir, "longbench_summary.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[longbench_sweep] wrote {out_path}")
        # Brief table
        print("\nPhase A (policy x rate, tier=medium):")
        for label, s in sorted(cell_summary.items(), key=lambda kv: kv[0]):
            if s["phase"] != "A":
                continue
            sp = speedups.get(label, {})
            print(f"  {label:38s}  prefill={s['prefill_time_s']['mean']:.3f}s  "
                  f"speedup={sp.get('prefill_speedup') or '-':>6}  "
                  f"e2e_speedup={sp.get('e2e_speedup') or '-'}")
        print("Phase B (bypass sweep):")
        for label, s in sorted(cell_summary.items(), key=lambda kv: kv[0]):
            if s["phase"] != "B":
                continue
            sp = speedups.get(label, {})
            print(f"  {label:38s}  prefill={s['prefill_time_s']['mean']:.3f}s  "
                  f"speedup={sp.get('prefill_speedup') or '-'}")
        print("Phase C (length tier):")
        for label, s in sorted(cell_summary.items(), key=lambda kv: kv[0]):
            if s["phase"] != "C":
                continue
            sp = speedups.get(label, {})
            print(f"  {label:38s}  prefill={s['prefill_time_s']['mean']:.3f}s  "
                  f"speedup={sp.get('prefill_speedup') or '-'}")

    wandb_logger.finish()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
