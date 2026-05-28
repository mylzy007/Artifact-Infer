"""LongBench drop sweep v2 — full policy set + accuracy.

Differences from longbench_sweep.py:
  - Tests all 7 ALL_DROP_POLICIES (3 GPU + 4 CPU-grouped):
      tail_weight, random, cross_numa_first,
      per_expert_uniform, hot_expert_relief, per_expert_tailtoken, hotspot_relief
  - Computes token-level F1 score against `reference_answer` to track accuracy.
  - Uses max_new_tokens=128 (long enough for LEval Generation_multidoc_qa answers).
  - Drops phase B (bypass) and phase C (tier) — those were settled in v1.

One engine load handles all cells; only `dispatch.drop_policy/rate/min_replicas`
are mutated between cells. Drop impl is "auto": GPU for simple policies, CPU
for grouped policies (the dispatcher picks per-policy at apply_drop time).

Usage:

    CUDA_HOME=/usr/local/cuda-12.8 \\
        PATH=/usr/local/cuda-12.8/bin:$PATH \\
        FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_tier2 \\
        /home/lzy/miniconda3/envs/vllm/bin/torchrun --nproc_per_node=8 \\
        --master_port=29514 -m eval.drop.longbench_sweep_v2 \\
        --output-dir eval_results/owner_local_ep_phase4_drop_longbench_v2 \\
        --dataset /home/lzy/datasets/moe_benchmarks/prepared/leval.Generation_multidoc_qa.custom.jsonl \\
        --expert-overlap-path eval_results/.../moe_overlap_plan_smoke_plan_20260521_222623.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
import time

# Drop env vars before importing dispatch_ep_ht.
os.environ.setdefault("MOE_DROP_IMPL", "auto")        # auto-route GPU vs CPU per policy
os.environ.setdefault("MOE_DROP_GPU_STATS", "0")
os.environ.setdefault("MOE_PROFILE_OVERLAP_RUNTIME", "0")
os.environ.setdefault("MOE_RECORD_TIMING", "1")
os.environ.setdefault("MOE_PROFILE_ROUTING", "0")

import torch
import torch.distributed as dist

from eval.drop.shared import WandbLogger, agg_stats, truncate_file, write_jsonl


ALL_POLICIES = [
    "tail_weight",          # GPU
    "random",               # GPU
    "cross_numa_first",     # GPU
    "per_expert_uniform",   # CPU grouped
    "hot_expert_relief",    # CPU grouped
    "per_expert_tailtoken", # CPU grouped
    "hotspot_relief",       # CPU grouped
]
GPU_POLICIES = {
    "tail_weight", "random", "cross_numa_first",
    "weighted_tail", "cross_numa_uniform",  # added v3
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-30B-A3B"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--num-samples", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--min-prompt-tokens", type=int, default=2048)
    p.add_argument("--max-prompt-tokens", type=int, default=4096)
    p.add_argument("--expert-overlap-path", default="")
    p.add_argument("--expert-overlap-strategy", default="greedy_balance")
    p.add_argument("--max-num-batched-tokens", type=int, default=6144)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-batches", type=int, default=3)
    p.add_argument("--wandb-project", default="")
    p.add_argument("--drop-rates", default="0.1,0.3,0.5")
    p.add_argument("--drop-policies", default=",".join(ALL_POLICIES))
    p.add_argument("--min-replicas", type=int, default=512,
                   help="Phase 1 finding: 512 bypasses decode without losing prefill")
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


def make_engine(args, world: int):
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


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, collapse whitespace, return token list."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return [tok for tok in text.split() if tok]


def f1_score(pred: str, ref: str) -> float:
    pred_tokens = _normalize(pred)
    ref_tokens = _normalize(ref)
    if not ref_tokens or not pred_tokens:
        return 0.0
    pred_set = {}
    for t in pred_tokens:
        pred_set[t] = pred_set.get(t, 0) + 1
    ref_set = {}
    for t in ref_tokens:
        ref_set[t] = ref_set.get(t, 0) + 1
    common = 0
    for t, c in pred_set.items():
        common += min(c, ref_set.get(t, 0))
    if common == 0:
        return 0.0
    precision = common / sum(pred_set.values())
    recall = common / sum(ref_set.values())
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(dataset_path: str, tokenizer, args) -> list[dict]:
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
                "reference_answer": str(d.get("reference_answer", "")),
            })
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    out = []
    for r in rows:
        ids = tokenizer.encode(r["prompt"], add_special_tokens=True)
        if args.min_prompt_tokens <= len(ids) <= args.max_prompt_tokens:
            r["prompt_token_ids"] = ids
            r["prompt_len"] = len(ids)
            out.append(r)
        if len(out) >= args.num_samples:
            break
    return out


def run_batch(engine, prompt_token_ids: list[list[int]], max_new_tokens: int):
    from workshop.nanovllm_moe.services.sampling_params import SamplingParams
    engine.reset()
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    t0 = time.perf_counter()
    outputs = engine.generate(prompt_token_ids, sp, use_tqdm=False)
    wall_s = time.perf_counter() - t0
    metrics = getattr(engine, "last_generation_metrics", {}) or {}
    return wall_s, outputs, metrics


def main() -> int:
    args = parse_args()
    rank, local_rank, world = setup_distributed()

    if rank == 0:
        print(f"[v2] world={world}  dataset={os.path.basename(args.dataset)}")

    t_load = time.perf_counter()
    engine = make_engine(args, world)
    if rank == 0:
        print(f"[v2] engine ready in {time.perf_counter() - t_load:.1f}s")

    tokenizer = engine.tokenizer
    prompts = load_prompts(args.dataset, tokenizer, args)
    if rank == 0:
        lens = [p["prompt_len"] for p in prompts]
        p50 = sorted(lens)[len(lens) // 2] if lens else 0
        print(f"[v2] {len(prompts)} prompts, len p50={p50}, range=[{min(lens)},{max(lens)}]")

    obj = [prompts]
    dist.broadcast_object_list(obj, src=0)
    prompts = obj[0]

    rates = sorted({float(x) for x in args.drop_rates.split(",")})
    policies = [s.strip() for s in args.drop_policies.split(",") if s.strip()]

    cells = [{"label": "baseline", "policy": "none", "rate": 0.0}]
    for pol in policies:
        for rate in rates:
            cells.append({
                "label": f"{pol}_r{rate}",
                "policy": pol,
                "rate": rate,
            })

    rng = random.Random(args.seed)
    rng.shuffle(cells)
    if rank == 0:
        print(f"[v2] running {len(cells)} cells; min_replicas={args.min_replicas}")

    batches = [
        prompts[i:i + args.batch_size]
        for i in range(0, len(prompts), args.batch_size)
    ]
    if rank == 0:
        print(f"[v2] {len(batches)} batches × {args.batch_size} prompts/batch")

    jsonl_path = os.path.join(args.output_dir, "v2_rows.jsonl")
    truncate_file(jsonl_path)
    dist.barrier()

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=f"v2_w{world}_seed{args.seed}",
        config=vars(args),
        enabled=bool(args.wandb_project),
    )

    # Warmup with baseline.
    if args.warmup_batches > 0:
        configure_drop(engine, "none", 0.0, args.min_replicas)
        if rank == 0:
            print(f"[v2] warmup {args.warmup_batches} baseline batches")
        for w in range(args.warmup_batches):
            batch = batches[w % len(batches)]
            tokens_list = [p["prompt_token_ids"] for p in batch]
            wm_s, _, wm_metrics = run_batch(engine, tokens_list, args.max_new_tokens)
            torch.cuda.synchronize()
            if rank == 0:
                pt = wm_metrics.get("prefill_time_s") or 0.0
                print(f"  warmup {w+1}/{args.warmup_batches}  prefill={pt:.3f}s (discarded)")
            dist.barrier()

    # ---- sweep ----
    for ci, cell in enumerate(cells):
        n_changed = configure_drop(engine, cell["policy"], cell["rate"], args.min_replicas)
        is_cpu_path = cell["policy"] not in GPU_POLICIES and cell["policy"] != "none"
        if rank == 0:
            print(f"\n[cell {ci+1}/{len(cells)}] {cell['label']:38s}  "
                  f"({'CPU' if is_cpu_path else 'GPU' if cell['policy'] != 'none' else 'baseline'})  "
                  f"applied to {n_changed} layers")
        for bi, batch in enumerate(batches):
            tokens_list = [p["prompt_token_ids"] for p in batch]
            wall_s, outputs, metrics = run_batch(engine, tokens_list, args.max_new_tokens)
            torch.cuda.synchronize()

            # Compute F1 on rank0 only (outputs gathered there).
            f1_per_prompt = []
            generations = []
            if rank == 0 and outputs:
                for ref_prompt, out in zip(batch, outputs):
                    gen_text = (out or {}).get("text", "")
                    ref_ans = ref_prompt.get("reference_answer", "")
                    f1 = f1_score(gen_text, ref_ans) if ref_ans else None
                    f1_per_prompt.append(f1)
                    generations.append({
                        "request_id": ref_prompt["request_id"],
                        "prompt_len": ref_prompt["prompt_len"],
                        "ref_text": ref_ans,                # v4: full reference for offline re-score
                        "gen_text": gen_text,               # v4: store full generation
                        "f1": f1,
                    })

            row = {
                "cell_id": ci,
                "label": cell["label"],
                "policy": cell["policy"],
                "rate": cell["rate"],
                "min_replicas": args.min_replicas,
                "is_cpu_path": is_cpu_path,
                "batch_id": bi,
                "batch_size": len(batch),
                "world_size": world,
                "prompt_lens": [p["prompt_len"] for p in batch],
                "total_prompt_tokens": sum(p["prompt_len"] for p in batch),
                "wall_s": wall_s,
                "prefill_tokens": metrics.get("prefill_tokens"),
                "prefill_time_s": metrics.get("prefill_time_s"),
                "prefill_tok_s": metrics.get("prefill_throughput_tok_s"),
                "decode_tokens": metrics.get("decode_tokens"),
                "decode_time_s": metrics.get("decode_time_s"),
                "decode_tok_s": metrics.get("decode_throughput_tok_s"),
                "e2e_total_time_s": metrics.get("e2e_total_time_s"),
                "f1_per_prompt": f1_per_prompt,
                "f1_mean": (sum(v for v in f1_per_prompt if v is not None) / max(1, len([v for v in f1_per_prompt if v is not None])))
                           if f1_per_prompt and any(v is not None for v in f1_per_prompt) else None,
                "generations": generations,
            }
            if rank == 0:
                write_jsonl(jsonl_path, row)
                wandb_logger.log({k: v for k, v in row.items() if k != "generations"})
                pt = row["prefill_time_s"] or 0.0
                et = row["e2e_total_time_s"] or 0.0
                f1 = row["f1_mean"]
                print(f"    b{bi+1}/{len(batches)}  tok={row['total_prompt_tokens']:6d}  "
                      f"prefill={pt:.3f}s  e2e={et:.3f}s  "
                      f"F1={'%.3f' % f1 if f1 is not None else '-':>6}")
            dist.barrier()

    # ---- aggregate ----
    if rank == 0:
        from collections import defaultdict
        rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
        per_cell = defaultdict(list)
        for r in rows:
            per_cell[r["label"]].append(r)
        cell_agg = {}
        for label, rs in per_cell.items():
            f1_vals = [r["f1_mean"] for r in rs if r["f1_mean"] is not None]
            cell_agg[label] = {
                "policy": rs[0]["policy"],
                "rate": rs[0]["rate"],
                "is_cpu_path": rs[0]["is_cpu_path"],
                "n_batches": len(rs),
                "prefill_time_s": agg_stats([r["prefill_time_s"] for r in rs if r["prefill_time_s"] is not None]),
                "prefill_tok_s": agg_stats([r["prefill_tok_s"] for r in rs if r["prefill_tok_s"] is not None]),
                "e2e_total_time_s": agg_stats([r["e2e_total_time_s"] for r in rs if r["e2e_total_time_s"] is not None]),
                "decode_time_s": agg_stats([r["decode_time_s"] for r in rs if r["decode_time_s"] is not None]),
                "f1_mean": agg_stats(f1_vals) if f1_vals else None,
                "f1_per_prompt_flat": [v for r in rs for v in r["f1_per_prompt"] if v is not None],
            }

        # speedups & accuracy delta vs baseline
        base = cell_agg.get("baseline")
        rel = {}
        if base is not None:
            base_pt = base["prefill_time_s"]["mean"]
            base_e2e = base["e2e_total_time_s"]["mean"]
            base_f1 = base["f1_mean"]["mean"] if base["f1_mean"] else None
            for label, s in cell_agg.items():
                if label == "baseline":
                    continue
                pt = s["prefill_time_s"]["mean"]
                e2e = s["e2e_total_time_s"]["mean"]
                f1m = s["f1_mean"]["mean"] if s["f1_mean"] else None
                rel[label] = {
                    "prefill_speedup": base_pt / pt if pt > 0 else None,
                    "prefill_delta_pct": (pt - base_pt) / base_pt if base_pt > 0 else None,
                    "e2e_speedup": base_e2e / e2e if e2e > 0 else None,
                    "e2e_delta_pct": (e2e - base_e2e) / base_e2e if base_e2e > 0 else None,
                    "f1_baseline": base_f1,
                    "f1_drop": f1m,
                    "f1_delta_abs": (f1m - base_f1) if (f1m is not None and base_f1 is not None) else None,
                    "f1_delta_pct": ((f1m - base_f1) / base_f1) if (f1m is not None and base_f1) else None,
                }

        out = {
            "config": vars(args),
            "cells": cell_agg,
            "relative_to_baseline": rel,
        }
        out_path = os.path.join(args.output_dir, "v2_summary.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[v2] wrote {out_path}")

        # Print compact table
        print(f"\n{'label':<32}{'path':>5}{'prefill_sp':>12}{'e2e_sp':>9}{'f1_mean':>10}{'f1_Δ':>9}")
        if base is not None:
            print(f"{'baseline':<32}{'-':>5}{'1.000':>12}{'1.000':>9}{base['f1_mean']['mean'] if base['f1_mean'] else 0:>10.3f}{'-':>9}")
        for label in sorted(rel.keys()):
            s = cell_agg[label]
            r = rel[label]
            path = "CPU" if s["is_cpu_path"] else "GPU"
            psp = r["prefill_speedup"] or 0
            esp = r["e2e_speedup"] or 0
            f1m = (s["f1_mean"]["mean"] if s["f1_mean"] else 0)
            f1d = r["f1_delta_abs"]
            print(f"{label:<32}{path:>5}{psp:>12.3f}{esp:>9.3f}{f1m:>10.3f}"
                  f"{('%+.3f' % f1d) if f1d is not None else '-':>9}")

    wandb_logger.finish()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
