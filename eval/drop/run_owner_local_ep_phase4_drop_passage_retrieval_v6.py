"""Drop sweep on LongBench passage_retrieval_en_e with OFFICIAL evaluation.

Why this task:
  - LongBench official task; native binary metric (`retrieval_score`)
  - 300 prompts in extended subset; 112 fall within our 24GB memory budget
    ([500, 5800] context tokens)
  - Answer format: "Paragraph N" — no CoT-extraction issues
  - max_new_tokens=32 (LongBench official) is plenty since reply is short

Pipeline matches THUDM/LongBench eval exactly:
  - Template: `dataset2prompt["passage_retrieval_en"]` verbatim
  - max_new_tokens: `dataset2maxlen["passage_retrieval_en"] = 32`
  - Metric: `retrieval_score` from `LongBench/metrics.py`

Sweeps:
  - 2 Phase 3 overlap plans (RR/min_comm + LBG/greedy_balance)
  - 5 GPU drop policies + baseline (1 + 5×3 = 16 cells per plan)
  - 64 prompts × 8 batches per cell
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict

os.environ.setdefault("MOE_DROP_IMPL", "auto")
os.environ.setdefault("MOE_DROP_GPU_STATS", "0")
os.environ.setdefault("MOE_PROFILE_OVERLAP_RUNTIME", "0")
os.environ.setdefault("MOE_RECORD_TIMING", "1")
os.environ.setdefault("MOE_PROFILE_ROUTING", "0")

import torch
import torch.distributed as dist

from eval.drop.shared import WandbLogger, agg_stats, truncate_file, write_jsonl


# ----------------------------------------------------------------------
# LongBench official config for passage_retrieval_en (verbatim)
# ----------------------------------------------------------------------

OFFICIAL_PROMPT = (
    "Here are 30 paragraphs from Wikipedia, along with an abstract. "
    "Please determine which paragraph the abstract is from.\n\n{context}\n\n"
    "The following is an abstract.\n\n{input}\n\n"
    "Please enter the number of the paragraph that the abstract is from. "
    "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\n"
    "The answer is: "
)
OFFICIAL_MAX_NEW_TOKENS = 32


def retrieval_score(prediction: str, ground_truth: str) -> float:
    """Official `retrieval_score` from THUDM/LongBench/metrics.py.

    Extract ground-truth paragraph id from `Paragraph N` pattern in
    ground_truth; find all integers in prediction; return 1/k if the gt id
    appears k>0 times, else 0.
    """
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    gt_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for x in numbers if x == gt_id)
    if right_num == 0:
        return 0.0
    return 1.0 / right_num


def retrieval_score_strict(prediction: str, ground_truth: str) -> float:
    """Stricter variant: 1.0 only if FIRST number in prediction == gt_id, else 0.0.

    Complements the official score: official gives 0.5 for "Paragraph 5 or 7"
    (one match in two numbers); strict gives 0 unless first guess is correct.
    """
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    gt_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return 1.0 if numbers[0] == gt_id else 0.0


# ----------------------------------------------------------------------
# Distributed setup + engine + drop config
# ----------------------------------------------------------------------

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


def build_engine(args, world):
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


def run_batch(engine, prompt_token_ids, max_new_tokens):
    from workshop.nanovllm_moe.services.sampling_params import SamplingParams
    engine.reset()
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    t0 = time.perf_counter()
    outs = engine.generate(prompt_token_ids, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    metrics = getattr(engine, "last_generation_metrics", {}) or {}
    return wall, outs, metrics


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-30B-A3B"))
    p.add_argument("--dataset", default="/home/lzy/datasets/moe_benchmarks/longbench/extracted/data/passage_retrieval_en_e.jsonl")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--min-prompt-tokens", type=int, default=500)
    p.add_argument("--max-prompt-tokens", type=int, default=5800)
    p.add_argument("--max-new-tokens", type=int, default=OFFICIAL_MAX_NEW_TOKENS)
    p.add_argument("--drop-rates", default="0.1,0.3,0.5")
    p.add_argument("--drop-policies",
                   default="tail_weight,random,cross_numa_first,weighted_tail,cross_numa_uniform")
    p.add_argument("--min-replicas", type=int, default=512)
    p.add_argument("--expert-overlap-path", required=True)
    p.add_argument("--expert-overlap-strategy", default="greedy_balance")
    p.add_argument("--max-num-batched-tokens", type=int, default=6144)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--warmup-batches", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rank, local_rank, world = setup_distributed()

    if rank == 0:
        print(f"[passage_retrieval] dataset={args.dataset}")
        print(f"[passage_retrieval] OFFICIAL max_new_tokens={OFFICIAL_MAX_NEW_TOKENS}")

    # Build engine first to get tokenizer
    t_load = time.perf_counter()
    engine = build_engine(args, world)
    if rank == 0:
        print(f"[passage_retrieval] engine ready in {time.perf_counter() - t_load:.1f}s")
    tok = engine.tokenizer

    # Load + wrap + filter
    rng = random.Random(args.seed)
    raw = [json.loads(line) for line in open(args.dataset) if line.strip()]
    rng.shuffle(raw)

    prompts = []
    skipped = 0
    for r in raw:
        ctx = r.get("context", "")
        inp = r.get("input", "")
        answers = r.get("answers", [])
        if not ctx or not inp or not answers:
            continue
        wrapped = OFFICIAL_PROMPT.format(context=ctx, input=inp)
        token_ids = tok.encode(wrapped, add_special_tokens=True)
        n = len(token_ids)
        if n < args.min_prompt_tokens or n > args.max_prompt_tokens:
            skipped += 1; continue
        prompts.append({
            "_id": r.get("_id", f"row-{len(prompts)}"),
            "context_head": ctx[:200],
            "input": inp,
            "wrapped_prompt": wrapped,
            "token_ids": token_ids,
            "prompt_len": n,
            "reference": answers[0],  # passage_retrieval_en has single answer
            "all_answers": answers,
        })
        if len(prompts) >= args.num_samples:
            break

    if rank == 0:
        if not prompts:
            print(f"[passage_retrieval] NO PROMPTS in [{args.min_prompt_tokens}, {args.max_prompt_tokens}]")
            return 1
        lens = [p["prompt_len"] for p in prompts]
        print(f"[passage_retrieval] kept {len(prompts)} prompts (skipped {skipped})")
        print(f"  prompt_len p50={sorted(lens)[len(lens)//2]} range=[{min(lens)}, {max(lens)}]")
        p0 = prompts[0]
        print(f"  sample input: {p0['input'][:150]!r}")
        print(f"  sample reference: {p0['reference']!r}")

    obj = [prompts]
    dist.broadcast_object_list(obj, src=0)
    prompts = obj[0]

    batches = [prompts[i:i + args.batch_size]
               for i in range(0, len(prompts), args.batch_size)]
    if rank == 0:
        print(f"[passage_retrieval] {len(batches)} batches × {args.batch_size}")

    # Cell plan: 1 baseline + 5 policies × 3 rates
    rates = sorted({float(x) for x in args.drop_rates.split(",")})
    policies = [s.strip() for s in args.drop_policies.split(",") if s.strip()]
    cells = [{"label": "baseline", "policy": "none", "rate": 0.0}]
    for pol in policies:
        for rate in rates:
            cells.append({"label": f"{pol}_r{rate}", "policy": pol, "rate": rate})
    rng = random.Random(args.seed)
    rng.shuffle(cells)
    if rank == 0:
        print(f"[passage_retrieval] running {len(cells)} cells, min_replicas={args.min_replicas}")

    jsonl_path = os.path.join(args.output_dir, "passage_retrieval_rows.jsonl")
    truncate_file(jsonl_path)
    dist.barrier()

    # Warmup (baseline)
    if args.warmup_batches > 0:
        configure_drop(engine, "none", 0.0, args.min_replicas)
        if rank == 0:
            print(f"[passage_retrieval] warmup {args.warmup_batches} baseline batches")
        for w in range(args.warmup_batches):
            batch = batches[w % len(batches)]
            ids = [p["token_ids"] for p in batch]
            wall, _, m = run_batch(engine, ids, args.max_new_tokens)
            torch.cuda.synchronize()
            if rank == 0:
                print(f"  warmup {w+1}: prefill={m.get('prefill_time_s'):.2f}s (discarded)")
            dist.barrier()

    # Sweep
    for ci, cell in enumerate(cells):
        n_changed = configure_drop(engine, cell["policy"], cell["rate"], args.min_replicas)
        if rank == 0:
            print(f"\n[cell {ci+1}/{len(cells)}] {cell['label']:35s} "
                  f"applied to {n_changed} layers")
        for bi, batch in enumerate(batches):
            ids = [p["token_ids"] for p in batch]
            wall, outputs, metrics = run_batch(engine, ids, args.max_new_tokens)
            torch.cuda.synchronize()

            # Score on rank 0
            batch_scores_official = []
            batch_scores_strict = []
            gens = []
            if rank == 0 and outputs:
                for src, out in zip(batch, outputs):
                    gen = (out or {}).get("text", "")
                    ref = src["reference"]
                    s_off = retrieval_score(gen, ref)
                    s_str = retrieval_score_strict(gen, ref)
                    batch_scores_official.append(s_off)
                    batch_scores_strict.append(s_str)
                    gens.append({
                        "_id": src["_id"],
                        "reference": ref,
                        "gen_text": gen,
                        "score_official": s_off,
                        "score_strict": s_str,
                        "prompt_len": src["prompt_len"],
                    })

            row = {
                "cell_id": ci,
                "label": cell["label"],
                "policy": cell["policy"],
                "rate": cell["rate"],
                "min_replicas": args.min_replicas,
                "batch_id": bi,
                "batch_size": len(batch),
                "world_size": world,
                "total_prompt_tokens": sum(p["prompt_len"] for p in batch),
                "wall_s": wall,
                "prefill_tokens": metrics.get("prefill_tokens"),
                "prefill_time_s": metrics.get("prefill_time_s"),
                "prefill_tok_s": metrics.get("prefill_throughput_tok_s"),
                "decode_time_s": metrics.get("decode_time_s"),
                "decode_tok_s": metrics.get("decode_throughput_tok_s"),
                "e2e_total_time_s": metrics.get("e2e_total_time_s"),
                "batch_acc_official": (sum(batch_scores_official) / len(batch_scores_official)
                                        if batch_scores_official else None),
                "batch_acc_strict": (sum(batch_scores_strict) / len(batch_scores_strict)
                                      if batch_scores_strict else None),
                "generations": gens,
            }
            if rank == 0:
                write_jsonl(jsonl_path, row)
                pt = row["prefill_time_s"] or 0
                et = row["e2e_total_time_s"] or 0
                ao = row["batch_acc_official"] or 0
                a_s = row["batch_acc_strict"] or 0
                print(f"  b{bi+1}/{len(batches)}  tok={row['total_prompt_tokens']:6d}  "
                      f"prefill={pt:.2f}s e2e={et:.2f}s  "
                      f"acc(off)={ao:.3f} acc(strict)={a_s:.3f}")
            dist.barrier()

    # Summary on rank 0
    if rank == 0:
        rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
        per_cell = defaultdict(list)
        for r in rows:
            per_cell[r["label"]].append(r)

        cell_agg = {}
        for label, rs in per_cell.items():
            n_prompts = sum(r["batch_size"] for r in rs)
            # Collect per-prompt scores
            off_scores = [g["score_official"] for r in rs for g in r["generations"]]
            str_scores = [g["score_strict"] for r in rs for g in r["generations"]]
            cell_agg[label] = {
                "policy": rs[0]["policy"],
                "rate": rs[0]["rate"],
                "n_batches": len(rs),
                "n_prompts": n_prompts,
                "prefill_time_s": agg_stats([r["prefill_time_s"] for r in rs if r["prefill_time_s"] is not None]),
                "prefill_tok_s": agg_stats([r["prefill_tok_s"] for r in rs if r["prefill_tok_s"] is not None]),
                "e2e_total_time_s": agg_stats([r["e2e_total_time_s"] for r in rs if r["e2e_total_time_s"] is not None]),
                "decode_time_s": agg_stats([r["decode_time_s"] for r in rs if r["decode_time_s"] is not None]),
                "acc_official": {
                    "mean": (sum(off_scores)/len(off_scores)) if off_scores else None,
                    "sem": (statistics.stdev(off_scores)/(len(off_scores)**0.5)) if len(off_scores) > 1 else 0.0,
                    "n": len(off_scores),
                },
                "acc_strict": {
                    "mean": (sum(str_scores)/len(str_scores)) if str_scores else None,
                    "sem": (statistics.stdev(str_scores)/(len(str_scores)**0.5)) if len(str_scores) > 1 else 0.0,
                    "n": len(str_scores),
                },
            }

        base = cell_agg.get("baseline")
        rel = {}
        if base is not None:
            base_pt = base["prefill_time_s"]["mean"]
            base_e2e = base["e2e_total_time_s"]["mean"]
            base_acc_o = base["acc_official"]["mean"]
            base_acc_s = base["acc_strict"]["mean"]
            for label, s in cell_agg.items():
                if label == "baseline":
                    continue
                pt = s["prefill_time_s"]["mean"]
                e2e = s["e2e_total_time_s"]["mean"]
                ao = s["acc_official"]["mean"]
                a_s = s["acc_strict"]["mean"]
                rel[label] = {
                    "prefill_speedup": base_pt / pt if pt > 0 else None,
                    "prefill_delta_pct": (pt - base_pt) / base_pt if base_pt > 0 else None,
                    "e2e_speedup": base_e2e / e2e if e2e > 0 else None,
                    "e2e_delta_pct": (e2e - base_e2e) / base_e2e if base_e2e > 0 else None,
                    "acc_official_drop": ao,
                    "acc_official_delta": (ao - base_acc_o) if ao is not None and base_acc_o is not None else None,
                    "acc_strict_drop": a_s,
                    "acc_strict_delta": (a_s - base_acc_s) if a_s is not None and base_acc_s is not None else None,
                }

        summary = {
            "config": vars(args),
            "task": "passage_retrieval_en",
            "max_new_tokens_official": OFFICIAL_MAX_NEW_TOKENS,
            "metric_official": "retrieval_score from THUDM/LongBench/metrics.py",
            "cells": cell_agg,
            "relative_to_baseline": rel,
        }
        out_path = os.path.join(args.output_dir, "passage_retrieval_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[passage_retrieval] wrote {out_path}")

        # Print table
        print(f"\n{'label':<32}{'prefill_sp':>11}{'e2e_sp':>9}{'tok/s':>8}{'acc(off)':>10}{'Δoff':>9}{'acc(strict)':>13}{'Δstr':>9}")
        if base is not None:
            print(f"{'baseline':<32}{'1.000':>11}{'1.000':>9}{base['prefill_tok_s']['mean']:>8.0f}"
                  f"{base['acc_official']['mean']:>10.3f}{'—':>9}{base['acc_strict']['mean']:>13.3f}{'—':>9}")
        for label in sorted(rel.keys()):
            s = cell_agg[label]
            r = rel[label]
            print(f"{label:<32}{r['prefill_speedup']:>11.3f}{r['e2e_speedup']:>9.3f}"
                  f"{s['prefill_tok_s']['mean']:>8.0f}"
                  f"{s['acc_official']['mean']:>10.3f}{r['acc_official_delta']:>+9.3f}"
                  f"{s['acc_strict']['mean']:>13.3f}{r['acc_strict_delta']:>+9.3f}")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
