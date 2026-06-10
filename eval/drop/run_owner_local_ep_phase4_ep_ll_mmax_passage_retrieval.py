"""EP-LL M_max sweep on LongBench passage_retrieval_en_e.

Full-model owner_local_ep evaluation on 8 GPUs with official LongBench
`retrieval_score`, sweeping EP-LL fixed-capacity `M_max`.
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
from pathlib import Path

os.environ.setdefault("MOE_RECORD_TIMING", "1")
os.environ.setdefault("MOE_PROFILE_ROUTING", "0")

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from eval.drop.shared import agg_stats, truncate_file, write_jsonl


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
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    gt_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return 1.0 if numbers[0] == gt_id else 0.0


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


def resolve_ep_ll_placement(args) -> tuple[str, str | None]:
    if args.expert_placement_path:
        return args.moe_expert_placement, args.expert_placement_path
    if not args.expert_overlap_path:
        return args.moe_expert_placement, None
    payload = json.loads(Path(args.expert_overlap_path).read_text())
    placement_path = payload.get("base_placement_json_path")
    placement_name = payload.get("base_placement") or args.moe_expert_placement
    return str(placement_name), str(placement_path) if placement_path else None


def build_engine(args, world: int, *, m_max: int):
    from workshop.nanovllm_moe.services.engine.llm_engine import LLMEngine

    placement_name, placement_path = resolve_ep_ll_placement(args)
    local_max_num_seqs = int(args.max_num_seqs)
    max_model_len = int(args.max_model_len)
    return LLMEngine(
        model=args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=local_max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        data_parallel_size=world,
        enforce_eager=bool(args.enforce_eager),
        moe_impl=args.moe_impl,
        moe_runtime_mode="owner_local_ep",
        moe_expert_placement=placement_name,
        moe_expert_placement_path=placement_path,
        moe_ll_m_max=m_max,
        moe_ll_overflow_policy=args.moe_ll_overflow_policy,
        moe_drop_policy="none",
        moe_drop_rate=0.0,
    )


def run_batch(engine, prompt_token_ids, max_new_tokens):
    from workshop.nanovllm_moe.services.sampling_params import SamplingParams

    engine.reset()
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    t0 = time.perf_counter()
    outs = engine.generate(prompt_token_ids, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    metrics = getattr(engine, "last_generation_metrics", {}) or {}
    return wall, outs, metrics


def parse_mmax_values(value: str) -> list[int]:
    items: list[int] = []
    for raw in value.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token == "auto":
            items.append(-1)
        else:
            items.append(int(token))
    if not items:
        raise SystemExit("need at least one --m-max-values entry")
    return items


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-30B-A3B"))
    p.add_argument("--dataset", default="/home/lzy/datasets/moe_benchmarks/longbench/extracted/data/passage_retrieval_en_e.jsonl")
    p.add_argument("--num-samples", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--min-prompt-tokens", type=int, default=500)
    p.add_argument("--max-prompt-tokens", type=int, default=5800)
    p.add_argument("--max-new-tokens", type=int, default=OFFICIAL_MAX_NEW_TOKENS)
    p.add_argument("--max-num-batched-tokens", type=int, default=6144)
    p.add_argument("--max-model-len", type=int, default=0,
                   help="0 => auto as max_prompt_tokens + max_new_tokens + 32")
    p.add_argument("--max-num-seqs", type=int, default=0,
                   help="0 => auto as ceil(batch_size / world_size) for owner_local_ep")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--warmup-batches", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--moe-impl", default="ep_ll_triton", choices=["ep_ll_triton", "ep_ll_torch"])
    p.add_argument("--enforce-eager", type=int, default=-1,
                   help="-1 => auto (ep_ll_triton uses cuda graph, ep_ll_torch uses eager)")
    p.add_argument("--moe-ll-overflow-policy", default="drop", choices=["drop", "error"])
    p.add_argument("--m-max-values", default="auto,1024,768,512")
    p.add_argument(
        "--expert-overlap-path",
        default=None,
        help="optional EP-HT overlap plan; for EP-LL we reuse its base placement JSON",
    )
    p.add_argument("--expert-placement-path", default=None)
    p.add_argument("--moe-expert-placement", default="contiguous")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rank, local_rank, world = setup_distributed()
    mmax_values = parse_mmax_values(args.m_max_values)
    if args.max_num_seqs <= 0:
        args.max_num_seqs = max(1, (args.batch_size + world - 1) // world)
    if args.max_model_len <= 0:
        args.max_model_len = args.max_prompt_tokens + args.max_new_tokens + 32
    if args.max_num_batched_tokens < args.max_model_len:
        raise SystemExit(
            f"max_num_batched_tokens={args.max_num_batched_tokens} must be >= "
            f"max_model_len={args.max_model_len}"
        )
    if args.enforce_eager < 0:
        args.enforce_eager = 0 if args.moe_impl == "ep_ll_triton" else 1

    if rank == 0:
        print(f"[ep_ll_mmax] dataset={args.dataset}")
        print(f"[ep_ll_mmax] impl={args.moe_impl} m_max_values={mmax_values}")
        print(f"[ep_ll_mmax] overflow_policy={args.moe_ll_overflow_policy}")
        print(
            f"[ep_ll_mmax] max_num_batched_tokens={args.max_num_batched_tokens} "
            f"max_model_len={args.max_model_len} max_num_seqs={args.max_num_seqs} "
            f"enforce_eager={args.enforce_eager}"
        )

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    rng = random.Random(args.seed)
    raw = [json.loads(line) for line in open(args.dataset) if line.strip()]
    rng.shuffle(raw)

    prompts = []
    skipped = 0
    for row in raw:
        ctx = row.get("context", "")
        inp = row.get("input", "")
        answers = row.get("answers", [])
        if not ctx or not inp or not answers:
            continue
        wrapped = OFFICIAL_PROMPT.format(context=ctx, input=inp)
        token_ids = tok.encode(wrapped, add_special_tokens=True)
        prompt_len = len(token_ids)
        if prompt_len < args.min_prompt_tokens or prompt_len > args.max_prompt_tokens:
            skipped += 1
            continue
        prompts.append(
            {
                "_id": row.get("_id", f"row-{len(prompts)}"),
                "token_ids": token_ids,
                "prompt_len": prompt_len,
                "reference": answers[0],
            }
        )
        if len(prompts) >= args.num_samples:
            break

    if rank == 0:
        if not prompts:
            print(f"[ep_ll_mmax] NO PROMPTS in [{args.min_prompt_tokens}, {args.max_prompt_tokens}]")
            return 1
        lens = [p["prompt_len"] for p in prompts]
        print(f"[ep_ll_mmax] kept {len(prompts)} prompts (skipped {skipped})")
        print(f"  prompt_len p50={sorted(lens)[len(lens)//2]} range=[{min(lens)}, {max(lens)}]")

    obj = [prompts]
    dist.broadcast_object_list(obj, src=0)
    prompts = obj[0]
    batches = [prompts[i:i + args.batch_size] for i in range(0, len(prompts), args.batch_size)]
    if rank == 0:
        print(f"[ep_ll_mmax] {len(batches)} batches × {args.batch_size}")

    jsonl_path = os.path.join(args.output_dir, "ep_ll_mmax_rows.jsonl")
    truncate_file(jsonl_path)
    dist.barrier()

    init_mmax = mmax_values[0]
    if args.warmup_batches > 0:
        t_load = time.perf_counter()
        engine = build_engine(args, world, m_max=init_mmax)
        if rank == 0:
            print(f"[ep_ll_mmax] warmup engine ready in {time.perf_counter() - t_load:.1f}s")
            print(f"[ep_ll_mmax] warmup {args.warmup_batches} batches with m_max={init_mmax}")
        for warm_idx in range(args.warmup_batches):
            batch = batches[warm_idx % len(batches)]
            ids = [p["token_ids"] for p in batch]
            _, _, metrics = run_batch(engine, ids, args.max_new_tokens)
            torch.cuda.synchronize()
            if rank == 0:
                print(
                    f"  warmup {warm_idx + 1}: "
                    f"prefill={float(metrics.get('prefill_time_s') or 0.0):.2f}s "
                    f"e2e={float(metrics.get('e2e_total_time_s') or 0.0):.2f}s"
                )
            dist.barrier()
        dist.barrier()
        del engine
        torch.cuda.empty_cache()
        dist.barrier()

    cells = []
    for m_max in mmax_values:
        label = "mmax_auto" if m_max <= 0 else f"mmax_{m_max}"
        cells.append({"label": label, "m_max_cfg": m_max})

    for cell_idx, cell in enumerate(cells):
        torch.cuda.empty_cache()
        dist.barrier()
        engine = build_engine(args, world, m_max=cell["m_max_cfg"])
        if rank == 0:
            print(f"\n[cell {cell_idx + 1}/{len(cells)}] {cell['label']}")
        for batch_idx, batch in enumerate(batches):
            ids = [p["token_ids"] for p in batch]
            wall, outputs, metrics = run_batch(engine, ids, args.max_new_tokens)
            torch.cuda.synchronize()

            batch_scores_official = []
            batch_scores_strict = []
            generations = []
            if rank == 0 and outputs:
                for src, out in zip(batch, outputs):
                    gen = (out or {}).get("text", "")
                    ref = src["reference"]
                    score_off = retrieval_score(gen, ref)
                    score_str = retrieval_score_strict(gen, ref)
                    batch_scores_official.append(score_off)
                    batch_scores_strict.append(score_str)
                    generations.append(
                        {
                            "_id": src["_id"],
                            "reference": ref,
                            "gen_text": gen,
                            "score_official": score_off,
                            "score_strict": score_str,
                            "prompt_len": src["prompt_len"],
                        }
                    )

            ep_ll_stats = metrics.get("ep_ll_stats") or {}
            ep_ll_stats_by_rank = metrics.get("ep_ll_stats_by_rank") or []
            row = {
                "cell_id": cell_idx,
                "label": cell["label"],
                "impl": args.moe_impl,
                "m_max_cfg": cell["m_max_cfg"],
                "overflow_policy": args.moe_ll_overflow_policy,
                "batch_id": batch_idx,
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
                "batch_acc_official": (
                    sum(batch_scores_official) / len(batch_scores_official)
                    if batch_scores_official else None
                ),
                "batch_acc_strict": (
                    sum(batch_scores_strict) / len(batch_scores_strict)
                    if batch_scores_strict else None
                ),
                "ep_ll_m_max": ep_ll_stats.get("m_max"),
                "ep_ll_observed_bucket_max": ep_ll_stats.get("observed_bucket_max"),
                "ep_ll_overflowed_replicas": ep_ll_stats.get("overflowed_replicas"),
                "ep_ll_overflow_replica_fraction": ep_ll_stats.get("overflow_replica_fraction"),
                "ep_ll_workspace_bytes_total": ep_ll_stats.get("workspace_bytes_total"),
                "ep_ll_total_routed_replicas": ep_ll_stats.get("routed_replicas"),
                "ep_ll_stats_by_rank": ep_ll_stats_by_rank,
                "generations": generations,
            }
            if rank == 0:
                write_jsonl(jsonl_path, row)
                print(
                    f"  b{batch_idx + 1}/{len(batches)} tok={row['total_prompt_tokens']:6d} "
                    f"prefill={float(row['prefill_time_s'] or 0.0):.2f}s "
                    f"e2e={float(row['e2e_total_time_s'] or 0.0):.2f}s "
                    f"acc(off)={float(row['batch_acc_official'] or 0.0):.3f} "
                    f"mmax={row['ep_ll_m_max']} "
                    f"overflow={float(row['ep_ll_overflow_replica_fraction'] or 0.0):.4f}"
                )
            dist.barrier()
        del engine
        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        rows = [json.loads(line) for line in open(jsonl_path) if line.strip()]
        per_cell = defaultdict(list)
        for row in rows:
            per_cell[row["label"]].append(row)

        summary = {
            "config": vars(args),
            "task": "passage_retrieval_en",
            "metric_official": "retrieval_score from THUDM/LongBench/metrics.py",
            "cells": {},
        }
        for label, cell_rows in per_cell.items():
            off_scores = [g["score_official"] for row in cell_rows for g in row["generations"]]
            str_scores = [g["score_strict"] for row in cell_rows for g in row["generations"]]
            summary["cells"][label] = {
                "impl": cell_rows[0]["impl"],
                "m_max_cfg": cell_rows[0]["m_max_cfg"],
                "m_max_effective": cell_rows[0]["ep_ll_m_max"],
                "overflow_policy": cell_rows[0]["overflow_policy"],
                "n_batches": len(cell_rows),
                "n_prompts": sum(row["batch_size"] for row in cell_rows),
                "prefill_time_s": agg_stats([row["prefill_time_s"] for row in cell_rows if row["prefill_time_s"] is not None]),
                "prefill_tok_s": agg_stats([row["prefill_tok_s"] for row in cell_rows if row["prefill_tok_s"] is not None]),
                "e2e_total_time_s": agg_stats([row["e2e_total_time_s"] for row in cell_rows if row["e2e_total_time_s"] is not None]),
                "decode_time_s": agg_stats([row["decode_time_s"] for row in cell_rows if row["decode_time_s"] is not None]),
                "decode_tok_s": agg_stats([row["decode_tok_s"] for row in cell_rows if row["decode_tok_s"] is not None]),
                "ep_ll_observed_bucket_max": agg_stats([row["ep_ll_observed_bucket_max"] for row in cell_rows if row["ep_ll_observed_bucket_max"] is not None]),
                "ep_ll_overflow_replica_fraction": agg_stats([row["ep_ll_overflow_replica_fraction"] for row in cell_rows if row["ep_ll_overflow_replica_fraction"] is not None]),
                "ep_ll_workspace_bytes_total": cell_rows[0]["ep_ll_workspace_bytes_total"],
                "acc_official": {
                    "mean": (sum(off_scores) / len(off_scores)) if off_scores else None,
                    "sem": (statistics.stdev(off_scores) / (len(off_scores) ** 0.5)) if len(off_scores) > 1 else 0.0,
                    "n": len(off_scores),
                },
                "acc_strict": {
                    "mean": (sum(str_scores) / len(str_scores)) if str_scores else None,
                    "sem": (statistics.stdev(str_scores) / (len(str_scores) ** 0.5)) if len(str_scores) > 1 else 0.0,
                    "n": len(str_scores),
                },
            }

        out_path = os.path.join(args.output_dir, "ep_ll_mmax_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[ep_ll_mmax] wrote {out_path}")

        print(f"\n{'label':<16}{'score(off)':>11}{'score(str)':>11}{'e2e_s':>9}{'prefill_tok/s':>15}{'overflow':>11}{'M_max':>8}")
        for label in sorted(summary["cells"].keys()):
            cell = summary["cells"][label]
            print(
                f"{label:<16}"
                f"{float(cell['acc_official']['mean'] or 0.0):>11.3f}"
                f"{float(cell['acc_strict']['mean'] or 0.0):>11.3f}"
                f"{float(cell['e2e_total_time_s']['mean'] or 0.0):>9.2f}"
                f"{float(cell['prefill_tok_s']['mean'] or 0.0):>15.0f}"
                f"{float(cell['ep_ll_overflow_replica_fraction']['mean'] or 0.0):>11.4f}"
                f"{str(cell['m_max_cfg']):>8}"
            )

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
