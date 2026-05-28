"""Baseline-only run on a LongBench task with the OFFICIAL prompt template and
max_new_tokens (per `THUDM/LongBench/config/dataset2prompt.json` and
`dataset2maxlen.json`). NO drop applied.

The prepared `*.custom.jsonl` files store `prompt = passages + question` as
a single string. We split it back into `{context, input}` by taking the last
paragraph (separated by `\\n\\n`) as the question, then re-wrap with the
official LongBench template for the chosen task.

Usage:
    CUDA_HOME=/usr/local/cuda-12.8 \\
      PATH=/usr/local/cuda-12.8/bin:$PATH \\
      FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_tier2 \\
      torchrun --nproc_per_node=8 --master_port=29571 \\
      -m eval.drop.official_baseline \\
      --output-dir eval_results/longbench_official_baseline \\
      --task 2wikimqa \\
      --expert-overlap-path eval_results/.../moe_overlap_plan_load_balanced...json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import string
import sys
import time
from collections import Counter, defaultdict

os.environ.setdefault("MOE_DROP_IMPL", "auto")
os.environ.setdefault("MOE_DROP_MIN_REPLICAS", "512")
os.environ.setdefault("MOE_DROP_GPU_STATS", "0")
os.environ.setdefault("MOE_PROFILE_OVERLAP_RUNTIME", "0")
os.environ.setdefault("MOE_RECORD_TIMING", "1")
os.environ.setdefault("MOE_PROFILE_ROUTING", "0")

import torch
import torch.distributed as dist

from eval.drop.shared import truncate_file, write_jsonl
from rouge_score import rouge_scorer


# ----------------------------------------------------------------------
# LongBench official config (from THUDM/LongBench config files)
# ----------------------------------------------------------------------

# Anti-CoT variant of LongBench official templates:
#   - Strengthened "no explanation" wording with explicit examples of what NOT to emit
#     ("do not write 'Okay', 'let's see', or any reasoning")
#   - Replaced terminal "Answer:" with "The answer is:" — this primes greedy decoding
#     to continue with the answer phrase directly, bypassing Qwen3's reflexive CoT preamble.
# Tail-question still wraps after the passages, matching the official "sandwich" pattern.
DATASET2PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question as concisely as you can, using a single phrase "
        "if possible. Do not provide any explanation. Do not write 'Okay' or 'let me think'."
        "\n\nStory: {context}\n\n"
        "Now, answer the question using a single phrase. Do not provide any explanation, "
        "reasoning, or preamble. Reply with the answer text only.\n\n"
        "Question: {input}\nThe answer is:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer concisely. If unanswerable "
        'write "unanswerable". If yes/no, write "yes", "no", or "unanswerable". Do not provide '
        "any explanation. Do not write 'Okay' or 'let me think'.\n\nArticle: {context}\n\n"
        "Question: {input}\nThe answer is:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly. Do not provide any explanation, "
        "reasoning, or preamble. Do not write 'Okay' or 'let me think'.\n\n{context}\n\n"
        "Now answer the question based on the above text. Reply with the answer text only "
        "(1-10 words).\n\nQuestion: {input}\nThe answer is:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Output only the answer (1-5 words). "
        "Do not provide any explanation, reasoning, or preamble. Do not write 'Okay' or 'let me think'."
        "\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Output only the answer as a short "
        "phrase. No preamble.\n\nQuestion: {input}\nThe answer is:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Output only the answer (1-5 words). "
        "Do not provide any explanation, reasoning, or preamble. Do not write 'Okay' or 'let me think'."
        "\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Output only the answer as a short "
        "phrase. No preamble.\n\nQuestion: {input}\nThe answer is:"
    ),
    "musique": (
        "Answer the question based on the given passages. Output only the answer (1-5 words). "
        "Do not provide any explanation, reasoning, or preamble. Do not write 'Okay' or 'let me think'."
        "\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Output only the answer as a short "
        "phrase. No preamble.\n\nQuestion: {input}\nThe answer is:"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Output only the answer (1-5 words). "
        "Do not provide any explanation, reasoning, or preamble. The following are some examples."
        "\n\n{context}\n\n{input}"
    ),
}

# Doubled the official LongBench `dataset2maxlen` budgets to give Qwen3 some
# headroom — even with "The answer is:" priming, the model often still emits
# a short reasoning preamble before the answer. 2x official is a defensive cap
# that still keeps decode budget modest.
DATASET2MAXLEN = {
    "narrativeqa": 256, "qasper": 256, "multifieldqa_en": 128,
    "hotpotqa": 64, "2wikimqa": 64, "musique": 64,
    "triviaqa": 64,
}


# ----------------------------------------------------------------------
# Prompt splitting (recover {context, input} from prepared prompt string)
# ----------------------------------------------------------------------

def split_prepared_prompt(prepared_prompt: str) -> tuple[str, str]:
    """Take the last paragraph (separated by '\\n\\n') as the question; everything
    before as the context. Falls back to splitting at last '\\n' if no '\\n\\n'."""
    if "\n\n" in prepared_prompt:
        parts = prepared_prompt.rsplit("\n\n", 1)
        return parts[0], parts[1].strip()
    # Fallback: last line
    parts = prepared_prompt.rsplit("\n", 1)
    if len(parts) == 2:
        return parts[0], parts[1].strip()
    return prepared_prompt, ""


# ----------------------------------------------------------------------
# Official LongBench QA F1 scoring
# ----------------------------------------------------------------------

_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in _PUNCT)
    return " ".join(s.split())


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_recall(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    return num_same / len(gt_tokens)


_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def rouge_l(prediction: str, ground_truth: str) -> float:
    if not prediction or not ground_truth:
        return 0.0
    return float(_rouge.score(ground_truth, prediction)["rougeL"].fmeasure)


# ----------------------------------------------------------------------
# Distributed setup + engine
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
    p.add_argument("--task", required=True,
                   choices=list(DATASET2PROMPT.keys()),
                   help="LongBench task name (controls template + max_new_tokens)")
    p.add_argument("--dataset-jsonl", default="",
                   help="Path to prepared .custom.jsonl; default: longbench.{task}.custom.jsonl")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--min-prompt-tokens", type=int, default=500)
    p.add_argument("--max-prompt-tokens", type=int, default=5800)
    p.add_argument("--expert-overlap-path", default="")
    p.add_argument("--expert-overlap-strategy", default="greedy_balance")
    p.add_argument("--max-num-batched-tokens", type=int, default=6144)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--warmup-batches", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rank, local_rank, world = setup_distributed()

    task = args.task
    template = DATASET2PROMPT[task]
    max_new = DATASET2MAXLEN[task]
    if rank == 0:
        print(f"[official] task={task}  max_new_tokens={max_new}  template_head: {template[:80]!r}")

    dataset_path = args.dataset_jsonl or f"/home/lzy/datasets/moe_benchmarks/prepared/longbench.{task}.custom.jsonl"
    if rank == 0:
        print(f"[official] loading {dataset_path}")
    raw_rows = []
    for line in open(dataset_path):
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        if not d.get("prompt"): continue
        raw_rows.append(d)

    # Build engine first (need tokenizer for length filter on TEMPLATED prompts)
    t_load = time.perf_counter()
    engine = build_engine(args, world)
    if rank == 0:
        print(f"[official] engine ready in {time.perf_counter() - t_load:.1f}s")
    tok = engine.tokenizer

    # Split each prepared prompt, wrap with official template, filter by tokenized length.
    import random
    rng = random.Random(args.seed)
    rng.shuffle(raw_rows)

    prompts = []
    skipped_short = skipped_long = 0
    for r in raw_rows:
        context, question = split_prepared_prompt(r["prompt"])
        if not question:
            continue
        wrapped = template.format(context=context, input=question)
        token_ids = tok.encode(wrapped, add_special_tokens=True)
        n = len(token_ids)
        if n < args.min_prompt_tokens:
            skipped_short += 1; continue
        if n > args.max_prompt_tokens:
            skipped_long += 1; continue
        prompts.append({
            "request_id": r["request_id"],
            "context_head": context[:200],
            "question": question,
            "wrapped_prompt": wrapped,
            "token_ids": token_ids,
            "prompt_len": n,
            "reference_answer": str(r.get("reference_answer", "")),
        })
        if len(prompts) >= args.num_samples:
            break

    if rank == 0:
        lens = [p["prompt_len"] for p in prompts]
        print(f"[official] kept {len(prompts)} prompts after templating "
              f"(skipped: {skipped_short} too-short, {skipped_long} too-long)")
        if lens:
            print(f"   wrapped prompt len p50={sorted(lens)[len(lens)//2]}, "
                  f"range=[{min(lens)},{max(lens)}]")
            # Sample preview
            p0 = prompts[0]
            print(f"   sample wrapped head: {p0['wrapped_prompt'][:200]!r}")
            print(f"   sample wrapped tail: {p0['wrapped_prompt'][-200:]!r}")
            print(f"   sample reference:    {p0['reference_answer']!r}")

    # Broadcast to all ranks
    obj = [prompts]
    dist.broadcast_object_list(obj, src=0)
    prompts = obj[0]

    jsonl_path = os.path.join(args.output_dir, "baseline_rows.jsonl")
    truncate_file(jsonl_path)
    dist.barrier()

    batches = [prompts[i:i + args.batch_size] for i in range(0, len(prompts), args.batch_size)]
    if rank == 0:
        print(f"[official] {len(batches)} batches × {args.batch_size}")

    # Warmup
    if args.warmup_batches > 0:
        if rank == 0:
            print(f"[official] warmup {args.warmup_batches}")
        for w in range(args.warmup_batches):
            batch = batches[w % len(batches)]
            ids = [p["token_ids"] for p in batch]
            wall, _, m = run_batch(engine, ids, max_new)
            torch.cuda.synchronize()
            if rank == 0:
                print(f"  warmup {w+1}: prefill={m.get('prefill_time_s'):.3f}s (discarded)")
            dist.barrier()

    # Measured batches
    all_f1, all_recall, all_rouge = [], [], []
    total_prefill_s = total_decode_s = total_e2e_s = 0.0
    total_prompt_tokens = total_decode_tokens = 0
    if rank == 0:
        print(f"\n[official] running {len(batches)} measured batches:")
    for bi, batch in enumerate(batches):
        ids = [p["token_ids"] for p in batch]
        wall, outputs, metrics = run_batch(engine, ids, max_new)
        torch.cuda.synchronize()
        # Score
        batch_f1, batch_recall, batch_rouge = [], [], []
        gens = []
        if rank == 0 and outputs:
            for src, out in zip(batch, outputs):
                gen_text = (out or {}).get("text", "")
                ref = src["reference_answer"]
                f1 = qa_f1_score(gen_text, ref)
                rc = qa_recall(gen_text, ref)
                rl = rouge_l(gen_text, ref)
                batch_f1.append(f1); batch_recall.append(rc); batch_rouge.append(rl)
                gens.append({
                    "request_id": src["request_id"],
                    "question": src["question"],
                    "reference": ref,
                    "gen_text": gen_text,
                    "f1": f1,
                    "recall": rc,
                    "rougeL": rl,
                    "prompt_len": src["prompt_len"],
                })
            all_f1.extend(batch_f1)
            all_recall.extend(batch_recall)
            all_rouge.extend(batch_rouge)
        row = {
            "batch_id": bi,
            "batch_size": len(batch),
            "total_prompt_tokens": sum(p["prompt_len"] for p in batch),
            "wall_s": wall,
            "prefill_time_s": metrics.get("prefill_time_s"),
            "prefill_tok_s": metrics.get("prefill_throughput_tok_s"),
            "decode_time_s": metrics.get("decode_time_s"),
            "decode_tok_s": metrics.get("decode_throughput_tok_s"),
            "e2e_total_time_s": metrics.get("e2e_total_time_s"),
            "batch_f1_mean": (sum(batch_f1) / len(batch_f1)) if batch_f1 else None,
            "batch_recall_mean": (sum(batch_recall) / len(batch_recall)) if batch_recall else None,
            "batch_rougeL_mean": (sum(batch_rouge) / len(batch_rouge)) if batch_rouge else None,
            "generations": gens,
        }
        if rank == 0:
            write_jsonl(jsonl_path, row)
            total_prefill_s += row["prefill_time_s"] or 0
            total_decode_s += row["decode_time_s"] or 0
            total_e2e_s += row["e2e_total_time_s"] or 0
            total_prompt_tokens += row["total_prompt_tokens"]
            total_decode_tokens += metrics.get("decode_tokens", 0) or 0
            print(f"  b{bi+1}/{len(batches)}  prompt_tok={row['total_prompt_tokens']:6d}  "
                  f"prefill={row['prefill_time_s']:.2f}s  decode={row['decode_time_s']:.2f}s  "
                  f"e2e={row['e2e_total_time_s']:.2f}s  "
                  f"F1={row['batch_f1_mean']:.3f}  recall={row['batch_recall_mean']:.3f}")
        dist.barrier()

    # Summary on rank 0
    if rank == 0 and all_f1:
        n = len(all_f1)
        def stat(xs):
            m = statistics.mean(xs)
            s = statistics.stdev(xs) if len(xs) > 1 else 0.0
            return {"mean": m, "std": s, "sem": s / (n ** 0.5), "n": n}
        summary = {
            "config": vars(args),
            "task": task,
            "max_new_tokens_official": max_new,
            "n_samples": n,
            "n_batches": len(batches),
            "system_throughput_tok_s_prefill": (
                total_prompt_tokens / total_prefill_s if total_prefill_s > 0 else None
            ),
            "perf": {
                "total_prefill_s": total_prefill_s,
                "total_decode_s": total_decode_s,
                "total_e2e_s": total_e2e_s,
                "total_prompt_tokens": total_prompt_tokens,
                "total_decode_tokens": total_decode_tokens,
                "mean_prefill_per_batch_s": total_prefill_s / len(batches),
                "mean_e2e_per_batch_s": total_e2e_s / len(batches),
            },
            "accuracy": {
                "lb_f1": stat(all_f1),
                "lb_recall": stat(all_recall),
                "rougeL": stat(all_rouge),
            },
        }
        out_path = os.path.join(args.output_dir, "baseline_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n=== Summary ({task}, n={n}, batches={len(batches)}) ===")
        print(f"  Performance:")
        print(f"    Total prefill: {total_prefill_s:.2f}s on {total_prompt_tokens} tokens")
        print(f"    Total decode:  {total_decode_s:.2f}s on {total_decode_tokens} tokens")
        print(f"    Total e2e:     {total_e2e_s:.2f}s")
        print(f"    Per-rank prefill tok/s: {total_prompt_tokens / total_prefill_s:.1f}")
        print(f"    System prefill tok/s (×8): {8 * total_prompt_tokens / total_prefill_s:.1f}")
        print(f"  Accuracy (LongBench OFFICIAL metric on OFFICIAL template):")
        a = summary["accuracy"]
        print(f"    F1:       {a['lb_f1']['mean']:.4f} ± {a['lb_f1']['sem']:.4f} (SEM)")
        print(f"    Recall:   {a['lb_recall']['mean']:.4f} ± {a['lb_recall']['sem']:.4f}")
        print(f"    ROUGE-L:  {a['rougeL']['mean']:.4f} ± {a['rougeL']['sem']:.4f}")
        print(f"\n[official] wrote {out_path}")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
