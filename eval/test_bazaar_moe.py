"""AIME24 evaluation for the MoE engine via the bazaar combine.

Mirrors `eval/test_bazaar_v25.py` (which evaluates Qwen3-4B via the v2_5 bazaar)
but uses `bazaar/nanovllm_moe.py` and supports BOTH:

  - Single-process (small/dense or trimmed MoE):
      python -m eval.test_bazaar_moe \\
          --model-path /home/yyx/models/Qwen3-30B-A3B \\
          --moe-impl triton --num-layers 4

  - Multi-rank EP via mp.spawn (full Qwen3-30B-A3B on 8 GPUs):
      python -m eval.test_bazaar_moe \\
          --model-path /home/yyx/models/Qwen3-30B-A3B \\
          --world-size 8 --tp-size 1 \\
          --moe-impl ep_ll_triton --enforce-eager 0

  - TP × EP composition (TP=2 × EP=4 on 8 GPUs):
      python -m eval.test_bazaar_moe \\
          --world-size 8 --tp-size 2 \\
          --moe-impl ep_ll_triton --enforce-eager 0

Output:
  eval_results/aime24_<moe_impl>_tp<TP>_ep<EP>_layers<L>.{json,txt}

Defaults to a 1-problem smoke test (`--num-problems 1`) so it terminates quickly;
pass `--num-problems 30` for the full AIME24 split.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pprint import pformat

import datasets
import torch
from transformers import AutoTokenizer

# Add repo root for `bazaar.*` imports when run as `python -m eval.test_bazaar_moe`.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bazaar.nanovllm_moe import combine, spawn_eval
from eval.utils import evaluate


# ---------------------------------------------------------------------------
# Dataset loading — datasets/aime24.parquet has columns:
#   id, solution, answer (str), url, question, prompt (list of chat messages)
# Same shape `eval/test_bazaar_v25.py` consumes.
# ---------------------------------------------------------------------------

def load_aime24(path: str):
    """Load the parquet split (single 'train' split)."""
    return datasets.load_dataset("parquet", data_files=path, split="train")


def build_prompts(rows, tokenizer, *, num_problems: int) -> list[dict]:
    """Apply the chat template to each row's prebuilt `prompt` (list of messages).
    Returns list of {raw_prompt, question, answer}."""
    items = []
    n = min(num_problems, len(rows))
    for i in range(n):
        row = rows[i]
        prompt = row["prompt"]  # list of {role, content} dicts
        raw_prompt = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False,
        )
        items.append({
            "raw_prompt": raw_prompt,
            "question": row["question"],
            "answer": str(row["answer"]),
            "id": row["id"],
        })
    return items


# ---------------------------------------------------------------------------
# The actual eval body — same code path for single-process and SPMD.
# ---------------------------------------------------------------------------

def _generate_and_score(
    engine,
    SamplingParams,
    *,
    model_path: str,
    dataset_path: str,
    num_problems: int,
    temperature: float,
    top_k: int,
    top_p: float,
    max_tokens: int,
    out_prefix: str,
    rank: int = 0,
):
    """Run AIME24 evaluation. Only rank 0 writes results."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    rows = load_aime24(dataset_path)
    items = build_prompts(rows, tokenizer, num_problems=num_problems)
    prompts = [it["raw_prompt"] for it in items]

    sampling_params = SamplingParams(
        temperature=temperature, top_k=top_k, top_p=top_p, max_tokens=max_tokens,
    )

    if rank == 0:
        print(f"[rank 0] generating {len(prompts)} AIME24 problems "
              f"(max_tokens={max_tokens}, temperature={temperature})", flush=True)

    outputs = engine.generate(prompts, sampling_params, use_tqdm=(rank == 0))

    # All ranks ran .generate() collectively, but only rank 0 has token_ids back
    # (in the multi-rank case, the engine's sampling is broadcast within TP groups
    # so all ranks of an EP slot see the same tokens; we still only write rank 0).
    if rank != 0:
        return

    os.makedirs("eval_results", exist_ok=True)
    json_path = f"eval_results/{out_prefix}.json"
    txt_path = f"eval_results/{out_prefix}.txt"

    total_score = 0.0
    total_gen_tokens = 0
    per_item = []

    with open(txt_path, "w") as ftxt:
        for idx, (item, out) in enumerate(zip(items, outputs)):
            gen_text = out["text"]
            gen_ids = out["token_ids"]
            score, parsed = evaluate(gen_text, item["answer"])
            total_score += float(score)
            total_gen_tokens += len(gen_ids)

            per_item.append({
                "idx": idx,
                "id": item["id"],
                "question": item["question"],
                "ground_truth": item["answer"],
                "parsed_answer": str(parsed) if parsed is not None else None,
                "score": float(score),
                "num_generated_tokens": len(gen_ids),
                "generated_text": gen_text,
            })

            ftxt.write(
                f"=== problem {idx} (id={item['id']}) ===\n"
                f"PROMPT:\n{item['raw_prompt']}\n\n"
                f"GENERATED:\n{gen_text}\n\n"
                f"GROUND TRUTH: {item['answer']}\n"
                f"PARSED:       {parsed}\n"
                f"SCORE:        {score}\n"
                f"NUM TOKENS:   {len(gen_ids)}\n"
                + ("=" * 100) + "\n\n"
            )

    summary = {
        "model_path": model_path,
        "num_problems": len(items),
        "total_score": total_score,
        "average_score": total_score / max(1, len(items)),
        "total_generated_tokens": total_gen_tokens,
        "average_generated_tokens": total_gen_tokens / max(1, len(items)),
    }
    with open(json_path, "w") as fjson:
        json.dump({"summary": summary, "per_item": per_item}, fjson, indent=2)

    print()
    print("=" * 60)
    print("AIME24 evaluation complete")
    print("=" * 60)
    print(pformat(summary))
    print(f"  results: {json_path}")
    print(f"  rollouts: {txt_path}")


# ---------------------------------------------------------------------------
# Entry points: single-process vs SPMD
# ---------------------------------------------------------------------------

def run_single_process(args):
    """Direct combine() — like test_bazaar_v25.py."""
    engine_kwargs = _engine_kwargs_from_args(args)
    engine, SP = combine(**engine_kwargs)
    _generate_and_score(
        engine, SP,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        num_problems=args.num_problems,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        out_prefix=_out_prefix(args),
        rank=0,
    )


def _spmd_aime_eval(engine, SamplingParams, rank, world_size, *, payload):
    """Top-level function (so it's picklable for mp.spawn) that runs AIME24 eval.
    `payload` is a dict of all the per-eval kwargs."""
    _generate_and_score(engine, SamplingParams, rank=rank, **payload)


def run_distributed(args):
    """spawn N workers; each constructs the engine and runs the eval body."""
    engine_kwargs = _engine_kwargs_from_args(args)
    payload = dict(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        num_problems=args.num_problems,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        out_prefix=_out_prefix(args),
    )

    spawn_eval(
        world_size=args.world_size,
        engine_kwargs=engine_kwargs,
        eval_fn=_spmd_aime_eval,
        eval_fn_kwargs={"payload": payload},
    )


def _engine_kwargs_from_args(args) -> dict:
    kw = dict(
        model=args.model_path,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tp_size,
        enforce_eager=bool(args.enforce_eager),
        moe_impl=args.moe_impl,
        moe_block_size_m=args.moe_block_size_m,
    )
    if args.num_layers > 0:
        kw["num_hidden_layers_override"] = args.num_layers
    if args.moe_ll_m_max > 0:
        kw["moe_ll_m_max"] = args.moe_ll_m_max
    return kw


def _out_prefix(args) -> str:
    ep_size = max(1, args.world_size // max(1, args.tp_size))
    layer_tag = f"_layers{args.num_layers}" if args.num_layers > 0 else ""
    return (
        f"aime24_{args.moe_impl}"
        f"_tp{args.tp_size}_ep{ep_size}"
        f"_eager{int(bool(args.enforce_eager))}"
        f"{layer_tag}"
    )


def main():
    parser = argparse.ArgumentParser()
    # Model / dataset
    parser.add_argument("--model-path", type=str,
                        default="/home/yyx/models/Qwen3-30B-A3B")
    parser.add_argument("--dataset-path", type=str, default="datasets/aime24.parquet")
    parser.add_argument("--num-problems", type=int, default=1,
                        help="how many AIME24 problems to evaluate (default 1 for smoke test; 30 = full)")

    # Engine / parallelism
    parser.add_argument("--world-size", type=int, default=1,
                        help="total ranks; 1 = single-process, >1 = mp.spawn SPMD")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="tensor-parallel size within each EP slot (ep_size = world_size / tp_size)")
    parser.add_argument("--moe-impl", type=str, default="triton",
                        choices=["torch", "triton", "ep_ll_torch", "ep_ll_triton", "ep_ht"])
    parser.add_argument("--enforce-eager", type=int, default=1,
                        help="1 = no cuda graph, 0 = capture (only valid for triton/ep_ll_triton)")
    parser.add_argument("--moe-block-size-m", type=int, default=64)
    parser.add_argument("--moe-ll-m-max", type=int, default=-1,
                        help="EP-LL bucket size; -1 = auto")
    parser.add_argument("--num-layers", type=int, default=-1,
                        help=">0 trims to the first N decoder layers (for smoke testing on small GPUs)")

    # Memory / scheduling
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)

    # Sampling
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2048)

    args = parser.parse_args()

    if args.world_size == 1:
        run_single_process(args)
    else:
        if args.world_size % args.tp_size != 0:
            raise SystemExit(
                f"world_size={args.world_size} must be divisible by tp_size={args.tp_size}"
            )
        run_distributed(args)


if __name__ == "__main__":
    main()
