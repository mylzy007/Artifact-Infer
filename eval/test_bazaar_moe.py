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
import time
from pathlib import Path
from pprint import pformat

import datasets
import torch
from transformers import AutoTokenizer

# Add repo root for `bazaar.*` imports when run as `python -m eval.test_bazaar_moe`.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bazaar.nanovllm_moe import combine, spawn_eval
from eval.utils import evaluate, evaluate_answer_first
from workshop.nanovllm_moe.services.utils import overlap_runtime_stats, routing_profile


_ANSWER_FIRST_PROMPT_PREFIX = (
    "Solve the following math word problem.\n\n"
    "On the first line, output exactly:\n"
    "Final answer: \\boxed{your_answer}\n\n"
    "Then optionally write at most 2 short checking lines.\n"
    "Do not output any other boxed expression.\n\n"
    "Question:\n"
)

# Phase 4 v2: short reasoning first, then a final boxed line. Keeps the
# answer-extraction surface (`Final answer: \boxed{...}`) so the existing
# evaluator works, but forces the model to think briefly before committing.
_REASONING_BRIEF_PROMPT_PREFIX = (
    "Solve the following math word problem step by step.\n\n"
    "Think through it in at most 4 short sentences (no more). Be concise.\n"
    "Then, on a new final line, write exactly:\n"
    "Final answer: \\boxed{your_answer}\n\n"
    "Rules:\n"
    "- The Final answer line MUST be the last line of your response.\n"
    "- Do not output any other boxed expression.\n"
    "- Do not exceed 4 sentences of reasoning.\n\n"
    "Question:\n"
)


def _wrap_prompt(raw_prompt: str, style: str, question: str | None) -> str:
    if style == "answer_only":
        return raw_prompt
    if style == "answer_first":
        # Re-wrap from `question` if available (cleaner). Otherwise prepend the
        # instruction block to whatever raw_prompt the dataset built.
        if question:
            return _ANSWER_FIRST_PROMPT_PREFIX + question.strip()
        return _ANSWER_FIRST_PROMPT_PREFIX + raw_prompt.strip()
    raise ValueError(f"unknown prompt style {style!r}")


# ---------------------------------------------------------------------------
# Dataset loading — datasets/aime24.parquet has columns:
#   id, solution, answer (str), url, question, prompt (list of chat messages)
# Same shape `eval/test_bazaar_v25.py` consumes.
# ---------------------------------------------------------------------------

def load_aime24(path: str):
    """Load the parquet split (single 'train' split)."""
    return datasets.load_dataset("parquet", data_files=path, split="train")


def build_prompts(rows, tokenizer, *, num_problems: int, prompt_style: str = "answer_only") -> list[dict]:
    """Apply the chat template to each row's prebuilt `prompt` (list of messages).
    Returns list of {raw_prompt, question, answer}.

    When `prompt_style == "answer_first"`, replace the user content with the
    Phase 4 answer-first wrapper before applying the chat template.
    """
    items = []
    n = min(num_problems, len(rows))
    for i in range(n):
        row = rows[i]
        prompt = row["prompt"]  # list of {role, content} dicts
        question = row.get("raw_question") or row["question"]
        if prompt_style in ("answer_first", "reasoning_brief"):
            prefix = (
                _ANSWER_FIRST_PROMPT_PREFIX
                if prompt_style == "answer_first"
                else _REASONING_BRIEF_PROMPT_PREFIX
            )
            prompt = [{"role": "user", "content": prefix + str(question).strip()}]
            # Disable Qwen3 thinking mode: we don't want a long <think> block
            # burning max_tokens before the final boxed answer is written.
            try:
                raw_prompt = tokenizer.apply_chat_template(
                    prompt, add_generation_prompt=True, tokenize=False, enable_thinking=False,
                )
            except TypeError:
                raw_prompt = tokenizer.apply_chat_template(
                    prompt, add_generation_prompt=True, tokenize=False,
                )
        else:
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
    output_dir: str,
    profile_metadata: dict | None = None,
    rank: int = 0,
    prompt_style: str = "answer_only",
    score_mode: str = "default",
):
    """Run AIME24 evaluation. Only rank 0 writes results."""
    if profile_metadata:
        routing_profile.set_metadata(**profile_metadata)
        overlap_runtime_stats.set_metadata(**profile_metadata)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    rows = load_aime24(dataset_path)
    items = build_prompts(rows, tokenizer, num_problems=num_problems, prompt_style=prompt_style)
    prompts = [it["raw_prompt"] for it in items]

    sampling_params = SamplingParams(
        temperature=temperature, top_k=top_k, top_p=top_p, max_tokens=max_tokens,
    )

    if rank == 0:
        print(f"[rank 0] generating {len(prompts)} AIME24 problems "
              f"(max_tokens={max_tokens}, temperature={temperature})", flush=True)

    outputs = engine.generate(prompts, sampling_params, use_tqdm=(rank == 0))
    output_path = Path(output_dir)
    profile_path = routing_profile.save_profile(str(output_path))
    overlap_stats_path = overlap_runtime_stats.save_profile(str(output_path))

    # All ranks ran .generate() collectively, but only rank 0 has token_ids back
    # (in the multi-rank case, the engine's sampling is broadcast within TP groups
    # so all ranks of an EP slot see the same tokens; we still only write rank 0).
    if rank != 0:
        return

    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{out_prefix}.json"
    txt_path = output_path / f"{out_prefix}.txt"

    total_score = 0.0
    total_gen_tokens = 0
    per_item = []

    with open(txt_path, "w") as ftxt:
        for idx, (item, out) in enumerate(zip(items, outputs)):
            gen_text = out["text"]
            gen_ids = out["token_ids"]
            if score_mode == "answer_first":
                score, parsed = evaluate_answer_first(gen_text, item["answer"])
            else:
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
    generation_metrics = getattr(engine, "last_generation_metrics", {})
    if generation_metrics:
        summary["generation_metrics"] = generation_metrics
    if profile_path:
        summary["routing_profile_path"] = profile_path
    if overlap_stats_path:
        summary["overlap_runtime_stats_path"] = overlap_stats_path
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
        output_dir=args.output_dir,
        profile_metadata=_profile_metadata(args),
        rank=0,
        prompt_style=args.prompt_style,
        score_mode=args.score_mode,
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
        output_dir=args.output_dir,
        profile_metadata=_profile_metadata(args),
        prompt_style=args.prompt_style,
        score_mode=args.score_mode,
    )

    spawn_eval(
        world_size=args.world_size,
        engine_kwargs=engine_kwargs,
        eval_fn=_spmd_aime_eval,
        eval_fn_kwargs={"payload": payload},
        master_port=str(args.master_port),
    )


def _engine_kwargs_from_args(args) -> dict:
    kw = dict(
        model=args.model_path,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tp_size,
        data_parallel_size=args.data_parallel_size,
        enforce_eager=bool(args.enforce_eager),
        moe_impl=args.moe_impl,
        moe_runtime_mode=args.moe_runtime_mode,
        moe_block_size_m=args.moe_block_size_m,
        moe_expert_placement=args.moe_expert_placement,
        moe_expert_placement_seed=args.moe_expert_placement_seed,
        moe_expert_placement_path=args.moe_expert_placement_path,
        moe_expert_overlap_enabled=bool(args.moe_expert_overlap_path),
        moe_expert_overlap_path=args.moe_expert_overlap_path,
        moe_expert_overlap_strategy=args.moe_expert_overlap_strategy,
        moe_ll_overflow_policy=args.moe_ll_overflow_policy,
        moe_drop_policy=args.moe_drop_policy,
        moe_drop_rate=args.moe_drop_rate,
        moe_drop_seed=args.moe_drop_seed,
        moe_router_keff=args.moe_router_keff,
    )
    if args.num_layers > 0:
        kw["num_hidden_layers_override"] = args.num_layers
    if args.moe_ll_m_max > 0:
        kw["moe_ll_m_max"] = args.moe_ll_m_max
    return kw


def _out_prefix(args) -> str:
    ep_size = (
        args.world_size
        if args.moe_runtime_mode in ("vllm_dp_ep", "owner_local_ep")
        else max(1, args.world_size // max(1, args.tp_size))
    )
    layer_tag = f"_layers{args.num_layers}" if args.num_layers > 0 else ""
    dataset_label = args.dataset_label
    if not dataset_label:
        dataset_label = Path(args.dataset_path).stem
    dataset_label = "".join(
        ch if ch.isalnum() or ch in "._-" else "_"
        for ch in dataset_label
    ).strip("_") or "dataset"
    return (
        f"{dataset_label}_{args.moe_impl}"
        f"_{args.moe_expert_placement}"
        f"_ov{str(args.moe_expert_overlap_value).replace('.', 'p')}"
        f"_{args.moe_expert_overlap_strategy}"
        f"_tp{args.tp_size}_ep{ep_size}"
        f"_dp{args.data_parallel_size}"
        f"_eager{int(bool(args.enforce_eager))}"
        f"{layer_tag}"
    )


def _profile_metadata(args) -> dict:
    ep_size = (
        args.world_size
        if args.moe_runtime_mode in ("vllm_dp_ep", "owner_local_ep")
        else max(1, args.world_size // max(1, args.tp_size))
    )
    return {
        "model_path": args.model_path,
        "moe_impl": args.moe_impl,
        "runtime_mode": args.moe_runtime_mode,
        "world_size": args.world_size,
        "tp_size": args.tp_size,
        "dp_size": args.data_parallel_size,
        "ep_size": ep_size,
        "source_definition": (
            "dp_leader"
            if args.moe_runtime_mode == "vllm_dp_ep"
            else "owner_rank"
            if args.moe_runtime_mode == "owner_local_ep"
            else "ep_rank"
        ),
        "num_layers": args.num_layers if args.num_layers > 0 else None,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "num_prompts": args.num_problems,
        "placement_used_during_profiling": args.moe_expert_placement,
        "placement_json_path": args.moe_expert_placement_path,
        "overlap": args.moe_expert_overlap_value,
        "overlap_plan_json_path": args.moe_expert_overlap_path,
        "overlap_strategy": args.moe_expert_overlap_strategy,
    }


def main():
    parser = argparse.ArgumentParser()
    # Model / dataset
    parser.add_argument("--model-path", type=str,
                        default="/home/yyx/models/Qwen3-30B-A3B")
    parser.add_argument("--dataset-path", type=str, default="datasets/aime24.parquet")
    parser.add_argument("--dataset-label", type=str, default=None,
                        help="short label used in output filenames; defaults to dataset filename stem")
    parser.add_argument("--num-problems", type=int, default=1,
                        help="how many AIME24 problems to evaluate (default 1 for smoke test; 30 = full)")

    # Engine / parallelism
    parser.add_argument("--world-size", type=int, default=1,
                        help="total ranks; 1 = single-process, >1 = mp.spawn SPMD")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="tensor-parallel size within each EP slot (ep_size = world_size / tp_size)")
    parser.add_argument("--data-parallel-size", type=int, default=1,
                        help="data-parallel size for vllm_dp_ep mode")
    parser.add_argument("--master-port", type=int,
                        default=int(os.environ.get("MASTER_PORT", "29555")),
                        help="torch.distributed master port for this eval job")
    parser.add_argument("--moe-impl", type=str, default="triton",
                        choices=["torch", "triton", "ep_ll_torch", "ep_ll_triton", "ep_ht"])
    parser.add_argument("--enforce-eager", type=int, default=1,
                        help="1 = no cuda graph, 0 = capture (only valid for triton/ep_ll_triton)")
    parser.add_argument("--moe-block-size-m", type=int, default=64)
    parser.add_argument("--moe-ll-m-max", type=int, default=-1,
                        help="EP-LL bucket size; -1 = auto")
    parser.add_argument("--moe-ll-overflow-policy", type=str,
                        default=os.environ.get("MOE_LL_OVERFLOW_POLICY", "drop"),
                        choices=["drop", "error"],
                        help="EP-LL capacity overflow behavior: drop keeps LL-style fixed-capacity semantics; error is an eager debug guard")
    parser.add_argument("--moe-expert-placement", type=str,
                        default=os.environ.get("MOE_EXPERT_PLACEMENT", "contiguous"),
                        choices=[
                            "contiguous",
                            "round_robin",
                            "fixed_random_shuffle",
                            "load_balanced_greedy_with_locality_tiebreak",
                            "communication_aware_greedy",
                        ])
    parser.add_argument("--moe-expert-placement-seed", type=int,
                        default=int(os.environ.get("MOE_EXPERT_PLACEMENT_SEED", "0")))
    parser.add_argument("--moe-expert-placement-path", type=str,
                        default=os.environ.get("MOE_EXPERT_PLACEMENT_PATH"),
                        help="placement JSON generated from a routing profile")
    parser.add_argument("--moe-expert-overlap-path", type=str,
                        default=os.environ.get("MOE_EXPERT_OVERLAP_PATH"),
                        help="overlap plan JSON for owner_local_ep + ep_ht runtime")
    parser.add_argument("--moe-expert-overlap-strategy", type=str,
                        default=os.environ.get("MOE_EXPERT_OVERLAP_STRATEGY", "hybrid"),
                        choices=[
                            "disjoint",
                            "greedy_balance",
                            "min_communication",
                            "cv_aware",
                            "hybrid",
                            "numa_aware_min_communication",
                        ])
    parser.add_argument("--moe-expert-overlap-value", type=float, default=0.0,
                        help="informational overlap value recorded in output filenames/metadata")
    parser.add_argument("--moe-drop-policy", type=str,
                        default=os.environ.get("MOE_DROP_POLICY", "none"),
                        choices=[
                            "none",
                            "tail_weight",
                            "random",
                            "cross_numa_first",
                            "per_expert_uniform",
                            "hot_expert_relief",
                            "per_expert_tailtoken",
                            "hotspot_relief",
                        ])
    parser.add_argument("--moe-drop-rate", type=float,
                        default=float(os.environ.get("MOE_DROP_RATE", "0.0")))
    parser.add_argument("--moe-drop-seed", type=int,
                        default=int(os.environ.get("MOE_DROP_SEED", "0")))
    parser.add_argument("--moe-router-keff", type=int,
                        default=int(os.environ.get("MOE_ROUTER_KEFF", "0")),
                        help="if >0 and < model top_k, dispatch only top-K_eff branches per token")
    parser.add_argument("--prompt-style", type=str, default="answer_only",
                        choices=["answer_only", "answer_first", "reasoning_brief"],
                        help=(
                            "answer_first: boxed answer on line 1; "
                            "reasoning_brief: <=4 short reasoning sentences then a final boxed line."
                        ))
    parser.add_argument("--score-mode", type=str, default="default",
                        choices=["default", "answer_first"],
                        help="answer_first extracts boxed value after 'Final answer:' before falling back to math_verify")
    parser.add_argument("--moe-profile-routing", type=int,
                        default=int(os.environ.get("MOE_PROFILE_ROUTING", "0")),
                        help="1 writes eval_results/moe_routing_profile_<run>.json")
    parser.add_argument("--moe-profile-overlap-runtime", type=int,
                        default=int(os.environ.get("MOE_PROFILE_OVERLAP_RUNTIME", "0")),
                        help="1 writes overlap runtime stats JSON for EP-HT owner_local_ep")
    parser.add_argument("--moe-runtime-mode", type=str, default="legacy_tp_ep",
                        choices=["legacy_tp_ep", "vllm_dp_ep", "owner_local_ep"])
    parser.add_argument("--output-dir", default="eval_results",
                        help="directory for eval JSON, rollouts, and routing profiles")
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
    os.environ["MOE_PROFILE_ROUTING"] = "1" if args.moe_profile_routing else "0"
    os.environ["MOE_PROFILE_OVERLAP_RUNTIME"] = "1" if args.moe_profile_overlap_runtime else "0"
    os.environ.setdefault("MOE_PROFILE_RUN_ID", time.strftime("%Y%m%d_%H%M%S"))
    routing_profile.set_metadata(**_profile_metadata(args))
    overlap_runtime_stats.set_metadata(**_profile_metadata(args))

    if args.world_size == 1:
        run_single_process(args)
    else:
        if args.world_size % args.tp_size != 0:
            raise SystemExit(
                f"world_size={args.world_size} must be divisible by tp_size={args.tp_size}"
            )
        if args.moe_expert_overlap_path and (
            args.moe_impl != "ep_ht" or args.moe_runtime_mode != "owner_local_ep"
        ):
            raise SystemExit(
                "overlap runtime currently supports only --moe-impl ep_ht with "
                "--moe-runtime-mode owner_local_ep"
            )
        if args.moe_runtime_mode == "vllm_dp_ep":
            expected_dp = args.world_size // args.tp_size
            if args.data_parallel_size != expected_dp:
                raise SystemExit(
                    f"vllm_dp_ep expects data_parallel_size=world_size/tp_size={expected_dp}, "
                    f"got {args.data_parallel_size}"
                )
        if args.moe_runtime_mode == "owner_local_ep":
            if args.tp_size != 1:
                raise SystemExit("owner_local_ep expects tp_size=1")
            if args.data_parallel_size != args.world_size:
                raise SystemExit(
                    f"owner_local_ep expects data_parallel_size=world_size={args.world_size}, "
                    f"got {args.data_parallel_size}"
                )
        run_distributed(args)


if __name__ == "__main__":
    main()
