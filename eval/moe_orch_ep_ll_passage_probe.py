from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

from bazaar.nanovllm_moe_orch_ep_ll_triton import spawn_eval


OFFICIAL_PROMPT = (
    "Here are 30 paragraphs from Wikipedia, along with an abstract. "
    "Please determine which paragraph the abstract is from.\n\n{context}\n\n"
    "The following is an abstract.\n\n{input}\n\n"
    "Please enter the number of the paragraph that the abstract is from. "
    "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\n"
    "The answer is: "
)


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


def load_prompts(dataset_path: Path, model_path: str, limit: int) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    rows = []
    with dataset_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows = rows[:limit]
    prompts = []
    for row in rows:
        prompt = OFFICIAL_PROMPT.format(context=row["context"], input=row["input"])
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=True))
        prompts.append(
            {
                "_id": row["_id"],
                "prompt": prompt,
                "prompt_len": prompt_len,
                "reference": row["answers"][0],
            }
        )
    return prompts


def run_one(engine, SamplingParams, prompt: str, max_tokens: int):
    engine.reset()
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    engine.add_request(prompt, sp)
    outputs = {}
    metrics = {
        "prefill_tokens": 0,
        "prefill_time_s": 0.0,
        "decode_tokens": 0,
        "decode_time_s": 0.0,
    }
    t0 = time.perf_counter()
    while not engine.is_finished():
        output, num_tokens, execution_time = engine.step()
        if num_tokens > 0:
            metrics["prefill_tokens"] += int(num_tokens)
            metrics["prefill_time_s"] += float(execution_time)
        else:
            metrics["decode_tokens"] += -int(num_tokens)
            metrics["decode_time_s"] += float(execution_time)
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids
    metrics["e2e_total_time_s"] = time.perf_counter() - t0
    metrics["prefill_throughput_tok_s"] = (
        metrics["prefill_tokens"] / metrics["prefill_time_s"]
        if metrics["prefill_time_s"] > 0 else 0.0
    )
    metrics["decode_throughput_tok_s"] = (
        metrics["decode_tokens"] / metrics["decode_time_s"]
        if metrics["decode_time_s"] > 0 else 0.0
    )
    engine.reset()
    out = outputs[min(outputs)] if outputs else []
    return {
        "text": engine.tokenizer.decode(out),
        "token_ids": out,
        "metrics": metrics,
    }


def eval_fn(engine, SamplingParams, rank: int, world_size: int, *, prompts: list[dict], output_path: str, max_tokens: int):
    rows = []
    for item in prompts:
        result = run_one(engine, SamplingParams, item["prompt"], max_tokens)
        if rank == 0:
            score_off = retrieval_score(result["text"], item["reference"])
            score_str = retrieval_score_strict(result["text"], item["reference"])
            rows.append(
                {
                    "_id": item["_id"],
                    "prompt_len": item["prompt_len"],
                    "reference": item["reference"],
                    "gen_text": result["text"],
                    "score_official": score_off,
                    "score_strict": score_str,
                    "metrics": result["metrics"],
                }
            )
            print(
                f"[orch ep-ll] {_short(item['_id'])} "
                f"prefill={result['metrics']['prefill_time_s']:.2f}s "
                f"e2e={result['metrics']['e2e_total_time_s']:.2f}s "
                f"official={score_off:.3f} strict={score_str:.3f}",
                flush=True,
            )

    if rank != 0:
        return

    off = [row["score_official"] for row in rows]
    strict = [row["score_strict"] for row in rows]
    prefill = [row["metrics"]["prefill_time_s"] for row in rows]
    e2e = [row["metrics"]["e2e_total_time_s"] for row in rows]
    prefill_tok = [row["metrics"]["prefill_throughput_tok_s"] for row in rows]
    decode_tok = [row["metrics"]["decode_throughput_tok_s"] for row in rows]
    payload = {
        "summary": {
            "num_prompts": len(rows),
            "official_score_mean": statistics.mean(off) if off else 0.0,
            "strict_score_mean": statistics.mean(strict) if strict else 0.0,
            "prefill_time_s_mean": statistics.mean(prefill) if prefill else 0.0,
            "e2e_total_time_s_mean": statistics.mean(e2e) if e2e else 0.0,
            "prefill_tok_s_mean": statistics.mean(prefill_tok) if prefill_tok else 0.0,
            "decode_tok_s_mean": statistics.mean(decode_tok) if decode_tok else 0.0,
        },
        "rows": rows,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2))


def _short(value: str) -> str:
    return value[:8]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--dataset-path", required=True, type=Path)
    p.add_argument("--output-path", required=True)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--max-num-batched-tokens", type=int, default=4352)
    p.add_argument("--max-num-seqs", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4240)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--num-prompts", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.dataset_path, args.model_path, args.num_prompts)
    spawn_eval(
        world_size=args.world_size,
        engine_kwargs={
            "model": args.model_path,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": 1,
        },
        eval_fn=eval_fn,
        eval_fn_kwargs={
            "prompts": prompts,
            "output_path": args.output_path,
            "max_tokens": args.max_tokens,
        },
    )


if __name__ == "__main__":
    main()
