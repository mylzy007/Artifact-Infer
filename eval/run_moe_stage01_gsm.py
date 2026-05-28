"""Stage 0/1 MoE placement runner for GSM8K.

This is a thin driver around the existing EP-LL / EP-HT experiment path:

  Stage 0:
    - prepare a GSM8K parquet with columns expected by eval.test_bazaar_moe
    - run contiguous profile with the first requested implementation for each
      world size

  Stage 1:
    - generate profile-based static placements
    - evaluate requested implementations over the static placement set

The defaults are tuned for 8x RTX 4090 PCIe/no-P2P:
  - world sizes 4 and 8
  - TP=1, pure EP
  - max_model_len=max_num_batched_tokens=256
  - HT-only by default, because EP-LL fixed buffers are communication-heavy on
    4090 PCIe; run EP-LL later as a separate capacity/drop sweep
  - max_num_seqs=8, so decode can amortize per-step overhead while
    max_num_batched_tokens still caps prefill pressure
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics
import subprocess
import time
from pathlib import Path

import datasets
from transformers import AutoTokenizer


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/lzy/models/Qwen3-30B-A3B")
DEFAULT_GSM_TEST = Path("/home/lzy/datasets/moe_benchmarks/gsm8k/main/test-00000-of-00001.parquet")
DEFAULT_OUTPUT_ROOT = REPO / "eval_results"


def sanitize(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "run"


def extract_final_answer(text: object) -> str:
    value = "" if text is None else str(text)
    match = re.search(r"####\s*(.+?)\s*$", value, flags=re.S)
    if match:
        return normalize_final_answer(match.group(1))
    return normalize_final_answer(value)


def normalize_final_answer(text: object) -> str:
    value = "" if text is None else str(text)
    value = value.strip()
    value = value.replace("$", "").replace(",", "")
    return value.strip()


def build_gsm_prompt(question: str) -> str:
    return (
        f"{question.strip()}\n\n"
        "Solve the problem. Keep the reasoning concise, and put the final "
        "answer on the last line exactly in the form \\boxed{answer}."
    )


def load_source(path: Path):
    if path.suffix == ".jsonl":
        return datasets.load_dataset("json", data_files=str(path), split="train")
    if path.suffix == ".json":
        return datasets.load_dataset("json", data_files=str(path), split="train")
    if path.suffix == ".parquet":
        return datasets.load_dataset("parquet", data_files=str(path), split="train")
    raise SystemExit(f"unsupported GSM source format: {path}")


def normalize_gsm_dataset(source: Path, output_path: Path, *, limit: int) -> dict:
    rows = load_source(source)
    n = len(rows) if limit <= 0 else min(limit, len(rows))
    items = []
    for idx in range(n):
        row = rows[idx]
        if "conversations" in row:
            conv = row["conversations"]
            question = str(conv[0]["value"])
            solution = str(conv[-1]["value"])
            row_id = f"gsm8k-{idx:05d}"
        else:
            question = str(row["question"])
            solution = str(row.get("solution") or row.get("answer") or "")
            row_id = str(row.get("id") or f"gsm8k-{idx:05d}")
        answer = extract_final_answer(row.get("answer") if "answer" in row else solution)
        if not answer or "####" in answer:
            answer = extract_final_answer(solution)
        prompt_text = build_gsm_prompt(question)
        items.append(
            {
                "id": row_id,
                "solution": solution,
                "answer": answer,
                "url": str(row.get("url") or ""),
                "question": question,
                "prompt": [{"role": "user", "content": prompt_text}],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(items).to_parquet(str(output_path))
    return {"source": str(source), "prepared_path": str(output_path), "num_rows": len(items)}


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return sorted(values)[idx]


def prompt_stats(dataset_path: Path, model_path: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    rows = datasets.load_dataset("parquet", data_files=str(dataset_path), split="train")
    lengths = []
    for row in rows:
        raw = tokenizer.apply_chat_template(
            row["prompt"], add_generation_prompt=True, tokenize=False,
        )
        lengths.append(len(tokenizer.encode(raw)))
    return {
        "num_rows": len(lengths),
        "min": min(lengths) if lengths else 0,
        "p50": statistics.median(lengths) if lengths else 0,
        "p90": percentile(lengths, 0.90),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths) if lengths else 0,
        "mean": sum(lengths) / len(lengths) if lengths else 0.0,
    }


def build_runner_cmd(args: argparse.Namespace, dataset_path: Path, num_rows: int) -> list[str]:
    num_problems = num_rows if args.num_problems <= 0 else min(args.num_problems, num_rows)
    cmd = [
        args.python,
        "-m",
        "eval.run_moe_placement_experiments",
        "--model-path",
        str(args.model_path),
        "--dataset-path",
        str(dataset_path),
        "--dataset-label",
        args.dataset_label,
        "--world-sizes",
        args.world_sizes,
        "--tp-size",
        str(args.tp_size),
        "--impls",
        args.impls,
        "--profile-impl",
        args.profile_impl,
        "--placements",
        args.placements,
        "--num-problems",
        str(num_problems),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enforce-eager",
        str(args.enforce_eager),
        "--moe-ll-overflow-policy",
        args.moe_ll_overflow_policy,
        "--gpu-ids",
        args.gpu_ids,
        "--seed",
        str(args.seed),
        "--base-master-port",
        str(args.base_master_port),
        "--port-retries",
        str(args.port_retries),
        "--timeout-s",
        str(args.timeout_s),
        "--experiment-name",
        args.experiment_name,
        "--run-id",
        args.run_id,
        "--output-root",
        str(args.output_root),
    ]
    if args.moe_ll_m_max > 0:
        cmd += ["--moe-ll-m-max", str(args.moe_ll_m_max)]
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/home/lzy/miniconda3/envs/vllm/bin/python")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gsm-source", type=Path, default=DEFAULT_GSM_TEST)
    parser.add_argument("--dataset-label", default="gsm8k")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-name", default="gsm_stage01")
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--prepared-dir", type=Path, default=None)
    parser.add_argument(
        "--num-problems",
        type=int,
        default=256,
        help="-1 means the full prepared GSM source; use 8 for a quick same-config smoke run",
    )
    parser.add_argument("--world-sizes", default="4,8")
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--impls", default="ep_ht")
    parser.add_argument("--profile-impl", default="ep_ht")
    parser.add_argument(
        "--placements",
        default=(
            "contiguous,round_robin,fixed_random_shuffle,"
            "load_balanced_greedy_with_locality_tiebreak,communication_aware_greedy"
        ),
    )

    # 4090 PCIe-friendly defaults. The answer-oriented GSM prompt is longer than
    # the raw question, so keep enough room for concise reasoning plus boxed answer.
    # HT-only is the default first pass; EP-LL should be run later with explicit
    # capacity/drop settings because its fixed all-to-all buffer dominates decode.
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", type=int, default=1)
    parser.add_argument("--moe-ll-overflow-policy", choices=["error", "drop"], default="error")
    parser.add_argument(
        "--moe-ll-m-max",
        type=int,
        default=128,
        help=(
            "EP-LL fixed bucket size. Ignored by HT-only runs. For later EP-LL "
            "capacity experiments, use 128/192 with --moe-ll-overflow-policy drop, "
            "or 256 with error for strict no-drop comparison."
        ),
    )

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--base-master-port", type=int, default=29600)
    parser.add_argument("--port-retries", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=604800)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_num_batched_tokens < args.max_model_len:
        raise SystemExit("--max-num-batched-tokens must be >= --max-model-len")

    run_root = args.output_root / f"{sanitize(args.experiment_name)}_{sanitize(args.run_id)}"
    prepared_dir = args.prepared_dir or (run_root / "prepared")
    prepared_path = prepared_dir / f"{sanitize(args.dataset_label)}.stage01.parquet"
    dataset_info = normalize_gsm_dataset(args.gsm_source, prepared_path, limit=args.num_problems)
    stats = prompt_stats(prepared_path, args.model_path)

    notes = []
    if stats["max"] + args.max_tokens > args.max_model_len:
        notes.append(
            "Some prompts may exceed max_model_len after generation; "
            "increase --max-model-len and --max-num-batched-tokens together if needed."
        )
    effective_p50_batch = max(1, args.max_num_batched_tokens // max(1, int(stats["p50"])))
    effective_p95_batch = max(1, args.max_num_batched_tokens // max(1, int(stats["p95"])))
    auto_m_max = max(8, ((args.max_num_batched_tokens * 8 + 127) // 128) * 4)
    effective_m_max = args.moe_ll_m_max if args.moe_ll_m_max > 0 else auto_m_max
    batch_note = {
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "effective_prefill_batch_at_p50_prompt": min(args.max_num_seqs, effective_p50_batch),
        "effective_prefill_batch_at_p95_prompt": min(args.max_num_seqs, effective_p95_batch),
        "ep_ll_auto_m_max_for_qwen3_topk8_num_experts128": auto_m_max,
        "ep_ll_effective_m_max": effective_m_max,
        "ep_ll_no_drop_margin_note": (
            "M_max equal to max_num_batched_tokens is the strict no-drop setting "
            "for a single expert bucket, assuming each token has unique top-k experts."
        ),
    }

    cmd = build_runner_cmd(args, prepared_path, dataset_info["num_rows"])
    manifest = {
        "dataset": dataset_info,
        "prompt_token_stats": stats,
        "batch_and_buffer_notes": batch_note,
        "notes": notes,
        "runner_command": shlex.join(cmd),
        "stage_design": {
            "stage0": "profile contiguous with the first requested implementation once per world size",
            "stage1": "generate static placements from profile and evaluate requested implementations",
            "world_sizes": args.world_sizes,
            "impls": args.impls,
            "placements": args.placements,
        },
    }
    manifest_path = run_root / "stage01_manifest.json"
    write_manifest(manifest_path, manifest)

    print(f"[stage01] prepared dataset: {prepared_path}")
    print(f"[stage01] manifest: {manifest_path}")
    print(f"[stage01] prompt stats: {json.dumps(stats, ensure_ascii=False)}")
    print(f"[stage01] batch notes: {json.dumps(batch_note, ensure_ascii=False)}")
    if notes:
        for note in notes:
            print(f"[stage01][note] {note}")
    print("[stage01] runner command:")
    print(shlex.join(cmd))

    if args.prepare_only:
        return
    subprocess.run(cmd, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
