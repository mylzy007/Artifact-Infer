"""Phase A (next-token accuracy) — task-level accuracy on lcc test set.

Same 48-layer Qwen3-30B-A3B + same held-out 4096-token lcc test split used by
the Phase B/C combine sparsification sweeps. For each config we capture the
full per-position logits and compute:

  - next-token top-1 accuracy   = argmax(logits[t]) == ground_truth[t+1]
  - next-token top-5 accuracy   = ground_truth[t+1] in top-5(logits[t])
  - top-1 agreement with teacher (independent of correctness — answers "how
    often does the compressed model pick the SAME token as the uncompressed
    model"; this is the strict identity-level metric).
  - PPL (re-reported to anchor against the prior sweep tables).

Configs:
  teacher                           : no compression
  topk_l2_fp8  keep_frac=0.5        : 2x value
  topk_l2_fp8  keep_frac=0.25       : 4x value
  topk_l2_fp8  keep_frac=0.125      : 8x value

Outputs (under eval_results/compression_lowrank_stage0_taskacc_lcc/):
  - summary.json
  - report.md
  - accuracy_curve.png

Run:
  CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    PATH="/usr/local/cuda-12.8/bin:/home/lzy/miniconda3/envs/vllm/bin:$PATH" \\
    CUDA_HOME=/usr/local/cuda-12.8 \\
    /home/lzy/miniconda3/envs/vllm/bin/python eval/compression/stage0_taskacc_lcc.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

MODEL_PATH = "/home/lzy/models/Qwen3-30B-A3B"
OUT_DIR = Path("eval_results/compression_lowrank_stage0_taskacc_lcc")
LB_DIR = Path("/home/lzy/datasets/moe_benchmarks/longbench/extracted/data")

FP8_E4M3_MAX = 448.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--chunk-len", type=int, default=512)
    p.add_argument("--test-chunks", type=int, default=8)
    p.add_argument("--batch-chunks", type=int, default=4)
    p.add_argument("--keep-fracs", type=float, nargs="+", default=[0.5, 0.25, 0.125])
    p.add_argument("--gpu-mem-gib", type=int, default=22)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_lcc_text(min_chars: int) -> str:
    pieces, total = [], 0
    with (LB_DIR / "lcc.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            txt = rec.get("context") or rec.get("input") or ""
            if isinstance(txt, str) and len(txt) >= 200:
                pieces.append(txt)
                total += len(txt)
                if total >= min_chars:
                    break
    return "\n\n".join(pieces)


def chunk_ids(ids: list[int], chunk_len: int, n_chunks: int, skip: int = 0) -> list[list[int]]:
    out, pos = [], skip
    while pos + chunk_len <= len(ids) and len(out) < n_chunks:
        out.append(ids[pos : pos + chunk_len])
        pos += chunk_len
    return out


def sparsify_topk_l2_fp8(x: torch.Tensor, k: int) -> torch.Tensor:
    """topk_l2 (L2-norm-preserving rescale) + per-row FP8 E4M3 round trip.
    Matches the production CombineEPHT path."""
    abs_x = x.abs()
    _, idx = abs_x.topk(k, dim=-1)
    sparse = torch.zeros_like(x)
    sparse.scatter_(-1, idx, x.gather(-1, idx))
    # L2 rescale.
    full_n2 = (x.float() ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    sparse_n2 = (sparse.float() ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    sparse = sparse * (full_n2 / sparse_n2).to(x.dtype)
    # FP8 round trip (per-row amax scaling).
    xf = sparse.float()
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / FP8_E4M3_MAX
    x_scaled = xf / scale
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    x_back = x_fp8.to(torch.float32) * scale
    return x_back.to(x.dtype)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[taskacc] loading {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_mem = {i: f"{args.gpu_mem_gib}GiB" for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_mem,
        trust_remote_code=True,
    )
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    cfg = model.config
    H = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    E = int(cfg.num_experts)
    embed_dev = next(model.model.embed_tokens.parameters()).device
    print(f"[taskacc] H={H}, n_layers={n_layers}, num_experts={E}", flush=True)

    moe_blocks = [model.model.layers[L].mlp for L in range(n_layers)]
    all_experts: list = []
    for L in range(n_layers):
        for e in range(E):
            all_experts.append(moe_blocks[L].experts[e])

    # Tokenize lcc, same held-out split as prior sweeps (skip the first 16 chunks
    # used as calibration in stage_0 SVD experiments).
    text = load_lcc_text(min_chars=1_400_000)
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    if not isinstance(ids, list):
        ids = list(ids)
    skip = 16 * args.chunk_len
    test_chunks = chunk_ids(ids, args.chunk_len, args.test_chunks, skip=skip)
    n_test_tok = args.test_chunks * args.chunk_len
    print(f"[taskacc] test_tokens={n_test_tok} (same lcc held-out split)", flush=True)

    def run_one(label: str, keep_frac: float | None):
        """Run a forward pass for the given config; return per-token logits-derived metrics."""
        if keep_frac is not None and 0 < keep_frac < 1:
            k = max(1, int(round(H * keep_frac)))

            def make_hook(k_inner: int):
                def hook(_m, _inputs, output):
                    if output is None or output.numel() == 0:
                        return output
                    return sparsify_topk_l2_fp8(output, k_inner)
                return hook

            handles = [expert_mod.register_forward_hook(make_hook(k)) for expert_mod in all_experts]
        else:
            handles = []

        # We need raw logits per token. Run a custom loop instead of passing labels=.
        try:
            with torch.no_grad():
                ids_t = torch.tensor(test_chunks, dtype=torch.long, device=embed_dev)
                losses = []
                top1_correct = 0  # vs ground truth
                top5_correct = 0
                top1_argmax_cache: list[torch.Tensor] = []  # for teacher_agreement (rank 0)
                total = 0
                for i in range(0, ids_t.shape[0], args.batch_chunks):
                    ids_b = ids_t[i : i + args.batch_chunks]
                    out = model(input_ids=ids_b, use_cache=False)
                    logits = out.logits  # [B, T, V]
                    # Standard LM-loss formulation: shift so predictions of position t
                    # are evaluated against token at t+1.
                    shift_logits = logits[..., :-1, :].float()      # [B, T-1, V]
                    shift_labels = ids_b[..., 1:]                    # [B, T-1]
                    nll = torch.nn.functional.cross_entropy(
                        shift_logits.reshape(-1, shift_logits.shape[-1]),
                        shift_labels.reshape(-1),
                        reduction="mean",
                    )
                    losses.append(float(nll.item()))
                    top1 = shift_logits.argmax(dim=-1)              # [B, T-1]
                    top5_vals, top5 = shift_logits.topk(5, dim=-1)  # [B, T-1, 5]
                    top1_correct += int((top1 == shift_labels).sum().item())
                    top5_correct += int((top5 == shift_labels.unsqueeze(-1)).any(dim=-1).sum().item())
                    total += int(shift_labels.numel())
                    top1_argmax_cache.append(top1.cpu())
        finally:
            for h in handles:
                h.remove()

        avg_loss = float(np.mean(losses))
        return {
            "label": label,
            "keep_frac": keep_frac,
            "loss": avg_loss,
            "ppl": math.exp(avg_loss),
            "top1_correct": int(top1_correct),
            "top5_correct": int(top5_correct),
            "total": int(total),
            "top1_acc": top1_correct / max(total, 1),
            "top5_acc": top5_correct / max(total, 1),
            "top1_argmax": torch.cat([c.reshape(-1) for c in top1_argmax_cache], dim=0),
        }

    print("[taskacc] running teacher (no compression) ...", flush=True)
    teacher = run_one("teacher", keep_frac=None)
    print(
        f"[taskacc]   teacher: loss={teacher['loss']:.4f}  ppl={teacher['ppl']:.3f}  "
        f"top1={teacher['top1_acc']*100:.2f}%  top5={teacher['top5_acc']*100:.2f}%",
        flush=True,
    )

    results = {"teacher": {k: v for k, v in teacher.items() if k != "top1_argmax"}}
    students = []
    for kf in args.keep_fracs:
        label = f"keep={kf}_fp8_l2"
        print(f"[taskacc] running student {label} ...", flush=True)
        student = run_one(label, keep_frac=kf)
        # Compare top-1 vs teacher.
        agree = int((student["top1_argmax"] == teacher["top1_argmax"]).sum().item())
        agree_total = int(teacher["top1_argmax"].numel())
        agree_ratio = agree / max(agree_total, 1)
        student_summary = {k: v for k, v in student.items() if k != "top1_argmax"}
        student_summary["agreement_with_teacher_top1"] = agree_ratio
        student_summary["ppl_increase_pct"] = (student["ppl"] - teacher["ppl"]) / teacher["ppl"] * 100
        student_summary["top1_drop_pp"] = (teacher["top1_acc"] - student["top1_acc"]) * 100  # in percentage points
        student_summary["top5_drop_pp"] = (teacher["top5_acc"] - student["top5_acc"]) * 100
        results[label] = student_summary
        students.append(student_summary)
        print(
            f"[taskacc]   {label}: loss={student['loss']:.4f}  ppl={student['ppl']:.3f} "
            f"(+{student_summary['ppl_increase_pct']:.1f}%)  "
            f"top1={student['top1_acc']*100:.2f}% (drop {student_summary['top1_drop_pp']:.2f}pp)  "
            f"top5={student['top5_acc']*100:.2f}% (drop {student_summary['top5_drop_pp']:.2f}pp)  "
            f"teacher_agreement={agree_ratio*100:.2f}%",
            flush=True,
        )

    summary = {
        "model_path": args.model_path,
        "hidden_size": H,
        "n_layers": n_layers,
        "num_experts": E,
        "test_tokens_per_chunk": args.chunk_len,
        "test_chunks": args.test_chunks,
        "test_tokens": n_test_tok,
        "shifted_predictions": n_test_tok - args.test_chunks,  # one per (T-1) per chunk
        "domain": "lcc",
        "keep_fracs": args.keep_fracs,
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir/'summary.json'}", flush=True)

    # Markdown.
    lines = []
    lines.append("# Task-level next-token accuracy — combine-side compression on lcc")
    lines.append("")
    lines.append(f"Held-out lcc test: {n_test_tok} tokens ({args.test_chunks} chunks x {args.chunk_len} tokens).")
    lines.append(f"Teacher: loss={teacher['loss']:.4f}  ppl={teacher['ppl']:.3f}  top1={teacher['top1_acc']*100:.2f}%  top5={teacher['top5_acc']*100:.2f}%")
    lines.append("")
    lines.append("| config | value compr | PPL | PPL +% | top-1 acc | top-1 drop (pp) | top-5 acc | top-5 drop (pp) | teacher top-1 agree |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| teacher | - | {teacher['ppl']:.3f} | - | {teacher['top1_acc']*100:.2f}% | - | "
        f"{teacher['top5_acc']*100:.2f}% | - | - |"
    )
    for s in students:
        kf = s["keep_frac"]
        lines.append(
            f"| keep={kf} +FP8 +L2 | {int(round(1/kf))}x | "
            f"{s['ppl']:.3f} | {s['ppl_increase_pct']:+.1f}% | "
            f"{s['top1_acc']*100:.2f}% | {s['top1_drop_pp']:+.2f}pp | "
            f"{s['top5_acc']*100:.2f}% | {s['top5_drop_pp']:+.2f}pp | "
            f"{s['agreement_with_teacher_top1']*100:.2f}% |"
        )
    lines.append("")
    lines.append("## How to read")
    lines.append("")
    lines.append("- `top-1 acc` is fraction of held-out positions where the model's argmax-predicted next token equals the true next token. Standard LM eval.")
    lines.append("- `top-1 drop (pp)` is teacher_top1 - student_top1 in absolute percentage points. Small = compression preserves task.")
    lines.append("- `teacher top-1 agreement` is fraction of positions where student's top-1 prediction == teacher's top-1 prediction, even when both are wrong. Strict 'are the two models behaving identically' metric.")
    lines.append("- For greedy decoding (temperature=0) deployments, `top-1 agreement` is the most direct quality proxy.")
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"wrote {out_dir/'report.md'}", flush=True)

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ratios = [1 / kf for kf in args.keep_fracs]
        top1 = [results[f"keep={kf}_fp8_l2"]["top1_acc"] for kf in args.keep_fracs]
        top5 = [results[f"keep={kf}_fp8_l2"]["top5_acc"] for kf in args.keep_fracs]
        agree = [results[f"keep={kf}_fp8_l2"]["agreement_with_teacher_top1"] for kf in args.keep_fracs]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].axhline(teacher["top1_acc"], color="red", linestyle="--", linewidth=0.8, label=f"teacher top-1 = {teacher['top1_acc']*100:.1f}%")
        axes[0].plot(ratios, top1, marker="o", label="student top-1 acc")
        axes[0].plot(ratios, top5, marker="s", label="student top-5 acc")
        axes[0].axhline(teacher["top5_acc"], color="gray", linestyle=":", linewidth=0.8, label=f"teacher top-5 = {teacher['top5_acc']*100:.1f}%")
        axes[0].set_xlabel("value-only compression ratio (1/keep_frac)")
        axes[0].set_ylabel("accuracy")
        axes[0].set_xscale("log", base=2)
        axes[0].set_title("Next-token accuracy on lcc held-out")
        axes[0].grid(True, alpha=0.3, which="both")
        axes[0].legend(fontsize=8)

        axes[1].plot(ratios, agree, marker="o", color="C2")
        axes[1].set_xlabel("value-only compression ratio")
        axes[1].set_ylabel("top-1 agreement with teacher")
        axes[1].set_xscale("log", base=2)
        axes[1].set_ylim(0, 1.02)
        axes[1].set_title("Strict student == teacher top-1")
        axes[1].grid(True, alpha=0.3, which="both")
        fig.suptitle("Combine-side topk_l2+FP8 — task-level accuracy on lcc")
        fig.tight_layout()
        fig.savefig(out_dir / "accuracy_curve.png", dpi=140)
        plt.close(fig)
        print(f"wrote {out_dir/'accuracy_curve.png'}", flush=True)
    except Exception as e:
        print(f"plot failed: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
