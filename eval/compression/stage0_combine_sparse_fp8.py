"""Phase C: combine-side topk_l2 sparsify + FP8 (E4M3) quantization on top.

Adds per-row FP8 quantization of the kept values to the Phase A winner
(topk_l2). Quantization recipe (matches DeepEP / MoRI production):

  per-row scale s = amax / 448   (448 = FP8 E4M3 max)
  v_fp8 = (v / s).to(fp8_e4m3fn)
  v_back = v_fp8.to(bf16) * s

This is applied after L2-norm-preserving sparsification, so the per-row
rescale that topk_l2 introduces is part of what we quantize.

Compression accounting (value-only; receiver needs k indices + 1 scale + k
8-bit values per row):

  - sparse_value_only_ratio = H / k                          (1 byte vs 2 bytes per kept)
  - sparse_fp8_value_only_ratio = (2 * H) / k = 2 * sparse_value_only_ratio

For keep_frac in {0.25, 0.125, 0.0625} we get sparse+FP8 value-only ratios
of {16x, 32x, 64x}.

Compare against:
  - teacher (no compression)
  - topk_l2 alone (bf16 values, no FP8)
  - naive_topk + FP8 (sanity check that L2 fix helps even at FP8 precision)

Run:
  CUDA_VISIBLE_DEVICES=0,1,2,3 /home/lzy/miniconda3/envs/atom/bin/python \
      eval/compression/stage0_combine_sparse_fp8.py
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
OUT_DIR = Path("eval_results/compression_lowrank_stage0_sparse_fp8")
LB_DIR = Path("/home/lzy/datasets/moe_benchmarks/longbench/extracted/data")

FP8_E4M3_MAX = 448.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--chunk-len", type=int, default=512)
    p.add_argument("--test-chunks", type=int, default=8)
    p.add_argument("--batch-chunks", type=int, default=4)
    p.add_argument("--keep-fracs", type=float, nargs="+", default=[0.25, 0.125, 0.0625])
    p.add_argument(
        "--variants",
        nargs="+",
        default=["topk_l2_bf16", "topk_l2_fp8", "naive_topk_fp8", "fp8_only"],
    )
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


def fp8_round_trip(x: torch.Tensor) -> torch.Tensor:
    """Per-row FP8-E4M3 quantization round-trip in-place semantics: return x' such
    that x' has been through (cast to fp8 with per-row amax scaling, then back)."""
    if x.numel() == 0:
        return x
    # Per-row amax in fp32.
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / FP8_E4M3_MAX
    x_scaled = xf / scale
    # Round-trip cast.
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    x_back = x_fp8.to(torch.float32) * scale
    return x_back.to(x.dtype)


def sparsify_naive_topk(x: torch.Tensor, k: int) -> torch.Tensor:
    abs_x = x.abs()
    _, idx = abs_x.topk(k, dim=-1)
    sparse = torch.zeros_like(x)
    sparse.scatter_(-1, idx, x.gather(-1, idx))
    return sparse


def sparsify_topk_l2(x: torch.Tensor, k: int) -> torch.Tensor:
    abs_x = x.abs()
    _, idx = abs_x.topk(k, dim=-1)
    sparse = torch.zeros_like(x)
    sparse.scatter_(-1, idx, x.gather(-1, idx))
    full_n2 = (x.float() ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    sparse_n2 = (sparse.float() ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    return sparse * (full_n2 / sparse_n2).to(x.dtype)


def sparsify_topk_l2_fp8(x: torch.Tensor, k: int) -> torch.Tensor:
    """topk_l2 then quantize kept values via per-row FP8 round trip.

    We do FP8 only on the kept values, not on the zeros (the zeros stay exactly
    zero and don't cost any payload). For simplicity we apply the round-trip to
    the whole row (zeros are exactly representable in FP8, so this is fine)."""
    sparse = sparsify_topk_l2(x, k)
    return fp8_round_trip(sparse)


def sparsify_naive_topk_fp8(x: torch.Tensor, k: int) -> torch.Tensor:
    sparse = sparsify_naive_topk(x, k)
    return fp8_round_trip(sparse)


def fp8_only(x: torch.Tensor, k: int) -> torch.Tensor:
    # k is ignored; just round-trip the full row through FP8.
    return fp8_round_trip(x)


VARIANT_FN = {
    "topk_l2_bf16": sparsify_topk_l2,
    "topk_l2_fp8": sparsify_topk_l2_fp8,
    "naive_topk_fp8": sparsify_naive_topk_fp8,
    "fp8_only": fp8_only,
}


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[sparse_fp8] loading {args.model_path}", flush=True)
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
    print(f"[sparse_fp8] H={H}, n_layers={n_layers}, num_experts={E}", flush=True)

    moe_blocks = [model.model.layers[L].mlp for L in range(n_layers)]
    all_experts: list = []
    for L in range(n_layers):
        for e in range(E):
            all_experts.append(moe_blocks[L].experts[e])

    text = load_lcc_text(min_chars=1_400_000)
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    if not isinstance(ids, list):
        ids = list(ids)
    skip = 16 * args.chunk_len
    test_chunks = chunk_ids(ids, args.chunk_len, args.test_chunks, skip=skip)
    n_test_tok = args.test_chunks * args.chunk_len
    print(f"[sparse_fp8] test_tokens={n_test_tok} (held out, same split as prior sweeps)", flush=True)

    # Teacher.
    print("[sparse_fp8] teacher forward ...", flush=True)
    teacher_h: list[torch.Tensor] = []

    def teacher_norm_hook(_m, _inputs, output):
        teacher_h.append(output.detach().to("cpu", dtype=torch.float32).reshape(-1, output.shape[-1]))

    th = model.model.norm.register_forward_hook(teacher_norm_hook)
    teacher_losses = []
    with torch.no_grad():
        ids_t = torch.tensor(test_chunks, dtype=torch.long, device=embed_dev)
        labels_t = ids_t.clone()
        for i in range(0, ids_t.shape[0], args.batch_chunks):
            out = model(
                input_ids=ids_t[i : i + args.batch_chunks],
                labels=labels_t[i : i + args.batch_chunks],
                use_cache=False,
            )
            teacher_losses.append(float(out.loss.item()))
    th.remove()
    teacher_H = torch.cat(teacher_h, dim=0)
    teacher_loss = float(np.mean(teacher_losses))
    teacher_ppl = math.exp(teacher_loss)
    print(f"[sparse_fp8] teacher: loss={teacher_loss:.4f}  ppl={teacher_ppl:.3f}", flush=True)

    # Sweep.
    results: dict[str, dict[str, dict]] = {v: {} for v in args.variants}
    for variant in args.variants:
        fn = VARIANT_FN[variant]
        # For fp8_only, k is unused; still iterate keep_fracs to align table columns.
        for kf in args.keep_fracs:
            k = max(1, int(round(H * kf)))
            # Per-row payload (value-only) -- index bits and scale bits not accounted.
            if variant == "fp8_only":
                value_only_ratio = 2.0          # bf16 -> fp8 = 2x
                actual_keep = H
            elif "fp8" in variant:
                value_only_ratio = 2.0 * H / k  # sparse * fp8
                actual_keep = k
            else:
                value_only_ratio = H / k         # sparse only
                actual_keep = k
            print(f"[sparse_fp8] variant={variant} keep_frac={kf} (k={actual_keep}, value-only ~{value_only_ratio:.1f}x)", flush=True)

            student_h: list[torch.Tensor] = []

            def student_norm_hook(_m, _inputs, output):
                student_h.append(output.detach().to("cpu", dtype=torch.float32).reshape(-1, output.shape[-1]))

            def make_hook(k_inner: int, fn_inner):
                def hook(_m, _inputs, output):
                    if output is None or output.numel() == 0:
                        return output
                    return fn_inner(output, k_inner)
                return hook

            handles = []
            for expert_mod in all_experts:
                handles.append(expert_mod.register_forward_hook(make_hook(k, fn)))
            handles.append(model.model.norm.register_forward_hook(student_norm_hook))

            try:
                student_losses = []
                with torch.no_grad():
                    ids_t = torch.tensor(test_chunks, dtype=torch.long, device=embed_dev)
                    labels_t = ids_t.clone()
                    for i in range(0, ids_t.shape[0], args.batch_chunks):
                        out = model(
                            input_ids=ids_t[i : i + args.batch_chunks],
                            labels=labels_t[i : i + args.batch_chunks],
                            use_cache=False,
                        )
                        student_losses.append(float(out.loss.item()))
            finally:
                for h in handles:
                    h.remove()

            student_H = torch.cat(student_h, dim=0)
            s_loss = float(np.mean(student_losses))
            s_ppl = math.exp(s_loss)
            diff_sq = float(((student_H - teacher_H) ** 2).sum())
            norm_sq = float((teacher_H ** 2).sum())
            rel_h = diff_sq / max(norm_sq, 1e-12)
            ppl_inc = (s_ppl - teacher_ppl) / teacher_ppl * 100

            results[variant][str(kf)] = {
                "keep_frac": kf if variant != "fp8_only" else 1.0,
                "k": int(actual_keep),
                "value_only_compression": float(value_only_ratio),
                "student_loss": s_loss,
                "student_ppl": s_ppl,
                "ppl_increase_pct": ppl_inc,
                "final_hidden_relMSE": rel_h,
            }
            print(
                f"[sparse_fp8]   {variant:18s} keep={kf if variant!='fp8_only' else 1.0} k={actual_keep}: "
                f"value-only {value_only_ratio:.1f}x  "
                f"ppl={s_ppl:.3f} (+{ppl_inc:.1f}%)  hid_relMSE={rel_h:.4f}",
                flush=True,
            )

            if variant == "fp8_only":
                # Same result regardless of keep_frac -- copy and break.
                for kf2 in args.keep_fracs:
                    if str(kf2) != str(kf):
                        results[variant][str(kf2)] = dict(results[variant][str(kf)])
                break

    summary = {
        "model_path": args.model_path,
        "hidden_size": H,
        "n_layers": n_layers,
        "num_experts": E,
        "test_tokens": n_test_tok,
        "domain": "lcc",
        "keep_fracs": args.keep_fracs,
        "variants": args.variants,
        "teacher": {"loss": teacher_loss, "ppl": teacher_ppl},
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir/'summary.json'}", flush=True)

    # Markdown.
    lines = []
    lines.append(f"# Combine-side topk_l2 + FP8 stack (Qwen3-30B-A3B, lcc, {n_layers} MoE layers)")
    lines.append("")
    lines.append(f"Teacher PPL = {teacher_ppl:.3f}; held-out lcc test = {n_test_tok} tokens.")
    lines.append("")
    lines.append("FP8 = per-row E4M3 round-trip with amax/448 scaling (DeepEP/MoRI-style production recipe).")
    lines.append("")
    lines.append("## Sweep")
    lines.append("")
    lines.append("| variant | keep_frac | k | value-only ratio | student PPL | PPL +% | hidden relMSE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for v in args.variants:
        for kf in args.keep_fracs:
            r = results[v].get(str(kf))
            if not r:
                continue
            lines.append(
                f"| {v} | {r['keep_frac']:.4f} | {r['k']} | "
                f"{r['value_only_compression']:.1f}x | {r['student_ppl']:.3f} | "
                f"{r['ppl_increase_pct']:+.1f}% | {r['final_hidden_relMSE']:.4f} |"
            )
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"wrote {out_dir/'report.md'}", flush=True)

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for variant in args.variants:
            ratios, ppls, rels = [], [], []
            for kf in args.keep_fracs:
                r = results[variant].get(str(kf))
                if not r:
                    continue
                ratios.append(r["value_only_compression"])
                ppls.append(r["student_ppl"])
                rels.append(r["final_hidden_relMSE"])
            axes[0].plot(ratios, ppls, marker="o", label=variant)
            axes[1].plot(ratios, rels, marker="o", label=variant)
        axes[0].axhline(teacher_ppl, color="red", linestyle="--", linewidth=0.8, label=f"teacher = {teacher_ppl:.2f}")
        axes[0].set_xlabel("value-only compression ratio")
        axes[0].set_ylabel("test PPL")
        axes[0].set_xscale("log", base=2)
        axes[0].set_yscale("log")
        axes[0].set_title("PPL")
        axes[0].grid(True, alpha=0.3, which="both")
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("value-only compression ratio")
        axes[1].set_ylabel("final hidden relMSE")
        axes[1].set_xscale("log", base=2)
        axes[1].set_yscale("log")
        axes[1].set_title("Final hidden drift")
        axes[1].grid(True, alpha=0.3, which="both")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"Stacking FP8 on combine sparsification (Qwen3-30B-A3B, lcc, {n_layers} layers)")
        fig.tight_layout()
        fig.savefig(out_dir / "stack_curve.png", dpi=140)
        plt.close(fig)
        print(f"wrote {out_dir/'stack_curve.png'}", flush=True)
    except Exception as e:
        print(f"plot failed: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
