"""Phase A: error-feedback variants for combine-side sparsification.

Compare four sparsifiers (all applied to each expert's output, all 48 MoE
layers, on lcc test set):

  1. naive_topk    : keep top-k magnitude, zero rest. (baseline = prior sweep)
  2. topk_l2       : keep top-k magnitude, rescale kept values so that the row
                     L2 norm equals the un-sparsified row L2 norm.
  3. topk_l1       : keep top-k magnitude, rescale to preserve L1 norm.
  4. stochastic    : sample k positions WITHOUT replacement with probability
                     proportional to |v_i|; divide kept values by their
                     sampling probability so E[v_kept] = v_original (unbiased
                     estimator). Standard technique from sparse comm literature
                     (Wangni et al. 2018, Stich et al. 2018).

Sweep keep_frac in {0.25, 0.125, 0.0625} (= 4x, 8x, 16x value-only compression).

Outputs (under eval_results/compression_lowrank_stage0_ef_variants/):
  - summary.json
  - report.md
  - sweep_curve.png

Run:
  CUDA_VISIBLE_DEVICES=0,1,2,3 /home/lzy/miniconda3/envs/atom/bin/python \
      eval/compression/stage0_combine_ef_variants.py
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
OUT_DIR = Path("eval_results/compression_lowrank_stage0_ef_variants")
LB_DIR = Path("/home/lzy/datasets/moe_benchmarks/longbench/extracted/data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--chunk-len", type=int, default=512)
    p.add_argument("--test-chunks", type=int, default=8)
    p.add_argument("--batch-chunks", type=int, default=4)
    p.add_argument("--keep-fracs", type=float, nargs="+", default=[0.25, 0.125, 0.0625])
    p.add_argument("--variants", nargs="+", default=["naive_topk", "topk_l2", "topk_l1", "stochastic"])
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


def chunk_ids(ids: list[int], chunk_len: int, n_chunks: int) -> list[list[int]]:
    out, pos = [], 0
    while pos + chunk_len <= len(ids) and len(out) < n_chunks:
        out.append(ids[pos : pos + chunk_len])
        pos += chunk_len
    return out


# ---------------- Sparsifiers ----------------

def sparsify_naive_topk(x: torch.Tensor, k: int) -> torch.Tensor:
    """Zero everything outside top-k magnitudes."""
    abs_x = x.abs()
    _, idx = abs_x.topk(k, dim=-1)
    sparse = torch.zeros_like(x)
    sparse.scatter_(-1, idx, x.gather(-1, idx))
    return sparse


def sparsify_topk_l2(x: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k magnitudes, then rescale to preserve per-row L2 norm."""
    abs_x = x.abs()
    _, idx = abs_x.topk(k, dim=-1)
    sparse = torch.zeros_like(x)
    sparse.scatter_(-1, idx, x.gather(-1, idx))
    # per-row L2 norms (work in fp32 for stability).
    full_n2 = (x.float() ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    sparse_n2 = (sparse.float() ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    scale = (full_n2 / sparse_n2).to(x.dtype)
    return sparse * scale


def sparsify_topk_l1(x: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k magnitudes, then rescale to preserve per-row L1 norm."""
    abs_x = x.abs()
    _, idx = abs_x.topk(k, dim=-1)
    sparse = torch.zeros_like(x)
    sparse.scatter_(-1, idx, x.gather(-1, idx))
    full_n1 = x.float().abs().sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sparse_n1 = sparse.float().abs().sum(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (full_n1 / sparse_n1).to(x.dtype)
    return sparse * scale


def sparsify_stochastic(x: torch.Tensor, k: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample k positions with prob proportional to |v_i|, scale by 1/p_i.

    This is an unbiased compressor: E[output] = x. Variance depends on the
    magnitude distribution of x. Implemented per row via Gumbel top-k for
    sampling without replacement weighted by |v_i|.
    """
    fx = x.float()
    abs_x = fx.abs().clamp_min(1e-12)
    log_p = abs_x.log()
    # Gumbel-top-k for weighted sampling without replacement.
    g = -torch.empty_like(log_p).exponential_().log()
    scores = log_p + g
    _, idx = scores.topk(k, dim=-1)
    sampled_abs = abs_x.gather(-1, idx)
    # Approximate per-row probability of being in top-k.
    # For sampling-without-replacement the exact p is hard; use the standard
    # approximation p_i ~ k * |v_i| / sum(|v|). This is accurate when k <<
    # number of significant entries, which is our regime.
    row_sum = abs_x.sum(dim=-1, keepdim=True)
    p = (k * sampled_abs / row_sum).clamp_max(1.0)  # cap at 1 so we never amplify forever
    gathered = fx.gather(-1, idx)
    scaled = (gathered / p).to(x.dtype)
    sparse = torch.zeros_like(x)
    sparse.scatter_(-1, idx, scaled)
    return sparse


VARIANT_FN = {
    "naive_topk": sparsify_naive_topk,
    "topk_l2": sparsify_topk_l2,
    "topk_l1": sparsify_topk_l1,
    "stochastic": sparsify_stochastic,
}


# ---------------- Main ----------------

def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[ef_variants] loading {args.model_path}", flush=True)
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
    print(f"[ef_variants] H={H}, n_layers={n_layers}, num_experts={E}", flush=True)

    moe_blocks = [model.model.layers[L].mlp for L in range(n_layers)]
    all_experts: list = []
    for L in range(n_layers):
        for e in range(E):
            all_experts.append(moe_blocks[L].experts[e])

    # Tokenize lcc.
    text = load_lcc_text(min_chars=1_400_000)
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    if not isinstance(ids, list):
        ids = list(ids)
    skip = 16 * args.chunk_len
    test_chunks = chunk_ids(ids[skip:], args.chunk_len, args.test_chunks)
    n_test_tok = args.test_chunks * args.chunk_len
    print(f"[ef_variants] test_tokens={n_test_tok}", flush=True)

    # Teacher.
    print("[ef_variants] teacher forward ...", flush=True)
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
    print(f"[ef_variants] teacher: loss={teacher_loss:.4f}  ppl={teacher_ppl:.3f}", flush=True)

    # Sweep.
    results: dict[str, dict[str, dict]] = {v: {} for v in args.variants}
    for variant in args.variants:
        fn = VARIANT_FN[variant]
        for kf in args.keep_fracs:
            k = max(1, int(round(H * kf)))
            print(f"[ef_variants] variant={variant} keep_frac={kf} (k={k}) ...", flush=True)
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
            student_loss = float(np.mean(student_losses))
            student_ppl = math.exp(student_loss)
            diff_sq = float(((student_H - teacher_H) ** 2).sum())
            norm_sq = float((teacher_H ** 2).sum())
            rel_h = diff_sq / max(norm_sq, 1e-12)
            ppl_inc = (student_ppl - teacher_ppl) / teacher_ppl * 100

            results[variant][str(kf)] = {
                "keep_frac": kf,
                "k": int(k),
                "student_loss": student_loss,
                "student_ppl": student_ppl,
                "ppl_increase_pct": ppl_inc,
                "final_hidden_relMSE": rel_h,
            }
            print(
                f"[ef_variants]   {variant:12s} keep={kf} (k={k}): "
                f"ppl={student_ppl:.3f} (+{ppl_inc:.1f}%) hid_relMSE={rel_h:.4f}",
                flush=True,
            )

    # Decide winner at keep_frac = middle (smallest if not exactly middle).
    if len(args.keep_fracs) >= 2:
        target_kf = args.keep_fracs[len(args.keep_fracs) // 2]
    else:
        target_kf = args.keep_fracs[0]
    winners = sorted(
        args.variants,
        key=lambda v: results[v][str(target_kf)]["student_ppl"],
    )
    winner = winners[0]
    print(f"[ef_variants] WINNER at keep_frac={target_kf}: {winner} "
          f"(ppl={results[winner][str(target_kf)]['student_ppl']:.3f})", flush=True)

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
        "winner_at_keep_frac": float(target_kf),
        "winner": winner,
        "winner_ranking": winners,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir/'summary.json'}", flush=True)

    # Markdown report.
    lines = []
    lines.append(f"# Combine-side sparsification: error-feedback variants (Qwen3-30B-A3B, lcc, {n_layers} layers)")
    lines.append("")
    lines.append(f"Teacher PPL = {teacher_ppl:.3f}; held-out lcc test = {n_test_tok} tokens.")
    lines.append("")
    lines.append("## PPL by (variant, keep_frac)")
    lines.append("")
    hdr = "| variant | " + " | ".join(f"keep={kf}" for kf in args.keep_fracs) + " |"
    sep = "|---" * (1 + len(args.keep_fracs)) + "|"
    lines.append(hdr); lines.append(sep)
    for v in args.variants:
        cells = []
        for kf in args.keep_fracs:
            r = results[v][str(kf)]
            cells.append(f"{r['student_ppl']:.3f} (+{r['ppl_increase_pct']:.1f}%)")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## hidden relMSE by (variant, keep_frac)")
    lines.append("")
    lines.append(hdr); lines.append(sep)
    for v in args.variants:
        cells = []
        for kf in args.keep_fracs:
            r = results[v][str(kf)]
            cells.append(f"{r['final_hidden_relMSE']:.4f}")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"## Winner at keep_frac = {target_kf}")
    lines.append("")
    lines.append(f"**{winner}** (ranking by PPL ascending: {winners})")
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"wrote {out_dir/'report.md'}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ratios = [1 / kf for kf in args.keep_fracs]  # value-only ratio
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for v in args.variants:
            ppls = [results[v][str(kf)]["student_ppl"] for kf in args.keep_fracs]
            rels = [results[v][str(kf)]["final_hidden_relMSE"] for kf in args.keep_fracs]
            axes[0].plot(ratios, ppls, marker="o", label=v)
            axes[1].plot(ratios, rels, marker="o", label=v)
        axes[0].axhline(teacher_ppl, color="red", linestyle="--", linewidth=0.8, label=f"teacher = {teacher_ppl:.2f}")
        axes[0].set_xlabel("value-only compression ratio (1/keep_frac)")
        axes[0].set_ylabel("test PPL")
        axes[0].set_xscale("log", base=2)
        axes[0].set_yscale("log")
        axes[0].set_title("PPL")
        axes[0].grid(True, alpha=0.3, which="both")
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("value-only compression ratio (1/keep_frac)")
        axes[1].set_ylabel("final hidden relMSE")
        axes[1].set_xscale("log", base=2)
        axes[1].set_yscale("log")
        axes[1].set_title("Final hidden drift")
        axes[1].grid(True, alpha=0.3, which="both")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"Combine-side EF variants on Qwen3-30B-A3B (lcc, {n_layers} layers)")
        fig.tight_layout()
        fig.savefig(out_dir / "sweep_curve.png", dpi=140)
        plt.close(fig)
        print(f"wrote {out_dir/'sweep_curve.png'}", flush=True)
    except Exception as e:
        print(f"plot failed: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
