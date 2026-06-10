"""Phase B: cross-domain validation of topk_l2 sparsifier.

Run the Phase A winner (topk_l2) and the baseline (naive_topk) on four held-out
test sets covering distinct domains:

  - lcc          : code (LongBench long code completion)
  - multifieldqa : mixed-domain English QA
  - dureader     : Chinese long QA (the worst domain in Stage 0 SVD; tests robustness)
  - gov_report   : long English government reports

At each of {4x, 8x} compression. Goal: confirm topk_l2's combine-side win is
not lcc-specific.

Outputs (under eval_results/compression_lowrank_stage0_cross_domain/):
  - summary.json
  - report.md
  - cross_domain_curve.png

Run:
  CUDA_VISIBLE_DEVICES=0,1,2,3 /home/lzy/miniconda3/envs/atom/bin/python \
      eval/compression/stage0_combine_cross_domain.py
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
OUT_DIR = Path("eval_results/compression_lowrank_stage0_cross_domain")
LB_DIR = Path("/home/lzy/datasets/moe_benchmarks/longbench/extracted/data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--chunk-len", type=int, default=512)
    p.add_argument("--test-chunks", type=int, default=8)
    p.add_argument("--batch-chunks", type=int, default=4)
    p.add_argument("--keep-fracs", type=float, nargs="+", default=[0.25, 0.125])
    p.add_argument("--variants", nargs="+", default=["naive_topk", "topk_l2"])
    p.add_argument(
        "--domains",
        nargs="+",
        default=["lcc", "multifieldqa", "dureader", "gov_report"],
    )
    p.add_argument("--gpu-mem-gib", type=int, default=22)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_lb_text(fname: str, min_chars: int = 1_400_000) -> str:
    pieces, total = [], 0
    with (LB_DIR / fname).open("r", encoding="utf-8") as f:
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


def get_domain_text(name: str) -> str:
    return {
        "lcc": load_lb_text("lcc.jsonl"),
        "multifieldqa": load_lb_text("multifieldqa_en.jsonl"),
        "dureader": load_lb_text("dureader.jsonl"),
        "gov_report": load_lb_text("gov_report.jsonl"),
    }[name]


def chunk_ids(ids: list[int], chunk_len: int, n_chunks: int, skip: int = 0) -> list[list[int]]:
    out, pos = [], skip
    while pos + chunk_len <= len(ids) and len(out) < n_chunks:
        out.append(ids[pos : pos + chunk_len])
        pos += chunk_len
    return out


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


VARIANT_FN = {"naive_topk": sparsify_naive_topk, "topk_l2": sparsify_topk_l2}


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[cross_domain] loading {args.model_path}", flush=True)
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
    print(f"[cross_domain] H={H}, n_layers={n_layers}, num_experts={E}", flush=True)

    moe_blocks = [model.model.layers[L].mlp for L in range(n_layers)]
    all_experts: list = []
    for L in range(n_layers):
        for e in range(E):
            all_experts.append(moe_blocks[L].experts[e])

    # Build per-domain held-out chunks (skip first 16 chunks to be consistent with
    # prior experiments' calibration/test split).
    domain_chunks: dict[str, list[list[int]]] = {}
    skip = 16 * args.chunk_len
    for dn in args.domains:
        text = get_domain_text(dn)
        ids = tokenizer(text, return_tensors=None)["input_ids"]
        if not isinstance(ids, list):
            ids = list(ids)
        if len(ids) < skip + args.test_chunks * args.chunk_len:
            print(f"[cross_domain] warn: {dn} has only {len(ids)} tokens", flush=True)
            ch = chunk_ids(ids, args.chunk_len, args.test_chunks, skip=0)
        else:
            ch = chunk_ids(ids, args.chunk_len, args.test_chunks, skip=skip)
        domain_chunks[dn] = ch
        print(f"[cross_domain] {dn}: {len(text)} chars -> {len(ids)} tokens -> {len(ch)} held-out chunks", flush=True)

    # Compute teacher PPL per domain (need it to report relative drops cleanly).
    results: dict = {"teacher": {}, "variants": {v: {dn: {} for dn in args.domains} for v in args.variants}}

    for dn in args.domains:
        chunks = domain_chunks[dn]
        if not chunks:
            continue
        print(f"[cross_domain] teacher on {dn} ...", flush=True)
        teacher_h: list[torch.Tensor] = []

        def teacher_norm_hook(_m, _inputs, output):
            teacher_h.append(output.detach().to("cpu", dtype=torch.float32).reshape(-1, output.shape[-1]))

        th = model.model.norm.register_forward_hook(teacher_norm_hook)
        teacher_losses = []
        with torch.no_grad():
            ids_t = torch.tensor(chunks, dtype=torch.long, device=embed_dev)
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
        t_loss = float(np.mean(teacher_losses))
        t_ppl = math.exp(t_loss)
        results["teacher"][dn] = {"loss": t_loss, "ppl": t_ppl, "tokens": int(teacher_H.shape[0])}
        print(f"[cross_domain]   teacher {dn}: loss={t_loss:.4f} ppl={t_ppl:.3f} tokens={teacher_H.shape[0]}", flush=True)

        for variant in args.variants:
            fn = VARIANT_FN[variant]
            for kf in args.keep_fracs:
                k = max(1, int(round(H * kf)))
                print(f"[cross_domain] {dn} / {variant} / keep_frac={kf} (k={k}) ...", flush=True)
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
                        ids_t = torch.tensor(chunks, dtype=torch.long, device=embed_dev)
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
                ppl_inc = (s_ppl - t_ppl) / t_ppl * 100
                results["variants"][variant][dn][str(kf)] = {
                    "keep_frac": kf,
                    "k": int(k),
                    "student_loss": s_loss,
                    "student_ppl": s_ppl,
                    "ppl_increase_pct": ppl_inc,
                    "final_hidden_relMSE": rel_h,
                }
                print(
                    f"[cross_domain]   {dn:14s} {variant:12s} keep={kf} k={k}: "
                    f"ppl={s_ppl:.3f} (+{ppl_inc:.1f}%) hid_relMSE={rel_h:.4f}",
                    flush=True,
                )

    summary = {
        "model_path": args.model_path,
        "hidden_size": H,
        "n_layers": n_layers,
        "num_experts": E,
        "test_chunks_per_domain": args.test_chunks,
        "chunk_len": args.chunk_len,
        "keep_fracs": args.keep_fracs,
        "variants": args.variants,
        "domains": args.domains,
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir/'summary.json'}", flush=True)

    # Markdown.
    lines = []
    lines.append("# Cross-domain validation of combine-side sparsification (Qwen3-30B-A3B)")
    lines.append("")
    lines.append("All 48 MoE layers' expert outputs are sparsified; held-out test set per domain.")
    lines.append("")
    lines.append("## Teacher PPL per domain")
    lines.append("| domain | teacher PPL |")
    lines.append("|---|---:|")
    for dn in args.domains:
        if dn in results["teacher"]:
            lines.append(f"| {dn} | {results['teacher'][dn]['ppl']:.3f} |")
    lines.append("")
    for kf in args.keep_fracs:
        lines.append(f"## PPL @ keep_frac = {kf} (value-only {1/kf:.0f}x compression)")
        lines.append("")
        hdr = "| domain | teacher | " + " | ".join(args.variants) + " | best PPL +% |"
        sep = "|---|---:|" + "---:|" * len(args.variants) + "---:|"
        lines.append(hdr); lines.append(sep)
        for dn in args.domains:
            if dn not in results["teacher"]:
                continue
            t = results["teacher"][dn]["ppl"]
            cells = [f"{t:.3f}"]
            best_inc = math.inf
            for v in args.variants:
                r = results["variants"][v][dn].get(str(kf))
                if r is None:
                    cells.append("-")
                else:
                    cells.append(f"{r['student_ppl']:.3f} (+{r['ppl_increase_pct']:.1f}%)")
                    best_inc = min(best_inc, r['ppl_increase_pct'])
            cells.append(f"+{best_inc:.1f}%")
            lines.append(f"| {dn} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("If `topk_l2` strictly beats `naive_topk` across all four domains, the L2 rescale fix is domain-independent.")
    lines.append("If `dureader` (Chinese) and `gov_report` (long reports) have similar PPL +% to `lcc` (code), the combine-side advantage is not lcc-specific.")
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"wrote {out_dir/'report.md'}", flush=True)

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(args.keep_fracs), figsize=(5 * len(args.keep_fracs), 4), sharey=True)
        if len(args.keep_fracs) == 1:
            axes = [axes]
        for ax, kf in zip(axes, args.keep_fracs):
            xs = np.arange(len(args.domains))
            width = 0.35
            for i, v in enumerate(args.variants):
                ys = []
                for dn in args.domains:
                    r = results["variants"][v][dn].get(str(kf))
                    ys.append(r["ppl_increase_pct"] if r else 0)
                ax.bar(xs + (i - 0.5) * width, ys, width, label=v)
            ax.set_xticks(xs)
            ax.set_xticklabels(args.domains, rotation=20)
            ax.set_ylabel("PPL increase %")
            ax.set_title(f"keep_frac={kf} ({1/kf:.0f}x)")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Combine-side sparsification: PPL increase across domains")
        fig.tight_layout()
        fig.savefig(out_dir / "cross_domain_curve.png", dpi=140)
        plt.close(fig)
        print(f"wrote {out_dir/'cross_domain_curve.png'}", flush=True)
    except Exception as e:
        print(f"plot failed: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
