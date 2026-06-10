"""Direction B: combine-side top-magnitude sparsification sweep.

Hypothesis: errors added to the residual stream at combine (= what each expert
outputs into the per-token sum) are absorbed by the residual writes rather than
amplified multiplicatively through subsequent layers (as dispatch-side errors
were). So even aggressive sparsification of expert outputs should leave PPL
much closer to the teacher than equivalently-aggressive dispatch-side
compression.

Method:
  For every expert module Qwen3MoeMLP in every MoE layer (48 x 128 = 6144 hooks),
  install a forward hook that replaces each row of the expert output with its
  top-k magnitude sparsification (the remaining 1 - keep_frac of entries are
  zeroed). The downstream weighted-sum into `final_hidden_states` continues
  normally with the sparsified values.

  This simulates: each token-expert pair's combine payload sent over the wire
  is reduced to its top-k entries + indices. Receiver reconstructs by placing
  the values at the indices and zeros elsewhere.

Sweep: keep_frac in {0.5, 0.25, 0.125, 0.0625, 0.03125}.
Compare against teacher (no sparsification).

Outputs (under eval_results/compression_lowrank_stage0_combine_sparsify/):
  - summary.json
  - report.md
  - sweep_curve.png

Run:
  CUDA_VISIBLE_DEVICES=0,1,2,3 /home/lzy/miniconda3/envs/atom/bin/python \
      eval/compression/stage0_combine_sparsify_sweep.py
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
OUT_DIR = Path("eval_results/compression_lowrank_stage0_combine_sparsify")
LB_DIR = Path("/home/lzy/datasets/moe_benchmarks/longbench/extracted/data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--chunk-len", type=int, default=512)
    p.add_argument("--test-chunks", type=int, default=8)
    p.add_argument("--batch-chunks", type=int, default=4)
    p.add_argument(
        "--keep-fracs",
        type=float,
        nargs="+",
        default=[0.5, 0.25, 0.125, 0.0625, 0.03125],
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


def chunk_ids(ids: list[int], chunk_len: int, n_chunks: int) -> list[list[int]]:
    out, pos = [], 0
    while pos + chunk_len <= len(ids) and len(out) < n_chunks:
        out.append(ids[pos : pos + chunk_len])
        pos += chunk_len
    return out


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[combine_sparsify] loading {args.model_path}", flush=True)
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
    print(f"[combine_sparsify] H={H}, n_layers={n_layers}, num_experts={E}", flush=True)

    moe_blocks = [model.model.layers[L].mlp for L in range(n_layers)]
    # Each MoE block has .experts as ModuleList of Qwen3MoeMLP. Collect references.
    all_experts: list = []
    for L in range(n_layers):
        for e in range(E):
            all_experts.append(moe_blocks[L].experts[e])
    print(f"[combine_sparsify] total expert modules to hook = {len(all_experts)}", flush=True)

    # Tokenize lcc test set (no calibration needed — sparsification is data-free).
    text = load_lcc_text(min_chars=1_400_000)
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    if not isinstance(ids, list):
        ids = list(ids)
    # Skip the first 16 chunks (used as calibration in prior experiments) to keep the
    # held-out test split identical across SVD / routing-aware / combine sweeps.
    skip = 16 * args.chunk_len
    test_chunks = chunk_ids(ids[skip:], args.chunk_len, args.test_chunks)
    n_test_tok = args.test_chunks * args.chunk_len
    print(f"[combine_sparsify] test_tokens={n_test_tok} (held out, same split as prior sweeps)", flush=True)

    # -------------------- Teacher forward (no sparsification) ---------------------
    print("[combine_sparsify] teacher forward ...", flush=True)
    teacher_h: list[torch.Tensor] = []

    def teacher_norm_hook(_m, _inputs, output):
        teacher_h.append(output.detach().to("cpu", dtype=torch.float32).reshape(-1, output.shape[-1]))

    teacher_handles = [model.model.norm.register_forward_hook(teacher_norm_hook)]

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
    for h in teacher_handles:
        h.remove()

    teacher_H = torch.cat(teacher_h, dim=0)
    teacher_loss = float(np.mean(teacher_losses))
    teacher_ppl = math.exp(teacher_loss)
    print(f"[combine_sparsify] teacher: loss={teacher_loss:.4f}  ppl={teacher_ppl:.3f}", flush=True)

    # -------------------- Student forwards (one per keep_frac) -------------------
    results = {}
    for kf in args.keep_fracs:
        k = max(1, int(round(H * kf)))
        compression_value_only = H / k
        # With value (2B bf16) + index (2B uint16), payload per kept entry = 4B.
        # Original per-row payload = 2*H. So effective row compression ratio ~ 2H / (4k) = H / (2k).
        compression_value_plus_index = H / (2 * k)
        print(
            f"[combine_sparsify] sweep: keep_frac={kf} -> k={k}/{H} "
            f"(value-only {compression_value_only:.2f}x, value+index ~{compression_value_plus_index:.2f}x)",
            flush=True,
        )

        student_h: list[torch.Tensor] = []

        def student_norm_hook(_m, _inputs, output):
            student_h.append(output.detach().to("cpu", dtype=torch.float32).reshape(-1, output.shape[-1]))

        def make_sparsify_hook(k_inner: int):
            def hook(_m, _inputs, output):
                # output: [n_tokens_for_this_expert, H], bf16
                if output is None or output.numel() == 0:
                    return output
                # Top-k by magnitude per row.
                vals, idx = output.abs().topk(k_inner, dim=-1)
                # Build sparsified output: keep at those indices, zero elsewhere.
                sparse = torch.zeros_like(output)
                gathered = output.gather(-1, idx)
                sparse.scatter_(-1, idx, gathered)
                return sparse
            return hook

        handles = []
        for expert_mod in all_experts:
            handles.append(expert_mod.register_forward_hook(make_sparsify_hook(k)))
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
        ppl_inc_pct = (student_ppl - teacher_ppl) / teacher_ppl * 100

        results[str(kf)] = {
            "keep_frac": float(kf),
            "k": int(k),
            "compression_value_only": compression_value_only,
            "compression_value_plus_index": compression_value_plus_index,
            "student_loss": student_loss,
            "student_ppl": student_ppl,
            "ppl_increase_pct": ppl_inc_pct,
            "final_hidden_relMSE": rel_h,
        }
        print(
            f"[combine_sparsify]   keep={kf} (k={k}): "
            f"loss={student_loss:.4f} ppl={student_ppl:.3f} (+{ppl_inc_pct:.1f}% vs teacher) "
            f"hid_relMSE={rel_h:.4f}",
            flush=True,
        )

    # -------------------- Output --------------------
    summary = {
        "model_path": args.model_path,
        "hidden_size": H,
        "n_layers": n_layers,
        "num_experts": E,
        "test_tokens": n_test_tok,
        "domain": "lcc",
        "keep_fracs": args.keep_fracs,
        "teacher": {"loss": teacher_loss, "ppl": teacher_ppl},
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir/'summary.json'}", flush=True)

    lines = []
    lines.append(f"# Combine-side top-magnitude sparsification sweep (Qwen3-30B-A3B, lcc, {n_layers} MoE layers)")
    lines.append("")
    lines.append(f"- Teacher PPL = {teacher_ppl:.3f}; test = {n_test_tok} held-out lcc tokens.")
    lines.append(f"- Each of the {n_layers * E} expert modules has a forward hook that sparsifies its output row-wise to the top-k magnitude entries.")
    lines.append("- 'value-only' compression assumes infinite-bandwidth indices (purely the kept fraction).")
    lines.append("- 'value+index' compression assumes we ALSO send int16 indices alongside each kept bf16 value (so the per-kept payload is 4 bytes vs the original 2H bytes per row).")
    lines.append("")
    lines.append("## Sweep")
    lines.append("")
    lines.append("| keep_frac | k | value-only | value+index | student PPL | PPL +% | final hidden relMSE |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for kf in args.keep_fracs:
        r = results[str(kf)]
        lines.append(
            f"| {kf:.4f} | {r['k']} | {r['compression_value_only']:.2f}x | "
            f"{r['compression_value_plus_index']:.2f}x | {r['student_ppl']:.3f} | "
            f"{r['ppl_increase_pct']:+.1f}% | {r['final_hidden_relMSE']:.4f} |"
        )
    lines.append("")
    lines.append("## Cross-comparison vs dispatch-side SVD (from prior sweep)")
    lines.append("")
    lines.append("From `eval_results/compression_lowrank_stage0_full_sweep/`:")
    lines.append("- dispatch SVD 2x:  teacher 3.15 -> 6.92 PPL (+119%)")
    lines.append("- dispatch SVD 4x:  teacher 3.15 -> 25.6 PPL (+711%)")
    lines.append("- dispatch SVD 8x:  teacher 3.15 -> 4296 PPL (+136 158%)")
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"wrote {out_dir/'report.md'}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ratios = [results[str(kf)]["compression_value_plus_index"] for kf in args.keep_fracs]
        ppls = [results[str(kf)]["student_ppl"] for kf in args.keep_fracs]
        rels = [results[str(kf)]["final_hidden_relMSE"] for kf in args.keep_fracs]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(ratios, ppls, marker="o", color="C2", label="combine sparsify")
        # Overlay dispatch SVD prior numbers for context.
        axes[0].plot([2, 4, 8, 16], [6.918, 25.561, 4296.199, 2128823.862],
                     marker="x", color="C0", linestyle="--", label="dispatch SVD (prior)")
        axes[0].axhline(teacher_ppl, color="red", linestyle="--", linewidth=0.8, label=f"teacher = {teacher_ppl:.2f}")
        axes[0].set_xlabel("compression ratio (value+index)")
        axes[0].set_ylabel("test PPL")
        axes[0].set_xscale("log", base=2)
        axes[0].set_yscale("log")
        axes[0].set_title("Test PPL")
        axes[0].grid(True, alpha=0.3, which="both")
        axes[0].legend(fontsize=8)

        axes[1].plot(ratios, rels, marker="o", color="C2", label="combine sparsify")
        axes[1].plot([2, 4, 8, 16], [0.1523, 0.2652, 0.9941, 1.8479],
                     marker="x", color="C0", linestyle="--", label="dispatch SVD (prior)")
        axes[1].set_xlabel("compression ratio (value+index)")
        axes[1].set_ylabel("final hidden relMSE")
        axes[1].set_xscale("log", base=2)
        axes[1].set_yscale("log")
        axes[1].set_title("Final hidden drift")
        axes[1].grid(True, alpha=0.3, which="both")
        axes[1].legend(fontsize=8)

        fig.suptitle(f"Combine-side sparsify vs dispatch-side SVD (Qwen3-30B-A3B, lcc test, {n_layers} layers)")
        fig.tight_layout()
        fig.savefig(out_dir / "sweep_curve.png", dpi=140)
        plt.close(fig)
        print(f"wrote {out_dir/'sweep_curve.png'}", flush=True)
    except Exception as e:
        print(f"plot failed: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
