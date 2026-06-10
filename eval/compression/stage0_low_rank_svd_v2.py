"""Stage 0 v2: larger calibration set + per-expert spectrum.

Differences vs v1:
  - Pull text from LongBench passages (long, varied) instead of short hand-prompts,
    so N >> d after tokenization. Target ~ 25-32 K non-pad tokens (N / d >= 12-15).
  - Compute per-expert spectrum (tokens whose top-1 destination is expert e) as
    a bonus signal: per-destination compressors could in principle compress more.

Outputs (under eval_results/compression_lowrank_stage0_v2/):
  - activations_layer{L}.pt       : float32 [N, d] dispatch input
  - top1_layer{L}.pt              : int64 [N] top-1 expert assignment
  - svd_summary.json
  - svd_curve.png                 : per-layer global spectrum
  - svd_curve_per_expert.png      : per-layer min/median/max expert spectrum
  - report.md

Run:
  /home/lzy/miniconda3/envs/atom/bin/python eval/compression/stage0_low_rank_svd_v2.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

MODEL_PATH = "/home/lzy/models/Qwen3-30B-A3B"
OUT_DIR = Path("eval_results/compression_lowrank_stage0_v2")
LONGBENCH_DIR = Path("/home/lzy/datasets/moe_benchmarks/longbench/extracted/data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument(
        "--layers", type=int, nargs="+", default=[2, 6, 12, 24, 36, 46],
        help="MoE layer indices to hook (0-indexed)",
    )
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-chunks", type=int, default=64, help="number of (seq_len)-token chunks of text to feed")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-mem-gib", type=int, default=22)
    p.add_argument(
        "--longbench-files",
        nargs="+",
        default=[
            "gov_report.jsonl", "multi_news.jsonl", "multifieldqa_en.jsonl", "lcc.jsonl",
        ],
    )
    return p.parse_args()


def setup_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_longbench_text(files: list[str]) -> str:
    """Concatenate the 'context' field of a handful of LongBench files into a
    single big text blob. Stop once we have plenty of chars (the tokenizer will
    later truncate)."""
    import json as _json
    pieces: list[str] = []
    total = 0
    target_chars = 800_000  # ~ 200K tokens worth, way more than we need
    for fname in files:
        path = LONGBENCH_DIR / fname
        if not path.exists():
            print(f"[warn] missing {path}", flush=True)
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = _json.loads(line)
                except Exception:
                    continue
                # LongBench schema: 'context' is the document, 'input' is the question.
                txt = rec.get("context") or rec.get("input") or ""
                if not isinstance(txt, str):
                    continue
                if len(txt) < 200:
                    continue
                pieces.append(txt)
                total += len(txt)
                if total >= target_chars:
                    break
        if total >= target_chars:
            break
    blob = "\n\n".join(pieces)
    print(f"[stage0_v2] loaded {len(pieces)} passages, {len(blob)} chars total", flush=True)
    return blob


def chunk_token_ids(token_ids: list[int], chunk_len: int, n_chunks: int) -> list[list[int]]:
    chunks = []
    pos = 0
    while pos + chunk_len <= len(token_ids) and len(chunks) < n_chunks:
        chunks.append(token_ids[pos : pos + chunk_len])
        pos += chunk_len
    return chunks


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    dtype = setup_dtype(args.dtype)

    print(f"[stage0_v2] loading {args.model_path}  dtype={args.dtype}", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_mem = {i: f"{args.gpu_mem_gib}GiB" for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_mem,
        trust_remote_code=True,
    )
    model.eval()
    cfg = model.config
    hidden_size = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    n_experts = int(cfg.num_experts)
    print(
        f"[stage0_v2] loaded: hidden_size={hidden_size}, num_hidden_layers={n_layers}, "
        f"num_experts={n_experts}, top_k={cfg.num_experts_per_tok}",
        flush=True,
    )

    layers_to_hook = [L for L in args.layers if 0 <= L < n_layers]
    print(f"[stage0_v2] hooking MoE input on layers {layers_to_hook}", flush=True)

    activations: dict[int, list[torch.Tensor]] = {L: [] for L in layers_to_hook}
    routings: dict[int, list[torch.Tensor]] = {L: [] for L in layers_to_hook}

    def make_mlp_pre_hook(layer_id: int):
        def hook(module, inputs):
            x = inputs[0].detach()
            activations[layer_id].append(x.to("cpu", dtype=torch.float32).reshape(-1, x.shape[-1]))
            return None
        return hook

    def make_gate_hook(layer_id: int):
        def hook(module, inputs, output):
            logits = output.detach()
            top1 = logits.argmax(dim=-1).to("cpu").reshape(-1)
            routings[layer_id].append(top1)
            return None
        return hook

    handles = []
    for L in layers_to_hook:
        layer = model.model.layers[L]
        mlp = layer.mlp
        handles.append(mlp.register_forward_pre_hook(make_mlp_pre_hook(L)))
        if hasattr(mlp, "gate"):
            handles.append(mlp.gate.register_forward_hook(make_gate_hook(L)))

    # Load and tokenize text.
    blob = load_longbench_text(args.longbench_files)
    token_ids = tokenizer(blob, return_tensors=None)["input_ids"]
    if not isinstance(token_ids, list):
        token_ids = list(token_ids)
    print(f"[stage0_v2] tokenized: {len(token_ids)} tokens", flush=True)

    chunks = chunk_token_ids(token_ids, args.max_seq_len, args.n_chunks)
    print(f"[stage0_v2] formed {len(chunks)} chunks of {args.max_seq_len} tokens each "
          f"-> {len(chunks)*args.max_seq_len} total tokens", flush=True)

    input_ids = torch.tensor(chunks, dtype=torch.long)
    # All positions are valid; attention mask is all-ones.
    attn_mask = torch.ones_like(input_ids)
    embed_dev = next(model.model.embed_tokens.parameters()).device
    input_ids = input_ids.to(embed_dev)
    attn_mask = attn_mask.to(embed_dev)

    bsz = args.batch_size
    with torch.no_grad():
        for i in range(0, input_ids.shape[0], bsz):
            ids = input_ids[i : i + bsz]
            am = attn_mask[i : i + bsz]
            _ = model(input_ids=ids, attention_mask=am, use_cache=False)
            print(f"[stage0_v2]   batch {i // bsz + 1}/{(input_ids.shape[0] + bsz - 1) // bsz} done", flush=True)

    for h in handles:
        h.remove()

    # All positions are kept (no padding).
    n_tok_total = int(input_ids.numel())
    print(f"[stage0_v2] kept tokens (no pad): {n_tok_total}", flush=True)

    summary: dict = {
        "model_path": args.model_path,
        "hidden_size": hidden_size,
        "num_hidden_layers": n_layers,
        "num_experts": n_experts,
        "top_k": int(cfg.num_experts_per_tok),
        "n_chunks": len(chunks),
        "chunk_len": int(args.max_seq_len),
        "n_kept_tokens": n_tok_total,
        "longbench_files": args.longbench_files,
        "layers": {},
        "go_criteria": {
            "explained_variance_at_d_over_4": 0.90,
            "rule": "GO if explained variance at l = hidden_size/4 >= 0.90 in ALL probed layers",
        },
    }

    import numpy as np

    per_expert_curves: dict[int, dict[int, np.ndarray]] = {}

    for L in layers_to_hook:
        chunks_l = activations[L]
        if not chunks_l:
            continue
        X = torch.cat(chunks_l, dim=0)  # [N, H]
        N, d = X.shape
        print(f"[stage0_v2] layer {L}: activation matrix shape ({N}, {d})", flush=True)

        torch.save({"X": X, "layer": L}, out_dir / f"activations_layer{L}.pt")
        top1 = torch.cat(routings[L], dim=0) if routings[L] else torch.zeros(N, dtype=torch.long)
        if top1.shape[0] != X.shape[0]:
            # In Qwen3MoE, `gate` is called with hidden_states.view(-1, H), so
            # the count should match; keep this guard for safety.
            n = min(top1.shape[0], X.shape[0])
            top1 = top1[:n]
            X = X[:n]
            N = n
        torch.save({"top1": top1, "layer": L}, out_dir / f"top1_layer{L}.pt")

        X_centered = X - X.mean(dim=0, keepdim=True)
        sv_centered = torch.linalg.svdvals(X_centered).cpu().numpy()
        ev_c = sv_centered ** 2
        cum_c = np.cumsum(ev_c) / ev_c.sum()

        marks = {"d/16": d // 16, "d/8": d // 8, "d/4": d // 4, "d/2": d // 2}
        layer_info: dict = {
            "shape": [int(N), int(d)],
            "singular_values_centered_first16": [float(x) for x in sv_centered[:16]],
            "singular_values_centered_last4": [float(x) for x in sv_centered[-4:]],
            "explained_variance_centered_at": {
                k: float(cum_c[min(v, len(cum_c)) - 1]) for k, v in marks.items()
            },
            "rank_at_explained_variance": {
                "0.90": int(np.searchsorted(cum_c, 0.90) + 1),
                "0.95": int(np.searchsorted(cum_c, 0.95) + 1),
                "0.99": int(np.searchsorted(cum_c, 0.99) + 1),
                "0.999": int(np.searchsorted(cum_c, 0.999) + 1),
            },
        }

        np.save(out_dir / f"cum_var_centered_layer{L}.npy", cum_c)
        np.save(out_dir / f"sv_centered_layer{L}.npy", sv_centered)

        # Per-expert spectrum. Use top-1 routed tokens grouped by destination expert.
        per_expert_curves[L] = {}
        per_expert_ev_d4 = []
        per_expert_counts = []
        for e in range(n_experts):
            mask = top1 == e
            n_e = int(mask.sum().item())
            if n_e < 64:  # too few samples for SVD on a 2048-dim space
                continue
            Xe = X[mask]
            Xec = Xe - Xe.mean(dim=0, keepdim=True)
            sv_e = torch.linalg.svdvals(Xec).cpu().numpy()
            ev_e = sv_e ** 2
            cum_e = np.cumsum(ev_e) / ev_e.sum() if ev_e.sum() > 0 else None
            if cum_e is None:
                continue
            # Cap at len; for n_e < d, the spectrum has length min(n_e, d).
            d4 = d // 4
            v = float(cum_e[min(d4, len(cum_e)) - 1])
            per_expert_curves[L][e] = cum_e
            per_expert_ev_d4.append(v)
            per_expert_counts.append(n_e)

        if per_expert_ev_d4:
            arr = np.array(per_expert_ev_d4)
            layer_info["per_expert"] = {
                "n_experts_with_at_least_64_tokens": int(len(arr)),
                "explained_variance_at_d_over_4": {
                    "min": float(arr.min()),
                    "p25": float(np.percentile(arr, 25)),
                    "median": float(np.percentile(arr, 50)),
                    "p75": float(np.percentile(arr, 75)),
                    "max": float(arr.max()),
                    "mean": float(arr.mean()),
                },
                "median_token_count_per_expert": int(np.median(per_expert_counts)),
                "max_token_count_per_expert": int(np.max(per_expert_counts)),
            }

        summary["layers"][str(L)] = layer_info

    # Apply Go/No-Go.
    all_ge_90 = True
    layer_ev_d4 = {}
    for L_str, info in summary["layers"].items():
        v = info["explained_variance_centered_at"]["d/4"]
        layer_ev_d4[L_str] = v
        if v < 0.90:
            all_ge_90 = False
    summary["go_criteria"]["per_layer_explained_variance_at_d_over_4"] = layer_ev_d4
    summary["go_criteria"]["verdict"] = "GO" if all_ge_90 else "NO-GO"

    with (out_dir / "svd_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[stage0_v2] wrote {out_dir/'svd_summary.json'}", flush=True)

    # Plots.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Global spectrum plot.
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
        for L in layers_to_hook:
            p = out_dir / f"cum_var_centered_layer{L}.npy"
            if not p.exists():
                continue
            cum = np.load(p)
            xs = np.arange(1, len(cum) + 1) / hidden_size
            ax.plot(xs, cum, label=f"layer {L}")
        ax.axvline(0.25, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(0.90, color="red", linestyle="--", linewidth=0.8)
        ax.set_xlabel("l / d (kept rank fraction)")
        ax.set_ylabel("cumulative explained variance (centered)")
        ax.set_title(
            f"MoE dispatch activation spectrum — Qwen3-30B-A3B (d={hidden_size}, "
            f"N={n_tok_total} tokens)"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "svd_curve.png", dpi=140)
        plt.close(fig)
        print(f"[stage0_v2] wrote {out_dir/'svd_curve.png'}", flush=True)

        # Per-expert plot: median/min/max envelope per layer.
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers_to_hook)))
        for c, L in zip(colors, layers_to_hook):
            curves = per_expert_curves.get(L, {})
            if not curves:
                continue
            # Each curve may have different length (min(n_e, d)); pad with last value
            # to the global d so we can stack.
            target_len = hidden_size
            padded = []
            for cum in curves.values():
                if len(cum) < target_len:
                    pad = np.full(target_len - len(cum), cum[-1])
                    padded.append(np.concatenate([cum, pad]))
                else:
                    padded.append(cum[:target_len])
            A = np.stack(padded, axis=0)  # [n_experts, d]
            med = np.median(A, axis=0)
            mn = np.min(A, axis=0)
            mx = np.max(A, axis=0)
            xs = np.arange(1, target_len + 1) / hidden_size
            ax.plot(xs, med, color=c, label=f"layer {L} (median)")
            ax.fill_between(xs, mn, mx, color=c, alpha=0.12)
        ax.axvline(0.25, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(0.90, color="red", linestyle="--", linewidth=0.8)
        ax.set_xlabel("l / d")
        ax.set_ylabel("cumulative explained variance (centered)")
        ax.set_title("Per-expert spectrum (top-1 routed) — median ± envelope across experts")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "svd_curve_per_expert.png", dpi=140)
        plt.close(fig)
        print(f"[stage0_v2] wrote {out_dir/'svd_curve_per_expert.png'}", flush=True)
    except Exception as e:
        print(f"[stage0_v2] plot failed: {e}", flush=True)

    # Human-readable report.
    lines = []
    lines.append("# Stage 0 v2: dispatch activation low-rank check (Qwen3-30B-A3B)")
    lines.append("")
    lines.append(f"- Model: `{args.model_path}`")
    lines.append(
        f"- hidden_size d = {hidden_size}, num_hidden_layers = {n_layers}, "
        f"num_experts = {n_experts}, top_k = {cfg.num_experts_per_tok}"
    )
    lines.append(
        f"- Calibration: {summary['n_chunks']} chunks x {summary['chunk_len']} tokens = "
        f"{summary['n_kept_tokens']} tokens (no padding); N/d = "
        f"{summary['n_kept_tokens']/hidden_size:.1f}"
    )
    lines.append(f"- LongBench sources: {args.longbench_files}")
    lines.append(f"- Probed layers: {layers_to_hook}")
    lines.append("")
    lines.append("## Global explained variance (centered) at fractions of d")
    lines.append("| layer | d/16 | d/8 | d/4 | d/2 | rank@0.90 | rank@0.95 | rank@0.99 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for L, info in summary["layers"].items():
        ev = info["explained_variance_centered_at"]
        r = info["rank_at_explained_variance"]
        lines.append(
            f"| {L} | {ev['d/16']:.3f} | {ev['d/8']:.3f} | {ev['d/4']:.3f} | {ev['d/2']:.3f} | "
            f"{r['0.90']} | {r['0.95']} | {r['0.99']} |"
        )
    lines.append("")
    lines.append("## Per-expert explained variance at d/4 (top-1 destination)")
    lines.append("| layer | n_experts | min | p25 | median | p75 | max |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for L, info in summary["layers"].items():
        pe = info.get("per_expert", {})
        if not pe:
            lines.append(f"| {L} | 0 | - | - | - | - | - |")
            continue
        s = pe["explained_variance_at_d_over_4"]
        lines.append(
            f"| {L} | {pe['n_experts_with_at_least_64_tokens']} | "
            f"{s['min']:.3f} | {s['p25']:.3f} | {s['median']:.3f} | {s['p75']:.3f} | {s['max']:.3f} |"
        )
    lines.append("")
    lines.append("## Go/No-Go (rule: ≥0.90 explained variance at ℓ = d/4 in ALL probed layers)")
    lines.append("")
    lines.append(f"**Verdict: {summary['go_criteria']['verdict']}**")
    lines.append("")
    for L, v in summary["go_criteria"]["per_layer_explained_variance_at_d_over_4"].items():
        flag = "OK" if v >= 0.90 else "FAIL"
        lines.append(f"- layer {L}: explained variance at d/4 = {v:.3f}  [{flag}]")
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"[stage0_v2] wrote {out_dir/'report.md'}", flush=True)
    print(f"[stage0_v2] verdict: {summary['go_criteria']['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
