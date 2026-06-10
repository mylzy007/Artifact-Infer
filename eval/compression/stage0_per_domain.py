"""Stage 0 per-domain SVD: does a single-domain calibration give a steeper spectrum?

Loads Qwen3-30B-A3B once, then for each of several narrow domains feeds 16
chunks * 512 tokens = 8192 tokens (N/d = 4, well-conditioned for SVD on d=2048).
For each layer and each domain, computes the centered cumulative explained
variance curve.

Outputs (under eval_results/compression_lowrank_stage0_per_domain/):
  - svd_per_domain.json
  - svd_per_domain.png    : grid of {layer x domain} cumulative curves
  - report.md             : ranked table of best-domain low-rank

Domains:
  lcc           - LongBench code completion (long, single language flavor)
  repobench_p   - LongBench python-only code completion
  gov_report    - LongBench long English government reports
  multifieldqa  - LongBench mixed-domain QA passages (already in mixed run; keep as control)
  dureader      - LongBench Chinese long QA
  gsm8k         - GSM8K math: Q + chain-of-thought A concatenated
  passage_ret   - LongBench passage_retrieval_en (highly templated)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

MODEL_PATH = "/home/lzy/models/Qwen3-30B-A3B"
OUT_DIR = Path("eval_results/compression_lowrank_stage0_per_domain")
LB_DIR = Path("/home/lzy/datasets/moe_benchmarks/longbench/extracted/data")
GSM_PARQUET = "/home/lzy/datasets/moe_benchmarks/gsm8k/main/train-00000-of-00001.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--layers", type=int, nargs="+", default=[2, 12, 24, 36, 46])
    p.add_argument("--chunk-len", type=int, default=512)
    p.add_argument("--chunks-per-domain", type=int, default=16, help="16 * 512 = 8192 tokens / domain (N/d=4)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gpu-mem-gib", type=int, default=22)
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def setup_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_lb_context(fname: str, max_chars: int = 600_000) -> str:
    pieces, total = [], 0
    path = LB_DIR / fname
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            txt = rec.get("context") or rec.get("input") or ""
            if isinstance(txt, str) and len(txt) >= 200:
                pieces.append(txt)
                total += len(txt)
                if total >= max_chars:
                    break
    return "\n\n".join(pieces)


def load_gsm_text(max_examples: int = 400) -> str:
    import pandas as pd
    df = pd.read_parquet(GSM_PARQUET)
    rows = df.iloc[:max_examples]
    pieces = []
    for _, r in rows.iterrows():
        pieces.append(f"Question: {r['question']}\nAnswer: {r['answer']}")
    return "\n\n".join(pieces)


def get_domain_text(name: str) -> str:
    if name == "lcc":
        return load_lb_context("lcc.jsonl")
    if name == "repobench_p":
        return load_lb_context("repobench-p.jsonl")
    if name == "gov_report":
        return load_lb_context("gov_report.jsonl")
    if name == "multifieldqa":
        return load_lb_context("multifieldqa_en.jsonl")
    if name == "dureader":
        return load_lb_context("dureader.jsonl")
    if name == "passage_ret":
        return load_lb_context("passage_retrieval_en.jsonl")
    if name == "gsm8k":
        return load_gsm_text()
    raise ValueError(name)


DOMAINS = [
    ("lcc", "code (mixed languages)"),
    ("repobench_p", "code (python)"),
    ("gov_report", "long gov reports"),
    ("multifieldqa", "mixed-field QA"),
    ("dureader", "Chinese QA"),
    ("passage_ret", "passage retrieval (templated)"),
    ("gsm8k", "math chain-of-thought"),
]


def chunk_token_ids(ids: list[int], chunk_len: int, n_chunks: int) -> list[list[int]]:
    out, pos = [], 0
    while pos + chunk_len <= len(ids) and len(out) < n_chunks:
        out.append(ids[pos : pos + chunk_len])
        pos += chunk_len
    return out


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = setup_dtype(args.dtype)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_mem = {i: f"{args.gpu_mem_gib}GiB" for i in range(torch.cuda.device_count())}
    print(f"[per_domain] loading model from {args.model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_mem,
        trust_remote_code=True,
    )
    model.eval()
    cfg = model.config
    H = cfg.hidden_size
    nL = cfg.num_hidden_layers
    print(f"[per_domain] hidden_size={H}, num_layers={nL}, num_experts={cfg.num_experts}", flush=True)

    # Build domain chunks.
    domain_chunks: dict[str, list[list[int]]] = {}
    for name, _ in DOMAINS:
        text = get_domain_text(name)
        ids = tokenizer(text, return_tensors=None)["input_ids"]
        if not isinstance(ids, list):
            ids = list(ids)
        ch = chunk_token_ids(ids, args.chunk_len, args.chunks_per_domain)
        if len(ch) < args.chunks_per_domain:
            print(f"[per_domain]   {name}: only {len(ch)} chunks available (wanted {args.chunks_per_domain})", flush=True)
        domain_chunks[name] = ch
        print(f"[per_domain] {name}: {len(text):>8d} chars -> {len(ids):>7d} tokens -> {len(ch)} chunks", flush=True)

    # Concatenate in fixed order so we can slice activations by domain afterwards.
    order = [(name, ch) for (name, _) in DOMAINS for ch in domain_chunks[name]]
    flat_chunks = [c for _, c in order]
    # Map (domain -> list of chunk-indices in flat_chunks).
    domain_indices: dict[str, list[int]] = {n: [] for n, _ in DOMAINS}
    for i, (n, _) in enumerate(order):
        domain_indices[n].append(i)
    n_total_chunks = len(flat_chunks)
    print(f"[per_domain] total chunks: {n_total_chunks} ({n_total_chunks*args.chunk_len} tokens)", flush=True)

    # Activation capture per layer.
    activations: dict[int, list[torch.Tensor]] = {L: [] for L in args.layers}

    def make_hook(L: int):
        def hook(module, inputs):
            x = inputs[0].detach()
            # [B, T, H] -> save flattened [B*T, H] keeping batch order intact.
            activations[L].append(x.to("cpu", dtype=torch.float32).reshape(-1, x.shape[-1]))
            return None
        return hook

    handles = []
    for L in args.layers:
        h = model.model.layers[L].mlp.register_forward_pre_hook(make_hook(L))
        handles.append(h)

    input_ids = torch.tensor(flat_chunks, dtype=torch.long)
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
            print(f"[per_domain] batch {i // bsz + 1}/{(input_ids.shape[0] + bsz - 1) // bsz}", flush=True)

    for h in handles:
        h.remove()

    # Sanity: each layer's concatenated tensor should be (n_total_chunks * chunk_len, H).
    summary: dict = {
        "model_path": args.model_path,
        "hidden_size": H,
        "chunk_len": int(args.chunk_len),
        "chunks_per_domain": int(args.chunks_per_domain),
        "tokens_per_domain": int(args.chunk_len * args.chunks_per_domain),
        "layers": args.layers,
        "domains": [n for n, _ in DOMAINS],
        "results": {},  # layer -> domain -> {ev_at, rank_at}
    }

    # Per-layer, per-domain SVD.
    print("[per_domain] running SVDs...", flush=True)
    curves: dict[int, dict[str, np.ndarray]] = {L: {} for L in args.layers}
    for L in args.layers:
        X_all = torch.cat(activations[L], dim=0)  # [n_total_chunks * chunk_len, H]
        N_expected = n_total_chunks * args.chunk_len
        assert X_all.shape[0] == N_expected, (X_all.shape, N_expected)
        # Rearrange to [chunks, chunk_len, H]
        X_all = X_all.view(n_total_chunks, args.chunk_len, H)

        summary["results"][str(L)] = {}
        for name, _ in DOMAINS:
            idx = domain_indices[name]
            if not idx:
                continue
            Xd = X_all[idx].reshape(-1, H)  # [len(idx)*chunk_len, H]
            Xc = Xd - Xd.mean(dim=0, keepdim=True)
            sv = torch.linalg.svdvals(Xc).cpu().numpy()
            ev = sv ** 2
            if ev.sum() <= 0:
                continue
            cum = np.cumsum(ev) / ev.sum()
            curves[L][name] = cum
            marks = {"d/16": H // 16, "d/8": H // 8, "d/4": H // 4, "d/2": H // 2}
            summary["results"][str(L)][name] = {
                "n_tokens": int(Xd.shape[0]),
                "ev_at": {k: float(cum[min(v, len(cum)) - 1]) for k, v in marks.items()},
                "rank_at_0.90": int(np.searchsorted(cum, 0.90) + 1),
                "rank_at_0.95": int(np.searchsorted(cum, 0.95) + 1),
                "rank_at_0.99": int(np.searchsorted(cum, 0.99) + 1),
                "rank_at_0.999": int(np.searchsorted(cum, 0.999) + 1),
            }
            print(
                f"  layer {L} / {name:<14s} N={Xd.shape[0]:>5d}  "
                f"EV@d/8={cum[H//8 - 1]:.3f}  EV@d/4={cum[H//4 - 1]:.3f}  EV@d/2={cum[H//2 - 1]:.3f}",
                flush=True,
            )

    (out_dir / "svd_per_domain.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir/'svd_per_domain.json'}", flush=True)

    # Plot: grid of (rows=layers, cols=domains).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_rows, n_cols = len(args.layers), len(DOMAINS)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.0 * n_rows), sharex=True, sharey=True)
        if n_rows == 1:
            axes = np.array([axes])
        if n_cols == 1:
            axes = axes[:, None]
        for r, L in enumerate(args.layers):
            for c, (name, label) in enumerate(DOMAINS):
                ax = axes[r, c]
                cum = curves[L].get(name)
                if cum is None:
                    ax.set_visible(False)
                    continue
                xs = np.arange(1, len(cum) + 1) / H
                ax.plot(xs, cum, color="C0", linewidth=1.4)
                ax.axhline(0.90, color="red", linestyle=":", linewidth=0.6)
                ax.axvline(0.25, color="gray", linestyle="--", linewidth=0.6)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1.02)
                if r == 0:
                    ax.set_title(f"{label}", fontsize=9)
                if c == 0:
                    ax.set_ylabel(f"L{L}\nEV", fontsize=9)
                if r == n_rows - 1:
                    ax.set_xlabel("l/d", fontsize=8)
                v = summary["results"][str(L)][name]["ev_at"]["d/4"]
                color = "green" if v >= 0.90 else ("orange" if v >= 0.85 else "black")
                ax.text(0.05, 0.05, f"@d/4={v:.2f}", fontsize=8, color=color, transform=ax.transAxes)
                ax.grid(True, alpha=0.25)
        fig.suptitle("Per-domain dispatch activation spectrum — Qwen3-30B-A3B (d=2048, N=8192/domain)", fontsize=11, y=1.00)
        fig.tight_layout()
        fig.savefig(out_dir / "svd_per_domain.png", dpi=140)
        plt.close(fig)
        print(f"wrote {out_dir/'svd_per_domain.png'}", flush=True)

        # Compact table-style plot: heatmap of EV@d/4 across layers x domains.
        fig, ax = plt.subplots(figsize=(7, 3.5))
        mat = np.zeros((len(args.layers), len(DOMAINS)))
        for r, L in enumerate(args.layers):
            for c, (name, _) in enumerate(DOMAINS):
                mat[r, c] = summary["results"][str(L)][name]["ev_at"]["d/4"]
        im = ax.imshow(mat, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(DOMAINS)))
        ax.set_xticklabels([n for n, _ in DOMAINS], rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(args.layers)))
        ax.set_yticklabels([f"L{L}" for L in args.layers])
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if mat[r, c] > 0.7 else "white")
        ax.set_title("EV at l = d/4 (=512) — green ≥ 0.90 (Go threshold)")
        fig.colorbar(im, ax=ax, label="explained variance")
        fig.tight_layout()
        fig.savefig(out_dir / "svd_per_domain_heatmap.png", dpi=140)
        plt.close(fig)
        print(f"wrote {out_dir/'svd_per_domain_heatmap.png'}", flush=True)
    except Exception as e:
        print(f"plot failed: {e}", flush=True)

    # Markdown report.
    lines = []
    lines.append("# Stage 0 per-domain SVD (Qwen3-30B-A3B)")
    lines.append("")
    lines.append(f"- d = {H}, {args.chunks_per_domain} chunks/domain x {args.chunk_len} tokens = {args.chunks_per_domain*args.chunk_len} tokens/domain (N/d = {args.chunks_per_domain*args.chunk_len/H:.1f})")
    lines.append(f"- Probed layers: {args.layers}")
    lines.append("")
    lines.append("## EV @ d/4 across (layer, domain)")
    lines.append("| layer | " + " | ".join(n for n, _ in DOMAINS) + " |")
    lines.append("|---:|" + "|".join("---:" for _ in DOMAINS) + "|")
    for L in args.layers:
        cells = []
        for name, _ in DOMAINS:
            r = summary["results"][str(L)][name]
            v = r["ev_at"]["d/4"]
            mark = "**" if v >= 0.90 else ""
            cells.append(f"{mark}{v:.3f}{mark}")
        lines.append(f"| {L} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## EV @ d/2 across (layer, domain)")
    lines.append("| layer | " + " | ".join(n for n, _ in DOMAINS) + " |")
    lines.append("|---:|" + "|".join("---:" for _ in DOMAINS) + "|")
    for L in args.layers:
        cells = []
        for name, _ in DOMAINS:
            r = summary["results"][str(L)][name]
            v = r["ev_at"]["d/2"]
            mark = "**" if v >= 0.90 else ""
            cells.append(f"{mark}{v:.3f}{mark}")
        lines.append(f"| {L} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## rank needed for 0.99 variance")
    lines.append("| layer | " + " | ".join(n for n, _ in DOMAINS) + " |")
    lines.append("|---:|" + "|".join("---:" for _ in DOMAINS) + "|")
    for L in args.layers:
        cells = []
        for name, _ in DOMAINS:
            r = summary["results"][str(L)][name]["rank_at_0.99"]
            cells.append(str(r))
        lines.append(f"| {L} | " + " | ".join(cells) + " |")
    lines.append("")
    # Best domain per layer.
    lines.append("## Best domain per layer (highest EV @ d/4)")
    lines.append("| layer | best domain | EV @ d/4 |")
    lines.append("|---:|---|---:|")
    for L in args.layers:
        best_name, best_v = None, -1.0
        for name, _ in DOMAINS:
            v = summary["results"][str(L)][name]["ev_at"]["d/4"]
            if v > best_v:
                best_v, best_name = v, name
        lines.append(f"| {L} | {best_name} | {best_v:.3f} |")
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"wrote {out_dir/'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
