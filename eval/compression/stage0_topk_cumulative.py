"""Stage-0 E2: top-k cumulative weight protection.

Tests "important experts get full bf16, others get compressed" — generalizing
师兄's "top-1 全量" suggestion. For each token's top-k=8 routed experts, sort
by routing weight, sum cumulatively. Experts whose CUMULATIVE weight (up to
but not including them) is BELOW threshold T are kept FULL (no compression);
the rest are compressed at 4x (keep_frac=0.25 + FP8 + L2).

Threshold T spans the spectrum:
  T = 0.0          : no expert protected (= baseline: compress all 8 at 4x)
  T = top1_only    : protect only the rank-1 expert (师兄 original)
  T = 0.5          : protect cumulative weight up to 0.5
  T = 0.7          : ...up to 0.7
  T = 0.9          : ...up to 0.9
  T = 1.0          : protect all (= teacher)

The mask is computed per-(token, expert) pair, so the same monkey-patch
machinery as E1 works.

Also computes the EFFECTIVE average compression ratio per config (weighted by
how many (token, expert) rows end up compressed vs full).

Metric: next-token top-1 / top-5 accuracy on the same 4096-token lcc held-out
split used by Phase A / E1.

Outputs (under eval_results/compression_lowrank_stage0_topk_cumulative/):
  - summary.json
  - report.md
  - tradeoff_curve.png  (effective compression vs top-1 accuracy)

Run:
  CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    /home/lzy/miniconda3/envs/atom/bin/python eval/compression/stage0_topk_cumulative.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL_PATH = "/home/lzy/models/Qwen3-30B-A3B"
OUT_DIR = Path("eval_results/compression_lowrank_stage0_topk_cumulative")
LB_DIR = Path("/home/lzy/datasets/moe_benchmarks/longbench/extracted/data")

FP8_E4M3_MAX = 448.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--chunk-len", type=int, default=512)
    p.add_argument("--test-chunks", type=int, default=8)
    p.add_argument("--batch-chunks", type=int, default=4)
    p.add_argument("--keep-frac-when-compressed", type=float, default=0.25)
    # Special tokens: "top1_only" means protect rank-1 expert per token (regardless of weight)
    p.add_argument("--thresholds", nargs="+", default=["0.0", "top1_only", "0.5", "0.7", "0.9", "1.0"])
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


@dataclass
class CompressionState:
    threshold: str = "0.0"        # "0.0" / "top1_only" / "0.5" / ...
    keep_frac: float = 0.25
    use_fp8: bool = True
    use_l2: bool = True
    # Stats accumulators (reset per config).
    n_pairs_total: int = 0
    n_pairs_compressed: int = 0
    sum_weight_full: float = 0.0
    sum_weight_compressed: float = 0.0

_state = CompressionState()


def sparsify_topk_l2_fp8(x: torch.Tensor, k: int) -> torch.Tensor:
    abs_x = x.abs()
    _, idx = abs_x.topk(k, dim=-1)
    sparse = torch.zeros_like(x)
    sparse.scatter_(-1, idx, x.gather(-1, idx))
    full_n2 = (x.float() ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    sparse_n2 = (sparse.float() ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    sparse = sparse * (full_n2 / sparse_n2).to(x.dtype)
    xf = sparse.float()
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / FP8_E4M3_MAX
    x_scaled = xf / scale
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    return (x_fp8.to(torch.float32) * scale).to(x.dtype)


def compute_keep_full_mask(routing_weights_topk: torch.Tensor, threshold: str) -> torch.Tensor:
    """Return bool mask [B*T, top_k] — True means this (token, expert) is uncompressed.

    routing_weights_topk: [B*T, top_k] floats summing to 1 per row (after norm_topk_prob),
    sorted DESCENDING by weight.
    """
    n, k = routing_weights_topk.shape
    if threshold == "0.0":
        return torch.zeros((n, k), dtype=torch.bool, device=routing_weights_topk.device)
    if threshold == "1.0":
        return torch.ones((n, k), dtype=torch.bool, device=routing_weights_topk.device)
    if threshold == "top1_only":
        m = torch.zeros((n, k), dtype=torch.bool, device=routing_weights_topk.device)
        m[:, 0] = True
        return m
    # Numeric threshold T in (0, 1): include experts whose cumulative weight up to (but
    # not including) them is < T. This guarantees at least one expert per row is kept
    # full (the top-1), and adds more until cumulative >= T.
    T = float(threshold)
    cum = routing_weights_topk.cumsum(dim=-1)              # [n, k] (inclusive of self)
    shifted_cum = torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], dim=-1)  # exclusive of self
    return shifted_cum < T


def make_patched_forward(orig_forward):
    def patched_forward(self, hidden_states: torch.Tensor):
        B, T, H = hidden_states.shape
        hidden_states = hidden_states.view(-1, H)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        routing_weights_topk_bf16 = routing_weights_topk.to(hidden_states.dtype)

        keep_full_mask = compute_keep_full_mask(routing_weights_topk.float(), _state.threshold)  # [B*T, top_k]

        # Bookkeeping for effective compression ratio reporting.
        n_pairs = int(keep_full_mask.numel())
        n_compressed = int((~keep_full_mask).sum().item())
        _state.n_pairs_total += n_pairs
        _state.n_pairs_compressed += n_compressed
        # weight-bucket stats
        w_full = float((routing_weights_topk * keep_full_mask.float()).sum().item())
        w_comp = float((routing_weights_topk * (~keep_full_mask).float()).sum().item())
        _state.sum_weight_full += w_full
        _state.sum_weight_compressed += w_comp

        final_hidden_states = torch.zeros(B * T, H, dtype=hidden_states.dtype, device=hidden_states.device)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        if expert_hitted.dim() > 1:
            expert_hitted = expert_hitted.flatten()

        k_compress = max(1, int(round(H * _state.keep_frac)))
        for expert_idx in expert_hitted.tolist():
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, H)
            expert_output = expert_layer(current_state)

            row_keep_full = keep_full_mask[top_x, idx]
            if row_keep_full.all():
                final_part = expert_output
            elif (~row_keep_full).all():
                final_part = sparsify_topk_l2_fp8(expert_output, k_compress)
            else:
                final_part = expert_output.clone()
                to_compress_rows = expert_output[~row_keep_full]
                final_part[~row_keep_full] = sparsify_topk_l2_fp8(to_compress_rows, k_compress)

            weighted = final_part * routing_weights_topk_bf16[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, weighted.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(B, T, H)
        return final_hidden_states, router_logits

    return patched_forward


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[topk_cum] loading {args.model_path}", flush=True)
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
    embed_dev = next(model.model.embed_tokens.parameters()).device
    print(f"[topk_cum] H={H}, num_experts={cfg.num_experts}, top_k={cfg.num_experts_per_tok}", flush=True)

    # Patch instance-level _old_forward (accelerate wraps the public forward).
    orig_forwards: list = []
    for layer in model.model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "_old_forward"):
            orig = mlp._old_forward
            orig_forwards.append(("_old_forward", mlp, orig))
            mlp._old_forward = make_patched_forward(orig).__get__(mlp, type(mlp))
        else:
            orig = mlp.forward
            orig_forwards.append(("forward", mlp, orig))
            base = orig.__func__ if hasattr(orig, "__func__") else orig
            mlp.forward = make_patched_forward(base).__get__(mlp, type(mlp))
    print(f"[topk_cum] patched {len(orig_forwards)} MoE blocks", flush=True)

    text = load_lcc_text(min_chars=1_400_000)
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    if not isinstance(ids, list):
        ids = list(ids)
    skip = 16 * args.chunk_len
    test_chunks = chunk_ids(ids, args.chunk_len, args.test_chunks, skip=skip)
    n_test_tok = args.test_chunks * args.chunk_len
    print(f"[topk_cum] test_tokens={n_test_tok}", flush=True)

    def run_one(threshold: str) -> dict:
        _state.threshold = threshold
        _state.keep_frac = float(args.keep_frac_when_compressed)
        _state.use_fp8 = True
        _state.use_l2 = True
        _state.n_pairs_total = 0
        _state.n_pairs_compressed = 0
        _state.sum_weight_full = 0.0
        _state.sum_weight_compressed = 0.0

        losses, top1c, top5c, total = [], 0, 0, 0
        top1_cache: list[torch.Tensor] = []
        with torch.no_grad():
            ids_t = torch.tensor(test_chunks, dtype=torch.long, device=embed_dev)
            for i in range(0, ids_t.shape[0], args.batch_chunks):
                ids_b = ids_t[i : i + args.batch_chunks]
                out = model(input_ids=ids_b, use_cache=False)
                logits = out.logits
                shift_logits = logits[..., :-1, :].float()
                shift_labels = ids_b[..., 1:]
                nll = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    reduction="mean",
                )
                losses.append(float(nll.item()))
                top1 = shift_logits.argmax(dim=-1)
                _, top5 = shift_logits.topk(5, dim=-1)
                top1c += int((top1 == shift_labels).sum().item())
                top5c += int((top5 == shift_labels.unsqueeze(-1)).any(dim=-1).sum().item())
                total += int(shift_labels.numel())
                top1_cache.append(top1.cpu())

        avg_loss = float(np.mean(losses))
        frac_compressed_pairs = _state.n_pairs_compressed / max(_state.n_pairs_total, 1)
        # Effective wire bytes = full bytes for kept pairs + compressed bytes for the rest.
        # Per row: full = 2H bytes (bf16). Compressed value-only = 2H * keep_frac bytes
        # (matches earlier sweep). With FP8 + indices + scale (production wire): 3k+2 bytes,
        # i.e., roughly H/(2 * (3 * keep_frac + 2/H)) compression. Here we report just
        # the value-only equivalent compression ratio for an apples-to-apples comparison
        # with the headline 4x / 8x numbers in prior sweeps.
        avg_bytes_per_row = 2 * H * (1 - frac_compressed_pairs + frac_compressed_pairs * _state.keep_frac)
        avg_compression_value_only = (2 * H) / max(avg_bytes_per_row, 1e-12)

        return {
            "threshold": threshold,
            "loss": avg_loss,
            "ppl": math.exp(avg_loss),
            "top1": top1c / max(total, 1),
            "top5": top5c / max(total, 1),
            "frac_compressed_pairs": frac_compressed_pairs,
            "frac_full_pairs": 1 - frac_compressed_pairs,
            "sum_weight_full": _state.sum_weight_full,
            "sum_weight_compressed": _state.sum_weight_compressed,
            "avg_compression_value_only": avg_compression_value_only,
            "top1_argmax": torch.cat([c.reshape(-1) for c in top1_cache], dim=0),
            "total": total,
        }

    rows = []
    teacher = None
    for thr in args.thresholds:
        print(f"[topk_cum] running threshold={thr} ...", flush=True)
        r = run_one(thr)
        if thr == "1.0":
            teacher = r
        agree = None
        if teacher is not None and thr != "1.0":
            agree = float((r["top1_argmax"] == teacher["top1_argmax"]).float().mean().item())
        row = {k: v for k, v in r.items() if k != "top1_argmax"}
        if agree is not None:
            row["agreement_with_teacher_top1"] = agree
        rows.append(row)
        print(
            f"[topk_cum]   threshold={thr}: ppl={r['ppl']:.3f} "
            f"top1={r['top1']*100:.2f}% "
            f"compressed_pair_frac={r['frac_compressed_pairs']:.3f} "
            f"avg_compress~{r['avg_compression_value_only']:.2f}x",
            flush=True,
        )

    # Recompute drops vs teacher
    teacher_top1 = next(r for r in rows if r["threshold"] == "1.0")["top1"]
    teacher_ppl = next(r for r in rows if r["threshold"] == "1.0")["ppl"]
    for r in rows:
        r["top1_drop_pp"] = (teacher_top1 - r["top1"]) * 100
        r["ppl_increase_pct"] = (r["ppl"] - teacher_ppl) / teacher_ppl * 100

    # Restore.
    for attr, mlp, orig in orig_forwards:
        setattr(mlp, attr, orig)

    summary = {
        "model_path": args.model_path,
        "hidden_size": H,
        "num_experts": cfg.num_experts,
        "top_k": cfg.num_experts_per_tok,
        "test_tokens": n_test_tok,
        "domain": "lcc",
        "keep_frac_when_compressed": args.keep_frac_when_compressed,
        "thresholds": args.thresholds,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir/'summary.json'}", flush=True)

    # Report.
    lines = []
    lines.append("# Top-k cumulative weight protection on lcc (E2)")
    lines.append("")
    lines.append(f"Test = {n_test_tok} held-out lcc tokens. Teacher PPL = {teacher_ppl:.3f}, top-1 = {teacher_top1*100:.2f}%.")
    lines.append(f"Per-row compression (when applied) = topk_l2 + FP8 with keep_frac = {args.keep_frac_when_compressed} (4x value).")
    lines.append("")
    lines.append("**Threshold semantics**: for each token's top-8 experts sorted descending by routing weight, an expert is kept FULL iff the cumulative weight BEFORE it (exclusive) < threshold. `top1_only` is a special case that protects only the rank-1 expert per token regardless of weight.")
    lines.append("")
    lines.append("| threshold | top-1 acc | top-1 drop | PPL | PPL +% | compressed pair frac | avg compr (value-only) | weight % compressed |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        wf = r["sum_weight_full"]
        wc = r["sum_weight_compressed"]
        w_frac = wc / max(wf + wc, 1e-12)
        lines.append(
            f"| {r['threshold']} | {r['top1']*100:.2f}% | {r['top1_drop_pp']:+.2f}pp | "
            f"{r['ppl']:.3f} | {r['ppl_increase_pct']:+.1f}% | "
            f"{r['frac_compressed_pairs']*100:.1f}% | "
            f"{r['avg_compression_value_only']:.2f}x | "
            f"{w_frac*100:.1f}% |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("- `threshold=0.0` is the original baseline: every (token, expert) compressed at 4x → avg ~4x.")
    lines.append("- `threshold=top1_only` keeps just rank-1 expert per token full; with k=8 → 1/8 = 12.5% pairs full.")
    lines.append("- `threshold=0.5/0.7/0.9` keeps a growing prefix; weight % compressed shows what fraction of total routing mass got compressed.")
    lines.append("- `threshold=1.0` keeps everything full → teacher.")
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"wrote {out_dir/'report.md'}", flush=True)

    # Tradeoff curve: x = avg_compression_value_only, y = top-1 acc
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["avg_compression_value_only"] for r in rows]
        ys = [r["top1"] * 100 for r in rows]
        labels = [r["threshold"] for r in rows]
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.plot(xs, ys, marker="o", linestyle="-")
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(f"T={lbl}", (x, y), fontsize=8, textcoords="offset points", xytext=(5, 5))
        ax.axhline(teacher_top1 * 100, color="red", linestyle="--", linewidth=0.8, label=f"teacher = {teacher_top1*100:.2f}%")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("average value-only compression ratio")
        ax.set_ylabel("next-token top-1 accuracy (%)")
        ax.set_title("E2: top-k cumulative weight protection — accuracy vs compression")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "tradeoff_curve.png", dpi=140)
        plt.close(fig)
        print(f"wrote {out_dir/'tradeoff_curve.png'}", flush=True)
    except Exception as e:
        print(f"plot failed: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
