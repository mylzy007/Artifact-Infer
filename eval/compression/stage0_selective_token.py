"""Stage-0 E1: per-token selective compression.

Tests whether WHICH tokens are compressed matters (vs random). For each
strategy + compress_frac, we monkey-patch every Qwen3MoeSparseMoeBlock.forward
to skip compression on the "important" tokens and apply topk_l2+FP8 (4x value
compression, keep_frac=0.25) only to the "unimportant" ones.

Selection strategies (per-layer signals; one decision per (token, layer)):
  random       — random fraction selected for compression (baseline)
  router_conf  — high router top-1 probability tokens are compressed (high
                 confidence = decided = redundant)
  hidden_norm  — small hidden-state-L2-norm tokens are compressed (small
                 norm = small residual contribution)

For each strategy, sweep compress_frac in {0.0, 0.25, 0.5, 0.75, 1.0}.
(0.0 = teacher; 1.0 = current "compress everything at 4x" baseline.)

Metric: next-token top-1 / top-5 accuracy on the same 4096-token lcc held-out
split used by Phase A (D1) and Phase B/C sweeps.

Outputs (under eval_results/compression_lowrank_stage0_selective_token/):
  - summary.json
  - report.md
  - heatmap.png  (strategy x compress_frac heatmap of top-1 acc)

Run:
  CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    /home/lzy/miniconda3/envs/atom/bin/python eval/compression/stage0_selective_token.py
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
OUT_DIR = Path("eval_results/compression_lowrank_stage0_selective_token")
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
    p.add_argument("--compress-fracs", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--strategies", nargs="+", default=["random", "router_conf", "hidden_norm"])
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


# ---------------- Global config that the patched forward reads ----------------

@dataclass
class CompressionState:
    strategy: str = "none"          # none / random / router_conf / hidden_norm
    compress_frac: float = 0.0      # 0.0 = no token compressed; 1.0 = all compressed
    keep_frac: float = 0.25         # per-row compression rate when applied
    use_fp8: bool = True
    use_l2: bool = True
    seed: int = 0                   # for deterministic random strategy

_state = CompressionState()


def sparsify_topk_l2_fp8(x: torch.Tensor, k: int) -> torch.Tensor:
    """Matches the production CombineEPHT sparsifier and Phase A/B/C HF hooks."""
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


def select_compressed_tokens(
    hidden_states_flat: torch.Tensor,    # [B*T, H]
    routing_weights_topk: torch.Tensor,  # [B*T, top_k]
    strategy: str,
    compress_frac: float,
    seed: int,
    layer_id_for_seed: int = 0,
) -> torch.Tensor:
    """Return bool tensor [B*T] where True = this token should be compressed."""
    n = hidden_states_flat.shape[0]
    if compress_frac <= 0.0 or n == 0:
        return torch.zeros(n, dtype=torch.bool, device=hidden_states_flat.device)
    if compress_frac >= 1.0:
        return torch.ones(n, dtype=torch.bool, device=hidden_states_flat.device)
    if strategy == "random":
        g = torch.Generator(device=hidden_states_flat.device)
        g.manual_seed(seed + layer_id_for_seed * 9973)
        scores = torch.rand(n, generator=g, device=hidden_states_flat.device)
        # Compress the LOWEST `compress_frac` fraction
        thresh = torch.quantile(scores, compress_frac).item()
        return scores < thresh
    if strategy == "router_conf":
        # High top-1 probability = "decided" = compress
        top1 = routing_weights_topk[:, 0].float()
        thresh = torch.quantile(top1, 1 - compress_frac).item()
        return top1 >= thresh
    if strategy == "hidden_norm":
        # Small L2 norm = small contribution = compress
        norms = hidden_states_flat.float().norm(dim=-1)
        thresh = torch.quantile(norms, compress_frac).item()
        return norms < thresh
    raise ValueError(f"unknown strategy {strategy}")


# Layer counter — bumped each block forward so per-layer seeds differ in random mode.
_layer_call_counter = {"i": 0}


def make_patched_forward(orig_forward):
    """Build a patched forward that applies selective compression based on _state.

    NOTE: When the model is loaded with accelerate's device_map='auto', the real
    forward is cached as `module._old_forward` and the public `module.forward`
    becomes a partial wrapper. So we must patch the INSTANCE's `_old_forward`
    rather than the class. See `install_patches`.
    """
    def patched_forward(self, hidden_states: torch.Tensor):
        B, T, H = hidden_states.shape
        hidden_states = hidden_states.view(-1, H)                # [B*T, H]
        router_logits = self.gate(hidden_states)                  # [B*T, E]
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        routing_weights_topk = routing_weights_topk.to(hidden_states.dtype)

        # Per-token compress decision.
        layer_seed = _layer_call_counter["i"]
        _layer_call_counter["i"] += 1
        compress_token = select_compressed_tokens(
            hidden_states, routing_weights_topk.float(),
            _state.strategy, _state.compress_frac, _state.seed, layer_seed,
        )                                                          # [B*T]
        # Expand to per-(token, expert) pair.
        # keep_full_mask[t, k]: True means "do not compress this row".
        keep_full_mask = (~compress_token).unsqueeze(-1).expand(-1, self.top_k)  # [B*T, top_k]

        final_hidden_states = torch.zeros(B * T, H, dtype=hidden_states.dtype, device=hidden_states.device)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        # mypy noise
        if expert_hitted.dim() > 1:
            expert_hitted = expert_hitted.flatten()

        k_compress = max(1, int(round(H * _state.keep_frac)))
        for expert_idx in expert_hitted.tolist():
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])         # idx ∈[0,K), top_x ∈[0,B*T)
            current_state = hidden_states[None, top_x].reshape(-1, H)  # [n_for_e, H]
            expert_output = expert_layer(current_state)                # [n_for_e, H]

            # Compress only rows where keep_full_mask[top_x, idx] is False.
            row_keep_full = keep_full_mask[top_x, idx]                 # [n_for_e]
            if row_keep_full.all():
                final_part = expert_output
            elif (~row_keep_full).all():
                final_part = sparsify_topk_l2_fp8(expert_output, k_compress)
            else:
                final_part = expert_output.clone()
                to_compress_rows = expert_output[~row_keep_full]
                final_part[~row_keep_full] = sparsify_topk_l2_fp8(to_compress_rows, k_compress)

            weighted = final_part * routing_weights_topk[top_x, idx, None]
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
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    print(f"[selective] loading {args.model_path}", flush=True)
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
    print(f"[selective] H={H}, num_experts={cfg.num_experts}, top_k={cfg.num_experts_per_tok}", flush=True)

    # Monkey-patch every MoE block's REAL forward. Under accelerate's
    # device_map=auto, that's stored as `module._old_forward` (the public
    # `module.forward` is a partial wrapper that adds device-dispatch hooks).
    # If accelerate didn't hook this module, fall back to the public
    # `forward` attribute.
    orig_forwards: list = []
    for layer in model.model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "_old_forward"):
            orig_method = mlp._old_forward
            orig_forwards.append(("_old_forward", mlp, orig_method))
            mlp._old_forward = make_patched_forward(orig_method).__get__(mlp, type(mlp))
        else:
            orig_method = mlp.forward
            orig_forwards.append(("forward", mlp, orig_method))
            mlp.forward = make_patched_forward(orig_method.__func__ if hasattr(orig_method, "__func__") else orig_method).__get__(mlp, type(mlp))
    print(f"[selective] patched {len(orig_forwards)} MoE blocks", flush=True)

    # Tokenize.
    text = load_lcc_text(min_chars=1_400_000)
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    if not isinstance(ids, list):
        ids = list(ids)
    skip = 16 * args.chunk_len
    test_chunks = chunk_ids(ids, args.chunk_len, args.test_chunks, skip=skip)
    n_test_tok = args.test_chunks * args.chunk_len
    print(f"[selective] test_tokens={n_test_tok} (same lcc held-out split)", flush=True)

    def run_one(label: str, strategy: str, compress_frac: float) -> dict:
        _state.strategy = strategy
        _state.compress_frac = float(compress_frac)
        _state.keep_frac = float(args.keep_frac_when_compressed)
        _state.use_fp8 = True
        _state.use_l2 = True
        _state.seed = args.seed
        _layer_call_counter["i"] = 0

        losses, top1c, top5c, total = [], 0, 0, 0
        teacher_top1_cache: list[torch.Tensor] = []
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
                teacher_top1_cache.append(top1.cpu())

        avg_loss = float(np.mean(losses))
        return {
            "label": label,
            "strategy": strategy,
            "compress_frac": compress_frac,
            "loss": avg_loss,
            "ppl": math.exp(avg_loss),
            "top1": top1c / max(total, 1),
            "top5": top5c / max(total, 1),
            "top1_argmax": torch.cat([c.reshape(-1) for c in teacher_top1_cache], dim=0),
            "total": total,
        }

    # Run teacher (compress_frac=0 — strategy doesn't matter).
    print("[selective] running teacher (no compression) ...", flush=True)
    teacher = run_one("teacher", "random", 0.0)
    print(f"[selective]   teacher: ppl={teacher['ppl']:.3f} top1={teacher['top1']*100:.2f}% top5={teacher['top5']*100:.2f}%", flush=True)

    rows = [{**{k: v for k, v in teacher.items() if k != "top1_argmax"}, "agreement_with_teacher_top1": 1.0, "top1_drop_pp": 0.0, "top5_drop_pp": 0.0, "ppl_increase_pct": 0.0}]
    for strategy in args.strategies:
        for cf in args.compress_fracs:
            if cf == 0.0:
                continue  # already done as teacher
            label = f"{strategy}@cf={cf}"
            print(f"[selective] running {label} ...", flush=True)
            r = run_one(label, strategy, cf)
            agree = float((r["top1_argmax"] == teacher["top1_argmax"]).float().mean().item())
            row = {k: v for k, v in r.items() if k != "top1_argmax"}
            row["agreement_with_teacher_top1"] = agree
            row["ppl_increase_pct"] = (r["ppl"] - teacher["ppl"]) / teacher["ppl"] * 100
            row["top1_drop_pp"] = (teacher["top1"] - r["top1"]) * 100
            row["top5_drop_pp"] = (teacher["top5"] - r["top5"]) * 100
            rows.append(row)
            print(
                f"[selective]   {label}: ppl={r['ppl']:.3f} (+{row['ppl_increase_pct']:.1f}%) "
                f"top1={r['top1']*100:.2f}% (drop {row['top1_drop_pp']:.2f}pp) "
                f"agree={agree*100:.2f}%",
                flush=True,
            )

    # Unpatch (for cleanliness — though process is about to exit anyway).
    for attr, mlp, orig in orig_forwards:
        setattr(mlp, attr, orig)

    summary = {
        "model_path": args.model_path,
        "hidden_size": H,
        "test_tokens": n_test_tok,
        "domain": "lcc",
        "keep_frac_when_compressed": args.keep_frac_when_compressed,
        "strategies": args.strategies,
        "compress_fracs": args.compress_fracs,
        "teacher": {k: v for k, v in teacher.items() if k != "top1_argmax"},
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir/'summary.json'}", flush=True)

    # Markdown.
    lines = []
    lines.append("# Selective per-token compression on lcc (E1)")
    lines.append("")
    lines.append(f"Test = {n_test_tok} held-out lcc tokens. Teacher PPL = {teacher['ppl']:.3f}, top-1 = {teacher['top1']*100:.2f}%.")
    lines.append(f"Per-row compression (when applied) = topk_l2 + FP8 with keep_frac = {args.keep_frac_when_compressed} (4x value).")
    lines.append("")
    lines.append("## Next-token top-1 accuracy by (strategy, compress_frac)")
    lines.append("")
    hdr = "| strategy \\ compress_frac | " + " | ".join(str(cf) for cf in args.compress_fracs) + " |"
    sep = "|---|" + "---:|" * len(args.compress_fracs)
    lines.append(hdr); lines.append(sep)
    for strategy in args.strategies:
        cells = []
        for cf in args.compress_fracs:
            if cf == 0.0:
                cells.append(f"{teacher['top1']*100:.2f}%")
            else:
                match = [r for r in rows if r.get("strategy") == strategy and r.get("compress_frac") == cf]
                if match:
                    cells.append(f"{match[0]['top1']*100:.2f}% ({match[0]['top1_drop_pp']:+.2f}pp)")
                else:
                    cells.append("-")
        lines.append(f"| {strategy} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## PPL +% by (strategy, compress_frac)")
    lines.append("")
    lines.append(hdr); lines.append(sep)
    for strategy in args.strategies:
        cells = []
        for cf in args.compress_fracs:
            if cf == 0.0:
                cells.append("+0.0%")
            else:
                match = [r for r in rows if r.get("strategy") == strategy and r.get("compress_frac") == cf]
                if match:
                    cells.append(f"+{match[0]['ppl_increase_pct']:.1f}%")
                else:
                    cells.append("-")
        lines.append(f"| {strategy} | " + " | ".join(cells) + " |")
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"wrote {out_dir/'report.md'}", flush=True)

    # Heatmap.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mat = np.zeros((len(args.strategies), len(args.compress_fracs)))
        for r_idx, strategy in enumerate(args.strategies):
            for c_idx, cf in enumerate(args.compress_fracs):
                if cf == 0.0:
                    mat[r_idx, c_idx] = teacher["top1"] * 100
                else:
                    m = [r for r in rows if r.get("strategy") == strategy and r.get("compress_frac") == cf]
                    if m:
                        mat[r_idx, c_idx] = m[0]["top1"] * 100
        fig, ax = plt.subplots(figsize=(8, 4.5))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=mat.min(), vmax=teacher["top1"] * 100, aspect="auto")
        ax.set_xticks(range(len(args.compress_fracs)))
        ax.set_xticklabels([f"{cf*100:.0f}%" for cf in args.compress_fracs])
        ax.set_yticks(range(len(args.strategies)))
        ax.set_yticklabels(args.strategies)
        ax.set_xlabel("fraction of tokens compressed (4x each)")
        ax.set_ylabel("token selection strategy")
        ax.set_title(f"Next-token top-1 acc — teacher={teacher['top1']*100:.2f}%")
        for r_idx in range(mat.shape[0]):
            for c_idx in range(mat.shape[1]):
                ax.text(c_idx, r_idx, f"{mat[r_idx, c_idx]:.1f}", ha="center", va="center", fontsize=9,
                        color="black" if mat[r_idx, c_idx] > 70 else "white")
        fig.colorbar(im, ax=ax, label="top-1 acc (%)")
        fig.tight_layout()
        fig.savefig(out_dir / "heatmap.png", dpi=140)
        plt.close(fig)
        print(f"wrote {out_dir/'heatmap.png'}", flush=True)
    except Exception as e:
        print(f"plot failed: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
