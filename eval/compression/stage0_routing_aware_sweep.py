"""Routing-aware projection vs vanilla SVD: full-model sweep.

Hypothesis being tested:
  The reason vanilla SVD breaks Qwen3 at 2x compression is that it preserves
  the *bulk* of activation x but discards the (possibly small-norm) directions
  in which the router gate W_gate is sensitive. Top-k routing flips when those
  directions are dropped.

Fix:
  Build a projector P whose column space STRICTLY CONTAINS the row space of
  W_gate (so W_gate @ P @ x == W_gate @ x exactly). Specifically, with ell
  columns:
    - First r = rank(W_gate) directions = orthonormal basis of W_gate^T (via QR).
    - Remaining ell - r directions = top SVD directions of x, projected onto
      the orthogonal complement of the gate subspace.
  This is the smallest possible modification to SVD that preserves routing
  perfectly, and uses any remaining budget for best L^2 reconstruction.

Compare at ell in {1024, 512, 256} (= 2x, 4x, 8x compression) on all 48 MoE
layers simultaneously, lcc domain, against the vanilla-SVD numbers from the
prior sweep.

Outputs (under eval_results/compression_lowrank_stage0_routing_aware/):
  - summary.json
  - report.md
  - comparison_curve.png

Note: rank(W_gate) <= num_experts = 128. So ell >= 128 (2x and 4x) can preserve
routing exactly. ell = 256 (8x) also has 256 - 128 = 128 dims left for SVD.

Run:
  CUDA_VISIBLE_DEVICES=0,1,2,3 /home/lzy/miniconda3/envs/atom/bin/python \
      eval/compression/stage0_routing_aware_sweep.py
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
OUT_DIR = Path("eval_results/compression_lowrank_stage0_routing_aware")
LB_DIR = Path("/home/lzy/datasets/moe_benchmarks/longbench/extracted/data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--chunk-len", type=int, default=512)
    p.add_argument("--calib-chunks", type=int, default=16)
    p.add_argument("--test-chunks", type=int, default=8)
    p.add_argument("--batch-chunks", type=int, default=4)
    p.add_argument("--ell-list", type=int, nargs="+", default=[1024, 512, 256])
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


def build_svd_projector(Xc: torch.Tensor, ell: int) -> torch.Tensor:
    """Standard SVD projector. Returns P_up of shape [d, ell] with orthonormal cols.
    Reconstruction is x_hat = P_up @ P_up^T @ (x - mu) + mu."""
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Vh[:ell].T.contiguous()


def build_routing_aware_projector(Xc: torch.Tensor, W_gate: torch.Tensor, ell: int) -> torch.Tensor:
    """Projector whose column space strictly contains row(W_gate).

    W_gate: [E, d]. Steps:
      1. Q_g, _ = qr(W_gate^T)  ->  [d, E] orthonormal basis of row(W_gate).
      2. Compute SVD of Xc, take top many directions Vh^T = V_full.
      3. Subtract their projection onto Q_g: V_perp = V_full - Q_g (Q_g^T V_full).
      4. QR on V_perp to re-orthonormalize, take top (ell - E) of those.
      5. P_up = [Q_g | top of V_perp].
    """
    d = Xc.shape[1]
    E = W_gate.shape[0]
    # Promote to fp64 for QR/SVD stability since the projection cascade compounds errors.
    W64 = W_gate.to(torch.float64).cpu()
    X64 = Xc.to(torch.float64)
    Q_g, _ = torch.linalg.qr(W64.T, mode="reduced")        # [d, E]
    # The gate is full row-rank in Qwen3 (E=128 < d=2048), so Q_g has E cols.
    r_g = Q_g.shape[1]
    if ell <= r_g:
        # Pathological: ell smaller than gate rank. Truncate by gate singular order.
        U_w, S_w, _ = torch.linalg.svd(W64, full_matrices=False)
        # Project rows of W_gate to a smaller orthonormal basis by keeping top ell singular dirs.
        # W = U_w S_w Vh_w; rowspace top ell is span of first ell cols of Vh_w^T (= V_w[:, :ell])
        # Simplest: keep top-ell rows of Q_g (good enough as approximation).
        P_up = Q_g[:, :ell].contiguous().to(torch.float32)
        return P_up
    # Compute SVD on X.
    _, _, Vh = torch.linalg.svd(X64, full_matrices=False)
    V_full = Vh.T.contiguous()                             # [d, min(N, d)]
    # Subtract projection onto gate subspace.
    coef = Q_g.T @ V_full                                  # [E, k]
    V_perp = V_full - Q_g @ coef                           # [d, k]
    # Orthonormalize via QR (columns), drop zero / near-zero cols.
    Q_p, R_p = torch.linalg.qr(V_perp, mode="reduced")
    diag = torch.diagonal(R_p)
    # Sort columns by |R_diag| descending so we keep the directions with most residual energy.
    order = torch.argsort(diag.abs(), descending=True)
    Q_p = Q_p[:, order]
    n_extra = min(ell - r_g, Q_p.shape[1])
    P_up = torch.cat([Q_g, Q_p[:, :n_extra]], dim=1).contiguous().to(torch.float32)
    assert P_up.shape == (d, ell), (P_up.shape, d, ell)
    return P_up


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[routing_aware] loading {args.model_path}", flush=True)
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
    print(f"[routing_aware] H={H}, n_layers={n_layers}, num_experts={E}", flush=True)

    moe_blocks = [model.model.layers[L].mlp for L in range(n_layers)]
    moe_devs = [next(b.parameters()).device for b in moe_blocks]

    text = load_lcc_text(min_chars=1_400_000)
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    if not isinstance(ids, list):
        ids = list(ids)
    all_chunks = chunk_ids(ids, args.chunk_len, args.calib_chunks + args.test_chunks)
    calib_chunks = all_chunks[: args.calib_chunks]
    test_chunks = all_chunks[args.calib_chunks : args.calib_chunks + args.test_chunks]
    n_calib_tok = args.calib_chunks * args.chunk_len
    n_test_tok = args.test_chunks * args.chunk_len
    print(f"[routing_aware] calib_tokens={n_calib_tok}, test_tokens={n_test_tok}", flush=True)

    # -------------------- Phase 1: collect calibration activations ---------------
    print("[routing_aware] phase 1: collecting per-layer dispatch input on calib chunks ...", flush=True)
    calib_buf: dict[int, list[torch.Tensor]] = {L: [] for L in range(n_layers)}

    def make_calib_hook(L: int):
        def pre_hook(_m, inputs):
            x = inputs[0].detach()
            calib_buf[L].append(x.to("cpu", dtype=torch.float32).reshape(-1, x.shape[-1]))
            return None
        return pre_hook

    handles = [moe_blocks[L].register_forward_pre_hook(make_calib_hook(L)) for L in range(n_layers)]
    with torch.no_grad():
        ids_t = torch.tensor(calib_chunks, dtype=torch.long, device=embed_dev)
        for i in range(0, ids_t.shape[0], args.batch_chunks):
            _ = model(input_ids=ids_t[i : i + args.batch_chunks], use_cache=False)
    for h in handles:
        h.remove()

    # -------------------- Phase 2: build projectors -----------------------------
    # variant: 'svd' or 'routing'
    # projectors[variant][L][ell] = {P_up, mu}
    projectors: dict[str, dict[int, dict[int, dict]]] = {"svd": {}, "routing": {}}
    for L in range(n_layers):
        X = torch.cat(calib_buf[L], dim=0)
        calib_buf[L] = []
        mu = X.mean(dim=0, keepdim=True)
        Xc = X - mu
        # Grab the gate weight for this layer's MoE block.
        W_gate = moe_blocks[L].gate.weight.detach().to("cpu", dtype=torch.float32)   # [E, d]
        for v in ("svd", "routing"):
            projectors[v].setdefault(L, {})
        for ell in args.ell_list:
            P_up_svd = build_svd_projector(Xc, ell)                                  # [d, ell]
            P_up_rt = build_routing_aware_projector(Xc, W_gate, ell)                 # [d, ell]
            dev = moe_devs[L]
            projectors["svd"][L][ell] = {
                "P_up": P_up_svd.to(dev, dtype=torch.bfloat16),
                "mu": mu.to(dev, dtype=torch.bfloat16),
            }
            projectors["routing"][L][ell] = {
                "P_up": P_up_rt.to(dev, dtype=torch.bfloat16),
                "mu": mu.to(dev, dtype=torch.bfloat16),
            }
        if L % 8 == 0 or L == n_layers - 1:
            print(f"[routing_aware]   layer {L:>2d}/{n_layers}: projectors built", flush=True)

    # -------------------- Phase 3: teacher forward (no compression) --------------
    print("[routing_aware] phase 3: teacher forward ...", flush=True)
    teacher_h: list[torch.Tensor] = []
    teacher_top1: dict[int, list[torch.Tensor]] = {L: [] for L in range(n_layers)}

    def teacher_norm_hook(_m, _inputs, output):
        teacher_h.append(output.detach().to("cpu", dtype=torch.float32).reshape(-1, output.shape[-1]))

    def make_teacher_gate_hook(L: int):
        def hook(_m, _inputs, output):
            top1 = output.detach().argmax(dim=-1).to("cpu").reshape(-1)
            teacher_top1[L].append(top1)
        return hook

    teacher_handles = [model.model.norm.register_forward_hook(teacher_norm_hook)]
    for L in range(n_layers):
        teacher_handles.append(moe_blocks[L].gate.register_forward_hook(make_teacher_gate_hook(L)))

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
    teacher_T1 = {L: torch.cat(v, dim=0) for L, v in teacher_top1.items()}
    teacher_loss = float(np.mean(teacher_losses))
    teacher_ppl = math.exp(teacher_loss)
    print(f"[routing_aware] teacher: loss={teacher_loss:.4f}  ppl={teacher_ppl:.3f}", flush=True)

    # -------------------- Phase 4: student forwards -----------------------------
    results: dict[str, dict[str, dict]] = {"svd": {}, "routing": {}}
    for variant in ("svd", "routing"):
        for ell in args.ell_list:
            print(f"[routing_aware] phase 4: variant={variant} ell={ell} ({H//ell}x) ...", flush=True)
            student_h: list[torch.Tensor] = []
            student_top1: dict[int, list[torch.Tensor]] = {L: [] for L in range(n_layers)}

            def make_pre_replace(L: int):
                P_up = projectors[variant][L][ell]["P_up"]
                mu = projectors[variant][L][ell]["mu"]
                def pre_replace(_m, inputs):
                    x = inputs[0]
                    B, T, Hd = x.shape
                    xf = x.reshape(-1, Hd)
                    # x_hat = P_up @ P_up^T @ (x - mu) + mu  with orthonormal P_up cols
                    z = (xf - mu) @ P_up          # [N, ell]
                    x_hat = z @ P_up.T + mu       # [N, d]
                    x_hat = x_hat.view(B, T, Hd).to(x.dtype)
                    return (x_hat,) + tuple(inputs[1:])
                return pre_replace

            def student_norm_hook(_m, _inputs, output):
                student_h.append(output.detach().to("cpu", dtype=torch.float32).reshape(-1, output.shape[-1]))

            def make_student_gate_hook(L: int):
                def hook(_m, _inputs, output):
                    top1 = output.detach().argmax(dim=-1).to("cpu").reshape(-1)
                    student_top1[L].append(top1)
                return hook

            handles = []
            for L in range(n_layers):
                handles.append(moe_blocks[L].register_forward_pre_hook(make_pre_replace(L)))
                handles.append(moe_blocks[L].gate.register_forward_hook(make_student_gate_hook(L)))
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

            agree_per_layer = []
            for L in range(n_layers):
                t1 = torch.cat(student_top1[L], dim=0)
                t0 = teacher_T1[L]
                n = min(t1.shape[0], t0.shape[0])
                agree_per_layer.append(float((t1[:n] == t0[:n]).float().mean().item()))
            mean_agree = float(np.mean(agree_per_layer))

            results[variant][str(ell)] = {
                "ell": int(ell),
                "compression": H // ell,
                "student_loss": student_loss,
                "student_ppl": student_ppl,
                "ppl_increase_pct": (student_ppl - teacher_ppl) / teacher_ppl * 100,
                "final_hidden_relMSE": rel_h,
                "top1_routing_agreement_mean": mean_agree,
                "top1_routing_agreement_min": float(np.min(agree_per_layer)),
                "top1_routing_agreement_max": float(np.max(agree_per_layer)),
            }
            print(
                f"[routing_aware]   {variant:8s} ell={ell} ({H//ell}x): "
                f"ppl={student_ppl:.3f} (+{(student_ppl-teacher_ppl)/teacher_ppl*100:.1f}%) "
                f"hid_relMSE={rel_h:.4f} routing_agree={mean_agree:.3f}",
                flush=True,
            )

    # -------------------- Output --------------------
    summary = {
        "model_path": args.model_path,
        "hidden_size": H,
        "n_layers": n_layers,
        "num_experts": E,
        "calib_tokens": n_calib_tok,
        "test_tokens": n_test_tok,
        "domain": "lcc",
        "ell_list": args.ell_list,
        "teacher": {"loss": teacher_loss, "ppl": teacher_ppl},
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir/'summary.json'}", flush=True)

    # Report.
    lines = []
    lines.append(f"# Routing-aware vs vanilla SVD projection (Qwen3-30B-A3B, lcc, {n_layers} MoE layers)")
    lines.append("")
    lines.append(f"- Teacher PPL = {teacher_ppl:.3f}; calib={n_calib_tok} tokens, test={n_test_tok} tokens.")
    lines.append(f"- num_experts = {E}; routing-aware variant strictly preserves W_gate row space (rank ≤ {E}).")
    lines.append("")
    lines.append("## Side-by-side")
    lines.append("")
    lines.append("| ell | compression | variant | PPL | PPL +% | final hidden relMSE | top-1 routing agree (mean / min) |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|")
    for ell in args.ell_list:
        for variant in ("svd", "routing"):
            r = results[variant][str(ell)]
            lines.append(
                f"| {ell} | {H//ell}x | {variant} | {r['student_ppl']:.3f} | "
                f"{r['ppl_increase_pct']:+.1f}% | {r['final_hidden_relMSE']:.4f} | "
                f"{r['top1_routing_agreement_mean']:.3f} / {r['top1_routing_agreement_min']:.3f} |"
            )
    lines.append("")
    lines.append("## How to read")
    lines.append("")
    lines.append("- The routing-aware projector spends the first 128 (= num_experts) dimensions of its budget on the EXACT row space of W_gate, so the routing logits are mathematically unchanged.")
    lines.append("- 'routing agree' should be 1.000 for the routing variant if the math is implemented correctly (within numerical precision; bf16 cast and downstream chain may flip a few).")
    lines.append("- If the routing-aware variant achieves much lower PPL increase, the failure mode of vanilla SVD really is routing destruction; we should pivot the projector design.")
    lines.append("- If the routing-aware variant is still bad, the failure is elsewhere (multi-layer error compounding into the residual stream, not routing) and pure dispatch-side low-rank is dead regardless of projector choice.")
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"wrote {out_dir/'report.md'}", flush=True)

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ratios = [H // e for e in args.ell_list]
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
        for variant, marker, color in (("svd", "o", "C0"), ("routing", "s", "C2")):
            ppls = [results[variant][str(e)]["student_ppl"] for e in args.ell_list]
            rels = [results[variant][str(e)]["final_hidden_relMSE"] for e in args.ell_list]
            agrees = [results[variant][str(e)]["top1_routing_agreement_mean"] for e in args.ell_list]
            axes[0].plot(ratios, ppls, marker=marker, color=color, label=variant)
            axes[1].plot(ratios, rels, marker=marker, color=color, label=variant)
            axes[2].plot(ratios, agrees, marker=marker, color=color, label=variant)
        axes[0].axhline(teacher_ppl, color="red", linestyle="--", linewidth=0.8, label=f"teacher = {teacher_ppl:.2f}")
        for ax, title, ylab in zip(
            axes,
            ("Test PPL", "Final hidden relMSE", "Mean top-1 routing agreement"),
            ("PPL", "relMSE", "fraction"),
        ):
            ax.set_xlabel("compression (d / ell)")
            ax.set_ylabel(ylab)
            ax.set_xscale("log", base=2)
            ax.set_title(title)
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(fontsize=8)
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[2].set_ylim(0, 1.02)
        fig.suptitle(f"Routing-aware vs vanilla SVD (all {n_layers} MoE layers, lcc)")
        fig.tight_layout()
        fig.savefig(out_dir / "comparison_curve.png", dpi=140)
        plt.close(fig)
        print(f"wrote {out_dir/'comparison_curve.png'}", flush=True)
    except Exception as e:
        print(f"plot failed: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
