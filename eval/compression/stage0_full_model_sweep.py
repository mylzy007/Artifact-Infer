"""Full-model SVD compression sweep on Qwen3-30B-A3B.

For every MoE layer (all 48), compute a per-layer SVD projector from calibration
text (lcc). Then at evaluation time, install a forward pre-hook on EVERY MoE
layer that replaces its input x with the SVD reconstruction P_up @ P_down @
(x - mu) + mu. Measure:

  - Cross-entropy loss / perplexity on held-out lcc test chunks.
  - Final hidden-state relative MSE vs the unhooked teacher.
  - Top-1 routing agreement teacher/student (averaged across all MoE layers).

Sweep ell in {d/2, d/4, d/8, d/16} (= 1024, 512, 256, 128 for d=2048).
Compare to teacher (no compression).

Outputs (under eval_results/compression_lowrank_stage0_full_sweep/):
  - summary.json     full table
  - report.md        human-readable
  - sweep_curve.png  PPL and hidden-relMSE vs ell

Run:
  CUDA_VISIBLE_DEVICES=0,1,2,3 /home/lzy/miniconda3/envs/atom/bin/python \
      eval/compression/stage0_full_model_sweep.py
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
OUT_DIR = Path("eval_results/compression_lowrank_stage0_full_sweep")
LB_DIR = Path("/home/lzy/datasets/moe_benchmarks/longbench/extracted/data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--chunk-len", type=int, default=512)
    p.add_argument("--calib-chunks", type=int, default=16)
    p.add_argument("--test-chunks", type=int, default=8)
    p.add_argument("--batch-chunks", type=int, default=4)
    p.add_argument("--ell-list", type=int, nargs="+", default=[1024, 512, 256, 128])
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

    print(f"[full_sweep] loading {args.model_path}", flush=True)
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
    embed_dev = next(model.model.embed_tokens.parameters()).device
    print(f"[full_sweep] H={H}, n_layers={n_layers}", flush=True)

    # Per-layer MoE block and its device.
    moe_blocks = [model.model.layers[L].mlp for L in range(n_layers)]
    moe_devs = [next(b.parameters()).device for b in moe_blocks]
    print(f"[full_sweep] MoE blocks on devices: {sorted(set(str(d) for d in moe_devs))}", flush=True)

    # Tokenize lcc.
    text = load_lcc_text(min_chars=1_400_000)
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    if not isinstance(ids, list):
        ids = list(ids)
    needed = (args.calib_chunks + args.test_chunks) * args.chunk_len
    assert len(ids) >= needed, f"only {len(ids)} tokens"
    all_chunks = chunk_ids(ids, args.chunk_len, args.calib_chunks + args.test_chunks)
    calib_chunks = all_chunks[: args.calib_chunks]
    test_chunks = all_chunks[args.calib_chunks : args.calib_chunks + args.test_chunks]
    n_calib_tok = args.calib_chunks * args.chunk_len
    n_test_tok = args.test_chunks * args.chunk_len
    print(f"[full_sweep] calib_tokens={n_calib_tok}, test_tokens={n_test_tok}", flush=True)

    # -------------------- Phase 1: collect calibration activations ---------------
    print("[full_sweep] phase 1: collecting per-layer dispatch input on calib chunks ...", flush=True)
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

    # -------------------- Phase 2: per-layer SVD for each ell --------------------
    print(f"[full_sweep] phase 2: per-layer SVD, ell in {args.ell_list} ...", flush=True)
    # Store projectors on the SAME device as each MoE block, in bf16, so the
    # eval-hook adds no cross-device transfers.
    projectors: dict[int, dict[int, dict]] = {L: {} for L in range(n_layers)}
    train_ev: dict[int, dict[int, float]] = {L: {} for L in range(n_layers)}
    for L in range(n_layers):
        X = torch.cat(calib_buf[L], dim=0)
        calib_buf[L] = []  # release
        mu = X.mean(dim=0, keepdim=True)
        Xc = X - mu
        # Full SVD on (N, d) for d=2048 in fp32 cpu; ~1-3s/layer.
        # We only need Vh (right singular vectors).
        _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        total_var = float((S ** 2).sum())
        for ell in args.ell_list:
            V_l = Vh[:ell].T.contiguous()              # [d, ell]
            P_down = V_l.T.contiguous()                # [ell, d]
            P_up = V_l.contiguous()                    # [d, ell]
            dev = moe_devs[L]
            projectors[L][ell] = {
                "P_down": P_down.to(dev, dtype=torch.bfloat16),
                "P_up": P_up.to(dev, dtype=torch.bfloat16),
                "mu": mu.to(dev, dtype=torch.bfloat16),
            }
            train_ev[L][ell] = float((S[:ell] ** 2).sum()) / max(total_var, 1e-12)
        if L % 8 == 0 or L == n_layers - 1:
            print(f"[full_sweep]   layer {L:>2d} done", flush=True)

    # -------------------- Phase 3: teacher forward (no compression) --------------
    print("[full_sweep] phase 3: teacher (no compression) forward on test ...", flush=True)
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
        if hasattr(moe_blocks[L], "gate"):
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

    teacher_H = torch.cat(teacher_h, dim=0)                              # [test_tok, d]
    teacher_T1 = {L: torch.cat(v, dim=0) for L, v in teacher_top1.items()}
    teacher_loss = float(np.mean(teacher_losses))
    teacher_ppl = math.exp(teacher_loss)
    print(f"[full_sweep] teacher: loss={teacher_loss:.4f}  ppl={teacher_ppl:.3f}", flush=True)

    # -------------------- Phase 4: student forwards (compressed) -----------------
    results = {}
    for ell in args.ell_list:
        print(f"[full_sweep] phase 4: student forward with ell={ell} (compression {H//ell}x) ...", flush=True)
        student_h: list[torch.Tensor] = []
        student_top1: dict[int, list[torch.Tensor]] = {L: [] for L in range(n_layers)}

        def make_pre_replace(L: int, ell: int):
            P_down = projectors[L][ell]["P_down"]
            P_up = projectors[L][ell]["P_up"]
            mu = projectors[L][ell]["mu"]
            def pre_replace(_m, inputs):
                x = inputs[0]
                B, T, Hd = x.shape
                xf = x.reshape(-1, Hd)
                z = (xf - mu) @ P_down.T
                x_hat = z @ P_up.T + mu
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
            handles.append(moe_blocks[L].register_forward_pre_hook(make_pre_replace(L, ell)))
            if hasattr(moe_blocks[L], "gate"):
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

        # Hidden relMSE.
        diff_sq = float(((student_H - teacher_H) ** 2).sum())
        norm_sq = float((teacher_H ** 2).sum())
        rel_h = diff_sq / max(norm_sq, 1e-12)

        # Routing agreement averaged across layers.
        agree_per_layer = []
        for L in range(n_layers):
            t1 = torch.cat(student_top1[L], dim=0)
            t0 = teacher_T1[L]
            n = min(t1.shape[0], t0.shape[0])
            agree_per_layer.append(float((t1[:n] == t0[:n]).float().mean().item()))
        mean_agree = float(np.mean(agree_per_layer))

        # Train EV averaged across layers (informational).
        avg_train_ev = float(np.mean([train_ev[L][ell] for L in range(n_layers)]))

        ratio_h = rel_h  # already a ratio
        ppl_increase = (student_ppl - teacher_ppl) / teacher_ppl
        ppl_increase_abs = student_ppl - teacher_ppl

        results[str(ell)] = {
            "ell": int(ell),
            "compression": H // ell,
            "avg_train_explained_variance": avg_train_ev,
            "student_loss": student_loss,
            "student_ppl": student_ppl,
            "teacher_loss": teacher_loss,
            "teacher_ppl": teacher_ppl,
            "ppl_increase_pct": ppl_increase * 100,
            "ppl_increase_abs": ppl_increase_abs,
            "final_hidden_relMSE": rel_h,
            "top1_routing_agreement_mean": mean_agree,
            "top1_routing_agreement_per_layer_min": float(np.min(agree_per_layer)),
            "top1_routing_agreement_per_layer_max": float(np.max(agree_per_layer)),
        }
        print(
            f"[full_sweep]   ell={ell} ({H//ell}x): "
            f"loss={student_loss:.4f} ppl={student_ppl:.3f} (+{ppl_increase*100:.1f}% vs teacher) "
            f"hid_relMSE={rel_h:.4f} routing_agree={mean_agree:.3f} "
            f"avg_train_EV={avg_train_ev:.3f}",
            flush=True,
        )

    # -------------------- Output --------------------
    summary = {
        "model_path": args.model_path,
        "hidden_size": H,
        "n_layers": n_layers,
        "calib_tokens": n_calib_tok,
        "test_tokens": n_test_tok,
        "domain": "lcc",
        "ell_list": args.ell_list,
        "teacher": {"loss": teacher_loss, "ppl": teacher_ppl},
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir/'summary.json'}", flush=True)

    # Report markdown.
    lines = []
    lines.append("# Full-model SVD compression sweep (Qwen3-30B-A3B, lcc domain)")
    lines.append("")
    lines.append(f"- All {n_layers} MoE layers compressed simultaneously, per-layer SVD projector from {n_calib_tok} lcc calibration tokens.")
    lines.append(f"- Test: {n_test_tok} held-out lcc tokens. Teacher PPL = {teacher_ppl:.3f} (loss {teacher_loss:.4f}).")
    lines.append("")
    lines.append("## Sweep")
    lines.append("")
    lines.append("| ell | compression | avg train EV | student PPL | PPL +% | final hidden relMSE | mean top-1 routing agree | per-layer agree min |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ell in args.ell_list:
        r = results[str(ell)]
        lines.append(
            f"| {ell} | {H//ell}x | {r['avg_train_explained_variance']:.3f} | "
            f"{r['student_ppl']:.3f} | {r['ppl_increase_pct']:+.1f}% | "
            f"{r['final_hidden_relMSE']:.4f} | "
            f"{r['top1_routing_agreement_mean']:.3f} | "
            f"{r['top1_routing_agreement_per_layer_min']:.3f} |"
        )
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append("- `compression` = d / ell. 2x means each token's dispatch payload shrinks from 2048 floats to 1024.")
    lines.append("- `avg train EV` is the in-sample explained variance averaged across all 48 layers (the L^2 measure we used in earlier stages).")
    lines.append("- `student PPL +%` is what the user actually feels. <5% likely acceptable, <1% very tight.")
    lines.append("- `final hidden relMSE` is sum-of-squared-error / sum-of-squares on the post-norm output. Small means task-level damage is small.")
    lines.append("- `mean top-1 routing agreement` is the fraction of tokens whose top-1 expert assignment matches the uncompressed teacher, averaged across all 48 MoE layers. Big drop here is the canonical 'compression broke routing' signal.")
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"wrote {out_dir/'report.md'}", flush=True)

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ratios = [H // e for e in args.ell_list]
        ppls = [results[str(e)]["student_ppl"] for e in args.ell_list]
        rels = [results[str(e)]["final_hidden_relMSE"] for e in args.ell_list]
        agrees = [results[str(e)]["top1_routing_agreement_mean"] for e in args.ell_list]

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
        ax0, ax1, ax2 = axes
        ax0.plot(ratios, ppls, marker="o")
        ax0.axhline(teacher_ppl, color="red", linestyle="--", linewidth=0.8, label=f"teacher PPL = {teacher_ppl:.2f}")
        ax0.set_xlabel("compression ratio (d / ell)")
        ax0.set_ylabel("test PPL")
        ax0.set_xscale("log", base=2)
        ax0.set_title("Test PPL vs compression")
        ax0.grid(True, alpha=0.3)
        ax0.legend(fontsize=8)

        ax1.plot(ratios, rels, marker="o", color="C2")
        ax1.set_xlabel("compression ratio (d / ell)")
        ax1.set_ylabel("final hidden relMSE")
        ax1.set_xscale("log", base=2)
        ax1.set_yscale("log")
        ax1.set_title("Final hidden drift vs compression")
        ax1.grid(True, alpha=0.3, which="both")

        ax2.plot(ratios, agrees, marker="o", color="C3")
        ax2.set_xlabel("compression ratio (d / ell)")
        ax2.set_ylabel("mean top-1 routing agreement")
        ax2.set_xscale("log", base=2)
        ax2.set_title("Routing preservation vs compression")
        ax2.set_ylim(0, 1.02)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"All-{n_layers}-layer SVD compression (Qwen3-30B-A3B, lcc test, calib={n_calib_tok}/test={n_test_tok} tokens)")
        fig.tight_layout()
        fig.savefig(out_dir / "sweep_curve.png", dpi=140)
        plt.close(fig)
        print(f"wrote {out_dir/'sweep_curve.png'}", flush=True)
    except Exception as e:
        print(f"plot failed: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
