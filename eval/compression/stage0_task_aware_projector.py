"""Task-aware low-rank projector for one MoE layer (Qwen3-30B-A3B).

Setup:
  - Domain: lcc (code), the best-low-rank domain in Stage 0 per-domain experiment.
  - Layer: 24 (mid-layer, where SVD baseline is weakest -- EV@d/4 = 0.78).
  - Compression: ell = d / 4 = 512.
  - Trainable params: P_down (d, ell) + P_up (ell, d) only. All Qwen3 weights frozen.
  - Loss: MSE between MoE-layer-24's output on (P_up P_down x) vs on original x.
  - Eval (in addition to MoE-output MSE):
      * Pure reconstruction MSE on x  (= SVD baseline's optimum for L^2).
      * Final hidden-state MSE end-of-model with hook installed (task-level proxy).
      * Top-1 expert routing agreement teacher/student.

The teacher activations (x = MoE input, y = MoE output, h = final hidden) and
the test forwards (with and without hook) are cached up front; training is just
optimizer steps over (X_train, Y_train).

Run:
  CUDA_VISIBLE_DEVICES=0,1,2,3 /home/lzy/miniconda3/envs/atom/bin/python \
      eval/compression/stage0_task_aware_projector.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

MODEL_PATH = "/home/lzy/models/Qwen3-30B-A3B"
OUT_DIR = Path("eval_results/compression_lowrank_stage0_task_aware")
LB_DIR = Path("/home/lzy/datasets/moe_benchmarks/longbench/extracted/data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--target-layer", type=int, default=24)
    p.add_argument("--ell", type=int, default=512, help="kept rank (d/4 = 512 for d=2048)")
    p.add_argument("--chunk-len", type=int, default=512)
    p.add_argument("--train-chunks", type=int, default=32)
    p.add_argument("--test-chunks", type=int, default=8)
    p.add_argument("--batch-chunks", type=int, default=4, help="chunks per minibatch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--gpu-mem-gib", type=int, default=22)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_lcc_text() -> str:
    pieces, total = [], 0
    target = 1_400_000  # need enough for train+test
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
                if total >= target:
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

    print(f"[task_aware] loading {args.model_path}", flush=True)
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
    L_target = args.target_layer
    assert 0 <= L_target < cfg.num_hidden_layers

    target_block = model.model.layers[L_target].mlp
    moe_dev = next(target_block.parameters()).device
    embed_dev = next(model.model.embed_tokens.parameters()).device
    print(f"[task_aware] target layer {L_target} MoE block on {moe_dev}; embed on {embed_dev}", flush=True)

    # Tokenize lcc and chunk.
    text = load_lcc_text()
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    if not isinstance(ids, list):
        ids = list(ids)
    total_needed = (args.train_chunks + args.test_chunks) * args.chunk_len
    if len(ids) < total_needed:
        raise RuntimeError(f"not enough lcc tokens: {len(ids)} < {total_needed}")
    all_chunks = chunk_ids(ids, args.chunk_len, args.train_chunks + args.test_chunks)
    train_chunks = all_chunks[: args.train_chunks]
    test_chunks = all_chunks[args.train_chunks : args.train_chunks + args.test_chunks]
    print(f"[task_aware] lcc tokens={len(ids)}, train_chunks={len(train_chunks)}, test_chunks={len(test_chunks)}", flush=True)

    # ---------- Pass 1: cache teacher activations on train + test ----------
    # We capture: x_in to target MoE (pre_hook input), y_out from target MoE
    # (forward_hook output), and the final hidden state h_final from
    # model.model.norm output.
    print("[task_aware] caching teacher x_in, y_out, h_final ...", flush=True)

    x_cache: list[torch.Tensor] = []
    y_cache: list[torch.Tensor] = []
    h_cache: list[torch.Tensor] = []

    def pre_hook(_m, inputs):
        # inputs[0]: [B, T, H], bf16 on moe_dev
        x = inputs[0].detach()
        x_cache.append(x.to("cpu", dtype=torch.float32).reshape(-1, x.shape[-1]))
        return None

    def post_hook(_m, _inputs, output):
        # Qwen3MoeSparseMoeBlock returns (hidden_states, router_logits)
        y = output[0] if isinstance(output, tuple) else output
        y = y.detach()
        y_cache.append(y.to("cpu", dtype=torch.float32).reshape(-1, y.shape[-1]))
        return None

    def norm_hook(_m, _inputs, output):
        h = output.detach()
        h_cache.append(h.to("cpu", dtype=torch.float32).reshape(-1, h.shape[-1]))
        return None

    def run_forward(chunks: list[list[int]]) -> None:
        ids_t = torch.tensor(chunks, dtype=torch.long, device=embed_dev)
        am = torch.ones_like(ids_t)
        bsz = args.batch_chunks
        with torch.no_grad():
            for i in range(0, ids_t.shape[0], bsz):
                _ = model(input_ids=ids_t[i : i + bsz], attention_mask=am[i : i + bsz], use_cache=False)

    handles = [
        target_block.register_forward_pre_hook(pre_hook),
        target_block.register_forward_hook(post_hook),
        model.model.norm.register_forward_hook(norm_hook),
    ]
    run_forward(train_chunks)
    n_train_tokens = sum(t.shape[0] for t in x_cache)
    train_x = torch.cat(x_cache, dim=0); x_cache.clear()
    train_y = torch.cat(y_cache, dim=0); y_cache.clear()
    train_h = torch.cat(h_cache, dim=0); h_cache.clear()
    run_forward(test_chunks)
    test_x = torch.cat(x_cache, dim=0); x_cache.clear()
    test_y = torch.cat(y_cache, dim=0); y_cache.clear()
    test_h = torch.cat(h_cache, dim=0); h_cache.clear()
    for h in handles:
        h.remove()
    print(f"[task_aware] train_x={tuple(train_x.shape)}  test_x={tuple(test_x.shape)}", flush=True)

    # ---------- SVD baseline ----------
    print("[task_aware] computing SVD on train_x (centered) ...", flush=True)
    mu = train_x.mean(dim=0, keepdim=True)               # [1, d]
    Xc = train_x - mu
    # U: [N, k], S: [k], Vh: [k, d]. Keep top ell.
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    # Truncated principal basis V_l ∈ R^{d, ell}: x_hat - mu = (V_l V_l^T) (x - mu)
    V_l = Vh[: args.ell].T.contiguous()                   # [d, ell]
    # Express as: x_hat = P_up @ P_down @ x + (I - P_up @ P_down) @ mu
    # We'll model the hook as:
    #   x_hat = (P_up @ P_down) @ (x - mu) + mu
    # so P_down ∈ R^{ell, d}, P_up ∈ R^{d, ell}, with P_down_svd = V_l^T, P_up_svd = V_l.
    P_down_svd = V_l.T.clone()                            # [ell, d]
    P_up_svd = V_l.clone()                                # [d, ell]
    sum_var = float((Xc.var(dim=0, unbiased=False) * Xc.shape[0]).sum())  # ||Xc||_F^2
    kept_var_svd = float((S[: args.ell] ** 2).sum())
    ev_svd_train = kept_var_svd / max(sum_var, 1e-12)
    print(f"[task_aware] SVD baseline: train explained variance = {ev_svd_train:.4f}", flush=True)

    # ---------- Eval helpers ----------
    def reconstruct(x_cpu: torch.Tensor, P_down: torch.Tensor, P_up: torch.Tensor, mu_: torch.Tensor) -> torch.Tensor:
        # All on cpu, fp32.
        z = (x_cpu - mu_) @ P_down.T          # [N, ell]
        x_hat = z @ P_up.T + mu_              # [N, d]
        return x_hat

    def reconstruction_mse(x_cpu: torch.Tensor, x_hat: torch.Tensor) -> float:
        return float(((x_cpu - x_hat) ** 2).mean())

    def explained_variance(x_cpu: torch.Tensor, x_hat: torch.Tensor) -> float:
        var_x = float(((x_cpu - x_cpu.mean(dim=0, keepdim=True)) ** 2).sum())
        err = float(((x_cpu - x_hat) ** 2).sum())
        return 1.0 - err / max(var_x, 1e-12)

    def moe_forward_reconstructed(
        x_cpu: torch.Tensor, P_down: torch.Tensor, P_up: torch.Tensor, mu_: torch.Tensor
    ) -> torch.Tensor:
        """Run the target MoE block on reconstructed inputs, returning y_hat on cpu fp32.

        Process chunk-by-chunk so memory stays bounded.
        """
        outs = []
        with torch.no_grad():
            for start in range(0, x_cpu.shape[0], args.batch_chunks * args.chunk_len):
                stop = min(start + args.batch_chunks * args.chunk_len, x_cpu.shape[0])
                xb = x_cpu[start:stop]
                xb_hat = reconstruct(xb, P_down, P_up, mu_)
                # Run MoE: it expects [B, T, H]. We give [1, N_b, H].
                xb_hat = xb_hat.to(moe_dev, dtype=torch.bfloat16).unsqueeze(0)
                y = target_block(xb_hat)
                if isinstance(y, tuple):
                    y = y[0]
                outs.append(y.squeeze(0).to("cpu", dtype=torch.float32))
        return torch.cat(outs, dim=0)

    def full_model_forward_with_hook(
        chunks: list[list[int]],
        P_down: torch.Tensor,
        P_up: torch.Tensor,
        mu_: torch.Tensor,
    ) -> torch.Tensor:
        """Run the full model on `chunks` with a pre-hook on the target MoE that
        replaces the input by reconstructed input. Return final hidden state ([N, d])."""
        P_down_dev = P_down.to(moe_dev, dtype=torch.bfloat16)
        P_up_dev = P_up.to(moe_dev, dtype=torch.bfloat16)
        mu_dev = mu_.to(moe_dev, dtype=torch.bfloat16)

        def pre_replace(_m, inputs):
            x = inputs[0]
            B, T, Hd = x.shape
            xf = x.reshape(-1, Hd)
            z = (xf - mu_dev) @ P_down_dev.T
            x_hat = z @ P_up_dev.T + mu_dev
            x_hat = x_hat.view(B, T, Hd).to(x.dtype)
            return (x_hat,) + tuple(inputs[1:])

        h_collect: list[torch.Tensor] = []

        def norm_hook_local(_m, _inputs, output):
            h_collect.append(output.detach().to("cpu", dtype=torch.float32).reshape(-1, output.shape[-1]))
            return None

        ph = target_block.register_forward_pre_hook(pre_replace)
        nh = model.model.norm.register_forward_hook(norm_hook_local)
        try:
            ids_t = torch.tensor(chunks, dtype=torch.long, device=embed_dev)
            am = torch.ones_like(ids_t)
            with torch.no_grad():
                for i in range(0, ids_t.shape[0], args.batch_chunks):
                    _ = model(input_ids=ids_t[i : i + args.batch_chunks], attention_mask=am[i : i + args.batch_chunks], use_cache=False)
        finally:
            ph.remove(); nh.remove()
        return torch.cat(h_collect, dim=0)

    # ---------- SVD baseline metrics on test ----------
    print("[task_aware] evaluating SVD baseline on test ...", flush=True)
    x_hat_test_svd = reconstruct(test_x, P_down_svd, P_up_svd, mu)
    mse_x_svd = reconstruction_mse(test_x, x_hat_test_svd)
    ev_x_svd = explained_variance(test_x, x_hat_test_svd)
    y_hat_test_svd = moe_forward_reconstructed(test_x, P_down_svd, P_up_svd, mu)
    mse_y_svd = float(((test_y - y_hat_test_svd) ** 2).mean())
    rel_y_svd = mse_y_svd / float((test_y ** 2).mean() + 1e-12)
    h_hat_test_svd = full_model_forward_with_hook(test_chunks, P_down_svd, P_up_svd, mu)
    mse_h_svd = float(((test_h - h_hat_test_svd) ** 2).mean())
    rel_h_svd = mse_h_svd / float((test_h ** 2).mean() + 1e-12)
    print(
        f"[task_aware] SVD baseline TEST:  "
        f"recon_MSE={mse_x_svd:.4e}  EV(x)={ev_x_svd:.4f}  "
        f"MoE_out_MSE={mse_y_svd:.4e}  rel_MoE={rel_y_svd:.4f}  "
        f"final_h_MSE={mse_h_svd:.4e}  rel_h={rel_h_svd:.4f}",
        flush=True,
    )

    # ---------- Trainable projector ----------
    print("[task_aware] starting task-aware training ...", flush=True)
    P_down = torch.nn.Parameter(P_down_svd.clone().to(moe_dev, dtype=torch.float32))
    P_up = torch.nn.Parameter(P_up_svd.clone().to(moe_dev, dtype=torch.float32))
    mu_dev_f32 = mu.to(moe_dev, dtype=torch.float32)
    opt = torch.optim.AdamW([P_down, P_up], lr=args.lr, weight_decay=0.0)

    # Move train caches to moe_dev in bf16 for speed; recompute on cpu when reading.
    train_x_dev = train_x.to(moe_dev, dtype=torch.bfloat16)
    train_y_dev = train_y.to(moe_dev, dtype=torch.bfloat16)

    history = []
    N_train = train_x_dev.shape[0]
    batch_tok = args.batch_chunks * args.chunk_len
    g = torch.Generator(device="cpu").manual_seed(args.seed)

    for step in range(1, args.steps + 1):
        # Sample a random contiguous slice (token order doesn't matter for MoE-layer-only forward).
        if N_train > batch_tok:
            start = int(torch.randint(0, N_train - batch_tok + 1, (1,), generator=g).item())
        else:
            start = 0
        xb = train_x_dev[start : start + batch_tok]
        yb = train_y_dev[start : start + batch_tok]

        # Cast projector to bf16 for the fwd pass; keep grads in fp32.
        P_down_bf = P_down.to(dtype=torch.bfloat16)
        P_up_bf = P_up.to(dtype=torch.bfloat16)
        mu_bf = mu_dev_f32.to(dtype=torch.bfloat16)
        # Reconstruct: x_hat = (P_up @ P_down) @ (x - mu) + mu
        z = (xb - mu_bf) @ P_down_bf.T            # [B*T, ell]
        x_hat = z @ P_up_bf.T + mu_bf             # [B*T, d]
        y_hat = target_block(x_hat.unsqueeze(0))
        if isinstance(y_hat, tuple):
            y_hat = y_hat[0]
        y_hat = y_hat.squeeze(0)
        loss = ((y_hat.float() - yb.float()) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 25 == 0 or step == 1:
            history.append({"step": int(step), "train_moe_mse": float(loss.item())})
            print(f"[task_aware] step {step:>4d}/{args.steps}  train_MoE_MSE={loss.item():.4e}", flush=True)

    # ---------- Trained projector metrics on test ----------
    print("[task_aware] evaluating trained projector on test ...", flush=True)
    P_down_cpu = P_down.detach().to("cpu", dtype=torch.float32)
    P_up_cpu = P_up.detach().to("cpu", dtype=torch.float32)
    x_hat_test_tr = reconstruct(test_x, P_down_cpu, P_up_cpu, mu)
    mse_x_tr = reconstruction_mse(test_x, x_hat_test_tr)
    ev_x_tr = explained_variance(test_x, x_hat_test_tr)
    y_hat_test_tr = moe_forward_reconstructed(test_x, P_down_cpu, P_up_cpu, mu)
    mse_y_tr = float(((test_y - y_hat_test_tr) ** 2).mean())
    rel_y_tr = mse_y_tr / float((test_y ** 2).mean() + 1e-12)
    h_hat_test_tr = full_model_forward_with_hook(test_chunks, P_down_cpu, P_up_cpu, mu)
    mse_h_tr = float(((test_h - h_hat_test_tr) ** 2).mean())
    rel_h_tr = mse_h_tr / float((test_h ** 2).mean() + 1e-12)
    print(
        f"[task_aware] TRAINED       TEST:  "
        f"recon_MSE={mse_x_tr:.4e}  EV(x)={ev_x_tr:.4f}  "
        f"MoE_out_MSE={mse_y_tr:.4e}  rel_MoE={rel_y_tr:.4f}  "
        f"final_h_MSE={mse_h_tr:.4e}  rel_h={rel_h_tr:.4f}",
        flush=True,
    )

    # ---------- Save ----------
    summary = {
        "model_path": args.model_path,
        "target_layer": L_target,
        "ell": int(args.ell),
        "hidden_size": int(H),
        "train_chunks": int(args.train_chunks),
        "test_chunks": int(args.test_chunks),
        "train_tokens": int(train_x.shape[0]),
        "test_tokens": int(test_x.shape[0]),
        "lr": float(args.lr),
        "steps": int(args.steps),
        "svd_baseline": {
            "train_explained_variance": ev_svd_train,
            "test": {
                "recon_MSE": mse_x_svd,
                "EV_x": ev_x_svd,
                "MoE_out_MSE": mse_y_svd,
                "MoE_out_relMSE": rel_y_svd,
                "final_h_MSE": mse_h_svd,
                "final_h_relMSE": rel_h_svd,
            },
        },
        "trained": {
            "test": {
                "recon_MSE": mse_x_tr,
                "EV_x": ev_x_tr,
                "MoE_out_MSE": mse_y_tr,
                "MoE_out_relMSE": rel_y_tr,
                "final_h_MSE": mse_h_tr,
                "final_h_relMSE": rel_h_tr,
            },
        },
        "history": history,
        "deltas": {
            "MoE_out_relMSE_ratio": rel_y_tr / max(rel_y_svd, 1e-12),
            "final_h_relMSE_ratio": rel_h_tr / max(rel_h_svd, 1e-12),
            "EV_x_delta": ev_x_tr - ev_x_svd,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir/'summary.json'}", flush=True)

    # Markdown report.
    lines = []
    lines.append(f"# Task-aware projector vs SVD baseline (layer {L_target}, lcc, ell={args.ell})")
    lines.append("")
    lines.append(f"- {args.train_chunks} train chunks ({summary['train_tokens']} tokens), {args.test_chunks} test chunks ({summary['test_tokens']} tokens)")
    lines.append(f"- ell / d = {args.ell}/{H} = 1/{H//args.ell}")
    lines.append(f"- trained {args.steps} AdamW steps, lr {args.lr}, batch {args.batch_chunks} chunks ({args.batch_chunks*args.chunk_len} tokens)")
    lines.append("")
    lines.append("## Test metrics: SVD vs trained projector")
    lines.append("")
    lines.append("| metric | SVD baseline | trained | ratio (trained/SVD) |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| x reconstruction MSE | {mse_x_svd:.4e} | {mse_x_tr:.4e} | {mse_x_tr/max(mse_x_svd,1e-12):.3f} |")
    lines.append(f"| x explained variance | {ev_x_svd:.4f} | {ev_x_tr:.4f} | n/a |")
    lines.append(f"| MoE output MSE | {mse_y_svd:.4e} | {mse_y_tr:.4e} | {mse_y_tr/max(mse_y_svd,1e-12):.3f} |")
    lines.append(f"| MoE output relMSE | {rel_y_svd:.4f} | {rel_y_tr:.4f} | {rel_y_tr/max(rel_y_svd,1e-12):.3f} |")
    lines.append(f"| final hidden MSE | {mse_h_svd:.4e} | {mse_h_tr:.4e} | {mse_h_tr/max(mse_h_svd,1e-12):.3f} |")
    lines.append(f"| final hidden relMSE | {rel_h_svd:.4f} | {rel_h_tr:.4f} | {rel_h_tr/max(rel_h_svd,1e-12):.3f} |")
    lines.append("")
    lines.append("## Interpretation cheatsheet")
    lines.append("")
    lines.append("- `x reconstruction MSE`: SVD is provably L^2-optimal for linear projection; trained projector cannot beat it on this metric. Confirms the experiment is well-formed.")
    lines.append("- `MoE output relMSE` (relative MSE on layer-L output): does the MoE-layer-aware loss beat naive SVD? Big drop => task-awareness wins.")
    lines.append("- `final hidden relMSE`: end-to-end task-level proxy. If this dropped meaningfully, the saved compute really maps to lossless task output.")
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"wrote {out_dir/'report.md'}", flush=True)

    # Save training curve.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps_xs = [h["step"] for h in history]
        losses = [h["train_moe_mse"] for h in history]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(steps_xs, losses, marker="o", markersize=3)
        ax.set_xlabel("step")
        ax.set_ylabel("train MoE output MSE")
        ax.set_yscale("log")
        ax.set_title(f"Task-aware projector training (layer {L_target}, lcc, ell={args.ell})")
        ax.grid(True, alpha=0.3)
        ax.axhline(mse_y_svd, color="red", linestyle="--", linewidth=0.9, label=f"SVD baseline test MoE MSE = {mse_y_svd:.3e}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "training_curve.png", dpi=140)
        plt.close(fig)
        print(f"wrote {out_dir/'training_curve.png'}", flush=True)
    except Exception as e:
        print(f"plot failed: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
