"""Debug: verify routing-aware projector preserves W_gate @ x exactly.

Pulls one layer's W_gate + a small batch of x activations from Qwen3-30B-A3B,
builds the routing-aware projector in fp64, then checks at each precision:
  - P_up^T @ P_up = I (orthonormality)
  - ||W_gate - W_gate @ P_up @ P_up^T||_F (gate row-space preservation)
  - max | (W_gate @ x_hat) - (W_gate @ x) | over a sample of x
  - top-1 routing match rate teacher vs student
"""
from __future__ import annotations

import json
from pathlib import Path
import torch

LB_DIR = Path("/home/lzy/datasets/moe_benchmarks/longbench/extracted/data")


def load_text():
    pieces, total = [], 0
    with (LB_DIR / "lcc.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = rec.get("context") or ""
            if isinstance(t, str) and len(t) >= 200:
                pieces.append(t)
                total += len(t)
                if total >= 200_000:
                    break
    return "\n\n".join(pieces)


def build_routing_aware(Xc: torch.Tensor, W: torch.Tensor, ell: int) -> torch.Tensor:
    """fp64 throughout."""
    Q_g, _ = torch.linalg.qr(W.T, mode="reduced")          # [d, E]
    r_g = Q_g.shape[1]
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    V_full = Vh.T.contiguous()                              # [d, k]
    coef = Q_g.T @ V_full
    V_perp = V_full - Q_g @ coef
    Q_p, R_p = torch.linalg.qr(V_perp, mode="reduced")
    diag = torch.diagonal(R_p)
    order = torch.argsort(diag.abs(), descending=True)
    Q_p = Q_p[:, order]
    n_extra = min(ell - r_g, Q_p.shape[1])
    return torch.cat([Q_g, Q_p[:, :n_extra]], dim=1).contiguous()


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("loading model ...", flush=True)
    tok = AutoTokenizer.from_pretrained("/home/lzy/models/Qwen3-30B-A3B", trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    max_mem = {i: "22GiB" for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        "/home/lzy/models/Qwen3-30B-A3B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_mem,
        trust_remote_code=True,
    )
    model.eval()
    H = model.config.hidden_size
    L = 24
    block = model.model.layers[L].mlp
    embed_dev = next(model.model.embed_tokens.parameters()).device

    # Capture x at layer L on a small batch.
    text = load_text()
    ids = tok(text, return_tensors=None)["input_ids"]
    chunks = [ids[i : i + 512] for i in range(0, 16 * 512, 512)]
    ids_t = torch.tensor(chunks, dtype=torch.long, device=embed_dev)

    buf = []

    def pre(_m, inputs):
        buf.append(inputs[0].detach().to("cpu", dtype=torch.float32).reshape(-1, inputs[0].shape[-1]))

    h = block.register_forward_pre_hook(pre)
    with torch.no_grad():
        _ = model(input_ids=ids_t, use_cache=False)
    h.remove()
    X = torch.cat(buf, dim=0)
    print(f"got X: {tuple(X.shape)}", flush=True)

    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    W_gate = block.gate.weight.detach().to("cpu", dtype=torch.float32)
    E = W_gate.shape[0]
    print(f"W_gate shape: {tuple(W_gate.shape)}, num_experts={E}", flush=True)

    ell = 1024

    # Build projector in fp64.
    Xc64 = Xc.to(torch.float64)
    W64 = W_gate.to(torch.float64)
    P_up_64 = build_routing_aware(Xc64, W64, ell)
    print(f"P_up shape: {tuple(P_up_64.shape)}", flush=True)

    # --- Test 1: orthonormality ---
    PtP = P_up_64.T @ P_up_64
    err_ortho = float((PtP - torch.eye(ell, dtype=torch.float64)).abs().max())
    print(f"[fp64] orthonormality |P^T P - I|_max = {err_ortho:.3e}", flush=True)

    # --- Test 2: W_gate preservation ---
    Proj = P_up_64 @ P_up_64.T
    W_diff = W64 - W64 @ Proj
    err_gate = float(W_diff.abs().max())
    print(f"[fp64] |W_gate - W_gate @ P P^T|_max = {err_gate:.3e}", flush=True)

    # --- Test 3: routing on real x ---
    X_test = X[:512].to(torch.float64)
    mu64 = mu.to(torch.float64)
    Xc_test = X_test - mu64
    X_hat = Xc_test @ Proj + mu64
    logits_orig = X_test @ W64.T              # [N, E]
    logits_hat = X_hat @ W64.T
    diff_log = (logits_orig - logits_hat).abs()
    print(f"[fp64] |W x - W x_hat|_max = {float(diff_log.max()):.3e}", flush=True)
    top1_orig = logits_orig.argmax(dim=-1)
    top1_hat = logits_hat.argmax(dim=-1)
    agree = float((top1_orig == top1_hat).float().mean())
    print(f"[fp64] top-1 routing agree on test x = {agree:.4f}", flush=True)

    # --- Now downcast to bf16 and repeat ---
    P_up_bf = P_up_64.to(torch.bfloat16).to(torch.float32)  # round to bf16 then back
    mu_bf = mu64.to(torch.bfloat16).to(torch.float32)
    PtP_b = P_up_bf.T @ P_up_bf
    err_ortho_b = float((PtP_b - torch.eye(ell)).abs().max())
    print(f"[bf16-cast] orthonormality |P^T P - I|_max = {err_ortho_b:.3e}", flush=True)
    Proj_b = P_up_bf @ P_up_bf.T
    W_bf = W_gate
    err_gate_b = float((W_bf - W_bf @ Proj_b).abs().max())
    print(f"[bf16-cast] |W - W P P^T|_max = {err_gate_b:.3e}", flush=True)

    X_test32 = X[:512]
    Xc_test32 = X_test32 - mu_bf
    X_hat32 = Xc_test32 @ Proj_b + mu_bf
    logits_orig32 = X_test32 @ W_bf.T
    logits_hat32 = X_hat32 @ W_bf.T
    diff_log_b = (logits_orig32 - logits_hat32).abs()
    print(f"[bf16-cast] |W x - W x_hat|_max = {float(diff_log_b.max()):.3e}", flush=True)
    agree_b = float((logits_orig32.argmax(-1) == logits_hat32.argmax(-1)).float().mean())
    print(f"[bf16-cast] top-1 routing agree = {agree_b:.4f}", flush=True)

    # --- Mimic the actual hook arithmetic in bf16 throughout ---
    P_up_real = P_up_64.to(torch.bfloat16)
    mu_real = mu64.to(torch.bfloat16)
    X_real = X[:512].to(torch.bfloat16)
    W_real = W_gate.to(torch.bfloat16)
    z = (X_real - mu_real) @ P_up_real
    x_hat_real = z @ P_up_real.T + mu_real
    logits_orig_real = X_real.float() @ W_real.float().T
    logits_hat_real = x_hat_real.float() @ W_real.float().T
    diff_log_r = (logits_orig_real - logits_hat_real).abs()
    print(f"[bf16-runtime] |W x - W x_hat|_max = {float(diff_log_r.max()):.3e}", flush=True)
    agree_r = float((logits_orig_real.argmax(-1) == logits_hat_real.argmax(-1)).float().mean())
    print(f"[bf16-runtime] top-1 routing agree (in-place) = {agree_r:.4f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
