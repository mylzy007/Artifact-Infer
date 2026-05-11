"""Smoke test: torch_fused_moe vs hand-rolled HF-style reference."""

import torch
import torch.nn.functional as F

from workshop.nanovllm_moe.artifacts.moe_backend.torch_fused_moe import torch_fused_moe


def hf_reference(hidden_states, w1, w2, topk_weights, topk_ids):
    """Naive HF-style MoE: per-token loop, no padding tricks."""
    T, H = hidden_states.shape
    E, two_N, _ = w1.shape
    N = two_N // 2
    K = topk_ids.shape[1]
    out = torch.zeros((T, H), dtype=hidden_states.dtype, device=hidden_states.device)
    for t in range(T):
        for k in range(K):
            e = topk_ids[t, k].item()
            x = hidden_states[t : t + 1]
            gate_up = x @ w1[e].T
            gate, up = gate_up.split(N, dim=-1)
            inter = F.silu(gate) * up
            down = inter @ w2[e].T
            out[t] += down.squeeze(0) * topk_weights[t, k].to(down.dtype)
    return out


def main():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    T, H, N, E, K = 16, 64, 32, 8, 2

    hidden = torch.randn((T, H), dtype=dtype, device=device) * 0.1
    w1 = torch.randn((E, 2 * N, H), dtype=dtype, device=device) * 0.1
    w2 = torch.randn((E, H, N), dtype=dtype, device=device) * 0.1

    # Random routing
    logits = torch.randn((T, E), device=device, dtype=torch.float32)
    weights, ids = torch.topk(F.softmax(logits, dim=-1), K, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # norm_topk_prob=True
    weights = weights.float()
    ids = ids.int()

    # Allocate workspaces (only cache3 is used by torch backend, but pass all 3)
    T_cap = 32
    cache1 = torch.empty((T_cap, K, 2 * N), dtype=dtype, device=device)
    cache2 = torch.empty((T_cap, K, N), dtype=dtype, device=device)
    cache3 = torch.empty((T_cap, K, H), dtype=dtype, device=device)

    # Unused metadata (signature parity with triton)
    sorted_ids = torch.empty((T * K + E * 63,), dtype=torch.int32, device=device)
    expert_ids = torch.empty((1024,), dtype=torch.int32, device=device)
    npp = torch.empty((1,), dtype=torch.int32, device=device)

    out_TK_H = torch_fused_moe(
        hidden_states=hidden,
        w1=w1, w2=w2,
        topk_weights=weights, topk_ids=ids,
        sorted_token_ids=sorted_ids, expert_ids=expert_ids,
        num_tokens_post_padded=npp,
        intermediate_cache1=cache1, intermediate_cache2=cache2, intermediate_cache3=cache3,
        block_size_m=64,
    )
    out = out_TK_H.sum(dim=1)  # what Combine will do

    ref = hf_reference(hidden, w1, w2, weights, ids)

    abs_err = (out - ref).abs()
    rel_err = abs_err / (ref.abs().clamp_min(1e-3))
    print(f"out shape: {out.shape}, ref shape: {ref.shape}")
    print(f"max abs err: {abs_err.max().item():.5e}")
    print(f"max rel err: {rel_err.max().item():.5e}")
    print(f"mean abs err: {abs_err.mean().item():.5e}")
    assert torch.allclose(out, ref, atol=5e-3, rtol=5e-2), "torch_fused_moe diverges from HF reference"
    print("OK: torch_fused_moe matches HF reference within bf16 tolerance")


if __name__ == "__main__":
    main()
