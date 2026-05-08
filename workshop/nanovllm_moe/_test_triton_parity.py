"""Parity test: triton_fused_moe vs torch_fused_moe.

Both backends share the exact same buffer contract — we feed both with the
same sorted_token_ids/expert_ids produced by Dispatch and check that they
produce the same output."""

import os
import torch
import torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12399")
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
torch.cuda.set_device(0)
torch.set_default_device("cuda")

from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size
from sgl_kernel import topk_softmax as sgl_topk_softmax

from workshop.nanovllm_moe.artifacts.moe_backend.torch_fused_moe import torch_fused_moe
from workshop.nanovllm_moe.artifacts.moe_backend.triton_fused_moe import triton_fused_moe


def main():
    torch.manual_seed(0)
    H, N, E, K_top, BLOCK_M = 64, 32, 8, 4, 16
    T, T_cap = 24, 64
    dtype = torch.bfloat16

    hidden = (torch.randn((T, H), dtype=dtype) * 0.05).cuda()
    w1 = (torch.randn((E, 2 * N, H), dtype=dtype) * 0.05).cuda()
    w2 = (torch.randn((E, H, N), dtype=dtype) * 0.05).cuda()

    # --- Dispatch (real path with sgl_kernel) ---
    gate_w = (torch.randn((E, H), dtype=dtype) * 0.05).cuda()
    router_logits = torch.nn.functional.linear(hidden, gate_w)
    topk_w = torch.empty((T, K_top), dtype=torch.float32, device="cuda")
    topk_i = torch.empty((T, K_top), dtype=torch.int32, device="cuda")
    sgl_topk_softmax(topk_w, topk_i, router_logits, True, 0.0, None)
    max_padded = T * K_top + (E + 1) * (BLOCK_M - 1)
    max_blocks = (max_padded + BLOCK_M - 1) // BLOCK_M
    sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device="cuda")
    expert_ids = torch.empty((max_blocks,), dtype=torch.int32, device="cuda")
    npp = torch.zeros((1,), dtype=torch.int32, device="cuda")
    cumsum = torch.empty((E + 2,), dtype=torch.int32, device="cuda")
    sgl_moe_align_block_size(topk_i, E + 1, BLOCK_M, sorted_ids, expert_ids, npp, cumsum, True)
    n = npp.item()
    print(f"Dispatch produced n={n}, n_blocks={n // BLOCK_M}")

    # --- Torch ---
    cache1_t = torch.empty((T_cap, K_top, 2 * N), dtype=dtype, device="cuda")
    cache2_t = torch.empty((T_cap, K_top, N), dtype=dtype, device="cuda")
    cache3_t = torch.empty((T_cap, K_top, H), dtype=dtype, device="cuda")
    out_torch = torch_fused_moe(
        hidden_states=hidden, w1=w1, w2=w2, topk_weights=topk_w, topk_ids=topk_i,
        sorted_token_ids=sorted_ids[:n], expert_ids=expert_ids[:n // BLOCK_M],
        num_tokens_post_padded=npp,
        intermediate_cache1=cache1_t, intermediate_cache2=cache2_t, intermediate_cache3=cache3_t,
        block_size_m=BLOCK_M,
    )
    out_torch_TH = out_torch.sum(dim=1)  # [T, H]

    # --- Triton ---
    cache1 = torch.empty((T_cap, K_top, 2 * N), dtype=dtype, device="cuda")
    cache2 = torch.empty((T_cap, K_top, N), dtype=dtype, device="cuda")
    cache3 = torch.empty((T_cap, K_top, H), dtype=dtype, device="cuda")
    out_triton = triton_fused_moe(
        hidden_states=hidden, w1=w1, w2=w2, topk_weights=topk_w, topk_ids=topk_i,
        sorted_token_ids=sorted_ids[:n], expert_ids=expert_ids[:n // BLOCK_M],
        num_tokens_post_padded=npp,
        intermediate_cache1=cache1, intermediate_cache2=cache2, intermediate_cache3=cache3,
        block_size_m=BLOCK_M,
    )
    out_triton_TH = out_triton.sum(dim=1)  # [T, H]

    abs_err = (out_torch_TH - out_triton_TH).abs()
    rel_err = abs_err / out_torch_TH.abs().clamp_min(1e-3)
    print(f"\nout shape: {out_triton_TH.shape}")
    print(f"max abs err:  {abs_err.max().item():.4e}")
    print(f"max rel err:  {rel_err.max().item():.4e}")
    print(f"mean abs err: {abs_err.mean().item():.4e}")
    assert abs_err.max().item() < 5e-3, "triton diverges from torch"
    print("OK: triton_fused_moe matches torch_fused_moe within bf16 tolerance")


if __name__ == "__main__":
    main()
