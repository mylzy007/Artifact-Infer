"""Parity test: EP-LL pipeline (N=1 group) vs single-rank torch_fused_moe.

Single process, gloo group of size 1. Verifies:
  1. DispatchEPLL.forward produces a hidden_recv where every token-expert
     assignment lands in the right bucket.
  2. torch_masked_grouped_gemm on that hidden_recv produces the per-(t, k)
     expert outputs (correct math + correct masking).
  3. CombineEPLL scatter-reduces them back into [T, H] matching the existing
     single-rank reference.
"""
import os
import torch
import torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12399")
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
torch.cuda.set_device(0)
torch.set_default_device("cuda")

from workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ll import DispatchEPLL
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.combine_ep_ll import CombineEPLL
from workshop.nanovllm_moe.artifacts.moe_backend.torch_masked_grouped_gemm import (
    torch_masked_grouped_gemm,
)
from workshop.nanovllm_moe.artifacts.moe_backend.torch_fused_moe import torch_fused_moe


def reference_final(hidden, w1, w2, topk_ids, topk_weights, T_cap):
    """Single-rank reference: torch_fused_moe -> sum across K -> [T, H]."""
    T, H = hidden.shape
    E, two_N, _ = w1.shape
    N = two_N // 2
    K = topk_ids.shape[1]
    cache1 = torch.empty((T_cap, K, 2 * N), dtype=hidden.dtype, device=hidden.device)
    cache2 = torch.empty((T_cap, K, N), dtype=hidden.dtype, device=hidden.device)
    cache3 = torch.empty((T_cap, K, H), dtype=hidden.dtype, device=hidden.device)
    out_TKH = torch_fused_moe(
        hidden_states=hidden, w1=w1, w2=w2,
        topk_weights=topk_weights, topk_ids=topk_ids,
        sorted_token_ids=None, expert_ids=None, num_tokens_post_padded=None,
        intermediate_cache1=cache1, intermediate_cache2=cache2, intermediate_cache3=cache3,
        block_size_m=64,
    )
    return out_TKH.sum(dim=1)  # [T, H]


def main():
    torch.manual_seed(0)
    dtype = torch.bfloat16

    # Small case so the host loops in DispatchEPLL aren't slow.
    T = 8
    H = 64
    N_intermediate = 32
    E = 4
    K = 2
    M_max = T * K  # worst case for N=1: all tokens hit the same expert

    hidden = (torch.randn((T, H), dtype=torch.float32) * 0.1).to(dtype)
    w1 = (torch.randn((E, 2 * N_intermediate, H), dtype=torch.float32) * 0.05).to(dtype)
    w2 = (torch.randn((E, H, N_intermediate), dtype=torch.float32) * 0.05).to(dtype)

    # Random router_logits → drives identical routing on both paths.
    router_logits = (torch.randn((T, E), dtype=torch.float32) * 0.5).to(dtype)

    # Pre-compute topk on the host so we can feed the SAME ids to the reference.
    logits_fp32 = router_logits.float()
    topk_vals, topk_ids = torch.topk(logits_fp32, K, dim=-1)
    topk_weights = torch.softmax(topk_vals, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    topk_ids_int32 = topk_ids.to(torch.int32)
    topk_weights_fp32 = topk_weights

    # ---- Reference: single-rank torch_fused_moe + K-sum ----
    ref_final = reference_final(hidden, w1, w2, topk_ids_int32, topk_weights_fp32, T_cap=T)

    # ---- EP-LL pipeline (N=1) ----
    dispatch = DispatchEPLL(num_experts_global=E, top_k=K, m_max=M_max, norm_topk_prob=True)
    combine = CombineEPLL(hidden_size=H, top_k=K, num_experts_local=E, m_max=M_max)
    tok_meta = dispatch(hidden, router_logits)

    # Sanity 1: dispatch routing matches the reference topk_ids.
    diff_ids = (tok_meta.topk_ids.cpu() != topk_ids_int32.cpu()).sum().item()
    assert diff_ids == 0, f"DispatchEPLL.topk_ids diverges from reference (count={diff_ids})"

    # Sanity 2: hidden_recv has the right shape and contains all the expected tokens.
    assert tok_meta.hidden_recv.shape == (E, M_max, H), tok_meta.hidden_recv.shape
    # Verify every (t, k) appears exactly once in the dispatched buffer (with N=1, in
    # bucket [target_local_expert, slot_within_bucket]).
    seen = set()
    oi = tok_meta.original_indices.cpu()  # [N=1, E, M_max, 2]
    for e in range(E):
        for m in range(M_max):
            t_id = int(oi[0, e, m, 0])
            k_id = int(oi[0, e, m, 1])
            if t_id < 0:
                continue  # unused slot
            assert (t_id, k_id) not in seen, f"(t={t_id}, k={k_id}) routed twice"
            seen.add((t_id, k_id))
            # The slot must hold the right hidden_state.
            row = tok_meta.hidden_recv[e, m]
            assert torch.allclose(row.float(), hidden[t_id].float(), atol=1e-3), (
                f"slot (e={e}, m={m}) doesn't hold hidden[t={t_id}]"
            )
    assert len(seen) == T * K, f"Expected {T*K} (t,k) pairs in send_buf, got {len(seen)}"

    # ---- Inner kernel ----
    expert_out = torch_masked_grouped_gemm(
        tok_meta.hidden_recv, w1, w2, tok_meta.masked_m,
    )

    # ---- Combine ----
    ep_final = combine(expert_out, tok_meta)

    # ---- Compare ----
    diff = (ep_final.float() - ref_final.float()).abs()
    print(f"max abs diff: {diff.max().item():.4e}")
    print(f"mean abs diff: {diff.mean().item():.4e}")
    print(f"ref magnitude (mean abs): {ref_final.abs().mean().item():.4e}")

    assert diff.max().item() < 1e-3, (
        f"EP-LL N=1 diverges from single-rank ref by {diff.max().item():.4e} "
        f"(should be ≤ 1e-3 for bf16)"
    )

    print("OK: EP-LL pipeline (N=1) matches single-rank torch_fused_moe within bf16 noise")


if __name__ == "__main__":
    main()
