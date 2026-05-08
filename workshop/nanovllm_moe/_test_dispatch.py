"""Smoke test: Dispatch produces sane (topk_ids, sorted_token_ids, expert_ids).

Compares against a hand-rolled topk + sort reference."""

import os
import torch
import torch.distributed as dist
import torch.nn.functional as F

# Need a distributed group (single-rank) because ReplicatedLinear queries dist.get_rank/world_size.
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12399")
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
torch.cuda.set_device(0)
torch.set_default_device("cuda")

from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch import Dispatch


def main():
    torch.manual_seed(0)
    T, H, E, K, BLOCK_M = 64, 32, 8, 2, 16
    T_cap = 128

    disp = Dispatch(num_experts=E, top_k=K, block_size_m=BLOCK_M, norm_topk_prob=True).cuda()
    # Hand-build a "gate" linear for the test.
    from workshop.nanovllm_moe.artifacts.modeling.layers.linear import ReplicatedLinear
    gate = ReplicatedLinear(H, E, bias=False).cuda().bfloat16()
    torch.nn.init.normal_(gate.weight, std=0.02)

    # Manually attach the workspace buffers (in real life: via orchestrator)
    max_padded = T_cap * K + (E + 1) * (BLOCK_M - 1)
    max_blocks = (max_padded + BLOCK_M - 1) // BLOCK_M
    disp.sorted_token_ids_buf = torch.empty((max_padded,), dtype=torch.int32)
    disp.expert_ids_buf = torch.empty((max_blocks,), dtype=torch.int32)
    disp.num_tokens_post_padded = torch.zeros((1,), dtype=torch.int32)
    disp.cumsum_buffer = torch.empty((E + 2,), dtype=torch.int32)
    disp.topk_weights_buf = torch.empty((T_cap, K), dtype=torch.float32)
    disp.topk_ids_buf = torch.empty((T_cap, K), dtype=torch.int32)

    hidden = torch.randn((T, H), dtype=torch.bfloat16) * 0.1
    router_logits = gate(hidden)
    tm = disp(router_logits)

    # Reference
    logits_ref = router_logits.float()
    weights_ref, ids_ref = torch.topk(F.softmax(logits_ref, dim=-1), K, dim=-1)
    weights_ref = weights_ref / weights_ref.sum(dim=-1, keepdim=True)

    # ids must match
    assert torch.equal(tm.topk_ids.long(), ids_ref.long()), \
        f"topk_ids mismatch:\n got: {tm.topk_ids[:3]}\nref: {ids_ref[:3]}"
    assert torch.allclose(tm.topk_weights, weights_ref.float(), atol=1e-3), "topk_weights mismatch"

    # num_tokens_post_padded should be > 0 and divisible by BLOCK_M
    n = tm.num_tokens_post_padded.item()
    print(f"num_tokens_post_padded = {n}, T*K = {T*K}, padded extra = {n - T*K}")
    assert n >= T * K, f"padding too small: {n} < T*K={T*K}"
    assert n % BLOCK_M == 0, f"padded length not divisible by BLOCK_M: {n}"

    sorted_valid = tm.sorted_token_ids[:n]
    valid_mask = sorted_valid < T * K
    n_valid = int(valid_mask.sum().item())
    print(f"n_valid (< T*K={T*K}): {n_valid}; n_padding: {n - n_valid}")
    assert n_valid == T * K, f"expected {T*K} valid sorted ids, got {n_valid}"

    # expert_ids: each block's expert should match the dominant expert of its
    # token-ids slice (skipping fully-padding blocks marked -1).
    expert_ids = tm.expert_ids[: n // BLOCK_M]
    flat_ids = tm.topk_ids.view(-1)
    print(f"expert_ids[:n_blocks={n // BLOCK_M}]: {expert_ids.tolist()}")
    n_real_blocks = 0
    for b in range(n // BLOCK_M):
        e_block = expert_ids[b].item()
        block_sids = sorted_valid[b * BLOCK_M : (b + 1) * BLOCK_M]
        block_valid = block_sids[block_sids < T * K]
        if e_block == -1:
            assert block_valid.numel() == 0, f"block {b} marked -1 but has {block_valid.numel()} valid entries"
            continue
        block_experts = flat_ids[block_valid.long()]
        assert (block_experts == e_block).all(), \
            f"block {b}: expected all expert {e_block}, got {block_experts.unique()}"
        n_real_blocks += 1
    print(f"validated {n_real_blocks} real blocks (out of {n // BLOCK_M})")
    print("OK: Dispatch produces consistent topk + sorted index tensors")


if __name__ == "__main__":
    main()
