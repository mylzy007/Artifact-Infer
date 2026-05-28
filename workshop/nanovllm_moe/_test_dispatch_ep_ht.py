"""Smoke test for EP-HT dispatch metadata.

The key edge case is the highest local expert id. sgl_kernel.moe_align_block_size
uses an internal padding/filter slot, so DispatchEPHT must pass E_local + 1.
"""

import os

import torch
import torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12401")
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
torch.cuda.set_device(0)
torch.set_default_device("cuda")

from workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ht import DispatchEPHT


def main():
    torch.manual_seed(0)
    T, H, E, K, BLOCK_M = 8, 32, 8, 1, 16

    disp = DispatchEPHT(
        num_experts_global=E,
        top_k=K,
        block_size_m=BLOCK_M,
        norm_topk_prob=True,
        expert_placement="contiguous",
        layer_id=0,
    ).cuda()

    max_recv = T * K
    max_padded = max_recv + (E + 1) * (BLOCK_M - 1)
    max_blocks = (max_padded + BLOCK_M - 1) // BLOCK_M
    disp.sorted_token_ids_buf = torch.empty(max_padded, dtype=torch.int32)
    disp.expert_ids_buf = torch.empty(max_blocks, dtype=torch.int32)
    disp.num_tokens_post_padded = torch.zeros(1, dtype=torch.int32)
    disp.cumsum_buffer = torch.empty(E + 2, dtype=torch.int32)

    hidden = torch.randn((T, H), dtype=torch.bfloat16)
    router_logits = torch.full((T, E), -1000.0, dtype=torch.float32)
    router_logits[:, E - 1] = 1000.0

    tok_meta = disp(hidden, router_logits)
    n = int(tok_meta.num_tokens_post_padded.item())
    expert_ids = tok_meta.expert_ids[: n // BLOCK_M]
    assert (expert_ids == E - 1).any(), (
        f"highest local expert {E - 1} disappeared from aligned blocks: "
        f"{expert_ids.cpu().tolist()}"
    )
    print("OK: EP-HT dispatch preserves highest local expert id")


if __name__ == "__main__":
    main()
