"""Unit test: ExpertsEPLL._w1_loader / _w2_loader correctly slice global expert ids
into per-rank E_local slots.

Strategy:
  1. Build a "fake checkpoint": E_global stacked tensors (random per expert).
  2. Load them into a single-rank Experts (E_global slots) — this is the oracle.
  3. Simulate two ranks (no actual dist; we set rank/world_size by patching attrs).
     For each rank, build an ExpertsEPLL with E_local = E_global / 2 and call its
     loaders on EVERY checkpoint expert (the loader filters by rank itself).
  4. Concatenate the two rank slices along expert axis and assert equality with
     the single-rank oracle.
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

from workshop.nanovllm_moe.artifacts.modeling.layers.moe.experts import Experts
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.experts_ep_ll import ExpertsEPLL


def main():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    E_global = 8
    H = 32
    N = 16

    # ---- Fake "checkpoint": one tensor per expert per shard ----
    fake_ckpt = {}  # (expert_id, shard) -> tensor
    for e in range(E_global):
        fake_ckpt[(e, "gate")] = (torch.randn((N, H), dtype=torch.float32) * 0.1).to(dtype)
        fake_ckpt[(e, "up")]   = (torch.randn((N, H), dtype=torch.float32) * 0.1).to(dtype)
        fake_ckpt[(e, "w2")]   = (torch.randn((H, N), dtype=torch.float32) * 0.1).to(dtype)

    # ---- Oracle: single-rank Experts holds all E_global ----
    oracle = Experts(num_experts=E_global, hidden_size=H, moe_intermediate_size=N)
    for e in range(E_global):
        oracle.w1.weight_loader(oracle.w1, fake_ckpt[(e, "gate")], e, "gate")
        oracle.w1.weight_loader(oracle.w1, fake_ckpt[(e, "up")],   e, "up")
        oracle.w2.weight_loader(oracle.w2, fake_ckpt[(e, "w2")],   e, None)

    # ---- EP-LL: two "ranks", manually created by patching the module attrs ----
    ranks = []
    for rank in range(2):
        # Construct then patch (we're in a single-process test; can't actually init two groups).
        m = ExpertsEPLL(num_experts_global=E_global, hidden_size=H, moe_intermediate_size=N)
        m.world_size = 2
        m.rank = rank
        m.E_local = E_global // 2
        m.expert_id_lo = rank * m.E_local
        m.expert_id_hi = (rank + 1) * m.E_local
        m.w1 = torch.nn.Parameter(torch.empty((m.E_local, 2 * N, H), dtype=dtype))
        m.w2 = torch.nn.Parameter(torch.empty((m.E_local, H, N), dtype=dtype))
        m.w1.weight_loader = m._w1_loader
        m.w2.weight_loader = m._w2_loader

        # Simulate the loader iterating over EVERY checkpoint expert; the EPLL
        # weight_loader filters internally.
        for e in range(E_global):
            m.w1.weight_loader(m.w1, fake_ckpt[(e, "gate")], e, "gate")
            m.w1.weight_loader(m.w1, fake_ckpt[(e, "up")],   e, "up")
            m.w2.weight_loader(m.w2, fake_ckpt[(e, "w2")],   e, None)
        ranks.append(m)

    # ---- Concatenate and compare to oracle ----
    w1_concat = torch.cat([m.w1.data for m in ranks], dim=0)  # [E_global, 2N, H]
    w2_concat = torch.cat([m.w2.data for m in ranks], dim=0)  # [E_global, H, N]

    assert torch.equal(w1_concat, oracle.w1.data), "w1 EP-sliced concat doesn't match oracle"
    assert torch.equal(w2_concat, oracle.w2.data), "w2 EP-sliced concat doesn't match oracle"

    # ---- Out-of-range expert_ids must be silently dropped (no exception). ----
    m = ranks[0]
    before = m.w1.data.clone()
    m.w1.weight_loader(m.w1, fake_ckpt[(7, "gate")], 7, "gate")  # expert 7 is on rank 1
    assert torch.equal(m.w1.data, before), "out-of-range expert_id should not modify rank-0 weights"

    print(f"OK: ExpertsEPLL loaders correctly partition E_global={E_global} into "
          f"{2} ranks of E_local={E_global // 2} each (out-of-range silently dropped)")


if __name__ == "__main__":
    main()
