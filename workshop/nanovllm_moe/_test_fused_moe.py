"""Smoke test: full FusedMoE (Dispatch + Experts + Combine) wired through
the MoeBackend artifact via the RegistryOrchestrator. Compares against a
hand-rolled HF-style oracle."""

import os
import torch
import torch.distributed as dist
import torch.nn.functional as F

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12399")
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
torch.cuda.set_device(0)
torch.set_default_device("cuda")

from src.core.orchestrator import RegistryOrchestrator
from src.core.service import BaseService

from workshop.nanovllm_moe.artifacts.moe_backend import MoeBackend
from workshop.nanovllm_moe.artifacts.modeling.layers.moe import (
    Combine, Dispatch, Experts, FusedMoE,
)


# Minimal Config stand-in
class _Cfg:
    def __init__(self, T_cap, dtype):
        self.max_num_batched_tokens = T_cap
        self.moe_block_size_m = 16
        self.moe_impl = "torch"
        self.tensor_parallel_size = 1

        class _HFCfg:
            torch_dtype = dtype
        self.hf_config = _HFCfg()


class _Root(BaseService):
    @property
    def name(self): return "Root"


def hf_reference(hidden_states, w1, w2, gate_weight, K, norm_topk_prob=True):
    T, H = hidden_states.shape
    E, two_N, _ = w1.shape
    N = two_N // 2
    logits = F.linear(hidden_states.float(), gate_weight.float())  # fp32 like topk_softmax
    weights, ids = torch.topk(F.softmax(logits, dim=-1), K, dim=-1)
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    out = torch.zeros((T, H), dtype=hidden_states.dtype, device=hidden_states.device)
    for t in range(T):
        for k in range(K):
            e = ids[t, k].item()
            x = hidden_states[t : t + 1]
            gate_up = x @ w1[e].T
            gate, up = gate_up.split(N, dim=-1)
            inter = F.silu(gate) * up
            down = inter @ w2[e].T
            out[t] += down.squeeze(0) * weights[t, k].to(down.dtype)
    return out


def main():
    torch.manual_seed(0)
    H, N, E, K, BLOCK_M = 64, 32, 8, 2, 16
    T = 24
    T_cap = 64
    dtype = torch.bfloat16

    cfg = _Cfg(T_cap, dtype)
    fused = FusedMoE(
        hidden_size=H, moe_intermediate_size=N, num_experts=E,
        top_k=K, block_size_m=BLOCK_M, norm_topk_prob=True,
    ).cuda().to(dtype)

    # Initialize weights
    torch.nn.init.normal_(fused.gate.weight, std=0.02)
    torch.nn.init.normal_(fused.experts.w1, std=0.02)
    torch.nn.init.normal_(fused.experts.w2, std=0.02)

    # Wire MoeBackend buffers + run_experts through the orchestrator
    root = _Root()
    orch = RegistryOrchestrator()
    moe_backend = orch.add(MoeBackend(
        config=cfg, num_experts=E, top_k=K, hidden_size=H, moe_intermediate_size=N,
    ))

    for module in fused.modules():
        if isinstance(module, Dispatch):
            for name in (
                "sorted_token_ids_buf", "expert_ids_buf", "num_tokens_post_padded",
                "cumsum_buffer", "topk_weights_buf", "topk_ids_buf",
            ):
                orch.register(moe_backend, name, module)
        if isinstance(module, Experts):
            for name in ("intermediate_cache1", "intermediate_cache2", "intermediate_cache3", "run_experts"):
                orch.register(moe_backend, name, module)
        if isinstance(module, Combine):
            orch.register(moe_backend, "intermediate_cache3", module)
    orch.register(moe_backend, "prepare_metadata_for_moe", root)
    orch.finalize()

    # Forward
    hidden = torch.randn((T, H), dtype=dtype) * 0.1
    out = fused(hidden)

    ref = hf_reference(
        hidden, fused.experts.w1, fused.experts.w2, fused.gate.weight,
        K=K, norm_topk_prob=True,
    )

    abs_err = (out - ref).abs()
    rel_err = abs_err / (ref.abs().clamp_min(1e-3))
    print(f"out shape: {out.shape}, ref shape: {ref.shape}")
    print(f"max abs err: {abs_err.max().item():.5e}")
    print(f"max rel err: {rel_err.max().item():.5e}")
    print(f"mean abs err: {abs_err.mean().item():.5e}")
    # bf16 tolerance: max rel ~1e-2 expected, weights diverge by ~1e-3 from fp32 logit path.
    assert torch.allclose(out, ref, atol=5e-3, rtol=5e-2), \
        "FusedMoE diverges from HF-style reference (beyond bf16 tolerance)"
    print("OK: FusedMoE end-to-end matches HF-style reference within bf16 tolerance")


if __name__ == "__main__":
    main()
