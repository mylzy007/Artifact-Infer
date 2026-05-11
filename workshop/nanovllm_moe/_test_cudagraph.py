"""CUDA graph capture/replay validation for FusedMoE (triton backend).

Captures a decode-style call (small T) and verifies that the replayed output
matches eager output for several different input batches.

This exercises the persistent-buffer + device-side num_tokens_post_padded
contract that the design relies on (docs/moe/design.md §8).
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

from src.core.orchestrator import RegistryOrchestrator
from src.core.service import BaseService

from workshop.nanovllm_moe.artifacts.moe_backend import MoeBackend
from workshop.nanovllm_moe.artifacts.modeling.layers.moe import (
    Combine, Dispatch, Experts, FusedMoE,
)


class _Cfg:
    def __init__(self, dtype):
        self.max_num_batched_tokens = 64
        self.moe_block_size_m = 16
        self.moe_impl = "triton"
        self.tensor_parallel_size = 1
        class _HF:
            torch_dtype = dtype
        self.hf_config = _HF()


class _Root(BaseService):
    @property
    def name(self): return "Root"


def main():
    torch.manual_seed(0)
    H, N, E, K = 64, 32, 8, 2
    T = 4   # decode-style small batch
    dtype = torch.bfloat16

    cfg = _Cfg(dtype)
    torch.set_default_dtype(dtype)
    fused = FusedMoE(
        hidden_size=H, moe_intermediate_size=N, num_experts=E,
        top_k=K, block_size_m=cfg.moe_block_size_m, norm_topk_prob=True,
    )
    torch.nn.init.normal_(fused.gate.weight, std=0.05)
    torch.nn.init.normal_(fused.experts.w1, std=0.05)
    torch.nn.init.normal_(fused.experts.w2, std=0.05)

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

    # Persistent input buffer (typical CUDA-graph pattern).
    input_buf = torch.zeros((T, H), dtype=dtype, device="cuda")
    output_buf = torch.zeros((T, H), dtype=dtype, device="cuda")

    # Warm-up + capture
    moe_backend.prepare_metadata_for_moe(T)
    input_buf.copy_((torch.randn_like(input_buf) * 0.05))
    with torch.inference_mode():
        # Warm up the kernel(s)
        out = fused(input_buf)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        # Capture (re-run prepare just before so npp is zeroed inside the captured region too,
        # though actually we want it zeroed OUTSIDE; the design says prepare runs outside capture)
        moe_backend.prepare_metadata_for_moe(T)  # zeroes npp before capture
        with torch.cuda.graph(graph):
            output_buf.copy_(fused(input_buf))
        torch.cuda.synchronize()
        print("Captured CUDA graph for FusedMoE")

        # Now replay several times with new inputs and compare to eager
        max_err = 0.0
        for trial in range(5):
            new_input = (torch.randn_like(input_buf) * 0.05)
            input_buf.copy_(new_input)
            moe_backend.prepare_metadata_for_moe(T)  # zero npp before replay
            graph.replay()
            torch.cuda.synchronize()
            replayed = output_buf.clone()

            # Compare to eager re-run
            moe_backend.prepare_metadata_for_moe(T)
            eager = fused(new_input)
            err = (replayed - eager).abs().max().item()
            max_err = max(max_err, err)
            print(f"  trial {trial}: max abs err vs eager = {err:.4e}")

    assert max_err < 5e-3, f"replay diverged from eager: {max_err:.3e}"
    print(f"\nOK: CUDA-graph replay matches eager across 5 different inputs (max abs err {max_err:.3e})")


if __name__ == "__main__":
    main()
