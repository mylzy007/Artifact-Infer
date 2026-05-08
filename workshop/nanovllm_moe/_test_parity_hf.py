"""Bit-level parity test: our FusedMoE for layer 0 of Qwen3-30B-A3B vs the
HuggingFace `Qwen3MoeSparseMoeBlock` for the same layer."""

import os
import torch
import torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12399")
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
torch.cuda.set_device(0)
torch.set_default_device("cuda")

from transformers import AutoConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from src.core.orchestrator import RegistryOrchestrator
from src.core.service import BaseService
from workshop.nanovllm_moe.artifacts.moe_backend import MoeBackend
from workshop.nanovllm_moe.artifacts.modeling.layers.moe import (
    Combine, Dispatch, Experts, FusedMoE,
)


CKPT = "/home/yyx/models/Qwen3-30B-A3B"


class _Cfg:
    def __init__(self, hf_cfg):
        self.hf_config = hf_cfg
        self.max_num_batched_tokens = 256
        self.moe_block_size_m = 64
        self.moe_impl = "torch"
        self.tensor_parallel_size = 1


class _Root(BaseService):
    @property
    def name(self): return "Root"


def main():
    torch.manual_seed(0)
    hf_cfg = AutoConfig.from_pretrained(CKPT)
    torch.set_default_dtype(hf_cfg.torch_dtype)

    H, N, E, K = hf_cfg.hidden_size, hf_cfg.moe_intermediate_size, hf_cfg.num_experts, hf_cfg.num_experts_per_tok
    T = 8

    # Build *only* layer 0 of our FusedMoE (don't load whole model)
    cfg = _Cfg(hf_cfg)
    fused = FusedMoE(
        hidden_size=H, moe_intermediate_size=N, num_experts=E,
        top_k=K, block_size_m=cfg.moe_block_size_m,
        norm_topk_prob=hf_cfg.norm_topk_prob,
    )
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

    # Build HF reference (single SparseMoeBlock from the same config)
    hf_block = Qwen3MoeSparseMoeBlock(hf_cfg).to(hf_cfg.torch_dtype).cuda()

    # Load only layer 0 weights into both modules.
    print("Loading layer 0 weights into both modules...")
    _load_layer0(fused, hf_block)
    print("OK")

    # Forward
    moe_backend.prepare_metadata_for_moe(T)
    hidden = (torch.randn((1, T, H), dtype=hf_cfg.torch_dtype) * 0.05).cuda()

    with torch.inference_mode():
        ours = fused(hidden)
        # HF returns (final_hidden, router_logits)
        hf_out_tuple = hf_block(hidden)
        hf_out = hf_out_tuple[0] if isinstance(hf_out_tuple, tuple) else hf_out_tuple

    print(f"\nours shape: {ours.shape}, hf shape: {hf_out.shape}")
    abs_err = (ours - hf_out).abs()
    rel_err = abs_err / hf_out.abs().clamp_min(1e-3)
    print(f"max abs err:  {abs_err.max().item():.4e}")
    print(f"max rel err:  {rel_err.max().item():.4e}")
    print(f"mean abs err: {abs_err.mean().item():.4e}")
    # bf16 with K=8 expert summation: tolerance ~1e-2 abs
    assert abs_err.max().item() < 2e-2, f"divergence vs HF too large: {abs_err.max().item():.3e}"
    print("\nOK: FusedMoE matches HuggingFace Qwen3MoeSparseMoeBlock within bf16 tolerance")


def _load_layer0(fused: FusedMoE, hf_block: Qwen3MoeSparseMoeBlock):
    """Read layer 0 weights from /home/yyx/models/Qwen3-30B-A3B and stuff them
    into both `fused` (our stacked layout) and `hf_block` (HF's per-expert
    submodules). This way both see the exact same weights."""
    from glob import glob
    from safetensors import safe_open

    L = 0
    needed = {
        f"model.layers.{L}.mlp.gate.weight",
    }
    for e in range(fused.num_experts):
        needed |= {
            f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight",
            f"model.layers.{L}.mlp.experts.{e}.up_proj.weight",
            f"model.layers.{L}.mlp.experts.{e}.down_proj.weight",
        }

    found = {}
    for file in sorted(glob(os.path.join(CKPT, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for key in f.keys():
                if key in needed:
                    found[key] = f.get_tensor(key)

    assert len(found) == len(needed), f"missing keys: {needed - found.keys()}"

    # Gate
    fused.gate.weight.data.copy_(found[f"model.layers.{L}.mlp.gate.weight"])
    hf_block.gate.weight.data.copy_(found[f"model.layers.{L}.mlp.gate.weight"])

    # Both HF (newer Qwen3MoeExperts) and us use stacked [E, 2N, H] / [E, H, N].
    N = fused.moe_intermediate_size
    for e in range(fused.num_experts):
        gp = found[f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight"]
        up = found[f"model.layers.{L}.mlp.experts.{e}.up_proj.weight"]
        dp = found[f"model.layers.{L}.mlp.experts.{e}.down_proj.weight"]

        # Ours
        fused.experts.w1.data[e, :N].copy_(gp)
        fused.experts.w1.data[e, N:].copy_(up)
        fused.experts.w2.data[e].copy_(dp)

        # HF: experts.gate_up_proj[e] is [2N, H] but stored with gate first then up
        hf_block.experts.gate_up_proj.data[e, :N].copy_(gp)
        hf_block.experts.gate_up_proj.data[e, N:].copy_(up)
        hf_block.experts.down_proj.data[e].copy_(dp)


if __name__ == "__main__":
    main()
