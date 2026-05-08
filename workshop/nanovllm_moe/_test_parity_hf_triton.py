"""Parity test: triton FusedMoE vs HF Qwen3MoeSparseMoeBlock on real layer 0
of Qwen3-30B-A3B. This is the strongest correctness proof of the Triton path."""

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
    def __init__(self, hf_cfg, impl):
        self.hf_config = hf_cfg
        self.max_num_batched_tokens = 256
        self.moe_block_size_m = 64
        self.moe_impl = impl
        self.tensor_parallel_size = 1


class _Root(BaseService):
    @property
    def name(self): return "Root"


def _build(impl):
    hf_cfg = AutoConfig.from_pretrained(CKPT)
    torch.set_default_dtype(hf_cfg.torch_dtype)
    H, N, E, K = hf_cfg.hidden_size, hf_cfg.moe_intermediate_size, hf_cfg.num_experts, hf_cfg.num_experts_per_tok

    fused = FusedMoE(
        hidden_size=H, moe_intermediate_size=N, num_experts=E,
        top_k=K, block_size_m=64, norm_topk_prob=hf_cfg.norm_topk_prob,
    )
    cfg = _Cfg(hf_cfg, impl)
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
    return fused, moe_backend, hf_cfg


def main():
    torch.manual_seed(42)
    fused_torch, mb_torch, hf_cfg = _build("torch")
    fused_triton, mb_triton, _ = _build("triton")
    hf_block = Qwen3MoeSparseMoeBlock(hf_cfg).to(hf_cfg.torch_dtype).cuda()

    # Load layer 0 weights into all three.
    print("Loading layer 0 weights into torch / triton / hf modules...")
    _load_layer0(fused_torch, fused_triton, hf_block, hf_cfg)

    T = 32
    mb_torch.prepare_metadata_for_moe(T)
    mb_triton.prepare_metadata_for_moe(T)
    hidden = (torch.randn((1, T, hf_cfg.hidden_size), dtype=hf_cfg.torch_dtype) * 0.05).cuda()

    with torch.inference_mode():
        out_torch = fused_torch(hidden)
        out_triton = fused_triton(hidden)
        hf_out_t = hf_block(hidden)
        out_hf = hf_out_t[0] if isinstance(hf_out_t, tuple) else hf_out_t

    print(f"\nshapes: torch {out_torch.shape}, triton {out_triton.shape}, hf {out_hf.shape}")
    for name, a in [("torch", out_torch), ("triton", out_triton)]:
        abs_err = (a - out_hf).abs()
        rel_err = abs_err / out_hf.abs().clamp_min(1e-3)
        print(f"  vs HF — {name}: max abs {abs_err.max().item():.3e}  max rel {rel_err.max().item():.3e}  mean abs {abs_err.mean().item():.3e}")
    abs_err_tt = (out_triton - out_torch).abs()
    print(f"  triton vs torch: max abs {abs_err_tt.max().item():.3e}  mean abs {abs_err_tt.mean().item():.3e}")

    assert abs_err_tt.max().item() < 2e-2, "triton vs torch divergence too large"
    assert ((out_triton - out_hf).abs().max().item()) < 2e-2, "triton vs HF divergence too large"
    print("\nOK: Triton FusedMoE matches both torch reference and HuggingFace within bf16 tolerance")


def _load_layer0(fused_torch, fused_triton, hf_block, hf_cfg):
    from glob import glob
    from safetensors import safe_open

    L = 0
    needed = {f"model.layers.{L}.mlp.gate.weight"}
    for e in range(hf_cfg.num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            needed.add(f"model.layers.{L}.mlp.experts.{e}.{proj}.weight")

    found = {}
    for file in sorted(glob(os.path.join(CKPT, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for key in f.keys():
                if key in needed:
                    found[key] = f.get_tensor(key)
    assert len(found) == len(needed)

    gw = found[f"model.layers.{L}.mlp.gate.weight"]
    fused_torch.gate.weight.data.copy_(gw)
    fused_triton.gate.weight.data.copy_(gw)
    hf_block.gate.weight.data.copy_(gw)

    N = hf_cfg.moe_intermediate_size
    for e in range(hf_cfg.num_experts):
        gp = found[f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight"]
        up = found[f"model.layers.{L}.mlp.experts.{e}.up_proj.weight"]
        dp = found[f"model.layers.{L}.mlp.experts.{e}.down_proj.weight"]
        for fused in (fused_torch, fused_triton):
            fused.experts.w1.data[e, :N].copy_(gp)
            fused.experts.w1.data[e, N:].copy_(up)
            fused.experts.w2.data[e].copy_(dp)
        hf_block.experts.gate_up_proj.data[e, :N].copy_(gp)
        hf_block.experts.gate_up_proj.data[e, N:].copy_(up)
        hf_block.experts.down_proj.data[e].copy_(dp)


if __name__ == "__main__":
    main()
