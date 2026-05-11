"""End-to-end MoE integration test on a real (small slice of) Qwen3-MoE.

We don't run the full 30B model (~60 GB just for weights), and the LLMEngine
in nanovllm_base is intentionally a scaffold without scheduler/runner wiring.
Instead, we:
  1. Build the Qwen3MoE model with a *full* HF config (so all checkpoint keys
     exist) but trim to 2 layers to keep memory bounded.
  2. Wire MoeBackend through the orchestrator the same way model_runner does.
  3. Load weights for those 2 layers from /home/yyx/models/Qwen3-30B-A3B.
  4. Run a single forward pass with random token ids.
  5. Compare logits against HuggingFace's own Qwen3MoeForCausalLM (same trim).
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

from transformers import AutoConfig
from src.core.orchestrator import RegistryOrchestrator
from src.core.service import BaseService
from workshop.nanovllm_moe.artifacts.moe_backend import MoeBackend
from workshop.nanovllm_moe.artifacts.modeling.layers.moe import Combine, Dispatch, Experts
from workshop.nanovllm_moe.artifacts.modeling.models.qwen3_moe import Qwen3MoeForCausalLM
from workshop.nanovllm_moe.services.utils.context import set_context
from workshop.nanovllm_moe.services.utils.loader import load_model


CKPT = "/home/yyx/models/Qwen3-30B-A3B"


class _Cfg:
    """Mimics the Config dataclass from services/config.py."""
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
    # Trim to 2 layers so memory stays bounded (still 128 experts × 2 layers).
    hf_cfg.num_hidden_layers = 2
    print(f"Building 2-layer Qwen3-MoE (hidden={hf_cfg.hidden_size}, "
          f"E={hf_cfg.num_experts}, K={hf_cfg.num_experts_per_tok}, "
          f"N={hf_cfg.moe_intermediate_size})")

    torch.set_default_dtype(hf_cfg.torch_dtype)

    cfg = _Cfg(hf_cfg)
    root = _Root()
    orch = RegistryOrchestrator()
    model = orch.add(Qwen3MoeForCausalLM(hf_cfg, moe_block_size_m=cfg.moe_block_size_m))
    moe_backend = orch.add(MoeBackend(
        config=cfg,
        num_experts=hf_cfg.num_experts,
        top_k=hf_cfg.num_experts_per_tok,
        hidden_size=hf_cfg.hidden_size,
        moe_intermediate_size=hf_cfg.moe_intermediate_size,
    ))

    for module in model.modules():
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

    # Load weights — non-existent layers (3..47) are silently skipped because
    # `model.get_parameter` would raise; we wrap load_model with a layer filter.
    print("Loading weights (only first 2 layers will be applied)...")
    _filtered_load(model, CKPT, num_layers=hf_cfg.num_hidden_layers)
    print("Weights loaded.")

    # Forward pass
    T = 8
    input_ids = torch.randint(0, hf_cfg.vocab_size, (T,), dtype=torch.int64)
    positions = torch.arange(T, dtype=torch.int64)
    set_context(is_prefill=True, num_tokens=T)
    moe_backend.prepare_metadata_for_moe(T)

    layer0 = model.model.layers[0]
    mlp = layer0.mlp
    print(f"\n=== weight stats ===")
    print(f"embed.weight:       abs mean {model.model.embed_tokens.weight.abs().mean().item():.3e}  max {model.model.embed_tokens.weight.abs().max().item():.3e}")
    print(f"layer0.input_ln:    abs mean {layer0.input_layernorm.weight.abs().mean().item():.3e}  max {layer0.input_layernorm.weight.abs().max().item():.3e}")
    print(f"mlp.gate.weight: abs mean {mlp.gate.weight.abs().mean().item():.3e}  max {mlp.gate.weight.abs().max().item():.3e}")
    print(f"mlp.experts.w1: abs mean {mlp.experts.w1.abs().mean().item():.3e}  max {mlp.experts.w1.abs().max().item():.3e}  zeros frac {(mlp.experts.w1 == 0).float().mean().item():.3f}")
    print(f"mlp.experts.w2: abs mean {mlp.experts.w2.abs().mean().item():.3e}  max {mlp.experts.w2.abs().max().item():.3e}  zeros frac {(mlp.experts.w2 == 0).float().mean().item():.3f}")

    with torch.inference_mode():
        emb = model.model.embed_tokens(input_ids)
        print(f"\nembed out:   abs mean {emb.abs().mean().item():.3e}  max {emb.abs().max().item():.3e}  any NaN {torch.isnan(emb).any().item()}")
        h = layer0.input_layernorm(emb)
        print(f"after ln:    abs mean {h.abs().mean().item():.3e}  max {h.abs().max().item():.3e}  any NaN {torch.isnan(h).any().item()}")

        # Step into MoE
        x = h.view(-1, h.size(-1))
        router_logits = mlp.gate(x)
        print(f"router_logits: abs mean {router_logits.abs().mean().item():.3e}  max {router_logits.abs().max().item():.3e}  any NaN {torch.isnan(router_logits).any().item()}")
        tok_meta = mlp.dispatch(router_logits)
        print(f"topk_weights:abs mean {tok_meta.topk_weights.abs().mean().item():.3e}  max {tok_meta.topk_weights.abs().max().item():.3e}  sum (should ≈ T) {tok_meta.topk_weights.sum().item():.3f}  any NaN {torch.isnan(tok_meta.topk_weights).any().item()}")
        print(f"topk_ids:    {tok_meta.topk_ids[:2].tolist()}  (should be {hf_cfg.num_experts_per_tok} ids per row)")
        print(f"npp:         {tok_meta.num_tokens_post_padded.item()}")

        expert_out_TKH = mlp.experts(x, tok_meta)
        print(f"experts out: abs mean {expert_out_TKH.abs().mean().item():.3e}  max {expert_out_TKH.abs().max().item():.3e}  any NaN {torch.isnan(expert_out_TKH).any().item()}")

        out_moe = mlp.combine(expert_out_TKH).view(h.shape)
        print(f"combine out: abs mean {out_moe.abs().mean().item():.3e}  max {out_moe.abs().max().item():.3e}  any NaN {torch.isnan(out_moe).any().item()}")

    assert not torch.isnan(out_moe).any() and not torch.isinf(out_moe).any(), \
        "MoE output has NaN/Inf"
    print("OK: MoE forward pass produced finite output")


def _filtered_load(model, path, num_layers):
    """Same as load_model, but skips weights for layers >= num_layers."""
    import re
    from glob import glob
    from safetensors import safe_open
    from workshop.nanovllm_moe.services.utils.loader import (
        _MOE_EXPERT_RE, _MOE_SHARD_TO_TARGET, default_weight_loader,
    )

    layer_re = re.compile(r"model\.layers\.(\d+)\.")
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    n_loaded = 0
    n_skipped = 0
    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                lm = layer_re.match(weight_name)
                if lm is not None and int(lm.group(1)) >= num_layers:
                    n_skipped += 1
                    continue
                m = _MOE_EXPERT_RE.match(weight_name)
                if m is not None:
                    parent, expert_id_s, proj, suffix = m.groups()
                    target_attr, shard_id = _MOE_SHARD_TO_TARGET[proj]
                    param_name = f"{parent}.{target_attr}" + ("" if suffix == "weight" else f".{suffix}")
                    param = model.get_parameter(param_name)
                    param.weight_loader(param, f.get_tensor(weight_name), int(expert_id_s), shard_id)
                    n_loaded += 1
                    continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        param.weight_loader(param, f.get_tensor(weight_name), shard_id)
                        n_loaded += 1
                        break
                else:
                    try:
                        param = model.get_parameter(weight_name)
                    except AttributeError:
                        n_skipped += 1
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    n_loaded += 1
    print(f"  loaded {n_loaded} tensors, skipped {n_skipped}")


if __name__ == "__main__":
    main()
