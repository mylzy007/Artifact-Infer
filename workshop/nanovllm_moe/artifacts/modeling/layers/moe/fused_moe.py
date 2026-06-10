"""FusedMoE — the parent module that wires Dispatch → Experts → Combine.

The three submodules never reference each other: they communicate through a
small `TokMeta` `NamedTuple` and the workspace tensors registered onto each by
the orchestrator (see docs/moe/design.md §4).

Three flavors selected by `moe_mode`:
  - "single":  Dispatch    + Experts    + Combine      (block-tiled triton_fused_moe)
  - "ep_ll":   DispatchEPLL + ExpertsEPLL + CombineEPLL (dense a2a + masked GEMM)
  - "ep_ht":   DispatchEPHT + ExpertsEPHT + CombineEPHT (ragged a2a + standard triton_fused_moe)

The Experts module for ep_ht is essentially Experts with E=E_local; we factor it
into a thin subclass for clarity in the orchestrator wiring.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from workshop.nanovllm_moe.artifacts.modeling.layers.linear import ReplicatedLinear
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.combine import Combine
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.combine_ep_ht import CombineEPHT
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.combine_ep_ll import CombineEPLL
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch import Dispatch
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ht import DispatchEPHT
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ll import DispatchEPLL
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.experts import Experts
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.experts_ep_ht import ExpertsEPHT
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.experts_ep_ll import ExpertsEPLL
from workshop.nanovllm_moe.services.utils.parallel import get_runtime_mode, is_dp_leader


class FusedMoE(nn.Module):
    """Owns the routing gate (so HF's `mlp.gate.weight` matches at the same
    nesting depth) and wires Dispatch → Experts → Combine."""

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        block_size_m: int,
        norm_topk_prob: bool = True,
        moe_mode: str = "single",            # "single" | "ep_ll" | "ep_ht"
        m_max: int = 0,
        ep_ll_dispatch_kernel: str = "triton",
        expert_placement: str = "contiguous",
        expert_placement_seed: int = 0,
        expert_placement_path: str | None = None,
        expert_overlap_enabled: bool = False,
        expert_overlap_path: str | None = None,
        expert_overlap_strategy: str = "hybrid",
        ep_ll_overflow_policy: str = "drop",
        drop_policy: str = "none",
        drop_rate: float = 0.0,
        drop_seed: int = 0,
        router_keff: int = 0,
        combine_compress_keep_frac: float = 0.0,
        combine_compress_fp8: bool = True,
        combine_compress_use_l2: bool = True,
        layer_id: int = -1,
    ) -> None:
        super().__init__()
        assert moe_mode in ("single", "ep_ll", "ep_ht"), f"unknown moe_mode {moe_mode!r}"
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_mode = moe_mode
        self.layer_id = int(layer_id)
        self.is_ep_ll = moe_mode == "ep_ll"
        self.is_ep_ht = moe_mode == "ep_ht"

        # The routing gate. Replicated across all ranks (small matrix).
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)

        if self.is_ep_ll:
            assert m_max > 0, "EP-LL requires m_max > 0 (set in MoeBackend)"
            self.dispatch = DispatchEPLL(
                num_experts_global=num_experts,
                top_k=top_k,
                m_max=m_max,
                norm_topk_prob=norm_topk_prob,
                dispatch_kernel=ep_ll_dispatch_kernel,
                expert_placement=expert_placement,
                expert_placement_seed=expert_placement_seed,
                expert_placement_path=expert_placement_path,
                layer_id=self.layer_id,
                overflow_policy=ep_ll_overflow_policy,
            )
            self.experts = ExpertsEPLL(
                num_experts_global=num_experts,
                hidden_size=hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                expert_placement=expert_placement,
                expert_placement_seed=expert_placement_seed,
                expert_placement_path=expert_placement_path,
                layer_id=self.layer_id,
            )
            self.combine = CombineEPLL(
                hidden_size=hidden_size,
                top_k=top_k,
                num_experts_local=self.experts.E_local,
                m_max=m_max,
            )
        elif self.is_ep_ht:
            self.dispatch = DispatchEPHT(
                num_experts_global=num_experts,
                top_k=top_k,
                block_size_m=block_size_m,
                norm_topk_prob=norm_topk_prob,
                expert_placement=expert_placement,
                expert_placement_seed=expert_placement_seed,
                expert_placement_path=expert_placement_path,
                expert_overlap_enabled=expert_overlap_enabled,
                expert_overlap_path=expert_overlap_path,
                expert_overlap_strategy=expert_overlap_strategy,
                drop_policy=drop_policy,
                drop_rate=drop_rate,
                drop_seed=drop_seed,
                router_keff=router_keff,
                layer_id=self.layer_id,
            )
            self.experts = ExpertsEPHT(
                num_experts_global=num_experts,
                hidden_size=hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                expert_placement=expert_placement,
                expert_placement_seed=expert_placement_seed,
                expert_placement_path=expert_placement_path,
                expert_overlap_enabled=expert_overlap_enabled,
                expert_overlap_path=expert_overlap_path,
                layer_id=self.layer_id,
            )
            self.combine = CombineEPHT(
                hidden_size=hidden_size,
                top_k=top_k,
                compress_keep_frac=combine_compress_keep_frac,
                compress_fp8=combine_compress_fp8,
                compress_use_l2=combine_compress_use_l2,
            )
        else:
            self.dispatch = Dispatch(
                num_experts=num_experts,
                top_k=top_k,
                block_size_m=block_size_m,
                norm_topk_prob=norm_topk_prob,
            )
            self.experts = Experts(
                num_experts=num_experts,
                hidden_size=hidden_size,
                moe_intermediate_size=moe_intermediate_size,
            )
            self.combine = Combine(hidden_size=hidden_size, top_k=top_k)

    def _debug_sync(self, stage: str) -> None:
        if os.environ.get("MOE_EPHT_DEBUG_SYNC", "0") == "1":
            torch.cuda.synchronize()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        x = hidden_states.view(-1, original_shape[-1])
        if self.is_ep_ht:
            self._debug_sync("entry")
        # vllm_dp_ep uses one TP leader as the routing source. owner_local_ep does not:
        # every rank owns its local tokens and computes local router logits.
        if self.is_ep_ht and get_runtime_mode() == "vllm_dp_ep" and not is_dp_leader():
            router_logits = torch.empty((0,), dtype=x.dtype, device=x.device)
        else:
            router_logits = self.gate(x)                       # [T, E]
        if self.is_ep_ht:
            self._debug_sync("gate")

        if self.is_ep_ll:
            tok_meta = self.dispatch(x, router_logits)
            expert_out = self.experts(tok_meta)            # [E_local, N*M_max, H]
            out = self.combine(expert_out, tok_meta)       # [T, H]
        elif self.is_ep_ht:
            tok_meta = self.dispatch(x, router_logits)
            expert_out = self.experts(tok_meta)            # [total_recv, 1, H]
            out = self.combine(expert_out, tok_meta)       # [T, H]
        else:
            tok_meta = self.dispatch(router_logits)
            expert_out_TKH = self.experts(x, tok_meta)
            out = self.combine(expert_out_TKH)

        return out.view(original_shape)
