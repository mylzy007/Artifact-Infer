"""ExpertsEPHT — per-rank expert holder for EP-HT.

Same weight-sharding plan as ExpertsEPLL:
  w1 : [E_local, 2N, H]    bf16
  w2 : [E_local, H, N]     bf16
loaded with the global expert_id, dropping non-local experts.

Forward delegates to MoeBackend.run_experts (registered by the orchestrator)
which dispatches to triton_fused_moe — the same inner kernel used by the
single-rank path. EP-HT achieves "expert parallelism" purely by replacing
the surrounding Dispatch/Combine; the kernel itself doesn't know it's running
EP.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn

from src.core.artifact import Artifact
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ht import TokMetaEPHT
from workshop.nanovllm_moe.services.utils.expert_overlap import load_expert_overlap_plan
from workshop.nanovllm_moe.services.utils.expert_placement import build_expert_placement
from workshop.nanovllm_moe.services.utils.parallel import get_ep_rank, get_ep_world_size


class ExpertsEPHT(Artifact, nn.Module):
    @property
    def name(self) -> str:
        return "ExpertsEPHT"

    def __init__(
        self,
        num_experts_global: int,
        hidden_size: int,
        moe_intermediate_size: int,
        expert_placement: str = "contiguous",
        expert_placement_seed: int = 0,
        expert_placement_path: str | None = None,
        expert_overlap_enabled: bool = False,
        expert_overlap_path: str | None = None,
        layer_id: int = -1,
    ) -> None:
        super().__init__()
        self.E_global = int(num_experts_global)
        self.H = int(hidden_size)
        self.N = int(moe_intermediate_size)
        self.layer_id = int(layer_id)

        self.world_size = get_ep_world_size() if dist.is_initialized() else 1
        self.rank = get_ep_rank() if dist.is_initialized() else 0
        self.overlap_plan = load_expert_overlap_plan(
            num_experts=self.E_global,
            world_size=self.world_size,
            expert_placement=expert_placement,
            expert_placement_seed=expert_placement_seed,
            expert_placement_path=expert_placement_path,
            overlap_path=expert_overlap_path if expert_overlap_enabled else None,
        )
        self.E_local = len(self.overlap_plan.experts_by_rank[self.rank])
        if self.E_local <= 0:
            raise RuntimeError(f"rank {self.rank} hosts zero experts")
        if self.overlap_plan.enabled:
            local_experts = self.overlap_plan.experts_by_rank[self.rank]
        else:
            self.placement = build_expert_placement(
                self.E_global,
                self.world_size,
                expert_placement,
                seed=expert_placement_seed,
                placement_path=expert_placement_path,
                layer_id=self.layer_id if self.layer_id >= 0 else None,
            )
            local_experts = self.placement.local_to_global[self.rank]
        self.global_to_local = [-1] * self.E_global
        for local_id, expert_id in enumerate(local_experts):
            self.global_to_local[expert_id] = local_id

        self.w1 = nn.Parameter(
            torch.empty((self.E_local, 2 * self.N, self.H))
        )
        self.w2 = nn.Parameter(
            torch.empty((self.E_local, self.H, self.N))
        )
        self.w1.weight_loader = self._w1_loader
        self.w2.weight_loader = self._w2_loader

    def _debug_sync(self, stage: str) -> None:
        if os.environ.get("MOE_EPHT_DEBUG_SYNC", "0") == "1":
            torch.cuda.synchronize()

    def _expert_is_local(self, expert_id_global: int) -> bool:
        return self.global_to_local[expert_id_global] >= 0

    def _w1_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                   expert_id: int, shard_id: str) -> None:
        if not self._expert_is_local(expert_id):
            return
        local_id = self.global_to_local[expert_id]
        if shard_id == "gate":
            param.data[local_id, 0:self.N, :].copy_(loaded_weight)
        elif shard_id == "up":
            param.data[local_id, self.N:2 * self.N, :].copy_(loaded_weight)
        else:
            raise ValueError(f"w1 expects shard_id in {{'gate', 'up'}}, got {shard_id!r}")

    def _w2_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                   expert_id: int, shard_id: str | None) -> None:
        if not self._expert_is_local(expert_id):
            return
        local_id = self.global_to_local[expert_id]
        param.data[local_id].copy_(loaded_weight)

    def forward(self, tok_meta: TokMetaEPHT) -> torch.Tensor:
        """Returns expert_out: [total_recv, 1, H]   bf16.

        Calls MoeBackend.run_experts (set by orchestrator) which itself dispatches
        to triton_fused_moe with the receive-side metadata DispatchEPHT prepared.
        """
        # Empty-recv shortcut: if no tokens routed here, return zero shaped
        # to (0, 1, H). CombineEPHT handles total_recv=0 fine.
        if tok_meta.recv_hidden.shape[0] == 0:
            return torch.zeros(
                (0, 1, self.H),
                dtype=tok_meta.recv_hidden.dtype,
                device=tok_meta.recv_hidden.device,
            )

        assert hasattr(self, "run_experts"), (
            "ExpertsEPHT requires MoeBackend.run_experts registered via the orchestrator"
        )
        # run_experts returns intermediate_cache3 view of shape [T_in, K, H].
        # Here T_in = total_recv, K=1.
        out = self.run_experts(
            hidden_states=tok_meta.recv_hidden,
            w1=self.w1,
            w2=self.w2,
            topk_weights=tok_meta.recv_topk_weights,    # [total_recv, 1]   = 1.0
            topk_ids=tok_meta.recv_topk_ids,            # [total_recv, 1]
            sorted_token_ids=tok_meta.sorted_token_ids,
            expert_ids=tok_meta.expert_ids,
            num_tokens_post_padded=tok_meta.num_tokens_post_padded,
        )
        self._debug_sync("run_experts")
        return out
