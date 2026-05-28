"""CombineEPHT — Expert-Parallel High-Throughput combine (eager-only).

Inverse of DispatchEPHT.

Inputs:
  expert_out: [total_recv, 1, H]  (or [total_recv, H])  bf16
  tok_meta : TokMetaEPHT (kept on this rank since DispatchEPHT)

Steps:
  1. Reverse all_to_all_single on expert_out using REVERSED counts:
     output_split_sizes=send_counts, input_split_sizes=recv_counts.
     -> rev_perm[T*K, H] in the SAME sorted-by-target-rank order we had on send.
  2. Un-permute: gather by sort_perm to get back [T, K, H] layout via index.
  3. Weight-and-reduce: out[t] = sum_k topk_weights[t, k] * unperm[t, k]
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn

from src.core.artifact import Artifact
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ht import TokMetaEPHT
from workshop.nanovllm_moe.services.utils.parallel import (
    get_ep_group,
    get_ep_world_size,
    get_runtime_mode,
    get_tp_group,
)


class CombineEPHT(Artifact, nn.Module):
    @property
    def name(self) -> str:
        return "CombineEPHT"

    def __init__(self, hidden_size: int, top_k: int) -> None:
        super().__init__()
        self.H = int(hidden_size)
        self.K = int(top_k)
        self.world_size = get_ep_world_size() if dist.is_initialized() else 1
        self.runtime_mode = get_runtime_mode()

    def _debug_sync(self, stage: str) -> None:
        if os.environ.get("MOE_EPHT_DEBUG_SYNC", "0") == "1":
            torch.cuda.synchronize()

    def forward(
        self,
        expert_out: torch.Tensor,    # [total_recv, 1, H]   or   [total_recv, H]
        tok_meta: TokMetaEPHT,
    ) -> torch.Tensor:
        N = self.world_size
        H = self.H
        # Phase 4 P1: when router K_eff < self.K, tok_meta.topk_weights has K_eff
        # columns. Read the actual K from tok_meta so combine matches dispatch.
        K = int(tok_meta.topk_weights.shape[1])
        T = tok_meta.T_local
        device = expert_out.device
        dtype = expert_out.dtype

        # Squeeze the K=1 dim if present.
        if expert_out.ndim == 3:
            assert expert_out.shape[1] == 1, (
                f"EP-HT expert_out should be [total_recv, 1, H], got {expert_out.shape}"
            )
            expert_out = expert_out.squeeze(1)   # [total_recv, H]

        total_recv = expert_out.shape[0]
        send_total = sum(tok_meta.send_counts)

        # 1. Reverse a2a: send back per-recv-rank slices.
        rev = torch.empty(send_total, H, dtype=dtype, device=device)
        if N > 1:
            dist.all_to_all_single(
                rev, expert_out.contiguous(),
                output_split_sizes=tok_meta.send_counts,
                input_split_sizes=tok_meta.recv_counts,
                group=get_ep_group(),
            )
        else:
            rev.copy_(expert_out)
        self._debug_sync("reverse_a2a")

        # owner_local_ep always combines locally on the original token owner.
        if self.runtime_mode == "vllm_dp_ep" and not tok_meta.is_source_leader:
            out = torch.empty(T, H, dtype=dtype, device=device)
        else:
            # zeros (not empty): when Phase 4 drop is enabled, sort_perm only
            # covers kept (token, expert) positions and the remaining slots must
            # contribute 0 to the weighted sum. When drop is off, sort_perm
            # spans every (t, k) and the zero init is overwritten in full.
            unperm = torch.zeros(T * K, H, dtype=dtype, device=device)
            unperm[tok_meta.sort_perm] = rev
            unperm_TKH = unperm.view(T, K, H)
            weights = tok_meta.topk_weights.to(dtype).unsqueeze(-1)  # [T, K, 1]
            out = (unperm_TKH * weights).sum(dim=1)                   # [T, H]

        # vllm_dp_ep: leader combines, then broadcasts within TP replica.
        # owner_local_ep: tp=1, so no output broadcast is allowed or needed.
        if self.runtime_mode == "vllm_dp_ep" and dist.is_initialized() and dist.get_world_size() > 1 and T > 0:
            dist.broadcast(out, src=tok_meta.source_leader_global_rank, group=get_tp_group())
        self._debug_sync("reduce")
        return out
