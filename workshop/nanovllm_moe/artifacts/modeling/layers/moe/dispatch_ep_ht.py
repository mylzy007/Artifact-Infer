"""DispatchEPHT — Expert-Parallel High-Throughput dispatcher (eager-only).

Differs from EP-LL in three key ways:
  1. Send/recv buffers are RAGGED (compact, no zero padding) — variable per-rank
     send sizes computed via bincount + host sync.
  2. Receive side runs the STANDARD single-rank fused_moe (block-tiled triton)
     by invoking sgl_kernel.moe_align_block_size on the received local-expert ids.
  3. NOT cuda-graph compatible (variable-size NCCL a2a needs Python int lists).
     The trade-off vs EP-LL: lower kernel FLOPs (no zero padding), higher latency
     per layer (host sync + variable a2a overhead), so this wins for large batches.

Pipeline (sender side):
  hidden[T, H], router_logits[T, E_global]
  -> topk_ids[T, K], topk_w[T, K]
  -> target_rank[T*K] = topk_ids // E_local
  -> sort token-replicas by target_rank:
       sorted_hidden  : [T*K, H]
       sorted_eid     : [T*K]            local expert id at destination
       sorted_topk_w  : [T*K]            weight for the (t, k) replica
       sort_perm      : [T*K]            inverse map for combine
  -> send_counts[N] = bincount(target_rank)
  -> all_to_all on counts to learn recv_counts[N]
  -> host-sync both, then all_to_all_single (3 calls, one per payload) with
     explicit input/output split sizes
  -> recv_hidden[total_recv, H], recv_eid[total_recv], recv_topk_w[total_recv]
  -> sgl_kernel.moe_align_block_size on recv_eid to build the standard
     (sorted_token_ids, expert_ids, num_tokens_post_padded) for the inner kernel.

Buffers:
  - All payloads are dynamically sized per-call (no persistent ragged buffer).
  - The receive-side (sorted_token_ids etc.) buffers are sized for
    ep_size * T_cap * K worst-case.
"""
from __future__ import annotations

import os
from typing import NamedTuple

import torch
import torch.distributed as dist
from torch import nn

from src.core.artifact import Artifact
from workshop.nanovllm_moe.services.utils import overlap_runtime_stats
from workshop.nanovllm_moe.services.utils.expert_drop import (
    ALL_DROP_POLICIES,
    apply_drop,
)
from workshop.nanovllm_moe.services.utils.expert_overlap import (
    ExpertOverlapRouter,
    load_expert_overlap_plan,
)
from workshop.nanovllm_moe.services.utils.expert_placement import build_expert_placement
from workshop.nanovllm_moe.services.utils.parallel import (
    get_dp_leader_global_rank,
    get_global_rank,
    get_ep_group,
    get_ep_rank,
    get_ep_world_size,
    get_runtime_mode,
    is_dp_leader,
    is_owner_local_ep_mode,
)


class TokMetaEPHT(NamedTuple):
    """EP-HT metadata flowing DispatchEPHT -> ExpertsEPHT -> CombineEPHT."""
    # Output of dispatch a2a, ready for the standard single-rank fused_moe.
    recv_hidden: torch.Tensor          # [total_recv, H]              bf16
    recv_topk_ids: torch.Tensor        # [total_recv, 1]              int32   (local expert id)
    recv_topk_weights: torch.Tensor    # [total_recv, 1]              fp32    (always 1.0)
    sorted_token_ids: torch.Tensor     # int32, output of moe_align
    expert_ids: torch.Tensor           # int32
    num_tokens_post_padded: torch.Tensor  # int32 [1]

    # Per-(t,k) topk weights for combine (kept on this rank).
    topk_weights: torch.Tensor         # [T, K]   fp32
    topk_ids: torch.Tensor             # [T, K]   int32

    # Sort permutation: position-in-sorted -> (t * K + k). Needed by combine
    # to undo the permutation on the reverse path.
    sort_perm: torch.Tensor            # [T*K]   int64
    send_counts: list[int]             # [N]     host-side
    recv_counts: list[int]             # [N]     host-side
    T_local: int
    is_source_leader: bool
    source_leader_global_rank: int


class DispatchEPHT(Artifact, nn.Module):
    @property
    def name(self) -> str:
        return "DispatchEPHT"

    def __init__(
        self,
        num_experts_global: int,
        top_k: int,
        block_size_m: int,
        norm_topk_prob: bool = True,
        expert_placement: str = "contiguous",
        expert_placement_seed: int = 0,
        expert_placement_path: str | None = None,
        expert_overlap_enabled: bool = False,
        expert_overlap_path: str | None = None,
        expert_overlap_strategy: str = "hybrid",
        drop_policy: str = "none",
        drop_rate: float = 0.0,
        drop_seed: int = 0,
        router_keff: int = 0,
        layer_id: int = -1,
    ) -> None:
        super().__init__()
        self.E_global = int(num_experts_global)
        self.K_model = int(top_k)
        # Phase 4 P1 router K_eff: when > 0 and < model top_k, only the top
        # K_eff branches are routed (no keep_mask machinery needed). This is a
        # router-level reduction that directly cuts a2a / expert / combine work.
        keff = int(router_keff)
        if keff > 0 and keff < self.K_model:
            self.K = keff
            self.router_keff_active = True
        else:
            self.K = self.K_model
            self.router_keff_active = False
        self.BLOCK_M = int(block_size_m)
        self.norm_topk_prob = norm_topk_prob
        self.layer_id = int(layer_id)
        if drop_policy not in ALL_DROP_POLICIES:
            raise ValueError(
                f"unknown drop_policy {drop_policy!r}; expected one of {ALL_DROP_POLICIES}"
            )
        self.drop_policy = str(drop_policy)
        self.drop_rate = float(drop_rate)
        self.drop_enabled = self.drop_policy != "none" and self.drop_rate > 0.0
        self.drop_seed = int(drop_seed)
        # Seed is per-layer so each layer has its own RNG stream when policy is
        # random. Reseeding every layer means the sequence is deterministic
        # across runs with the same seed and same routing.
        self._drop_rng = __import__("random").Random(self.drop_seed + self.layer_id * 1009)
        # Lazy GPU generator; created on first forward when we know the device.
        self._drop_torch_generator: torch.Generator | None = None
        self._drop_torch_generator_device: torch.device | None = None
        # Resolve impl + small-batch bypass from env once. Override in tests by
        # setting the env vars before constructing the layer.
        self.drop_impl = os.environ.get("MOE_DROP_IMPL", "auto").lower()
        try:
            self.drop_min_replicas = int(os.environ.get("MOE_DROP_MIN_REPLICAS", "0"))
        except ValueError:
            self.drop_min_replicas = 0
        self.drop_collect_stats = os.environ.get("MOE_DROP_GPU_STATS", "0") == "1"

        self.world_size = get_ep_world_size() if dist.is_initialized() else 1
        self.rank = get_ep_rank() if dist.is_initialized() else 0
        self.runtime_mode = get_runtime_mode()
        if dist.is_initialized() and is_owner_local_ep_mode():
            # owner_local_ep: every rank is the local token owner/source.
            self.is_source_leader = True
            self.source_leader_global_rank = get_global_rank()
        else:
            self.is_source_leader = is_dp_leader() if dist.is_initialized() else True
            self.source_leader_global_rank = get_dp_leader_global_rank() if dist.is_initialized() else 0
        self.overlap_plan = load_expert_overlap_plan(
            num_experts=self.E_global,
            world_size=self.world_size,
            expert_placement=expert_placement,
            expert_placement_seed=expert_placement_seed,
            expert_placement_path=expert_placement_path,
            overlap_path=expert_overlap_path if expert_overlap_enabled else None,
        )
        self.E_local = len(self.overlap_plan.experts_by_rank[self.rank])
        self.overlap_router = ExpertOverlapRouter(
            self.overlap_plan,
            "disjoint" if self.overlap_plan.is_disjoint else expert_overlap_strategy,
        )
        if self.overlap_plan.enabled:
            local_index = self.overlap_plan.local_index_by_rank_expert
            self.register_buffer(
                "local_index_by_rank_expert",
                torch.tensor(local_index, dtype=torch.int32),
                persistent=False,
            )
        else:
            self.placement = build_expert_placement(
                self.E_global,
                self.world_size,
                expert_placement,
                seed=expert_placement_seed,
                placement_path=expert_placement_path,
                layer_id=self.layer_id if self.layer_id >= 0 else None,
            )
            self.register_buffer(
                "expert_to_rank",
                torch.tensor(self.placement.expert_to_rank, dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "expert_to_local",
                torch.tensor(self.placement.expert_to_local, dtype=torch.int32),
                persistent=False,
            )

        # Buffers populated by orchestrator (registered from MoeBackend).
        self.sorted_token_ids_buf: torch.Tensor    # int32
        self.expert_ids_buf: torch.Tensor          # int32
        self.num_tokens_post_padded: torch.Tensor  # int32 [1]
        self.cumsum_buffer: torch.Tensor           # int32 [E_local + 2]

    def _debug_sync(self, stage: str) -> None:
        if os.environ.get("MOE_EPHT_DEBUG_SYNC", "0") == "1":
            torch.cuda.synchronize()

    def _ensure_buffers(self, T_cap: int, device: torch.device):
        """Lazy allocation for standalone tests."""
        if hasattr(self, "sorted_token_ids_buf") and isinstance(
            self.sorted_token_ids_buf, torch.Tensor
        ):
            return
        # Worst case: every source rank routes all local replicas to this rank.
        max_in = self.world_size * T_cap * self.K
        max_padded = max_in + (self.E_local + 1) * (self.BLOCK_M - 1)
        max_blocks = (max_padded + self.BLOCK_M - 1) // self.BLOCK_M
        self.sorted_token_ids_buf = torch.empty(max_padded, dtype=torch.int32, device=device)
        self.expert_ids_buf = torch.empty(max_blocks, dtype=torch.int32, device=device)
        self.num_tokens_post_padded = torch.zeros(1, dtype=torch.int32, device=device)
        self.cumsum_buffer = torch.empty(self.E_local + 2, dtype=torch.int32, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [T, H]   bf16
        router_logits: torch.Tensor,   # [T, E_global]
    ) -> TokMetaEPHT:
        from sgl_kernel import moe_align_block_size

        T, H = hidden_states.shape
        K = self.K
        N = self.world_size
        E_local = self.E_local
        device = hidden_states.device
        dtype = hidden_states.dtype

        T_cap = T  # for standalone tests; lazy buffers sized to current T_cap is fine
        self._ensure_buffers(T_cap, device)
        if hasattr(self, "topk_ids_buf") and T > self.topk_ids_buf.size(0):
            raise RuntimeError(
                f"DispatchEPHT got T={T}, exceeding topk buffer capacity "
                f"{self.topk_ids_buf.size(0)}"
            )
        self._debug_sync("entry")

        # ---- 1. topk + softmax (cuda-graph compatible — pure GPU ops) ----
        if self.runtime_mode == "vllm_dp_ep" and not self.is_source_leader:
            topk_ids = torch.empty((T, K), dtype=torch.int32, device=device)
            topk_weights = torch.empty((T, K), dtype=torch.float32, device=device)
        else:
            logits_fp32 = router_logits.float()
            self._debug_sync("logits_float")
            topk_vals, topk_ids = torch.topk(logits_fp32, K, dim=-1)         # [T, K]
            self._debug_sync("torch_topk")
            topk_weights = torch.softmax(topk_vals, dim=-1)
            self._debug_sync("softmax")
            if self.norm_topk_prob:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                self._debug_sync("norm_topk")
            topk_ids = topk_ids.to(torch.int32)
            from workshop.nanovllm_moe.services.utils.routing_profile import record_routing
            record_routing(self.layer_id, topk_ids, self.E_global)
            self._debug_sync("topk_ids")

        # ---- 2. flatten + sort by target rank ----
        if self.runtime_mode == "vllm_dp_ep" and not self.is_source_leader:
            flat_eid = torch.empty((0,), dtype=torch.int32, device=device)
            target_rank = torch.empty((0,), dtype=torch.int64, device=device)
            flat_topk_w = torch.empty((0,), dtype=torch.float32, device=device)
            flat_local_eid = torch.empty((0,), dtype=torch.int32, device=device)
            sort_perm = torch.empty((0,), dtype=torch.int64, device=device)
            sorted_local_eid = torch.empty((0,), dtype=torch.int32, device=device)
            sorted_topk_w = torch.empty((0,), dtype=torch.float32, device=device)
            sorted_hidden = torch.empty((0, H), dtype=dtype, device=device)
        else:
            flat_eid = topk_ids.view(-1)                           # [T*K]   int32
            if self.overlap_plan.enabled:
                routed = self.overlap_router.route(flat_eid, self.rank)
                target_rank = routed.target_rank
                flat_local_eid = routed.local_expert_index
            else:
                target_rank = self.expert_to_rank[flat_eid.long()].long()  # [T*K]
                flat_local_eid = self.expert_to_local[flat_eid.long()].to(torch.int32)
            flat_topk_w = topk_weights.view(-1)                    # [T*K]   fp32

            # ---- 2.5 (Phase 4) optional token-replica drop ----
            # Small-batch bypass: when T*K is too small (decode steps), the drop
            # machinery costs more than it saves. Skip entirely.
            drop_active = (
                self.drop_enabled
                and target_rank.numel() > 0
                and (self.drop_min_replicas <= 0 or target_rank.numel() > self.drop_min_replicas)
            )
            if drop_active:
                # Lazy-init a per-layer torch.Generator on the right device for
                # the GPU random path. Seed kept consistent with the CPU RNG so
                # comparable across impls.
                if (
                    self._drop_torch_generator is None
                    or self._drop_torch_generator_device != device
                ):
                    self._drop_torch_generator = torch.Generator(device=device)
                    self._drop_torch_generator.manual_seed(
                        self.drop_seed + self.layer_id * 1009
                    )
                    self._drop_torch_generator_device = device
                drop_result = apply_drop(
                    flat_expert_ids=flat_eid,
                    target_rank=target_rank,
                    flat_topk_w=flat_topk_w,
                    source_rank=self.rank,
                    world_size=self.world_size,
                    K=K,
                    drop_policy=self.drop_policy,
                    drop_rate=self.drop_rate,
                    rng=self._drop_rng,
                    impl=self.drop_impl,
                    min_replicas=self.drop_min_replicas,
                    torch_generator=self._drop_torch_generator,
                    collect_stats=self.drop_collect_stats,
                    num_experts_global=self.E_global,
                )
                keep_mask = drop_result.keep_mask                  # [T*K]  bool
                # Renormalize topk_weights row-wise over kept positions; dropped
                # entries become 0 and the surviving weights sum to 1 per token.
                keep_mask_TK = keep_mask.view(T, K).to(topk_weights.dtype)
                topk_weights = topk_weights * keep_mask_TK
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                flat_topk_w = topk_weights.view(-1)
                kept_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)  # [L_kept]
                tr_kept = target_rank[kept_indices]
                kept_sort_local = torch.argsort(tr_kept, stable=True)
                sort_perm = kept_indices[kept_sort_local]          # [L_kept]
                overlap_runtime_stats.record_drop(self.layer_id, drop_result, self.rank)
            else:
                sort_perm = torch.argsort(target_rank, stable=True)  # [T*K]
            sorted_local_eid = flat_local_eid[sort_perm].contiguous()
            sorted_topk_w = flat_topk_w[sort_perm].contiguous()
            sorted_token_idx = (sort_perm // K).contiguous()       # [L_kept]  int64
            sorted_hidden = hidden_states.index_select(0, sorted_token_idx)  # [L_kept, H]  bf16

        # ---- 3. Per-rank counts, exchange them ----
        # Counts must reflect ONLY kept replicas (post-drop). Using sort_perm —
        # which addresses kept rows after drop, or all rows when drop is off —
        # makes this automatic.
        send_counts_t = (
            torch.bincount(target_rank[sort_perm], minlength=N).to(torch.int32)
            if sort_perm.numel() > 0
            else torch.zeros(N, dtype=torch.int32, device=device)
        )
        recv_counts_t = torch.empty_like(send_counts_t)
        if N > 1:
            dist.all_to_all_single(recv_counts_t, send_counts_t, group=get_ep_group())
        else:
            recv_counts_t.copy_(send_counts_t)
        self._debug_sync("count_a2a")

        # Host sync — REQUIRED to feed Python int lists to all_to_all_single.
        # This is what makes EP-HT incompatible with cuda graph.
        send_counts = send_counts_t.tolist()
        recv_counts = recv_counts_t.tolist()
        if self.overlap_plan.enabled:
            overlap_runtime_stats.record_dispatch(self.layer_id, send_counts, self.rank)
        if len(send_counts) != N or len(recv_counts) != N:
            raise RuntimeError(
                f"EP-HT split vector length mismatch: send={len(send_counts)}, "
                f"recv={len(recv_counts)}, ep_size={N}"
            )
        expected_local_replicas = (
            0
            if (self.runtime_mode == "vllm_dp_ep" and not self.is_source_leader)
            else int(sort_perm.numel())
        )
        if int(sum(send_counts)) != expected_local_replicas:
            raise RuntimeError(
                f"EP-HT send split sum {sum(send_counts)} != kept replicas {expected_local_replicas}"
            )
        total_recv = int(sum(recv_counts))
        recv_capacity = self.sorted_token_ids_buf.size(0) - (E_local + 1) * (self.BLOCK_M - 1)
        if total_recv > recv_capacity:
            raise RuntimeError(
                f"EP-HT total_recv={total_recv} exceeds per-rank workspace capacity "
                f"{recv_capacity}; increase max_num_batched_tokens capacity or reduce batch size"
            )

        # ---- 4. Variable-size all_to_all on the three payloads ----
        recv_hidden = torch.empty(total_recv, H, dtype=dtype, device=device)
        recv_local_eid = torch.empty(total_recv, dtype=torch.int32, device=device)
        recv_topk_w = torch.empty(total_recv, dtype=torch.float32, device=device)

        if N > 1:
            ep_grp = get_ep_group()
            dist.all_to_all_single(
                recv_hidden, sorted_hidden,
                output_split_sizes=recv_counts, input_split_sizes=send_counts,
                group=ep_grp,
            )
            dist.all_to_all_single(
                recv_local_eid, sorted_local_eid,
                output_split_sizes=recv_counts, input_split_sizes=send_counts,
                group=ep_grp,
            )
            dist.all_to_all_single(
                recv_topk_w, sorted_topk_w,
                output_split_sizes=recv_counts, input_split_sizes=send_counts,
                group=ep_grp,
            )
        else:
            recv_hidden.copy_(sorted_hidden)
            recv_local_eid.copy_(sorted_local_eid)
            recv_topk_w.copy_(sorted_topk_w)
        self._debug_sync("payload_a2a")

        # ---- 5. Build standard fused_moe metadata on recv tokens ----
        # Treat as K=1 routing: each received row goes to exactly one local expert.
        recv_topk_ids = recv_local_eid.view(-1, 1)                # [total_recv, 1]
        # Inner kernel applies weights via topk_weights * row_output. We pass 1.0
        # here and let CombineEPHT apply the real weights after the reverse a2a
        # (saves precision; the 1.0 weight just means "don't reweight in kernel").
        recv_topk_weights = torch.ones_like(recv_topk_w).view(-1, 1)

        # If total_recv is 0 (no tokens routed to this rank's experts), fall back
        # to a no-op: skip moe_align_block_size and let CombineEPHT see an empty
        # expert_out.
        if total_recv > 0:
            self.num_tokens_post_padded.zero_()
            # sgl_kernel reserves internal expert slot 0 as a padding/filter
            # sentinel and maps real expert i to slot i + 1. Pass E_local + 1
            # so the highest local expert id is represented.
            moe_align_block_size(
                recv_topk_ids,
                E_local + 1,
                self.BLOCK_M,
                self.sorted_token_ids_buf,
                self.expert_ids_buf,
                self.num_tokens_post_padded,
                self.cumsum_buffer,
                True,  # pad_sorted_token_ids: prevent stale padding from being consumed
            )
        self._debug_sync("moe_align")

        return TokMetaEPHT(
            recv_hidden=recv_hidden,
            recv_topk_ids=recv_topk_ids,
            recv_topk_weights=recv_topk_weights,
            sorted_token_ids=self.sorted_token_ids_buf,
            expert_ids=self.expert_ids_buf,
            num_tokens_post_padded=self.num_tokens_post_padded,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            sort_perm=sort_perm,
            send_counts=send_counts,
            recv_counts=recv_counts,
            T_local=T,
            is_source_leader=self.is_source_leader,
            source_leader_global_rank=self.source_leader_global_rank,
        )
