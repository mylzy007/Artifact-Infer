import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 65536
    max_num_seqs: int = 512
    max_model_len: int = 40960
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    enforce_eager: bool = True
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 1
    num_kvcache_blocks: int = -1

    # MoE: which inner-kernel implementation to use.
    #   "torch"        — slow reference, single-rank only.
    #   "triton"       — Triton fused_moe (block-tiled), single-rank.
    #   "ep_ll_torch"  — EP-LL with torch-bmm masked grouped GEMM (multi-rank reference).
    #   "ep_ll_triton" — EP-LL with Triton masked grouped GEMM (multi-rank fast,
    #                    cuda-graph capturable when paired with the triton dispatch).
    #   "ep_ht"        — EP-HT (high throughput): ragged a2a + standard triton_fused_moe.
    #                    Eager-only (variable-size NCCL needs Python int splits).
    moe_impl: str = "triton"
    # Static tile size used by Triton fused_moe + sgl_kernel.moe_align_block_size.
    moe_block_size_m: int = 64
    # EP-LL only: M_max per (rank, local_expert) bucket. -1 means auto-size as
    # ceil(T_cap * K / (N_ranks * E_local)) * 4 (4x imbalance budget).
    moe_ll_m_max: int = -1
    # EP-LL fixed-capacity overflow policy:
    #   "drop"  — LL-style behavior: replicas beyond M_max are not written.
    #   "error" — debug/correctness guard in eager mode; graph capture still
    #             cannot raise from host-side checks.
    moe_ll_overflow_policy: str = "drop"
    # Expert placement shared by EP dispatch and expert weight loading.
    moe_expert_placement: str = "contiguous"
    moe_expert_placement_seed: int = 0
    moe_expert_placement_path: str | None = None
    moe_expert_overlap_enabled: bool = False
    moe_expert_overlap_path: str | None = None
    moe_expert_overlap_strategy: str = "hybrid"
    # Phase 4: token-replica drop. drop_policy="none" disables (default).
    moe_drop_policy: str = "none"
    moe_drop_rate: float = 0.0
    moe_drop_seed: int = 0
    # Phase 4 P1: router K_eff. When > 0 and < model top_k, dispatch only
    # routes the top K_eff branches per token (no keep_mask machinery needed).
    moe_router_keff: int = 0
    # Optional: trim model to first N layers (useful for testing big MoE on small GPUs;
    # generated text won't be coherent but the pipeline is exercised end-to-end).
    num_hidden_layers_override: int = -1
    # Runtime mode:
    #   legacy_tp_ep   — existing layout: rank = ep_rank * tp_rank, EP subgroup can be strided.
    #   vllm_dp_ep     — owner-based DP+EP: TP within each DP replica, EP over all ranks.
    #   owner_local_ep — local-token owner + world expert parallel:
    #                    tp=1, dp=world, ep=world. Each rank keeps only its local
    #                    sequences/tokens and local expert shard.
    moe_runtime_mode: str = "legacy_tp_ep"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        # assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert 1 <= self.data_parallel_size <= 8
        assert self.moe_runtime_mode in ("legacy_tp_ep", "vllm_dp_ep", "owner_local_ep")
        if self.moe_runtime_mode == "owner_local_ep":
            assert self.tensor_parallel_size == 1, (
                "owner_local_ep requires tensor_parallel_size == 1"
            )
        if self.moe_expert_overlap_enabled and not self.moe_expert_overlap_path:
            raise ValueError("moe_expert_overlap_enabled requires moe_expert_overlap_path")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.num_hidden_layers_override > 0:
            self.hf_config.num_hidden_layers = self.num_hidden_layers_override
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
