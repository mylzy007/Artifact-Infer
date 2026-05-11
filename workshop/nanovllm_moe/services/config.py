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
    # Optional: trim model to first N layers (useful for testing big MoE on small GPUs;
    # generated text won't be coherent but the pipeline is exercised end-to-end).
    num_hidden_layers_override: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        # assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.num_hidden_layers_override > 0:
            self.hf_config.num_hidden_layers = self.num_hidden_layers_override
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
