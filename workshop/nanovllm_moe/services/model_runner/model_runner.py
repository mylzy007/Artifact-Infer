from logging import config
import pickle
from xml.parsers.expat import model
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from ..config import Config
from ..engine.sequence import Sequence

from ..utils.context import set_context, get_context, reset_context
from ..utils.loader import load_model

from src.core.service import BaseService
from src.core.orchestrator import RegistryOrchestrator

import itertools

from workshop.nanovllm_moe.artifacts.attention_backend.flashinfer_attention import Attention as FlashinferAttention
from workshop.nanovllm_moe.artifacts.modeling.models.qwen3 import Qwen3ForCausalLM
from workshop.nanovllm_moe.artifacts.modeling.models.qwen3_moe import Qwen3MoeForCausalLM
from workshop.nanovllm_moe.artifacts.modeling.layers.sampler import Sampler
from workshop.nanovllm_moe.artifacts.modeling.layers.moe import Combine, Dispatch, Experts
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ll import DispatchEPLL
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.experts_ep_ll import ExpertsEPLL
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.combine_ep_ll import CombineEPLL
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ht import DispatchEPHT
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.experts_ep_ht import ExpertsEPHT
from workshop.nanovllm_moe.artifacts.moe_backend import MoeBackend
from workshop.nanovllm_moe.services.utils.parallel import (
    get_dp_leader_global_rank,
    get_dp_rank,
    get_tp_group,
    get_tp_rank,
    is_dp_leader,
    is_owner_local_ep_mode,
)

from enum import Enum


class RunningStage(Enum):
    WARMUP = 1
    INFERENCE = 2


class ModelRunner(BaseService):
    @property
    def name(self):
        return f"ModelRunner-Rank{self.rank}"

    def __init__(self, config: Config):
        super().__init__()
        self.tp_rank = get_tp_rank()
        self.dp_rank = get_dp_rank()
        self.dp_leader = is_dp_leader()
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0
        # Keep `rank` as the TP rank for existing TP-sharded layer assumptions.
        self.rank = self.tp_rank
        self.config = config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.runtime_mode = config.moe_runtime_mode
        
        torch.set_default_dtype(self.config.hf_config.torch_dtype)
        
        orch = RegistryOrchestrator()

        attention = orch.add(FlashinferAttention(config))

        is_moe = getattr(config.hf_config, "model_type", "") == "qwen3_moe"
        is_ep_ll = is_moe and config.moe_impl in ("ep_ll_torch", "ep_ll_triton")
        is_ep_ht = is_moe and config.moe_impl == "ep_ht"
        moe_mode = "ep_ll" if is_ep_ll else ("ep_ht" if is_ep_ht else "single")

        if is_moe:
            hf = config.hf_config
            # MoeBackend must be constructed FIRST in EP-LL mode so M_max is decided
            # before the model layers are created (FusedMoE needs m_max in __init__).
            moe_backend = orch.add(MoeBackend(
                config=config,
                num_experts=hf.num_experts,
                top_k=hf.num_experts_per_tok,
                hidden_size=hf.hidden_size,
                moe_intermediate_size=hf.moe_intermediate_size,
            ))
            self.moe_backend = moe_backend
            m_max = moe_backend.M_max if is_ep_ll else 0
            # ep_ll_torch -> torch dispatch (host loops, eager only).
            # ep_ll_triton -> triton dispatch (cuda-graph compatible).
            ep_ll_dispatch_kernel = "torch" if config.moe_impl == "ep_ll_torch" else "triton"
            self.model = orch.add(Qwen3MoeForCausalLM(
                hf,
                moe_block_size_m=config.moe_block_size_m,
                moe_mode=moe_mode,
                m_max=m_max,
                ep_ll_dispatch_kernel=ep_ll_dispatch_kernel,
                expert_placement=config.moe_expert_placement,
                expert_placement_seed=config.moe_expert_placement_seed,
                expert_placement_path=config.moe_expert_placement_path,
                expert_overlap_enabled=config.moe_expert_overlap_enabled,
                expert_overlap_path=config.moe_expert_overlap_path,
                expert_overlap_strategy=config.moe_expert_overlap_strategy,
                ep_ll_overflow_policy=config.moe_ll_overflow_policy,
                drop_policy=config.moe_drop_policy,
                drop_rate=config.moe_drop_rate,
                drop_seed=config.moe_drop_seed,
                router_keff=config.moe_router_keff,
                combine_compress_keep_frac=config.moe_combine_compress_keep_frac,
                combine_compress_fp8=config.moe_combine_compress_fp8,
                combine_compress_use_l2=config.moe_combine_compress_use_l2,
            ))
        else:
            self.model = orch.add(Qwen3ForCausalLM(config.hf_config))
            moe_backend = None
            self.moe_backend = None

        orch.register(attention, "init_forward_metadata_capture_cuda_graph", self)
        orch.register(attention, "init_forward_metadata_replay_cuda_graph", self)
        orch.register(attention, "prepare_metadata_for_attn_decode", self)
        orch.register(attention, "prepare_metadata_for_attn_prefill", self)

        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                 orch.register(attention, "attn", module)

            if moe_backend is None:
                continue

            # ---- Single-rank wiring ----
            if isinstance(module, Dispatch):
                for name in (
                    "sorted_token_ids_buf", "expert_ids_buf", "num_tokens_post_padded",
                    "cumsum_buffer", "topk_weights_buf", "topk_ids_buf",
                ):
                    orch.register(moe_backend, name, module)
            if isinstance(module, Experts):
                for name in (
                    "intermediate_cache1", "intermediate_cache2", "intermediate_cache3",
                    "run_experts",
                ):
                    orch.register(moe_backend, name, module)
            if isinstance(module, Combine):
                orch.register(moe_backend, "intermediate_cache3", module)

            # ---- EP-LL wiring (parallel branch; same MoeBackend, different attrs) ----
            if isinstance(module, DispatchEPLL):
                for name in (
                    "send_buf", "recv_buf", "original_indices", "local_counts",
                    "topk_weights_buf", "topk_ids_buf",
                    "hidden_recv",  # persistent inner-kernel input workspace
                ):
                    orch.register(moe_backend, name, module)
            if isinstance(module, ExpertsEPLL):
                orch.register(moe_backend, "run_experts_ll", module)
            if isinstance(module, CombineEPLL):
                for name in ("rev_send", "rev_recv"):
                    orch.register(moe_backend, name, module)

            # ---- EP-HT wiring ----
            # DispatchEPHT needs the moe_align_block_size buffers (sized for E_local)
            # and the topk-output workspaces. ExpertsEPHT needs run_experts (which
            # itself uses the intermediate_cache* sized for total_recv).
            if isinstance(module, DispatchEPHT):
                for name in (
                    "sorted_token_ids_buf", "expert_ids_buf", "num_tokens_post_padded",
                    "cumsum_buffer",
                ):
                    orch.register(moe_backend, name, module)
            if isinstance(module, ExpertsEPHT):
                for name in (
                    "intermediate_cache1", "intermediate_cache2", "intermediate_cache3",
                    "run_experts",
                ):
                    orch.register(moe_backend, name, module)

        if moe_backend is not None:
            orch.register(moe_backend, "prepare_metadata_for_moe", self)
        orch.finalize()
                
        self.__post_init__()
    
    def __post_init__(self):
        self._mem_print("[MEM] before load_model")
        load_model(self.model, self.config.model)
        self._mem_print("[MEM] after  load_model")
        self.sampler = Sampler()
        self.allocate_kv_cache()
        self._mem_print("[MEM] after  allocate_kv_cache")

        # self.stage = RunningStage.WARMUP
        # self.warmup_model()
        # print("after warmup")
        self.stage = RunningStage.INFERENCE

        
        self.graphs = {}
        self.graph_pool = None

        if not self.enforce_eager:
            self.capture_cudagraph()
        self._mem_print("[MEM] after  capture_cudagraph")
        print("after capturing cuda graph")

    def _is_vllm_dp_ep(self) -> bool:
        return self.runtime_mode == "vllm_dp_ep" and dist.is_initialized() and dist.get_world_size() > 1

    def _is_owner_local_ep(self) -> bool:
        return is_owner_local_ep_mode() and dist.is_initialized() and dist.get_world_size() > 1

    def _tp_leader_src(self) -> int:
        return get_dp_leader_global_rank(self.dp_rank)

    def _broadcast_tp_object(self, payload):
        if not self._is_vllm_dp_ep():
            return payload
        obj_list = [payload if self.dp_leader else None]
        dist.broadcast_object_list(obj_list, src=self._tp_leader_src(), group=get_tp_group())
        return obj_list[0]

    def _mem_print(self, tag: str) -> None:
        """One-line GPU memory snapshot for this rank.

        Prints peak/current/free along with cuda-graph private-pool usage so
        you can pinpoint which startup phase blew the budget. Set MOE_NO_MEM=1
        to silence."""
        import os
        if os.environ.get("MOE_NO_MEM"):
            return
        free, total = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats()
        cur = stats["allocated_bytes.all.current"]
        peak = stats["allocated_bytes.all.peak"]
        gib = 1024 ** 3
        rk = self.rank
        print(
            f"{tag} | rank{rk}: cur={cur/gib:.2f}GiB peak={peak/gib:.2f}GiB "
            f"free={free/gib:.2f}GiB total={total/gib:.2f}GiB",
            flush=True,
        )
        # default_dtype = torch.get_default_dtype()
        # torch.set_default_device("cpu")
        # torch.set_default_dtype(default_dtype)
    
    def cleanup(self): 
        del self.graphs, self.graph_pool


    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        seqs = [Sequence.from_prompt([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        print(f"Allocating KV cache on cuda {torch.get_default_device()}...")
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * hf_config.head_dim
            * hf_config.torch_dtype.itemsize
        )
        memory_budget_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        # Do not consume all remaining memory with KV cache when the configured
        # workload could only ever touch a much smaller number of blocks.
        max_blocks_per_seq = (config.max_model_len + self.block_size - 1) // self.block_size
        workload_cap_blocks = max(1, int(config.max_num_seqs) * int(max_blocks_per_seq))
        config.num_kvcache_blocks = min(memory_budget_blocks, workload_cap_blocks)
        assert config.num_kvcache_blocks > 0        
        
        self.kv_cache = torch.zeros(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            hf_config.head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        max_len = max(len(seq.block_table) for seq in seqs)
        if max_len > max_num_blocks:
            raise RuntimeError(
                f"block table length {max_len} exceeds max_model_len capacity "
                f"{max_num_blocks}; sequence length likely exceeded max_model_len="
                f"{self.config.max_model_len}"
            )
        for seq in seqs:
            for block_id in seq.block_table:
                if block_id < 0 or block_id >= self.config.num_kvcache_blocks:
                    raise RuntimeError(
                        f"KV block id {block_id} outside allocated range "
                        f"[0, {self.config.num_kvcache_blocks})"
                    )
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        if self._is_vllm_dp_ep() and not self.dp_leader:
            payload = self._broadcast_tp_object(None)
            input_ids = torch.tensor(payload["input_ids"], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            positions = torch.tensor(payload["positions"], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            cu_seqlens_q = torch.tensor(payload["cu_seqlens_q"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            cu_seqlens_k = torch.tensor(payload["cu_seqlens_k"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            slot_mapping = torch.tensor(payload["slot_mapping"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            block_tables = None
            if payload["block_tables"] is not None:
                block_tables = torch.tensor(payload["block_tables"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            set_context(
                True,
                cu_seqlens_q,
                cu_seqlens_k,
                payload["max_seqlen_q"],
                payload["max_seqlen_k"],
                slot_mapping,
                None,
                block_tables,
                num_tokens=int(input_ids.size(0)),
            )
            self.prepare_metadata_for_attn_prefill(seqs)
            if hasattr(self, "prepare_metadata_for_moe"):
                self.prepare_metadata_for_moe(int(input_ids.size(0)))
            return input_ids, positions

        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            if seqlen > self.config.max_model_len:
                raise RuntimeError(
                    f"sequence length {seqlen} exceeds max_model_len={self.config.max_model_len}"
                )
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        cache_slots = self.config.num_kvcache_blocks * self.block_size
        if len(input_ids) > self.config.max_num_batched_tokens:
            raise RuntimeError(
                f"prefill batch has {len(input_ids)} tokens, exceeding "
                f"max_num_batched_tokens={self.config.max_num_batched_tokens}"
            )
        bad_slot = next((slot for slot in slot_mapping if slot < 0 or slot >= cache_slots), None)
        if bad_slot is not None:
            raise RuntimeError(
                f"prefill slot_mapping contains {bad_slot}, outside KV cache slots "
                f"[0, {cache_slots})"
            )
        if seqs and cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
            num_tokens=int(input_ids.size(0)),
        )

        self.prepare_metadata_for_attn_prefill(seqs)
        if hasattr(self, "prepare_metadata_for_moe"):
            self.prepare_metadata_for_moe(int(input_ids.size(0)))

        if self._is_vllm_dp_ep():
            self._broadcast_tp_object(
                {
                    "input_ids": input_ids.cpu().tolist(),
                    "positions": positions.cpu().tolist(),
                    "cu_seqlens_q": cu_seqlens_q.cpu().tolist(),
                    "cu_seqlens_k": cu_seqlens_k.cpu().tolist(),
                    "max_seqlen_q": max_seqlen_q,
                    "max_seqlen_k": max_seqlen_k,
                    "slot_mapping": slot_mapping.cpu().tolist(),
                    "block_tables": block_tables.cpu().tolist() if block_tables is not None else None,
                }
            )

        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        if self._is_vllm_dp_ep() and not self.dp_leader:
            payload = self._broadcast_tp_object(None)
            input_ids = torch.tensor(payload["input_ids"], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            positions = torch.tensor(payload["positions"], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            slot_mapping = torch.tensor(payload["slot_mapping"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            context_lens = torch.tensor(payload["context_lens"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            block_tables = torch.tensor(payload["block_tables"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            set_context(
                False,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                num_tokens=int(input_ids.size(0)),
            )
            if hasattr(self, "prepare_metadata_for_moe"):
                self.prepare_metadata_for_moe(int(input_ids.size(0)))
            if not self.enforce_eager and self.stage != RunningStage.WARMUP and len(seqs) > 0:
                bs = len(seqs)
                page_counts = torch.tensor(
                    [0] + [len(seq.block_table) for seq in seqs],
                    device="cuda",
                    dtype=torch.int32,
                )
                cu_page_indices = torch.tensor(
                    list(itertools.chain(*[seq.block_table for seq in seqs])),
                    device="cuda",
                    dtype=torch.int32,
                )
                kv_last_page_lens = torch.tensor(
                    [seq.last_block_num_tokens for seq in seqs],
                    device="cuda",
                    dtype=torch.int32,
                ).to(torch.int32)
                self.init_forward_metadata_replay_cuda_graph(
                    bs,
                    page_counts,
                    cu_page_indices,
                    kv_last_page_lens,
                )
            else:
                self.prepare_metadata_for_attn_decode(seqs)
            return input_ids, positions

        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            if len(seq) > self.config.max_model_len:
                raise RuntimeError(
                    f"sequence length {len(seq)} exceeds max_model_len="
                    f"{self.config.max_model_len}"
                )
            if not seq.block_table:
                raise RuntimeError("decode sequence has no allocated KV blocks")
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        cache_slots = self.config.num_kvcache_blocks * self.block_size
        bad_slot = next((slot for slot in slot_mapping if slot < 0 or slot >= cache_slots), None)
        if bad_slot is not None:
            raise RuntimeError(
                f"decode slot_mapping contains {bad_slot}, outside KV cache slots "
                f"[0, {cache_slots})"
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs) if seqs else torch.empty((0, 0), dtype=torch.int32, device="cuda")
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            num_tokens=int(input_ids.size(0)),
        )

        if hasattr(self, "prepare_metadata_for_moe"):
            self.prepare_metadata_for_moe(int(input_ids.size(0)))

        if self._is_vllm_dp_ep():
            self._broadcast_tp_object(
                {
                    "input_ids": input_ids.cpu().tolist(),
                    "positions": positions.cpu().tolist(),
                    "slot_mapping": slot_mapping.cpu().tolist(),
                    "context_lens": context_lens.cpu().tolist(),
                    "block_tables": block_tables.cpu().tolist(),
                }
            )

        if not self.enforce_eager and self.stage != RunningStage.WARMUP and len(seqs) > 0:
            # cuda_graph enabled
            bs = len(seqs)
            page_counts = torch.tensor(
                [0] + [len(seq.block_table) for seq in seqs],
                device="cuda",
                dtype=torch.int32,
            )
            cu_page_indices = torch.tensor(
                list(itertools.chain(*[seq.block_table for seq in seqs])),
                device="cuda",
                dtype=torch.int32,
            )
            kv_last_page_lens = torch.tensor(
                [seq.last_block_num_tokens for seq in seqs],
                device="cuda",
                dtype=torch.int32,
            ).to(torch.int32)
            self.init_forward_metadata_replay_cuda_graph(
                bs,
                page_counts,
                cu_page_indices,
                kv_last_page_lens,
            )
        else:
            self.prepare_metadata_for_attn_decode(seqs)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        if input_ids.numel() == 0:
            return self.model.compute_logits(self.model(input_ids, positions))
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs) if (
            self.dp_leader if self._is_vllm_dp_ep()
            else True if self._is_owner_local_ep()
            else self.global_rank == 0
        ) else None

        logits = self.run_model(input_ids, positions, is_prefill)
        if self._is_vllm_dp_ep():
            token_ids = self.sampler(logits, temperatures).tolist() if self.dp_leader else None
            obj_list = [token_ids]
            dist.broadcast_object_list(obj_list, src=self._tp_leader_src(), group=get_tp_group())
            token_ids = obj_list[0]
        elif self._is_owner_local_ep():
            token_ids = self.sampler(logits, temperatures).tolist()
        else:
            token_ids = (
                self.sampler(logits, temperatures).tolist()
                if self.global_rank == 0 else None
            )
            if dist.is_initialized() and dist.get_world_size() > 1:
                obj_list = [token_ids]
                dist.broadcast_object_list(obj_list, src=0)
                token_ids = obj_list[0]

        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        # capture cudagraph for decode only
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)

        seqs = [Sequence.for_capture([0]) for _ in range(max_bs)]
        cu_page_indices = torch.tensor(
            list(itertools.chain(*[seq.block_table for seq in seqs])), device="cuda"
        ).to(torch.int32)
        page_counts = torch.tensor(
            [0] + [len(seq.block_table) for seq in seqs],
            device="cuda",
            dtype=torch.int32,
        )
        kv_last_page_lens = torch.ones(max_bs, dtype=torch.int32, device="cuda")

        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # Capture sizes 1..min(max_bs, 8). max_bs comes from max_num_seqs and may be
        # smaller than 8, in which case our pre-allocated buffers can't hold a larger BS.
        self.graph_bs = list(range(1, min(max_bs, 8) + 1))

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
                num_tokens=bs,
            )
            self.init_forward_metadata_capture_cuda_graph(
                bs, page_counts[: bs + 1], cu_page_indices, kv_last_page_lens[:bs]
            )
            # MoE: prepare moe metadata BEFORE warmup AND capture; dispatch will refill
            # num_tokens_post_padded inside the captured region, so the captured launches
            # bind to that buffer pointer (which is persistent for the engine's lifetime).
            if hasattr(self, "prepare_metadata_for_moe"):
                self.prepare_metadata_for_moe(bs)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
