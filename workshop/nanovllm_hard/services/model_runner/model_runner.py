import torch
import torch.distributed as dist

import pickle
from src.core.utils import cdiv

from ..config import Config
from ..engine.sequence import Sequence, torch_rotl_uint8

from ..utils.context import (set_context, 
                             get_context, 
                             reset_context, 
                             set_cuda_graph_flag, 
                             init_packed_wise_mask_for_cudagraph)
from ..utils.loader import load_model
from ..utils.logging import get_log, reset_log

from src.core.service import BaseService
from src.core.orchestrator import RegistryOrchestrator
from ..engine.io_struct import SamplingInfo, ModelRunnerOutput

from workshop.nanovllm_hard.artifacts.modeling.models.qwen3 import Qwen3ForCausalLM
from workshop.nanovllm_hard.artifacts.modeling.layers.sampler import Sampler
from workshop.nanovllm_hard.artifacts.cache_mngr.headwise import CacheManager
from workshop.nanovllm_hard.artifacts.attention_backend.flashinfer_attention_headflatten import Attention as FlashinferAttention

import os

from enum import Enum


class RunningStage(Enum):
    WARMUP = 1
    INFERENCE = 2


stage = RunningStage.WARMUP

import dataclasses

@dataclasses.dataclass
class SeqMetadata:
    num_blocks_head: torch.Tensor
    headwise_mask_layer_transpose: torch.Tensor
    next_mask: torch.Tensor
    
    def __init__(self, num_blocks_head, headwise_mask_layer_transpose, next_mask):
        self.num_blocks_head = num_blocks_head
        self.headwise_mask_layer_transpose = headwise_mask_layer_transpose
        self.next_mask = next_mask

    @property
    def num_blocks_max_heads(self):
        return self.num_blocks_head.max().item()

class ModelRunner(BaseService):
    @property
    def name(self):
        return f"ModelRunner-Rank{self.rank}"

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        hf_config = config.hf_config
        self.query_window_size = config.query_window_size
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        
        self.rank = dist.get_rank()
        self.world_size = config.tensor_parallel_size
        
        self.num_layers = hf_config.num_hidden_layers
        self.num_kv_heads = hf_config.num_key_value_heads // self.world_size    
        
        self.trace_count = 0
                
        torch.set_default_dtype(hf_config.torch_dtype)

        orch = RegistryOrchestrator()
        
        attention = orch.add(FlashinferAttention(config))
        self.model = orch.add(Qwen3ForCausalLM(hf_config))
        
        load_model(self.model, config.model)

        self.p_attn = config.p_attn
        
        self.seq_metadata_dict = {}
        
        if self.config.compress_method == "none":
            from workshop.nanovllm_hard.artifacts.compression.nocompress import NoCompress
            self.compressor = NoCompress(config, window_size=config.query_window_size, budget=config.layer_budget)
        elif self.config.compress_method == "rkv":
            if self.config.if_fake_compress:
                from workshop.nanovllm_hard.artifacts.compression.RKV_topp import RKV
            else:
                from workshop.nanovllm_hard.artifacts.compression.RKV_topp_rewrite import RKV
            self.compressor = RKV(config, window_size=config.query_window_size, budget=config.layer_budget, upper_budget=config.layer_upper_budget)
        elif self.config.compress_method == "snapkv":
            if self.config.if_fake_compress:
                from workshop.nanovllm_hard.artifacts.compression.snapKV_topp import SnapKV
            else:
                from workshop.nanovllm_hard.artifacts.compression.snapKV_topp_rewrite import SnapKV
            self.compressor = SnapKV(config, window_size=config.query_window_size, budget=config.layer_budget, upper_budget=config.layer_upper_budget)
        elif self.config.compress_method == "vanilla": 
            if self.config.if_fake_compress:
                from workshop.nanovllm_hard.artifacts.compression.vanilla_topp import VanillaToppKV
            else:
                from workshop.nanovllm_hard.artifacts.compression.vanilla_topp_rewrite import VanillaToppKV
            self.compressor = VanillaToppKV(
                config, window_size=config.query_window_size, budget=config.layer_budget, upper_budget=config.layer_upper_budget
            )
        else:
            raise ValueError(f"Unknown compress method: {self.config.compress_method}")
        
        cache_mngr = orch.add(CacheManager(config, self.compressor))
        
        orch.register(attention, "prepare_metadata_for_attn_prefill", cache_mngr)
        orch.register(attention, "prepare_metadata_for_attn_decode", cache_mngr)
        orch.register(attention, "init_forward_metadata_capture_cuda_graph", cache_mngr)
        orch.register(attention, "init_forward_metadata_replay_cuda_graph", cache_mngr)

        orch.register(cache_mngr, "allocate_page_indices", self)
        orch.register(cache_mngr, "allocate_page_indices_cudagraph", self)
        orch.register(cache_mngr, "read_and_store_cache", self)
        orch.register(cache_mngr, "update_indices", self)
        orch.register(cache_mngr, "update_indices_capture", self)
        orch.register(cache_mngr, "update_indices_replay", self)
        orch.register(cache_mngr, "organize", self)
        orch.register(cache_mngr, "cu_seqs", self)
        
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                 orch.register(attention, "attn", module)
        
        orch.finalize()
        
        self.sampler = Sampler()
        global stage
        stage = RunningStage.WARMUP
        # self.warmup_model()
        stage = RunningStage.INFERENCE

        self.allocate_kv_cache()
        init_packed_wise_mask_for_cudagraph(
            hf_config.num_hidden_layers, 
            config.max_num_seqs, 
            config.max_model_len
        )
        self.graphs = {}
        self.graph_pool = None
        if not self.enforce_eager:
            set_cuda_graph_flag()
            self.capture_cudagraph()
            print("cuda graph captured")
            cache_mngr.reset()
        
        reset_log()
        
    def cleanup(self):
        del self.graphs, self.graph_pool
        
    def save_num_blocks(self):
        os.makedirs(self.config.log_path, exist_ok=True)
        global_log = get_log()
        if 1 - self.p_attn >= 0.01:
            p_attn_string = f"{int(self.p_attn * 100)}"
        else:
            p_attn_string = f"{int(self.p_attn * 1000)}"
        
        num_blocks_head = getattr(global_log, "num_blocks_head", None)
        if num_blocks_head is not None:
            save_path = os.path.join(self.config.log_path, f"{self.config.attn_reduce_method}_num_blocks_head_p{p_attn_string}_budget_{self.config.layer_budget}_steps_{self.config.steps_between_cache_compressions}.pt")
            torch.save(num_blocks_head, save_path)
    
    def save_num_topp(self):
        os.makedirs(self.config.log_path, exist_ok=True)
        global_log = get_log()
        assert getattr(self, "p_attn", None) is not None, "please specify p_attn when logging selected_topp_indices"
        if 1 - self.p_attn >= 0.01:
            p_attn_string = f"{int(self.p_attn * 100)}"
        else:
            p_attn_string = f"{int(self.p_attn * 1000)}"
        num_topp = getattr(global_log, "num_topp_log", None)
        if num_topp is not None:
            save_path = os.path.join(self.config.log_path, f"{self.config.attn_reduce_method}_num_topp_p{p_attn_string}_{self.config.layer_budget}_{not self.config.if_fake_compress}.pt")
            torch.save(num_topp, save_path)
        selected_topp_indices = getattr(global_log, "selected_topp_indices", None)
        if selected_topp_indices is not None:
            save_path = os.path.join(self.config.log_path, f"{self.config.attn_reduce_method}_selected_topp_indices_p{p_attn_string}_{self.config.layer_budget}_{not self.config.if_fake_compress}.pt")
            torch.save(selected_topp_indices, save_path)
        temperatures = getattr(global_log, "temperatures", None)
        if temperatures is not None:
            save_path = os.path.join(self.config.log_path, f"{self.config.attn_reduce_method}_temperatures_topp_p{p_attn_string}_{self.config.layer_budget}_{not self.config.if_fake_compress}.pt")
            torch.save(temperatures, save_path)
        
    def save_lse_log(self):
        global_log = get_log()
        lse = getattr(global_log, "lse_log", None)
        if lse is not None:
            save_path = os.path.join(self.config.log_path, f"{self.config.attn_reduce_method}_lse_{self.config.layer_budget}_{not self.config.if_fake_compress}.pt")
            if not os.path.exists(self.config.log_path):
                os.makedirs(self.config.log_path)
            torch.save(lse, save_path)
        lse_topp = getattr(global_log, "lse_topp_log", None)
        if lse_topp is not None:
            assert getattr(self, "p_attn", None) is not None, "please specify p_attn when logging selected_topp_indices"
            if 1 - self.p_attn >= 0.01:
                p_attn_string = f"{int(self.p_attn * 100)}"
            else:
                p_attn_string = f"{int(self.p_attn * 1000)}"
            save_path = os.path.join(self.config.log_path, f"{self.config.attn_reduce_method}_lse_topp_p{p_attn_string}_{self.config.layer_budget}_{not self.config.if_fake_compress}.pt")
            if not os.path.exists(self.config.log_path):
                os.makedirs(self.config.log_path)
            torch.save(lse_topp, save_path)

    def reset(self):
        if hasattr(self.compressor, "reset_indices"):
            self.compressor.reset_indices()
        reset_log()
        reset_context()
    
    # @torch.inference_mode()
    def compress(self):
        for module in self.model.modules():
            if (
                hasattr(module, "k_cache")
                and hasattr(module, "v_cache")
                and hasattr(module, "q_cache")
            ):                
                self.read_and_store_cache(
                    module.q_cache, module.k_cache, module.v_cache, module.layer_id
                )
        
        self.organize()
        
        for seq in self.cu_seqs:
            if self.world_size > 0:
                num_blocks = [torch.zeros(self.num_kv_heads, device="cuda", dtype=torch.int64) for _ in range(self.world_size)]
                # print(f"before all_gather")
                dist.all_gather(num_blocks, seq.num_blocks_head)
                num_blocks_max_heads = torch.cat(num_blocks, dim=0).max(dim=0).values.item()
                # print(f"[rank {self.rank}] num_blocks_max_heads after compression:", num_blocks_max_heads)
            else:
                num_blocks_max_heads = seq.num_blocks_max_heads
            # reduce across all ranks to get num_blocks_max_heads
            if dist.get_rank() == 0:
                self.update_blocks_post_compression(seq, num_blocks_max_heads)
        self.sync_metadata(self.cu_seqs)
        
        # self.trace_count += 1
        # prof.export_chrome_trace(f"cpu_trace_{self.trace_count}.json")

    def save_compress_distribution(self, steps):
        save_path = os.path.join(self.config.log_path, f"compress_distribution_{steps}.pt")
        if not os.path.exists(self.config.log_path):
            os.makedirs(self.config.log_path)
        torch.save(self.compressor.cache_pos_to_seq_pos, save_path)

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
        config = self.config
        hf_config = config.hf_config
        try:
            self.q_cache = torch.zeros(
                hf_config.num_hidden_layers,
                self.config.max_num_seqs,
                self.query_window_size,
                hf_config.num_attention_heads,
                hf_config.head_dim,
                dtype=hf_config.torch_dtype,
            )
        except:
            raise ValueError(
                'Not enough memory for q_cache, try to lower memory occupation of other process or lower "config.max_num_seqs"'
            )
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
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks, 
            num_kv_heads,
            self.block_size,
            # num_kv_heads,
            1, 
            hf_config.head_dim,
        )

        if config.if_compress_kvcache and not config.if_fake_compress:
            config.lazy_max_num_seqs = cdiv(config.num_kvcache_blocks, (config.layer_budget + config.steps_between_cache_compressions))
        else:
            config.lazy_max_num_seqs = cdiv(config.num_kvcache_blocks, config.max_model_len) 
        
        layer_id = 0
        for module in self.model.modules():
            if (
                hasattr(module, "k_cache")
                and hasattr(module, "v_cache")
                and hasattr(module, "q_cache")
            ):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                module.q_cache = self.q_cache[layer_id]
                layer_id += 1

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        query_window_pos = []
        query_slot_mapping = []

        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # prepare query window metadata
            query_window_pos.extend(
                list(
                    range(seq.num_tokens - seq.query_window_num_tokens, seq.num_tokens)
                )
            )
            query_slot_mapping.extend(
                list(
                    range(
                        seq.query_block_id * self.query_window_size,
                        seq.query_block_id * self.query_window_size
                        + seq.query_window_num_tokens,
                    )
                )
            )

            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks_max_heads):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks_max_heads - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
                
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

        query_slot_mapping = torch.tensor(
            query_slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        query_window_pos = torch.tensor(
            query_window_pos, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            query_slot_mapping,
            query_window_pos,
        )
        
        self.allocate_page_indices(seqs)
        self.update_indices()
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        query_slot_mapping = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )

            # prepare query window metadata
            query_slot_mapping.append(
                seq.query_block_id * self.query_window_size
                + seq.last_query_window_index
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

        query_slot_mapping = torch.tensor(
            query_slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            query_slot_mapping=query_slot_mapping,
        )

        # self.allocate_page_indices(seqs)
        self.decode_time = 0
        self.decode_time += 1
        self.allocate_page_indices(seqs)
        if not self.enforce_eager and stage != RunningStage.WARMUP:
            # cuda_graph enabled
            # self.allocate_page_indices_cudagraph(seqs)
            self.update_indices_replay(bs=len(seqs))
        else:
            
            self.update_indices()
        return input_ids, positions

    # def prepare_sample(self, seqs: list[Sequence]):
    #     temperatures = []
    #     for seq in seqs:
    #         temperatures.append(seq.temperature)
    #     temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
    #     return temperatures

    def prepare_sample(self, seqs: list[Sequence]):
        return SamplingInfo.from_sequence(seqs)

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
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
            graph_vars["query_slot_mapping"][:bs] = context.query_slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def create_metadata_to_seq(self, metadata: SeqMetadata, num_blocks):
        for i in range(num_blocks):
            if i % 8 == 0 and i != 0:
                metadata.headwise_mask_layer_transpose = torch.cat(
                    [metadata.headwise_mask_layer_transpose, torch.zeros((self.num_layers, self.num_kv_heads, 1), device="cuda", dtype=torch.uint8)], dim=2
                )
            metadata.headwise_mask_layer_transpose[:, :, i // 8] += metadata.next_mask.to(metadata.headwise_mask_layer_transpose.device)
            metadata.next_mask = torch_rotl_uint8(metadata.next_mask, 1)

    def append_metadata_to_seq(self, metadata: SeqMetadata):
        metadata.num_blocks_head += 1
        # there must be at least one token after prefilling (allocate)
        if (metadata.num_blocks_max_heads - 1) // 8 >= metadata.headwise_mask_layer_transpose.shape[2]:
            metadata.headwise_mask_layer_transpose = torch.cat(
                [metadata.headwise_mask_layer_transpose, torch.zeros((self.num_layers, self.num_kv_heads, 1), device="cuda", dtype=torch.uint8)], dim=2
            )
        # print(metadata.headwise_mask_layer_transpose[0])
        if self.config.if_fake_compress:
            metadata.headwise_mask_layer_transpose[:, torch.arange(0, self.num_kv_heads), (metadata.num_blocks_head - 1) // 8] += metadata.next_mask.to(metadata.headwise_mask_layer_transpose.device)
        else:
            metadata.headwise_mask_layer_transpose[:, torch.arange(0, self.num_kv_heads), (metadata.num_blocks_max_heads - 1) // 8] += metadata.next_mask.to(metadata.headwise_mask_layer_transpose.device)

        metadata.next_mask = torch_rotl_uint8(metadata.next_mask, 1)

    def prepare_seqs_metadata(self, seq: Sequence, is_prefill: bool):
        if is_prefill:
            self.seq_metadata_dict[seq.seq_id] = SeqMetadata(
                num_blocks_head = torch.ones((seq.num_kv_heads,), device="cuda", dtype=torch.int32) * seq.num_tokens, 
                headwise_mask_layer_transpose = torch.zeros((seq.num_layers, seq.num_kv_heads, 1), device="cuda", dtype=torch.uint8), 
                next_mask = torch.ones((seq.num_kv_heads,), device="cpu", dtype=torch.uint8)
            )
            self.create_metadata_to_seq(self.seq_metadata_dict[seq.seq_id], seq.num_blocks_max_heads)
        else:
            self.append_metadata_to_seq(self.seq_metadata_dict[seq.seq_id])
        
        seq.num_blocks_head = self.seq_metadata_dict[seq.seq_id].num_blocks_head
        seq.headwise_mask_layer_transpose = self.seq_metadata_dict[seq.seq_id].headwise_mask_layer_transpose
        seq.next_mask = self.seq_metadata_dict[seq.seq_id].next_mask
        
        # print(seq.num_blocks_head)
        # print(seq.next_mask)
        # print("-" * 100)

    @torch.inference_mode()
    def prepare_and_run(self, seqs, is_prefill: bool):
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        # temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        sampling_infos = (
            self.prepare_sample(seqs).to(input_ids.device) if self.rank == 0 else None
        )
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = (
            self.sampler(logits, sampling_infos).tolist() if self.rank == 0 else None
        )
        return token_ids
    
    def sync_metadata(self, seqs):
        for seq in seqs:
            metadata = self.seq_metadata_dict[seq.seq_id]
            metadata.num_blocks_head = seq.num_blocks_head
            metadata.headwise_mask_layer_transpose = seq.headwise_mask_layer_transpose
            metadata.next_mask = seq.next_mask

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        for seq in seqs:
            self.prepare_seqs_metadata(seq, is_prefill)
        
        token_ids = self.prepare_and_run(seqs, is_prefill)
        # reset_context()
        # return ModelRunnerOutput(token_ids=token_ids, logits=logits)
        return ModelRunnerOutput(token_ids=token_ids, logits=None)

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        query_slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)

        seqs = [Sequence.for_capture([0], self.num_layers, self.num_kv_heads) for _ in range(max_bs)]
        
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = list(range(1, 52))
        # self.graph_bs = [1]

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                True,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                query_slot_mapping=query_slot_mapping[:bs],
            )
            # self.init_forward_metadata_capture_cuda_graph(bs, seq_lens[:bs], cu_page_indices)
            self.allocate_page_indices(seqs[:bs])
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                query_slot_mapping=query_slot_mapping[:bs],
            )
            # print("-" * 100)
            # print(get_context().packed_headwise_mask.shape)
            # print("-" * 100)

            self.allocate_page_indices(seqs[:bs])
            self.update_indices_capture(bs)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            
            reset_context()
            init_packed_wise_mask_for_cudagraph(
                hf_config.num_hidden_layers,
                config.max_num_seqs,
                config.max_model_len,
            )
            

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            query_slot_mapping=query_slot_mapping,
            context_lens=context_lens,
            outputs=outputs,
        )
