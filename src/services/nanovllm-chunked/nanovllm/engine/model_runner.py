import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        
        chunk_size = min(max_num_batched_tokens, max_model_len)
        num_seqs = max(1, min(max_num_batched_tokens // chunk_size, self.config.max_num_seqs))
        
        # [核心修正]：手动将 dummy 序列的进度设置为 chunk_size
        # 这样 ModelRunner 才能通过 end_idx - chunk_size 倒推出 0
        seqs = []
        for _ in range(num_seqs):
            seq = Sequence([0] * chunk_size)
            seq.num_computed_tokens = chunk_size  # 模拟 Scheduler 的提前更新
            seqs.append((seq, chunk_size))
            
        self.run(seqs)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        if not seqs:
            return None
        max_len = max((len(seq.block_table) for seq in seqs), default=0)
        if max_len == 0:
            return None
        block_tables = [seq.block_table + [0] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    # 统一的 prepare_inputs，取代 prefill 和 decode
    def prepare_inputs(self, scheduled_seqs: list[tuple[Sequence, int]]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        context_lens = []
        
        logits_indices = []  # 用于只抽取每个 chunk 最后一个 token 去算 Logits
        discard_mask = []    # 丢弃掩码：标记哪些是中间 chunk

        is_decode_batch = True  # 如果所有 chunk_size 都是 1，说明是纯 Decode 批次，可以开 CUDA Graph

        for seq, chunk_size in scheduled_seqs:
            start_idx = seq.num_computed_tokens - chunk_size
            end_idx = min(start_idx + chunk_size, seq.num_tokens)
            actual_chunk = end_idx - start_idx  # 算出实际真正要算的长度 (比如 4)
            
            if actual_chunk != 1:
                is_decode_batch = False

            input_ids.extend(seq[start_idx:end_idx])
            positions.extend(list(range(start_idx, end_idx)))
            
            # 以下全部用 actual_chunk 替代原来的 chunk_size
            cu_seqlens_q.append(cu_seqlens_q[-1] + actual_chunk)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end_idx)
            
            max_seqlen_q = max(max_seqlen_q, actual_chunk)
            max_seqlen_k = max(max_seqlen_k, end_idx)
            
            context_lens.append(end_idx)

            for i in range(start_idx, end_idx):
                if not seq.block_table:
                    slot_mapping.append(-1)
                else:
                    block_number = seq.block_table[i // self.block_size]
                    block_offset = i % self.block_size
                    slot_mapping.append(block_number * self.block_size + block_offset)

            logits_indices.append(cu_seqlens_q[-1] - 1)
            discard_mask.append(end_idx < seq.num_tokens)
            print(f"endidx: {end_idx}, num_tokens: {seq.num_tokens}, discard mask: {discard_mask}")

        # 统一准备 block_tables，供 PagedAttention 跨块读取历史 KV
        seqs = [seq for seq, _ in scheduled_seqs]
        block_tables = self.prepare_block_tables(seqs)

        # 张量化并移至 GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        logits_indices = torch.tensor(logits_indices, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

        # is_prefill 参数直接传 not is_decode_batch。
        # 底层算子只要收到 block_tables，就能自动执行 "Paged Prefill" 或 "Paged Decode"
        set_context(
            not is_decode_batch, 
            cu_seqlens_q, cu_seqlens_k, 
            max_seqlen_q, max_seqlen_k, 
            slot_mapping, context_lens, block_tables
        )
        
        return input_ids, positions, logits_indices, discard_mask, is_decode_batch

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_decode_batch: bool, logits_indices: torch.Tensor):
        if not is_decode_batch or self.enforce_eager or input_ids.size(0) > 512:
            hidden_states = self.model(input_ids, positions)
            
            # 删除了二次切片，直接把完整的 hidden_states 交给原模型的 lm_head 处理
            return self.model.compute_logits(hidden_states)
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            # 在纯 Decode 中，bs 就是请求数，前 bs 个 output 就是我们需要的
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, scheduled_seqs: list[tuple[Sequence, int]]) -> list[int]:
        # 准备所有输入张量，包括用于切片的 logits_indices
        input_ids, positions, logits_indices, discard_mask, is_decode_batch = self.prepare_inputs(scheduled_seqs)
        
        seqs = [seq for seq, _ in scheduled_seqs]
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # 拿到包含了所有请求 Token 的全局 logits 面条
        logits = self.run_model(input_ids, positions, is_decode_batch, logits_indices)
        
        if self.rank == 0:
            # 【修复 4】：精准切片逻辑
            # 如果面条长度大于请求数量，说明这是一个包含了多 Token 的连续批处理 (Prefill)
            if logits.size(0) > len(logits_indices):
                sampled_logits = logits[logits_indices]
            else:
                sampled_logits = logits

            # 采样器现在只会针对每个请求真正的“最后一个词”进行采样
            raw_token_ids = self.sampler(sampled_logits, temperatures).tolist()
            token_ids = []
            
            # 结合 discard_mask，把还在跑 Chunked Prefill 中间块的输出拦截掉
            for token_id, is_discard in zip(raw_token_ids, discard_mask):
                if is_discard:
                    token_ids.append(-1)
                else:
                    token_ids.append(token_id)
            return token_ids
            
        return []

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
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
