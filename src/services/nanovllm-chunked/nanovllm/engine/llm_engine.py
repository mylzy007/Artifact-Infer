import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # 1. 调度与计算
        scheduled_seqs = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", scheduled_seqs)
        
        # 2. 原始后处理（把非 -1 的 token 追加到各个 seq 中）
        seqs = [seq for seq, _ in scheduled_seqs]
        self.scheduler.postprocess(seqs, token_ids)
        
        # --- 【新增：强制拦截与内存回收逻辑】 ---
        # Qwen 系列有两个常见的结束符：151643 (Base) 和 151645 (Instruct)
        eos_ids = {self.tokenizer.eos_token_id, 151643, 151645}
        
        for seq in seqs:
            # 兼容不同写法的完成状态判断
            is_finished = getattr(seq, "is_finished", False)
            
            if not is_finished and seq.num_tokens > seq.num_prompt_tokens:
                # 从你 Sequence 类的 token_ids 里拿最后一个词
                last_token = seq.token_ids[-1]
                
                # 计算已经生成了多少个新词
                generated_len = seq.num_tokens - seq.num_prompt_tokens
                
                # 触发刹车：如果不忽略 EOS 且命中结束符，或达到了最大长度
                hit_eos = (not seq.ignore_eos) and (last_token in eos_ids)
                hit_max_len = generated_len >= seq.max_tokens
                
                if hit_eos or hit_max_len:
                    # 标记完成（根据你的类，可能需要改 status）
                    seq.is_finished = True  
                    if hasattr(seq, "status"):
                        from nanovllm.engine.sequence import SequenceStatus
                        seq.status = SequenceStatus.FINISHED
                    
                    # 极其重要：通知 Scheduler 释放物理显存块！
                    if hasattr(self.scheduler, "free"):
                        self.scheduler.free(seq)
                    elif hasattr(self.scheduler, "free_seq"):
                        self.scheduler.free_seq(seq)
        # --- 【拦截逻辑结束】 ---

        # 3. 收集完成的请求
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        
        # 4. 动态统计真正的 Prefill 和 Decode 吞吐量
        prefill_tokens = sum(chunk_size for _, chunk_size in scheduled_seqs if chunk_size > 1)
        decode_tokens = sum(chunk_size for _, chunk_size in scheduled_seqs if chunk_size == 1)
        
        return outputs, prefill_tokens, decode_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
            
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        
        while not self.is_finished():
            t = perf_counter()
            # 接收更细粒度的 throughput 统计
            output, prefill_tokens, decode_tokens = self.step()
            elapsed_time = perf_counter() - t
            
            if use_tqdm:
                # 分别计算吞吐量，即使在混合 Batch 中也能准确显示
                if prefill_tokens > 0:
                    prefill_throughput = prefill_tokens / elapsed_time
                if decode_tokens > 0:
                    decode_throughput = decode_tokens / elapsed_time
                    
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)} tok/s",
                    "Decode": f"{int(decode_throughput)} tok/s",
                })
                
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
                    
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        
        if use_tqdm:
            pbar.close()
            
        return outputs