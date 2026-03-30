import atexit
from dataclasses import fields
from itertools import count
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from ..config import Config
from ..sampling_params import SamplingParams
from .sequence import Sequence

from ..utils.logging import get_log, set_log, LogCollector
from src.core.service import BaseService
import numpy as np

class LLMEngine(BaseService):

    def __init__(self, model, **kwargs):
        super().__init__()
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = config = Config(model, **config_kwargs)
        if not config.enforce_eager and config.if_log_lse_in_attn:
            print("Warning: LSE cannot be logged when cuda graph is enabled.") 
        
        self.log_collector = LogCollector()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        
        self.counter = count()
                        
        self.log_steps = []
        self.cur_step = 0
        
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence.from_prompt(next(self.counter), prompt, sampling_params, self.config.kvcache_block_size)
        self.add(seq)

    def step(self):
        seqs, is_prefill = self.schedule()        
        modelrunner_output = self.run(seqs, is_prefill)
        self.postprocess(seqs, modelrunner_output.token_ids, modelrunner_output.logits)
        if not self.is_finished() and self.config.if_compress_kvcache and self.cur_step % self.config.steps_between_cache_compressions == 0:
            # print(f"[DEBUG] Compressing KVCaches cur_step: {self.cur_step}")
            self.compress()
            # self.model_runner.call("save_compress_distribution", self.cur_step)
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.logits) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        self.cur_step += 1
                
        return outputs, num_tokens
    
    def save_logits_log(self):
        save_path = f"{self.config.log_path}/logits_log.npy"
        log = get_log()
        logits_log = log.logits_log
        np.save(logits_log.as_dict(), save_path)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        log_steps: list[int] | None = None, 
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
            output, num_tokens = self.step()
            # self.log_collector.append(perf_counter(), get_log().occupied_pages)
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                    "Occupied Pages": get_log().occupied_pages,
                })
            for seq_id, token_ids, logits in output:
                outputs[seq_id] = (token_ids, logits)
            pbar.update(1)            
        # NOTE
        # if self.config.if_log_num_topp:
        #     self.model_runner.call("save_num_topp")
        #     self.model_runner.call("save_lse_log")
        # self.log_collector.save(self.config.log_path)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids, "logits": logits} for token_ids, logits in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
