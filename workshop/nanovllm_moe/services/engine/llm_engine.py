import atexit
from dataclasses import fields
import os
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..config import Config
from ..sampling_params import SamplingParams
from .sequence import Sequence
from .scheduler import Scheduler
from ..model_runner.model_runner import ModelRunner
from ...artifacts.block_mngr.block_manager import BlockManager
from src.core.service import BaseService
from src.core.orchestrator import RegistryOrchestrator
from workshop.nanovllm_moe.services.utils import ep_ll_runtime_stats
from workshop.nanovllm_moe.services.utils.parallel import (
    get_dp_world_size,
    get_dp_rank,
    is_dp_leader,
)

DUMMY_CREATION = os.getenv("DUMMY_CREATION", False)


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return 0
    return int(tensor.numel() * tensor.element_size())


def _ep_ll_workspace_summary(model_runner) -> dict:
    moe_backend = getattr(model_runner, "moe_backend", None)
    if moe_backend is None or not getattr(moe_backend, "is_ep_ll", False):
        return {}
    buffer_names = [
        "send_buf",
        "recv_buf",
        "rev_send",
        "rev_recv",
        "original_indices",
        "local_counts",
        "masked_m_buf",
        "hidden_recv",
        "topk_weights_buf",
        "topk_ids_buf",
        "ll_inter_workspace",
        "ll_out_workspace",
    ]
    bytes_by_buffer = {
        name: _tensor_bytes(getattr(moe_backend, name, None))
        for name in buffer_names
    }
    return {
        "m_max": int(getattr(moe_backend, "M_max", 0)),
        "m_max_config": int(getattr(model_runner.config, "moe_ll_m_max", -1)),
        "overflow_policy": str(getattr(model_runner.config, "moe_ll_overflow_policy", "drop")),
        "world_size": int(getattr(moe_backend, "world_size", 1)),
        "num_experts_global": int(getattr(moe_backend, "E", 0)),
        "num_experts_local": int(getattr(moe_backend, "E_local", 0)),
        "hidden_size": int(getattr(moe_backend, "H", 0)),
        "moe_intermediate_size": int(getattr(moe_backend, "N", 0)),
        "t_cap": int(getattr(moe_backend, "T_cap", 0)),
        "workspace_bytes_by_buffer": bytes_by_buffer,
        "workspace_bytes_total": int(sum(bytes_by_buffer.values())),
    }


def _gather_ep_ll_rank_stats(local_stats: dict) -> list[dict] | None:
    if not local_stats or not dist.is_initialized():
        return None
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_stats)
    cleaned: list[dict] = []
    for rank_id, item in enumerate(gathered):
        if not isinstance(item, dict) or not item:
            continue
        entry = dict(item)
        entry.setdefault("global_rank", rank_id)
        cleaned.append(entry)
    return cleaned


def _ensure_distributed(tp_size: int = 1, data_parallel_size: int = 1, runtime_mode: str = "legacy_tp_ep"):
    """Initialize torch.distributed and set up the TP × EP subgroups.

    Single-GPU path: initialize a one-rank gloo group so `dist.get_rank()` works,
    and put us on cuda:0 so subsequent `torch.empty(...)` lands on GPU.

    Multi-GPU path: the parent (e.g. an mp.spawn'd worker) is expected to have
    already called `dist.init_process_group(backend='nccl', world_size=N, rank=r,
    device_id=torch.device(f'cuda:{r}'))` and `torch.cuda.set_device(r)`. We then
    construct TP × EP subgroups (see services/utils/parallel.py) where:
      - `tp_size == 1`           → pure EP (current default).
      - `tp_size == world_size`  → pure TP (every rank holds 1/N of every expert).
      - `1 < tp_size < world_size` → TP × EP composition.

    Layout convention: rank = ep_rank * tp_size + tp_rank.
    """
    import torch
    from workshop.nanovllm_moe.services.utils.parallel import init_parallel_groups

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12399")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
        torch.cuda.set_device(0)

    world_size = dist.get_world_size()
    init_parallel_groups(
        tp_size=tp_size,
        world_size=world_size,
        data_parallel_size=data_parallel_size,
        runtime_mode=runtime_mode,
    )

    torch.set_default_device(f"cuda:{torch.cuda.current_device()}")


class LLMEngine(BaseService):

    def __init__(self, model, **kwargs):
        super().__init__()
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)
        if not DUMMY_CREATION:
            self.__post_init__()

    def __post_init__(self):
        _ensure_distributed(
            tp_size=self.config.tensor_parallel_size,
            data_parallel_size=self.config.data_parallel_size,
            runtime_mode=self.config.moe_runtime_mode,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, use_fast=True)
        self.config.eos = self.tokenizer.eos_token_id

        # ModelRunner internally wires Attention <-> Model (and MoE <-> Dispatch/Experts/Combine).
        # It also allocates KV cache, which fixes self.config.num_kvcache_blocks.
        orch = RegistryOrchestrator()
        self.model_runner = orch.add(ModelRunner(self.config))
        self.block_mngr = orch.add(BlockManager(
            self.config.num_kvcache_blocks, self.config.kvcache_block_size,
        ))
        self.scheduler = orch.add(Scheduler(self.config))

        # BlockManager methods are consumed by Scheduler.{schedule, postprocess, preempt}.
        for name in ("can_allocate", "allocate", "can_append", "may_append", "deallocate"):
            orch.register(self.block_mngr, name, self.scheduler)

        # Scheduler methods are exposed on the engine itself so step() can call self.<m>().
        for name in ("add", "schedule", "postprocess", "is_finished"):
            orch.register(self.scheduler, name, self)

        # ModelRunner.run is exposed on the engine.
        orch.register(self.model_runner, "run", self)
        orch.finalize()
        atexit.register(self.reset)

    def reset(self):
        """Clear scheduler and block-manager state. Idempotent."""
        if not hasattr(self, "scheduler"):
            return
        self.scheduler.waiting.clear()
        self.scheduler.running.clear()
        self.block_mngr.reset()

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        *,
        seq_id: int | None = None,
    ):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence.from_prompt(
            prompt,
            sampling_params,
            self.config.kvcache_block_size,
            seq_id=seq_id,
        )
        self.add(seq)

    def _is_vllm_dp_ep(self) -> bool:
        return self.config.moe_runtime_mode == "vllm_dp_ep" and dist.is_initialized() and dist.get_world_size() > 1

    def _is_owner_local_ep(self) -> bool:
        return self.config.moe_runtime_mode == "owner_local_ep" and dist.is_initialized() and dist.get_world_size() > 1

    def _is_owner_sharded_mode(self) -> bool:
        return self._is_vllm_dp_ep() or self._is_owner_local_ep()

    def _local_prompt_indices(self, num_prompts: int) -> list[int]:
        if not self._is_owner_sharded_mode():
            return list(range(num_prompts))
        dp_rank = get_dp_rank()
        dp_size = get_dp_world_size()
        return [idx for idx in range(num_prompts) if idx % dp_size == dp_rank]

    def _all_dp_finished(self) -> bool:
        local_finished = int(self.is_finished())
        if not self._is_owner_sharded_mode():
            return bool(local_finished)
        flag = torch.tensor([local_finished], device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def _gather_outputs_owner_sharded(self, outputs: dict[int, list[int]]) -> list[dict]:
        payload = outputs if (self._is_owner_local_ep() or is_dp_leader()) else {}
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, payload)
        if dist.get_rank() != 0:
            return []
        merged: dict[int, list[int]] = {}
        for item in gathered:
            if isinstance(item, dict):
                merged.update(item)
        return [
            {"text": self.tokenizer.decode(merged[seq_id]), "token_ids": merged[seq_id]}
            for seq_id in sorted(merged)
        ]

    def step(self):
        if self._is_owner_sharded_mode() and self.is_finished():
            record_timing = os.environ.get("MOE_RECORD_TIMING", "1") == "1"
            if record_timing:
                torch.cuda.synchronize()
            t = perf_counter()
            self.run([], False)
            if record_timing:
                torch.cuda.synchronize()
            excution_time = perf_counter() - t
            return [], 0, excution_time
        seqs, is_prefill = self.schedule()
        record_timing = os.environ.get("MOE_RECORD_TIMING", "1") == "1"
        if record_timing:
            torch.cuda.synchronize()
        t = perf_counter()
        token_ids = self.run(seqs, is_prefill)
        if record_timing:
            torch.cuda.synchronize()
        excution_time = perf_counter() - t
        self.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens, excution_time

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        ep_ll_runtime_stats.reset()
        ep_ll_runtime_stats.set_static_metadata(
            moe_impl=self.config.moe_impl,
            **_ep_ll_workspace_summary(self.model_runner),
        )
        if use_tqdm and (not self._is_owner_sharded_mode() or dist.get_rank() == 0):
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for idx in self._local_prompt_indices(len(prompts)):
            self.add_request(prompts[idx], sampling_params[idx], seq_id=idx)
        outputs = {}
        metrics = {
            "prefill_tokens": 0,
            "prefill_time_s": 0.0,
            "decode_tokens": 0,
            "decode_time_s": 0.0,
            "steps": [],
        }
        e2e_t0 = perf_counter()
        prefill_throughput = decode_throughput = 0.
        while True:
            if self._all_dp_finished():
                break
            output, num_tokens, excution_time = self.step()
            step = {
                "is_prefill": num_tokens > 0,
                "tokens": abs(int(num_tokens)),
                "time_s": float(excution_time),
            }
            metrics["steps"].append(step)
            if num_tokens > 0:
                metrics["prefill_tokens"] += int(num_tokens)
                metrics["prefill_time_s"] += float(excution_time)
            else:
                metrics["decode_tokens"] += -int(num_tokens)
                metrics["decode_time_s"] += float(excution_time)
            if use_tqdm and (not self._is_owner_sharded_mode() or dist.get_rank() == 0):
                if num_tokens > 0:
                    prefill_throughput = num_tokens / excution_time
                else:
                    decode_throughput = -num_tokens / excution_time
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
            if use_tqdm and (not self._is_owner_sharded_mode() or dist.get_rank() == 0):
                pbar.update(1) 
        metrics["e2e_total_time_s"] = perf_counter() - e2e_t0
        metrics["prefill_throughput_tok_s"] = (
            metrics["prefill_tokens"] / metrics["prefill_time_s"]
            if metrics["prefill_time_s"] > 0 else 0.0
        )
        metrics["decode_throughput_tok_s"] = (
            metrics["decode_tokens"] / metrics["decode_time_s"]
            if metrics["decode_time_s"] > 0 else 0.0
        )
        ep_ll_stats = ep_ll_runtime_stats.summarize()
        if ep_ll_stats:
            local_ep_ll_stats = ep_ll_runtime_stats.summarize_local()
            metrics["ep_ll_stats"] = ep_ll_stats
            metrics["ep_ll_stats_by_rank"] = _gather_ep_ll_rank_stats(local_ep_ll_stats)
            metrics["ep_ll_m_max"] = ep_ll_stats.get("m_max")
            metrics["ep_ll_overflowed_replicas"] = ep_ll_stats.get("overflowed_replicas")
            metrics["ep_ll_overflow_replica_fraction"] = ep_ll_stats.get("overflow_replica_fraction")
            metrics["ep_ll_observed_bucket_max"] = ep_ll_stats.get("observed_bucket_max")
            metrics["ep_ll_workspace_bytes_total"] = ep_ll_stats.get("workspace_bytes_total")
        self.last_generation_metrics = metrics
        self.reset()
        if self._is_owner_sharded_mode():
            outputs = self._gather_outputs_owner_sharded(outputs)
        else:
            outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
            outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm and (not self._is_owner_sharded_mode() or dist.get_rank() == 0):
            pbar.close()
        return outputs
