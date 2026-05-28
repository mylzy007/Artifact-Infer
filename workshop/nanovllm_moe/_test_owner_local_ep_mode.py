"""Unit tests for the owner_local_ep runtime mode.

Focus:
  1. Prompt ownership shards by rank with no duplication.
  2. owner_local_ep never relies on TP broadcast for sampling.
  3. Final gather restores original prompt order.
  4. DispatchEPLL + CombineEPLL preserve token-owner semantics locally.
  5. legacy_tp_ep / vllm_dp_ep helper behavior does not regress.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.append(REPO)

from workshop.nanovllm_moe.artifacts.modeling.layers.moe.combine_ep_ll import CombineEPLL
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ll import DispatchEPLL
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.combine_ep_ht import CombineEPHT
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ht import DispatchEPHT, TokMetaEPHT
from workshop.nanovllm_moe.services.engine.llm_engine import LLMEngine
from workshop.nanovllm_moe.services.model_runner.model_runner import ModelRunner
from workshop.nanovllm_moe.services.sampling_params import SamplingParams
from workshop.nanovllm_moe.services.utils import parallel
from workshop.nanovllm_moe.services.utils import routing_profile


class FakeDist:
    def __init__(self, rank: int, world_size: int, gathered=None):
        self._rank = rank
        self._world_size = world_size
        self._gathered = gathered or []
        self.broadcast_called = False

    def is_initialized(self):
        return True

    def get_rank(self, group=None):
        return self._rank

    def get_world_size(self, group=None):
        return self._world_size

    def all_gather_object(self, out_list, payload):
        for idx in range(self._world_size):
            out_list[idx] = self._gathered[idx]

    def broadcast_object_list(self, obj_list, src=0, group=None):
        self.broadcast_called = True
        raise AssertionError("owner_local_ep should not broadcast_object_list")


def _make_engine(mode: str, *, rank: int = 0, world_size: int = 4):
    engine = object.__new__(LLMEngine)
    engine.config = SimpleNamespace(moe_runtime_mode=mode)
    engine.tokenizer = SimpleNamespace(decode=lambda token_ids: f"decoded:{token_ids}")
    engine.is_finished = lambda: True
    return engine, FakeDist(rank=rank, world_size=world_size)


def test_owner_local_prompt_sharding():
    engine, fake_dist = _make_engine("owner_local_ep", rank=2, world_size=4)
    with patch("workshop.nanovllm_moe.services.engine.llm_engine.dist", fake_dist):
        with patch("workshop.nanovllm_moe.services.engine.llm_engine.get_dp_rank", lambda: 2):
            with patch("workshop.nanovllm_moe.services.engine.llm_engine.get_dp_world_size", lambda: 4):
                assert engine._local_prompt_indices(10) == [2, 6]


def test_owner_local_output_gather_restores_order():
    engine, fake_dist = _make_engine("owner_local_ep", rank=0, world_size=4)
    fake_dist._gathered = [
        {0: [10], 4: [14]},
        {1: [11], 5: [15]},
        {2: [12]},
        {3: [13]},
    ]
    with patch("workshop.nanovllm_moe.services.engine.llm_engine.dist", fake_dist):
        outputs = engine._gather_outputs_owner_sharded({0: [10], 4: [14]})
    assert [x["token_ids"] for x in outputs] == [[10], [11], [12], [13], [14], [15]]


def test_owner_local_sampling_is_local_only():
    runner = object.__new__(ModelRunner)
    runner.runtime_mode = "owner_local_ep"
    runner.dp_leader = True
    runner.global_rank = 1
    runner.prepare_prefill = lambda seqs: (torch.tensor([1, 2]), torch.tensor([0, 1]))
    runner.prepare_decode = lambda seqs: (torch.tensor([1]), torch.tensor([0]))
    runner.prepare_sample = lambda seqs: torch.tensor([0.0, 0.0])
    runner.run_model = lambda input_ids, positions, is_prefill: torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    runner.sampler = lambda logits, temperatures: torch.tensor([1, 0])
    seqs = [SimpleNamespace(seq_id=0), SimpleNamespace(seq_id=1)]

    with patch("workshop.nanovllm_moe.services.model_runner.model_runner.dist", FakeDist(rank=1, world_size=4)):
        with patch("workshop.nanovllm_moe.services.model_runner.model_runner.is_owner_local_ep_mode", lambda: True):
            with patch("workshop.nanovllm_moe.services.model_runner.model_runner.reset_context", lambda: None):
                token_ids = ModelRunner.run(runner, seqs, True)
    assert token_ids == [1, 0]


def test_dispatch_combine_owner_round_trip_local():
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    router_logits = torch.tensor([[5.0, -5.0], [6.0, -6.0]], dtype=torch.float32)
    dispatch = DispatchEPLL(
        num_experts_global=2,
        top_k=1,
        m_max=4,
        norm_topk_prob=True,
        dispatch_kernel="torch",
    )
    combine = CombineEPLL(hidden_size=2, top_k=1, num_experts_local=2, m_max=4)
    tok_meta = dispatch(hidden, router_logits)
    out = combine(tok_meta.hidden_recv.clone(), tok_meta)
    assert torch.allclose(out, hidden, atol=1e-6), (out, hidden)


def test_parallel_owner_local_ep_layout():
    class FakeParallelDist:
        def is_initialized(self):
            return True

        def get_rank(self, group=None):
            return 3

        def get_world_size(self, group=None):
            return 4

        def new_group(self, ranks):
            return tuple(ranks)

    fake_dist = FakeParallelDist()
    with patch.object(parallel, "dist", fake_dist):
        parallel.init_parallel_groups(
            tp_size=1,
            world_size=4,
            data_parallel_size=4,
            runtime_mode="owner_local_ep",
        )
        assert parallel.is_owner_local_ep_mode()
        assert parallel.get_tp_rank() == 0
        assert parallel.get_dp_rank() == 3
        assert parallel.get_ep_rank() == 3
        assert parallel.get_dp_world_size() == 4
        assert parallel.get_ep_world_size() == 4


def test_owner_local_epht_dispatch_counts_for_all_sources():
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    router_logits = torch.tensor([[5.0, -5.0], [6.0, -6.0]], dtype=torch.float32)

    def fake_moe_align_block_size(*args, **kwargs):
        sorted_token_ids_buf = args[3]
        expert_ids_buf = args[4]
        num_tokens_post_padded = args[5]
        recv_topk_ids = args[0].view(-1)
        n = recv_topk_ids.numel()
        if n:
            sorted_token_ids_buf[:n].copy_(torch.arange(n, device=sorted_token_ids_buf.device, dtype=torch.int32))
            expert_ids_buf[:1].fill_(1)
            num_tokens_post_padded[0] = n

    with patch.dict(sys.modules, {"sgl_kernel": SimpleNamespace(moe_align_block_size=fake_moe_align_block_size)}):
        with patch("workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ht.get_runtime_mode", lambda: "owner_local_ep"):
            with patch("workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ht.is_owner_local_ep_mode", lambda: True):
                with patch("workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ht.get_global_rank", lambda: 0):
                    disp = DispatchEPHT(
                        num_experts_global=2,
                        top_k=1,
                        block_size_m=64,
                        norm_topk_prob=True,
                        expert_placement="contiguous",
                    )
                    tok_meta = disp(hidden, router_logits)
    assert tok_meta.is_source_leader is True
    assert sum(tok_meta.send_counts) == hidden.size(0) * 1
    assert tok_meta.topk_ids.shape == (2, 1)


def test_owner_local_epht_combine_has_no_broadcast_dependency():
    tok_meta = TokMetaEPHT(
        recv_hidden=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        recv_topk_ids=torch.tensor([[0]], dtype=torch.int32),
        recv_topk_weights=torch.tensor([[1.0]], dtype=torch.float32),
        sorted_token_ids=torch.tensor([0], dtype=torch.int32),
        expert_ids=torch.tensor([1], dtype=torch.int32),
        num_tokens_post_padded=torch.tensor([1], dtype=torch.int32),
        topk_weights=torch.tensor([[1.0]], dtype=torch.float32),
        topk_ids=torch.tensor([[0]], dtype=torch.int32),
        sort_perm=torch.tensor([0], dtype=torch.int64),
        send_counts=[1],
        recv_counts=[1],
        T_local=1,
        is_source_leader=True,
        source_leader_global_rank=0,
    )
    fake_dist = FakeDist(rank=0, world_size=1)
    combine = CombineEPHT(hidden_size=2, top_k=1)
    with patch("workshop.nanovllm_moe.artifacts.modeling.layers.moe.combine_ep_ht.dist", fake_dist):
        out = combine(torch.tensor([[1.0, 2.0]], dtype=torch.float32), tok_meta)
    assert torch.allclose(out, torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    assert fake_dist.broadcast_called is False


def test_owner_local_routing_profile_uses_owner_rank_sources():
    routing_profile.reset()
    with patch("workshop.nanovllm_moe.services.utils.routing_profile.get_runtime_mode", lambda: "owner_local_ep"):
        with patch("workshop.nanovllm_moe.services.utils.routing_profile.get_dp_world_size", lambda: 2):
            with patch("workshop.nanovllm_moe.services.utils.routing_profile.get_dp_rank", lambda: 1):
                with patch("workshop.nanovllm_moe.services.utils.routing_profile.dist", SimpleNamespace(is_initialized=lambda: False)):
                    with patch.dict(os.environ, {"MOE_PROFILE_ROUTING": "1"}):
                        routing_profile.record_routing(0, torch.tensor([[0, 1]], dtype=torch.int32), 4)
    payload = routing_profile._rank_payload()
    assert len(payload["traffic"][0]) == 2
    assert sum(payload["traffic"][0][0]) == 0
    assert sum(payload["traffic"][0][1]) == 2


def test_non_owner_modes_keep_expected_helpers():
    engine, fake_dist = _make_engine("legacy_tp_ep", rank=0, world_size=4)
    with patch("workshop.nanovllm_moe.services.engine.llm_engine.dist", fake_dist):
        assert engine._local_prompt_indices(5) == [0, 1, 2, 3, 4]

    engine, fake_dist = _make_engine("vllm_dp_ep", rank=0, world_size=4)
    with patch("workshop.nanovllm_moe.services.engine.llm_engine.dist", fake_dist):
        with patch("workshop.nanovllm_moe.services.engine.llm_engine.get_dp_rank", lambda: 1):
            with patch("workshop.nanovllm_moe.services.engine.llm_engine.get_dp_world_size", lambda: 4):
                assert engine._local_prompt_indices(6) == [1, 5]


def main():
    test_owner_local_prompt_sharding()
    test_owner_local_output_gather_restores_order()
    test_owner_local_sampling_is_local_only()
    test_dispatch_combine_owner_round_trip_local()
    test_parallel_owner_local_ep_layout()
    test_owner_local_epht_dispatch_counts_for_all_sources()
    test_owner_local_epht_combine_has_no_broadcast_dependency()
    test_owner_local_routing_profile_uses_owner_rank_sources()
    test_non_owner_modes_keep_expected_helpers()
    print("OK: owner_local_ep unit tests passed")


if __name__ == "__main__":
    main()
