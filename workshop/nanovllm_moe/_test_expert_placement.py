"""Smoke tests for EP expert placement maps.

Run:
  python -m workshop.nanovllm_moe._test_expert_placement
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from workshop.nanovllm_moe.services.utils.expert_placement import build_expert_placement


def test_contiguous():
    p = build_expert_placement(8, 2, "contiguous")
    assert p.expert_to_rank == [0, 0, 0, 0, 1, 1, 1, 1]
    assert p.expert_to_local == [0, 1, 2, 3, 0, 1, 2, 3]
    assert p.local_to_global == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_round_robin():
    p = build_expert_placement(8, 2, "round_robin")
    assert p.expert_to_rank == [0, 1, 0, 1, 0, 1, 0, 1]
    assert p.expert_to_local == [0, 0, 1, 1, 2, 2, 3, 3]
    assert p.local_to_global == [[0, 2, 4, 6], [1, 3, 5, 7]]


def test_invalid_policy():
    try:
        build_expert_placement(8, 2, "checkerboard")
    except ValueError as exc:
        assert "unknown expert placement policy" in str(exc)
    else:
        raise AssertionError("invalid placement policy should fail")


def test_fixed_random_shuffle_reproducible():
    p1 = build_expert_placement(8, 2, "fixed_random_shuffle", seed=123)
    p2 = build_expert_placement(8, 2, "fixed_random_shuffle", seed=123)
    assert p1.local_to_global == p2.local_to_global
    assert sorted(p1.local_to_global[0] + p1.local_to_global[1]) == list(range(8))


def test_json_placement():
    payload = {
        "policy": "load_balanced_greedy_with_locality_tiebreak",
        "ep_size": 2,
        "num_experts": 4,
        "E_local": 2,
        "layer_mode": "global",
        "placement": {
            "expert_to_rank": [1, 0, 1, 0],
            "expert_to_local": [0, 0, 1, 1],
            "local_to_global": [[1, 3], [0, 2]],
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "placement.json"
        path.write_text(json.dumps(payload))
        p = build_expert_placement(
            4,
            2,
            "load_balanced_greedy_with_locality_tiebreak",
            placement_path=str(path),
        )
    assert p.expert_to_rank == [1, 0, 1, 0]
    assert p.local_to_global == [[1, 3], [0, 2]]


def main():
    test_contiguous()
    test_round_robin()
    test_fixed_random_shuffle_reproducible()
    test_json_placement()
    test_invalid_policy()
    print("expert placement tests PASSED")


if __name__ == "__main__":
    main()
