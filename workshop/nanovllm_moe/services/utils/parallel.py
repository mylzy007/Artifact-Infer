"""Parallel-group helpers for legacy TP×EP, vLLM-style DP+EP, and owner-local EP."""
from __future__ import annotations

import torch.distributed as dist

_TP_GROUP = None
_EP_GROUP = None
_TP_SIZE = 1
_DP_SIZE = 1
_EP_SIZE = 1
_DP_RANK = 0
_TP_RANK = 0
_EP_RANK = 0
_WORLD_SIZE = 1
_GLOBAL_RANK = 0
_RUNTIME_MODE = "legacy_tp_ep"


def init_parallel_groups(
    tp_size: int,
    world_size: int,
    *,
    data_parallel_size: int = 1,
    runtime_mode: str = "legacy_tp_ep",
) -> None:
    global _TP_GROUP, _EP_GROUP
    global _TP_SIZE, _DP_SIZE, _EP_SIZE, _DP_RANK, _TP_RANK, _EP_RANK
    global _WORLD_SIZE, _GLOBAL_RANK, _RUNTIME_MODE

    _TP_GROUP = None
    _EP_GROUP = None
    _TP_SIZE = int(tp_size)
    _DP_SIZE = int(data_parallel_size)
    _EP_SIZE = 1
    _DP_RANK = 0
    _TP_RANK = 0
    _EP_RANK = 0
    _WORLD_SIZE = int(world_size)
    _GLOBAL_RANK = 0
    _RUNTIME_MODE = str(runtime_mode)

    if not dist.is_initialized():
        return

    if _RUNTIME_MODE not in ("legacy_tp_ep", "vllm_dp_ep", "owner_local_ep"):
        raise ValueError(f"unknown runtime_mode={_RUNTIME_MODE!r}")

    _GLOBAL_RANK = dist.get_rank()
    _WORLD_SIZE = dist.get_world_size()
    if _WORLD_SIZE != int(world_size):
        raise ValueError(f"dist world_size={_WORLD_SIZE} != requested world_size={world_size}")

    if _RUNTIME_MODE == "owner_local_ep":
        if _TP_SIZE != 1:
            raise ValueError(f"owner_local_ep requires tp_size=1, got {_TP_SIZE}")
        if _DP_SIZE != _WORLD_SIZE:
            raise ValueError(
                f"owner_local_ep expects data_parallel_size=world_size={_WORLD_SIZE}, got {_DP_SIZE}"
            )
        _EP_SIZE = _WORLD_SIZE
        _DP_RANK = _GLOBAL_RANK
        _TP_RANK = 0
        _EP_RANK = _GLOBAL_RANK
        _TP_GROUP = None
        _EP_GROUP = None
        return

    if _RUNTIME_MODE == "vllm_dp_ep":
        if _WORLD_SIZE % _TP_SIZE != 0:
            raise ValueError(f"world_size={_WORLD_SIZE} must be divisible by tp_size={_TP_SIZE}")
        expected_dp = _WORLD_SIZE // _TP_SIZE
        if _DP_SIZE != expected_dp:
            raise ValueError(
                f"vllm_dp_ep expects data_parallel_size=world_size/tp_size={expected_dp}, "
                f"got {_DP_SIZE}"
            )
        _EP_SIZE = _WORLD_SIZE
        _DP_RANK = _GLOBAL_RANK // _TP_SIZE
        _TP_RANK = _GLOBAL_RANK % _TP_SIZE
        _EP_RANK = _GLOBAL_RANK

        my_tp_group = None
        for dp_rank in range(_DP_SIZE):
            ranks = list(range(dp_rank * _TP_SIZE, (dp_rank + 1) * _TP_SIZE))
            grp = dist.new_group(ranks=ranks)
            if _GLOBAL_RANK in ranks:
                my_tp_group = grp
        _TP_GROUP = my_tp_group
        _EP_GROUP = None  # world group
        return

    # Legacy layout: rank = ep_rank * tp_rank + tp_rank, EP is strided.
    if _WORLD_SIZE % _TP_SIZE != 0:
        raise ValueError(f"world_size={_WORLD_SIZE} must be divisible by tp_size={_TP_SIZE}")
    _EP_SIZE = _WORLD_SIZE // _TP_SIZE
    _DP_SIZE = _EP_SIZE
    _DP_RANK = _GLOBAL_RANK // _TP_SIZE
    _TP_RANK = _GLOBAL_RANK % _TP_SIZE
    _EP_RANK = _DP_RANK

    my_tp_group = None
    for g in range(_EP_SIZE):
        ranks = list(range(g * _TP_SIZE, (g + 1) * _TP_SIZE))
        grp = dist.new_group(ranks=ranks)
        if _GLOBAL_RANK in ranks:
            my_tp_group = grp

    my_ep_group = None
    for t in range(_TP_SIZE):
        ranks = list(range(t, _WORLD_SIZE, _TP_SIZE))
        grp = dist.new_group(ranks=ranks)
        if _GLOBAL_RANK in ranks:
            my_ep_group = grp

    _TP_GROUP = my_tp_group
    _EP_GROUP = my_ep_group


def get_runtime_mode() -> str:
    return _RUNTIME_MODE


def is_vllm_dp_ep_mode() -> bool:
    return _RUNTIME_MODE == "vllm_dp_ep"


def is_owner_local_ep_mode() -> bool:
    return _RUNTIME_MODE == "owner_local_ep"


def set_tp_group(group) -> None:
    global _TP_GROUP
    _TP_GROUP = group


def set_ep_group(group) -> None:
    global _EP_GROUP
    _EP_GROUP = group


def get_tp_group():
    return _TP_GROUP


def get_ep_group():
    return _EP_GROUP


def _gws(group) -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=group)


def _grank(group) -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank(group=group)


def get_global_rank() -> int:
    return _GLOBAL_RANK if dist.is_initialized() else 0


def get_world_size() -> int:
    return _WORLD_SIZE if dist.is_initialized() else 1


def get_tp_world_size() -> int:
    return _TP_SIZE if dist.is_initialized() else 1


def get_tp_rank() -> int:
    return _TP_RANK if dist.is_initialized() else 0


def get_dp_world_size() -> int:
    return _DP_SIZE


def get_dp_rank() -> int:
    return _DP_RANK if dist.is_initialized() else 0


def get_dp_leader_global_rank(dp_rank: int | None = None) -> int:
    dp_rank = get_dp_rank() if dp_rank is None else int(dp_rank)
    return dp_rank * _TP_SIZE


def is_dp_leader() -> bool:
    return get_tp_rank() == 0


def get_ep_world_size() -> int:
    return _EP_SIZE if dist.is_initialized() else 1


def get_ep_rank() -> int:
    return _EP_RANK if dist.is_initialized() else 0
