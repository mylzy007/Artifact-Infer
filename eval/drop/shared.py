"""Shared helpers for Prefill L Sweep benches.

- CUDAEventTimer: context manager around torch.cuda.Event-based timing.
- write_jsonl: append a row to a JSONL file, rank0-only when distributed.
- agg_stats: mean/std/p10/p50/p90/p99/min/max over a list of floats.
- WandbLogger: optional W&B logger that no-ops when not configured.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


class CUDAEventTimer:
    """Single-segment CUDA timing via torch.cuda.Event.

    Usage:
        timer = CUDAEventTimer()
        with timer:
            do_gpu_work()
        # ... later, after torch.cuda.synchronize()
        elapsed_us = timer.read()
    """

    def __init__(self) -> None:
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.elapsed_us: float | None = None

    def __enter__(self) -> "CUDAEventTimer":
        self.start.record()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.end.record()
        return False

    def read(self) -> float:
        """Read elapsed time in microseconds.

        Caller must have run torch.cuda.synchronize() before this.
        """
        self.elapsed_us = float(self.start.elapsed_time(self.end) * 1000.0)
        return self.elapsed_us


def _is_rank0() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def write_jsonl(path: str, row: dict[str, Any]) -> None:
    """Append a JSON row to ``path``. No-op on non-rank0 ranks."""
    if not _is_rank0():
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def truncate_file(path: str) -> None:
    """Truncate file at ``path``. No-op on non-rank0 ranks."""
    if not _is_rank0():
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    open(path, "w").close()


def agg_stats(values: list[float]) -> dict[str, float]:
    """Aggregate a list of floats into a stats dict."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0,
                "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


class WandbLogger:
    """Optional W&B logger. No-op when wandb missing, unconfigured, or non-rank0."""

    def __init__(
        self,
        project: str,
        name: str,
        config: dict[str, Any],
        enabled: bool = True,
    ) -> None:
        self.run = None
        if not enabled or not project:
            return
        if not _is_rank0():
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            print("[wandb] not installed; logging disabled")
            return
        # Detect credentials: env var or stored API key.
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            stored = getattr(getattr(wandb, "api", None), "api_key", None)
            api_key = stored
        if not api_key:
            print("[wandb] no API key configured; logging disabled")
            return
        try:
            self.run = wandb.init(
                project=project, name=name, config=config, reinit=True
            )
            print(f"[wandb] run started: {self.run.name} (project={project})")
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] init failed: {exc}; logging disabled")
            self.run = None

    def log(self, row: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            self.run.log(row)
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] log failed: {exc}")

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] finish failed: {exc}")
        self.run = None
