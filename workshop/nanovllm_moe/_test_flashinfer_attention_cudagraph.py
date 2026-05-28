"""Standalone smoke for FlashInfer decode cudagraph metadata replay.

This covers the attention-only path that previously tripped illegal memory
access in decode. It also isolates FlashInfer JIT into a fresh workspace so a
stale build.ninja cannot pin an older nvcc path.

Run:
  /home/lzy/miniconda3/envs/vllm/bin/python -m workshop.nanovllm_moe._test_flashinfer_attention_cudagraph
"""

import atexit
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _parse_nvcc_version(nvcc: Path) -> tuple[int, int] | None:
    try:
        output = subprocess.check_output([str(nvcc), "--version"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release (\d+)\.(\d+),", output)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _pick_cuda_home() -> str | None:
    candidates: list[str] = []
    for key in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(key)
        if value:
            candidates.append(value)
    candidates.extend(
        str(path)
        for path in sorted(Path("/usr/local").glob("cuda-*"), reverse=True)
    )
    candidates.append("/usr/local/cuda")

    best: tuple[tuple[int, int], str] | None = None
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        nvcc = Path(candidate) / "bin" / "nvcc"
        version = _parse_nvcc_version(nvcc)
        if version is None:
            continue
        if best is None or version > best[0]:
            best = (version, candidate)
    return None if best is None else best[1]


def _configure_flashinfer_env() -> str | None:
    cuda_home = _pick_cuda_home()
    if cuda_home is not None:
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["CUDA_PATH"] = cuda_home
        os.environ["FLASHINFER_NVCC"] = str(Path(cuda_home) / "bin" / "nvcc")
        path_entries = [
            str(Path(sys.executable).resolve().parent),
            str(Path(cuda_home) / "bin"),
        ]
        existing_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(path_entries + [existing_path])

    workspace = tempfile.mkdtemp(prefix="flashinfer_ws_", dir="/tmp")
    atexit.register(shutil.rmtree, workspace, ignore_errors=True)
    os.environ["FLASHINFER_WORKSPACE_BASE"] = workspace
    return cuda_home


CUDA_HOME = _configure_flashinfer_env()

import torch

from workshop.nanovllm_moe.artifacts.attention_backend.flashinfer_attention import Attention


class _HF:
    num_attention_heads = 8
    num_key_value_heads = 1
    head_dim = 128
    hidden_size = 1024
    max_position_embeddings = 64


class _Cfg:
    hf_config = _HF()
    tensor_parallel_size = 1
    kvcache_block_size = 1
    max_num_seqs = 4
    max_model_len = 64
    num_kvcache_blocks = 64


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")

    torch.cuda.set_device(0)
    torch.manual_seed(0)

    att = Attention(_Cfg())
    bs = 2
    page_counts = torch.tensor([0, 3, 2], device="cuda", dtype=torch.int32)
    page_indices = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.int32)
    last_page_lens = torch.ones(bs, device="cuda", dtype=torch.int32)

    att.init_forward_metadata_capture_cuda_graph(
        bs,
        page_counts,
        page_indices,
        last_page_lens,
    )

    q = torch.randn(bs, att.num_heads, att.head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        _Cfg.num_kvcache_blocks,
        _Cfg.kvcache_block_size,
        att.num_kv_heads,
        att.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    out = torch.empty_like(q)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            out.copy_(att.forward_wrapper.forward(q, (k, v)))
    warmup_stream.synchronize()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(att.forward_wrapper.forward(q, (k, v)))

    graph.replay()
    torch.cuda.synchronize()

    replay_page_counts = torch.tensor([0, 4, 2], device="cuda", dtype=torch.int32)
    replay_page_indices = torch.tensor(
        [5, 6, 7, 8, 9, 10],
        device="cuda",
        dtype=torch.int32,
    )
    att.init_forward_metadata_replay_cuda_graph(
        bs,
        replay_page_counts,
        replay_page_indices,
        last_page_lens,
    )

    graph.replay()
    torch.cuda.synchronize()
    print(
        "cuda_graph_attention_decode_smoke=ok",
        tuple(out.shape),
        out.dtype,
        f"cuda_home={CUDA_HOME}",
    )


if __name__ == "__main__":
    main()
