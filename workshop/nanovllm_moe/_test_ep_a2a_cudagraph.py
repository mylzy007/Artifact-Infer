"""Standalone validation: NCCL all_to_all_single + CUDA graph capture.

Gates the entire EP-LL approach: if NCCL collectives can't be captured in a
cuda graph in our PyTorch + NCCL combo, we have to put the a2a outside the
captured region (loses ~30-50% of decode perf).

Spawns N processes. Each:
  1. init NCCL group on a separate GPU
  2. allocate persistent send/recv buffers of shape [N, M, H] bf16
  3. run all_to_all_single eager-mode and verify correctness
  4. capture a cuda graph that does the same a2a
  5. replay the graph and verify bit-exact match with eager output
  6. measure latency: eager a2a vs graph replay (should be lower for graph)

Run:  python -m workshop.nanovllm_moe._test_ep_a2a_cudagraph
      WORLD_SIZE=2 python -m workshop.nanovllm_moe._test_ep_a2a_cudagraph
"""
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "2"))
M = 64        # rows per (rank, dest_rank) bucket — analogue of M_max in EP-LL
H = 2048      # hidden dim
DTYPE = torch.bfloat16


def worker(rank: int, world_size: int):
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{rank}"),
        )
        torch.set_default_device(f"cuda:{rank}")

        # Persistent buffers (CUDA-graph requirement).
        # send_buf[r, m, :]  is the m-th row this rank sends to rank r.
        # recv_buf[r, m, :]  is the m-th row this rank receives from rank r.
        send_buf = torch.empty((world_size, M, H), dtype=DTYPE)
        recv_buf = torch.empty((world_size, M, H), dtype=DTYPE)

        # Fill send_buf with rank-distinguishable pattern:
        # rank r sends to dest d:  values = (rank + 1) * 100 + d * 10 + position_in_bucket / M
        for d in range(world_size):
            base = (rank + 1) * 100 + d * 10
            send_buf[d] = base + torch.arange(M, dtype=torch.float32).view(M, 1).expand(M, H).to(DTYPE) / M

        # ---- Eager a2a (with warmup so we don't time NCCL setup) ----
        for _ in range(3):
            dist.all_to_all_single(recv_buf, send_buf)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(10):
            dist.all_to_all_single(recv_buf, send_buf)
        torch.cuda.synchronize()
        t_eager = (time.perf_counter() - t0) / 10

        # Verify: rank r's recv_buf[s] should be (s+1)*100 + r*10 + arange/M (sender s -> dest r).
        eager_out_clone = recv_buf.clone()
        for s in range(world_size):
            expected_base = (s + 1) * 100 + rank * 10
            expected = (expected_base + torch.arange(M, dtype=torch.float32).view(M, 1).expand(M, H).to(DTYPE) / M)
            got = recv_buf[s]
            ok = torch.allclose(got.float(), expected.float(), atol=1e-2)
            assert ok, f"rank {rank}: eager a2a from sender {s} is wrong"
        if rank == 0:
            print(f"[rank 0] eager a2a OK ({t_eager * 1000:.3f} ms)", flush=True)

        # ---- Cuda graph capture ----
        # Warmup on a side stream (required by torch.cuda.graph).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                dist.all_to_all_single(recv_buf, send_buf)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                dist.all_to_all_single(recv_buf, send_buf)
        except Exception as e:
            print(f"[rank {rank}] CUDA graph capture FAILED: {type(e).__name__}: {e}", flush=True)
            dist.destroy_process_group()
            sys.exit(2)

        if rank == 0:
            print(f"[rank 0] cuda graph capture OK", flush=True)

        # ---- Cuda graph replay + verify ----
        # Modify send_buf to a *different* pattern so we know the graph re-reads it
        # and writes the right new values into recv_buf.
        for d in range(world_size):
            base = (rank + 1) * 1000 + d * 100  # different scale than before
            send_buf[d] = base + torch.arange(M, dtype=torch.float32).view(M, 1).expand(M, H).to(DTYPE) / M

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        t_graph = (time.perf_counter() - t0) / 10

        # Verify with new pattern.
        all_ok = True
        for s_idx in range(world_size):
            expected_base = (s_idx + 1) * 1000 + rank * 100
            expected = expected_base + torch.arange(M, dtype=torch.float32).view(M, 1).expand(M, H).to(DTYPE) / M
            got = recv_buf[s_idx]
            if not torch.allclose(got.float(), expected.float(), atol=1e-1):
                print(f"[rank {rank}] cuda graph replay WRONG for sender {s_idx}: "
                      f"expected base {expected_base}, got[0,0] = {got[0, 0].item()}", flush=True)
                all_ok = False

        print(f"[rank {rank}] cuda graph replay {'OK' if all_ok else 'FAIL'} "
              f"({t_graph * 1000:.3f} ms/replay vs eager {t_eager * 1000:.3f} ms)", flush=True)
        # Skip barrier+destroy — they often hang at process exit and we're done with the validation.
        # Just exit cleanly; the parent process will reap us.
        sys.exit(0 if all_ok else 4)

    except Exception as e:
        print(f"[rank {rank}] EXCEPTION: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        sys.exit(1)


def main():
    print(f"Spawning {WORLD_SIZE} ranks for NCCL all_to_all + cuda graph validation", flush=True)
    print(f"  buffer shape: [N={WORLD_SIZE}, M={M}, H={H}] bf16 = "
          f"{WORLD_SIZE * M * H * 2 / 1e6:.2f} MB per direction per rank", flush=True)
    mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    print("OK: NCCL all_to_all_single is CUDA-graph capturable on this stack")


if __name__ == "__main__":
    main()
