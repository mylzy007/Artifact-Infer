"""Unit test for torch_masked_grouped_gemm.

Spec: per-expert SwiGLU FFN with `masked_m[e]` valid rows; padding rows zero in output.
We validate both the math (vs hand-rolled per-expert reference) and that padding rows
are exactly zero.
"""
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12399")
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
torch.cuda.set_device(0)
torch.set_default_device("cuda")

from workshop.nanovllm_moe.artifacts.moe_backend.torch_masked_grouped_gemm import (
    torch_masked_grouped_gemm,
)


def hand_reference(hidden_states, w1, w2, masked_m):
    """Unfused per-expert reference with explicit masking."""
    E, M_max, H = hidden_states.shape
    N = w2.size(2)
    out = torch.zeros((E, M_max, H), dtype=hidden_states.dtype, device=hidden_states.device)
    for e in range(E):
        m = int(masked_m[e].item())
        for row in range(m):
            x = hidden_states[e, row : row + 1]              # [1, H]
            gate_up = x @ w1[e].t()                          # [1, 2N]
            gate, up = gate_up.split(N, dim=-1)
            inter = F.silu(gate) * up                         # [1, N]
            out[e, row : row + 1] = inter @ w2[e].t()         # [1, H]
    return out


def main():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    E, M_max, H, N = 4, 16, 64, 32
    hidden = (torch.randn((E, M_max, H), dtype=torch.float32) * 0.1).to(dtype)
    w1 = (torch.randn((E, 2 * N, H), dtype=torch.float32) * 0.05).to(dtype)
    w2 = (torch.randn((E, H, N), dtype=torch.float32) * 0.05).to(dtype)

    # Heterogeneous valid counts including 0 (one expert receives nothing).
    masked_m = torch.tensor([7, 0, M_max, 3], dtype=torch.int32)

    # Fill padding rows of `hidden` with random garbage to verify the kernel
    # doesn't propagate it into output.
    for e in range(E):
        m = int(masked_m[e].item())
        hidden[e, m:].normal_(0.0, 100.0).to(dtype)  # huge values in padding

    out_ref = hand_reference(hidden, w1, w2, masked_m)
    out = torch_masked_grouped_gemm(hidden, w1, w2, masked_m)

    # 1. Valid rows match within bf16 noise.
    for e in range(E):
        m = int(masked_m[e].item())
        if m == 0:
            continue
        diff = (out[e, :m].float() - out_ref[e, :m].float()).abs()
        assert diff.max().item() < 1e-2, (
            f"expert {e}: max abs diff {diff.max().item()} (mean {diff.mean().item()})"
        )

    # 2. Padding rows are exactly zero (not random garbage from `hidden`).
    for e in range(E):
        m = int(masked_m[e].item())
        if m < M_max:
            pad = out[e, m:]
            assert (pad == 0).all(), f"expert {e}: padding rows not zero (got max {pad.abs().max()})"

    # 3. Empty expert (masked_m=0) is fully zero.
    assert (out[1] == 0).all(), "expert with masked_m=0 should be all-zero in output"

    # 4. Output dtype/shape check.
    assert out.dtype == dtype
    assert out.shape == (E, M_max, H)
    assert out.device.type == "cuda"

    print(f"OK: torch_masked_grouped_gemm matches per-expert reference (E={E}, M_max={M_max}, "
          f"H={H}, N={N}; padding stays zero, empty expert handled)")


if __name__ == "__main__":
    main()
