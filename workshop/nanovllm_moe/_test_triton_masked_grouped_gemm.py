"""Parity test: triton_masked_grouped_gemm vs torch_masked_grouped_gemm."""
import os
import torch
import torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12399")
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
torch.cuda.set_device(0)
torch.set_default_device("cuda")

from workshop.nanovllm_moe.artifacts.moe_backend.torch_masked_grouped_gemm import (
    torch_masked_grouped_gemm,
)
from workshop.nanovllm_moe.artifacts.moe_backend.triton_masked_grouped_gemm import (
    triton_masked_grouped_gemm,
)


def run_case(name, E, M_max, H, N, masked_m_list, scale_h=0.1, scale_w=0.05):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    hidden = (torch.randn((E, M_max, H), dtype=torch.float32) * scale_h).to(dtype)
    w1 = (torch.randn((E, 2 * N, H), dtype=torch.float32) * scale_w).to(dtype)
    w2 = (torch.randn((E, H, N), dtype=torch.float32) * scale_w).to(dtype)
    masked_m = torch.tensor(masked_m_list, dtype=torch.int32)

    # Garbage in padding rows of `hidden` — both kernels must zero output padding.
    for e in range(E):
        m = masked_m_list[e]
        if m < M_max:
            hidden[e, m:].normal_(0.0, 100.0).to(dtype)

    ref = torch_masked_grouped_gemm(hidden, w1, w2, masked_m)
    got = triton_masked_grouped_gemm(hidden, w1, w2, masked_m)

    # Padding rows must be zero in triton output too.
    for e in range(E):
        m = masked_m_list[e]
        if m < M_max:
            pad = got[e, m:]
            assert (pad == 0).all(), f"[{name}] expert {e}: triton padding rows not zero"

    # Compare valid rows. bf16 GEMMs differ from torch's bmm slightly due to
    # different accumulation order. Use a generous atol/rtol.
    diffs = []
    for e in range(E):
        m = masked_m_list[e]
        if m == 0:
            continue
        d = (got[e, :m].float() - ref[e, :m].float()).abs()
        diffs.append((e, d.max().item(), d.mean().item(), ref[e, :m].abs().mean().item()))

    print(f"[{name}] E={E}, M_max={M_max}, H={H}, N={N}, masked_m={masked_m_list}")
    for e, dmax, dmean, refmag in diffs:
        rel = dmax / max(refmag, 1e-6)
        print(f"  expert {e}: max abs diff {dmax:.4e} (mean {dmean:.4e}), rel-to-ref {rel:.2e}")

    # Threshold: 1e-2 absolute or 5e-2 relative — bf16 with two GEMMs has noisy lower bits.
    for e, dmax, _, refmag in diffs:
        if dmax > 1e-2 and dmax / max(refmag, 1e-6) > 5e-2:
            raise AssertionError(f"[{name}] expert {e}: divergence too large (max {dmax:.4e}, ref mag {refmag:.4e})")

    print(f"  OK\n")


def main():
    # Tiny case (sanity).
    run_case("tiny", E=4, M_max=16, H=64, N=32, masked_m_list=[7, 0, 16, 3])

    # Realistic Qwen3-30B-A3B per-rank shapes (8-way EP):
    #   E_local = 128 / 8 = 16, H = 2048, N = 768 (intermediate_size=768 per expert), M_max ~= 128.
    # Use smaller M_max for test speed.
    run_case(
        "qwen3_moe_per_rank",
        E=16, M_max=64, H=2048, N=768,
        masked_m_list=[64, 32, 16, 0, 50, 1, 64, 12, 8, 64, 30, 20, 0, 64, 7, 45],
    )

    # All-zero case (no tokens routed).
    run_case("all_zero", E=4, M_max=16, H=64, N=32, masked_m_list=[0, 0, 0, 0])

    # Single-expert hot.
    run_case("hot_one", E=4, M_max=64, H=128, N=64, masked_m_list=[64, 1, 0, 1])

    print("All triton vs torch masked-grouped-gemm parity tests PASSED.")


if __name__ == "__main__":
    main()
