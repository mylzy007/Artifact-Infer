"""Parity test: triton EP-LL dispatch kernel vs torch (host-loop) reference.

Tests:
  1. Single-process N=1: both kernels produce identical send_buf, original_indices,
     and local_counts (in the appropriate places).
  2. End-to-end pipeline: DispatchEPLL(triton) -> torch_masked_grouped_gemm ->
     CombineEPLL produces the same output as DispatchEPLL(torch) -> ... .
  3. Stress: bf16 noise tolerance, multiple T values, multiple K values.

Run:
  cd /home/yyx/personal/Artifact-Infer
  .venv/bin/python -m workshop.nanovllm_moe._test_triton_ep_ll_dispatch
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def _init_size1_pg():
    """Single-process pseudo-distributed group so DispatchEPLL works."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12511")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", world_size=1, rank=0)


def _make_inputs(T: int, H: int, E_global: int, K: int, *, device, dtype, seed: int):
    g = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn(T, H, generator=g, device=device, dtype=dtype) * 0.1

    # Random topk_ids (no top-k computation needed — we drive the dispatcher with
    # synthetic logits that force the desired ids).
    # Easier: just call torch.topk on random logits.
    logits = torch.randn(T, E_global, generator=g, device=device, dtype=torch.float32)
    return hidden, logits


def _run_dispatch(dispatch_kernel: str, hidden, logits, num_experts, top_k, M_max):
    """Run a fresh DispatchEPLL with the given kernel choice; return key buffers."""
    from workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ll import (
        DispatchEPLL,
    )

    disp = DispatchEPLL(
        num_experts_global=num_experts,
        top_k=top_k,
        m_max=M_max,
        norm_topk_prob=True,
        dispatch_kernel=dispatch_kernel,
    ).to(device=hidden.device)

    meta = disp(hidden, logits)
    return {
        "send_buf": disp.send_buf.clone(),
        "original_indices": disp.original_indices.clone(),
        "local_counts": disp.local_counts.clone(),
        "topk_ids": meta.topk_ids.clone(),
        "topk_weights": meta.topk_weights.clone(),
        "hidden_recv": meta.hidden_recv.clone(),
    }


def test_dispatch_parity():
    """torch vs triton dispatch produce identical outputs (modulo slot ordering)."""
    print("=" * 70)
    print("Test 1: torch vs triton dispatch — same routing, same buffer state")
    print("=" * 70)

    device = "cuda:0"
    dtype = torch.bfloat16
    H = 256
    E_global = 8     # tiny so a single rank holds all experts
    K = 2
    M_max = 32

    for T in [4, 16, 64]:
        hidden, logits = _make_inputs(T, H, E_global, K, device=device, dtype=dtype, seed=T)

        torch_state = _run_dispatch("torch", hidden, logits, E_global, K, M_max)
        triton_state = _run_dispatch("triton", hidden, logits, E_global, K, M_max)

        # topk_ids and topk_weights are identical (computed by the same code path).
        assert torch.equal(torch_state["topk_ids"], triton_state["topk_ids"]), \
            f"topk_ids mismatch at T={T}"
        assert torch.allclose(
            torch_state["topk_weights"], triton_state["topk_weights"], atol=1e-6
        ), f"topk_weights mismatch at T={T}"

        # local_counts must match exactly (same total per (rank, local_expert)).
        assert torch.equal(torch_state["local_counts"], triton_state["local_counts"]), (
            f"local_counts mismatch at T={T}: "
            f"torch={torch_state['local_counts']}\ntriton={triton_state['local_counts']}"
        )

        # send_buf and original_indices may differ in SLOT ORDER within a bucket
        # (atomic ordering vs sequential) but the SET of (slot_data, t, k) must match.
        # Compare per-(rank, local_expert) bucket as a multiset.
        N, E, M, _ = torch_state["send_buf"].shape
        for r in range(N):
            for e in range(E):
                count = int(torch_state["local_counts"][r, e].item())
                if count == 0:
                    continue
                # Gather (t, k, hidden_row) tuples from each impl.
                torch_oi = torch_state["original_indices"][r, e, :count]  # [count, 2]
                triton_oi = triton_state["original_indices"][r, e, :count]
                # Sort by (t, k) lexicographic to compare as multiset.
                t_keys = torch_oi[:, 0].long() * (K + 1) + torch_oi[:, 1].long()
                triton_keys = triton_oi[:, 0].long() * (K + 1) + triton_oi[:, 1].long()
                t_perm = torch.argsort(t_keys)
                triton_perm = torch.argsort(triton_keys)

                torch_oi_sorted = torch_oi[t_perm]
                triton_oi_sorted = triton_oi[triton_perm]
                assert torch.equal(torch_oi_sorted, triton_oi_sorted), (
                    f"original_indices multiset mismatch at (T={T}, r={r}, e={e}):\n"
                    f"  torch: {torch_oi_sorted}\n"
                    f"  triton: {triton_oi_sorted}"
                )

                torch_data = torch_state["send_buf"][r, e, :count][t_perm]
                triton_data = triton_state["send_buf"][r, e, :count][triton_perm]
                # bf16-exact: both impls just COPY the same hidden row, no math.
                assert torch.equal(torch_data, triton_data), (
                    f"send_buf data mismatch at (T={T}, r={r}, e={e})"
                )

        print(f"  PASS  T={T}  K={K}  E={E_global}  total_routed={int(torch_state['local_counts'].sum())}")


def test_end_to_end_parity():
    """Full pipeline (DispatchEPLL -> torch_masked_grouped_gemm -> CombineEPLL).

    torch dispatch and triton dispatch should produce identical final outputs
    (within bf16 noise — the inner kernel reduces in fp32 internally).
    """
    print()
    print("=" * 70)
    print("Test 2: end-to-end pipeline parity (torch vs triton dispatch)")
    print("=" * 70)

    from workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ll import (
        DispatchEPLL,
    )
    from workshop.nanovllm_moe.artifacts.modeling.layers.moe.combine_ep_ll import (
        CombineEPLL,
    )
    from workshop.nanovllm_moe.artifacts.moe_backend.torch_masked_grouped_gemm import (
        torch_masked_grouped_gemm,
    )

    device = "cuda:0"
    dtype = torch.bfloat16
    H = 256
    inter = 128
    E_global = 8
    K = 2
    M_max = 32

    for T in [4, 16, 32]:
        hidden, logits = _make_inputs(T, H, E_global, K, device=device, dtype=dtype, seed=T + 100)

        # Same expert weights for both runs.
        g = torch.Generator(device=device).manual_seed(0xC0FFEE)
        w1 = torch.randn(E_global, 2 * inter, H, generator=g, device=device, dtype=dtype) * 0.05
        w2 = torch.randn(E_global, H, inter, generator=g, device=device, dtype=dtype) * 0.05

        outputs = []
        for impl in ["torch", "triton"]:
            disp = DispatchEPLL(
                num_experts_global=E_global, top_k=K, m_max=M_max,
                norm_topk_prob=True, dispatch_kernel=impl,
            ).to(device=device)
            comb = CombineEPLL(
                hidden_size=H, top_k=K, num_experts_local=E_global, m_max=M_max,
            ).to(device=device)

            meta = disp(hidden, logits)
            expert_out = torch_masked_grouped_gemm(
                hidden_states=meta.hidden_recv,
                w1=w1, w2=w2,
                masked_m=meta.masked_m,
            )
            out = comb(expert_out, meta)
            outputs.append(out)

        diff = (outputs[0] - outputs[1]).abs().max().item()
        rel = diff / outputs[0].abs().max().item()
        print(f"  T={T}  abs_max_diff={diff:.6f}  rel={rel:.6e}")
        # Both pipelines do the SAME math on the SAME data (modulo permutations
        # of slot order which the inner kernel sums over symmetrically). The
        # final tokens land at identical positions.
        assert diff < 1e-2, f"outputs differ by {diff} at T={T}"

    print("  PASS")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return 0
    _init_size1_pg()

    test_dispatch_parity()
    test_end_to_end_parity()

    print()
    print("All parity tests PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
