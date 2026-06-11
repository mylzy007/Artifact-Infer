"""CombineEPHT — Expert-Parallel High-Throughput combine (eager-only).

Inverse of DispatchEPHT.

Inputs:
  expert_out: [total_recv, 1, H]  (or [total_recv, H])  bf16
  tok_meta : TokMetaEPHT (kept on this rank since DispatchEPHT)

Steps:
  1. Reverse all_to_all_single on expert_out using REVERSED counts:
     output_split_sizes=send_counts, input_split_sizes=recv_counts.
     -> rev_perm[T*K, H] in the SAME sorted-by-target-rank order we had on send.
  2. Un-permute: gather by sort_perm to get back [T, K, H] layout via index.
  3. Weight-and-reduce: out[t] = sum_k topk_weights[t, k] * unperm[t, k]

Optional combine-side compression (eager only):

  Config / engine kwarg:
    moe_combine_compress_keep_frac : float    default 0.0  (off)
    moe_combine_compress_fp8       : bool     default True
    moe_combine_compress_use_l2    : bool     default True

  Per-call env-var override (for quick A/B inside one engine load, no rebuild):
    MOE_COMBINE_COMPRESS_KEEP_FRAC=<f in (0, 1)>   overrides keep_frac
    MOE_COMBINE_COMPRESS_FP8=0/1                   overrides fp8
    MOE_COMBINE_COMPRESS_NO_L2=0/1                 1 = skip L2 rescale (naive top-k)

  Effective values per forward = env if set, else constructor default.

Phase 5: per-row SELECTIVE compression (env-var only, accuracy measurement mode).

  When MOE_COMBINE_SELECT_MODE is set to a non-default value, the dense reverse
  a2a is kept (wire format unchanged) but the lossy round-trip (sparsify + L2
  rescale + FP8) is applied ONLY to rows selected by the strategy. This lets us
  measure E1/E2-style "selective compression" task accuracy without changing
  the on-the-wire payload format.

    MOE_COMBINE_SELECT_MODE  ∈ {"uniform" (default), "row_weight", "row_norm"}
      uniform     — all rows compressed at keep_frac (original behavior)
      row_weight  — compress the lowest-`select_frac` rows by routing weight
                    (uses tok_meta.recv_topk_w_real). Approximates E2.
      row_norm    — compress the lowest-`select_frac` rows by recv_hidden L2
                    norm. Approximates E1 hidden_norm.
    MOE_COMBINE_SELECT_FRAC  ∈ float [0, 1]    fraction of rows to compress
                                               (default 1.0 = compress all)

When enabled, each row of expert_out is replaced by its top-k magnitude
sparsification (default with per-row L2 rescale; FP8 round-trip on the kept
values by default). The reverse all-to-all moves two or three smaller
payloads (indices / values / optional scale, all bit-cast to uint8 so NCCL is
happy with the dtype) instead of one dense [N, H] bf16 payload. The post-decode
rev has the same dense [send_total, H] shape, so the rest of combine is
unchanged.

Algorithm matches the Phase-A/B/C combine-side sparsification sweep in the HF
reference model (eval/compression/stage0_combine_*.py): top-k magnitude
sparsifier with optional per-row L2 norm rescale on the kept values and
optional per-row FP8 E4M3 quantization with amax/448 scaling.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn

from src.core.artifact import Artifact
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ht import TokMetaEPHT
from workshop.nanovllm_moe.services.utils.parallel import (
    get_ep_group,
    get_ep_world_size,
    get_runtime_mode,
    get_tp_group,
)

FP8_E4M3_MAX = 448.0


def _resolve_selective_mode() -> tuple[str, float]:
    """Return (mode, frac). mode in {'uniform', 'row_weight', 'row_norm'};
    frac is fraction of rows to compress (when mode != 'uniform')."""
    mode = os.environ.get("MOE_COMBINE_SELECT_MODE", "uniform").lower()
    if mode not in ("uniform", "row_weight", "row_norm"):
        mode = "uniform"
    try:
        frac = float(os.environ.get("MOE_COMBINE_SELECT_FRAC", "1.0"))
    except ValueError:
        frac = 1.0
    return mode, max(0.0, min(1.0, frac))


def _resolve_compression(
    cfg_keep_frac: float, cfg_fp8: bool, cfg_use_l2: bool,
) -> tuple[float, bool, bool]:
    """Effective (keep_frac, use_fp8, use_l2). Constructor values from Config are
    the default; per-call env vars override them so the knob can still be
    flipped at runtime without re-instantiating the engine. Useful for quick
    A/B comparisons within a single engine load."""
    env_keep_frac = os.environ.get("MOE_COMBINE_COMPRESS_KEEP_FRAC")
    if env_keep_frac is not None and env_keep_frac != "":
        try:
            keep_frac = float(env_keep_frac)
        except ValueError:
            keep_frac = cfg_keep_frac
    else:
        keep_frac = float(cfg_keep_frac)
    env_fp8 = os.environ.get("MOE_COMBINE_COMPRESS_FP8")
    use_fp8 = (env_fp8 == "1") if env_fp8 is not None and env_fp8 != "" else bool(cfg_fp8)
    env_no_l2 = os.environ.get("MOE_COMBINE_COMPRESS_NO_L2")
    if env_no_l2 is not None and env_no_l2 != "":
        use_l2 = env_no_l2 != "1"
    else:
        use_l2 = bool(cfg_use_l2)
    return keep_frac, use_fp8, use_l2


def _bytecast(t: torch.Tensor) -> torch.Tensor:
    """View any contiguous tensor as a 1D uint8 byte buffer. NCCL all_to_all_single
    is happy with uint8 on all backends; int16 / float8_e4m3fn / fp16 may not be.

    Returns a 1D uint8 view of the same memory; no copy."""
    return t.contiguous().view(torch.uint8).view(-1)


def _sparsify_for_send(
    x: torch.Tensor,                # [N_rows, H]   bf16
    k: int,
    use_fp8: bool,
    use_l2: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, int, int]:
    """Return uint8 byte buffers for (indices, values, optional scale) plus the
    per-row byte counts so the reverse a2a can split correctly.

    Per-row recipe:
      1. top-k magnitude. Gather kept values and their column indices.
      2. (optional) rescale kept values by ||x_full||_2 / ||x_sparse||_2  -> L2 preservation.
      3. (optional) per-row amax scaling -> FP8 E4M3 cast. The amax scale is sent
         alongside so the receiver can dequantize.

    Tuple layout:
      (indices_u8_flat, values_u8_flat, scale_u8_flat_or_None,
       idx_bytes_per_row, val_bytes_per_row, scale_bytes_per_row)
    where scale_bytes_per_row is 0 when FP8 is disabled.
    """
    N, H = x.shape
    if N == 0:
        empty = torch.empty(0, dtype=torch.uint8, device=x.device)
        idx_bpr = 2 * k
        val_bpr = k if use_fp8 else 2 * k
        scl_bpr = 2 if use_fp8 else 0
        return (
            empty,
            empty,
            empty if use_fp8 else None,
            idx_bpr,
            val_bpr,
            scl_bpr,
        )

    xf = x.float()
    abs_x = xf.abs()
    _, topk_idx = abs_x.topk(k, dim=-1)                        # [N, k] int64
    kept = xf.gather(-1, topk_idx)                              # [N, k]
    if use_l2:
        full_n2 = (xf ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        sparse_n2 = (kept ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        kept = kept * (full_n2 / sparse_n2)
    indices_i16 = topk_idx.to(torch.int16)                      # [N, k] int16
    idx_u8 = _bytecast(indices_i16)                             # [N * 2k]
    if use_fp8:
        amax = kept.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = (amax / FP8_E4M3_MAX).squeeze(-1).to(torch.float16)         # [N]
        scaled = (kept / amax) * FP8_E4M3_MAX
        values_fp8 = scaled.to(torch.float8_e4m3fn)                         # [N, k]
        val_u8 = _bytecast(values_fp8)                                       # [N * k]
        scl_u8 = _bytecast(scale)                                            # [N * 2]
        return idx_u8, val_u8, scl_u8, 2 * k, k, 2
    else:
        values_bf16 = kept.to(torch.bfloat16)                                # [N, k]
        val_u8 = _bytecast(values_bf16)                                      # [N * 2k]
        return idx_u8, val_u8, None, 2 * k, 2 * k, 0


def _reverse_ep_a2a_bytes(
    send_u8_flat: torch.Tensor,           # 1D uint8 of length sum(send_counts) * bytes_per_token
    send_counts: list[int],
    recv_counts: list[int],
    bytes_per_token: int,
    world_size: int,
) -> torch.Tensor:
    """Reverse-direction all_to_all on a flat uint8 byte buffer. Returns recv
    buffer of length sum(recv_counts) * bytes_per_token."""
    total_recv_bytes = sum(recv_counts) * bytes_per_token
    if total_recv_bytes == 0:
        return torch.empty(0, dtype=torch.uint8, device=send_u8_flat.device)
    recv = torch.empty(total_recv_bytes, dtype=torch.uint8, device=send_u8_flat.device)
    if world_size > 1:
        dist.all_to_all_single(
            recv,
            send_u8_flat.contiguous(),
            output_split_sizes=[c * bytes_per_token for c in recv_counts],
            input_split_sizes=[c * bytes_per_token for c in send_counts],
            group=get_ep_group(),
        )
    else:
        recv.copy_(send_u8_flat)
    return recv


def _sparsify_lossy_roundtrip(
    x: torch.Tensor,                # [N_rows, H]   bf16
    k: int,
    use_fp8: bool,
    use_l2: bool,
) -> torch.Tensor:
    """Apply the topk_l2 + (optional) FP8 sparsification + reconstruction as a
    dense in-place lossy round-trip. Output has SAME shape/dtype as input.

    Used by Phase-5 selective-compression mode where we want to measure the
    end-to-end accuracy of "compress these specific rows" without changing the
    on-the-wire payload format. The wire still carries dense bf16; we just
    pass each selected row through the same lossy transform first."""
    if x.numel() == 0:
        return x
    N, H = x.shape
    xf = x.float()
    abs_x = xf.abs()
    _, topk_idx = abs_x.topk(k, dim=-1)
    kept = xf.gather(-1, topk_idx)
    if use_l2:
        full_n2 = (xf ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        sparse_n2 = (kept ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        kept = kept * (full_n2 / sparse_n2)
    if use_fp8:
        amax = kept.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        scaled = (kept / amax) * FP8_E4M3_MAX
        kept_fp8 = scaled.to(torch.float8_e4m3fn)
        kept = (kept_fp8.to(torch.float32) / FP8_E4M3_MAX) * amax
    out = torch.zeros(N, H, dtype=torch.float32, device=x.device)
    out.scatter_(-1, topk_idx, kept)
    return out.to(x.dtype)


def _selective_compress_mask(
    tok_meta: TokMetaEPHT,
    expert_out: torch.Tensor,
    select_mode: str,
    select_frac: float,
) -> torch.Tensor | None:
    """Return bool mask [total_recv] — True = this row should be compressed.

    None if select_mode == 'uniform' (caller should use the full uniform path)."""
    if select_mode == "uniform":
        return None
    total_recv = expert_out.shape[0]
    device = expert_out.device
    if total_recv == 0 or select_frac <= 0.0:
        return torch.zeros(total_recv, dtype=torch.bool, device=device)
    if select_frac >= 1.0:
        return torch.ones(total_recv, dtype=torch.bool, device=device)
    if select_mode == "row_weight":
        scores = tok_meta.recv_topk_w_real
        if scores is None:
            return torch.zeros(total_recv, dtype=torch.bool, device=device)
        scores = scores.to(device).float()
    elif select_mode == "row_norm":
        if tok_meta.recv_hidden is None or tok_meta.recv_hidden.numel() == 0:
            return torch.zeros(total_recv, dtype=torch.bool, device=device)
        scores = tok_meta.recv_hidden.float().norm(dim=-1)
    else:
        return torch.zeros(total_recv, dtype=torch.bool, device=device)
    # Compress the LOWEST `select_frac` fraction (small score => less important).
    n_compress = int(round(total_recv * select_frac))
    n_compress = max(1, min(total_recv - 1, n_compress))
    # kthvalue: smallest k values, threshold = the k-th value.
    thresh = torch.kthvalue(scores, n_compress).values
    return scores <= thresh


def _decode_at_recv(
    indices_u8_flat: torch.Tensor,        # [N * 2k] uint8
    values_u8_flat: torch.Tensor,         # [N * k] (FP8) or [N * 2k] (bf16) uint8
    scale_u8_flat: torch.Tensor | None,   # [N * 2] uint8 (FP8) or None (bf16)
    N: int,
    H: int,
    k: int,
    use_fp8: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    device = indices_u8_flat.device
    if N == 0:
        return torch.empty(0, H, dtype=dtype, device=device)
    # Bitcast back to typed views.
    indices = indices_u8_flat.view(torch.int16).view(N, k).to(torch.int64)
    if use_fp8:
        assert scale_u8_flat is not None
        vals_fp8 = values_u8_flat.view(torch.float8_e4m3fn).view(N, k)
        vals_f32 = vals_fp8.to(torch.float32)
        scale_f32 = scale_u8_flat.view(torch.float16).view(N, 1).to(torch.float32)
        vals = vals_f32 * scale_f32
    else:
        vals = values_u8_flat.view(torch.bfloat16).view(N, k).to(torch.float32)
    out = torch.zeros(N, H, dtype=torch.float32, device=device)
    out.scatter_(-1, indices, vals)
    return out.to(dtype)


class CombineEPHT(Artifact, nn.Module):
    @property
    def name(self) -> str:
        return "CombineEPHT"

    def __init__(
        self,
        hidden_size: int,
        top_k: int,
        compress_keep_frac: float = 0.0,
        compress_fp8: bool = True,
        compress_use_l2: bool = True,
    ) -> None:
        super().__init__()
        self.H = int(hidden_size)
        self.K = int(top_k)
        self.world_size = get_ep_world_size() if dist.is_initialized() else 1
        self.runtime_mode = get_runtime_mode()
        # Config-time compression defaults. Env vars MOE_COMBINE_COMPRESS_*
        # override at runtime so a single engine instance can be A/B-tested.
        self.compress_keep_frac = float(compress_keep_frac)
        self.compress_fp8 = bool(compress_fp8)
        self.compress_use_l2 = bool(compress_use_l2)

    def _debug_sync(self, stage: str) -> None:
        if os.environ.get("MOE_EPHT_DEBUG_SYNC", "0") == "1":
            torch.cuda.synchronize()

    def forward(
        self,
        expert_out: torch.Tensor,    # [total_recv, 1, H]   or   [total_recv, H]
        tok_meta: TokMetaEPHT,
    ) -> torch.Tensor:
        N = self.world_size
        H = self.H
        # Phase 4 P1: when router K_eff < self.K, tok_meta.topk_weights has K_eff
        # columns. Read the actual K from tok_meta so combine matches dispatch.
        K = int(tok_meta.topk_weights.shape[1])
        T = tok_meta.T_local
        device = expert_out.device
        dtype = expert_out.dtype

        # Squeeze the K=1 dim if present.
        if expert_out.ndim == 3:
            assert expert_out.shape[1] == 1, (
                f"EP-HT expert_out should be [total_recv, 1, H], got {expert_out.shape}"
            )
            expert_out = expert_out.squeeze(1)   # [total_recv, H]

        total_recv = expert_out.shape[0]
        send_total = sum(tok_meta.send_counts)

        # 1. Reverse a2a: send back per-recv-rank slices.
        keep_frac, use_fp8, use_l2 = _resolve_compression(
            self.compress_keep_frac, self.compress_fp8, self.compress_use_l2,
        )
        compress_on = 0.0 < keep_frac < 1.0
        select_mode, select_frac = _resolve_selective_mode()
        # Phase-5 selective mode: apply lossy round-trip to a subset of rows,
        # but keep the wire format DENSE bf16 so we measure pure accuracy of
        # "compress these rows" without needing a per-row ragged packer.
        if compress_on and select_mode != "uniform":
            k_keep = max(1, min(H, int(round(H * keep_frac))))
            mask = _selective_compress_mask(tok_meta, expert_out, select_mode, select_frac)
            if mask is not None and mask.any():
                # Apply lossy round-trip on selected rows; leave others untouched.
                # Working on a clone to keep autograd / non-overwrite semantics if any caller cares.
                expert_out = expert_out.clone()
                rows_to_compress = expert_out[mask]
                expert_out[mask] = _sparsify_lossy_roundtrip(
                    rows_to_compress, k_keep, use_fp8, use_l2,
                )
            # Fall through to the dense bf16 a2a (compress_on disabled below).
            compress_on = False
        if compress_on:
            k_keep = max(1, min(H, int(round(H * keep_frac))))
            (
                idx_send, val_send, scl_send,
                idx_bpr, val_bpr, scl_bpr,
            ) = _sparsify_for_send(expert_out, k_keep, use_fp8, use_l2)
            # Three reverse a2a calls on uint8 byte buffers. Counts are per-token;
            # reverse direction => input_split = recv_counts, output_split = send_counts
            # (sending the *outputs* back to where the original tokens lived).
            idx_recv = _reverse_ep_a2a_bytes(
                idx_send, tok_meta.recv_counts, tok_meta.send_counts,
                bytes_per_token=idx_bpr, world_size=N,
            )
            val_recv = _reverse_ep_a2a_bytes(
                val_send, tok_meta.recv_counts, tok_meta.send_counts,
                bytes_per_token=val_bpr, world_size=N,
            )
            if use_fp8 and scl_send is not None:
                scl_recv = _reverse_ep_a2a_bytes(
                    scl_send, tok_meta.recv_counts, tok_meta.send_counts,
                    bytes_per_token=scl_bpr, world_size=N,
                )
            else:
                scl_recv = None
            rev = _decode_at_recv(
                idx_recv, val_recv, scl_recv,
                N=send_total, H=H, k=k_keep,
                use_fp8=use_fp8, dtype=dtype,
            )
        else:
            rev = torch.empty(send_total, H, dtype=dtype, device=device)
            if N > 1:
                dist.all_to_all_single(
                    rev, expert_out.contiguous(),
                    output_split_sizes=tok_meta.send_counts,
                    input_split_sizes=tok_meta.recv_counts,
                    group=get_ep_group(),
                )
            else:
                rev.copy_(expert_out)
        self._debug_sync("reverse_a2a")

        # owner_local_ep always combines locally on the original token owner.
        if self.runtime_mode == "vllm_dp_ep" and not tok_meta.is_source_leader:
            out = torch.empty(T, H, dtype=dtype, device=device)
        else:
            # zeros (not empty): when Phase 4 drop is enabled, sort_perm only
            # covers kept (token, expert) positions and the remaining slots must
            # contribute 0 to the weighted sum. When drop is off, sort_perm
            # spans every (t, k) and the zero init is overwritten in full.
            unperm = torch.zeros(T * K, H, dtype=dtype, device=device)
            unperm[tok_meta.sort_perm] = rev
            unperm_TKH = unperm.view(T, K, H)
            weights = tok_meta.topk_weights.to(dtype).unsqueeze(-1)  # [T, K, 1]
            out = (unperm_TKH * weights).sum(dim=1)                   # [T, H]

        # vllm_dp_ep: leader combines, then broadcasts within TP replica.
        # owner_local_ep: tp=1, so no output broadcast is allowed or needed.
        if self.runtime_mode == "vllm_dp_ep" and dist.is_initialized() and dist.get_world_size() > 1 and T > 0:
            dist.broadcast(out, src=tok_meta.source_leader_global_rank, group=get_tp_group())
        self._debug_sync("reduce")
        return out
