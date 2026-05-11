# MoE on top of `nanovllm_base` — Design

> Status: pre-implementation design sketch (v2 — block-tiled Triton path; FlashInfer paths deferred to F1).
> Target workshop variant: `workshop/nanovllm_moe/`.
> Source review the design borrows from: `sglang/python/sglang/srt/layers/moe/` and `vllm/vllm/model_executor/layers/fused_moe/`.
> Hardware target: RTX 4090 (sm89, Ada). See §2.1 for why FlashInfer MoE kernels are not reachable on this arch.

This document captures the locked decisions, the resulting module layout, the per-call data flow, the metadata that crosses the engine ↔ layer boundary, and a short orientation on how Expert Parallel (EP) re-uses the same `Dispatch` / `Experts` / `Combine` skeleton — so we can later swap the single-rank backend for an EP one (or for a FlashInfer kernel on Hopper/Blackwell) without re-architecting.

The goal is the **smallest possible delta against `workshop/nanovllm_base/`**: copy the tree, add a `moe_backend` artifact and a `moe/` layer subtree, and make the `model_runner` register the new backend the same way it already registers attention.

---

## 1. Locked decisions

| Q | Decision | Implication |
|---|---|---|
| Q1 — checkpoint | `~/models/Qwen3-30B-A3B` | bf16, 128 experts, top-k 8, hidden 2048, moe_intermediate 768, 48 layers, all-MoE (no `mlp_only_layers`), `norm_topk_prob = true` |
| Q2 — quantization | **No FP4** | FlashInfer's `cute_dsl.blockscaled_gemm.grouped_gemm_nt_masked` requires FP4 weights → cannot be used. See §2. |
| Q3 — permute / kernel backend | Both `torch` and `triton`; ship `torch` first | One artifact, two interchangeable backend functions chosen by config flag. |
| Q4 — buffer ownership | **Full registration** through the orchestrator | Workspace tensors live on the `MoeBackend` artifact and are propagated as `StateCell`s onto every `Dispatch` / `Experts` / `Combine` instance. |
| Q5 — `Combine` shape | `nn.Module` (not free function) | Symmetric reads in the parent (see §6) and Q4 also requires it (so the orchestrator can register cells onto it). |
| Q6 — inner kernel choice | **Adopt the vllm/sglang Triton `fused_moe` path (block-tiled metadata)** | Rules out the masked-m layout the original cutedsl plan used. Different metadata contract — see §2.2. |

---

## 2. Backend kernel choice

### 2.1 Why no FlashInfer kernel on RTX 4090

We checked all four FlashInfer MoE entry points against the Ada (sm89) target:

| Kernel | Required arch | Runs on RTX 4090? |
|---|---|---|
| `flashinfer.fused_moe.cutlass_fused_moe` | sm90 / sm100 / sm110 / sm120 (vllm `_supports_current_device`) | No |
| `flashinfer.fused_moe.trtllm_bf16_moe` | sm100+ (`get_trtllm_moe_sm100_module`) | No |
| `flashinfer.cute_dsl.blockscaled_gemm.grouped_gemm_nt_masked` | sm100, FP4 hardware | No (also FP4) |
| `flashinfer.deep_gemm.m_grouped_fp8_gemm_nt_masked` | sm90, FP8 hardware | No (also FP8) |

The local FlashInfer build agrees — `cutlass_fused_moe`'s `else` (unquantized) branch in vllm exists but `is_valid_flashinfer_cutlass_fused_moe` (`vllm/.../flashinfer_cutlass_moe.py:36-60`) gates it on `w1.dtype == torch.uint8` and `_supports_current_device` gates the whole class on sm90+. So FlashInfer is not reachable here.

### 2.2 Chosen kernel: vllm/sglang Triton `fused_moe` (block-tiled)

We use the same Triton kernel that vllm and sglang use as their reference path. It is sm75+, bf16-friendly, autotuned, and battle-tested. The metadata contract is:

| Tensor | Shape | Produced by |
|---|---|---|
| `topk_weights`, `topk_ids` | `[T, K]`, `[T, K]` | `sgl_kernel.topk_softmax` (or torch fallback) |
| `sorted_token_ids` | `[T*K + pad]` | `sgl_kernel.moe_align_block_size` |
| `expert_ids` | `[(T*K + pad) // BLOCK_M]` | same |
| `num_tokens_post_padded` | `[1]` (scalar tensor) | same |
| `intermediate_cache1` | `[T, K, 2N]` workspace | `MoeBackend` |
| `intermediate_cache2` | `[T, K, N]` workspace | `MoeBackend` |
| `intermediate_cache3` | `[T, K, H]` workspace | `MoeBackend` |
| `out` | `[T, H]` | reduction over K |

This replaces the `[E, M_cap, H] + masked_m` contract from v1 of the doc. Note: **the kernel itself folds permute and unpermute into the GEMM**: it reads `hidden_states[sorted_token_ids // K]` inside the first GEMM and writes to per-`(token, k)` rows in the second GEMM with `topk_weights` already multiplied in. So `Dispatch` does *not* materially permute hidden_states; it only computes the index tensors. `Combine` does *not* materially unpermute; it only reduces over the K axis.

Available building blocks (already installed at `.venv/.../sgl_kernel/`):
- `sgl_kernel.moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum_buffer)` — C++/CUDA op, in-place.
- `sgl_kernel.topk_softmax(topk_weights, topk_ids, gating_output, renormalize, ...)` — C++/CUDA op, in-place.
- `sgl_kernel.moe_sum(input, output)` — C++/CUDA per-token sum over the K axis.

What we vendor (≈ 250 LOC, bf16-only, no quant variants):
- The `@triton.jit` `fused_moe_kernel` and its launcher `invoke_fused_moe_kernel(...)` — adapted from `sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py` (which itself is adapted from vllm). We strip out FP4/FP8/AWQ/GPTQ branches and keep only the bf16 path.
- A trimmed `try_get_optimal_moe_config(...)` that returns a fixed config (`BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_warps=4, num_stages=3`) — autotuning is a follow-up.

### 2.3 Two interchangeable inner-kernel implementations (Q3)

Per Q3, `MoeBackend` exposes two interchangeable `run_experts` bodies under one signature:

1. **`torch_fused_moe`** (MVP / reference): pure-torch sort-based permute + `torch.bmm` + activation + `torch.bmm` + `index_add_` reduce. Slow, but the reference implementation against which the Triton path is correctness-tested.
2. **`triton_fused_moe`** (default once tested): vendored `invoke_fused_moe_kernel` × 2 around `silu_and_mul`, then `sgl_kernel.moe_sum` for the K-axis reduction.

Both share the same buffer contract (`intermediate_cache{1,2,3}` + `sorted_token_ids` + `expert_ids` + `num_tokens_post_padded`), so the swap is one line in `MoeBackend.__init__`.

### 2.4 Trade-off note vs the cutedsl masked-m layout

The block-tiled layout is what vllm/sglang use for *non-EP local* MoE. It is **not** the EP-LL layout — DeepEP-LL produces `[E_local, M_cap, H] + masked_m`, which is the cutedsl/CuteDSL contract. So:

- Today (single rank, RTX 4090): block-tiled is the right choice. Faster to ship, no custom kernels, no EP code.
- When/if we add EP-LL or move to Hopper/Blackwell with FlashInfer: we'll need a *second* `MoeBackend` impl on the masked-m contract. The orchestration stays the same (Dispatch / Experts / Combine, full registration) but `Dispatch.permute` and `Combine.unpermute` swap their metadata structure.

This is captured as F1/F2 in §13.

---

## 3. Module layout

```
workshop/nanovllm_moe/
├── artifacts/
│   ├── attention_backend/                # ← copied verbatim
│   │   └── flashinfer_attention.py
│   ├── moe_backend/                      # ← NEW
│   │   ├── __init__.py                   # MoeBackend factory; selects torch | triton impl
│   │   ├── moe_backend.py                # MoeBackend (Artifact): owns workspaces + per-batch prep
│   │   ├── torch_fused_moe.py            # bf16 reference (sort + bmm + silu_and_mul + bmm + index_add_)
│   │   └── triton_fused_moe.py           # vendored bf16 Triton fused_moe kernel + launcher
│   ├── modeling/
│   │   ├── layers/
│   │   │   ├── activation.py             # ← copied
│   │   │   ├── embed_head.py             # ← copied
│   │   │   ├── layernorm.py              # ← copied
│   │   │   ├── linear.py                 # ← copied
│   │   │   ├── rotary_embedding.py       # ← copied
│   │   │   ├── sampler.py                # ← copied
│   │   │   └── moe/                      # ← NEW
│   │   │       ├── __init__.py           # exports FusedMoE + (Dispatch, Experts, Combine)
│   │   │       ├── dispatch.py           # Dispatch: gate + topk_softmax + moe_align_block_size
│   │   │       ├── experts.py            # Experts: thin wrapper around MoeBackend.run_experts
│   │   │       ├── combine.py            # Combine: K-axis reduction (sgl_kernel.moe_sum or torch.sum)
│   │   │       └── fused_moe.py          # FusedMoE = Dispatch + Experts + Combine
│   │   └── models/
│   │       ├── qwen3.py                  # ← copied
│   │       └── qwen3_moe.py              # ← NEW (Qwen3MoE wrapping FusedMoE)
│   └── block_mngr/                       # ← copied
└── services/
    ├── config.py                         # ← + one knob: `moe_impl: "torch" | "triton" = "triton"`
    ├── engine/                           # ← copied
    ├── llm.py                            # ← copied
    ├── sampling_params.py                # ← copied
    ├── model_runner/
    │   └── model_runner.py               # ← MODIFIED
    └── utils/
        ├── context.py                    # ← MODIFIED (adds 4 fields, see §5)
        └── loader.py                     # ← MODIFIED (one branch for stacked MoE weights)
```

Net delta vs. `nanovllm_base`:
- `services/config.py`: +1 field.
- `services/utils/context.py`: +4 fields.
- `services/utils/loader.py`: +1 branch for `experts.{i}.{w}_proj` → stacked `(E, …)`.
- `services/model_runner/model_runner.py`: ~15 LOC for one extra artifact + one extra registration loop.
- All other code is net-new under `artifacts/moe_backend/` and `artifacts/modeling/layers/moe/`.

Nothing in `engine/scheduler.py`, `engine/sequence.py`, `block_mngr/`, or any existing layer changes.

---

## 4. Per-call data flow

```
hidden_states [T, H]
        │
        ▼
┌───────────────────── FusedMoE ─────────────────────────────────────────────┐
│                                                                            │
│   ┌──────────── Dispatch ──────────────────────────────────────────────┐   │
│   │ router_logits = gate(x)                       # [T, E]             │   │
│   │ topk_weights, topk_ids = topk_softmax(rl, K)  # [T, K], [T, K]     │   │
│   │ moe_align_block_size(topk_ids, BLOCK_M, E):                        │   │
│   │   writes  sorted_token_ids_buf  [T*K + pad]                        │   │
│   │   writes  expert_ids_buf        [(T*K + pad) // BLOCK_M]           │   │
│   │   writes  num_tokens_post_padded                                   │   │
│   │ → tok_meta = TokMeta(topk_weights, topk_ids,                       │   │
│   │                      sorted_token_ids, expert_ids,                 │   │
│   │                      num_tokens_post_padded)                       │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│   ┌──────────── Experts ────────────────────────────────────────────────┐   │
│   │ x_out_TKH = run_experts(hidden_states, w1, w2, tok_meta,            │   │
│   │                          intermediate_cache1/2/3)                   │   │
│   │   GEMM1 (gate+up):  hidden_states[sorted//K]  · w1[expert_ids]     │   │
│   │                     →  cache1 [T*K, 2N]                            │   │
│   │   silu_and_mul:     cache1 → cache2 [T*K, N]                       │   │
│   │   GEMM2 (down, weighted):  cache2 · w2[expert_ids]  * topk_weights │   │
│   │                            →  cache3 [T*K, H]                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│   ┌──────────── Combine ────────────────────────────────────────────────┐   │
│   │ moe_sum(cache3.view(T, K, H), out)            # [T, H]              │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
out [T, H]
```

Three things to internalize about this layout:

1. **`Dispatch` does not move hidden_states.** It only computes index tensors. The kernel reads `hidden_states[sorted_token_ids // K]` *internally* during GEMM1.
2. **`Experts` writes a `[T, K, H]`-shaped output**, *already weighted by `topk_weights`*. The per-route weighting is folded into GEMM2.
3. **`Combine` is just a sum across the K axis** — `sgl_kernel.moe_sum` (CUDA) or `cache3.sum(dim=1)` (torch fallback). No unpermute, no weighting.

The three submodules **never reference each other directly**; they communicate through:
- the **shared workspace tensors** owned by the `MoeBackend` artifact and registered onto each submodule (§5), and
- a small **`tok_meta` `NamedTuple`** that flows from `Dispatch` through `Experts` to `Combine`.

---

## 5. Metadata: who computes what, where it lives

### 5.1 Engine-level (ModelRunner → Context)

We add four fields to `services/utils/context.py` that compose with the existing ones:

```python
@dataclass
class Context:
    # === existing fields, unchanged ===
    is_prefill: bool = False
    cu_seqlens_q:  torch.Tensor | None = None
    cu_seqlens_k:  torch.Tensor | None = None
    max_seqlen_q:  int = 0
    max_seqlen_k:  int = 0
    slot_mapping:  torch.Tensor | None = None
    context_lens:  torch.Tensor | None = None
    block_tables:  torch.Tensor | None = None
    no_prefix:     bool | None = None

    # === NEW: MoE-relevant ===
    num_tokens:           int = 0          # = input_ids.size(0); read by Combine for view shape
    moe_t_cap:            int = 0          # static T capacity = max_num_batched_tokens
    moe_block_size_m:     int = 0          # static BLOCK_SIZE_M used by Triton kernel (default 64)
    moe_capture_buf_id:   int | None = None  # reserved for per-BS captured workspaces; None in MVP
```

Notes:
- `num_tokens` is set in both `prepare_prefill` and `prepare_decode` from `input_ids.size(0)`. It is only a convenience — `Combine` could derive it, but pulling it from `Context` keeps `Combine.forward()` shape-agnostic.
- `moe_t_cap` is **set once** by `MoeBackend.__post_init__` and never changes. It is `max_num_batched_tokens` and drives the size of `intermediate_cache{1,2,3}` and `sorted_token_ids_buf`.
- `moe_block_size_m` is the constant tile size used by the Triton kernel. Fixed at construction (default 64). `sgl_kernel.moe_align_block_size` requires this value.
- `moe_capture_buf_id` is reserved for a future variant that allocates one workspace per captured BS — it is `None` in the MVP.

### 5.2 Per-batch (ModelRunner → MoeBackend, before forward)

A single small method called from `prepare_prefill` / `prepare_decode` right next to `prepare_metadata_for_attn_*`:

```python
def prepare_metadata_for_moe(self, num_tokens: int):
    # The Triton kernel reads num_tokens_post_padded[0] to bound its grid,
    # so we must zero it before each batch (otherwise stale values leak across calls).
    self.num_tokens_post_padded.zero_()
    # sorted_token_ids_buf is filled by sgl_kernel.moe_align_block_size from valid prefix only,
    # but we initialize the unused tail to num_tokens (a "padding sentinel" the kernel ignores).
    # In practice sgl_kernel handles this internally — we just need the buffer present.
```

That is the **entire** "plan" step for MoE. Per-batch index tensors are written by `sgl_kernel.moe_align_block_size` *inside* `Dispatch.forward`, not here. CUDA-graph capture is trivial because all buffers are persistent and shape-static (see §8).

### 5.3 Per-call (inside `Dispatch.forward`)

These all use backend-owned buffers (registered via the orchestrator); `Dispatch` itself allocates nothing per call:

| Tensor | Shape | Where written |
|---|---|---|
| `router_logits` | `[T, E]` | `self.gate(x)` |
| `topk_weights` | `[T, K]` | `sgl_kernel.topk_softmax(...)` (with `renormalize=True` for `norm_topk_prob`) |
| `topk_ids` | `[T, K]` | `sgl_kernel.topk_softmax(...)` (int32) |
| `sorted_token_ids_buf[: T*K + pad]` | `[T_max*K + pad_max]` | `sgl_kernel.moe_align_block_size(...)` |
| `expert_ids_buf[: ...]` | `[(T_max*K + pad_max) // BLOCK_M]` | same call |
| `num_tokens_post_padded` | `[1]` (scalar tensor, int32) | same call |
| `cumsum_buf` | `[E + 1]` (scratch for moe_align) | same call |
| `tok_meta` | `NamedTuple` | `(topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded)` |

Outputs that escape `Dispatch`:
- All five fields of `tok_meta` are consumed by `Experts.forward`.
- `topk_ids` (and the implicit `num_tokens` from Context) are also consumed by `Combine.forward` for the K-axis reshape.

### 5.4 Per-call (inside `Experts.forward`)

`Experts` reads `tok_meta` plus the three intermediate caches and the stacked weights `w1` / `w2`:

| Tensor | Shape | Role |
|---|---|---|
| `intermediate_cache1` | `[T_max, K, 2N]` | GEMM1 output buffer |
| `intermediate_cache2` | `[T_max, K, N]` | post-`silu_and_mul` buffer |
| `intermediate_cache3` | `[T_max, K, H]` | GEMM2 output buffer (weighted by `topk_weights`) |

The Triton path is two `invoke_fused_moe_kernel(...)` calls bracketing one `silu_and_mul` call. The torch path uses `torch.bmm` with a manual sort/scatter-style permute and an explicit `* topk_weights` before the K-axis reduce. Both backends agree on the contract — they read the same buffers and write to `intermediate_cache3`.

### 5.5 Per-call (inside `Combine.forward`)

```python
def forward(self, tok_meta: TokMeta) -> torch.Tensor:
    T = get_context().num_tokens
    cache3 = self.intermediate_cache3[:T]      # [T, K, H]
    out = torch.empty((T, H), dtype=cache3.dtype, device=cache3.device)
    sgl_kernel.moe_sum(cache3, out)            # or: torch.sum(cache3, dim=1, out=out)
    return out
```

That is the entire body. `Combine` does not need `topk_weights` because GEMM2 already folded it in.

### 5.6 Weight loading (`services/utils/loader.py`)

The HF Qwen3-MoE checkpoint exposes per-expert per-projection tensors:

```
model.layers.{l}.mlp.gate.weight                       # routing
model.layers.{l}.mlp.experts.{e}.gate_proj.weight      # → stacked w1[e, 0:N]
model.layers.{l}.mlp.experts.{e}.up_proj.weight        # → stacked w1[e, N:2N]
model.layers.{l}.mlp.experts.{e}.down_proj.weight      # → stacked w2[e]
```

We add **one branch** in `load_model` that detects the `experts.{e}.{proj}` substring and dispatches to the per-expert weight loader on `Experts.w1` / `Experts.w2`. The existing `packed_modules_mapping` mechanism is enough to handle the gate/up packing along the expert dim.

---

## 6. Why `Combine` is an `nn.Module` (Q5 elaboration)

Three concrete reasons keep `Combine` as a Module rather than a free function:

1. **Symmetric read in the parent.** With Module, `FusedMoE.forward` reads as one rhythm:
   ```python
   x_in, masked_m, perm_meta = self.dispatch(hidden_states)
   x_out                     = self.experts(x_in, masked_m)
   out                       = self.combine(x_out, perm_meta)
   ```
   Each line is `self.X(...)` — the parent doesn't need to know which X is stateful and which isn't. With a free function the third line becomes `combine_experts(x_out, perm_meta, num_tokens=...)`, breaking the rhythm and forcing the caller to plumb `num_tokens` explicitly. (The Module variant reads `num_tokens` from `Context`.)

2. **Q4 requires it.** Since you opted for full registration of buffers, `Combine` must be a child of `nn.Module` for the orchestrator to walk it via `model.modules()` and register state cells (e.g. the `perm_meta` scratch buffer or, later, an EP combine workspace) onto it. A free function is invisible to `for module in self.model.modules()`.

3. **Future-proofing for EP / fused combine.** When we add EP, `Combine` becomes the side that owns the **reverse all-to-all workspace** and may hold a `comm_handle` — it stops being stateless. Keeping it as a Module from day one means the call site does not change when this swap happens; only `Combine.__init__` and `Combine.forward` do.

The cost is one extra `nn.Module` in `model.modules()`; for a 48-layer 128-expert model that's 48 extra Module instances. Negligible.

---

## 7. EP, briefly — and why our three-module shape is still EP-friendly

> This is informational; the MVP is single-rank. Block-tiled metadata (Q6) is **not** the EP-LL contract — but the *module-level* split into `Dispatch` / `Experts` / `Combine` matches both vllm's `prepare_finalize × experts` split and sglang's `dispatcher × runner` split, so adding EP is a matter of adding a new `MoeBackend` impl, not redesigning.

### 7.1 The two flavors of EP in vllm/sglang

There are two EP regimes in production:

| Regime | sglang dispatcher | vllm prepare/finalize | Layout produced | Pairs with |
|---|---|---|---|---|
| **High-throughput EP** (DeepEP-HT, Mori, NVLink-2-sided) | `DeepEPNormalDispatchOutput` etc. | `deepep_ht.py`, `mori_prepare_finalize.py`, `flashinfer_nvlink_two_sided.py` | Block-tiled / "standard" ragged: `[total_local_tokens_after_a2a, H]` + `expert_offsets` (or sorted_token_ids/expert_ids style) | Triton `fused_moe`, DeepGEMM contiguous |
| **Low-latency EP** (DeepEP-LL, FlashInfer NVLink-1-sided) | `DeepEPLLDispatchOutput` | `deepep_ll.py`, `flashinfer_nvlink_one_sided.py` | Masked / batched: `[E_local, M_max, H]` + `masked_m: [E_local]` | FlashInfer cutedsl, DeepGEMM masked |

So our block-tiled choice (Q6) is **the right pair for EP-HT, the wrong pair for EP-LL**:
- For **EP-HT**: the all-to-all delivers ragged tokens that look like a *bigger* batch on each rank. Our `Dispatch.moe_align_block_size` step still applies — it just operates on the rank-local concatenated tokens. `Experts` and `Combine` work unchanged. So **adding EP-HT is "wrap `Dispatch` with an A2A and `Combine` with the reverse A2A"**, with no metadata-shape change.
- For **EP-LL**: the all-to-all already delivers `[E_local, M_max, H] + masked_m`. To use that, we'd need a *second* `MoeBackend` impl (e.g. `flashinfer_cutedsl_moe.py`, F1) and a different `tok_meta` flavor. Our `Dispatch` and `Combine` would short-circuit on the EP-LL path because the kernel folds permute/unpermute into the A2A.

### 7.2 What EP means for our three modules (block-tiled / EP-HT path)

In an EP-HT-enabled `Dispatch`:
1. Local routing as before (gate + topk).
2. **All-to-all dispatch** (e.g. DeepEP-HT): every rank sends its (token, k) pairs to the rank that owns the chosen expert. Output is ragged-concatenated tokens on each rank.
3. Rank-local `moe_align_block_size` over the *received* tokens — same code as today, different `T`.

In EP-HT-enabled `Experts`:
- Identical to single-rank — same `run_experts(...)`, but `E` is `num_local_experts` (e.g. 16 of 128 for `tp_size = 8` w/ EP).

In EP-HT-enabled `Combine`:
1. K-axis reduction as today, **then** reverse all-to-all to ship per-token outputs back to the original ranks. (Or the comm op fuses both — depends on transport.)

So the engine-level shape is the **same three boxes** with `Dispatch` and `Combine` wrapping their work in collectives. In our codebase that would mean adding `artifacts/moe_backend/deepep_ht.py` and selecting it by config flag — **the `FusedMoE` parent and the model definitions stay untouched**.

### 7.3 Are dispatch/combine already involved in EP code today?

Yes, in both engines:

- **sglang**: `BaseDispatcher` is the integration point. `StandardDispatcher` is the local pass-through, `FlashinferDispatcher` is the NVLink A2A, `DeepEPLLDispatcher` is the LL A2A. The runner only sees `dispatch_output: DispatchOutput` and dispatches by `format`. There is no "extra" EP machinery on top of `FusedMoE.forward` — the dispatcher *is* the EP machinery.
- **vllm**: same idea, different naming. `prepare_finalize/no_dp_ep.py` is local; `prepare_finalize/deepep_ht.py` and `prepare_finalize/deepep_ll.py` are EP variants; all three implement `FusedMoEPrepareAndFinalize` so `FusedMoEKernel.apply` doesn't change.

For us, this means the **single-rank `Dispatch` and `Combine` we are about to write are the rank-local degenerate case of the EP-HT versions**, and a *separate* `MoeBackend(impl="flashinfer_cutedsl")` covers the EP-LL pairing on Hopper/Blackwell. Both are F-items in §13, neither in MVP scope.

---

## 8. CUDA graph

Three observations make MoE almost free under CUDA graph for the block-tiled path:

1. **Workspace tensors are persistent.** `intermediate_cache{1,2,3}`, `sorted_token_ids_buf`, `expert_ids_buf`, `num_tokens_post_padded`, `cumsum_buf` are all allocated once in `MoeBackend.__post_init__` with capacities sized to `T_cap = max_num_batched_tokens`, and registered as `StateCell`s on every `Dispatch` / `Experts` / `Combine` instance. Nothing is reallocated.
2. **Triton kernel launch grid is data-dependent on `num_tokens_post_padded`.** This is a scalar tensor value, not a Python int — Triton reads it from device memory at launch. So the kernel grid is *not* statically known to the host but **is** statically the same memory location every call, which makes it CUDA-graph compatible (the captured launch reads the value at replay).
3. **`prepare_metadata_for_moe` runs outside the captured region.** It only zeros `num_tokens_post_padded`. The graph captures `Dispatch.gate`, `topk_softmax`, `moe_align_block_size`, the two GEMM launches, `silu_and_mul`, and `moe_sum`. All of those use device-side state that varies across decode steps without changing tensor shapes.

If we later need different `T_cap` per captured BS (memory-saving), we mirror `decode_cuda_graph_metadata[bs]` with a `moe_buffers[bs]` dict and toggle `Context.moe_capture_buf_id`. Not in MVP.

---

## 9. Buffer ownership (Q4 in detail)

The orchestrator pattern from `nanovllm_base`:

```48:62:/home/yyx/personal/Artifact-Infer/workshop/nanovllm_base/services/model_runner/model_runner.py
        attention = orch.add(FlashinferAttention(config))
        ...
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                 orch.register(attention, "attn", module)
```

becomes (added):

```python
moe_backend = orch.add(MoeBackend(config, impl=config.moe_impl))   # "torch" | "triton"
...
for module in self.model.modules():
    # Attention registration — unchanged
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        orch.register(attention, "attn", module)
    # MoE registration — each of the three submodules receives the buffers it needs
    if isinstance(module, Dispatch):
        orch.register(moe_backend, "sorted_token_ids_buf",   module)
        orch.register(moe_backend, "expert_ids_buf",         module)
        orch.register(moe_backend, "num_tokens_post_padded", module)
        orch.register(moe_backend, "cumsum_buf",             module)
    if isinstance(module, Experts):
        orch.register(moe_backend, "intermediate_cache1",    module)
        orch.register(moe_backend, "intermediate_cache2",    module)
        orch.register(moe_backend, "intermediate_cache3",    module)
        orch.register(moe_backend, "run_experts",            module)
    if isinstance(module, Combine):
        orch.register(moe_backend, "intermediate_cache3",    module)
orch.register(moe_backend, "prepare_metadata_for_moe", self)
orch.finalize()
```

What this gives us:
- **Dispatch** has `self.sorted_token_ids_buf`, `self.expert_ids_buf`, `self.num_tokens_post_padded`, `self.cumsum_buf` writeable through the `StateCell` mechanism. (It does not need `intermediate_cache*` — those are kernel workspaces.)
- **Experts** has all three intermediate caches plus the `run_experts` `MethodCell` (which the orchestrator already wraps as a `MethodProxy` — same pattern as `attn`).
- **Combine** has `self.intermediate_cache3` to read from.
- **MoeBackend** is the single source of truth for the buffers; if we later resize them or swap to per-BS sets, only `MoeBackend.__post_init__` changes.

Compared to "buffers attached to a `moe_backend` reference held by `FusedMoE`", this is a few extra registration calls but gives us:
- one orchestrator-visible owner,
- automatic propagation when we go multi-process via `DistOrchestrator`,
- a uniform pattern that matches how attention already works.

This is the nano-vllm-friendly idiom and is what makes Q4 worthwhile.

---

## 10. `model_runner.py` diff (preview)

Only the additions; the existing attention path is untouched.

```python
# imports
from workshop.nanovllm_moe.artifacts.moe_backend import MoeBackend
from workshop.nanovllm_moe.artifacts.modeling.layers.moe import Dispatch, Experts, Combine
from workshop.nanovllm_moe.artifacts.modeling.models.qwen3_moe import Qwen3MoeForCausalLM

# inside ModelRunner.__init__, alongside the existing orchestrator block
moe_backend = orch.add(MoeBackend(config, impl=config.moe_impl))   # "torch" | "triton"
self.model = orch.add(Qwen3MoeForCausalLM(config.hf_config))      # was Qwen3ForCausalLM

# MoE registration loop (alongside the attention loop)
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        orch.register(attention, "attn", module)
    if isinstance(module, Dispatch):
        orch.register(moe_backend, "expert_in_buf",  module)
        orch.register(moe_backend, "masked_m_buf",   module)
    if isinstance(module, Experts):
        orch.register(moe_backend, "expert_in_buf",  module)
        orch.register(moe_backend, "expert_out_buf", module)
        orch.register(moe_backend, "masked_m_buf",   module)
        orch.register(moe_backend, "run_experts",    module)
    if isinstance(module, Combine):
        orch.register(moe_backend, "expert_out_buf", module)

orch.register(moe_backend, "prepare_metadata_for_moe", self)
orch.finalize()
```

```python
# inside prepare_prefill / prepare_decode, after set_context(...)
self.prepare_metadata_for_moe(input_ids.size(0))
```

(Plus setting `Context.num_tokens` inside `set_context` — one extra arg.)

---

## 11. What is explicitly out of scope

- `FusedMoEMethodBase` / `quant_method` abstraction.
- `MoeRunnerBackend` enum, `FusedOpPool` registry, `MoeA2ABackend`.
- Any FlashInfer MoE kernel (impossible on sm89; entered as F1/F4 once on supported hardware).
- Any EP transport (DeepEP-HT, DeepEP-LL, FlashInfer NVLink, Mori, Nixl). EP enters as a *future* `MoeBackend` subclass, not as a refactor.
- Quantization (FP4, FP8, AWQ, GPTQ, Marlin).
- Triton kernel autotuning (we ship one fixed config; autotuning is F5).
- Grouped top-k, biased top-k, EPLB, simulator router. Plain `topk + softmax (+ optional norm)`.
- Shared-experts overlap, two-stream shared experts, fused router GEMM.
- ROCm / CPU / XPU paths, LoRA on MoE, telemetry capturer.
- Chunking the T dimension. One workspace covers `max_num_batched_tokens × top_k`.

---

## 12. Implementation plan (post sign-off)

1. **Skeleton**: copy `nanovllm_base` → `nanovllm_moe`. Confirm baseline `qwen3.py` still runs end-to-end against `Qwen3-4B`.
2. **Context + loader + config**: add the four `Context` fields, the `moe_impl` config flag, and the stacked-expert loader branch. Smoke-test by loading `Qwen3-30B-A3B` weights into dummy stacked tensors and checking checksums.
3. **`MoeBackend` (torch impl)**: implement `MoeBackend` with `impl="torch"` — buffer allocation, `prepare_metadata_for_moe`, `run_experts` calling pure-torch sort + `bmm` + `silu_and_mul` + `bmm` + `* topk_weights`. Unit test with random topk_ids on a 2-expert toy.
4. **`Dispatch`** with `sgl_kernel.topk_softmax` + `sgl_kernel.moe_align_block_size`. Unit-test that the index tensors match a pure-torch reference.
5. **`Experts` / `Combine` / `FusedMoE`**: wire the three submodules. Unit test against the HF `Qwen3MoeSparseMoeBlock` (single layer, single batch).
6. **`Qwen3MoeForCausalLM`**: write the model file (clone `qwen3.py`, swap MLP for `FusedMoE`). Run end-to-end on a single prompt; compare logits against HF.
7. **`model_runner` integration**: register buffers + `run_experts` + `prepare_metadata_for_moe`. Run end-to-end through `LLM.generate`.
8. **Triton kernel** (`triton_fused_moe.py`): vendor a bf16-only `invoke_fused_moe_kernel` (≈ 250 LOC) plus its `@triton.jit` body, adapted from `sglang/.../fused_moe_triton_kernels.py` with FP4/FP8/AWQ/GPTQ branches stripped. Wire into `MoeBackend` under `impl="triton"`.
9. **Parity test** between `impl="torch"` and `impl="triton"` on identical inputs. Then end-to-end on Qwen3-30B-A3B with `impl="triton"`.
10. **CUDA graph**: enable `enforce_eager=False`; verify decode-mode capture/replay matches eager output.

Every step is independent of the next at the orchestration level — only the *body* of one Module / one method changes between steps.

---

## 13. Open follow-ups (not blockers)

- **F1.** When on Hopper/Blackwell **with FP4 weights**, add `MoeBackend(impl="cutedsl")` using `flashinfer.cute_dsl.blockscaled_gemm.grouped_gemm_nt_masked`. `Dispatch` and `Combine` will need a *masked-m* sibling path because the cutedsl contract is `[E_local, M_max, H] + masked_m`, not block-tiled. Wire selection by hardware probe.
- **F2.** When on Hopper/Blackwell **with bf16 weights**, add `MoeBackend(impl="flashinfer_cutlass")` using `flashinfer.fused_moe.cutlass_fused_moe` (sm90+) — monolithic, so `Dispatch` reduces to gate+topk and `Combine` becomes a no-op. Sm100+: also `MoeBackend(impl="trtllm_bf16")` which is even more monolithic (kernel does TopK).
- **F3.** Add EP-HT (DeepEP-HT or FlashInfer NVLink two-sided) as `MoeBackend(impl="deepep_ht")`. Wraps the existing block-tiled `Dispatch.moe_align_block_size` with an A2A; wraps the existing `Combine.moe_sum` with the reverse A2A. No metadata-shape change.
- **F4.** Add EP-LL (DeepEP-LL or FlashInfer NVLink one-sided) — paired with the cutedsl backend (F1) on the masked-m contract.
- **F5.** Triton kernel autotuning: port `try_get_optimal_moe_config` and the `configs/` JSON tables from sglang/vllm so the BLOCK_SIZE_M/N/K and num_warps adapt to the (T, E, K, N, H) shape.
- **F6.** `routed_experts_capturer`-style telemetry can be added by inserting a hook inside `Dispatch.forward` after `topk_ids` is computed; it does not need a separate Module.

---

## Appendix A — Glossary of shapes

| Symbol | Meaning | Qwen3-30B-A3B value |
|---|---|---|
| `T` | tokens in the batch (varies) | up to `max_num_batched_tokens` |
| `T_cap` | static workspace capacity along T | `max_num_batched_tokens` |
| `H` | hidden size | 2048 |
| `N` | MoE intermediate size (per expert) | 768 |
| `E` | total experts | 128 |
| `K` | top-k (experts per token) | 8 |
| `BLOCK_M` | Triton tile size along the (T*K)-axis | 64 (fixed in MVP) |
| `pad_max` | max padding from `moe_align_block_size` | `E × (BLOCK_M − 1)` |
| `num_tokens_post_padded[0]` | per-batch padded length, ≤ `T*K + pad_max` | runtime |

Workspace sizes:
- `intermediate_cache1`: `[T_cap, K, 2N]`
- `intermediate_cache2`: `[T_cap, K, N]`
- `intermediate_cache3`: `[T_cap, K, H]`
- `sorted_token_ids_buf`: `[T_cap × K + pad_max]`
- `expert_ids_buf`: `[(T_cap × K + pad_max) // BLOCK_M]`
- `num_tokens_post_padded`: `[1]` int32
- `cumsum_buf`: `[E + 1]` int32

---

## Appendix B — Expected files at end of MVP

```
workshop/nanovllm_moe/artifacts/moe_backend/__init__.py           ~ 20 LOC (factory selecting impl)
workshop/nanovllm_moe/artifacts/moe_backend/moe_backend.py        ~ 120 LOC (Artifact: workspaces + prep + run_experts dispatch)
workshop/nanovllm_moe/artifacts/moe_backend/torch_fused_moe.py    ~ 80 LOC (reference run_experts)
workshop/nanovllm_moe/artifacts/moe_backend/triton_fused_moe.py   ~ 250 LOC (vendored bf16 Triton kernel + launcher)
workshop/nanovllm_moe/artifacts/modeling/layers/moe/dispatch.py   ~ 50 LOC
workshop/nanovllm_moe/artifacts/modeling/layers/moe/experts.py    ~ 25 LOC
workshop/nanovllm_moe/artifacts/modeling/layers/moe/combine.py    ~ 20 LOC
workshop/nanovllm_moe/artifacts/modeling/layers/moe/fused_moe.py  ~ 35 LOC
workshop/nanovllm_moe/artifacts/modeling/models/qwen3_moe.py      ~ 80 LOC (clone of qwen3.py)
workshop/nanovllm_moe/services/model_runner/model_runner.py       ~ +20 LOC delta vs base
workshop/nanovllm_moe/services/config.py                          ~ +1 LOC delta vs base
workshop/nanovllm_moe/services/utils/context.py                   ~ +6 LOC delta vs base
workshop/nanovllm_moe/services/utils/loader.py                    ~ +15 LOC delta vs base
```

≈ 470 LOC of net-new code for the full MVP (≈ 220 if you ship `impl="torch"` only and defer `triton_fused_moe.py`), with **zero deletions** and no edits to existing files outside `services/config.py`, `services/utils/{context,loader}.py`, and `services/model_runner/model_runner.py`.
