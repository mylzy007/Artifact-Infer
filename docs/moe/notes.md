# MoE on `nanovllm_moe` — Implementation Notes

> Companion to `docs/moe/design.md`. The design doc is the **plan**; this is the **journal** — what surprised us, what we changed, and the small landmines we stepped on.

Audience: someone re-reading or extending this code six months from now. Skim the **TL;DR** in each section if you only have a minute.

---

## 1. Architecture refinements made during implementation

### 1.1 Gate hoisted from `Dispatch` → `FusedMoE` ❗ (deviation from design.md §4)

**TL;DR:** the routing gate `nn.Linear` lives on `FusedMoE`, not on `Dispatch`. `Dispatch.forward` takes `router_logits` (already projected) instead of raw `hidden_states`.

**Why we changed it.** HF stores the gate at `model.layers.{l}.mlp.gate.weight`. The original design put `gate` inside `Dispatch`, which would have made our parameter path `model.layers.{l}.mlp.dispatch.gate.weight`. Two ways out:

1. Add a **rename rule** to the loader (`mlp.gate` → `mlp.dispatch.gate`).
2. Move the gate **up one level** to `FusedMoE`.

We picked (2) — it adds zero loader complexity and makes `Dispatch` purely stateless logic (topk + block-align). It also matches sglang's mental model where `gate` is treated as part of the parent layer and the dispatcher only handles routing decisions on already-computed logits.

**Catch.** The first end-to-end test on the real Qwen3-30B-A3B checkpoint produced **all-NaN MoE outputs**. Cause: the gate weight path didn't match — the loader fell through to `model.get_parameter("model.layers.0.mlp.gate.weight")` which raised → was caught silently → gate stayed at `torch.empty()` (uninitialized memory, which on this run happened to be NaN). Fix took 3 minutes once spotted; spotting took 30 minutes because the symptom (all NaN at the *end* of the MoE) suggested a numerical issue, not a loader miss.

**Lesson.** When integrating new model parameter paths, the **loaded-tensor count** is a much better health signal than examining outputs. After the fix, the count went 787 → 789 (+2 = the 2 missing gate weights for the 2-layer test slice).

### 1.2 `set_moe_capacity` and `Context` field policy

**TL;DR:** `Context` has two classes of fields — *per-batch* (cleared by `reset_context()`) and *static* (set once by `MoeBackend.__post_init__` and survive `reset_context()`).

**Why.** The 4 new MoE fields split cleanly:

- `num_tokens` — per-batch (set by `set_context(..., num_tokens=...)` in `prepare_prefill/decode`).
- `moe_t_cap`, `moe_block_size_m`, `moe_capture_buf_id` — static (set once at init via `set_moe_capacity`).

The original `reset_context()` in `nanovllm_base` blindly resets the whole dataclass, which would wipe the static fields between batches. We added a static-aware `reset_context()` that preserves them.

**Catch.** This is easy to miss — `reset_context()` is called every step. If you add another static field later, remember to thread it through `reset_context()` too.

### 1.3 `Combine` reads `T` from the input tensor, not from `Context.num_tokens`

**TL;DR:** the design doc said `Combine` reads `T` from `Context`. We instead read it from `expert_out_TKH.size(0)`.

**Why.** The input is already shape `[T, K, H]` so the size is right there. Reading from `Context` adds a (tiny) coupling that buys nothing here. We kept `Context.num_tokens` because `Dispatch` will need it eventually for capture-mode buffer slicing, but `Combine` doesn't.

**No catch — just smaller delta vs the design.**

---

## 2. Orchestrator pits

### 2.1 Latent bug: `_propagate` debug print assumed `cell.origin` ⚠️

**TL;DR:** `src/core/orchestrator.py:_propagate` printed `cell.origin.name`, which only exists on `MethodCell`s (not `StateCell`s). Triggered the moment we propagated our first state cell.

**Why it never fired before.** `nanovllm_base` only calls `orch.register(attention, "attn", module)` — and `attn` is a method, so it always becomes a `MethodCell`. No state cells ever flowed through `_propagate` until MoE arrived (we propagate ~10 workspace tensors per layer).

**Fix:** one-line `hasattr(cell, "origin")` guard. Confined to a debug log, no semantic change.

### 2.2 Don't `orch.add(...)` non-`Artifact` modules ⚠️

**TL;DR:** `orch.add(fused)` on a plain `nn.Module` (which `FusedMoE` is) breaks `_check_cycles` because it tries to read `fused.name` and `fused.parents`.

**Pattern.** The orchestrator's registry is **only** for `Artifact`s. To wire MoE buffers onto plain `nn.Module`s (like `Dispatch`, `Experts`, `Combine`), make those modules **inherit from `Artifact, nn.Module`** (so they have `_cells`, `parents`, `name`), then walk `model.modules()` and `orch.register(moe_backend, "<attr>", module)`. Don't `orch.add()` anything that doesn't subclass `Artifact`.

`FusedMoE` itself is intentionally a plain `nn.Module` — it owns no registered state, only submodules.

### 2.3 `Dispatch`/`Experts`/`Combine` MUST inherit from `Artifact`

**TL;DR:** if a module is a *consumer* of a registered cell, it must subclass `Artifact` (so it has `_cells` to receive the propagation, and `__getattr__` to look up cells when accessed).

This is the same pattern as `Attention(Artifact, nn.Module)` in the existing `flashinfer_attention.py`.

**Pit if you forget:** the registration silently writes the cell into a `_cells` dict that never gets read (because the module's `__getattr__` is just `nn.Module`'s, which doesn't know about cells). You'd see `AttributeError: <Module> has no attribute 'sorted_token_ids_buf'` on the first access.

### 2.4 MethodProxy attribute resolution rule

`MethodProxy.__getattr__(name)` checks **origin first, host second**. So inside `MoeBackend.run_experts(self, ...)`, `self.intermediate_cache1` resolves to the cell on `MoeBackend` (the origin) — even though we also propagated that cell onto `Experts` (the host).

This is what we want: the inner kernel function doesn't care which physical module owns the buffer, but it always gets the same `MoeBackend`-owned tensor. The Experts-side cell registration is what makes `Experts.intermediate_cache3` available **outside** the method body (e.g., for `Combine` to read by following the propagation chain).

---

## 3. `sgl_kernel.moe_align_block_size` is full of footguns 🚧

This kernel is the most error-prone surface we touched. Three independent gotchas, all detected via the dispatch unit test (max-detail dispatch test caught all three).

### 3.1 Pass `num_experts + 1`, not `num_experts` ⚠️

**The bug.** Internally the kernel uses a 1-indexed expert space where slot 0 is reserved as a "filtered/padding" sentinel. Passing `num_experts = E` means slots `1..E` are available for real experts → only `E` real experts fit, but Qwen3 has 128 experts numbered `0..127`, and the kernel maps real expert `i` to internal slot `i + 1`. Real expert 127 needs internal slot 128, which doesn't exist if you only allocate space for `num_experts = 128` slots. Result: experts at the high end silently drop their tokens.

**Fix:** pass `num_experts + 1` to the kernel. Allocate `cumsum_buffer` of size `(E + 1) + 1 = E + 2`. Padding budget grows from `E * (BLOCK_M - 1)` to `(E + 1) * (BLOCK_M - 1)`.

This is what sgl's own wrapper (`sglang/.../moe_align_block_size.py`) does:

```python
sgl_moe_align_block_size(topk_ids, num_experts + 1, block_size, ...)
```

We copied that pattern verbatim.

### 3.2 Pass `pad_sorted_token_ids=True` ⚠️

**The bug.** The kernel **does not** initialize the padding entries of `sorted_token_ids` unless `pad_sorted_token_ids=True`. Without that flag, padded slots contain whatever uninitialized memory was in the buffer.

The Triton consumer kernel does `token_mask = offs_token < num_valid_tokens` to mask padding. For that to work, the padding entries must hold a value `≥ num_valid_tokens` (the sentinel is `topk_ids.numel()`). If they hold uninitialized memory, the mask will silently mis-classify some padding slots as valid → the consumer reads `hidden_states` at a random index.

**Fix:** always pass `pad_sorted_token_ids=True`.

### 3.3 Padding-only blocks have `expert_id = -1` ✅ (handled)

This one is documented behavior, not a pit, but worth noting. After moe_align, some blocks may have `expert_ids[b] = -1` — they consist entirely of padding. The Triton fused_moe kernel skips them with an early `if off_experts == -1: write_zeros; return`.

### 3.4 What we test to catch all three at once

`_test_dispatch.py` validates: (a) every block's `expert_ids[b]` matches the actual expert of the valid token-ids in that block; (b) blocks with `-1` expert have zero valid tokens; (c) total valid count equals `T*K`. This caught all three pits during bring-up and is the regression test for any future refactor of the dispatch path.

---

## 4. Triton kernel pit: `c_sorted` semantics 🔥

**TL;DR:** for both GEMMs in `triton_fused_moe`, set `c_sorted=False`. `True` was the original choice — it's wrong here, and the failure mode is subtle (~100× larger error vs HF, but still passing a loose `< 2e-2` tolerance).

### What `c_sorted` actually controls

The kernel writes its output via either:

- `c_sorted=True`:  `c_ptrs = c_ptr + offs_token_id[:, None] * stride_cm + ...`  (output indexed by **sorted position**)
- `c_sorted=False`: `c_ptrs = c_ptr + offs_token[:, None] * stride_cm + ...`  (output indexed by **flat (token, k) id**)

These two are different coordinate systems linked by `offs_token = sorted_token_ids[offs_token_id]`.

### The bug pattern we hit

GEMM1 with `c_sorted=True` writes `cache1` indexed by sorted position. Then `silu_and_mul` does element-wise reads/writes in the same coordinate system → `cache2` is also indexed by sorted position. Then GEMM2's `a_ptrs` formula is `a_ptr + offs_token // top_k * stride_am`. With `top_k=1`, that becomes `a_ptr + offs_token * stride_am` — but `offs_token` is in the **flat (token, k)** coordinate system, and `cache2` is in the **sorted-position** system. So GEMM2 reads `cache2` at the wrong rows.

The result still kind of works because the indices are at least within bounds and the values are bf16-y, but the per-element accuracy goes from `~1e-04` (correct) to `~1e-02` (wrong). This **passed** our initial tolerance check; we caught it because triton vs torch divergence was ~1.7e-2 while torch vs HF was 1.2e-4 — a 100× gap that shouldn't exist between two implementations of the same math.

### The fix

Set `c_sorted=False` for **both** GEMMs. Now `cache1`, `cache2`, `cache3` all live in the flat-(token,k) coordinate system. The downstream `Combine` can `cache3.view(T, K, H)` because the layout matches the natural row-major reshape.

After the fix:

- triton vs torch: max abs `9.5e-07` (down from 1.7e-02)
- triton vs HF: max abs `1.2e-04` (matches torch, both at bf16 noise floor)

### Lesson

When two backends implement the same math and disagree by more than ~3× the noise floor, **one of them is wrong**. Don't widen the tolerance.

---

## 5. Triton kernel: `top_k=1` trick for GEMM2

GEMM1's `a_ptrs` formula reads `hidden_states` by **token-id** (one entry per real token, regardless of how many experts route to it):

```python
a_ptrs = a_ptr + offs_token // top_k * stride_am + ...
```

Here `offs_token` is the flat (token, k) id. Dividing by `top_k` gives the original token id. Correct for GEMM1.

GEMM2's `a` is `cache2` which has one row per (token, k) pair (size `[T*K, N]`), not per-token. So we want `a_ptrs = a_ptr + offs_token * stride_am` (no divide). Achieved by passing `top_k=1` to the same kernel — `offs_token // 1 = offs_token`. The kernel doesn't need to know "this is GEMM2"; the divisor parameter encodes the difference.

**Pit if you ever try to fuse GEMM1+GEMM2 into one launch:** they use the same kernel with **different `top_k`** values. Don't naively share.

---

## 6. Weight loading pits

### 6.1 HF Qwen3-MoE checkpoint key pattern

```
model.layers.{l}.mlp.gate.weight                       — routing gate
model.layers.{l}.mlp.experts.{e}.gate_proj.weight       — per-expert gate
model.layers.{l}.mlp.experts.{e}.up_proj.weight         — per-expert up
model.layers.{l}.mlp.experts.{e}.down_proj.weight       — per-expert down
```

The per-expert keys must be regex-matched (E=128 means 128 × 3 = **384 keys per layer**, all going into 2 stacked tensors). We added `_MOE_EXPERT_RE = r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$"` and a per-expert weight loader that takes `(param, loaded, expert_id, shard_id)`.

### 6.2 Why we removed `gate_proj`/`up_proj` from `packed_modules_mapping`

In `Qwen3ForCausalLM` (dense), the loader uses `packed_modules_mapping` to remap `gate_proj`→`gate_up_proj` (column 0) and `up_proj`→`gate_up_proj` (column 1). For the MoE model we drop those entries — the `_MOE_EXPERT_RE` branch matches first (it's in the `if/continue` clause before the `packed_modules_mapping` loop) and routes them per-expert, so the dense remap would never fire anyway. Removing them keeps `Qwen3MoeForCausalLM.packed_modules_mapping` minimal and prevents accidental misroutes if a future model has both dense MLPs and MoE MLPs sharing a checkpoint.

### 6.3 HF `Qwen3MoeExperts` already uses stacked layout

Newer HF (transformers ≥ recent commits) stores Qwen3MoE expert weights as **stacked** `[E, 2N, H]` (gate_up) and `[E, H, N]` (down) — exactly our layout. So the HF loader does the same de-stacking we do. This was a useful sanity check during the parity test: same layout, same answer.

---

## 7. Environment pits (pre-existing, not MoE)

### 7.1 `flash_attn` symbols missing

`workshop/nanovllm_base/artifacts/attention_backend/{flashinfer,fa}_attention.py` has dead `from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache` imports. The installed `flash_attn` package is empty (no symbols exported) → `ImportError` at module load.

The actual attention path uses `flashinfer.BatchPrefillWithPagedKVCacheWrapper` etc., not `flash_attn`. These imports are vestigial. We wrapped them in `try/except` so MoE bring-up isn't blocked. Did **not** modify `nanovllm_base` — only the `nanovllm_moe` copy.

### 7.2 `get_rope` is `@lru_cache`-d → fails on dict args

`get_rope(rope_scaling=dict)` raises `TypeError: unhashable type: 'dict'` because of `@lru_cache(1)`. Qwen3-MoE's HF config has `rope_scaling = {"rope_type": "default", "rope_theta": 1000000.0}` (the rope_theta is nested inside, not at the top level).

Fix in `qwen3_moe.py`: `_resolve_rope_theta(config)` extracts the theta from either the top-level `rope_theta` or the nested `rope_scaling['rope_theta']`, then passes `rope_scaling=None` to `get_rope`.

The dense `qwen3.py` doesn't hit this because Qwen3 (non-MoE) checkpoints don't use the nested form.

---

## 8. CUDA-graph compatibility checklist

For future reference — the design relies on these all being true. They were validated by `_test_cudagraph.py`:

- ✅ `intermediate_cache{1,2,3}`, `sorted_token_ids_buf`, `expert_ids_buf`, `num_tokens_post_padded`, `cumsum_buffer`, `topk_weights_buf`, `topk_ids_buf` are all allocated **once** in `MoeBackend.__post_init_`_ with `T_cap`-sized capacities. No per-call allocations.
- ✅ `prepare_metadata_for_moe` runs **outside** the captured region (it just zeros `num_tokens_post_padded`).
- ✅ Triton kernel reads `num_tokens_post_padded[0]` from device memory inside its grid → captured launch points at the same address every replay → reads the fresh value.
- ✅ `sgl_kernel.topk_softmax`, `sgl_kernel.moe_align_block_size`, `sgl_kernel.moe_sum` are all CUDA-graph compatible (in-place writes to fixed buffers, no host syncs).

If you ever add a per-batch host→device copy inside `MoeBackend` or `Dispatch`/`Experts`/`Combine`, you've broken capture. Use a persistent buffer + an in-place update kernel instead.

---

## 9. Things explicitly **not** done that the design called out as "out of scope"

Re-listed for searchability — design doc §11 is authoritative:

- No EP transport (DeepEP-HT/LL, FlashInfer NVLink, Mori, Nixl)
- No quantization (FP4/FP8/INT8/AWQ/GPTQ/Marlin)
- No FlashInfer MoE kernel (sm90+ only; F1/F2/F4 in design §13 once on supported hardware)
- No Triton kernel autotuning (one fixed config: `BLOCK_SIZE_M=64, N=64, K=32, GROUP_SIZE_M=8, num_warps=4, num_stages=3`)
- No grouped/biased top-k, no EPLB
- No shared experts (Qwen3-30B-A3B has none)
- No T-axis chunking (one workspace covers `max_num_batched_tokens × top_k`)
- No telemetry / routed-experts capturer

---

## 10. Validation hierarchy used during bring-up

For future MoE work — this is the order that gave us fastest debug signal:

1. **Index correctness** (`_test_dispatch.py`): does `moe_align_block_size` produce `sorted_token_ids` and `expert_ids` consistent with `topk_ids`? Catches all three sgl_kernel pits in §3.
2. **Backend self-parity** (`_test_torch_fused_moe.py`): does `torch_fused_moe` match a hand-rolled per-token loop on random small inputs? Catches stacked-weight layout bugs.
3. **Orchestrator end-to-end** (`_test_fused_moe.py`): does the registered-cell propagation actually route buffers into `Dispatch/Experts/Combine`? Catches `Artifact` inheritance and `_cells` lookup bugs.
4. **Backend cross-parity** (`_test_triton_parity.py`): do `triton_fused_moe` and `torch_fused_moe` agree to `~1e-6`? Catches `c_sorted` and `top_k` indexing bugs in the Triton kernel.
5. **HuggingFace parity on real weights** (`_test_parity_hf.py`, `_test_parity_hf_triton.py`): do we match HF's `Qwen3MoeSparseMoeBlock` to bf16 noise floor (`1.2e-04`) on real Qwen3-30B-A3B layer 0 weights? Catches weight-loader name-mismatch bugs and any subtle math divergence.
6. **CUDA graph** (`_test_cudagraph.py`): does capture/replay match eager bit-exactly across multiple input batches? Catches non-persistent buffer or host-side state leaks.

Going in this order, each new test's failure mode points at a narrower set of suspects than the last.

---

## 11. End-to-end engine integration (post-design.md §12 step 10)

After all 10 unit-level steps in the design plan were green, the next jump was wiring the MoE pipeline through the actual `LLMEngine` (scheduler + block manager + sampler + cuda graph). The notes below cover what was needed and what tripped us up.

### 11.1 LLMEngine was an empty scaffold — orchestrator wiring required

The `nanovllm_base.LLMEngine.step()` method calls `self.schedule()`, `self.run()`, `self.postprocess()`, `self.is_finished()`, but the class only defines `__post_init_`_, `add_request`, `step`, `generate`, and `reset` — none of the called methods are defined locally. They have to come in via the orchestrator, propagated from `Scheduler` (add/schedule/postprocess/is_finished) and `ModelRunner` (run). And `Scheduler.schedule` itself calls `self.can_allocate/allocate/can_append/may_append/deallocate` which have to come from `BlockManager`.

Wiring sequence (in `LLMEngine.__post_init_`_):

1. Build `ModelRunner` first (it allocates KV cache, which sets `config.num_kvcache_blocks`).
2. Build `BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)`.
3. Build `Scheduler(config)`.
4. Register `BlockManager.{can_allocate, allocate, can_append, may_append, deallocate}` onto `Scheduler`.
5. Register `Scheduler.{add, schedule, postprocess, is_finished}` onto `LLMEngine`.
6. Register `ModelRunner.run` onto `LLMEngine`.
7. `orch.finalize()` — propagates BlockManager methods all the way up to `LLMEngine` via the chained `parents`.

After this, `engine.step()` works because `__getattr__` resolves each name through `_cells` and the `MethodProxy` translates `self` correctly inside each method body.

**Pit:** the method-proxy resolves `self.X` first against the **origin** then against the **host**. For `Scheduler.schedule(proxy, ...)` calling `self.allocate(seq)`, the proxy first checks scheduler. Scheduler doesn't have `allocate` as a regular attribute, but it does have it in `_cells` (after step 4 above), and `Scheduler.__getattr__` (inherited from `Artifact`) finds it there. So the lookup chain is:

```
self.allocate                                     (inside Scheduler.schedule)
  → MethodProxy.__getattr__("allocate")           (since `self` is a proxy)
  → hasattr(scheduler, "allocate")                (origin first)
  → scheduler.__getattr__("allocate")             (Artifact.__getattr__)
  → scheduler._cells["allocate"]                  (set by step 4)
  → MethodCell wrapper
  → BlockManager.allocate(MethodProxy(block_mngr, scheduler), seq)
```

This works as long as every level in the chain inherits `Artifact` (so it has `_cells` and `__getattr__`). `LLMEngine` and `Scheduler` and `BlockManager` all inherit `BaseService` (which inherits `Artifact`).

### 11.2 `_ensure_distributed()` must also `set_default_device("cuda")`

`ModelRunner.__init__` calls `dist.get_rank()` (needs init'd process group) and `torch.set_default_dtype(...)` but never sets the default **device**. With default device = cpu, `nn.Parameter(torch.empty(...))` lands on cpu, and the first `embed_tokens(input_ids)` errors with "tensors on different devices" because input_ids comes from `prepare_prefill` (which explicitly does `.cuda(non_blocking=True)`).

The unit tests all set both `torch.cuda.set_device(0)` and `torch.set_default_device("cuda")` at module top. The `LLMEngine` path needed the same — so `_ensure_distributed()` does both.

**Pit if you don't:** the model weights silently land on CPU, the load is slow but works, and the first forward pass errors deep inside `F.embedding`.

### 11.3 FlashInfer JIT needs CUDA 12.x nvcc — `/usr/bin/nvcc` is too old

System nvcc is CUDA 11.5 (no `compute_89`). FlashInfer JIT-compiles its prefill kernel on first use → fails with `nvcc fatal: Unsupported gpu architecture 'compute_89'`.

Fix: prepend the CUDA 12.8 toolkit dir to `PATH` before running anything that touches FlashInfer:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
```

This is an environment problem, not a code problem. Documented here so the next person hits it immediately rather than 30 minutes in.

### 11.4 Import cycle from `services/__init__.py`

The `services/__init__.py` originally did `from .llm import LLM` (and `from .sampling_params import SamplingParams`). `LLM` extends `LLMEngine`, which now imports `ModelRunner`, which imports `MoeBackend`. `MoeBackend` imports `from workshop.nanovllm_moe.services.utils.context import set_moe_capacity` — and *importing anything under `services`* triggers `services/__init__.py`, which is **still in the middle of importing `LLM`**. The result:

```
ImportError: cannot import name 'MoeBackend' from partially initialized module
'workshop.nanovllm_moe.artifacts.moe_backend' (most likely due to a circular import)
```

The cycle is real but only fires for tests that import from `artifacts.*` first (since they don't go through the engine path). Any test that imports `LLMEngine` first works fine, because by then `services/__init__.py` has fully resolved.

**Fix:** removed the eager imports from `services/__init__.py`. Users now import explicitly:

```python
from workshop.nanovllm_moe.services.llm import LLM
from workshop.nanovllm_moe.services.sampling_params import SamplingParams
```

Less convenient but no cycle.

### 11.5 `Config.num_hidden_layers_override` for memory-bound testing

Qwen3-30B-A3B is ~60 GB bf16; one RTX 4090 has 24 GB. Full-model e2e on a single GPU is impossible without TP. We added a `num_hidden_layers_override` field to `Config`: post-init trims `hf_config.num_hidden_layers` to the override. Combined with a loader that **gracefully skips** weights for layers ≥ N (rather than erroring), this gives us a "shape-compatible but text-incoherent" model that exercises every code path including dispatch + experts + combine + cuda graph + scheduler + block manager.

Loader bookkeeping: now prints `[loader] loaded N tensors, skipped M (model has fewer layers than ckpt)`. For NUM_LAYERS=8 we see ~3.1k loaded vs ~15.7k skipped (since the checkpoint has 48 layers).

For real-text e2e you need either a smaller MoE checkpoint that fits on one GPU, or TP across multiple GPUs (out of MVP scope).

### 11.6 `graph_bs = list(range(1, 8))` was hardcoded > `max_bs`

`capture_cudagraph` builds graphs for batch sizes 1..7. But `max_bs = min(max_num_seqs, 512)`. If `max_num_seqs < 7`, the per-bs buffers (sized at `max_bs`) can't accommodate `bs=7` graph. Symptom:

```
RuntimeError: The expanded size of the tensor (5) must match the existing size (8)
at non-singleton dimension 0.  Target sizes: [5].  Tensor sizes: [8]
```

(triggered inside `flashinfer_attention._update_indices` when assigning `qo_indptr[:bs+1] = arange(bs+1)`.)

Fix: cap with `min(max_bs, 8)`:

```python
self.graph_bs = list(range(1, min(max_bs, 8) + 1))
```

This was a pre-existing bug in `nanovllm_base` masked by the fact that no nanovllm_base test ever ran the engine end-to-end with `max_num_seqs < 7`.

### 11.7 `prepare_metadata_for_moe` must also fire inside `capture_cudagraph`

The captured graph's first kernel will read whatever value `num_tokens_post_padded[0]` happens to hold at capture time (and it gets baked into the launch's grid-dim derivation as a device pointer). At replay time, the runtime re-derives the grid by re-reading the buffer at the same address, so the buffer's **address** matters at capture, but its **value** can change between replays.

We must call `prepare_metadata_for_moe(bs)` before both the warmup and the capture so that:

- the buffer is in a known state,
- the first dispatch.forward inside the captured region writes the correct value, and
- all subsequent kernels in the captured region (for that bs) read that value rather than uninitialized memory.

Done by inserting `if hasattr(self, "prepare_metadata_for_moe"): self.prepare_metadata_for_moe(bs)` right before the warmup forward in `capture_cudagraph`.

### 11.8 `moe_impl="torch"` is not graph-capturable

The torch reference iterates `for e in range(E):` and uses `topk_ids.nonzero()` which forces a host sync. `torch.cuda.graph` capture refuses to record that:

```
torch.AcceleratorError: CUDA error: operation failed due to a previous error during capture
```

We added an upfront assertion in `MoeBackend.__post_init__`:

```python
if self.impl == "torch" and not getattr(self.config, "enforce_eager", True):
    raise RuntimeError("moe_impl='torch' is not compatible with cuda graph capture ...")
```

Fast fail with a fix recipe is better than a generic CUDA error.

### 11.9 Engine validation matrix (what we measured)


| backend         | dense Qwen3-4B | MoE 8-layer torch | MoE 8-layer triton | MoE 8-layer triton+graph |
| --------------- | -------------- | ----------------- | ------------------ | ------------------------ |
| build time      | 4.4 s          | 11.1 s            | 7.7 s              | 22.1 s (+graph capture)  |
| 64 toks at bs=2 | n/a            | 16.1 s (4.0 t/s)  | 4.8 s (13.4 t/s)   | 1.36 s (47.2 t/s)        |


The triton + cuda graph result is **bit-exact identical** to triton + eager on the same inputs (greedy). This confirms the entire MoE pipeline (dispatch → experts → combine → sampler) is captured correctly.

The torch backend's first 3 tokens match the triton backend on the same prompts (greedy bf16 noise floor diverges for later tokens). This is the engine-level analogue of the per-block parity tests in §10.

### 11.10 What "real" MoE e2e would look like

Beyond what's tested here, you'd want:

- A model that fits on one GPU (e.g. a smaller MoE) so the generated text is meaningful.
- Or TP across 2-8 GPUs, which requires updating `_ensure_distributed` to do `world_size > 1`, deploying `ModelRunner` per rank via `DistOrchestrator.deploy_distributed_runner`, and validating that the `Experts` weight loader correctly shards `[E, 2N, H]` across TP ranks (currently it doesn't — it copies the full tensor on every rank). That's design.md §11's "EP" item: out of MVP scope.
- Streaming output (currently `generate` collects all completions before returning).
- Batched continuous decoding throughput (we tested with 2 short prompts; the scheduler's preempt path was never exercised).

