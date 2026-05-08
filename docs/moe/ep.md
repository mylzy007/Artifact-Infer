# Expert Parallelism for `nanovllm_moe`

This is the design + journal for the EP implementation that ships in `workshop/nanovllm_moe`. Three flavors are implemented:


| flavor                              | dispatch                                                          | inner kernel                                             | cuda graph        | trade-off                                               |
| ----------------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------- | ----------------- | ------------------------------------------------------- |
| **EP-LL** (`ep_ll_triton`)          | dense [N, E_local, M_max, H] a2a + atomic-bucketing Triton kernel | `triton_masked_grouped_gemm` (sets `masked_m = N*M_max`) | yes               | low latency, wastes FLOPs on zero-padded slots          |
| **EP-LL torch ref** (`ep_ll_torch`) | same layout, host-side python loops                               | torch `bmm` per expert                                   | no (host loops)   | reference-only                                          |
| **EP-HT** (`ep_ht`)                 | ragged variable-size a2a + sgl_kernel `moe_align_block_size`      | reuses single-rank `triton_fused_moe`                    | no (variable a2a) | higher throughput at large batches; classical EP layout |


Plus **TP × EP composition**: every rank belongs to a TP subgroup AND an EP subgroup, layout `rank = ep_rank * TP + tp_rank`. TP-aware layers (attention, embed, lm_head) shard within TP; MoE collectives go through EP.

It complements `docs/moe/design.md` (single-rank MoE) and `docs/moe/notes.md` (single-rank pits).

If you only read one section: **§2 (data flow), §6 (validation), §13 (Triton dispatch + cuda graph), §14 (EP-HT), §15 (TP × EP)**.

---

## 0. Decisions (recap)

User choices made before implementation:

1. **EP-LL first** (not EP-HT) — driven by the cleaner cuda-graph layout and future-proofing for FlashInfer/DeepGEMM masked kernels, not by latency on this hardware (see §7).
2. **NCCL `all_to_all_single`** as transport.
3. **Pure EP**, no TP within attention (each rank holds the full attention/embed/lm_head).
4. **8 GPUs** (RTX 4090, sm89).
5. **Both `torch` and `triton` inner kernels**.
6. **Numerical tolerance 1e-4** (bf16-noise OK).

---

## 1. Module layout

```
workshop/nanovllm_moe/
├── artifacts/modeling/layers/moe/
│   ├── fused_moe.py            # branches: single-rank vs EP-LL
│   ├── dispatch.py             # single-rank (block-tiled fused_moe)
│   ├── experts.py              # single-rank
│   ├── combine.py              # single-rank
│   ├── dispatch_ep_ll.py       # NEW — EP-LL dispatcher (torch ref)
│   ├── experts_ep_ll.py        # NEW — owns rank's E_local expert weights
│   └── combine_ep_ll.py        # NEW — reverse a2a + scatter-reduce
├── artifacts/moe_backend/
│   ├── moe_backend.py          # branches by config.moe_impl
│   ├── torch_fused_moe.py      # single-rank torch ref
│   ├── triton_fused_moe.py     # single-rank triton
│   ├── torch_masked_grouped_gemm.py    # NEW — EP-LL inner kernel torch ref
│   └── triton_masked_grouped_gemm.py   # NEW — EP-LL inner kernel triton
└── services/
    ├── utils/parallel.py       # NEW — TP subgroup helper (set_tp_group, get_tp_*)
    ├── engine/llm_engine.py    # _ensure_distributed creates per-rank size-1 TP group
    └── model_runner/model_runner.py  # wires EP-LL buffers from MoeBackend
```

`config.moe_impl` selector:


| value          | inner kernel                 | EP? | cuda graph                        |
| -------------- | ---------------------------- | --- | --------------------------------- |
| `torch`        | `torch_fused_moe`            | no  | no (host loops)                   |
| `triton`       | `triton_fused_moe`           | no  | yes                               |
| `ep_ll_torch`  | `torch_masked_grouped_gemm`  | yes | no (host dispatch)                |
| `ep_ll_triton` | `triton_masked_grouped_gemm` | yes | no (still host dispatch — see §8) |


---

## 2. Data flow per layer (EP-LL)

```
                           rank r (E_local = E_global / N experts here)
                           ────────────────────────────────────────────────
hidden[T, H]  ──── gate ──▶ router_logits[T, E_global]
                  ┌──────────── DispatchEPLL ────────────┐
                  │ topk → (topk_ids[T, K], topk_w[T, K])│
                  │ bucket: send_buf[N, E_local, M_max, H]│
                  │ also fill original_indices[N, E_local, M_max, 2] (t, k)│
                  │ NCCL all_to_all_single(send_buf -> recv_buf)│
                  │ recv_buf.permute(1,0,2,3).view(E_local, N*M_max, H)│
                  │ masked_m = N * M_max  (over-approximation, see §4)│
                  └────────┬─────────────────────────────┘
                           ▼
                  ┌──────────── ExpertsEPLL ─────────────┐
                  │ run_experts_ll: masked grouped GEMM x2 │
                  │ around silu_and_mul                  │
                  │ (torch ref or triton kernel)         │
                  └────────┬─────────────────────────────┘
                           ▼  expert_out[E_local, N*M_max, H]
                  ┌──────────── CombineEPLL ─────────────┐
                  │ rev_send = expert_out.view(E_local, N, M_max, H)│
                  │                  .permute(1, 0, 2, 3).contiguous()│
                  │ NCCL all_to_all_single(rev_send -> rev_recv)│
                  │ scatter-reduce: for each (r, e, m) where original_indices[r,e,m,0] >= 0,│
                  │   final[t] += rev_recv[r, e, m] * topk_w[t, k]│
                  └────────┬─────────────────────────────┘
                           ▼
                  out[T, H]
```

Two NCCL collectives per MoE layer (forward a2a + reverse a2a). Both are dense fixed-size — no device-side split sizes, no ragged inputs.

---

## 3. Buffer ownership

All EP-LL workspaces live on `MoeBackend` (single per-engine artifact) and are registered onto the per-layer modules via the `RegistryOrchestrator`. This matches the Q4 decision in `design.md`.

Per-rank workspace (sized for max load — see §4 for `M_max`):


| name                 | shape                                | dtype | bytes (Qwen3-30B-A3B, N=8, E_local=16, M_max=512, H=2048) |
| -------------------- | ------------------------------------ | ----- | --------------------------------------------------------- |
| `send_buf`           | `[N, E_local, M_max, H]`             | bf16  | 256 MB                                                    |
| `recv_buf`           | `[N, E_local, M_max, H]`             | bf16  | 256 MB                                                    |
| `rev_send`           | `[N, E_local, M_max, H]`             | bf16  | 256 MB                                                    |
| `rev_recv`           | `[N, E_local, M_max, H]`             | bf16  | 256 MB                                                    |
| `original_indices`   | `[N, E_local, M_max, 2]`             | int32 | 256 KB                                                    |
| `ll_inter_workspace` | `[E_local, N*M_max, N_intermediate]` | bf16  | 96 MB                                                     |
| `ll_out_workspace`   | `[E_local, N*M_max, H]`              | bf16  | 256 MB                                                    |
| `topk_weights_buf`   | `[T_cap, K]`                         | fp32  | small                                                     |
| `topk_ids_buf`       | `[T_cap, K]`                         | int32 | small                                                     |
| **total**            |                                      |       | **~1.4 GB / rank**                                        |


Reuses across the 48 layers (workspaces are per-engine, not per-layer). With Qwen3-30B-A3B's per-rank weights at ~3.6 GB plus the ~1.4 GB EP workspace plus KV cache, comfortably fits in 24 GB.

---

## 4. M_max policy

`M_max` is the static upper bound on the number of `(token, k)` pairs that any one rank can route to any one of its local experts in a single batch.

Auto-sized in `MoeBackend._init_ep_ll_buffers`:

```
avg_per_bucket = ceil(T_cap * K / (N_ranks * E_local))
M_max          = max(8, avg_per_bucket * 4)        # 4× imbalance budget
```

For Qwen3-30B-A3B at `T_cap=2048, K=8, N=8, E_local=16`: avg = 128 → `M_max = 512`. Override with `Config.moe_ll_m_max`.

**Overflow policy: hard error in eager mode.** `DispatchEPLL.forward` raises `RuntimeError` if any bucket exceeds `M_max`. Production would either (a) increase `M_max`, (b) apply Expert Parallel Load Balancing (EPLB) at training time, or (c) drop overflow tokens (correctness-affecting; out of scope).

---

## 5. The two TP groups problem

The existing `linear.py` / `embed_head.py` / `qwen3_moe.py` code reads `dist.get_world_size()` and shards everything by it — assumed `world_size == tp_size`. **Pure EP breaks this assumption**: `world_size = 8` for EP collectives but `tp_size = 1` for attention/embed (replicated).

Fix (`services/utils/parallel.py`):

```python
_TP_GROUP = None
def get_tp_world_size():  return dist.get_world_size(group=_TP_GROUP)
def get_tp_rank():        return dist.get_rank(group=_TP_GROUP)
def set_tp_group(group):  global _TP_GROUP; _TP_GROUP = group
```

`LLMEngine._ensure_distributed`:

```python
if dist.get_world_size() > 1:
    # collective: every rank calls dist.new_group(ranks=[r]) for r in range(N).
    for r in range(world_size):
        g = dist.new_group(ranks=[r])
        if r == my_rank: my_tp_group = g
    set_tp_group(my_tp_group)
```

Touched 4 files (`linear.py`, `embed_head.py`, `qwen3.py`, `qwen3_moe.py`) to swap `dist.get_world_size()` → `get_tp_world_size()` and `dist.all_reduce(y)` → `dist.all_reduce(y, group=get_tp_group())`. The TP-aware layers now treat themselves as un-sharded; `MoeBackend` and the `*EPLL` modules read the real `dist.get_world_size()` for the EP collectives.

This is a small refactor that keeps single-rank tests bit-exact while making pure-EP work cleanly.

---

## 6. Validation matrix

All tests live in `workshop/nanovllm_moe/_test_*.py`.


| step | test                                                              | what it gates                                                                                            | result                                          |
| ---- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| 1    | `_test_ep_a2a_cudagraph`                                          | NCCL all_to_all_single + cuda graph capture (2 and 8 GPUs)                                               | bit-exact, captures cleanly                     |
| 2    | `_test_masked_grouped_gemm`                                       | torch ref of inner kernel matches per-expert hand reference; padding rows zero; empty experts handled    | exact match                                     |
| 3    | `_test_triton_masked_grouped_gemm`                                | Triton inner kernel parity vs torch ref across 4 cases including realistic Qwen3-30B-A3B per-rank shapes | max abs diff ~1e-3, max rel ~3% (bf16 expected) |
| 4-5  | `_test_ep_ll_single_process`                                      | full EP-LL pipeline (Dispatch → kernel → Combine) at N=1 vs single-rank `torch_fused_moe`                | bit-exact (max diff = 0.0)                      |
| 6    | `_test_ep_ll_loader`                                              | E_local slicing in `ExpertsEPLL` loaders, concat across ranks == single-rank weights                     | exact match                                     |
| 7    | `_test_engine_ep_ll` (2 GPU, NUM_LAYERS=2)                        | full LLMEngine multi-process                                                                             | runs, produces 8 tokens                         |
| 8    | `_test_engine_ep_ll` (8 GPU, full 48 layers, eager, ep_ll_torch)  | first coherent text                                                                                      | `' Paris. The capital of the United Kingdom'`   |
| 10   | `_test_engine_ep_ll` (8 GPU, full 48 layers, eager, ep_ll_triton) | inner kernel swap                                                                                        | identical text, 60.4 s vs 65.8 s                |
| 9    | (skipped)                                                         | cuda graph for EP-LL                                                                                     | needs vectorized dispatch — see §8              |


---

## 7. Hardware reality on RTX 4090

EP-LL's "low-latency" claim assumes kernel-level a2a fusion (FlashInfer NVLink one-sided), which needs sm90+. Without it, dense `[N_ranks, E_local, M_max, H]` a2a transmits MORE bytes than EP-HT's ragged a2a:


|                                    | bytes per rank per direction per layer | at ~25 GB/s PCIe |
| ---------------------------------- | -------------------------------------- | ---------------- |
| EP-HT (ragged)                     | ~8 MB                                  | 0.3 ms           |
| EP-LL dense (M_max=512, our setup) | ~256 MB                                | 10 ms            |


For prefill of one token across 48 layers that's roughly **+480 ms/token** of pure transfer overhead vs what EP-HT would do. We measured **~7.5–8.2 s/token** end-to-end, which is dominated by:

1. Host-side dispatch loops (Python iteration in `DispatchEPLL`)
2. NCCL all_to_all overhead (256 MB × 2 directions × 48 layers)
3. Pure EP overhead (every rank duplicates attention work)
4. Inner kernel (small fraction)

So the perf gap between `ep_ll_torch` and `ep_ll_triton` is small (8.22 → 7.55 s/tok): the inner kernel isn't the bottleneck. Production wins would come from:

- Triton dispatch kernel (eliminates Python loops)
- CUDA graph capture (saves launch overhead)
- True compaction on receive (cuts kernel FLOPs by 4–8×)
- Fused a2a+kernel on Hopper/Blackwell

---

## 8. Pits and decisions

### 8.1 ❗ TP layers cached `dist.get_world_size()` at init

**Symptom:**  `RuntimeError: shape '[-1, 32, 128]' is invalid for input of size 10240` when running multi-process. With WORLD_SIZE=2 the QKVParallelLinear sharded its output but `Qwen3MoeAttention.q_size` was computed assuming non-sharded.

Two cooperating bugs: (a) `linear.py` used `dist.get_world_size()`, (b) `qwen3_moe.py:Qwen3MoeAttention.__init_`_ *also* used `dist.get_world_size()` directly to compute `q_size = (total_num_heads // tp_size) * head_dim`. Even after we fixed `linear.py`, `q_size` was still wrong.

**Fix:** route both through `get_tp_world_size()`. See §5.

### 8.2 ❗ ModelRunner.rank used the world rank for sampling decisions

`if self.rank == 0: sample()` — true only on world rank 0. In our SPMD setup every rank runs its own engine and needs the next token to feed back into attention.

**Fix:** `self.rank = get_tp_rank()` (always 0 in pure EP), so every rank samples.

### 8.3 ❗ EP-LL inner kernel correctness with non-compacted layout

After a2a, `recv_buf` is `[N, E_local, M_max, H]` where each per-source slice has only `recv_counts[s, e]` valid rows in its prefix and the rest are zero-padding. Reshape to `[E_local, N*M_max, H]` makes the valid rows interleaved with padding (NOT a clean prefix).

**Solution:** set `masked_m[e] = N*M_max` (process EVERYTHING). The padding rows are zero-padded by the sender, so they produce zero outputs through the linear-then-silu*linear-then-linear math. Those zeros land in slots whose `original_indices == -1`, which `CombineEPLL.scatter_reduce` skips.

**Cost:** kernel FLOPs scale with `N*M_max` per expert instead of the actual count. With M_max=512 and avg count ~128, that's 4× wasted FLOPs. Acceptable for the correctness-first reference.

**Optimization (future):** compact on receive (per-expert prefix-sum + scatter), set tight `masked_m[e]`. Either a small Triton kernel or a host loop (kills cuda-graph). For Hopper/Blackwell the FlashInfer/DeepGEMM masked kernels assume the same dense-padded layout we have, so they don't need compaction.

### 8.4 ❗ NCCL `destroy_process_group` hangs at shutdown

When ranks exit at slightly different times after their last collective, `destroy_process_group` blocks waiting for the TCPStore. The `[W TCPStore.cpp:125] recvValue failed` warnings are cosmetic.

**Workaround:** `sys.exit(0)` directly without `destroy_process_group()` from each rank. NCCL prints a "did not call destroy" warning but it's benign.

### 8.5 EP-LL test passes at N=1 with bit-exact match (Step 4-5)

Surprised by zero divergence vs `torch_fused_moe`. Both paths use the same accumulation order (`hidden @ w1.T` then `silu(g)*u @ w2.T`), and at N=1 the bucketing is just an in-order placement of (t, k) pairs. So the numerics are literally identical — not just within tolerance. Good safety net.

### 8.6 8-GPU NCCL all_to_all has skewed per-rank latency in cuda graph (Step 1)

In the validation test, rank 0 reported ~5 ms per replay while ranks 4 and 2 reported ~0.5 ms. The collective itself is correct (bit-exact across ranks); the per-rank measurement skew is due to staggered re-entry into the timing loop. Not blocking.

### 8.7 8-GPU torch vs triton inner kernel: tiny perf delta

`ep_ll_torch` 8.22 s/tok → `ep_ll_triton` 7.55 s/tok at full Qwen3-30B-A3B 48 layers. The inner kernel is only ~50 ms per layer (out of ~140 ms per layer total) and the swap saves maybe 30% of that, which adds up to ~700 ms over 48 layers — matches the measurement.

The dominant time goes to (1) NCCL transfers, (2) Python dispatch loops, (3) duplicated attention work across all 8 ranks. The triton kernel is a no-regret swap, but not where the headline perf comes from.

### 8.8 No cuda graph for EP-LL yet (Step 9 cancelled)

`DispatchEPLL.forward` has `for t in range(T): for k in range(K):` host loops to build `send_buf` and `original_indices`. These are incompatible with cuda-graph capture.

To enable cuda graph: write a Triton dispatch kernel that uses atomic counters to compute slot positions and scatter `hidden_states` + `original_indices` in parallel. ~150 LOC of Triton. Deferred — not blocking the design validation, and the hardware doesn't have the kernel-level a2a fusion that would make EP-LL truly low-latency anyway.

---

## 9. What changed in single-rank code

To support the TP subgroup pattern (§5):

- `artifacts/modeling/layers/linear.py`: 3 spots using `dist.get_world_size()` → `get_tp_world_size()`; one `dist.all_reduce(y)` → `dist.all_reduce(y, group=get_tp_group())`
- `artifacts/modeling/layers/embed_head.py`: 1 `dist.get_world_size()` and 1 `dist.get_rank()` swap; 1 `all_reduce` and 1 `gather` get explicit `group=get_tp_group()`
- `artifacts/modeling/models/qwen3.py`: 1 swap
- `artifacts/modeling/models/qwen3_moe.py`: 1 swap; plus `Qwen3MoeForCausalLM/Qwen3MoeModel/Qwen3MoeDecoderLayer` accept `is_ep_ll` and `m_max` kwargs and forward them to `FusedMoE`

To support EP-LL backend selection:

- `services/config.py`: `moe_impl` allowed values extended to include `ep_ll_torch`, `ep_ll_triton`. New `moe_ll_m_max` field.
- `artifacts/moe_backend/moe_backend.py`: split `__post_init_`_ into `_init_single_rank_buffers` and `_init_ep_ll_buffers`; added `run_experts_ll(...)` method.
- `artifacts/modeling/layers/moe/fused_moe.py`: branches in `__init__` between single-rank and EP-LL submodules; `forward` branches on `is_ep_ll`.
- `services/model_runner/model_runner.py`: parallel EP-LL wiring branch in the orchestrator loop; `MoeBackend` now constructed before `Qwen3MoeForCausalLM` so `M_max` is known to `FusedMoE`.
- `services/engine/llm_engine.py`: `_ensure_distributed` creates the per-rank size-1 TP subgroup when `dist.get_world_size() > 1`.

`_test_engine_dense.py` and `_test_engine_moe.py` continue to pass without modification.

---

## 10. How to run

Single-rank (regression check):

```bash
CUDA_VISIBLE_DEVICES=0 python -m workshop.nanovllm_moe._test_engine_moe
```

EP-LL on 2 GPUs, trimmed to 2 layers (smoke test):

```bash
WORLD_SIZE=2 NUM_LAYERS=2 python -m workshop.nanovllm_moe._test_engine_ep_ll
```

Full Qwen3-30B-A3B on 8 GPUs:

```bash
WORLD_SIZE=8 NUM_LAYERS=-1 MOE_IMPL=ep_ll_triton \
    python -m workshop.nanovllm_moe._test_engine_ep_ll
```

---

## 11. What's next (out of scope for this PR)

In rough priority order:

1. ~~**Triton dispatch kernel** to eliminate the Python loops and unblock CUDA graph capture.~~ → **DONE**, see §13.
2. ~~**TP × EP composition.**~~ → **DONE**, see §15.
3. **Compaction on receive** to reduce inner-kernel FLOPs by 4–8×. Pairs naturally with the Triton dispatch kernel (it can produce per-(rank, expert) counts as a side output). The sender-side cost is now zero (we skip `send_buf.zero_()`); the receiver-side gain would be in inner-kernel FLOPs.
4. **DeepEPLB** (load-balanced expert placement) to keep `M_max` tight in production.
5. **Hopper/Blackwell port:** swap `triton_masked_grouped_gemm` for FlashInfer `cute_dsl.blockscaled_gemm.grouped_gemm_nt_masked` (FP4) or DeepGEMM `m_grouped_fp8_gemm_nt_masked` (FP8). Layout is identical, so it's a one-file change.
6. **Shared experts** (Qwen3-MoE doesn't have any; DeepSeek-V3 does).
7. **Quantization** along the EP axis (FP8 weights, AWQ, etc.).
8. **Variable-size NCCL with device-resident split sizes** (via `ProcessGroupNCCL.allToAll` with `output_split_sizes` as a CUDA tensor) — would unlock cuda graph for EP-HT. Currently PyTorch's `dist.all_to_all_single` requires Python int lists.

---

## 12. (placeholder)

(left intentionally empty so the new sections start at a clear new number; numbering is not load-bearing)

---

## 13. Triton dispatch kernel + CUDA graph for EP-LL

**Goal.** Replace the host-side bucketing loop in `DispatchEPLL` with a pure-GPU Triton kernel so that the entire EP-LL forward (dispatch + a2a + masked GEMM + reverse a2a + scatter-reduce) is CUDA-graph capturable.

### 13.1 Investigation: existing implementations

Surveyed `sglang` and `vllm`:

- `**sglang/srt/layers/moe/ep_moe/kernels.py:608–911`** — `_fwd_kernel_ep_scatter_2` and `_fwd_kernel_ep_gather` are the canonical "EP scatter / gather" Triton kernels. Pattern:
  - 1 program per (received) token; processes the WHOLE `H` row at once (no H-tiling — fits in registers for `H ≤ 8K bf16`).
  - For each top-k expert: `tl.atomic_add(per_expert_counter, 1)` returns a unique slot index. Write the row at that slot.
  - Output is **compacted** (each expert's tokens contiguous, no zero-padding interior). Pairs with DeepGEMM's `m_grouped_fp8_gemm_nt_contiguous`.
  - There's also a separate `cumsum + m_indices` kernel (`_fwd_kernel_ep_scatter_1`) that runs first to publish `expert_start_loc` for the scatter.
- `**sglang/srt/layers/moe/token_dispatcher/standard.py:86–192`** — `StandardDispatcher` is NOT classical EP-HT. It's actually **EDP** (Expert-Data-Parallel): every rank sees all tokens, masks `topk_ids → -1` for non-local experts, runs the standard fused_moe, then `reduce_scatter` on the output. No per-token a2a at all. We did NOT adopt this — it's a different design point that doesn't reduce per-token bandwidth.
- `**vllm/model_executor/layers/fused_moe/fused_moe.py`** — uses the standard sgl_kernel path for single-rank; EP is delegated to DeepEP (separate library) which we don't link.

**Decision:** adopt sglang's `ep_scatter` pattern, simplified for our no-compaction layout. We keep the dense `[N, E_local, M_max, H]` send buffer (so the a2a payload size is static and graph-friendly) but use the SAME atomic-bucketing kernel pattern to produce `original_indices` and `local_counts` on-GPU.

### 13.2 The kernel

`workshop/nanovllm_moe/artifacts/moe_backend/triton_ep_ll_dispatch.py`:

- 1 program per token, iterate top-K experts; for each, atomic-bump `local_counts[target_rank, target_local]`, write row to `send_buf[target_rank, target_local, slot, :]` and `(t, k)` to `original_indices[target_rank, target_local, slot, :]`.
- Slot >= M_max: silently dropped (graph mode can't raise). Caller must size `M_max` conservatively. The torch-reference `ep_ll_torch` path still raises in eager mode for development.
- Grid capped at `min(T, 4096)` programs; each handles `ceil(T/4096)` tokens via stride.

### 13.3 The send-buffer skip-zero optimization

`DispatchEPLL.forward` no longer calls `send_buf.zero_()`. Reasoning (proven correct end-to-end):

- Stale slots in `send_buf` produce stale outputs through the kernel and reverse a2a.
- These stale outputs land at slots on the original sender rank where `original_indices == -1`.
- `CombineEPLL.scatter_reduce` filters by `original_indices >= 0` and skips those slots.
- Net effect: stale data flows through the pipeline harmlessly; we save ~256 MB of writes per layer.

Only `original_indices.fill_(-1)` (small) and `local_counts.zero_()` (tiny) are needed each call.

### 13.4 Combine made graph-friendly

The original `CombineEPLL.scatter_reduce` used `rec[valid_mask]` (boolean indexing → dynamic shapes → not graph capturable). Replaced with a dense version: clamp invalid `t_idx, k_idx` to 0, multiply contribution by a 0/1 valid mask, then `index_add_(0, t_safe, contrib)`. All shapes static. Invalid slots map to `t=0` with zero contribution → harmless extra adds.

### 13.5 Validation

```bash
# Parity: triton dispatch vs torch reference (single process, N=1)
.venv/bin/python -m workshop.nanovllm_moe._test_triton_ep_ll_dispatch
# → bit-exact on send_buf, original_indices, local_counts (modulo slot ordering
#   within a bucket, which the inner kernel handles symmetrically)

# 2 GPUs, 2 layers, eager:
WORLD_SIZE=2 NUM_LAYERS=2 MOE_IMPL=ep_ll_triton ENFORCE_EAGER=1 \
    python -m workshop.nanovllm_moe._test_engine_ep_ll
# → tokens [537, 3602, 7225, 28910, 57216, 1368, 87500, 269] in 14.05s

# 2 GPUs, 2 layers, CUDA graph:
WORLD_SIZE=2 NUM_LAYERS=2 MOE_IMPL=ep_ll_triton ENFORCE_EAGER=0 \
    python -m workshop.nanovllm_moe._test_engine_ep_ll
# → SAME tokens in 7.27s  (~2× speedup)

# 8 GPUs, CUDA graph:
WORLD_SIZE=8 NUM_LAYERS=2 MOE_IMPL=ep_ll_triton ENFORCE_EAGER=0 \
    python -m workshop.nanovllm_moe._test_engine_ep_ll
# → SAME tokens in 8.66s  (NCCL overhead dominates the small-batch decode)
```

### 13.6 Pits encountered

- `**local_counts` not allocated.** Initial run failed with `'DispatchEPLL' object has no attribute 'local_counts'`. The buffer was added to `MoeBackend._init_ep_ll_buffers` and to the orchestrator's `DispatchEPLL` registration list (alongside `send_buf`, `recv_buf`, `original_indices`, `topk_*_buf`).
- **Boolean-mask indexing in combine.** Quickly identified by trial — capture failed silently with no clear error. Replaced with the dense `clamp_min(0)` + multiply pattern.
- `**prepare_metadata_for_moe` for EP-LL must be a no-op.** It IS — `MoeBackend.prepare_metadata_for_moe` early-returns when `is_ep_ll`. (DispatchEPLL handles its own per-layer reset.)

---

## 14. EP-HT (high-throughput): ragged a2a, classical layout

**Goal.** Provide the canonical EP-HT design — variable-size NCCL `all_to_all_single`, sort by expert id, reuse the standard block-tiled `triton_fused_moe`. Eager-only.

### 14.1 Why eager-only?

`dist.all_to_all_single(output, input, output_split_sizes=..., input_split_sizes=...)` requires Python int lists for the splits. Reading `bincount(target_rank).tolist()` forces a host sync, which is incompatible with cuda graph capture. There is a path to lift this (NCCL supports device-resident sizes via `ProcessGroupNCCL` C++APIs), but it requires writing C++ glue or using `SymmetricMemory` (DeepEP's strategy). Out of scope here.

### 14.2 Modules

`workshop/nanovllm_moe/artifacts/modeling/layers/moe/`:

- `dispatch_ep_ht.py` — `DispatchEPHT`:
  - topk + softmax (cuda-graph compatible — pure GPU)
  - flatten `[T, K] → [T*K]` and sort by `target_rank = topk_ids // E_local`
  - `index_select` to permute hidden by sort order (replicates each token K times)
  - `bincount(target_rank, minlength=N)` → `send_counts[N]`
  - exchange `send_counts` via small `all_to_all_single` to learn `recv_counts[N]`
  - **host sync** (the only one): `send_counts.tolist()`, `recv_counts.tolist()`
  - 3 variable-size `all_to_all_single` calls: hidden, local-expert-id, topk-weight
  - `sgl_kernel.moe_align_block_size` on `recv_local_eid` (treated as `K=1`)
- `experts_ep_ht.py` — `ExpertsEPHT`:
  - holds `[E_local, 2N, H]` and `[E_local, H, N]` (same as ExpertsEPLL)
  - `forward(tok_meta)` → calls `MoeBackend.run_experts(...)` which dispatches to the SAME single-rank `triton_fused_moe`. Just with `K_in=1`.
- `combine_ep_ht.py` — `CombineEPHT`:
  - reverse `all_to_all_single` (same `output/input_split_sizes` swapped)
  - un-permute: `unperm[sort_perm] = rev`
  - weight-and-reduce: `out = (unperm.view(T, K, H) * topk_weights.unsqueeze(-1)).sum(dim=1)`

### 14.3 MoeBackend buffers

`MoeBackend._init_ep_ht_buffers`:

- `intermediate_cache1/2/3` sized `[2 * T_cap * K, 1, X]` (worst-case recv size with 2× imbalance budget).
- `moe_align_block_size` buffers sized for `E_local` experts and `2 * T_cap * K` recv rows.
- All variable-size payloads (recv_hidden, recv_local_eid, recv_topk_w) are allocated PER-CALL in `DispatchEPHT.forward` — they have batch-dependent shape that can't be persistent.

### 14.4 Validation

```bash
# 2 GPUs:
WORLD_SIZE=2 NUM_LAYERS=2 MOE_IMPL=ep_ht ENFORCE_EAGER=1 \
    python -m workshop.nanovllm_moe._test_engine_ep_ll
# → tokens [537, 40196, 398, ...] in 11.02s (PASS)

# 8 GPUs:
WORLD_SIZE=8 NUM_LAYERS=2 MOE_IMPL=ep_ht ENFORCE_EAGER=1 \
    python -m workshop.nanovllm_moe._test_engine_ep_ll
# → tokens [537, 3602, 7225, 7225, ...] in 15.83s (PASS)
```

Tokens differ from EP-LL because the bf16 reduction order in the inner kernel is different (compacted vs zero-padded). First few tokens match, then divergence amplifies through layers. With temperature=0 (greedy) sampling, even small numerical perturbations flip subsequent tokens — this is expected behavior for any EP-LL ↔ EP-HT comparison and not a correctness bug.

### 14.5 Pits encountered

- `**recv_topk_weights` precision.** Initially considered passing the real per-(t,k) weights as `topk_weights` to `triton_fused_moe`, but the inner kernel multiplies bf16 weight × bf16 row before reducing. To preserve precision, we now pass `weights = 1.0` to the inner kernel and apply the real fp32 weights AFTER the reverse a2a in `CombineEPHT.forward`. This also avoids re-sending weights through the reverse a2a.
- **Empty-recv shortcut.** If a rank ends up with zero received rows (no token routed to its experts), `moe_align_block_size` would assert; `ExpertsEPHT.forward` short-circuits to a `zeros((0, 1, H))` tensor and `CombineEPHT` handles `total_recv == 0` cleanly.

---

## 15. TP × EP composition

**Goal.** Allow `tp_size > 1` in the engine config. Each rank holds 1/TP of attention/embed/lm_head AND 1/EP of the experts, where world_size = TP × EP.

### 15.1 Layout convention

```
rank = ep_rank * TP + tp_rank
TP subgroup = [ep_rank * TP, ..., ep_rank * TP + TP - 1]   (size TP, contiguous)
EP subgroup = [tp_rank, tp_rank + TP, ..., tp_rank + (EP-1)*TP]   (size EP, strided)
```

Within an EP slot (= one TP subgroup), all ranks hold the SAME slice of expert weights (replicated across TP). Across EP slots, expert weights are sharded.

### 15.2 `services/utils/parallel.py` extension

Replaced the single `_TP_GROUP` global with two: `_TP_GROUP` and `_EP_GROUP`. New entry point:

```python
def init_parallel_groups(tp_size: int, world_size: int) -> None:
    # Constructs TP and EP subgroups collectively across all ranks.
    # All ranks call new_group(...) the same number of times in the same order.
```

Layout-derived helpers: `get_tp_world_size()`, `get_tp_rank()`, `get_ep_world_size()`, `get_ep_rank()`. Pure-EP (TP=1) and pure-TP (EP=1) are degenerate cases that fall through to size-1 subgroups.

### 15.3 Layer changes


| Layer                                                                        | Group used            |
| ---------------------------------------------------------------------------- | --------------------- |
| `LinearBase`, `QKVParallelLinear`, `RowParallelLinear` (`o_proj` all-reduce) | `get_tp_*`            |
| `VocabParallelEmbedding`, `ParallelLMHead` (gather)                          | `get_tp_*`            |
| `Qwen3MoeAttention`, `Qwen3Attention` (head sharding)                        | `get_tp_world_size()` |
| `MoeBackend` (`E_local = E / world_size`)                                    | `get_ep_*` ← changed  |
| `DispatchEPLL`, `DispatchEPHT`, `CombineEPLL`, `CombineEPHT` (a2a)           | `get_ep_*` ← changed  |
| `ExpertsEPLL._w*_loader`, `ExpertsEPHT._w*_loader` (which experts to load)   | `get_ep_*` ← changed  |


### 15.4 Sampling broadcast within TP group

For TP > 1, only `tp_rank == 0` of each EP slot has the gathered logits to sample. After sampling, `ModelRunner.run` broadcasts the token list within the TP subgroup so all TP ranks have matching tokens to update their KV cache state:

```python
if get_tp_world_size() > 1:
    obj_list = [token_ids]
    src_global = dist.get_global_rank(get_tp_group(), 0)
    dist.broadcast_object_list(obj_list, src=src_global, group=get_tp_group())
    token_ids = obj_list[0]
```

### 15.5 `dist.gather` `dst` is a GLOBAL rank

The original `ParallelLMHead.forward` had `dist.gather(logits, all_logits, 0, group=get_tp_group())`. PyTorch interprets `dst=0` as a GLOBAL rank, not a group rank — this fails for any TP subgroup that doesn't include world rank 0 (every EP slot except `ep_rank=0`). Fix:

```python
src_global = dist.get_global_rank(tp_grp, 0) if tp_grp is not None else 0
dist.gather(logits, all_logits, src_global, group=tp_grp)
```

### 15.6 Validation

```bash
# TP=2 × EP=4, EP-LL Triton, CUDA graph:
TP_SIZE=2 EP_SIZE=4 NUM_LAYERS=2 MOE_IMPL=ep_ll_triton ENFORCE_EAGER=0 \
    python -m workshop.nanovllm_moe._test_engine_tp_x_ep
# → tokens [537, 3602, 7225, 28910, 57216, 1368, 87500, 269]   ← IDENTICAL to pure EP=8
# (2-layer prefix; demonstrates TP+EP correctness end-to-end)

# TP=4 × EP=2, EP-LL Triton, CUDA graph:
TP_SIZE=4 EP_SIZE=2 NUM_LAYERS=2 MOE_IMPL=ep_ll_triton ENFORCE_EAGER=0 \
    python -m workshop.nanovllm_moe._test_engine_tp_x_ep
# → IDENTICAL tokens

# TP=2 × EP=4, EP-HT, eager:
TP_SIZE=2 EP_SIZE=4 NUM_LAYERS=2 MOE_IMPL=ep_ht ENFORCE_EAGER=1 \
    python -m workshop.nanovllm_moe._test_engine_tp_x_ep
# → PASS (tokens differ from EP-LL by reduction-order noise, as expected)
```

The fact that **TP=2×EP=4 and TP=4×EP=2 produce IDENTICAL tokens to pure EP=8** is the strongest correctness check — the math is partition-invariant.

### 15.7 Pits encountered

- The `dist.gather(... dst=0 ...)` bug (§15.5) was the only material issue. Caught immediately on the first multi-rank EP-slot run. Tests on pure EP missed it because every TP subgroup was size-1 (and rank 0 is trivially in any size-1 group containing rank 0).
- Sampling broadcast (§15.4) was the second issue; without it, only TP rank 0 of each EP slot had token_ids and the others returned None, breaking sequence-state updates.

