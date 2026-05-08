# EP-LL Memory Analysis & OOM Debugging Journal

This document captures one concrete OOM debugging session for the EP-LL path of
Qwen3-30B-A3B on 8 × RTX 4090 (24 GB each) and the per-rank memory budget the
engine needs in steady state. It is the "what happens, where it goes, what
fails first" companion to `docs/moe/ep.md` (which is the design / data-flow /
correctness document).

The bug fixed here is documented at the bottom (§6) — but the goal of writing
this doc is to make the *next* per-rank OOM diagnosable by reading one file.

---

## 1. The reproducer

```bash
python -m eval.test_bazaar_moe \
  --world-size 8 --tp-size 1 \
  --moe-impl ep_ll_triton --enforce-eager 0 \
  --num-problems 30 --max-tokens 8192 --max-model-len 8192
```

Hardware: 8 × RTX 4090, 24 GB / GPU. Model: `Qwen3-30B-A3B` bf16 (60 GB total
parameters).

This *should* fit easily for weight-only sharding (60 GB / 8 = 7.5 GB / rank
just for experts, plus ~1.5 GB / rank for replicated non-expert layers).
It originally OOM'd at startup during `capture_cudagraph` warmup with
`free=248 MiB, allocated=22.20 GiB, in cuda-graph pools=3.02 GiB`.

The fix is at the bottom (§6). The rest of the doc is the analysis you do
*before* you know there is a fix — i.e. the next time something OOMs.

---

## 2. Per-rank memory budget (Qwen3-30B-A3B, EP=8, TP=1)

Numbers below are computed from the model config and the explicit allocations
in `MoeBackend._init_ep_ll_buffers`. They match what `[MEM]` prints emit at
startup (see `_mem_print` in `services/model_runner/model_runner.py` — set
`MOE_NO_MEM=1` to silence).

### 2.1 Model weights per rank (~9.25 GB)

```
Qwen3-30B-A3B config:
  num_hidden_layers   = 48
  hidden_size H       = 2048
  moe_intermediate_size N_inter = 768
  num_experts E       = 128       → E_local = 16 per rank at EP=8
  num_attention_heads = 32        (q)
  num_key_value_heads = 4         (kv)
  head_dim            = 128
  vocab_size          = 151_936
  num_experts_per_tok K = 8
```

Per layer (replicated across all EP ranks since TP=1 inside an EP slot):

| Component | Bytes | Note |
|---|---|---|
| q_proj    | (32·128) × 2048 × 2 = 16 MB | replicated |
| k/v_proj  | 2 × (4·128) × 2048 × 2 = 4 MB | replicated |
| o_proj    | 2048 × 2048 × 2 = 8 MB | replicated |
| q_norm/k_norm + RMSNorms | < 1 MB | replicated |
| router gate | 2048 × 128 × 2 = 0.5 MB | replicated |
| 16 local experts × (w1: 1536×2048 + w2: 2048×768) × 2 = | **144 MB** | sharded — only owned 16/128 |
| **per-layer total** | **~172 MB** | |

× 48 layers = **~8.1 GB**

Embed + LM head (replicated): 2 × 151_936 × 2048 × 2 ≈ **1.2 GB**

**Weights per rank: ~9.25 GB.** This is the irreducible floor.

### 2.2 EP-LL persistent workspaces (~5.4 GB at default M_max)

`M_max` is the static per-`(rank, local_expert)` bucket capacity used by
`DispatchEPLL` / `CombineEPLL`. Default policy in `MoeBackend._init_ep_ll_buffers`:

```python
avg_per_bucket = ceil(T_cap * K / (N * E_local))
M_max          = max(8, avg_per_bucket * 4)     # 4× imbalance budget
```

For `T_cap = max_num_batched_tokens = 8192`, `K = 8`, `N = 8`, `E_local = 16`:

```
avg_per_bucket = ceil(8192 · 8 / (8 · 16)) = 512
M_max          = 2048
```

Each `[N=8, E_local=16, M_max=2048, H=2048]` bf16 tensor is **exactly 1 GB**.
Persistent buffers in `MoeBackend`:

| Buffer | Shape | Size |
|---|---|---|
| `send_buf`   | [N, E_local, M, H]  bf16 | 1.00 GB |
| `recv_buf`   | [N, E_local, M, H]  bf16 | 1.00 GB |
| `rev_send`   | [N, E_local, M, H]  bf16 | 1.00 GB |
| `rev_recv`   | [N, E_local, M, H]  bf16 | 1.00 GB |
| `hidden_recv` (added in §6) | [E_local, N·M, H] bf16 | 1.00 GB |
| `ll_inter_workspace` | [E_local, N·M, N_inter] bf16 | 0.38 GB |
| `ll_out_workspace`   | [E_local, N·M, H]       bf16 | 1.00 GB |
| `original_indices`   | [N, E_local, M, 2] int32 | 2 MB |
| `local_counts`, `masked_m_buf`, topk bufs | (small) | < 1 MB |
| **TOTAL** | | **~5.4 GB** (or 6.4 GB w/ hidden_recv) |

This dominates the per-rank "non-weight, non-KV" footprint. **Halving `M_max`
halves this** — 2× balance budget instead of 4× saves ~3 GB / rank, at the
cost of an assertion-failure risk if routing imbalance ever exceeds 2×.

### 2.3 KV cache (greedy fill, ~5.7 GB at gpu_memory_utilization=0.85)

Allocated by `ModelRunner.allocate_kv_cache`:

```python
budget_bytes = total_mem * gpu_memory_utilization − used − peak + current
num_kvcache_blocks = budget_bytes // block_bytes

block_bytes = 2 (k+v) × num_hidden_layers × kvcache_block_size
              × (num_kv_heads // tp_size) × head_dim × 2 bytes
```

For Qwen3-30B-A3B at TP=1: `2 × 48 × 1 × 4 × 128 × 2 = 49,152 B/block`.

The KV cache is **greedy** — it fills whatever is left under
`gpu_memory_utilization`. So if you raise `M_max` or `T_cap`, the KV cache
shrinks 1:1 (and your max effective `max_model_len * max_num_seqs` shrinks
with it).

### 2.4 CUDA-graph private pool (after §6 fix: ~150 MB)

Allocated during `capture_cudagraph` for transients live inside captured
graphs. Pre-fix: ~3 GB (see §5). Post-fix: ~150 MB (just the small per-batch
metadata tensors).

### 2.5 Steady-state breakdown (the post-fix numbers we actually observed)

```
[MEM] before load_model      | rank0: cur=16.55GiB peak=16.55GiB free=6.01GiB
[MEM] after  load_model      | rank0: cur=16.55GiB peak=16.55GiB free=6.01GiB
[MEM] after  allocate_kv_cache | rank0: cur=19.03GiB peak=19.03GiB free=3.53GiB
[MEM] after  capture_cudagraph | rank0: cur=19.33GiB peak=19.34GiB free=3.06GiB
```

Reading this:
- `before load_model = 16.55 GB` already includes weight tensors (allocated by
  `nn.Module.__init__`, not yet *populated*) **plus** the entire `MoeBackend`
  EP-LL workspace. So 16.55 ≈ 9.25 (weights) + 6.4 (workspaces incl. hidden_recv) + ~1 GB misc.
- `load_model` is a no-op for the allocator — it copies disk → already-allocated
  tensors.
- KV cache grew the footprint by ~2.5 GB (small because `gpu_memory_utilization=0.85`
  leaves only ~3 GB headroom after weights+workspaces; the budget calc
  consumes most of it).
- Cuda-graph capture grew by only **~300 MB** (was 3 GB before §6).

---

## 3. Where it actually OOM'd (pre-fix)

The OOM stack pinpointed `dispatch_ep_ll.py:196`:

```python
hidden_recv = self.recv_buf.permute(1, 0, 2, 3).contiguous().view(E_local, N * M_max, H)
```

`.contiguous()` on a non-contiguous permute materializes a *new* `[E_local, N, M_max, H]`
bf16 tensor — exactly 1 GB at this config. The `combine_ep_ll.py` pair did the
same trick twice more:

```python
# L70:
rev_send_view = expert_out.view(E, N, M, H).permute(1, 0, 2, 3).contiguous()  # 1 GB
self.rev_send.copy_(rev_send_view)

# L98:
contrib = rec * (w * valid_mask)            # 1 GB
out.index_add_(0, t_safe, contrib)
```

So *per layer per forward call* the layer body needed 3 × 1 GB transient
allocations. In eager mode this is fine — each is freed before the next layer.

**During `torch.cuda.graph(...)` capture, however, all allocations made on the
capture stream go into the graph's private memory pool, and that pool's HWM
must hold whatever can be simultaneously live during replay**. PyTorch's
allocator tracks this conservatively and the pool ended up at 3 GB across our
8 captured decode batch sizes (1..8).

The OOM crash showed:
```
GPU 5: 23.52 GiB total, 248 MiB free
PyTorch allocated 22.20 GiB (3.02 GiB in private pools)
Tried to allocate 1024.00 MiB  ← the next layer's hidden_recv transient
```

So we were doing fine on the static budget (~20.4 GB after KV) and the
captured graphs (~3 GB private pool brought us to ~23.4 GB). The next 1 GB
transient inside the *eager warmup pass* immediately preceding the next graph
capture had nowhere to go (`capture_cudagraph` does an eager warmup *then*
captures, see `model_runner.py:470-472`).

### Why the total of 22.20 GiB makes sense

```
weights (9.25)
+ MoeBackend EP-LL persistent (5.4 incl. send/recv/rev_send/rev_recv/ll_workspaces)
+ KV cache (~5.7 at 0.85 utilization with the workspaces present)
+ cuda-graph private pool (3.02)
≈ 23.4 GiB ← within 23.52 GiB capacity by ~120 MiB
+ 1024 MiB transient request → OOM
```

---

## 4. The recipe for diagnosing the next per-rank OOM

When you see a multi-GPU EP run go OOM:

1. **Read the error message.** PyTorch's OOM error includes
   `total / free / allocated / in private pools` — those four numbers tell you
   the static-vs-graph split. Note which file/line is in the traceback.

2. **Look at the `[MEM]` snapshots from `_mem_print`.** They print at four
   milestones: `before load_model` (covers weights + MoeBackend allocs),
   `after load_model` (no-op for allocator), `after allocate_kv_cache`
   (greedy fill up to `gpu_memory_utilization`), `after capture_cudagraph`
   (graph private pool size = `cur` − `cur after KV`).

3. **Compute the budget by hand.** Use the table in §2 with your config's
   `T_cap`, `K`, `N`, `E_local`, `H`, `N_inter`. Buffers scale linearly in
   `M_max` and proportionally in `H` and `N_inter`. The two big knobs are:
   - `T_cap = max_num_batched_tokens` → drives `M_max` → drives all 4 a2a
     buffers + ll_inter + ll_out.
   - `gpu_memory_utilization` → caps total static usage and squeezes KV cache.

4. **If the OOM is from a transient inside a layer body**, the fix is one of:
   (a) Hoist it to a persistent MoeBackend workspace, (b) replace the operation
   with an in-place / strided one that avoids `.contiguous()` and avoids
   creating a new same-sized tensor (see §6), (c) shrink `M_max` if routing
   tolerance allows.

5. **If the OOM is during `allocate_kv_cache`**, your weights + MoeBackend
   allocations exceed `gpu_memory_utilization * total - peak`, and the KV
   budget went negative. Lower `gpu_memory_utilization`, or shrink workspace
   sizes by lowering `T_cap` / `M_max`.

---

## 5. Cuda-graph private pool: pitfalls we hit

Two relevant facts about `torch.cuda.graph(g, pool=...)`:

1. **Allocations made during capture go into the graph's private pool**. The
   pool's HWM (high-water mark) is what gets reserved for the graph's lifetime
   — it does NOT shrink after capture even if individual allocations
   alloc/free repeatedly within the capture.

2. **The eager warmup pass right before each capture (`model_runner.py:470`)
   runs in eager mode, but on the same stream as the upcoming capture and
   right after a previous capture finished**. If a previous capture's pool is
   already at 3 GB and the eager warmup needs another 1 GB transient, you can
   easily overshoot the `gpu_memory_utilization` headroom. This is what bit us.

The fix below removes the per-layer 1 GB transients entirely — the cuda-graph
private pool drops from 3 GB → 150 MB and we get 3+ GB of headroom back.

---

## 6. The fix (commit-level summary)

Three small changes, no architectural surgery, no kernel rewrites:

### 6.1 `MoeBackend._init_ep_ll_buffers` — add a persistent input workspace

```python
# Layout: [E_local, N*M_max, H], same as the inner-kernel expects.
# Without this, every DispatchEPLL.forward materializes
# `recv_buf.permute(1,0,2,3).contiguous()` — a fresh ~1 GB tensor that OOMs
# cuda-graph capture on tight budgets.
self.hidden_recv = torch.empty(
    (E_local, N * M, H), dtype=self.dtype, device=self.device,
)
```

Wired through `RegistryOrchestrator` in `ModelRunner.__init__`:

```python
if isinstance(module, DispatchEPLL):
    for name in (..., "hidden_recv"):
        orch.register(moe_backend, name, module)
```

### 6.2 `DispatchEPLL.forward` — write into the persistent buffer in place

```python
# Before:
hidden_recv = self.recv_buf.permute(1, 0, 2, 3).contiguous().view(E_local, N * M_max, H)

# After:
self.hidden_recv.view(E_local, N, M_max, H).copy_(self.recv_buf.permute(1, 0, 2, 3))
hidden_recv = self.hidden_recv.view(E_local, N * M_max, H)
```

`copy_` between the strided permute view and the contiguous workspace performs
the exact same data movement (transposed-read, contiguous-write) without
allocating a new tensor.

### 6.3 `CombineEPLL.forward` — eliminate `rev_send_view` and `contrib`

Two sibling transients in combine, both eliminated:

```python
# (1) Was: materialize transposed expert_out into a new contiguous tensor,
#          then copy_ into rev_send. Now: write through a transposed VIEW of
#          rev_send directly. No transient.
#
# Before:
rev_send_view = expert_out.view(E, N, M, H).permute(1, 0, 2, 3).contiguous()
self.rev_send.copy_(rev_send_view)

# After:
self.rev_send.permute(1, 0, 2, 3).copy_(expert_out.view(E, N, M, H))
```

```python
# (2) Was: materialize a per-row scaled copy of rev_recv into `contrib`,
#          then index_add_. Now: scale rev_recv IN-PLACE (it's about to be
#          overwritten by the next layer's all_to_all anyway).
#
# Before:
contrib = rec * (w * valid_mask)            # 1 GB transient
out.index_add_(0, t_safe, contrib)

# After:
rec.mul_(w * valid_mask)                    # in-place, no transient
out.index_add_(0, t_safe, rec)
```

`(w * valid_mask)` is `[N*E*M, 1]` bf16 — a few hundred KB. The broadcast
multiply against `rec: [N*E*M, H]` happens in place via `Tensor.mul_`.

### 6.4 Net effect

| Phase (per rank) | Pre-fix | Post-fix |
|---|---|---|
| After `load_model` | 16.55 GB | 16.55 GB |
| After `allocate_kv_cache` | 20.4 GB | 19.03 GB |
| Cuda-graph private pool growth | **OOM at 3.02 GB** | ~150 MB |
| Peak memory | **OOM at 22.2 GB** | 19.34 GB |
| Free at end | 0.24 GB | 3.06 GB |

The post-fix `after KV` is *lower* than pre-fix because `hidden_recv` (1 GB)
is now persistent and accounted for in the KV budget, so the greedy KV
allocation gives back ~1 GB. That's a fair trade — predictable and stable
versus a non-deterministic OOM during graph capture.

Correctness was verified by re-running the 2-rank `--num-layers 2` parity
smoke test (bit-identical output to pre-fix), then end-to-end on 8 GPUs
with cuda-graph capture (the originally-OOMing config).

---

## 7. Knobs you have

When you hit a per-rank memory wall, in roughly the order to try:

1. **`--gpu-memory-utilization 0.80`** (default 0.85). Frees 1.2 GB / 24 GB
   for transients without changing model semantics.
2. **`--moe-ll-m-max <value>`** (default `4 × balanced_avg`). Halving it
   halves all 7 of the EP-LL workspaces. **Only safe if EPLB / training-time
   load-balancing keeps your routing imbalance under 2×**; otherwise you'll
   hit the in-kernel overflow assertion.
3. **`--max-num-batched-tokens <smaller>`** (default 65536, capped to whatever
   fits). Caps `T_cap` which directly determines the auto-`M_max` and the
   intermediate-cache sizes (single-rank / EP-HT only).
4. **`--enforce-eager 1`**. Skips cuda-graph capture entirely → frees the
   private pool budget but ~2× decode latency.
5. **Switch to `ep_ht`** if your traffic is throughput-heavy. EP-HT does not
   pre-allocate the dense `[N, E_local, M_max, H]` buffers, so it has a much
   smaller persistent footprint at the cost of being eager-only (variable-size
   NCCL needs Python-int splits).
6. **Sharding choice** (TP × EP). Increasing TP shrinks per-rank weights and
   KV cache (`num_kv_heads // tp_size`); see `docs/moe/ep.md` §15. With TP=2
   the KV cache shrinks proportionally; with TP=4 you can fit dramatically
   wider sequence lengths.

---

## 8. Things to do next (notes-to-self)

- **Revisit the default `M_max` policy.** 4× balance is conservative for a
  trained-and-load-balanced MoE like Qwen3-30B-A3B. 2× would save ~3 GB / rank
  at almost zero correctness risk. Sane defaults matter — most users will
  hit the OOM wall before they think to tune `moe_ll_m_max`.
- **Alias hidden_recv with rev_recv.** Their lifetimes don't overlap (see
  §6 of `docs/moe/ep.md`'s data flow). Aliasing would save 1 GB per rank but
  requires a small refactor (the kernel input view needs to come from the
  aliased storage with the right strides).
- **Write a `--print-budget` flag** that prints the §2 table for the current
  config before any allocation happens, so users can size up before hitting
  an OOM.
- **Dump cuda-graph private pool size per captured batch_size** during
  `capture_cudagraph` to make graph-pool growth observable directly.

---

## 9. Provenance

This doc was written after a single OOM debug session on 2026-05-08. The
reproducer command, the exact line numbers and the before/after `[MEM]`
snapshots are all from that session. `services/model_runner/model_runner.py`'s
`_mem_print` was added as part of the same session — keep it around; the
prints are cheap and the next per-rank OOM will thank you for them.
