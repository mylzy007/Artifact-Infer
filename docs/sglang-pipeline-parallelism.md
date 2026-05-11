# Pipeline Parallelism in SGLang — Implementation Reference

> Source: [`sglang`](https://github.com/sgl-project/sglang) main branch, April 2026.

---

## 1. Overview

SGLang implements **end-to-end pipeline parallelism (PP)** for LLM serving.
The design splits a model's transformer layers across multiple GPUs (stages), with each stage running its own scheduler process. Activations flow stage-to-stage via P2P tensor sends over NCCL, while scheduling metadata propagates via CPU-side pickling over Gloo.

Key design choices:

- **One scheduler process per (pp_rank, tp_rank)** — PP composes with TP.
- **Microbatched event loop** — the scheduler cycles through `pp_size + pp_async_batch_depth` microbatch slots.
- **Async send / sync recv** — avoids desynchronisation while hiding send latency.
- **Typed tensor dict multiplexing** — proxy (hidden-state) tensors and output (token-id) tensors share the same PP channel, demultiplexed by a `__msg_type__` tag.

---

## 2. Process Group Setup

**File**: `sglang/srt/distributed/parallel_state.py`

### 2.1 Rank Mapping

```
global_rank = tp_size * pp_rank + tp_rank
world_size  = tp_size * pp_size
```

### 2.2 Group Creation

`initialize_model_parallel(pipeline_model_parallel_size=N)` creates PP groups with **strided** ranks:

```python
# For world_size=8, tp_size=2, pp_size=4:
#   TP groups (consecutive):   [0,1], [2,3], [4,5], [6,7]
#   PP groups (strided):       [0,2,4,6], [1,3,5,7]
num_pp_groups = world_size // pp_size
for i in range(num_pp_groups):
    ranks = list(range(i, world_size, num_pp_groups))
    # → [0,2,4,6] and [1,3,5,7]
```

The PP group is a `GroupCoordinator` with `group_name="pp"`, **without** custom all-reduce (only P2P needed).

### 2.3 GroupCoordinator PP Primitives

| Method | Transport | Purpose |
|--------|-----------|---------|
| `send_tensor_dict` | NCCL (GPU tensors) + Gloo (CPU metadata) | Send hidden states / output tokens to next stage |
| `recv_tensor_dict` | NCCL + Gloo | Receive from previous stage |
| `send_object` / `recv_object` | Gloo (CPU pickle) | Send scheduling metadata (request lists, RIDs) |
| `barrier` | Gloo CPU | Synchronise stages |

`send_tensor_dict` splits a `Dict[str, Tensor | Any]` into:
1. **Metadata list** — `(key, TensorMetadata(device, dtype, size))` or `(key, value)` for non-tensors — sent as a pickled object via Gloo.
2. **Tensor payload** — each tensor sent individually via NCCL `isend`/`send`.

Optional **send-allgather** optimisation: when an `all_gather_group` is provided (the attention-TP group), only a 1/N slice is sent and the receiver does an all-gather to reconstruct.

---

## 3. Scheduler PP Event Loop

**File**: `sglang/srt/managers/scheduler_pp_mixin.py`

Class `SchedulerPPMixin` is mixed into `Scheduler` and provides three event loop variants:

| Loop | When |
|------|------|
| `event_loop_pp` | Normal serving |
| `event_loop_pp_disagg_prefill` | Disaggregated prefill server |
| `event_loop_pp_disagg_decode` | Disaggregated decode server |

### 3.1 Microbatch State

```python
pp_loop_size = pp_size + pp_async_batch_depth

mbs           = [None] * pp_loop_size   # current batch per slot
last_mbs      = [None] * pp_loop_size   # previous batch per slot
running_mbs   = [ScheduleBatch()] * pp_loop_size
mb_metadata   = [None] * pp_loop_size   # PPBatchMetadata (cuda-graph flag)
pp_outputs    = None                     # PPProxyTensors from last stage
last_rank_comm_queue = deque()           # (event, outputs) buffer on last rank
```

### 3.2 Per-Microbatch Iteration (Unified Schedule)

```
for mb_id in range(pp_loop_size):
  ── recv requests from tokenizer (first rank) or previous stage ──
  ── forward recv_reqs to next stage (async send) ──
  ── schedule: get_next_batch_to_run() ──
  ── recv proxy tensors from previous stage (if not first rank) ──
  ── commit previous send-output work ──
  ── recv + preprocess previous microbatch's output tensors ──
  ── launch current batch on GPU (forward_stream) ──
  ── D2H sync on copy_stream for previous microbatch ──
  ── process_batch_result for previous microbatch ──
  ── send proxy tensors to next stage (async send, if not last rank) ──
  ── send output tensors (last rank → rank 0, or passthrough) ──
```

Key ordering invariant: a batch's **output** is processed one full `pp_size` round later, enabling the GPU computation of the current batch to overlap with the CPU post-processing of the previous.

### 3.3 Communication Helpers

```python
def _pp_send_pyobj_to_next_stage(data, async_send):
    """CPU-side pickle send via point_to_point_pyobj (Gloo)."""

def _pp_recv_pyobj_from_prev_stage():
    """CPU-side pickle recv, then broadcast within attn-TP / attn-CP groups."""

def _pp_send_dict_to_next_stage(tensor_dict, msg_type="proxy"|"output"):
    """GPU-side tensor send via pp_group.send_tensor_dict."""
    tensor_dict["__msg_type__"] = msg_type  # tag for demux

def _pp_recv_typed_dict(expected_kind):
    """Recv and demultiplex by __msg_type__; stash mismatched messages."""

def _pp_recv_proxy_tensors() -> PPProxyTensors:
    """Wrapper that calls _pp_recv_typed_dict(expected_kind="proxy")."""
```

### 3.4 Event Loop Dispatch

**File**: `sglang/srt/managers/scheduler.py :: dispatch_event_loop`

```python
if pp_size > 1:
    if disagg == NULL:   scheduler.event_loop_pp()
    elif disagg == PREFILL: scheduler.event_loop_pp_disagg_prefill()
    elif disagg == DECODE:  scheduler.event_loop_pp_disagg_decode()
```

---

## 4. Worker-Level Forwarding

**File**: `sglang/srt/managers/tp_worker.py`

`TpModelWorker.forward_batch_generation` branches on PP rank:

```python
if self.pp_group.is_last_rank:
    out = self.model_runner.forward(forward_batch, pp_proxy_tensors=...)
    logits_output = out.logits_output
    next_token_ids = self.model_runner.sample(logits_output, forward_batch)
    return GenerationBatchResult(logits_output=logits_output,
                                 next_token_ids=next_token_ids, ...)
else:
    out = self.model_runner.forward(forward_batch, pp_proxy_tensors=...)
    pp_proxy_tensors = out.logits_output  # actually hidden states
    return GenerationBatchResult(pp_hidden_states_proxy_tensors=pp_proxy_tensors, ...)
```

Non-last ranks **never sample** — they return intermediate `PPProxyTensors` instead of logits.

---

## 5. PPProxyTensors

**File**: `sglang/srt/model_executor/forward_batch_info.py`

```python
class PPProxyTensors:
    tensors: Dict[str, torch.Tensor]
    # Typically: {"hidden_states": ..., "residual": ...}
    # Or for output: {"next_token_ids": ..., optional logprob tensors}
```

A thin wrapper around a dict of tensors that flows between stages. Supports `__getitem__`, `__setitem__`, slicing.

---

## 6. Model-Level Layer Partitioning

### 6.1 `get_pp_indices`

**File**: `sglang/srt/distributed/utils.py`

```python
def get_pp_indices(num_hidden_layers, pp_rank, pp_size) -> (start, end):
    # Respects SGLANG_PP_LAYER_PARTITION env var for manual splits.
    # Otherwise: evenly distribute, extra layers go to last N partitions.
```

### 6.2 `make_layers`

**File**: `sglang/srt/utils/common.py`

```python
def make_layers(num_hidden_layers, layer_fn, pp_rank, pp_size, ...):
    start, end = get_pp_indices(num_hidden_layers, pp_rank, pp_size)
    modules = ModuleList(
        [PPMissingLayer() for _ in range(start)]          # placeholder before
        + [layer_fn(idx) for idx in range(start, end)]    # real layers
        + [PPMissingLayer() for _ in range(end, total)]   # placeholder after
    )
    return modules, start, end
```

### 6.3 `PPMissingLayer`

**File**: `sglang/srt/layers/utils/common.py`

```python
class PPMissingLayer(torch.nn.Identity):
    """Pass-through placeholder for layers not on this PP stage."""
    def forward(self, *args, **kwargs):
        return args[0]  # or wrapped in tuple if return_tuple=True
```

### 6.4 Per-Model Integration Pattern (e.g. `LlamaModel`)

**File**: `sglang/srt/models/llama.py`

```python
class LlamaModel(nn.Module):
    def __init__(self, ...):
        self.pp_group = get_pp_group()

        # Embedding: only on first rank
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(...)
        else:
            self.embed_tokens = PPMissingLayer()

        # Layers: partitioned
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers, ...,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
        )

        # Final norm: only on last rank
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(...)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def forward(self, input_ids, ..., pp_proxy_tensors=None):
        if self.pp_group.is_first_rank:
            hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(hidden_states, residual, ...)

        if not self.pp_group.is_last_rank:
            return PPProxyTensors({"hidden_states": hidden_states,
                                   "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

**60+ model files** follow this exact pattern.

---

## 7. Weight Loading

**File**: `sglang/srt/model_loader/loader.py`

The weight loader skips parameters whose layer index falls outside `[start_layer, end_layer)`. The layer ID is extracted from weight names like `model.layers.10.self_attn.qkv_proj.weight`.

Embedding and LM-head weights are only loaded on the first/last rank respectively.

---

## 8. Process Spawning

**File**: `sglang/srt/managers/data_parallel_controller.py`

`DataParallelController` spawns one `mp.Process` per `(pp_rank, tp_rank)`:

```python
for pp_rank in range(pp_size):
    for tp_rank in range(tp_size):
        gpu_id = pp_rank * tp_size + tp_rank  # or node-aware mapping
        Process(target=run_scheduler_process,
                args=(pp_rank, tp_rank, gpu_id, ...))
```

Each process initialises torch.distributed with:
```python
init_distributed_environment(
    world_size=tp_size * pp_size,
    rank=tp_size * pp_rank + tp_rank,
    local_rank=gpu_id, ...
)
initialize_model_parallel(
    tensor_model_parallel_size=tp_size,
    pipeline_model_parallel_size=pp_size, ...
)
```

---

## 9. CLI Flags & Configuration

**File**: `sglang/srt/server_args.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--pipeline-parallel-size` / `--pp-size` | 1 | Number of PP stages |
| `--pp-max-micro-batch-size` | None | Max tokens per microbatch |
| `--pp-async-batch-depth` | 0 | Extra microbatch slots for async overlap |
| `SGLANG_PP_LAYER_PARTITION` (env) | None | Comma-separated layer counts per stage |

### Validation Constraints

```python
def _handle_pipeline_parallelism(self):
    if self.pp_size > 1:
        self.disable_overlap_schedule = True
        # Also: piecewise cuda graph disabled

# Other guards:
assert pp_size == 1 or not enable_context_parallel
assert pp_size == 1 or not enable_pdmux
assert pp_size == 1 or speculative_algorithm != "dflash"
```

---

## 10. Component Dependency Graph

```
CLI (--pp-size N)
 └─ ServerArgs
     └─ DataParallelController (spawn per pp_rank × tp_rank)
         └─ Scheduler
             ├─ SchedulerPPMixin (microbatch event loop)
             │   ├─ _pp_send_pyobj_to_next_stage()  ← Gloo (CPU metadata)
             │   ├─ _pp_send_dict_to_next_stage()    ← NCCL (GPU tensors)
             │   └─ _pp_recv_typed_dict()             ← demux by msg_type
             └─ TpModelWorker
                 └─ ModelRunner
                     ├─ init_torch_distributed()
                     │   └─ initialize_model_parallel(pp_size=N)
                     │       └─ GroupCoordinator("pp")
                     │           ├─ send_tensor_dict / recv_tensor_dict
                     │           └─ send_object / recv_object
                     ├─ Model (e.g. LlamaForCausalLM)
                     │   ├─ embed_tokens   → PPMissingLayer (if not first)
                     │   ├─ layers         → make_layers(pp_rank, pp_size)
                     │   │                   └─ get_pp_indices → [start, end)
                     │   ├─ norm / lm_head → PPMissingLayer (if not last)
                     │   └─ forward: emit PPProxyTensors (non-last) or logits (last)
                     └─ Weight Loader (skip layers outside [start, end))
```

---

## 11. Data Flow Diagram

```
Stage 0 (PP rank 0)          Stage 1 (PP rank 1)          Stage 2 (PP rank 2, last)
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│ embed_tokens     │          │                  │          │                  │
│ layers[0:10]     │─proxy──▶│ layers[10:20]    │─proxy──▶│ layers[20:30]    │
│ PPMissingLayer   │ tensors  │ PPMissingLayer   │ tensors  │ norm + lm_head   │
│ (norm=Identity)  │          │ (embed=Identity) │          │ sample tokens    │
└──────────────────┘          └──────────────────┘          └────────┬─────────┘
        ▲                                                            │
        │                      output tensor dict                    │
        └────────────────────(next_token_ids)─────────────────◀──────┘
```

Each arrow is a `send_tensor_dict` / `recv_tensor_dict` call on the PP `GroupCoordinator`. Proxy tensors carry `{"hidden_states": ..., "residual": ...}`, output tensors carry `{"next_token_ids": ..., optional logprob tensors}`.

---

## 12. Known Limitations

| Limitation | Where |
|-----------|-------|
| Some models assert `pp_size == 1` | `sdar.py`, `sdar_moe.py` |
| Speculative decoding workers hardcode `pp_rank=0` | `eagle_worker.py` (FIXME) |
| Context parallelism + PP forbidden | `server_args.py` validation |
| DFLASH + PP forbidden | `server_args.py` validation |
| PD-multiplexing + PP forbidden | `server_args.py` validation |
| Overlap schedule disabled under PP | `_handle_pipeline_parallelism()` |
| HF `transformers.py` wrapper may raise | If HF model lacks PP support |

---

## 13. Key File Index

| Component | Path |
|-----------|------|
| Process groups & P2P | `sglang/srt/distributed/parallel_state.py` |
| PP layer indices | `sglang/srt/distributed/utils.py :: get_pp_indices` |
| Scheduler PP event loops | `sglang/srt/managers/scheduler_pp_mixin.py` |
| Event loop dispatch | `sglang/srt/managers/scheduler.py :: dispatch_event_loop` |
| Worker PP branching | `sglang/srt/managers/tp_worker.py` |
| PPProxyTensors | `sglang/srt/model_executor/forward_batch_info.py` |
| `make_layers` | `sglang/srt/utils/common.py` |
| `PPMissingLayer` | `sglang/srt/layers/utils/common.py` |
| Model example (Llama) | `sglang/srt/models/llama.py` |
| Weight loader | `sglang/srt/model_loader/loader.py` |
| Server args & validation | `sglang/srt/server_args.py` |
| Process spawning | `sglang/srt/managers/data_parallel_controller.py` |
| Distributed init | `sglang/srt/model_executor/model_runner.py :: init_torch_distributed` |
| PP output helpers | `sglang/srt/managers/utils.py :: get_logprob_from_pp_outputs` |
| Tests | `test/registered/distributed/test_pp_single_node.py` |
