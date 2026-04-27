# Local kernel-level MoE balanceness test

Notes for running the kernel-level MoE balanceness benchmark
(`tests/kernels/moe/modular_kernel_tools/benchmark_eplb_multigpu.py`) on this
machine. Verified end-to-end on **8x RTX 4090 / driver 580.82.07 / CUDA 13.0**.

## TL;DR

```bash
# from this workspace root (/home/yyx/personal/Artifact-Infer)
bash workshop/e2e_bench/scripts/setup_kernel_bench_env.sh
source /home/yyx/personal/inference/vllm-bench/.venv/bin/activate

# small smoke test (8 GPUs, ~90 s)
cd /home/yyx/personal/inference/vllm
python -m tests.kernels.moe.modular_kernel_tools.benchmark_eplb_multigpu \
    --world-size 8 \
    --routing-space physical_gpu \
    --backend allgather_reducescatter \
    --num-experts 64 --topk 6 \
    -k 2048 -n 1408 \
    --total-tokens 8192 \
    --target-dest-cv 0.10 \
    --num-random-trials 1 --sample-attempts 200 \
    --warmup-iters 3 --iters 5 \
    --mode off
```

A successful run prints one JSON line per trial with fields like
`realized_physical_cv`, `global_tokens_per_s`, `max_rank_mean_step_ms`, etc.

## What the benchmark does

For one synthetic MoE layer the benchmark:

1. Spawns `--world-size` processes (one per GPU) and forms a vLLM EP+DP group.
2. Constructs router IDs that produce a target physical-GPU token distribution
  (parameterized by `--target-dest-cv` / `--target-dest-std` / `--target-dest-entropy`
   when sampled via Monte Carlo, or directly via `--physical-rank-probs`).
3. Builds a real `FusedMoE` layer with bf16 weights and runs it under
  `set_forward_context`, with the same custom routing on every step.
4. Measures latency / throughput, plus the realized CV / entropy of the
  per-rank token load. Optionally turns EPLB on for an `eplb_on` case.

The wrapper `scripts/run_moe_kernel_cv_sweep.py` invokes this for a sweep of
target CV values and writes a tidy CSV/JSONL summary.

## Why the setup is non-trivial

The benchmark imports recent vLLM internals that are only present on
post-`v0.18` `main`:

- `vllm.distributed.eplb.eplb_communicator.create_eplb_communicator`
- `vllm.distributed.eplb.eplb_state.{EplbState, compute_logical_maps}`
- `vllm.distributed.eplb.rebalance_execute.rearrange_expert_weights_inplace`
- `vllm.distributed.parallel_state.get_eplb_group`
- `vllm.v1.worker.workspace.{init_workspace_manager, is_workspace_manager_initialized}`
- `vllm.model_executor.layers.fused_moe.layer.FusedMoE` (with `set_eplb_state`,
`maybe_init_modular_kernel`, `get_expert_weights`)

These already exist in the local checkout at `/home/yyx/personal/inference/vllm`
(commit `ad720aefe`, ahead of v0.18). However the corresponding C++/CUDA
kernels need to match the Python layer. We avoid a multi-hour `pip install -e .`
build by using vLLM's official "precompiled" path:

```bash
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=nightly \
  uv pip install --editable /home/yyx/personal/inference/vllm \
                  --torch-backend=auto --index-strategy unsafe-best-match
```

`uv` resolves PyTorch automatically (here: `torch 2.11.0+cu130` to match the
host driver), then installs the latest pre-built `vllm` C++ extensions while
keeping the local Python source on `sys.path`. Total install time is roughly
4 minutes on this box.

## Files added

- `scripts/setup_kernel_bench_env.sh` — idempotent setup script that
creates the venv, installs vLLM, and extracts the benchmark file from
`e2e_artifact.diff`.
- `scripts/capture_routing_trace.py` — HuggingFace-based capture tool that
hooks each MoE block during a real prefill and writes per-layer
`(topk_ids, topk_weights)` into a torch trace. Supports `qwen3_moe`
(`--mode hf`) and `qwen3_5_moe` (`--mode router-only`, since transformers
4.57 cannot yet load the full Qwen3.5 architecture).
- (Created at runtime, not in this workspace)
`/home/yyx/personal/inference/vllm-bench/.venv/` — Python 3.12 venv with
`vllm 0.19.1rc1.dev180+gad720aefe.precompiled`, `torch 2.11.0+cu130`,
`triton 3.6.0`, `accelerate 1.13.0`, etc.
- (Created at runtime, in the vLLM source tree)
`/home/yyx/personal/inference/vllm/tests/kernels/moe/modular_kernel_tools/benchmark_eplb_multigpu.py`
— the benchmark itself, extracted from `e2e_artifact.diff` and extended
in-place with `--model-config`, `--num-moe-layers`, and
`--routing-trace`. The wrapper scripts under `workshop/e2e_bench/scripts/`
invoke this module via
`python -m tests.kernels.moe.modular_kernel_tools.benchmark_eplb_multigpu`,
so it must live inside the vLLM source tree (which is what the editable
install also points to).

## Sanity-check results

A 2-point CV sweep on this box produced the expected balanceness signal:


| target_cv | realized_cv | entropy | tokens/s | max step (ms) |
| --------- | ----------- | ------- | -------- | ------------- |
| 0.10      | 0.1003      | 0.998   | 562634.6 | 29.12         |
| 0.30      | 0.2987      | 0.977   | 526838.6 | 31.14         |


Latency rises and throughput drops when load is more skewed across GPUs, which
is the expected effect of imbalance on the all-gather/reduce-scatter
modular MoE kernel.

## Whole-model multi-layer mode

The benchmark can now play back **all MoE layers of a real model** in
sequence per timed iteration, either with synthetic routing skew or with
**real per-layer routing decisions** captured from a HuggingFace prefill.

### New benchmark flags


| Flag                              | Effect                                                                                                                                                                                                                                      |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model-config <hf-config.json>` | Reads `hidden_size`, `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `num_hidden_layers` (+ `mtp_num_hidden_layers` for Qwen3.5) and pre-fills the equivalent CLI flags whenever those flags are still at their CLI default. |
| `--num-moe-layers N`              | Stack N MoE layer kernel calls per timed iteration. Each layer uses its own routing decisions (per-layer fresh seed in synthetic mode, per-layer trace slice in trace mode).                                                                |
| `--routing-trace <trace.pt>`      | Replay per-layer `(topk_ids, topk_weights)` from a `capture_routing_trace.py` output. Replaces synthetic Monte Carlo routing entirely.                                                                                                      |


The output JSON gains:

- `model_name`, `num_moe_layers`, `routing_trace` provenance fields.
- `per_layer_mean_step_ms` / `per_layer_std_step_ms` (length = num layers).
- `per_layer_logical_cv` / `per_layer_logical_entropy` — coefficient of
variation of the global expert-load histogram for each individual layer.
- `per_layer_physical_cv` / `per_layer_physical_entropy` — same, after
bucketing experts to the EP rank that owns them.

### Wrapper presets

`run_moe_kernel_cv_sweep.py` learned a `--model {custom,deepseek-v2-lite,qwen3,qwen3.5}`
preset. Each preset auto-fills `--num-experts/--topk/--hidden-size/`
`--intermediate-size/--num-moe-layers/--model-config` from the local
`/home/yyx/models/<name>/config.json`.

### Capturing real routing traces

Two modes are supported in `capture_routing_trace.py`:

- `**--mode hf`** — Loads the full HuggingFace model with `device_map="auto"`,
hooks every `*SparseMoeBlock`/`*MoeBlock`'s `forward`, runs one prefill
on `--prompt` truncated/padded to `--max-tokens`, then dumps the per-layer
`(topk_ids, topk_weights)` recomputed from the captured `router_logits`.
Tested on `Qwen3-30B-A3B` (`qwen3_moe`) with transformers 4.57.6.
- `**--mode router-only`** — Loads only `mlp.gate.weight` shards from the
safetensors index (ignoring the experts entirely), then runs a tiny
`softmax(hidden @ gate.T) -> topk` on a synthetic Gaussian hidden stream.
Useful when the installed transformers does not yet recognise the
architecture (e.g. Qwen3.5 with transformers 4.57.x). The trained expert
preferences are real; only the per-layer hidden_states are stylised.

Trace file schema (created by either mode, consumed by the benchmark via
`--routing-trace`):

```python
torch.save({
    "metadata": {
        "model_name": str,
        "num_layers": int,
        "num_experts": int,
        "topk": int,
        "hidden_size": int,
        "moe_intermediate_size": int,
        "global_num_tokens": int,
        # plus mode-specific fields like prompt_excerpt / synthetic_tokens
    },
    "layers": [
        {"layer_idx": int,
         "topk_ids":     LongTensor[global_num_tokens, topk],
         "topk_weights": FloatTensor[global_num_tokens, topk]},
        ...
    ],
}, path)
```

### Demo: Qwen3.5-35B-A3B (router-only, all 41 layers)

```bash
source /home/yyx/personal/inference/vllm-bench/.venv/bin/activate

# 1) capture (~20 s, no full model load needed)
python workshop/e2e_bench/scripts/capture_routing_trace.py \
    --model /home/yyx/models/Qwen3.5-35B-A3B \
    --mode router-only --num-tokens 16384 \
    --output /home/yyx/personal/inference/vllm-bench/traces/qwen3p5_router.pt

# 2) play back through the kernel bench across all 41 MoE layers
cd /home/yyx/personal/inference/vllm
python -m tests.kernels.moe.modular_kernel_tools.benchmark_eplb_multigpu \
    --world-size 8 --routing-space physical_gpu \
    --backend allgather_reducescatter \
    --model-config /home/yyx/models/Qwen3.5-35B-A3B/config.json \
    --routing-trace /home/yyx/personal/inference/vllm-bench/traces/qwen3p5_router.pt \
    --total-tokens 16384 \
    --warmup-iters 1 --iters 3 --mode off
```

Result on this box (8x4090, 16384 tokens / 2048 tokens-per-rank):


| metric                 | value                                         |
| ---------------------- | --------------------------------------------- |
| `num_moe_layers`       | 41                                            |
| total step time        | 1147 ms (28.0 ms / layer avg)                 |
| per-layer logical CV   | 0.42 .. 0.96 (mean ~0.74)                     |
| per-layer physical CV  | 0.06 .. 0.20 (mean ~0.11)                     |
| aggregated physical CV | 0.023 (very balanced after summing 41 layers) |
| throughput             | 14 280 tokens/s                               |


### Demo: Qwen3-30B-A3B (real HF capture, all 48 layers)

```bash
# 1) capture from a real prefill (~62 s including model load + 256-token prefill)
python workshop/e2e_bench/scripts/capture_routing_trace.py \
    --model /home/yyx/models/Qwen3-30B-A3B \
    --mode hf \
    --prompt "Write a long technical essay about how mixture-of-experts ..." \
    --max-tokens 256 --dtype bf16 --device-map auto \
    --output /home/yyx/personal/inference/vllm-bench/traces/qwen3_hf.pt

# 2) play back across all 48 MoE layers
cd /home/yyx/personal/inference/vllm
python -m tests.kernels.moe.modular_kernel_tools.benchmark_eplb_multigpu \
    --world-size 8 --routing-space physical_gpu \
    --backend allgather_reducescatter \
    --model-config /home/yyx/models/Qwen3-30B-A3B/config.json \
    --routing-trace /home/yyx/personal/inference/vllm-bench/traces/qwen3_hf.pt \
    --total-tokens 2048 \
    --warmup-iters 1 --iters 3 --mode off
```

Comparison on this box (8x4090, 256 tokens / rank, 2048 total):


| metric                | real HF trace | synthetic CV=0.05 |
| --------------------- | ------------- | ----------------- |
| total step time (ms)  | 201.5         | 198.1             |
| per-layer mean (ms)   | 4.20          | 4.11              |
| per-layer logical CV  | 1.3 .. 2.0    | 0.30 .. 0.40      |
| per-layer physical CV | 0.16 .. 0.66  | 0.05 (target)     |
| aggregated phys. CV   | 0.102         | 0.053             |
| throughput (tokens/s) | 10 164        | 10 336            |


Take-aways:

- Per-layer physical CV is **substantially higher** than aggregated physical
CV (0.16-0.66 per layer vs. 0.10 aggregated). Token loads sum across
layers, hiding most of the pain at the model level even though each
layer's kernel still pays for its own skew.
- Per-layer logical CV (over 128 Qwen3 experts / 256 Qwen3.5 experts) is
high in real prefills: the most-loaded expert in a layer easily sees
5-10x the tokens of the least-loaded one. This is invisible if you only
look at the steady-state aggregate.
- The aggregated 0.10 vs. 0.05 difference still shows up as ~1.7% higher
total kernel time (201.5 vs. 198.1 ms), giving a useful signal for
whether expert-parallel load balancing investments would actually help
this model.

## Running larger sweeps

Use `scripts/run_moe_kernel_cv_sweep.py` from this workspace, but pass
`--repo-root /home/yyx/personal/inference/vllm` so it dispatches to the
benchmark module from the vLLM source tree:

```bash
source /home/yyx/personal/inference/vllm-bench/.venv/bin/activate

python workshop/e2e_bench/scripts/run_moe_kernel_cv_sweep.py \
    --output-dir /home/yyx/personal/inference/vllm-bench/results/cv_sweep_full \
    --cv-values 0.05,0.10,0.15,0.20,0.30,0.40,0.70,1.00,1.50 \
    --world-size 8 \
    --num-experts 64 --topk 6 \
    --hidden-size 2048 --intermediate-size 1408 \
    --total-tokens 131072 \
    --num-random-trials 60 --sample-attempts 10000 \
    --warmup-iters 15 --iters 50 \
    --repo-root /home/yyx/personal/inference/vllm
```

The wrapper accepts a `--model {qwen3,qwen3.5}` preset to do the same
sweep on the actual MoE dimensions of those models, and forwards
`--num-moe-layers` / `--routing-trace` to each benchmark invocation:

```bash
python workshop/e2e_bench/scripts/run_moe_kernel_cv_sweep.py \
    --output-dir /home/yyx/personal/inference/vllm-bench/results/qwen3p5_cv_sweep \
    --model qwen3.5 \
    --routing-trace /home/yyx/personal/inference/vllm-bench/traces/qwen3p5_router.pt \
    --cv-values 0.05,0.10,0.20,0.40 \
    --total-tokens 16384 --num-random-trials 4 --sample-attempts 1000 \
    --warmup-iters 1 --iters 3 \
    --repo-root /home/yyx/personal/inference/vllm
```

When a `--routing-trace` is passed, `--cv-values` is effectively ignored
on the routing side (the bench replays trace routing instead of sampling
CVs), but the wrapper still emits one row per CV value so you can
diff-check against synthetic baselines.

Each CV point with the default 60 Monte Carlo trials takes a few minutes on
8x4090. The defaults above match what the e2e bench README suggests for
DeepSeek-V2-Lite shapes (64 experts, topk=6, hidden=2048, moe-intermediate=1408).

## Troubleshooting

- `**No module named 'vllm._C'**`: the wheel didn't install. Re-run the setup
script and check `uv pip list | grep vllm` shows the `precompiled`
variant. If `VLLM_USE_PRECOMPILED=1` was missing, uv tried to build from
source (slow).
- `**Used [...] CV but realized CV != target**`: bump `--sample-attempts`
(Monte Carlo rejection sampling tightens with more attempts). The smoke
test uses 200; production sweeps use 10000.
- `**No tests/kernels/moe/modular_kernel_tools/benchmark_eplb_multigpu**`:
the benchmark file is added by `setup_kernel_bench_env.sh`. Re-run that
script; it's idempotent and only writes if the file is missing.
- **GPU OOM at `--total-tokens 131072`**: reduce to `65536` or `32768`. With
8x 24 GiB cards, full DeepSeek-V2-Lite shapes at 131072 tokens/world-size
fit, but anything quantized + larger hidden sizes may not.
- `**Captured 0 layers; hooks were not triggered**` in the HF capture: the
installed transformers version probably does not understand the model
(e.g. `qwen3_5_moe`). Use `--mode router-only` instead, which only
needs the `mlp.gate.weight` shards and works without a transformers
modeling file.
- **HF capture OOMs at large `--max-tokens`**: lower it (try 256, 128, 64).
Real router patterns are similar across token counts, and the kernel
benchmark wraps the trace tokens around if it needs more per-rank
tokens than the trace contains.
- `**Trace topk=X does not match bench topk=Y**`: the `--model-config` you
pass to the benchmark must match the model from which the trace was
captured. Use the same Qwen3 / Qwen3.5 preset on both sides.

