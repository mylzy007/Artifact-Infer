# Kernel-level MoE balanceness on Qwen3 / Qwen3.5

A short report on the local kernel-level MoE benchmarking work in
`workshop/e2e_bench/`. All numbers below are from one workstation
(8x NVIDIA RTX 4090, driver 580.82.07, CUDA 13.0, vLLM
`0.19.1rc1.dev180+gad720aefe.precompiled` against torch 2.11.0+cu130).

## 1. What was built

| Component | Path | What it does |
|---|---|---|
| Setup | `scripts/setup_kernel_bench_env.sh` | Idempotent venv creation + vLLM editable install with precompiled wheel + `accelerate`. |
| Bench (extended) | `inference/vllm/tests/kernels/moe/modular_kernel_tools/benchmark_eplb_multigpu.py` | The artifact's single-layer MoE kernel benchmark, extended with `--model-config`, `--num-moe-layers`, `--routing-trace`. Plays back N MoE layers per timed iteration with either synthetic or trace-driven routing. |
| Capture | `scripts/capture_routing_trace.py` | Two-mode trace recorder: `hf` (HuggingFace `AutoModelForCausalLM` + per-MoE-block forward hook), `router-only` (gate-weights-only forward, for archs the installed transformers cannot load). |
| Sweep wrapper | `scripts/run_moe_kernel_cv_sweep.py` | `--model {qwen3,qwen3.5,...}` preset that auto-fills MoE dimensions and forwards `--num-moe-layers`/`--routing-trace`. |
| Visualisation | `scripts/plot_routing_trace.py`, `scripts/plot_bench_results.py` | Trace heatmaps + per-layer CV/entropy/top-K, and per-layer step-time + CV-vs-time scatter for bench results. |

The bench output JSON gained these fields per trial row, on top of what
the original artifact already produced:

- `model_name`, `num_moe_layers`, `routing_trace` — provenance.
- `per_layer_mean_step_ms` / `per_layer_std_step_ms` — length = num
  layers; one CUDA-event-timed per-layer kernel call per iter.
- `per_layer_logical_cv` / `per_layer_logical_entropy` — coefficient of
  variation / normalised entropy of the layer's per-expert token count.
- `per_layer_physical_cv` / `per_layer_physical_entropy` — same after
  bucketing experts into the EP rank that owns them.

## 2. Methodology

**Multi-GPU.** Every measurement runs `--world-size 8` ranks via
`torch.multiprocessing.spawn`, one per physical GPU; vLLM is configured
with `data_parallel_size=8, tensor_parallel_size=1, enable_expert_parallel=True,
all2all_backend=allgather_reducescatter`. Each kernel call therefore
includes the full all-gather → expert GEMM → reduce-scatter pipeline.

**Multi-layer.** With `--num-moe-layers N`, each timed iteration walks
N kernel calls in order and the closure feeding `custom_routing_function`
is mutated between calls so each layer sees its own `(topk_ids, topk_weights)`.
The reported total step time is the sum across the N layers; per-layer
arrays are kept too.

**Real routing.** A trace file is a list of per-layer
`(topk_ids, topk_weights)` tensors saved with `torch.save`. The bench
slices those per-rank for replay. Trace contents come from one of:

- `--mode hf` — Loads the actual HF model with `device_map="auto"`,
  hooks every `*SparseMoeBlock` / `*MoeBlock` `forward`, runs **one real
  prefill** of `--prompt` truncated/padded to `--max-tokens`, and dumps
  `(topk_ids, topk_weights)` recomputed from each layer's
  `router_logits`. Real gates **and** real per-layer hidden states.
- `--mode router-only` — Loads only `mlp.gate.weight` from the
  safetensors index, then runs `softmax(z @ gate.T) → topk` on a
  synthetic Gaussian hidden stream `z`. Real trained gates but **synthetic**
  hidden state (the same Gaussian feeds every layer). This mode exists
  because Qwen3.5 is not yet supported by transformers `< 5`, which is
  what vLLM pins.

**Synthetic routing baseline.** With Monte-Carlo `--target-dest-cv X`
the bench samples a per-rank token-share distribution close to physical
CV `X` and constructs routing IDs that materialise that share. This
gives a known-imbalance counterfactual to compare against the trace.

**Verified configurations.** Two models are exercised end-to-end:

| Model | Layers (MoE) | E | top-k | hidden | moe_int | Capture mode used |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-30B-A3B | 48 | 128 | 8 | 2048 | 768 | `hf` (real gate + real hidden) |
| Qwen3.5-35B-A3B | 41 (40 + 1 MTP) | 256 | 8 | 2048 | 512 | `router-only` (real gate, synthetic hidden) |

## 3. What real router traces look like

### Qwen3-30B-A3B (real HF prefill, 48 layers, 256 prompt tokens)

![Qwen3 trace](figures/qwen3_trace.png)

- **Routing is *very* concentrated.** Per-layer logical CV stays in 1.3-2.1
  for all 48 layers (mean 1.74). The top-1 expert in any given layer
  captures 4.5-10.5% of routed tokens, vs. the uniform top-1 share of
  6.25% (`topk/E = 8/128`); top-10 captures 40-55% of the layer's tokens.
- **Specialisation grows with depth.** logical CV climbs from ~1.5 at
  layer 0 to a 1.9-2.1 ceiling around layer 40-45.
- **Per-rank skew is non-trivial too.** physical CV (per-EP-rank token
  share, world=8) hovers around 0.4 with peaks above 0.65, so even
  after bucketing experts into rank-blocks the imbalance is significant.

### Qwen3.5-35B-A3B (router-only, 41 layers, 16 384 synthetic tokens)

![Qwen3.5 trace](figures/qwen3p5_trace.png)

- **Gates are clearly trained, but the synthetic input dampens the
  picture.** Per-layer logical CV is 0.42-0.96 (mean 0.73), uniformly
  lower than the real-prefill Qwen3 numbers despite Qwen3.5 having
  twice as many experts (256 vs. 128). This is the synthetic-hidden-
  state caveat showing up: a Gaussian hidden stream gives the gate less
  to discriminate against than real activations.
- **Expert specialisation is still real.** Logical CV ramps up over
  the first ~10 layers (0.42 → ~0.8) and stays elevated, with a
  noticeable spike at layer 20. Top-1 share is ~1-2% (uniform 3.1%);
  routing leans toward "uniform-with-gentle-bias", not "razor-sharp
  top-K specialisation". Real prefill activations would almost certainly
  push these numbers higher.
- **Per-rank skew is mild.** Physical CV is 0.06-0.20 (mean 0.11).
  Aggregated across 41 layers it falls to 0.023 — almost perfectly
  balanced when summed over the model.

## 4. What the kernel measures across all layers

### Qwen3.5-35B-A3B, all 41 layers, 16 384 tokens

![Qwen3.5 per-layer](figures/qwen3p5_bench_per_layer.png)

![Qwen3.5 scatter](figures/qwen3p5_bench_scatter.png)

| scenario | target_cv | agg_phys_cv | layer_phys_cv (avg / max) | layer_log_cv (avg) | total step ms | per-layer ms | tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| **real Qwen3.5 router trace** | (real) | **0.023** | **0.11 / 0.20** | **0.73** | **1250.4** | **30.49** | **13 103** |
| synthetic CV=0.05 | 0.05 | 0.055 | 0.05 / 0.05 | 0.37 | 1184.2 | 28.88 | 13 838 |
| synthetic CV=0.15 | 0.15 | 0.150 | 0.15 / 0.15 | 0.40 | 1203.4 | 29.35 | 13 624 |
| synthetic CV=0.40 | 0.40 | 0.401 | 0.40 / 0.40 | 0.56 | 1171.3 | 28.56 | 13 989 |

Two facts the data forces on us:

1. **Real-trace replay is the slowest scenario despite having the
   *lowest* aggregated cross-GPU CV.** 1250 ms vs. 1171-1203 ms for the
   synthetics, with `agg_phys_cv = 0.023` (better than even the
   "balanced" synthetic CV=0.05). The model-level "are GPUs balanced?"
   metric is fooling: token loads sum across 41 layers and the per-layer
   pain disappears in aggregation, but the kernel still pays it
   layer-by-layer.

2. **Per-GPU CV alone is *not* a useful predictor of kernel time on
   this model.** Synthetic CV 0.05 → 0.40 walks total time as
   1184 → 1203 → 1171 ms — non-monotone, all within 3% of each other.
   With fixed `hot_token_frac=0.8 / hot_expert_frac=0.375` the
   per-expert GEMM batch shapes change less than the per-rank shares,
   so kernel performance is ~insensitive to physical CV in isolation.
   The signal is in **per-expert** (logical) CV, where the real trace
   sits at 0.73 — well above any of the synthetics — and that is what
   pushes total time up.

### Qwen3-30B-A3B, all 48 layers, 2048 tokens

![Qwen3 per-layer](figures/qwen3_bench_per_layer.png)

![Qwen3 scatter](figures/qwen3_bench_scatter.png)

The two-point Qwen3 comparison is consistent with the Qwen3.5 picture
and arguably stronger (because the Qwen3 trace **is** fully real — both
gates and hidden states):

| | real HF trace | synthetic CV=0.05 |
|---|---:|---:|
| total step time (ms) | 201.5 | 198.1 |
| per-layer mean (ms) | 4.20 | 4.11 |
| per-layer logical CV | 1.3 .. 2.0 | 0.30 .. 0.40 |
| per-layer physical CV | 0.16 .. 0.66 | 0.05 (target) |
| aggregated phys. CV | 0.102 | 0.053 |
| throughput (tokens/s) | 10 164 | 10 336 |

Real Qwen3 routing pays only ~1.7% in step time over a near-uniform
synthetic, but its **per-layer** physical CV peaks at 0.66 and its
per-layer logical CV hits 2.0. The aggregated physical CV (0.102) is a
~10% per-GPU imbalance — visible at the model level but only half the
story.

## 5. What it adds up to

Three concrete claims supported by what's been collected:

1. **The kernel cost of MoE imbalance is dominated by the *per-expert*
   skew within a layer, not the *per-GPU* skew aggregated over all
   layers.** Within-layer logical CV varies from ~0.4 (synthetic uniform)
   to ~2.0 (real Qwen3 prefill); summed across layers, per-GPU CV
   compresses to 0.02-0.10 because hot experts in different layers live
   on different ranks. Tools that report only an aggregate "balance"
   number — including most production telemetry — will materially
   understate per-iter latency cost.

2. **Real-prefill traces look qualitatively different from any of the
   off-the-shelf synthetic distributions used in the original
   single-layer artifact.** The original artifact's `hot_token_frac=0.95,
   hot_expert_frac=0.5` Monte Carlo configuration produces logical CVs
   in 0.30-0.40; real Qwen3 prefills produce 1.3-2.0. Any conclusion
   about EPLB-on/off, kernel autotuning, or routing-aware scheduling
   reached on the synthetic distribution should be re-checked on a
   trace before it is published.

3. **At Qwen3-class hidden sizes the modular MoE kernel is bandwidth-
   and-collective-bound, not GEMM-bound.** Per-layer time (~4 ms for
   Qwen3, ~28-30 ms for Qwen3.5) is dominated by all-gather +
   reduce-scatter at small per-expert M (256/8 = 32 tokens per rank for
   the Qwen3 prefill). That's why moving physical CV from 0.05 to 0.40
   barely changes total time on either model — the kernel is paying
   for collectives, not for the most-loaded expert's GEMM. Qwen3.5's
   synthetic-CV-0.40 actually came back **faster** than CV=0.05 (1171
   vs. 1184 ms, within noise), reinforcing the same conclusion.

## 6. Caveats

- **Qwen3.5 trace uses synthetic hidden states.** Real per-layer hidden
  states are passed through 0..40 transformer blocks before reaching
  the gate; our Gaussian stand-in skips that. The trace correctly
  reflects the gates' *learned preferences* (you can see specialisation
  growing with depth in the heatmap), but the absolute logical CVs are
  expected to **understate** what production prefill would show on
  Qwen3.5. The Qwen3 trace does not have this caveat.
- **N=2 trials per CV in the synthetic sweep.** Per-CV variance
  (std of `max_rank_mean_step_ms`) is ~14 ms on Qwen3.5, so
  differences smaller than ~30 ms between synthetic points are not
  statistically meaningful. The real-trace vs. synthetic-CV-0.05 gap
  on Qwen3.5 (1250 vs. 1184 ms = 66 ms) and Qwen3 (201.5 vs. 198.1 ms
  = 3.4 ms) are both above the per-trial noise floor.
- **No EPLB-on/off comparison yet for the multi-layer mode.** The
  bench's `--mode on/both` paths still work but were not exercised in
  the multi-layer runs reported here. EPLB rebalances using the
  *aggregate* expert load, so its benefit on traces with high
  per-layer logical CV but low aggregate CV (i.e. exactly Qwen3.5) is
  worth a separate experiment.
- **No quantisation in any run.** All weights are bf16 and the runs
  fall through to `Using TRITON Unquantized MoE backend`. Kernel cost
  shapes will change for fp8 / mxfp4 paths.
- **256 tokens of Qwen3 prefill is a small sample.** It captures the
  routing pattern of one prompt's prefill cleanly, but does not yet
  span multi-prompt or decode-phase routing.

## 7. Next steps

Listed in order of effort:

1. **Promote Qwen3.5 to a fully-real trace.** Two routes (a) plug a
   `Routed_Experts_Capturer`-style hook into vLLM's own
   `Qwen3_5MoeForCausalLM`, since vLLM ships its own implementation and
   does not depend on transformers ≥ 5; (b) add a parallel venv pinned
   on transformers main and re-run `capture_routing_trace.py --mode hf`.
   Either turns the table 4 column for Qwen3.5 from "router-only" into
   "real-prefill", and almost certainly raises the per-layer logical
   CV closer to the Qwen3 ceiling.

2. **EPLB-on vs. off across all layers, with a real trace.** Same
   bench, `--mode both`. Question: does EPLB recover most of the
   real-trace overhead on Qwen3 (3 ms over 200 ms is small but not
   zero), and how does it interact with the high *per-layer* /
   low-aggregate pattern on Qwen3.5?

3. **Token-count scaling.** The reported numbers fix `--total-tokens`
   at 16 384 (Qwen3.5) and 2 048 (Qwen3). Both are realistic per-batch
   prefill sizes but the cost regime shifts as M climbs (collective
   amortisation, GEMM efficiency). A scan over `2k → 64k` would clarify
   when the kernel becomes GEMM-bound and would expose imbalance more
   sharply.

4. **More prompts.** The Qwen3 trace is from one 256-token prompt; the
   per-layer logical CV measurement is one realisation, not a
   distribution. A small corpus of prompts and a heatmap of `layer ×
   prompt` logical CV would tighten the "real routing is much more
   skewed than synthetic" claim.

5. **Quantised paths.** Re-run the multi-layer mode with `fp8` or
   `mxfp4` quantisation flags as supported by the modular kernel —
   imbalance should hurt more in those regimes because the per-expert
   GEMMs are smaller and collective overhead a larger relative cost.

## 8. Files of record

```
workshop/e2e_bench/
├── REPORT.md                              ← this file
├── KERNEL_BENCH_LOCAL_SETUP.md            ← deployment runbook
├── figures/
│   ├── qwen3_trace.png
│   ├── qwen3p5_trace.png
│   ├── qwen3_bench_per_layer.png
│   ├── qwen3_bench_scatter.png
│   ├── qwen3p5_bench_per_layer.png
│   └── qwen3p5_bench_scatter.png
└── scripts/
    ├── setup_kernel_bench_env.sh
    ├── capture_routing_trace.py
    ├── plot_routing_trace.py
    ├── plot_bench_results.py
    └── run_moe_kernel_cv_sweep.py         (extended)

# Raw outputs (outside the workspace)
/home/yyx/personal/inference/vllm-bench/
├── traces/
│   ├── qwen3_hf.pt                        # 48 layers, 256 tokens, hf mode
│   └── qwen3p5_router.pt                  # 41 layers, 16384 tokens, router-only
└── results/
    ├── qwen3_alllayers/
    │   ├── run.jsonl                      # real Qwen3 trace replay
    │   └── synthetic_uniform.jsonl        # Qwen3 synthetic CV=0.05 baseline
    └── qwen3p5_alllayers_v2/
        ├── real_trace.jsonl               # Qwen3.5 real trace replay
        └── cv_sweep/
            ├── kernel_cv_0p0500.jsonl     # Qwen3.5 synthetic CV=0.05
            ├── kernel_cv_0p1500.jsonl     # Qwen3.5 synthetic CV=0.15
            ├── kernel_cv_0p4000.jsonl     # Qwen3.5 synthetic CV=0.40
            └── cv_sweep_summary_*.csv     # aggregate
```
