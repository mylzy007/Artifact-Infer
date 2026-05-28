# MoE Placement Experiment Report

## 1. Environment

- Model: `/home/lzy/models/Qwen3-30B-A3B`
- Eval entry: `python -m eval.test_bazaar_moe`
- Profile switch: `--moe-profile-routing 1` or `MOE_PROFILE_ROUTING=1`
- Default baseline placement remains `contiguous`.

Runtime environment fields to fill per run: GPU model, CUDA, torch, triton,
flashinfer, peak memory, and per-rank memory snapshots.

## 2. Experiment Matrix

Minimum smoke matrix:

| impl | placement | world_size | tp_size | layers | num_prompts | max_tokens | max_model_len |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ep_ll_triton | contiguous | 2 | 1 | 1 | 1 | 8 | 512 |
| ep_ll_triton | round_robin | 2 | 1 | 1 | 1 | 8 | 512 |
| ep_ll_triton | fixed_random_shuffle | 2 | 1 | 1 | 1 | 8 | 512 |
| ep_ll_triton | load_balanced_greedy_with_locality_tiebreak | 2 | 1 | 1 | 1 | 8 | 512 |
| ep_ll_triton | communication_aware_greedy | 2 | 1 | 1 | 1 | 8 | 512 |
| ep_ht | same placements | 2 | 1 | 1 | 1 | 8 | 512 |

Optional: repeat with `--world-size 4 --tp-size 2`.

## 3. Placement Policies

- `contiguous`: built in, no profile required.
- `round_robin`: built in, no profile required.
- `fixed_random_shuffle`: built in for eval with `--moe-expert-placement-seed`; can also be exported as JSON.
- `load_balanced_greedy_with_locality_tiebreak`: generated from profile JSON and passed back with `--moe-expert-placement-path`.
- `communication_aware_greedy`: generated from profile JSON and passed back with `--moe-expert-placement-path`.

## 4. Profiling Method

Routing profile records `traffic[layer][src_ep_rank][expert_id] += count(topk_ids == expert_id)`
inside EP-LL and EP-HT dispatch after `topk_ids` is computed. Profiling is off by
default and writes:

`eval_results/moe_routing_profile_<run_id>.json`

Rank-local files are also preserved as:

`eval_results/moe_routing_profile_<run_id>_rank<R>.json`

## 5. Metrics Definition

- `expert_cv`: CV over per-expert routed replicas.
- `rank_load_cv`: CV over estimated per-rank expert load after placement.
- `cross_traffic_ratio`: cross-rank routed replicas divided by total routed replicas.
- `prefill_throughput_tok_s`: real prefill tokens divided by measured prefill step time.
- `decode_throughput_tok_s`: real decode tokens divided by measured decode step time.
- `e2e_total_time_s`: wall time for `engine.generate`.

Current first version records true prefill/decode step throughput and estimated
traffic/load metrics. Dispatch/compute/combine timing and GPU CV are still P1
follow-ups.

## 6. Commands

Profile contiguous:

```bash
MOE_PROFILE_ROUTING=1 python -m eval.test_bazaar_moe \
  --model-path /home/lzy/models/Qwen3-30B-A3B \
  --world-size 2 --tp-size 1 \
  --moe-impl ep_ll_triton \
  --moe-expert-placement contiguous \
  --num-layers 1 --num-problems 1 --max-tokens 8 --max-model-len 512
```

Generate placements:

```bash
python -m eval.generate_moe_placement \
  --profile eval_results/moe_routing_profile_<run_id>.json \
  --policy fixed_random_shuffle --seed 1234

python -m eval.generate_moe_placement \
  --profile eval_results/moe_routing_profile_<run_id>.json \
  --policy load_balanced_greedy_with_locality_tiebreak

python -m eval.generate_moe_placement \
  --profile eval_results/moe_routing_profile_<run_id>.json \
  --policy communication_aware_greedy
```

Eval generated placement:

```bash
python -m eval.test_bazaar_moe \
  --model-path /home/lzy/models/Qwen3-30B-A3B \
  --world-size 2 --tp-size 1 \
  --moe-impl ep_ll_triton \
  --moe-expert-placement load_balanced_greedy_with_locality_tiebreak \
  --moe-expert-placement-path eval_results/moe_placement_<policy>_<timestamp>.json \
  --num-layers 1 --num-problems 1 --max-tokens 8 --max-model-len 512
```

## 7. Results Table

| impl | placement | world_size | tp_size | ep_size | layers | num_prompts | max_tokens | pass/fail | e2e_time_s | prefill_tok_s | decode_tok_s | dispatch_ms/token | compute_ms/token | combine_ms/token | gpu_cv | expert_cv | rank_load_cv | cross_traffic_ratio | drop_ratio | peak_mem_gib | output_path | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | P1 | P1 | P1 | P1 | TBD | TBD | TBD | P2 | TBD | TBD | TBD |

## 8. Runtime Breakdown

P0 records e2e, prefill, and decode timing from actual scheduler steps. P1 will
add dispatch, expert compute, combine, and per-rank straggler timing.

## 9. Throughput

Throughput is not derived from e2e. It is recorded from real prefill/decode step
timers in eval output JSON under `summary.generation_metrics`.

## 10. Load Balance

Placement JSON contains `estimated_metrics` with expert CV, GPU/rank CV, and
per-rank estimated compute load. Profile JSON contains per-layer expert load.

## 11. Communication

Placement JSON estimates local traffic, cross traffic, cross traffic ratio,
per-source outgoing replicas, and per-destination incoming replicas.

## 12. EP-LL Capacity / Drop

P2: record `M_max`, local bucket count max/p95/mean, overflow count, and drop ratio.
Current overflow behavior is still controlled by `--moe-ll-overflow-policy`.

## 13. Findings

Pending real runs.

## 14. Bugs / Risks

- Profile aggregation currently reports combined prefill/decode routing, not split.
- Layerwise placement JSON loader is supported, but the generator currently emits global placement.
- Dispatch/compute/combine timing and GPU CV are not implemented yet.
- Generated profiled policies require `--moe-expert-placement-path`; the policy name alone is not enough to reconstruct profile-derived placement.

## 15. Next Steps

1. Run minimum matrix and populate the results table.
2. Add low-overhead MoE module timing for dispatch/experts/combine.
3. Add EP-LL capacity/drop counters.
4. Add layerwise placement generation after global placement is validated.


## Run 20260516_015440

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_20260516_015440/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_20260516_015440/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3', '4,5,6,7']


## Run drycheck

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_drycheck/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_drycheck/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3', '4,5,6,7']


## Run dry_full_default

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_full_default/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_full_default/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3', '4,5,6,7']


## Run dry_cuda_home

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_cuda_home/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_cuda_home/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3', '4,5,6,7']


## Run full_ep4_fiws_20260516_0230

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_full_ep4_fiws_20260516_0230/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_full_ep4_fiws_20260516_0230/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3', '4,5,6,7']


## Run dry_ws_sweep_check_ws8

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_ws_sweep_check_ws8/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_ws_sweep_check_ws8/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3,4,5,6,7']


## Run dry_ws_sweep_check_ws4

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_ws_sweep_check_ws4/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_ws_sweep_check_ws4/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3', '4,5,6,7']


## Run dry_scale_check_p1_t8_ws8

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_scale_check_p1_t8_ws8/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_scale_check_p1_t8_ws8/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3,4,5,6,7']


## Run dry_scale_check_p1_t8_ws4

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_scale_check_p1_t8_ws4/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_scale_check_p1_t8_ws4/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3', '4,5,6,7']


## Run dry_scale_check_p4_t32_ws8

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_scale_check_p4_t32_ws8/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_scale_check_p4_t32_ws8/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3,4,5,6,7']


## Run dry_scale_check_p4_t32_ws4

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_scale_check_p4_t32_ws4/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_scale_check_p4_t32_ws4/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3', '4,5,6,7']


## Run dry_scale_check_p8_t64_ws8

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_scale_check_p8_t64_ws8/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_scale_check_p8_t64_ws8/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3,4,5,6,7']


## Run dry_scale_check_p8_t64_ws4

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_scale_check_p8_t64_ws4/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_scale_check_p8_t64_ws4/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3', '4,5,6,7']


## Run perf_ws8_ws4_20260516_0228_p1_t32_ws8

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_perf_ws8_ws4_20260516_0228_p1_t32_ws8/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_perf_ws8_ws4_20260516_0228_p1_t32_ws8/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3,4,5,6,7']


## Run perf_ws8_ws4_20260516_0228_p1_t32_ws4

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_perf_ws8_ws4_20260516_0228_p1_t32_ws4/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_perf_ws8_ws4_20260516_0228_p1_t32_ws4/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3', '4,5,6,7']


## Run dry_port_check

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_port_check/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_port_check/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3', '4,5,6,7']


## Run retry_p1_t32_ws4_epht_round_robin_20260516_1412

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_retry_p1_t32_ws4_epht_round_robin_20260516_1412/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_retry_p1_t32_ws4_epht_round_robin_20260516_1412/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3']


## Run dry_full_rerun_port_check_p1_t32_ws8

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_full_rerun_port_check_p1_t32_ws8/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_full_rerun_port_check_p1_t32_ws8/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3,4,5,6,7']


## Run dry_full_rerun_port_check_p1_t32_ws4

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_full_rerun_port_check_p1_t32_ws4/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_full_rerun_port_check_p1_t32_ws4/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3', '4,5,6,7']


## Run dry_full_rerun_port_check_p4_t64_ws8

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_full_rerun_port_check_p4_t64_ws8/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_full_rerun_port_check_p4_t64_ws8/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3,4,5,6,7']


## Run dry_full_rerun_port_check_p4_t64_ws4

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_full_rerun_port_check_p4_t64_ws4/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_full_rerun_port_check_p4_t64_ws4/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3', '4,5,6,7']


## Run dry_full_rerun_port_check_p8_t64_ws8

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_full_rerun_port_check_p8_t64_ws8/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_full_rerun_port_check_p8_t64_ws8/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3,4,5,6,7']


## Run dry_full_rerun_port_check_p8_t64_ws4

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_full_rerun_port_check_p8_t64_ws4/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_dry_full_rerun_port_check_p8_t64_ws4/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3', '4,5,6,7']


## Run perf_ws8_ws4_rerun_20260516_1428_p1_t32_ws8

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_perf_ws8_ws4_rerun_20260516_1428_p1_t32_ws8/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_perf_ws8_ws4_rerun_20260516_1428_p1_t32_ws8/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3,4,5,6,7']


## Run perf_ws8_ws4_rerun_20260516_1428_p1_t32_ws4

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_perf_ws8_ws4_rerun_20260516_1428_p1_t32_ws4/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_perf_ws8_ws4_rerun_20260516_1428_p1_t32_ws4/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3', '4,5,6,7']


## Run perf_ws8_ws4_rerun_20260516_1516_p1_t32_ws8

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_perf_ws8_ws4_rerun_20260516_1516_p1_t32_ws8/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_perf_ws8_ws4_rerun_20260516_1516_p1_t32_ws8/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3,4,5,6,7']


## Run perf_ws8_ws4_rerun_20260516_1516_p1_t32_ws4

- Summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_perf_ws8_ws4_rerun_20260516_1516_p1_t32_ws4/summary.json`
- Per-run report: `/home/lzy/Artifact-Infer/eval_results/moe_placement_run_perf_ws8_ws4_rerun_20260516_1516_p1_t32_ws4/moe_placement_experiment_report.md`
- Dataset: `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet`
- GPU groups: ['0,1,2,3', '4,5,6,7']
