# MoE Placement Result Analysis: perf_ws8_ws4_20260516_0228

## Scope

- Model: `/home/lzy/models/Qwen3-30B-A3B`
- Dataset: `datasets/gsm8k_moe_smoke.parquet`
- Completed scale: `num_problems=1`, `max_tokens=32`, full model layers
- World sizes: `8`, then `4`
- Implementations: `ep_ll_triton`, `ep_ht`
- Placements: `contiguous`, `round_robin`, `fixed_random_shuffle`, `load_balanced_greedy_with_locality_tiebreak`, `communication_aware_greedy`
- Sweep stopped before larger scales because `ws=4 / ep_ht / round_robin` failed with `EADDRINUSE` on master port `29701`.

## Stability

| scale | world_size | status | note |
| --- | ---: | --- | --- |
| 1x32 | 8 | pass | profile, placement generation, and 10/10 evals passed |
| 1x32 | 4 | partial | profile and placement generation passed; 9/10 evals passed |

Failure summary:

```text
ep_ht + round_robin + ws=4 failed before model execution:
DistNetworkError: address already in use, port: 29701
```

This is a runner/port reuse issue, not evidence that `round_robin` is slower or incorrect for EP-HT.

## Runtime Results

| ws | impl | placement | pass | e2e_s | prefill_tok_s | decode_tok_s | generated |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 8 | ep_ll_triton | contiguous | pass | 116.676 | 7.018 | 0.292 | 32 |
| 8 | ep_ll_triton | round_robin | pass | 115.228 | 8.042 | 0.292 | 32 |
| 8 | ep_ll_triton | fixed_random_shuffle | pass | 115.640 | 7.687 | 0.292 | 32 |
| 8 | ep_ll_triton | load_balanced_greedy_with_locality_tiebreak | pass | 113.969 | 7.719 | 0.297 | 32 |
| 8 | ep_ll_triton | communication_aware_greedy | pass | 118.520 | 7.176 | 0.287 | 32 |
| 8 | ep_ht | contiguous | pass | 13.700 | 11.879 | 4.155 | 32 |
| 8 | ep_ht | round_robin | pass | 13.205 | 12.314 | 4.316 | 32 |
| 8 | ep_ht | fixed_random_shuffle | pass | 11.897 | 15.455 | 4.371 | 32 |
| 8 | ep_ht | load_balanced_greedy_with_locality_tiebreak | pass | 13.464 | 11.466 | 4.431 | 32 |
| 8 | ep_ht | communication_aware_greedy | pass | 13.448 | 11.929 | 4.288 | 32 |
| 4 | ep_ll_triton | contiguous | pass | 94.317 | 11.172 | 0.354 | 32 |
| 4 | ep_ll_triton | round_robin | pass | 89.942 | 9.401 | 0.378 | 32 |
| 4 | ep_ll_triton | fixed_random_shuffle | pass | 94.220 | 9.786 | 0.358 | 32 |
| 4 | ep_ll_triton | load_balanced_greedy_with_locality_tiebreak | pass | 89.777 | 9.360 | 0.379 | 32 |
| 4 | ep_ll_triton | communication_aware_greedy | pass | 93.695 | 11.464 | 0.355 | 32 |
| 4 | ep_ht | contiguous | pass | 13.254 | 13.453 | 4.006 | 32 |
| 4 | ep_ht | round_robin | fail | - | - | - | - |
| 4 | ep_ht | fixed_random_shuffle | pass | 10.797 | 16.380 | 4.946 | 32 |
| 4 | ep_ht | load_balanced_greedy_with_locality_tiebreak | pass | 10.787 | 17.921 | 4.664 | 32 |
| 4 | ep_ht | communication_aware_greedy | pass | 23.345 | 11.473 | 1.839 | 32 |

## Delta vs Contiguous

Lower e2e delta is better. Higher prefill/decode delta is better.

| ws | impl | placement | e2e_delta | prefill_delta | decode_delta |
| ---: | --- | --- | ---: | ---: | ---: |
| 8 | ep_ll_triton | round_robin | -1.2% | +14.6% | +0.1% |
| 8 | ep_ll_triton | fixed_random_shuffle | -0.9% | +9.5% | +0.1% |
| 8 | ep_ll_triton | load_balanced_greedy_with_locality_tiebreak | -2.3% | +10.0% | +1.7% |
| 8 | ep_ll_triton | communication_aware_greedy | +1.6% | +2.3% | -1.9% |
| 8 | ep_ht | round_robin | -3.6% | +3.7% | +3.9% |
| 8 | ep_ht | fixed_random_shuffle | -13.2% | +30.1% | +5.2% |
| 8 | ep_ht | load_balanced_greedy_with_locality_tiebreak | -1.7% | -3.5% | +6.6% |
| 8 | ep_ht | communication_aware_greedy | -1.8% | +0.4% | +3.2% |
| 4 | ep_ll_triton | round_robin | -4.6% | -15.8% | +6.9% |
| 4 | ep_ll_triton | fixed_random_shuffle | -0.1% | -12.4% | +1.2% |
| 4 | ep_ll_triton | load_balanced_greedy_with_locality_tiebreak | -4.8% | -16.2% | +7.1% |
| 4 | ep_ll_triton | communication_aware_greedy | -0.7% | +2.6% | +0.5% |
| 4 | ep_ht | fixed_random_shuffle | -18.5% | +21.8% | +23.5% |
| 4 | ep_ht | load_balanced_greedy_with_locality_tiebreak | -18.6% | +33.2% | +16.4% |
| 4 | ep_ht | communication_aware_greedy | +76.1% | -14.7% | -54.1% |

## Profile-Estimated Placement Metrics

| ws | placement | estimated_gpu_cv | cross_traffic_ratio |
| ---: | --- | ---: | ---: |
| 8 | contiguous | 0.0570 | 0.8750 |
| 8 | round_robin | 0.0830 | 0.8749 |
| 8 | fixed_random_shuffle | 0.1167 | 0.8750 |
| 8 | load_balanced_greedy_with_locality_tiebreak | 0.0024 | 0.8748 |
| 8 | communication_aware_greedy | 0.0739 | 0.8736 |
| 4 | contiguous | 0.0539 | 0.7497 |
| 4 | round_robin | 0.0181 | 0.7499 |
| 4 | fixed_random_shuffle | 0.0435 | 0.7501 |
| 4 | load_balanced_greedy_with_locality_tiebreak | 0.0009 | 0.7500 |
| 4 | communication_aware_greedy | 0.0922 | 0.7479 |

## Findings

1. EP-HT is much faster than EP-LL on this short decode workload. Across comparable successful placements, EP-HT has about `8.5x-9.7x` lower e2e time at `ws=8`, and roughly `11x-15x` higher decode throughput.
2. `load_balanced_greedy_with_locality_tiebreak` is the most consistent placement. It gives the lowest estimated GPU load CV at both `ws=8` and `ws=4`, and is also the best EP-LL e2e placement in both world sizes.
3. `fixed_random_shuffle` looks surprisingly strong for EP-HT. It is best e2e at `ws=8` and essentially tied for best e2e at `ws=4`. This suggests contiguous expert id order is not always favorable for EP-HT.
4. `communication_aware_greedy` reduces estimated cross traffic only slightly, but can hurt load balance. The `ws=4 / ep_ht` result is a clear outlier: e2e becomes `+76.1%` worse and decode throughput drops `54.1%` vs contiguous. On this setup, reducing cross traffic by about `0.2%` does not compensate for the load imbalance.
5. `ws=4` is generally better than `ws=8` for EP-LL on this machine. EP-LL decode throughput improves by about `21%-29%` and e2e drops by about `18%-22%` for matching successful placements. This matches the communication-limited 4090/no-NVLink intuition.
6. `ws=8` is not clearly better for EP-HT either at this scale. EP-HT results are mixed; `ws=4` improves `fixed_random_shuffle` and load-balanced e2e, but `communication_aware_greedy` is much worse due to load imbalance.

## Next Checks

1. Fix or avoid master-port reuse for parallel `ws=4` jobs, then rerun the first scale to recover the missing `ep_ht + round_robin` result.
2. Use `load_balanced_greedy_with_locality_tiebreak` and `fixed_random_shuffle` as the main candidates for larger-scale runs.
3. Treat `communication_aware_greedy` as risky unless it is constrained by a load-balance cap.
4. Continue progressive scale after the port issue is fixed: `4x64`, then `8x64`.
