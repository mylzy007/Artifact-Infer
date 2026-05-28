# MoE Placement Restats - 2026-05-16 Early Morning

## Run Audit

| label | run_id | ws | max_tokens | batch_tokens | max_seqs | eval_pass | note |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| strict_max8_completed_ws4 | full_ep4_fiws_20260516_0230 | 4 | 8 | 512 | 1 | 10/10 | matches max_tokens=8, but only ws=4 exists |
| ws8_ws4_completed_mismatch_max32_ws8 | perf_ws8_ws4_20260516_0228_p1_t32_ws8 | 8 | 32 | 1024 | 4 | 10/10 | has ws8/ws4, but max_tokens=32 not 8 |
| ws8_ws4_completed_mismatch_max32_ws4 | perf_ws8_ws4_20260516_0228_p1_t32_ws4 | 4 | 32 | 1024 | 4 | 9/10 | has ws8/ws4, but max_tokens=32 not 8 |

## Strict max_tokens=8 Completed Data

These rows are from `moe_placement_run_full_ep4_fiws_20260516_0230`; no completed ws=8 sibling for the pasted command was found on disk.

| ws | impl | placement | pass | e2e_s | prefill_tok_s | decode_tok_s | compute_total | gpu_cv | cross_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | ep_ht | communication_aware_greedy | True | 15.170 | 7.546 | 1.308 | 122880 | 0.0693 | 0.7478 |
| 4 | ep_ht | contiguous | True | 11.063 | 13.499 | 1.256 | 122880 | 0.0575 | 0.7500 |
| 4 | ep_ht | fixed_random_shuffle | True | 12.800 | 9.797 | 1.337 | 122880 | 0.0414 | 0.7499 |
| 4 | ep_ht | load_balanced_greedy_with_locality_tiebreak | True | 10.850 | 12.993 | 1.361 | 122880 | 0.0012 | 0.7498 |
| 4 | ep_ht | round_robin | True | 13.104 | 7.372 | 2.288 | 122880 | 0.0235 | 0.7500 |
| 4 | ep_ll_triton | communication_aware_greedy | True | 18.615 | 10.004 | 0.624 | 122880 | 0.0693 | 0.7478 |
| 4 | ep_ll_triton | contiguous | True | 17.984 | 10.559 | 0.638 | 122880 | 0.0575 | 0.7500 |
| 4 | ep_ll_triton | fixed_random_shuffle | True | 19.382 | 8.943 | 0.631 | 122880 | 0.0414 | 0.7499 |
| 4 | ep_ll_triton | load_balanced_greedy_with_locality_tiebreak | True | 19.697 | 8.801 | 0.621 | 122880 | 0.0012 | 0.7498 |
| 4 | ep_ll_triton | round_robin | True | 21.662 | 7.201 | 0.615 | 122880 | 0.0235 | 0.7500 |

## Completed ws8/ws4 Data Found At 02:28

These rows are included only as the completed four/eight-card early-morning run present on disk. They do not match the pasted `max_tokens=8` command: they use `max_tokens=32`, `max_num_batched_tokens=1024`, `max_num_seqs=4`.

| ws | impl | placement | pass | e2e_s | prefill_tok_s | decode_tok_s | compute_total | gpu_cv | cross_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | ep_ht | communication_aware_greedy | True | 23.345 | 11.473 | 1.839 | 159744 | 0.0922 | 0.7479 |
| 4 | ep_ht | contiguous | True | 13.254 | 13.453 | 4.006 | 159744 | 0.0539 | 0.7497 |
| 4 | ep_ht | fixed_random_shuffle | True | 10.797 | 16.380 | 4.946 | 159744 | 0.0435 | 0.7501 |
| 4 | ep_ht | load_balanced_greedy_with_locality_tiebreak | True | 10.787 | 17.921 | 4.664 | 159744 | 0.0009 | 0.7500 |
| 4 | ep_ht | round_robin | False |  |  |  | 159744 | 0.0181 | 0.7499 |
| 4 | ep_ll_triton | communication_aware_greedy | True | 93.695 | 11.464 | 0.355 | 159744 | 0.0922 | 0.7479 |
| 4 | ep_ll_triton | contiguous | True | 94.317 | 11.172 | 0.354 | 159744 | 0.0539 | 0.7497 |
| 4 | ep_ll_triton | fixed_random_shuffle | True | 94.220 | 9.786 | 0.358 | 159744 | 0.0435 | 0.7501 |
| 4 | ep_ll_triton | load_balanced_greedy_with_locality_tiebreak | True | 89.777 | 9.360 | 0.379 | 159744 | 0.0009 | 0.7500 |
| 4 | ep_ll_triton | round_robin | True | 89.942 | 9.401 | 0.378 | 159744 | 0.0181 | 0.7499 |
| 8 | ep_ht | communication_aware_greedy | True | 13.448 | 11.929 | 4.288 | 319488 | 0.0739 | 0.8736 |
| 8 | ep_ht | contiguous | True | 13.700 | 11.879 | 4.155 | 319488 | 0.0570 | 0.8750 |
| 8 | ep_ht | fixed_random_shuffle | True | 11.897 | 15.455 | 4.371 | 319488 | 0.1167 | 0.8750 |
| 8 | ep_ht | load_balanced_greedy_with_locality_tiebreak | True | 13.464 | 11.466 | 4.431 | 319488 | 0.0024 | 0.8748 |
| 8 | ep_ht | round_robin | True | 13.205 | 12.314 | 4.316 | 319488 | 0.0830 | 0.8749 |
| 8 | ep_ll_triton | communication_aware_greedy | True | 118.520 | 7.176 | 0.287 | 319488 | 0.0739 | 0.8736 |
| 8 | ep_ll_triton | contiguous | True | 116.676 | 7.018 | 0.292 | 319488 | 0.0570 | 0.8750 |
| 8 | ep_ll_triton | fixed_random_shuffle | True | 115.640 | 7.687 | 0.292 | 319488 | 0.1167 | 0.8750 |
| 8 | ep_ll_triton | load_balanced_greedy_with_locality_tiebreak | True | 113.969 | 7.719 | 0.297 | 319488 | 0.0024 | 0.8748 |
| 8 | ep_ll_triton | round_robin | True | 115.228 | 8.042 | 0.292 | 319488 | 0.0830 | 0.8749 |

## Metric Notes

- `e2e_s`, `prefill_*`, and `decode_*` come from `summary.eval_results[].metrics`.
- `compute_total`, `gpu_cv`, and `cross_ratio` are profile-estimated placement metrics from the routing profile, not runtime CUDA kernel timing. The runner did not record runtime `compute_ms` or `comm_ms` fields for these runs.
- `cross_ratio` is `estimated_cross_traffic_ratio`; it is the communication proxy available in this experiment output.
