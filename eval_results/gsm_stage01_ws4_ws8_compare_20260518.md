# GSM Stage01 WS4 vs WS8 Comparison

Sources:
- ws4 runtime data: `/home/lzy/Artifact-Infer/eval_results/gsm_stage01_20260517_150323_ws8failed/ws4/summary.json`
- ws8 runtime data: `/home/lzy/Artifact-Infer/eval_results/gsm_stage01_20260518_210739_onlyws8/ws8/summary.json`

Notes:
- `impl` is `ep_ht` for all rows.
- `e2e_s`, `prefill_tok_s`, and `decode_tok_s` come from each run's `metrics`.
- `score` is `average_score` from each run result JSON. This is not semantic similarity; it is the evaluator's correctness score after parsing the predicted final answer and verifying it against the ground-truth answer. In practice here it is an average 0/1 correctness rate over problems.
- `gpu_cv` and `cross_traffic_ratio` for generated placements come from each run's `placement_generation[].estimated_metrics`.
- `gpu_cv` and `cross_traffic_ratio` for `contiguous` and `round_robin` were recomputed from the saved routing profile with the same `estimated_metrics(...)` utility used by placement generation, so the metric definition is consistent across all rows.
- `gpu_cv_layer_mean` is the new per-layer-average load-balance metric: first compute each layer's `gpu_cv`, then average across layers.
- `cross_layer_mean` is the new per-layer-average communication metric: first compute each layer's `cross_traffic_ratio`, then average across layers.

| world_size | impl | placement | score | e2e_s | prefill_tok_s | decode_tok_s | gpu_cv | gpu_cv_layer_mean | cross_traffic_ratio | cross_layer_mean |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | ep_ht | communication_aware_greedy | 0.003906 | 671.660 | 798.225 | 50.915 | 0.157562 | 0.226614 | 0.742656 | 0.742656 |
| 4 | ep_ht | contiguous | 0.007812 | 773.942 | 780.280 | 43.946 | 0.084679 | 0.193893 | 0.751142 | 0.751142 |
| 4 | ep_ht | fixed_random_shuffle | 0.003906 | 747.073 | 853.388 | 45.407 | 0.010750 | 0.196068 | 0.749859 | 0.749859 |
| 4 | ep_ht | load_balanced_greedy_with_locality_tiebreak | 0.003906 | 775.809 | 949.832 | 43.488 | 0.000509 | 0.174392 | 0.749869 | 0.749869 |
| 4 | ep_ht | round_robin | 0.000000 | 672.219 | 892.304 | 50.608 | 0.029673 | 0.164639 | 0.749533 | 0.749533 |
| 8 | ep_ht | communication_aware_greedy | 0.019531 | 742.122 | 679.829 | 46.270 | 0.210176 | 0.362565 | 0.875000 | 0.875000 |
| 8 | ep_ht | contiguous | 0.035156 | 803.968 | 691.442 | 42.510 | 0.059353 | 0.298912 | 0.875000 | 0.875000 |
| 8 | ep_ht | fixed_random_shuffle | 0.054688 | 743.547 | 616.265 | 46.447 | 0.099877 | 0.318633 | 0.875000 | 0.875000 |
| 8 | ep_ht | load_balanced_greedy_with_locality_tiebreak | 0.035156 | 732.772 | 654.749 | 46.993 | 0.000586 | 0.281830 | 0.875000 | 0.875000 |
| 8 | ep_ht | round_robin | 0.050781 | 747.784 | 703.407 | 45.825 | 0.045644 | 0.266028 | 0.875000 | 0.875000 |

WS8 minus WS4:

| placement | delta_e2e_s | delta_prefill_tok_s | delta_decode_tok_s | delta_gpu_cv | delta_cross_traffic_ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| communication_aware_greedy | 70.461 | -118.396 | -4.645 | 0.052614 | 0.132344 |
| contiguous | 30.026 | -88.838 | -1.435 | -0.025326 | 0.123858 |
| fixed_random_shuffle | -3.526 | -237.123 | 1.040 | 0.089127 | 0.125141 |
| load_balanced_greedy_with_locality_tiebreak | -43.037 | -295.083 | 3.505 | 0.000077 | 0.125131 |
| round_robin | 75.565 | -188.898 | -4.783 | 0.015971 | 0.125467 |

Conclusions:
- The cross-rank traffic floor moves sharply upward from ws4 to ws8. In ws4, cross traffic is around `0.743-0.751`; in ws8 it is effectively pinned at `0.875` for every placement. That means placement policy has much less room to reduce communication once EP is spread across 8 ranks.
- In ws4, runtime tracks communication more clearly. `communication_aware_greedy` and `round_robin` are the two fastest placements, and both also have the best decode throughput (`50.915` and `50.608 tok/s`).
- In ws8, decode throughput becomes much tighter across placements (`42.510-46.993 tok/s`), and the best overall runtime is `load_balanced_greedy_with_locality_tiebreak` rather than `communication_aware_greedy`. This suggests ws8 is no longer dominated by cross-traffic differences alone; load balance and per-rank compute skew matter more.
- `load_balanced_greedy_with_locality_tiebreak` is the most stable placement across both scales. It has the lowest `gpu_cv` in both ws4 and ws8, and it improves from being worst in ws4 e2e to best in ws8 e2e, which is consistent with higher-rank EP becoming more sensitive to compute imbalance.
- `communication_aware_greedy` helps most in ws4, where it meaningfully lowers `cross_traffic_ratio` (`0.742656`) and wins e2e. In ws8 its communication advantage disappears because all policies sit at `0.875`, while its `gpu_cv` becomes the worst (`0.210176`), and its runtime falls behind the more balanced placements.
- `fixed_random_shuffle` is not competitive in ws4 but becomes middle-of-pack in ws8. Its decode throughput improves relative to contiguous and round_robin in ws8, but its `gpu_cv` is still much worse than the balanced greedy placement.
- Score and runtime are not tightly coupled here. In ws4 the fastest placements are not the highest-scoring ones, while in ws8 `fixed_random_shuffle` and `round_robin` get the best scores even though `load_balanced_greedy_with_locality_tiebreak` has the best e2e/runtime profile. So current placement effects are changing model behavior enough that throughput-optimal and accuracy-optimal placements are not the same.
