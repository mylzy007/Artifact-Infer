# v3 — comprehensive table

| plan | policy | rate | prefill_sp | e2e_sp | per-rank tok/s | F1 | F1 Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| RR / numa_local_first / min_communication | baseline | 0.0 | 1.000 | 1.000 | 709 | 0.108 | — |
| RR / numa_local_first / min_communication | cross_numa_first | 0.1 | 1.055 | 1.115 | 748 | 0.110 | +0.002 |
| RR / numa_local_first / min_communication | cross_numa_first | 0.3 | 1.238 | 1.117 | 878 | 0.108 | -0.000 |
| RR / numa_local_first / min_communication | cross_numa_first | 0.5 | 1.428 | 1.133 | 1013 | 0.098 | -0.010 |
| RR / numa_local_first / min_communication | cross_numa_uniform | 0.1 | 1.065 | 1.016 | 754 | 0.110 | +0.002 |
| RR / numa_local_first / min_communication | cross_numa_uniform | 0.3 | 1.229 | 1.125 | 873 | 0.104 | -0.004 |
| RR / numa_local_first / min_communication | cross_numa_uniform | 0.5 | 1.440 | 1.036 | 1021 | 0.097 | -0.011 |
| RR / numa_local_first / min_communication | random | 0.1 | 1.059 | 1.016 | 750 | 0.106 | -0.002 |
| RR / numa_local_first / min_communication | random | 0.3 | 1.211 | 1.121 | 859 | 0.110 | +0.001 |
| RR / numa_local_first / min_communication | random | 0.5 | 1.440 | 1.142 | 1021 | 0.098 | -0.010 |
| RR / numa_local_first / min_communication | tail_weight | 0.1 | 1.059 | 1.080 | 750 | 0.105 | -0.003 |
| RR / numa_local_first / min_communication | tail_weight | 0.3 | 1.226 | 1.057 | 869 | 0.106 | -0.002 |
| RR / numa_local_first / min_communication | tail_weight | 0.5 | 1.434 | 1.030 | 1017 | 0.108 | -0.000 |
| RR / numa_local_first / min_communication | weighted_tail | 0.1 | 1.059 | 1.120 | 750 | 0.109 | +0.001 |
| RR / numa_local_first / min_communication | weighted_tail | 0.3 | 1.225 | 1.028 | 868 | 0.109 | +0.001 |
| RR / numa_local_first / min_communication | weighted_tail | 0.5 | 1.441 | 1.149 | 1022 | 0.095 | -0.013 |
| LBG / numa_local_first / greedy_balance | baseline | 0.0 | 1.000 | 1.000 | 711 | 0.108 | — |
| LBG / numa_local_first / greedy_balance | cross_numa_first | 0.1 | 1.060 | 1.031 | 754 | 0.107 | -0.001 |
| LBG / numa_local_first / greedy_balance | cross_numa_first | 0.3 | 1.227 | 1.064 | 873 | 0.108 | -0.000 |
| LBG / numa_local_first / greedy_balance | cross_numa_first | 0.5 | 1.435 | 1.028 | 1021 | 0.094 | -0.014 |
| LBG / numa_local_first / greedy_balance | cross_numa_uniform | 0.1 | 1.056 | 1.025 | 751 | 0.108 | -0.000 |
| LBG / numa_local_first / greedy_balance | cross_numa_uniform | 0.3 | 1.222 | 1.037 | 869 | 0.096 | -0.012 |
| LBG / numa_local_first / greedy_balance | cross_numa_uniform | 0.5 | 1.428 | 1.042 | 1016 | 0.104 | -0.005 |
| LBG / numa_local_first / greedy_balance | random | 0.1 | 1.057 | 1.004 | 751 | 0.104 | -0.004 |
| LBG / numa_local_first / greedy_balance | random | 0.3 | 1.217 | 1.046 | 866 | 0.105 | -0.003 |
| LBG / numa_local_first / greedy_balance | random | 0.5 | 1.424 | 1.046 | 1013 | 0.094 | -0.014 |
| LBG / numa_local_first / greedy_balance | tail_weight | 0.1 | 1.060 | 1.016 | 754 | 0.107 | -0.001 |
| LBG / numa_local_first / greedy_balance | tail_weight | 0.3 | 1.218 | 1.041 | 866 | 0.109 | +0.001 |
| LBG / numa_local_first / greedy_balance | tail_weight | 0.5 | 1.429 | 1.049 | 1017 | 0.108 | +0.000 |
| LBG / numa_local_first / greedy_balance | weighted_tail | 0.1 | 1.052 | 1.005 | 748 | 0.104 | -0.004 |
| LBG / numa_local_first / greedy_balance | weighted_tail | 0.3 | 1.221 | 1.030 | 868 | 0.110 | +0.002 |
| LBG / numa_local_first / greedy_balance | weighted_tail | 0.5 | 1.424 | 1.029 | 1013 | 0.108 | -0.000 |