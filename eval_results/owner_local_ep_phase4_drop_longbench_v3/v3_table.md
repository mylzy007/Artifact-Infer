# v3 — comprehensive table

| plan | policy | rate | prefill_sp | e2e_sp | per-rank tok/s | F1 | F1 Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| RR / numa_local_first / min_communication | baseline | 0.0 | 1.000 | 1.000 | 723 | 0.208 | — |
| RR / numa_local_first / min_communication | cross_numa_first | 0.1 | 1.057 | 1.029 | 765 | 0.207 | -0.001 |
| RR / numa_local_first / min_communication | cross_numa_first | 0.3 | 1.226 | 1.063 | 887 | 0.221 | +0.013 |
| RR / numa_local_first / min_communication | cross_numa_first | 0.5 | 1.422 | 1.103 | 1031 | 0.191 | -0.018 |
| RR / numa_local_first / min_communication | cross_numa_uniform | 0.1 | 1.060 | 1.019 | 767 | 0.208 | -0.001 |
| RR / numa_local_first / min_communication | cross_numa_uniform | 0.3 | 1.229 | 1.056 | 890 | 0.211 | +0.002 |
| RR / numa_local_first / min_communication | cross_numa_uniform | 0.5 | 1.427 | 1.087 | 1034 | 0.205 | -0.003 |
| RR / numa_local_first / min_communication | random | 0.1 | 1.056 | 1.012 | 764 | 0.220 | +0.012 |
| RR / numa_local_first / min_communication | random | 0.3 | 1.219 | 1.067 | 883 | 0.212 | +0.004 |
| RR / numa_local_first / min_communication | random | 0.5 | 1.271 | 1.018 | 985 | 0.212 | +0.004 |
| RR / numa_local_first / min_communication | tail_weight | 0.1 | 1.053 | 0.944 | 762 | 0.210 | +0.002 |
| RR / numa_local_first / min_communication | tail_weight | 0.3 | 1.219 | 1.032 | 882 | 0.204 | -0.005 |
| RR / numa_local_first / min_communication | tail_weight | 0.5 | 1.427 | 1.031 | 1035 | 0.213 | +0.004 |
| RR / numa_local_first / min_communication | weighted_tail | 0.1 | 1.056 | 1.012 | 764 | 0.210 | +0.001 |
| RR / numa_local_first / min_communication | weighted_tail | 0.3 | 1.220 | 1.053 | 884 | 0.202 | -0.007 |
| RR / numa_local_first / min_communication | weighted_tail | 0.5 | 1.428 | 1.097 | 1035 | 0.211 | +0.003 |
| LBG / numa_local_first / greedy_balance | baseline | 0.0 | 1.000 | 1.000 | 720 | 0.208 | — |
| LBG / numa_local_first / greedy_balance | cross_numa_first | 0.1 | 1.058 | 1.019 | 762 | 0.213 | +0.005 |
| LBG / numa_local_first / greedy_balance | cross_numa_first | 0.3 | 1.243 | 1.057 | 896 | 0.205 | -0.003 |
| LBG / numa_local_first / greedy_balance | cross_numa_first | 0.5 | 1.451 | 1.104 | 1046 | 0.194 | -0.014 |
| LBG / numa_local_first / greedy_balance | cross_numa_uniform | 0.1 | 1.062 | 1.022 | 765 | 0.211 | +0.003 |
| LBG / numa_local_first / greedy_balance | cross_numa_uniform | 0.3 | 1.232 | 1.048 | 888 | 0.218 | +0.010 |
| LBG / numa_local_first / greedy_balance | cross_numa_uniform | 0.5 | 1.453 | 1.095 | 1048 | 0.201 | -0.007 |
| LBG / numa_local_first / greedy_balance | random | 0.1 | 1.059 | 1.015 | 763 | 0.219 | +0.010 |
| LBG / numa_local_first / greedy_balance | random | 0.3 | 1.228 | 1.055 | 885 | 0.219 | +0.011 |
| LBG / numa_local_first / greedy_balance | random | 0.5 | 1.448 | 1.092 | 1044 | 0.206 | -0.002 |
| LBG / numa_local_first / greedy_balance | tail_weight | 0.1 | 1.058 | 1.022 | 762 | 0.208 | -0.001 |
| LBG / numa_local_first / greedy_balance | tail_weight | 0.3 | 1.172 | 0.984 | 857 | 0.212 | +0.004 |
| LBG / numa_local_first / greedy_balance | tail_weight | 0.5 | 1.451 | 1.096 | 1047 | 0.230 | +0.021 |
| LBG / numa_local_first / greedy_balance | weighted_tail | 0.1 | 1.057 | 1.015 | 761 | 0.214 | +0.006 |
| LBG / numa_local_first / greedy_balance | weighted_tail | 0.3 | 1.228 | 1.056 | 885 | 0.201 | -0.007 |
| LBG / numa_local_first / greedy_balance | weighted_tail | 0.5 | 1.450 | 1.097 | 1045 | 0.218 | +0.009 |