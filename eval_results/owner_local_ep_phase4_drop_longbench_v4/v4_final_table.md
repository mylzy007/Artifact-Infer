# v4 — final per-cell table (perf + recall + sub-metrics)

| plan | policy | rate | prefill_sp | e2e_sp | tok/s | recall | rec Δ | F1 | substr | ROUGE-L |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RR/min_comm | baseline | 0.0 | 1.000 | 1.000 | 709 | 0.577 | — | 0.108 | 0.062 | 0.087 |
| RR/min_comm | cross_numa_first | 0.1 | 1.055 | 1.115 | 748 | 0.587 | +0.010 | 0.110 | 0.078 | 0.090 |
| RR/min_comm | cross_numa_first | 0.3 | 1.238 | 1.117 | 878 | 0.587 | +0.010 | 0.108 | 0.062 | 0.092 |
| RR/min_comm | cross_numa_first | 0.5 | 1.428 | 1.133 | 1013 | 0.539 | -0.038 | 0.098 | 0.078 | 0.081 |
| RR/min_comm | cross_numa_uniform | 0.1 | 1.065 | 1.016 | 754 | 0.598 | +0.021 | 0.110 | 0.078 | 0.089 |
| RR/min_comm | cross_numa_uniform | 0.3 | 1.229 | 1.125 | 873 | 0.579 | +0.002 | 0.104 | 0.094 | 0.085 |
| RR/min_comm | cross_numa_uniform | 0.5 | 1.440 | 1.036 | 1021 | 0.522 | -0.055 | 0.097 | 0.031 | 0.080 |
| RR/min_comm | random | 0.1 | 1.059 | 1.016 | 750 | 0.571 | -0.006 | 0.106 | 0.062 | 0.090 |
| RR/min_comm | random | 0.3 | 1.211 | 1.121 | 859 | 0.609 | +0.032 | 0.110 | 0.125 | 0.093 |
| RR/min_comm | random | 0.5 | 1.440 | 1.142 | 1021 | 0.551 | -0.026 | 0.098 | 0.047 | 0.083 |
| RR/min_comm | tail_weight | 0.1 | 1.059 | 1.080 | 750 | 0.585 | +0.007 | 0.105 | 0.094 | 0.087 |
| RR/min_comm | tail_weight | 0.3 | 1.226 | 1.057 | 869 | 0.578 | +0.001 | 0.106 | 0.047 | 0.088 |
| RR/min_comm | tail_weight | 0.5 | 1.434 | 1.030 | 1017 | 0.605 | +0.028 | 0.108 | 0.125 | 0.091 |
| RR/min_comm | weighted_tail | 0.1 | 1.059 | 1.120 | 750 | 0.583 | +0.006 | 0.109 | 0.031 | 0.089 |
| RR/min_comm | weighted_tail | 0.3 | 1.225 | 1.028 | 868 | 0.591 | +0.014 | 0.109 | 0.078 | 0.094 |
| RR/min_comm | weighted_tail | 0.5 | 1.441 | 1.149 | 1022 | 0.523 | -0.054 | 0.095 | 0.047 | 0.078 |
| LBG/greedy_balance | baseline | 0.0 | 1.000 | 1.000 | 711 | 0.577 | — | 0.108 | 0.062 | 0.087 |
| LBG/greedy_balance | cross_numa_first | 0.1 | 1.060 | 1.031 | 754 | 0.585 | +0.008 | 0.107 | 0.078 | 0.089 |
| LBG/greedy_balance | cross_numa_first | 0.3 | 1.227 | 1.064 | 873 | 0.604 | +0.027 | 0.108 | 0.094 | 0.090 |
| LBG/greedy_balance | cross_numa_first | 0.5 | 1.435 | 1.028 | 1021 | 0.516 | -0.062 | 0.094 | 0.016 | 0.076 |
| LBG/greedy_balance | cross_numa_uniform | 0.1 | 1.056 | 1.025 | 751 | 0.597 | +0.020 | 0.108 | 0.078 | 0.090 |
| LBG/greedy_balance | cross_numa_uniform | 0.3 | 1.222 | 1.037 | 869 | 0.541 | -0.036 | 0.096 | 0.094 | 0.083 |
| LBG/greedy_balance | cross_numa_uniform | 0.5 | 1.428 | 1.042 | 1016 | 0.537 | -0.040 | 0.104 | 0.047 | 0.083 |
| LBG/greedy_balance | random | 0.1 | 1.057 | 1.004 | 751 | 0.575 | -0.002 | 0.104 | 0.062 | 0.087 |
| LBG/greedy_balance | random | 0.3 | 1.217 | 1.046 | 866 | 0.585 | +0.008 | 0.105 | 0.078 | 0.084 |
| LBG/greedy_balance | random | 0.5 | 1.424 | 1.046 | 1013 | 0.503 | -0.074 | 0.094 | 0.031 | 0.076 |
| LBG/greedy_balance | tail_weight | 0.1 | 1.060 | 1.016 | 754 | 0.585 | +0.008 | 0.107 | 0.047 | 0.089 |
| LBG/greedy_balance | tail_weight | 0.3 | 1.218 | 1.041 | 866 | 0.596 | +0.018 | 0.109 | 0.078 | 0.093 |
| LBG/greedy_balance | tail_weight | 0.5 | 1.429 | 1.049 | 1017 | 0.611 | +0.033 | 0.108 | 0.062 | 0.089 |
| LBG/greedy_balance | weighted_tail | 0.1 | 1.052 | 1.005 | 748 | 0.575 | -0.002 | 0.104 | 0.078 | 0.087 |
| LBG/greedy_balance | weighted_tail | 0.3 | 1.221 | 1.030 | 868 | 0.595 | +0.018 | 0.110 | 0.078 | 0.091 |
| LBG/greedy_balance | weighted_tail | 0.5 | 1.424 | 1.029 | 1013 | 0.574 | -0.003 | 0.108 | 0.078 | 0.090 |