# Per-cell summary

| phase | label | policy | rate | min_replicas | tier | n_batches | prefill_mean (s) | prefill_std | prefill_speedup | e2e_speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| A | A_baseline | none | 0.0 | 512 | medium | 3 | 4.484 | 0.147 | — | — |
| A | A_cross_numa_first_r0.1 | cross_numa_first | 0.1 | 512 | medium | 3 | 4.261 | 0.160 | 1.052 | 1.010 |
| A | A_cross_numa_first_r0.2 | cross_numa_first | 0.2 | 512 | medium | 3 | 3.925 | 0.140 | 1.142 | 1.053 |
| A | A_cross_numa_first_r0.3 | cross_numa_first | 0.3 | 512 | medium | 3 | 3.653 | 0.140 | 1.227 | 1.096 |
| A | A_cross_numa_first_r0.5 | cross_numa_first | 0.5 | 512 | medium | 3 | 3.177 | 0.122 | 1.411 | 1.151 |
| A | A_random_r0.1 | random | 0.1 | 512 | medium | 3 | 4.297 | 0.159 | 1.043 | 1.001 |
| A | A_random_r0.2 | random | 0.2 | 512 | medium | 3 | 3.961 | 0.135 | 1.131 | 1.049 |
| A | A_random_r0.3 | random | 0.3 | 512 | medium | 3 | 3.693 | 0.125 | 1.214 | 1.077 |
| A | A_random_r0.5 | random | 0.5 | 512 | medium | 3 | 3.158 | 0.114 | 1.419 | 1.145 |
| A | A_tail_weight_r0.1 | tail_weight | 0.1 | 512 | medium | 3 | 4.244 | 0.162 | 1.056 | 1.017 |
| A | A_tail_weight_r0.2 | tail_weight | 0.2 | 512 | medium | 3 | 4.038 | 0.128 | 1.110 | 1.039 |
| A | A_tail_weight_r0.3 | tail_weight | 0.3 | 512 | medium | 3 | 3.699 | 0.100 | 1.212 | 1.087 |
| A | A_tail_weight_r0.5 | tail_weight | 0.5 | 512 | medium | 3 | 3.161 | 0.108 | 1.418 | 1.157 |
| B | B_bypass0 | tail_weight | 0.3 | 0 | medium | 3 | 3.692 | 0.137 | 1.214 | 0.975 |
| B | B_bypass128 | tail_weight | 0.3 | 128 | medium | 3 | 3.699 | 0.135 | 1.212 | 1.087 |
| B | B_bypass2048 | tail_weight | 0.3 | 2048 | medium | 3 | 3.731 | 0.115 | 1.201 | 1.073 |
| C | C_long_baseline | none | 0.0 | 512 | long | 1 | 5.550 | 0.000 | — | — |
| C | C_long_drop | tail_weight | 0.3 | 512 | long | 1 | 4.538 | 0.000 | 1.223 | 1.089 |
| C | C_medium_baseline | none | 0.0 | 512 | medium | 3 | 4.482 | 0.164 | — | — |
| C | C_medium_drop | tail_weight | 0.3 | 512 | medium | 3 | 3.735 | 0.128 | 1.200 | 1.072 |