# v2 sweep — comprehensive per-cell table

| policy | path | rate | prefill (s) | prefill_sp | e2e_sp | F1 | F1 Δ (abs) |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | - | 0.0 | 4.334 | 1.000 | 1.000 | 0.224 | — |
| cross_numa_first | GPU | 0.1 | 4.114 | 1.053 | 1.003 | 0.224 | +0.000 |
| cross_numa_first | GPU | 0.3 | 3.532 | 1.227 | 1.053 | 0.212 | -0.012 |
| cross_numa_first | GPU | 0.5 | 3.197 | 1.355 | 1.042 | 0.241 | +0.017 |
| hot_expert_relief | CPU | 0.1 | 6.461 | 0.671 | 0.860 | 0.220 | -0.004 |
| hot_expert_relief | CPU | 0.3 | 6.119 | 0.708 | 0.878 | 0.208 | -0.016 |
| hot_expert_relief | CPU | 0.5 | 5.934 | 0.730 | 0.874 | 0.226 | +0.002 |
| hotspot_relief | CPU | 0.1 | 6.384 | 0.679 | 0.857 | 0.221 | -0.003 |
| hotspot_relief | CPU | 0.3 | 6.287 | 0.689 | 0.878 | 0.225 | +0.001 |
| hotspot_relief | CPU | 0.5 | 6.852 | 0.633 | 0.815 | 0.225 | +0.001 |
| per_expert_tailtoken | CPU | 0.1 | 6.478 | 0.669 | 0.865 | 0.219 | -0.005 |
| per_expert_tailtoken | CPU | 0.3 | 6.092 | 0.711 | 0.887 | 0.231 | +0.007 |
| per_expert_tailtoken | CPU | 0.5 | 6.342 | 0.683 | 0.864 | 0.231 | +0.007 |
| per_expert_uniform | CPU | 0.1 | 6.360 | 0.681 | 0.870 | 0.219 | -0.005 |
| per_expert_uniform | CPU | 0.3 | 6.053 | 0.716 | 0.882 | 0.224 | -0.001 |
| per_expert_uniform | CPU | 0.5 | 5.841 | 0.742 | 0.881 | 0.229 | +0.005 |
| random | GPU | 0.1 | 4.195 | 1.033 | 0.993 | 0.223 | -0.001 |
| random | GPU | 0.3 | 3.629 | 1.194 | 1.033 | 0.200 | -0.024 |
| random | GPU | 0.5 | 3.108 | 1.394 | 1.076 | 0.205 | -0.019 |
| tail_weight | GPU | 0.1 | 4.146 | 1.045 | 0.997 | 0.213 | -0.011 |
| tail_weight | GPU | 0.3 | 3.673 | 1.180 | 1.019 | 0.202 | -0.022 |
| tail_weight | GPU | 0.5 | 3.043 | 1.424 | 1.035 | 0.229 | +0.005 |