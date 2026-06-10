# Selective per-token compression on lcc (E1)

Test = 4096 held-out lcc tokens. Teacher PPL = 3.153, top-1 = 76.15%.
Per-row compression (when applied) = topk_l2 + FP8 with keep_frac = 0.25 (4x value).

## Next-token top-1 accuracy by (strategy, compress_frac)

| strategy \ compress_frac | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---:|---:|---:|---:|---:|
| random | 76.15% | 75.54% (+0.61pp) | 74.41% (+1.74pp) | 73.51% (+2.64pp) | 72.21% (+3.94pp) |
| router_conf | 76.15% | 75.39% (+0.76pp) | 73.73% (+2.42pp) | 73.43% (+2.72pp) | 72.21% (+3.94pp) |
| hidden_norm | 76.15% | 75.59% (+0.56pp) | 75.34% (+0.81pp) | 74.46% (+1.69pp) | 72.21% (+3.94pp) |

## PPL +% by (strategy, compress_frac)

| strategy \ compress_frac | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---:|---:|---:|---:|---:|
| random | +0.0% | +3.8% | +7.5% | +11.6% | +17.2% |
| router_conf | +0.0% | +5.7% | +10.6% | +13.7% | +17.2% |
| hidden_norm | +0.0% | +1.9% | +3.6% | +7.6% | +17.2% |
