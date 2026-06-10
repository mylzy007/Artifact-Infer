# Combine-side sparsification: error-feedback variants (Qwen3-30B-A3B, lcc, 48 layers)

Teacher PPL = 3.153; held-out lcc test = 4096 tokens.

## PPL by (variant, keep_frac)

| variant | keep=0.25 | keep=0.125 | keep=0.0625 |
|---|---|---|---|
| naive_topk | 3.803 (+20.6%) | 5.827 (+84.8%) | 15.350 (+386.8%) |
| topk_l2 | 3.720 (+18.0%) | 5.197 (+64.8%) | 10.421 (+230.5%) |
| topk_l1 | 4.731 (+50.0%) | 74.696 (+2269.0%) | 24765.175 (+785351.6%) |
| stochastic | 11.372 (+260.7%) | 12009.240 (+380784.7%) | 1925763.945 (+61077373.0%) |

## hidden relMSE by (variant, keep_frac)

| variant | keep=0.25 | keep=0.125 | keep=0.0625 |
|---|---|---|---|
| naive_topk | 0.1412 | 0.2777 | 0.4149 |
| topk_l2 | 0.1417 | 0.2697 | 0.4043 |
| topk_l1 | 0.3058 | 0.7218 | 1.1571 |
| stochastic | 0.4904 | 1.2162 | 1.8914 |

## Winner at keep_frac = 0.125

**topk_l2** (ranking by PPL ascending: ['topk_l2', 'naive_topk', 'topk_l1', 'stochastic'])
