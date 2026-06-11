# HumanEval pass@1 — E1/E2 selective compression (real EP=8)

164 problems, greedy, max_new_tokens=384.

| config | mode | frac | pass@1 | drop vs teacher | drop vs 4x baseline |
|---|---|---:|---:|---:|---:|
| teacher | uniform | 1.000 | 55.49% (91/164) | +0.00pp | -9.15pp |
| baseline_uniform_4x | uniform | 1.000 | 46.34% (76/164) | +9.15pp | +0.00pp |
| selective_row_weight_50 | row_weight | 0.500 | 47.56% (78/164) | +7.93pp | -1.22pp |
| selective_row_weight_875 | row_weight | 0.875 | 50.61% (83/164) | +4.88pp | -4.27pp |
| selective_row_norm_50 | row_norm | 0.500 | 47.56% (78/164) | +7.93pp | -1.22pp |