# Cross-domain validation of combine-side sparsification (Qwen3-30B-A3B)

All 48 MoE layers' expert outputs are sparsified; held-out test set per domain.

## Teacher PPL per domain
| domain | teacher PPL |
|---|---:|
| lcc | 3.153 |
| multifieldqa | 6.092 |
| dureader | 21.801 |
| gov_report | 6.622 |

## PPL @ keep_frac = 0.25 (value-only 4x compression)

| domain | teacher | naive_topk | topk_l2 | best PPL +% |
|---|---:|---:|---:|---:|
| lcc | 3.153 | 3.803 (+20.6%) | 3.720 (+18.0%) | +18.0% |
| multifieldqa | 6.092 | 7.347 (+20.6%) | 7.189 (+18.0%) | +18.0% |
| dureader | 21.801 | 28.969 (+32.9%) | 27.367 (+25.5%) | +25.5% |
| gov_report | 6.622 | 7.852 (+18.6%) | 7.558 (+14.1%) | +14.1% |

## PPL @ keep_frac = 0.125 (value-only 8x compression)

| domain | teacher | naive_topk | topk_l2 | best PPL +% |
|---|---:|---:|---:|---:|
| lcc | 3.153 | 5.827 (+84.8%) | 5.197 (+64.8%) | +64.8% |
| multifieldqa | 6.092 | 15.187 (+149.3%) | 11.964 (+96.4%) | +96.4% |
| dureader | 21.801 | 59.141 (+171.3%) | 45.151 (+107.1%) | +107.1% |
| gov_report | 6.622 | 13.792 (+108.3%) | 11.275 (+70.3%) | +70.3% |

## Reading

If `topk_l2` strictly beats `naive_topk` across all four domains, the L2 rescale fix is domain-independent.
If `dureader` (Chinese) and `gov_report` (long reports) have similar PPL +% to `lcc` (code), the combine-side advantage is not lcc-specific.