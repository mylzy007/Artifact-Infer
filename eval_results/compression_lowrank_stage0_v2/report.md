# Stage 0 v2: dispatch activation low-rank check (Qwen3-30B-A3B)

- Model: `/home/lzy/models/Qwen3-30B-A3B`
- hidden_size d = 2048, num_hidden_layers = 48, num_experts = 128, top_k = 8
- Calibration: 64 chunks x 512 tokens = 32768 tokens (no padding); N/d = 16.0
- LongBench sources: ['gov_report.jsonl', 'multi_news.jsonl', 'multifieldqa_en.jsonl', 'lcc.jsonl']
- Probed layers: [2, 6, 12, 24, 36, 46]

## Global explained variance (centered) at fractions of d
| layer | d/16 | d/8 | d/4 | d/2 | rank@0.90 | rank@0.95 | rank@0.99 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.629 | 0.718 | 0.821 | 0.925 | 862 | 1234 | 1772 |
| 6 | 0.396 | 0.530 | 0.692 | 0.867 | 1176 | 1477 | 1854 |
| 12 | 0.367 | 0.498 | 0.661 | 0.846 | 1263 | 1556 | 1898 |
| 24 | 0.385 | 0.511 | 0.669 | 0.848 | 1257 | 1554 | 1900 |
| 36 | 0.435 | 0.557 | 0.702 | 0.863 | 1204 | 1519 | 1892 |
| 46 | 0.550 | 0.660 | 0.781 | 0.903 | 1007 | 1374 | 1850 |

## Per-expert explained variance at d/4 (top-1 destination)
| layer | n_experts | min | p25 | median | p75 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 54 | 0.816 | 0.996 | 1.000 | 1.000 | 1.000 |
| 6 | 35 | 0.780 | 0.990 | 1.000 | 1.000 | 1.000 |
| 12 | 53 | 0.787 | 0.965 | 1.000 | 1.000 | 1.000 |
| 24 | 56 | 0.773 | 0.966 | 1.000 | 1.000 | 1.000 |
| 36 | 57 | 0.829 | 0.966 | 1.000 | 1.000 | 1.000 |
| 46 | 42 | 0.879 | 0.982 | 1.000 | 1.000 | 1.000 |

## Go/No-Go (rule: ≥0.90 explained variance at ℓ = d/4 in ALL probed layers)

**Verdict: NO-GO**

- layer 2: explained variance at d/4 = 0.821  [FAIL]
- layer 6: explained variance at d/4 = 0.692  [FAIL]
- layer 12: explained variance at d/4 = 0.661  [FAIL]
- layer 24: explained variance at d/4 = 0.669  [FAIL]
- layer 36: explained variance at d/4 = 0.702  [FAIL]
- layer 46: explained variance at d/4 = 0.781  [FAIL]
