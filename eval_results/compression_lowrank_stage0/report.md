# Stage 0: MoE dispatch activation low-rank check (Qwen3-30B-A3B)

- Model: `/home/lzy/models/Qwen3-30B-A3B`
- hidden_size d = 2048, num_hidden_layers = 48, num_experts = 128, top_k = 8
- Calibration: 44 prompts x max_seq_len=256 -> 3293 non-pad tokens
- Probed layers: [2, 12, 24, 36, 46]

## Cumulative explained variance (centered) at fractions of d
| layer | d/8 | d/4 | d/2 | rank for 0.90 | rank for 0.95 | rank for 0.99 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.895 | 0.941 | 0.983 | 280 | 589 | 1210 |
| 12 | 0.607 | 0.777 | 0.931 | 872 | 1149 | 1622 |
| 24 | 0.633 | 0.790 | 0.934 | 849 | 1131 | 1613 |
| 36 | 0.646 | 0.798 | 0.937 | 834 | 1117 | 1605 |
| 46 | 0.675 | 0.815 | 0.943 | 791 | 1079 | 1584 |

## Go/No-Go (rule from report §6: ≥0.90 explained variance at ℓ = d/4)

**Verdict: NO-GO**

- layer 2: explained variance at d/4 = 0.941  [OK]
- layer 12: explained variance at d/4 = 0.777  [FAIL]
- layer 24: explained variance at d/4 = 0.790  [FAIL]
- layer 36: explained variance at d/4 = 0.798  [FAIL]
- layer 46: explained variance at d/4 = 0.815  [FAIL]
