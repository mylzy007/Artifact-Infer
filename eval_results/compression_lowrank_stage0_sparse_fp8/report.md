# Combine-side topk_l2 + FP8 stack (Qwen3-30B-A3B, lcc, 48 MoE layers)

Teacher PPL = 3.153; held-out lcc test = 4096 tokens.

FP8 = per-row E4M3 round-trip with amax/448 scaling (DeepEP/MoRI-style production recipe).

## Sweep

| variant | keep_frac | k | value-only ratio | student PPL | PPL +% | hidden relMSE |
|---|---:|---:|---:|---:|---:|---:|
| topk_l2_bf16 | 0.2500 | 512 | 4.0x | 3.720 | +18.0% | 0.1417 |
| topk_l2_bf16 | 0.1250 | 256 | 8.0x | 5.197 | +64.8% | 0.2697 |
| topk_l2_bf16 | 0.0625 | 128 | 16.0x | 10.421 | +230.5% | 0.4043 |
| topk_l2_fp8 | 0.2500 | 512 | 8.0x | 3.695 | +17.2% | 0.1405 |
| topk_l2_fp8 | 0.1250 | 256 | 16.0x | 5.166 | +63.9% | 0.2686 |
| topk_l2_fp8 | 0.0625 | 128 | 32.0x | 10.321 | +227.3% | 0.4037 |
| naive_topk_fp8 | 0.2500 | 512 | 8.0x | 3.803 | +20.6% | 0.1413 |
| naive_topk_fp8 | 0.1250 | 256 | 16.0x | 5.782 | +83.4% | 0.2749 |
| naive_topk_fp8 | 0.0625 | 128 | 32.0x | 15.532 | +392.6% | 0.4161 |
| fp8_only | 1.0000 | 2048 | 2.0x | 3.150 | -0.1% | 0.0070 |
| fp8_only | 1.0000 | 2048 | 2.0x | 3.150 | -0.1% | 0.0070 |
| fp8_only | 1.0000 | 2048 | 2.0x | 3.150 | -0.1% | 0.0070 |
