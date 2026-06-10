# Combine-side top-magnitude sparsification sweep (Qwen3-30B-A3B, lcc, 48 MoE layers)

- Teacher PPL = 3.153; test = 4096 held-out lcc tokens.
- Each of the 6144 expert modules has a forward hook that sparsifies its output row-wise to the top-k magnitude entries.
- 'value-only' compression assumes infinite-bandwidth indices (purely the kept fraction).
- 'value+index' compression assumes we ALSO send int16 indices alongside each kept bf16 value (so the per-kept payload is 4 bytes vs the original 2H bytes per row).

## Sweep

| keep_frac | k | value-only | value+index | student PPL | PPL +% | final hidden relMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5000 | 1024 | 2.00x | 1.00x | 3.257 | +3.3% | 0.0419 |
| 0.2500 | 512 | 4.00x | 2.00x | 3.803 | +20.6% | 0.1412 |
| 0.1250 | 256 | 8.00x | 4.00x | 5.827 | +84.8% | 0.2777 |
| 0.0625 | 128 | 16.00x | 8.00x | 15.350 | +386.8% | 0.4149 |
| 0.0312 | 64 | 32.00x | 16.00x | 49.171 | +1459.5% | 0.5211 |

## Cross-comparison vs dispatch-side SVD (from prior sweep)

From `eval_results/compression_lowrank_stage0_full_sweep/`:
- dispatch SVD 2x:  teacher 3.15 -> 6.92 PPL (+119%)
- dispatch SVD 4x:  teacher 3.15 -> 25.6 PPL (+711%)
- dispatch SVD 8x:  teacher 3.15 -> 4296 PPL (+136 158%)
