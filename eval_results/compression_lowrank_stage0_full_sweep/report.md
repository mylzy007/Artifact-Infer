# Full-model SVD compression sweep (Qwen3-30B-A3B, lcc domain)

- All 48 MoE layers compressed simultaneously, per-layer SVD projector from 8192 lcc calibration tokens.
- Test: 4096 held-out lcc tokens. Teacher PPL = 3.153 (loss 1.1483).

## Sweep

| ell | compression | avg train EV | student PPL | PPL +% | final hidden relMSE | mean top-1 routing agree | per-layer agree min |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 2x | 0.938 | 6.918 | +119.4% | 0.1523 | 0.654 | 0.536 |
| 512 | 4x | 0.826 | 25.561 | +710.7% | 0.2652 | 0.548 | 0.430 |
| 256 | 8x | 0.693 | 4296.199 | +136158.1% | 0.9941 | 0.352 | 0.205 |
| 128 | 16x | 0.561 | 2128823.862 | +67517615.4% | 1.8479 | 0.189 | 0.013 |

## How to read this

- `compression` = d / ell. 2x means each token's dispatch payload shrinks from 2048 floats to 1024.
- `avg train EV` is the in-sample explained variance averaged across all 48 layers (the L^2 measure we used in earlier stages).
- `student PPL +%` is what the user actually feels. <5% likely acceptable, <1% very tight.
- `final hidden relMSE` is sum-of-squared-error / sum-of-squares on the post-norm output. Small means task-level damage is small.
- `mean top-1 routing agreement` is the fraction of tokens whose top-1 expert assignment matches the uncompressed teacher, averaged across all 48 MoE layers. Big drop here is the canonical 'compression broke routing' signal.