# Task-aware projector vs SVD baseline (layer 24, lcc, ell=512)

- 32 train chunks (16384 tokens), 8 test chunks (4096 tokens)
- ell / d = 512/2048 = 1/4
- trained 500 AdamW steps, lr 0.001, batch 4 chunks (2048 tokens)

## Test metrics: SVD vs trained projector

| metric | SVD baseline | trained | ratio (trained/SVD) |
|---|---:|---:|---:|
| x reconstruction MSE | 1.4435e-01 | 4.0385e-01 | 2.798 |
| x explained variance | 0.4010 | -0.6759 | n/a |
| MoE output MSE | 2.6810e-03 | 5.6031e-03 | 2.090 |
| MoE output relMSE | 0.7388 | 1.5441 | 2.090 |
| final hidden MSE | 1.0342e-01 | 1.6395e-01 | 1.585 |
| final hidden relMSE | 0.0131 | 0.0207 | 1.585 |

## Interpretation cheatsheet

- `x reconstruction MSE`: SVD is provably L^2-optimal for linear projection; trained projector cannot beat it on this metric. Confirms the experiment is well-formed.
- `MoE output relMSE` (relative MSE on layer-L output): does the MoE-layer-aware loss beat naive SVD? Big drop => task-awareness wins.
- `final hidden relMSE`: end-to-end task-level proxy. If this dropped meaningfully, the saved compute really maps to lossless task output.
