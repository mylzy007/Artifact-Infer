# Routing-aware vs vanilla SVD projection (Qwen3-30B-A3B, lcc, 48 MoE layers)

- Teacher PPL = 3.153; calib=8192 tokens, test=4096 tokens.
- num_experts = 128; routing-aware variant strictly preserves W_gate row space (rank ≤ 128).

## Side-by-side

| ell | compression | variant | PPL | PPL +% | final hidden relMSE | top-1 routing agree (mean / min) |
|---:|---:|---|---:|---:|---:|---:|
| 1024 | 2x | svd | 6.918 | +119.4% | 0.1523 | 0.654 / 0.536 |
| 1024 | 2x | routing | 30.372 | +863.3% | 0.4393 | 0.395 / 0.174 |
| 512 | 4x | svd | 25.561 | +710.7% | 0.2652 | 0.548 / 0.430 |
| 512 | 4x | routing | 2429.326 | +76948.4% | 0.7555 | 0.205 / 0.056 |
| 256 | 8x | svd | 4296.199 | +136158.1% | 0.9941 | 0.352 / 0.205 |
| 256 | 8x | routing | 201070.526 | +6377046.9% | 1.3181 | 0.092 / 0.007 |

## How to read

- The routing-aware projector spends the first 128 (= num_experts) dimensions of its budget on the EXACT row space of W_gate, so the routing logits are mathematically unchanged.
- 'routing agree' should be 1.000 for the routing variant if the math is implemented correctly (within numerical precision; bf16 cast and downstream chain may flip a few).
- If the routing-aware variant achieves much lower PPL increase, the failure mode of vanilla SVD really is routing destruction; we should pivot the projector design.
- If the routing-aware variant is still bad, the failure is elsewhere (multi-layer error compounding into the residual stream, not routing) and pure dispatch-side low-rank is dead regardless of projector choice.