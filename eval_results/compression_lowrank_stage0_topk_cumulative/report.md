# Top-k cumulative weight protection on lcc (E2)

Test = 4096 held-out lcc tokens. Teacher PPL = 3.153, top-1 = 76.15%.
Per-row compression (when applied) = topk_l2 + FP8 with keep_frac = 0.25 (4x value).

**Threshold semantics**: for each token's top-8 experts sorted descending by routing weight, an expert is kept FULL iff the cumulative weight BEFORE it (exclusive) < threshold. `top1_only` is a special case that protects only the rank-1 expert per token regardless of weight.

| threshold | top-1 acc | top-1 drop | PPL | PPL +% | compressed pair frac | avg compr (value-only) | weight % compressed |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 72.21% | +3.94pp | 3.695 | +17.2% | 100.0% | 4.00x | 100.0% |
| top1_only | 74.41% | +1.74pp | 3.368 | +6.8% | 87.5% | 2.91x | 76.9% |
| 0.5 | 75.61% | +0.54pp | 3.200 | +1.5% | 59.8% | 1.81x | 43.3% |
| 0.7 | 75.88% | +0.27pp | 3.161 | +0.3% | 39.1% | 1.41x | 25.0% |
| 0.9 | 75.98% | +0.17pp | 3.151 | -0.1% | 12.8% | 1.11x | 7.1% |
| 1.0 | 76.15% | +0.00pp | 3.153 | +0.0% | 0.0% | 1.00x | 0.0% |

## Reading
- `threshold=0.0` is the original baseline: every (token, expert) compressed at 4x → avg ~4x.
- `threshold=top1_only` keeps just rank-1 expert per token full; with k=8 → 1/8 = 12.5% pairs full.
- `threshold=0.5/0.7/0.9` keeps a growing prefix; weight % compressed shows what fraction of total routing mass got compressed.
- `threshold=1.0` keeps everything full → teacher.