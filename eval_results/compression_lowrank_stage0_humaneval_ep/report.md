# HumanEval pass@1 under combine-side compression (Qwen3-30B-A3B, real EP=8)

164 problems, greedy decoding, max_new_tokens=384.

| config | value compr | pass@1 | drop (pp) | relative drop | gen time |
|---|---:|---:|---:|---:|---:|
| teacher | - | 55.49% (91/164) | +0.00pp | +0.0% | 1078.8s |
| keep=0.5_fp8_l2 | 2x | 47.56% (78/164) | +7.93pp | +14.3% | 1329.4s |
| keep=0.25_fp8_l2 | 4x | 46.34% (76/164) | +9.15pp | +16.5% | 1292.4s |
| keep=0.125_fp8_l2 | 8x | 18.29% (30/164) | +37.20pp | +67.0% | 1283.1s |
