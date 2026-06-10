# Task-level next-token accuracy — combine-side compression on lcc

Held-out lcc test: 4096 tokens (8 chunks x 512 tokens).
Teacher: loss=1.1483  ppl=3.153  top1=76.15%  top5=90.80%

| config | value compr | PPL | PPL +% | top-1 acc | top-1 drop (pp) | top-5 acc | top-5 drop (pp) | teacher top-1 agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| teacher | - | 3.153 | - | 76.15% | - | 90.80% | - | - |
| keep=0.5 +FP8 +L2 | 2x | 3.226 | +2.3% | 75.49% | +0.66pp | 90.83% | -0.02pp | 94.84% |
| keep=0.25 +FP8 +L2 | 4x | 3.695 | +17.2% | 72.21% | +3.94pp | 88.80% | +2.01pp | 86.47% |
| keep=0.125 +FP8 +L2 | 8x | 5.166 | +63.9% | 66.73% | +9.42pp | 84.47% | +6.34pp | 75.78% |

## How to read

- `top-1 acc` is fraction of held-out positions where the model's argmax-predicted next token equals the true next token. Standard LM eval.
- `top-1 drop (pp)` is teacher_top1 - student_top1 in absolute percentage points. Small = compression preserves task.
- `teacher top-1 agreement` is fraction of positions where student's top-1 prediction == teacher's top-1 prediction, even when both are wrong. Strict 'are the two models behaving identically' metric.
- For greedy decoding (temperature=0) deployments, `top-1 agreement` is the most direct quality proxy.