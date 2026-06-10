# Trace — Dimension 5 Collision GPT QC

- Date: 2026-05-29
- Reviewer backend: codex, reasoning_effort=high
- Thread ID: 019e6fd2-104d-7f82-a1ba-5c5b4dbf418e
- Sent ideas: D5-I9 (decode-hopelessness theorem), D5-I10 (byte-counted primitive)

## GPT raw response summary

```
D5-I9 missed priors:
- MoE-Spec (arXiv:2602.16052) — Expert Budgeting for Efficient SD
- Sem-MoE / Semantic Parallelism (arXiv:2503.04398) — communication-aware MoE
  inference
Critical already-cited:
- qs Inequality (arXiv:2603.08960)
- MoESD (arXiv:2505.19645) NeurIPS'25 spotlight
- Utility-Driven Speculative MoE (arXiv:2506.20675) — Theorem 4.2 net negative SD
Salvage: communication-operator no-benefit certificate; SD/MTP = escape hatch
(changes T per step, not a byte reducer); decision table per operator.
Verdict: 接近, SD/MTP part partially 完全撞.

D5-I10 missed priors:
- UCCL-EP (arXiv:2512.19849) — portable DeepEP alternative
- Capacity-Aware Inference (arXiv:2503.05066) — inference token drop
- Occult (arXiv:2505.13345) — MoE comm-cost + collab pruning
- Sem-MoE (arXiv:2503.04398) — token-expert co-scheduling for a2a
- TensorRT-LLM NVLinkOneSided AlltoAll (NVIDIA blog, operational)
Salvage: budgeted semantic collective (BudgetedDispatchCombine / BudgetedAllToAllV);
formal byte-budget contract + error bounds + router-score-aware degradation +
unified lowering of drop/top-k/quant into budget optimizer.
Verdict: 接近到完全撞 as primitive; 互补 if reframed as semantics layer above EP libs.
```

## Claude independent verification of GPT-cited papers

- MoE-Spec (arXiv:2602.16052) — verified=YES.
- UCCL-EP (arXiv:2512.19849) — verified=YES.
- Occult (arXiv:2505.13345) — verified=YES.
- Semantic Parallelism / Sem-MoE (arXiv:2503.04398) — verified via D1 trace.
- Capacity-Aware Inference (arXiv:2503.05066) — verified via D1 trace.
- TensorRT-LLM NVLinkOneSided AlltoAll — NVIDIA blog (operational, not arXiv).

Zero verification failures.

## Verdict update

- D5-I9: from 接近-high risk → confirmed 接近, salvage = byte-side certificate +
  decision table per operator. SD/MTP framing collides hard with MoESD and
  Utility-Driven SD.
- D5-I10: from 互补 → 接近-完全撞 as primitive; 互补 only as semantics layer.
