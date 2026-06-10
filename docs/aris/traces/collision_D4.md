# Trace — Dimension 4 Collision GPT QC

- Date: 2026-05-29
- Reviewer backend: codex, reasoning_effort=high
- Thread ID: 019e6fcd-1131-7c00-8826-1c263de6039c
- Sent ideas: D4-I1 (L*_ℓ artifact), D4-I7 (BOCPD joint slope)

## GPT raw response summary

```
D4-I1 missed priors:
- LLMCompass (arXiv:2312.03134) — hardware-aware LLM latency/cost modeling
- TensorRT timing/tactic cache — engineering artifact pattern
- TVM/Ansor — autotuning logs
- Vidur, MoE-CAP, DeepSpeed-MoE, Tutel — existing MoE cost / runtime work
Salvage: only as MoE-specific control-plane ABI with versioning, drift hooks,
semantic break-even boundary; NOT a "per-layer cost table" claim.
Verdict: "near-dead as novelty; usable as engineering mechanism inside D4".

D4-I7 missed priors:
- BOCPD itself (Adams-MacKay)
- BOCPD scalable robust extensions (Altamirano ICML 2023)
- DriftBench for DB workload drift (arXiv:2510.10858) — adjacency
- AIOps workload-latency anomaly detection (Roots IEEE TCC 2018) — background
Salvage: physics-informed MoE L*-drift diagnostic; canary microprobes or
synthetic-L_recv replay required to neutralize queueing/congestion/batching
confounds.
Verdict: "borderline salvageable; algorithm is prior-heavy".
```

## Claude independent verification of GPT-cited papers

- LLMCompass (arXiv:2312.03134) — verified=YES.
- DriftBench (DB workload, arXiv:2510.10858) — verified=YES.
- BOCPD (Adams-MacKay arXiv:0710.3742) — already verified D4.
- TVM/Ansor / TensorRT — engineering artifacts, not arXiv papers; flagged as
  generic background.

Zero verification failures.

## Verdict update

- D4-I1: from 接近-高风险 → confirmed near-dead novelty. Engineering substrate
  only.
- D4-I7: from 接近-差异 → borderline salvageable; needs canary microprobes.
