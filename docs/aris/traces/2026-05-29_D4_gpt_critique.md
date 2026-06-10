# Trace — Dimension 4 GPT Idea Critique

- Date: 2026-05-29
- Stage: A (idea generation), Dimension 4 — L*-Aware Cost Model and Control Plane
- Reviewer backend: codex (mcp__codex__codex), reasoning_effort=high
- Thread ID: 019e6f85-3bef-7eb2-b95f-417ae30fdf49
- Rounds per idea: 1 (per ARIS config; user lifted budget cap mid-session,
  protocol retained)
- Sent ideas: D4-I1, D4-I2, D4-I4, D4-I7, D4-I8 (top 5 by Claude self-score)

## Prompt sent (summary)

Phase-A idea critique. No experiments, no plans, no implementations, no collision
verdicts. User finding paste-in: 97% bytes / 2.8% GEMM split; L*≈3271 break-even;
above L* drop pays off, below L* drop hurts. D4 thesis: turn the empirical L*
into a cost-model + control-plane (predictor, decision boundary, regime
classifier), NOT a new drop policy. For each idea: (a) enough system
contribution or just packaged threshold rule?; (b) new measurable benefit or
re-explain existing ±2% / +12%?; (c) mechanism specified vs hand-waved?;
(d) collision risk ≤2 papers + URL; (e) strongest objection; (f) sharp question.

## Raw GPT response

```
D4-I1: Per-layer L*_ℓ calibration as deployable artifact
  (a) Marginal — shipped threshold table reads like profiled config, not a
      full system mechanism.
  (b) Partly — prevents bad drop activation but mostly re-explains existing
      +12% / -5% boundary unless tied to real lever selection.
  (c) Partly — inputs / failure direction clear; confidence interval semantics
      and transfer axes hand-waved.
  (d) Collision risk:
      - Vidur, MLSys 2024,
        https://proceedings.mlsys.org/paper_files/paper/2024/file/b74a8de47d2b3c928360e0a011f48351-Paper-Conference.pdf
      - MoE-CAP, https://arxiv.org/abs/2412.07067
  (e) Strongest objection: "Just hardware-specific autotuning metadata, not a
      research contribution."
  (f) Sharp question: what invariant makes L*_ℓ scientifically meaningful
      beyond this exact model-hardware-runtime triple?

D4-I2: L_recv estimator from microbatch composition + router histogram
  (a) Potentially enough — prediction-before-dispatch is a real control-plane
      primitive, not just post-hoc thresholding.
  (b) Partly — new benefit exists only if prediction beats simple live
      routed-token measurement or token-count heuristics.
  (c) Partly — inputs concrete; router histogram stationarity, prompt mix,
      layer/expert correlation under-specified.
  (d) Collision risk:
      - MoE-Infinity, https://arxiv.org/abs/2401.14361
      - Fast MoE Inference via Predictive Prefetching and Expert Replication,
        https://arxiv.org/abs/2605.11537
  (e) Strongest objection: "if router outputs are already available before
      dispatch, why predict L_recv from stale histograms?"
  (f) Sharp question: what decision must be made before actual routed counts
      are observable?

D4-I4: One-way "don't-drop" safety guard
  (a) Weak standalone — valuable engineering guard, close to
      "if L_recv < L* + margin: disable".
  (b) Partly — produces regression avoidance, not a new positive speedup
      source.
  (c) Partly — decision and asymmetric failure costs clear; "classifier" not
      clearer than a deterministic margin rule.
  (d) Collision risk:
      - Faster MoE LLM Inference for Extremely Large Models,
        https://arxiv.org/abs/2505.03531
      - MoE-CAP, https://arxiv.org/abs/2412.07067
  (e) Strongest objection: "safety wrapper around another scheduler, not a
      paper-level mechanism."
  (f) Sharp question: what does the guard learn or model that a fixed
      0.15·L* margin does not?

D4-I7: Online L*_ℓ recalibration
  (a) More system-like — nonstationary break-even tracking is a legitimate
      control-plane problem, but risks becoming generic changepoint detection
      glued onto MoE.
  (b) Partly — benefit is avoiding stale-threshold regressions under shifts,
      not improving steady-state beyond the known drop gain.
  (c) Partly — rolling inputs / alerts clear; per-layer latency attribution is
      confounded by NCCL contention, batching, replicas, cross-layer overlap.
  (d) Collision risk:
      - Bayesian Online Changepoint Detection (Adams & MacKay 2007),
        https://arxiv.org/abs/0710.3742
      - Vidur (MLSys 2024)
  (e) Strongest objection: "known online changepoint detection; MoE-specific
      novelty is only the monitored scalar."
  (f) Sharp question: can production traces identify true L* drift separately
      from workload-mix drift?

D4-I8: L_recv uncertainty band — conformal-style decision
  (a) Moderate — uncertainty-aware boundary cleaner than point-threshold
      control, still a statistical wrapper unless tied to latency loss.
  (b) Partly — can reduce bad activations under uncertainty; conservatism may
      erase much of the +12% opportunity.
  (c) Partly — lower-bound decision clear; conformal score, calibration
      window, nonstationarity, per-layer dependence not.
  (d) Collision risk:
      - Adaptive Conformal Inference Under Distribution Shift (Gibbs & Candès),
        https://arxiv.org/abs/2106.00170
      - Fast MoE Inference via Predictive Prefetching and Expert Replication,
        https://arxiv.org/abs/2605.11537
  (e) Strongest objection: "coverage of L_recv is not a latency-regret
      guarantee, especially under adaptive serving streams."
  (f) Sharp question: what exactly is guaranteed — L_recv coverage,
      drop-decision precision, or bounded e2e latency regression?
```

## Claude independent verification of GPT-cited papers

- Vidur (MLSys 2024 / arXiv:2405.05465) — verified=YES via web search and
  WebFetch (paper exists, dense LLM focus; MoE coverage unclear).
- MoE-CAP (arXiv:2412.07067) — verified=YES (independently web_fetched; note:
  same paper also at 2505.11415, a withdrawn duplicate).
- MoE-Infinity (arXiv:2401.14361) — verified=YES via D2 trace.
- Fast MoE Inference via Predictive Prefetching and Expert Replication
  (arXiv:2605.11537) — verified=YES (Jyothish et al., 2026, dynamic expert
  replication).
- Faster MoE LLM Inference for Extremely Large Models (arXiv:2505.03531) —
  verified=YES via D4 search.
- Bayesian Online Changepoint Detection (Adams & MacKay, arXiv:0710.3742) —
  verified=YES (2007 classic).
- Adaptive Conformal Inference Under Distribution Shift (Gibbs & Candès,
  arXiv:2106.00170) — verified=YES.

No verification failures.

## Stage-A scope discipline

- No experiments, no source edits, no plan-mode transition, no Stage-B
  collision verdicts.

## Codex usage estimate (cumulative)

- D1 idea critique: 1 call. D2 idea critique: 1 call. D3 idea critique: 1 call.
- D1 QC challenge: 1 call. D2 QC challenge: 1 call. D3 QC challenge: 1 call.
- D4 idea critique: 1 call.
- Connectivity ping: 1 call.
- Total: 7 substantive + 1 ping. User lifted budget cap mid-session; the
  "top 5 / 1 round each" protocol is a content rule and was retained.
