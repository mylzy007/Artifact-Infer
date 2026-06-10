# Trace — Dimension 5 GPT Idea Critique

- Date: 2026-05-29
- Stage: A (idea generation), Dimension 5 — Minimal Deepening, Not Component Stacking
- Reviewer backend: codex (mcp__codex__codex), reasoning_effort=high
- Thread ID: 019e6f98-7108-7422-bc68-3c5bf21d46e9
- Rounds per idea: 1 (per ARIS config)
- Sent ideas: D5-I2, D5-I3, D5-I4, D5-I5, D5-I9 (top 5 by Claude self-score)

## Prompt sent (summary)

Phase-A idea critique. Strict rules. Dimension 5 = "做减法不做加法" — paper-grade
contributions deepening the drop finding via ONE dominant mechanism each, no
stacking. Reward minimal-component + sharp falsifiability; penalize "and we also
add X" / "explore the possibility of Y". For each idea: (a) really doing 减法?;
(b) new conceptual artifact or rewording existing +12% / ±2%?; (c) MLSys-2026
paper-level?; (d) collision risk ≤2 papers + URL; (e) strongest objection;
(f) sharp question.

## Raw GPT response

```
D5-I2: Drop ↔ byte-conservative router-perturbation equivalence
  (a) 减法: YES, if the entire paper is the equivalence class.
  (b) New artifact: PARTLY. Real equivalence theorem is new; "drop = stochastic
      mask/noisy router" is close to tautology unless it preserves weighted
      expert output under top-k renormalization.
  (c) MLSys-level: PARTLY. Theory-only can work, but output-L2 R² is a weak
      systems artifact.
  (d) Collision risk:
      - Shazeer et al. noisy top-k routing, https://arxiv.org/abs/1701.06538
      - Capacity-Aware Inference (token drop at inference),
        https://arxiv.org/abs/2503.05066
  (e) Strongest objection: "Every drop policy is a router perturbation" is
      either false for arbitrary policies or vacuous by construction.
  (f) Sharp question: what exactly is invariant — expert identity distribution,
      weighted expert output expectation, or final residual stream distribution?

D5-I3: Information-theoretic lower bound on per-step a2a bytes
  (a) 减法: YES.
  (b) New artifact: YES, if mathematically nontrivial and tight enough.
  (c) MLSys-level: PARTLY. A loose Shannon-style bound will be dismissed as
      obvious gap theater.
  (d) Collision risk:
      - Rate-distortion model-compression theory,
        https://arxiv.org/abs/1810.06401
      - LatentMoE (cost/accuracy limits for MoE routed paths),
        https://arxiv.org/abs/2601.18089
  (e) Strongest objection: token/router entropy does not determine the bits
      needed to preserve nonlinear expert outputs; the real object is remote
      activation transformation, not token identity.
  (f) Sharp question: what random variable is being lower-bounded — dispatch
      activations, expert outputs, router decisions, or task loss?

D5-I4: Single-knob byte-budget runtime
  (a) 减法: NO. This is three levers under one knob — disguised stacking.
  (b) New artifact: PARTLY. "Byte budget" is a clean abstraction, but Pareto
      matching is controller packaging, not a concept.
  (c) MLSys-level: NO as stated. Reads like an autotuner over known knobs.
  (d) Collision risk:
      - Dynamic top-k MoE routing, https://arxiv.org/abs/2403.07652
      - FP8 post-training quantization, https://arxiv.org/abs/2309.14592
  (e) Strongest objection: dominant mechanism is not byte budget; it is
      whichever of drop/K/quant happens to win at each point.
  (f) Sharp question: after removing the three underlying knobs, what theorem
      or abstraction remains?

D5-I5: Replica-minimality above L*
  (a) 减法: YES — removes replica enthusiasm above L*.
  (b) New artifact: PARTLY. Regime law, but only if more than one-system
      observation.
  (c) MLSys-level: PARTLY. Strong taste; narrow hardware/model dependence is
      the danger.
  (d) Collision risk:
      - vLLM EPLB (redundant experts), vllm docs (not arXiv)
      - CRAFT (expert replica allocation), https://arxiv.org/abs/2603.28768
  (e) Strongest objection: replicas can change locality, queueing, and hotspot
      contention; "bytes dominate" does not prove replicas are irrelevant.
  (f) Sharp question: does L* rule out replica benefit in topology/queueing
      terms, or only in pure payload-byte terms?

D5-I9: Decode-hopelessness theorem
  (a) 减法: YES. Clean theorem attempt; no new mechanism.
  (b) New artifact: YES. Turns ±2% decode failure into a structural inequality.
  (c) MLSys-level: PARTLY. Potentially sharp, but "any byte-reduction operator"
      is overclaimed.
  (d) Collision risk:
      - METRO (decode-serving in memory-bound MoE),
        https://arxiv.org/abs/2512.09277
      - Scaling Multi-Node MoE Inference via Activation Patterns,
        https://arxiv.org/abs/2604.23150
  (e) Strongest objection: decode L_recv is not structurally fixed; batching,
      speculative verification, MTP, and serving mix can push decode into
      different regimes.
  (f) Sharp question: is the theorem about Qwen3/8×4090/ep_ht constants, or a
      model-independent MoE decode law?
```

## Claude independent verification of GPT-cited papers

- Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer" (arXiv:1701.06538) — verified=YES.
- Capacity-Aware Inference (arXiv:2503.05066) — verified=YES via D1 trace.
- Rate Distortion For Model Compression (arXiv:1810.06401) — verified=YES.
- LatentMoE (arXiv:2601.18089) — verified=YES.
- Dynamic top-k MoE routing (arXiv:2403.07652) — verified=YES.
- FP8 post-training quantization (arXiv:2309.14592) — verified=YES.
- CRAFT (arXiv:2603.28768) — verified=YES via D2 trace.
- METRO (arXiv:2512.09277) — verified=YES.
- Scaling Multi-Node MoE Inference via Activation Patterns (arXiv:2604.23150) —
  verified=YES via D1 trace.

vLLM EPLB doc page (https://docs.vllm.ai/...) is documentation, not an arXiv
paper — flagged but not formally verified.

No verification failures.

## Stage-A scope discipline

- No experiments, no source edits, no plan-mode transition.
- No "已撞车 / 没撞车" verdicts; only preliminary collision risk.

## Codex usage estimate (cumulative)

- D1+D2+D3+D4 idea critiques: 4 calls.
- D1+D2+D3+D4 QC challenges: 4 calls.
- D5 idea critique: 1 call.
- Connectivity ping: 1 call.
- Total: 9 substantive + 1 ping. User lifted budget cap mid-session; "top 5 /
  1 round each" protocol kept as content rule.
