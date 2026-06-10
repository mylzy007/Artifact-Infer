# Trace — Stage D1 Round 2b: Codex PLAN Attack on D2-I5

- Date: 2026-05-29
- Reviewer: Codex (`mcp__codex__codex`), `model_reasoning_effort=xhigh`, sandbox=read-only
- Thread ID: 019e742c-786f-7072-b603-769b0dc9995c
- cwd: `/home/lzy/Artifact-Infer-pilot`

## What Codex was given
The D2-I5 PLAN as drafted (research question + pre-registered claims + method summary + comparison groups + fail-fasts + known limitations). Instructed to attack as hostile MLSys reviewer.

## Codex bottom line (verbatim)
> "As written, the plan can support only a narrow statement: 'On logged traces, using microbench-derived L_recv thresholds reduces a router-weight proxy at a comparable dropped-cell budget versus one static baseline.' It cannot support 'quality,' 'Pareto,' 'offline-decidable policy,' or 'mechanism' without online validation, grouped holdouts, real task metrics, and exact semantic equivalence tests for the replayed drop policy."

## 30 attacks (by category)

**Counterfactual / Replay validity (1-2)**: offline replay is not counterfactual (drops change future state); L_recv gate is endogenous (computed from no-drop trace but the policy itself would change L_recv).

**Claim framing (3-5)**: "strictly better Pareto" overclaimed (two-threshold ≠ dominance); static r=0.3 may be strawman; 80%/50% thresholds arbitrary.

**Quality proxy (6-9)**: weight-mass-loss ≠ accuracy; proxy ignores expert output magnitude; aggregate denominator hides bad cases; top-k sidecar may be insufficient (no rerouting modeled).

**L*_ℓ bridge (10-12)**: D1-I10 bridge underdefined (what does microbench optimize?); microbench-to-workload domain shift; per-layer thresholds may encode layer scale, not congestion.

**Splits (13-16)**: step_id//5 splits leak; not stratified by phase/length/load; "Seed 42" meaningless for deterministic split; holdout sizes (150/50) too small.

**Drop semantics (17-19)**: EXP protected-branch semantics may not be replicated; LBG reconstruction weak; bytes-saved ≠ communication-saved (no NCCL bytes).

**Baselines (20-23)**: B-MR matched-random underspecified (matching granularity); B-MB risks leak (calibrating on holdout); B-TH threshold sweep risks post-hoc rescue; ORC-LB framing.

**Negative control (24-26)**: GSM8K too weak (low load by construction); <1% with 50 steps ≈ exact zero; need long-context reasoning negative control too.

**Statistics + reporting (27-30)**: no CIs; rank/layer aggregation hides regressions; phase-gated baseline may leak signal; no latency claim despite byte-motivation.

The PLAN's Devil's Advocate Review section records Claude's response and the corresponding PLAN edits. Full text retained in conversation transcript (Thread ID 019e742c).
