# Trace — Stage D1 Round 2a: Codex PLAN Attack on D1-I10

- Date: 2026-05-29
- Reviewer: Codex (`mcp__codex__codex`), `model_reasoning_effort=xhigh`, sandbox=read-only
- Thread ID: 019e742a-8180-7272-b340-f24b1dd710e1
- cwd: `/home/lzy/Artifact-Infer-pilot`

## What Codex was given
The D1-I10 PLAN as drafted (research question + pre-registered claims + method summary + fail-fasts + known limitations). Instructed to attack as a hostile MLSys reviewer.

## Codex bottom line (verbatim)
> "This plan is not publication-ready as written. The biggest problems are: 'causal' is overstated, logit replay may not reproduce production routing semantics, `worst_peer_bytes`, `L_recv`, and `FLOPs_proxy` are mechanically entangled, the `L*_ℓ` calibration is underspecified and may leak, and the success thresholds look invented."

## 48 attacks (by category; see PLAN's Devil's Advocate Review for response + edit)

**Definitional / Mechanical (1-10)**: "causal" overstated; worst-peer mechanically coupled to L_recv; FLOPs_proxy ≡ L_recv × const; 20pp/10pp/50% thresholds arbitrary; tertile stability weak; no layer fixed effects; no token-shape controls; rank-max confounds straggler effects; worst-peer may not be the right network primitive.

**Logit-encoding replay (11-18)**: equal +1000 ties destroy ordering; topk_weights looked up but ignored; ±1000 numerical edge cases; send-count match insufficient validation; 1% mismatch tolerance loose & undefined unit; LBG validation indirect; random hidden states break compute; replay misses upstream runtime state.

**Calibration / L*_ℓ (19-23)**: split (step_id//5 mod 10 < 7) creates contiguous blocks → leak; L*_ℓ calibrated on same cells as regression; drop_rate=0.3 arbitrary; tail-weight semantics may differ; per-layer thresholds assume separability.

**Regression methodology (24-30)**: N=864 ≠ N=864 independent (correlated by workload/step/layer); stratified sampling distorts distribution; R² dominated by between-bucket variance; linear assumption unjustified; no outlier policy; no CIs; scale/units not normalized.

**Runtime / Hardware (31-35)**: 8×4090 fragile platform; dispatcher as fixed truth; overlap not accounted for; warmup/iters look optimistic; STOP wall-time bias.

**Failure handling (36-40)**: "scope-limit framing" too soft; anti-claim language confusing; STOP biases dataset; N<300 too low; ≥24 distinct cells too weak.

**Definitions (41-48)**: L_recv undefined; skew bucket undefined; segment latency undefined; dispatch/combine collapsed prematurely; expert compute dominates; no held-out validation; no multi-run reproducibility; claimed generality exceeds experiment.

**Top 10 PLAN edits Codex demanded**:
1. Remove "causal" or add true route-matrix interventions
2. Define L_recv, skew, segment latency, mismatch, tier1 baseline, cell
3. Fix replay: preserve top-k order and weights, or bypass logits
4. Validate replay at token-level, all ranks, exact runtime mode
5. Separate regression data, L*_ℓ calibration data, downstream D2 data
6. Add held-out grouped validation and clustered uncertainty
7. Add layer fixed effects and controls for token shape, rank, workload, run
8. Report collinearity diagnostics and residualized worst-peer analysis
9. Replace arbitrary 20pp/10pp bars with justified thresholds plus CIs
10. Treat STOP-triggered cells as censored data, not silent exclusions

Full attack text retained in conversation transcript (turn that produced Thread ID 019e742a). The PLAN's Devil's Advocate Review section records Claude's response and the corresponding PLAN edits.
