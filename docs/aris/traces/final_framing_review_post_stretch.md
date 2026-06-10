# Trace — Final-Final Framing Review (post-stretch)

- Date: 2026-05-30
- Reviewer: Codex (`mcp__codex__codex`), `model_reasoning_effort=xhigh`
- Thread ID: 019e78aa-9aee-7551-9b31-c4c6d48a8adf

## Codex bottom line (verbatim)
> "This moves the work materially upward: from 'workshop / systems note' to **borderline MLSys main**, but I would not call it a clean main-track paper yet. The stretch result fixes the biggest weakness in the earlier story: D2-I5 was previously proxy-only, so the registered C2.main failure could look like 'interesting but unusable.' Now you have a production result showing: no strict-accuracy loss on the measured task, real end-to-end speedup, replay proxy pessimism of about 18 percentage points, and consistency with the prior Phase 4 v6 production result. ... My revised assessment: **not just workshop anymore; borderline main if expanded slightly and framed as proxy failure + production validation, not as a universal dropping algorithm.**"

## Recommended title (Codex's pick)
> **"When Router Mass Misleads: Production-Validated Expert Dropping for Faster MoE Prefill"**

## Codex 200-word abstract (verbatim)
> Mixture-of-Experts inference performance is often limited by distributed expert communication, but existing optimization signals can poorly predict task-level behavior. We study this gap using production measurements and offline replay for large-scale MoE serving. First, across all-to-all communication experiments, we find that received-token load, `L_recv`, is the dominant predictor of communication latency, achieving CV R² of 0.87, while a worst-peer hypothesis is not supported. Second, offline replay on prefill-dominated workloads shows that gate-triggered expert redundancy creates exploitable tail behavior: tail-weight dropping outperforms random dropping at matched communication budgets, although router-weight-mass loss predicts substantial quality risk. Finally, we validate a phase-gated production policy using LongBench `passage_retrieval_en`. Against a no-drop baseline, phase-gated tail-weight dropping preserves strict accuracy exactly over 64 samples, increases the official retrieval score from 0.257 to 0.383, improves prefill throughput by 1.232x, and improves end-to-end speed by 1.111x. We do not interpret the retrieval-score increase as a quality gain; rather, the result shows no measurable task-quality reduction under real serving. Notably, the offline router-mass proxy predicted 17.979% lost mass, while production strict-accuracy degradation was 0.000%, revealing substantial proxy pessimism. These results motivate task-calibrated MoE dropping policies and caution against using router mass as a standalone quality bound.

## To reach MLSys main (Codex's 5 must-haves)
1. **Scale production validation** from 64 to 256-512 samples with paired bootstrap CIs.
2. **Add 2-4 additional production tasks** (retrieval, QA, summarization, code/multi-doc) — map where proxy pessimism holds and where dropping becomes unsafe.
3. **Ablate the production policy**: no-drop, random matched-byte, tail_weight, phase-gated tail_weight, non-phase-gated tail_weight.
4. **Proxy-vs-real calibration figure**: x = router-mass loss, y = task-metric degradation — first-class deliverable.
5. **System-cost breakdown**: prefill vs decode, A2A bytes, L_recv, GPU util, expert load, e2e latency — ties D1-I10 and D2-I5 into one causal story.

Stretch for stronger main: replicate on second MoE model OR second serving stack.

## Codex's write-up time estimate
- Workshop-ready paper: **3-5 days**.
- Borderline MLSys main with current data + polish: **1-1.5 weeks**.
- Strong MLSys main with broader production validation: **2-4 weeks** (experiment-dominated).

## Codex's section layout (verbatim)
1. Introduction (MoE comm bottleneck + proxy overstates risk + prod-calibrated dropping accelerates prefill)
2. Background and Problem
3. D1-I10: What Predicts A2A Latency? (L_recv result, CV R² 0.87, worst-peer refutation)
4. D2-I5 Offline Replay (gate-trigger redundancy, tail vs random, C2.main proxy failure)
5. **Production Stretch Validation** (B0 vs B-PG main result table; zero strict-acc gap; 11.1% e2e; 23.2% prefill; no quality-loss claim)
6. **Proxy Pessimism Analysis** (router-mass 17.979% risk predicted vs 0.000% measured)
7. Ablations and Sensitivity
8. Limitations
9. Conclusion
