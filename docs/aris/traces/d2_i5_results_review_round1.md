# Trace — D2-I5 RESULTS round 1 (Codex attack)

- Date: 2026-05-30
- Reviewer: Codex (`mcp__codex__codex`), `model_reasoning_effort=xhigh`, sandbox=read-only
- Thread ID: 019e77fd-751d-7881-ab5e-91b1fe605671
- cwd: `/home/lzy/Artifact-Infer-pilot`

## What Codex was given
Full D2-I5 replay results across 27 policy points × 3 workloads + C2 mechanical verdicts (C2.main FAIL, C2.neg PARTIAL, C2.sample PASS). Asked: is FAIL the only honest reading; what's the right paper framing.

## Codex bottom line (verbatim)
> "C2.main FAIL is the only honest preregistered reading. The data cannot support 'L_recv gating strictly dominates static r=0.3.' It supports a narrower claim: on these long-prompt traces, all reasonable gates collapse to 'drop only on prefill,' and the real value comes from dropping low-weight/tail cells rather than random cells. That is paperable only if framed as a negative/mechanistic finding, not as a win for D2-I5's gate."

## Codex's key calls (responses in RESULTS.md)

1. **C2.main FAIL is the only honest reading.** Don't soften.
2. **Gate-trigger redundancy is both finding and failure.** Honest framing: "long-prompt workloads make prefill trivially separable, so gate location is over-determined."
3. **B-MR (tail_weight vs random at matched bytes) is the strongest result** — but TAIL_WEIGHT IS PARTIALLY OPTIMIZED FOR THE EVALUATION METRIC (weight_mass_loss). Claim "selection matters" is defensible; "tail_weight is the central contribution" is NOT yet defensible — need stronger selector baselines (rank-local tail, expert-local tail, activation-based, etc.) to escape the tautology.
4. **GSM8K 1pp threshold failure is legitimate**, but reveals a methodological artifact: per-step gate-open-rate is a poor safety metric when the few open steps are prefill steps carrying most mass. Suggest: byte-weighted or mass-weighted gate-open-rate as the right negative-control metric in future.
5. **Best title-level claim**: "In prefill-dominated MoE replay, drop selection matters more than gate trigger."
6. **What NOT to claim**: L_recv uniquely effective; strict dominance over B1; negative control passes; oracle agreement proves optimality.
7. **Salvage paths (out of scope for this pilot)**: mixed-length replay with intermediate L_recv; boundary stress around L*=3271; matched-byte-budget gate-location comparison; byte-weighted safety metric; stronger non-random selector baselines.

Verdict accepted by Claude. Round 2 will be a check on the RESULTS.md draft to ensure no overclaim slipped in.
