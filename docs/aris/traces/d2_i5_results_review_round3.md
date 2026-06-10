# Trace — D2-I5 RESULTS round 3 (Codex convergence check)

- Date: 2026-05-30
- Reviewer: Codex (`mcp__codex__codex`), `model_reasoning_effort=xhigh`
- Thread ID: 019e7803-34a1-7b93-8b6a-72eadc85c7c2

Round 3 asked Codex to verify all 12 round-2 fixes landed and check for remaining overclaim.

## Codex round 3 verdict
> "Convergence call: not done yet, but close. I would make the small wording fixes above plus reconcile the C2.neg record-count explanation. After that, I'd call the framing done; no further structural/material edits needed."

## 5 remaining issues Codex flagged → all applied

1. **Old C2.neg "Methodological artifact" / "correct safety metric" language reappeared at line 54** — rewritten to match new framing ("necessary companion metric ... does not erase the failure").
2. **Numeric inconsistency**: 0.44% × 49,536 ≈ 218 records ≠ "3 prefill steps' worth" (1,152) → rewrote as "a subset of the high-mass prefill records ... less than one full prefill step's worth of 384 (rank, layer) combos".
3. **"selection-dominates" heading + "attributable purely" + "1.7× quality" + "selection mechanism is robust"** — minor handles → softened to "matched-bytes selector benefit vs random" / "1.7× lower router-mass loss" / "router-mass gap is stable in this bootstrap".
4. **"decode-heavy uncovered"** contradicts GSM8K being decode-heavy → "other decode-heavy workloads beyond GSM8K".
5. **"Establish algorithmic dominance"** reintroduces tautology trap → "test whether the router-mass advantage survives stronger selector baselines".

All 5 applied. Round 4 will confirm convergence.
