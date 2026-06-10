# Trace — D2-I5 RESULTS round 2 (Codex review of draft)

- Date: 2026-05-30
- Reviewer: Codex (`mcp__codex__codex`), `model_reasoning_effort=xhigh`
- Thread ID: 019e7800-d3f4-7163-82f6-6442218d4bdb

Round 2 asked Codex to check the RESULTS.md draft for any overclaim that slipped in past the round 1 reframing. Codex returned 12 specific issues; all 12 were accepted by Claude.

## 12 fixes Codex demanded → applied:

1. **TL;DR C2.neg "methodological artifact"** softens FAIL → rewritten as "bound is confounded by rare high-mass prefill records".
2. **Title claim** "drop selection matters more than gate trigger" too universal → narrowed to "In these offline, prefill-dominated traces and tested policy family, once gates collapse to prefill-only behavior, tail-weight selection preserves more router mass than random at matched bytes."
3. **"the only differentiator"** overclaims → "the main observed aggregate differentiator in this suite".
4. **Tail_weight tautology** — frame as metric-aligned selection benefit vs random, NOT algorithmic superiority.
5. **"Any other reasonable gate"** → "the tested alternative gates" (applied throughout).
6. **"All reduce to drop only on prefill"** implies mechanistic equivalence → "are aggregate-indistinguishable because droppable volume is overwhelmingly prefill".
7. **C2.main FAIL prominence** → final verdict opens with "The pre-registered dominance claim fails."
8. **Honest Limitations missing uncertainty** → added (#12) "headline numbers are point estimates; full sensitivity analysis is future work".
9. **Honest Limitations missing weak-baseline** → added (#7) "Random is a weak selector baseline".
10. **Tautology vs proxy-quality** separated into (#6) and (#8).
11. **Generalization boundary** added (#11).
12. **"Correct metric"** softened → "necessary companion metric, does not erase the failure".
13. (Bonus) added "What would validate the paper claim" section (4 items: interventional/online replay, stronger selectors, downstream quality, non-prefill workloads).
