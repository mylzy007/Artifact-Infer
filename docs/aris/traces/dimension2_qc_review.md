# Trace — Dimension 2 QC GPT Challenge

- Date: 2026-05-29
- Stage: A (idea pool QC), Dimension 2 — Phase-Asymmetric MoE Serving
- Reviewer backend: codex (mcp__codex__codex), reasoning_effort=high
- Thread ID: 019e6f5c-4492-7b90-94f8-31eb0cb25a30
- Mode: senior MLSys reviewer challenges Claude's QC scores on top-3 by (novelty + story_fit)
- Sent ideas: D2-I5, D2-I2, D2-I3 (top 3)

## Prompt sent (summary)

QC challenge prompt: don't be polite, no experiments, no code. For each of three D2
idea cards (D2-I5/I2/I3), answer (A) is novelty too high? closest prior?; (B) is
story_fit too high? does it advance phase-aware POLICY or just adjacent
engineering/scheduling/characterization?; (C) is the one-sentence bet structurally
unsafe — even if all experiments confirm, does the idea fail?; (D) is it secretly
an empty rhetorical move?; (E) is it distinguishable from "just DistServe-for-MoE
with extra mechanism"?. End with a "loser" pick.

## Raw GPT response

```
(A) Novelty challenge:
- D2-I5: Novelty slightly high. Closest priors: SMIDT-style dynamic MoE policy +
  DuoServe-MoE dual-phase handling. But L_recv > L* as control variable is a
  real hook.
- D2-I2: Novelty too high. Closest priors: CRAFT/DPMoE memory-aware replica
  allocation + MoE-Infinity expert caching. "Shadow replica" is mostly cache
  residency language.
- D2-I3: Novelty too high. Closest priors: SMIDT dynamic Top-K + decode-side
  routing-change work like SERE / Opportunistic Expert Activation.
  Phase-static K_decode<K_prefill is thinner than it sounds.

(B) Story_fit challenge:
- D2-I5: Story fit real. Directly converts the empirical break-even into a
  phase-aware policy mechanism inside one instance.
- D2-I2: Story fit too high. This is mostly memory management / expert cache
  residency. Phase is used as a justification, not the core policy result.
- D2-I3: Story fit decent but narrow. Advances one phase-aware knob (K_eff),
  but risks becoming "decode approximation works" rather than "phase-aware
  serving policy".

(C) Structurally unsafe bets:
- D2-I5: Structurally mostly safe. If it wins on chunked/mixed workloads, it
  actually delivers something binary phase tags cannot.
- D2-I2: Structurally unsafe. Even if 15% VRAM is reclaimable, "yields ≥1
  additional concurrent decode session" can fail due to KV fragmentation,
  scheduler limits, batch shape, or latency SLOs.
- D2-I3: Structurally unsafe. GSM8K accuracy + softmax mass does not prove
  general serving validity; it can win latency and still fail as a
  model-quality policy.

(D) Empty rhetorical move?
- D2-I5: NOT empty. Replaces phase rhetoric with a measurable gate.
- D2-I2: Borderline empty. "Mark replicas decode-cold" can collapse into
  "we run two static memory states" unless there is real online arbitration.
- D2-I3: Borderline empty. "K=8 prefill, K=4 decode" is just two static configs
  unless tied tightly to the launch-bound decode diagnosis.

(E) Distinguishable from DistServe-for-MoE?
- D2-I5: Distinguishable. Intra-instance, per-layer, per-step, payload-threshold
  driven.
- D2-I2: Weakly distinguishable. Looks like DistServe-for-MoE + expert-cache
  paging unless the same live instance continuously trades replica residency
  against KV pages.
- D2-I3: Somewhat distinguishable, but not by much. Phase-specific approximation
  knob, not a serving architecture.

LOSER: D2-I2 is the weakest. Most prior overlap, weakest policy story, most
dangerous claim chain: cold replicas → reclaimable VRAM → allocator usability →
extra decode session. Too many places for the thesis to be technically true and
still not matter.
```

## Claude independent verification of GPT-cited papers

GPT did not introduce any new papers beyond those already verified in earlier
traces (`2026-05-28_D1_gpt_critique.md`, `2026-05-28_D2_gpt_critique.md`,
`dimension1_qc_review.md`). All cited priors (SMIDT, DuoServe-MoE, CRAFT,
MoE-Infinity, SERE, Opportunistic Expert Activation, Nexus) have prior
`verified=YES` records. No new web_fetch needed.

## Claude reaction to GPT QC

- **D2-I5**: agree on novelty downgrade (4 → 3.5). Story_fit=5 confirmed by GPT
  as "real". Structurally safe per GPT. **Keep as #1.**
- **D2-I2**: agree on both downgrades — novelty (3.5 → 3) and story_fit (4.5 →
  3.5). GPT picked it as loser; I accept the framing: shadow-replica IS mostly
  cache residency language, and the "extra concurrent decode session" claim
  chain is fragile. Downgrade to **PASS-WEAK** with explicit "empty rhetoric
  risk" flag; the only way to revive it as a research contribution (vs CRAFT /
  MoE-Infinity) is to commit to *online arbitration* between replica residency
  and KV pages as the central mechanism, not just "mark cold at transition".
- **D2-I3**: agree on novelty downgrade (3 → 2.5). On story_fit (4.5 → 4):
  partial agreement — "narrow" is fair; the contribution must be packaged as
  *the launch-bound diagnosis applied to K*, not as a generic decode
  approximation. Keep as PASS but with the framing risk noted.

**Score shuffle after GPT**: D2-I5 stays clear top 1. D2-I2 falls out of top 3
(6.5). D2-I3 also 6.5. The unreviewed D2-I4 (placement bi-modal, 7.5) and D2-I8
(re-co-location, 7.5) are now higher than I2/I3 by composite. Honestly report:
the GPT challenge re-shuffled top 2-3 to be D2-I5 → D2-I4/I8 → D2-I3, with I2
demoted.

## Stage-A scope discipline

- No experiments, no source edits, no plan-mode transition.
- No "已撞车 / 没撞车" verdicts in QC; only preliminary collision risk.

## Codex usage estimate (cumulative)

- D1 idea critique: 1 call.
- D2 idea critique: 1 call.
- D1 QC challenge: 1 call.
- D2 QC challenge: 1 call.
- Connectivity ping: 1 call.
- Total: 4 substantive + 1 ping. Still well below the 15-call alarm.
