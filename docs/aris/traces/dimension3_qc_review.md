# Trace — Dimension 3 QC GPT Challenge

- Date: 2026-05-29
- Stage: A (idea pool QC), Dimension 3 — Workload-Adaptive MoE Scheduling
- Reviewer backend: codex (mcp__codex__codex), reasoning_effort=high
- Thread ID: 019e6f7c-9ff5-71b3-ad5c-7df83c2c4cbf
- Mode: senior MLSys reviewer challenges Claude's QC scores on top-3 by
  (novelty + story_fit)
- Sent ideas: D3-I2, D3-I3, D3-I8 (top 3)

## Prompt sent (summary)

QC challenge prompt: don't be polite, no experiments, no code. For each of three
D3 idea cards, answer (A) is novelty too high? closest prior?; (B) is story_fit
too high relative to the umbrella "a2a as first-class objective" thesis?;
(C) structurally unsafe bet?; (D) empty rhetorical move / heuristic dressed up?;
(E) distinguishable from a one-line patch to vLLM / SGLang / TensorRT-LLM /
Sarathi-Serve?. End with a loser pick.

## Raw GPT response

```
(A) Novelty challenge:
- D3-I2: Novelty too high. Closest priors: Capacity-Aware Inference (online
  dynamic drop control); Semantic Parallelism (weakens "early routing
  predicts later routing" angle).
- D3-I3: Novelty too high. Orca/Sarathi already make iteration-level scheduling
  the unit. This is mostly "also switch the MoE comm kernel at that unit".
- D3-I8: Novelty too high. Sarathi-Serve + Dynamic SplitFuse already own
  chunked-prefill adaptation space. "Target MoE break-even" is a retargeted
  heuristic, not a new mechanism.

(B) Story_fit challenge:
- D3-I2: Story fit somewhat high but not absurd. Uses L_recv and break-even
  directly, but still a feedback controller for a drop knob, not a serving-
  system reframing.
- D3-I3: Best story fit of the three. Makes byte volume the runtime decision
  variable and keeps accuracy invariant. Thin, but on-thesis.
- D3-I8: Story fit too high. Chunk sizing is mostly scheduling/stall
  management. The MoE byte thesis is attached to an existing scheduler knob,
  not made first-class.

(C) Structurally unsafe bets:
- D3-I2: Unsafe. Mean prediction does not protect layer tails, expert skew, or
  request-level regressions. "Degenerates to static" also means upside can
  vanish.
- D3-I3: Structurally safer. If byte-identical outputs are real, claim can
  deliver. Risk: ≥5% depends entirely on heterogeneity and static-baseline
  weakness.
- D3-I8: Unsafe. Hitting [L*, sweet_spot] can still lose if stall-free
  constraints, KV pressure, batching fairness, or TTFT dominate. Confirming
  the chunk target does not prove serving benefit.

(D) Empty rhetorical move?
- D3-I2: Partly rhetorical. "Online L_recv feedback" can collapse into a
  threshold heuristic with confidence gating.
- D3-I3: Least empty. Has a concrete invariant: switch only MoE comm path,
  preserve router/top-K/expert outputs.
- D3-I8: Most rhetorical. "Make chunk size adaptive to MoE break-even" is a
  scheduler heuristic dressed as a thesis shift.

(E) Distinguishable from engine patch?
- D3-I2: Hard to distinguish unless it shows nontrivial stability logic across
  layers and workloads. Otherwise: measure early L_recv, flip policy.
- D3-I3: Barely distinguishable, but strongest candidate: microbatch-level
  comm-path switching could be a real systems primitive if integrated cleanly.
- D3-I8: Not distinguishable enough. Looks like one more adaptive chunk-size
  rule in Sarathi-Serve/SGLang/vLLM.

LOSER: D3-I8. Existing chunked-prefill scheduler with a MoE-colored target
metric. Mechanism generic, thesis fit bolted on, success can be explained
without buying the "a2a as first-class objective" claim.
```

## Claude independent verification of GPT-cited papers

GPT introduced no new papers beyond those already verified in earlier traces:
- Capacity-Aware Inference (arXiv:2503.05066) — verified D1.
- Semantic Parallelism (arXiv:2503.04398) — verified D1.
- Orca (OSDI'22) — verified D3 trace via web search (USENIX page).
- SARATHI (arXiv:2308.16369), Sarathi-Serve (arXiv:2403.02310) — verified D3.
- DeepSpeed-FastGen / Dynamic SplitFuse (arXiv:2401.08671) — verified D3.

No new web_fetch needed.

## Claude reaction to GPT QC

- **D3-I3**: GPT confirmed it is the strongest candidate of the three —
  "best story fit", "least empty", "structurally safer", "barest distinguishable
  but real systems primitive". Accept novelty downgrade 3.5 → 3 (Orca/Sarathi
  own iteration-level scheduling); keep story_fit at 5 (GPT independently said
  "best story fit"). Becomes new top 1.
- **D3-I2**: Accept novelty downgrade 4 → 3.5 (Capacity-Aware Inference closer
  than initially assessed) and story_fit downgrade 5 → 4.5 (GPT: "still a
  feedback controller for a drop knob, not a serving-system reframing"). The
  "mean prediction doesn't protect tails" objection is real and the mitigation
  must be explicit: extend the bet to commit to per-layer tail variance
  monitoring on top of mean, not just confidence gating on mean. Still PASS.
- **D3-I8**: Substantial downgrade. Accept novelty 4 → 3 and story_fit 5 → 3
  per GPT framing ("most rhetorical", "MoE thesis bolted on", "not
  distinguishable enough"). The contribution becomes "MoE byte break-even
  reframes Sarathi chunk objective" — that's a comparative finding, not a new
  mechanism. **Downgrade to PASS-WEAK**; explicitly note that the strongest
  publishable form is a Sarathi-extension empirical study rather than a new
  serving system.

**Score shuffle after GPT**: D3-I3 (8) becomes #1; D3-I2 (8) #2; D3-I8 (6)
falls from tied-#1 to mid-pack. The top-3 reshuffles to D3-I3 → D3-I2 → D3-I5
(7), with D3-I8 becoming a borderline candidate alongside D3-I4.

## Stage-A scope discipline

- No experiments, no source edits, no plan-mode transition.
- No "已撞车 / 没撞车" verdicts in QC; only preliminary collision risk.

## Codex usage estimate (cumulative)

- D1 idea critique: 1 call.
- D2 idea critique: 1 call.
- D1 QC challenge: 1 call.
- D2 QC challenge: 1 call.
- D3 idea critique: 1 call.
- D3 QC challenge: 1 call.
- Connectivity ping: 1 call.
- Total: 6 substantive + 1 ping. User lifted budget cap mid-session; the
  "top 3 / 1 round each" protocol was kept since it is a content rule, not a
  budget rule.
