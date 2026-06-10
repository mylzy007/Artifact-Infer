# Trace — Dimension 2 GPT Idea Critique

- Date: 2026-05-28
- Stage: A (idea generation), Dimension 2 — Phase-Asymmetric MoE Serving
- Reviewer backend: codex (mcp__codex__codex), reasoning_effort=high
- Thread ID: 019e6e5e-624d-7ff0-a48a-80fbf8638618
- Rounds per idea: 1 (per ARIS config budget cap)
- Sent ideas: D2-I5, D2-I2, D2-I3, D2-I9, D2-I4 (top 5 by Claude self-score)

## Prompt sent to GPT (summary)

Phase-A idea critique only. Strict rules: no experiments, no plans, no implementations,
no scoop verdicts (only preliminary collision risk with concrete prior work). User
finding paste-in: 97% / 2.8% bytes-vs-GEMM split; L_recv asymmetry between phases is
100×; decode L_recv ≈ 8 explains all Phase-4 ±2% e2e results. Required for each idea:
(a) is it just DistServe/SplitWise-for-MoE? (b) is it just an engineering policy
switch? (c) mechanism specified+falsifiable yes/partly/no, (d) collision risk (≤2
papers + URL), (e) strongest reviewer objection, (f) sharp question.

## Raw GPT response

```
D2-I5: L_recv-Gated Continuous Phase Detection
  (a) No — layer-local within one EP instance, vs DistServe/SplitWise resource-pool split.
  (b) Partial — threshold switch, but using measured L_recv_l instead of phase label
      is a real causal claim.
  (c) Yes — L_recv_l > L* gives crisp on/off; weak point: whether one L* is stable
      across layers, skew, overlap, hardware.
  (d) Collision risk:
      - DuoServe-MoE, https://arxiv.org/abs/2509.07379 — different MoE expert
        scheduling for prefill and decode.
      - SMIDT (AAAI'26 / 39403) — automates MoE inference config; treats prefill/decode
        performance separately.
  (e) Strongest objection: L*≈3271 may be hardware/runtime artifact, not structural.
  (f) Sharp question: what makes L_recv alone sufficient when routing skew, peer
      imbalance, kernel fusion, overlap also shift the break-even?

D2-I2: Phase-Specific Expert Replica Budget
  (a) No — varies MoE replica + KV capacity in one instance, not phase disaggregation.
  (b) Partial — "512 prefill / 128 decode" sounds like a config switch unless memory
      fungibility / contention argument is central.
  (c) Partly — replica-count states named; mixed prefill/decode batch assignment is
      not specified.
  (d) Collision risk:
      - CRAFT, https://arxiv.org/abs/2603.28768 — replica allocation with layerwise
        cost estimates.
      - SMIDT (AAAI'26 / 39403) — DPMoE component is memory-aware replication.
  (e) Strongest objection: replicas are global shared resources; per-session decode
      reclamation may conflict with concurrent prefills; freed VRAM may not become
      usable KV without allocator/residency costs.
  (f) Sharp question: in a mixed workload, whose phase controls the replica budget
      for a layer simultaneously serving prefill and decode traffic?

D2-I3: Phase-Asymmetric Top-K
  (a) No — changes model's effective routing per phase, not infra layout.
  (b) No — model-system semantic tradeoff, not infra policy.
  (c) Partly — K_decode=4 / K_prefill=8 clear; renormalization, router calibration,
      accuracy-delta definition underspecified.
  (d) Collision risk:
      - SMIDT (AAAI'26 / 39403) — dynamic Top-K in title and mechanism.
      - SERE, https://arxiv.org/abs/2602.07616 — similarity-based expert re-routing
        for batch decoding in MoE.
  (e) Strongest objection: if global K reduction was negative-leverage, claiming
      decode-only K reduction is accuracy-safe needs stronger model-side reason than
      launch overhead.
  (f) Sharp question: why are experts ranked 5–8 dispensable during decode when
      decode logits directly determine generated content?

D2-I9: Phase-Transition Memory Rebalancing
  (a) No — intra-instance replica/KV reallocation, not separate engines.
  (b) Partial — freeing cold replicas is cache/offload policy unless trajectory
      predictability is precise.
  (c) Partly — transition point clear; "unlikely activated" and miss handling
      underspecified.
  (d) Collision risk:
      - MoE-Infinity, https://arxiv.org/abs/2401.14361 — sequence-level activation
        tracing + expert offloading for MoE serving.
      - DuoServe-MoE, https://arxiv.org/abs/2509.07379 — dual-phase MoE expert
        prefetch/cache scheduling.
  (e) Strongest objection: future decode expert use is not knowable at transition;
      a cold-expert miss either hurts latency via reload or changes behavior.
  (f) Sharp question: if a freed expert replica is routed later, is the token served
      remotely, stalled for reload, or approximated?

D2-I4: Phase-Asymmetric Expert Placement
  (a) No — one model instance, switches MoE placement metadata, not resource-pool
      separation.
  (b) Partial — two plans are more than a switch only if prefill byte objective and
      decode remote-hit objective provably select different layouts.
  (c) Partly — two objectives stated; "pointer-only re-tagging" assumes replicas
      already exist and mixed-phase batches do not need both plans simultaneously.
  (d) Collision risk:
      - Cluster Topology-Driven Placement of Experts Reduces Network Traffic in MoE
        Inference, https://arxiv.org/abs/2508.09229.
      - MoETuner, https://arxiv.org/abs/2502.06643.
  (e) Strongest objection: decode "remote-rank hits" still go through collective /
      a2a paths where bytes, sync, launch overhead are not separable.
  (f) Sharp question: under the same replica budget, what concrete routing
      distribution makes the byte-minimizing placement and hit-minimizing placement
      diverge?
```

## Claude independent verification of GPT-cited papers

- DuoServe-MoE (arXiv:2509.07379) — verified=YES (independently web_fetched).
- SMIDT (AAAI-26 / 39403, https://ojs.aaai.org/index.php/AAAI/article/view/39403) —
  verified=YES (independently web_fetched; AAAI-26 record present).
- CRAFT (arXiv:2603.28768) — verified=YES (independently web_fetched).
- SERE (arXiv:2602.07616) — verified=YES (independently web_fetched).
- MoE-Infinity (arXiv:2401.14361) — verified=YES (paper exists; actual title is
  "Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache",
  GPT-rendered title was inexact).
- Cluster Topology-Driven Placement (arXiv:2508.09229) — verified=YES.
- MoETuner (arXiv:2502.06643) — verified=YES (D1 trace already verified).

## Stage-A scope discipline

- No GPU experiments, no source edits, no experiment-bridge/run-experiment calls.
- No "已撞车 / 没撞车" verdicts; only preliminary collision risk.
- No automatic Stage-B continuation.

## Codex usage estimate (cumulative across dimensions)

- D1: 1 substantive call + 1 connectivity ping = 2 calls.
- D2: 1 substantive call. Cumulative: 3 substantive + 1 ping. Well below 15-call alarm.
