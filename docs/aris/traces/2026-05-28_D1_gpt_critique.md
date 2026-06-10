# Trace — Dimension 1 GPT Idea Critique

- Date: 2026-05-28
- Stage: A (idea generation), Dimension 1 — Communication-Centric MoE Framework
- Reviewer backend: codex (mcp__codex__codex)
- Reviewer model: as configured by Codex MCP (config reasoning_effort=high)
- Thread ID: 019e6e39-6e42-76c0-8d64-2eacde822d7a
- Rounds per idea: 1 (per ARIS config budget cap)
- Sandbox: read-only, approval-policy=never
- Sent ideas: D1-I5, D1-I4, D1-I6, D1-I10, D1-I2 (top 5 by Claude self-score)

## Prompt sent to GPT (summary)

Phase-A idea critique only. Strict rules: no experiments, no plans, no implementations,
no "scooped / not scooped" verdicts (only preliminary collision risk with concrete prior
work). User finding paste-in: 97% of MoE-drop e2e benefit from a2a payload reduction
(dispatch −47% + combine −50% bytes), 2.8% from expert GEMM, break-even L* ≈ 3271
rows/rank, prod config tail_weight@r=0.3 → prefill +23% / e2e +12% / accuracy zero loss.
For each of 5 ideas, asked for: strongest reviewer objection, likely failure mode,
mechanism-specified yes/partly/no, preliminary collision risk (≤2 closest papers + URL),
one sharp question.

## Raw GPT response

```
D1-I5: Payload-min-cut expert replica placement
  (a) Strongest objection: Objective underspecified — cross-GPU bytes are driven by
      token-source rank, replica selection, per-peer skew, and collective impl, not
      just expert co-activation. Reviewer may say min-cut graph is optimizing the
      wrong abstraction.
  (b) Likely failure mode: Placement reduces theoretical remote-expert hits but not
      critical-path A2A latency because bytes concentrate on a few rank pairs or
      create load imbalance that erases gains.
  (c) Mechanism specified? Partly. Min-cut + VRAM capacity is clear; replica routing,
      per-peer byte model, topology, load constraints are not.
  (d) Preliminary collision risk:
      - Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data
        Co-Scheduling, ICLR 2026, openreview MSHPrMpIHZ
      - Scaling Multi-Node Mixture-of-Experts Inference Using Expert Activation
        Patterns, arXiv 2604.23150
  (e) Sharp question: What exactly makes expert co-activation graph cuts equivalent
      to minimizing dispatch+combine bytes under real EP collectives?

D1-I4: Wire-byte budget runtime controller
  (a) Strongest objection: Single byte budget is not a quality-aware control target;
      two layers/requests with identical bytes can have very different accuracy
      sensitivity.
  (b) Likely failure mode: Controller oscillates or over-drops on semantically fragile
      layers because previous-layer L_recv is a lagging, shape-only proxy.
  (c) Mechanism specified? Partly. Controlled variable / action space named;
      controller law, stability criterion, and quality guard absent.
  (d) Preliminary collision risk:
      - Toward Efficient Inference for Mixture of Experts, NeurIPS 2024
      - Capacity-Aware Inference: Mitigating the Straggler Effect in MoE,
        arXiv 2503.05066 / ICLR 2026
  (e) Sharp question: Why should bytes alone be a sufficient runtime state variable
      for both latency and accuracy preservation?

D1-I6: Combine-asymmetric drop
  (a) Strongest objection: Combine-only drop pays dispatch and expert compute before
      saving bytes, so the latency claim is structurally weaker than symmetric drop
      unless combine is independently dominant.
  (b) Likely failure mode: Low router weight is not low contribution after the expert;
      dropped outputs have high norm or task-critical direction, causing worse
      accuracy than expected.
  (c) Mechanism specified? Partly. "Drop after expert compute" clear; post-expert
      importance criterion and aggregation semantics not specified.
  (d) Preliminary collision risk:
      - Not All Experts are Equal: Efficient Expert Pruning and Skipping for MoE
        LLMs, ACL 2024, arXiv 2402.14800
      - Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping
        Strategies and Observations, arXiv 2504.05586
  (e) Sharp question: What post-expert signal justifies dropping a contribution
      better than the pre-expert router score after dispatch+GEMM are already paid?

D1-I10: A2A-bytes as first-class MoE performance model
  (a) Strongest objection: "Bytes explain latency" may be a confounded characterization
      result, not a systems contribution; A2A latency depends on max per-peer bytes,
      imbalance, topology, launch overhead, overlap.
  (b) Likely failure mode: Model fits one Qwen3/4090/drop grid but breaks across
      decode, different EP layouts, NCCL/DeepEP backends, multi-node topology.
  (c) Mechanism specified? Partly. Feature and baselines clear; latency target,
      aggregation granularity, hardware/backend invariants not specified.
  (d) Preliminary collision risk:
      - Toward Efficient Inference for Mixture of Experts, NeurIPS 2024
      - MixServe: An Automatic Distributed Serving System for MoE Models with
        Hybrid Parallelism Based on Fused Communication Algorithm, arXiv 2601.08800
  (e) Sharp question: Is total wire bytes the causal feature, or merely a proxy for
      the true bottleneck feature: worst-rank/per-peer A2A pressure?

D1-I2: Cross-layer dispatch permutation reuse
  (a) Strongest objection: Reusing a dispatch permutation does not itself skip
      dispatch bytes — layer ℓ+1 has new hidden activations and layer-specific
      experts; combine returns tokens to owner rank.
  (b) Likely failure mode: Routing Jaccard overlap exists statistically but
      reusable cases do not align with rank locality or byte savings once
      combine/attention ordering is respected.
  (c) Mechanism specified? No. Does not specify how tokens legally remain on the
      right rank across layers or how new activations avoid being sent.
  (d) Preliminary collision risk:
      - Exploiting Inter-Layer Expert Affinity for Accelerating MoE Model Inference
        (ExFlow), IPDPS 2024, arXiv 2401.08383
      - Path-Constrained Mixture-of-Experts, arXiv 2603.18297
  (e) Sharp question: What exact tensor movement is eliminated if the next layer
      still needs fresh hidden states routed to its own experts?
```

## Claude independent verification of GPT-cited papers

All 8 papers GPT cited were independently web_fetched. All verified=YES (existence;
title; year; URL). See `02_idea_pool.md` literature references blocks for per-paper
results.

## Stage-A scope discipline

- No GPU experiments run.
- No source code modified.
- No experiment-bridge / run-experiment / experiment-queue calls.
- No "已撞车 / 没撞车" verdicts; only preliminary collision risk.
- No automatic continuation to Stage B.

## Codex usage estimate

- 1 substantive Codex round (5 ideas bundled in one prompt).
- 1 trivial Codex round for connectivity test (pre-task, "1+1").
- Total: 2 Codex calls; well under the 15-call alarm threshold.
