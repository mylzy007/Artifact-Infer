# Trace — Dimension 3 GPT Idea Critique

- Date: 2026-05-29
- Stage: A (idea generation), Dimension 3 — Workload-Adaptive MoE Scheduling
- Reviewer backend: codex (mcp__codex__codex), reasoning_effort=high
- Thread ID: 019e6f6f-5bd9-7f00-a111-506c8e741e74
- Rounds per idea: 1 (per ARIS config; user lifted budget cap mid-session but
  task spec retained "top 5 / 1 round each" protocol)
- Sent ideas: D3-I2, D3-I3, D3-I4, D3-I5, D3-I8 (top 5 by Claude self-score)

## Prompt sent (summary)

Phase-A idea critique. Strict rules: no experiments, no plans, no implementations,
no collision verdicts. User finding paste-in: 97% / 2.8% bytes-vs-GEMM split;
long-prompt+short-decode +12% e2e via drop; short-prompt+long-decode zero benefit
(decode L_recv ≈ 8). D3 thesis: workload-adaptive runtime MoE policy at request
or microbatch granularity, distinct from DistServe/SplitWise (cross-instance) and
distinct from D2 (within-request phase). For each idea: (a) just if-else
heuristic?; (b) is mechanism systems-paper grade?; (c) distinguishable from
vLLM/SGLang/TensorRT-LLM/Sarathi-Serve/DistServe?; (d) collision risk (≤2 papers
+ URL); (e) strongest objection; (f) sharp question.

## Raw GPT response

```
D3-I2: Online L_recv Early-Layer Feedback
  (a) Partial — boundary-crossing on L_recv > L* is basically if-else unless
      the early-layer signal is formalized as a calibrated predictor of later-
      layer communication.
  (b) Partly — inputs/knobs are clear; mapping is still hand-waved unless
      "first k layers predict remaining L_recv envelope" has a principled
      error model.
  (c) Partly distinguishable — MoE-specific feedback at layer granularity is not
      plain vLLM/Sarathi scheduling, but "switch a kernel/drop flag mid-request"
      can collapse into an engine knob.
  (d) Collision risk:
      - Speculative MoE: Communication Efficient Parallel MoE Inference with
        Speculative Token and Expert Pre-scheduling, arXiv:2503.04398
      - Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of
        Experts, openreview LuYFpySWA2
  (e) Strongest objection: early-layer L_recv may be a weak or unstable predictor
      across layers/prompts/phases; reviewers may call the switch opportunistic.
  (f) Sharp question: what invariant makes first-2/3-layer L_recv predictive
      enough that a wrong switch cannot silently damage quality or latency?

D3-I3: Continuous-Batching Microbatch-Level MoE Policy
  (a) Partial — f(total prefill tokens + decode tokens) risks looking like a
      threshold heuristic, but microbatch-level control is a legitimate systems
      unit.
  (b) Partly — composition/decisions clear; mapping needs more than "deterministic
      function".
  (c) High collision with existing engines; reads as "add MoE-aware per-step
      policy to vLLM/SGLang/TensorRT-LLM" unless MoE byte model is central.
  (d) Collision risk:
      - Orca: A Distributed Serving System for Transformer-Based Generative
        Models, OSDI 2022, USENIX osdi22-yu.pdf
      - SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked
        Prefills, arXiv:2308.16369
  (e) Strongest objection: microbatch policy may optimize the step but violate
      per-request consistency when different requests in the same batch want
      different MoE behavior.
  (f) Sharp question: is the decision optimizing MoE communication bytes,
      per-request latency, or global batch throughput when those disagree?

D3-I4: Workload-Conditioned Elastic Replica Pool
  (a) No if replica/KV tradeoff is real-time resource allocation; partial if
      "long prompt gets replicas, decode gets KV" is the whole mechanism.
  (b) Partly — "spawn/release replicas" is expensive and underspecified unless
      replica materialization/remapping cost is modeled.
  (c) Partly distinguishable — MoE expert replicas are specific, but elastic
      GPU/KV resource pooling is crowded; may collapse into MoE-aware memory
      manager.
  (d) Collision risk:
      - ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts
        Models, arXiv:2510.02613
      - MemServe: Context Caching for Disaggregated LLM Serving with Elastic
        Memory Pool, arXiv:2406.17565
  (e) Strongest objection: replica elasticity at per-session timescale may be
      too slow or disruptive on 4090-class PCIe systems to matter online.
  (f) Sharp question: what is the unit of elasticity — physical weight copy,
      logical routing to an existing replica, or remapped HBM reservation?

D3-I5: MoE-Aware Admission Control with Byte-Budget SLO
  (a) Partial — admission control is classic if it is just "queue when bytes
      exceed threshold"; MoE-specific worst-peer bytes gives it a sharper
      resource metric.
  (b) Partly — p99 mapping from byte budget to latency margin is hand-waved.
  (c) Mostly collapses unless framed as a new MoE network/EP pressure model;
      vLLM/SGLang front-door admission already has natural hooks.
  (d) Collision risk:
      - SCORPIO: Serving the Right Requests at the Right Time for Heterogeneous
        SLOs in LLM Inference, arXiv:2505.23022
      - DistServe (arXiv:2401.09670)
  (e) Strongest objection: "byte budget" may be a proxy, not a control law;
      p99 latency is affected by queueing, KV pressure, routing skew, kernel
      contention.
  (f) Sharp question: why is worst-peer MoE wire bytes the admission bottleneck
      rather than a feature inside a broader latency predictor?

D3-I8: Adaptive Chunked-Prefill Chunk Size Targeting MoE Break-Even
  (a) Partial — targeting [L*, sweet_spot] is threshold tuning, but tying chunk
      size to MoE a2a break-even is a concrete systems angle.
  (b) Partly — "per request per layer" chunk sizing may be too complex unless
      constraints are cleanly defined.
  (c) Partly distinguishable — close to Sarathi-Serve/vLLM chunked prefill;
      distinct only if MoE L_recv break-even, not generic stall-free scheduling,
      drives the design.
  (d) Collision risk:
      - Sarathi-Serve, arXiv:2403.02310
      - DeepSpeed-FastGen with Dynamic SplitFuse, arXiv:2401.08671
  (e) Strongest objection: chunk size changes L_recv indirectly and may fight
      continuous batching constraints; reviewers may see this as "tuning Sarathi
      for MoE".
  (f) Sharp question: is L_recv ≈ 4k stable across layers and routing skew, or
      does the chunk controller chase a moving target?
```

## Claude independent verification of GPT-cited papers

- Speculative MoE / Semantic Parallelism (arXiv:2503.04398) — verified=YES via
  D1 trace. Note: GPT used title "Speculative MoE"; the paper at that URL is
  actually "Semantic Parallelism: Redefining Efficient MoE Inference via
  Model-Data Co-Scheduling" — same paper, different title rendering.
- Capacity-Aware Inference (arXiv:2503.05066) — verified=YES via D1 trace.
- Orca, OSDI 2022 — verified=YES via web search (USENIX OSDI22 page + multiple
  reading-list pages confirm Yu et al. 2022).
- SARATHI / Sarathi-Serve (arXiv:2308.16369 / 2403.02310) — verified=YES.
- ElasticMoE (arXiv:2510.02613) — verified=YES (independently web_fetched).
- MemServe (arXiv:2406.17565) — verified=YES (independently web_fetched).
- SCORPIO (arXiv:2505.23022) — verified=YES (independently web_fetched).
- DistServe (arXiv:2401.09670) — verified=YES via D2 trace.
- DeepSpeed-FastGen (arXiv:2401.08671) — verified=YES (independently web_fetched).

## Codex usage estimate (cumulative across all dimensions)

- D1 idea critique: 1 call.
- D2 idea critique: 1 call.
- D1 QC challenge: 1 call.
- D2 QC challenge: 1 call.
- D3 idea critique: 1 call.
- Connectivity ping: 1 call.
- Total: 5 substantive + 1 ping. Even after the user lifted the budget cap
  mid-session, the original "top 5 / 1 round each" protocol was kept since it
  is a content-level rule, not just a budget rule.
