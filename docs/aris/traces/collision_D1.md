# Trace — Dimension 1 Collision GPT QC

- Date: 2026-05-29
- Stage: Collision check (post-Stage-A QC)
- Reviewer backend: codex (mcp__codex__codex), reasoning_effort=high
- Thread ID: 019e6fbb-ee39-7ce0-acff-2413059c2cc2
- Sent ideas: D1-I7 (rated 完全撞 by Claude), D1-I8 (rated 接近-高风险 by Claude)

## Why these two

Claude's pre-QC collision rating ranked D1-I7 and D1-I8 as the two highest-risk
ideas in D1 (rest are 接近-差异 or 互补). GPT was asked specifically: name
priors Claude likely missed; is collision rating too optimistic?

## GPT raw response

```
D1-I7 missed priors:
- Adaptive Gating in MoE-based LMs, EMNLP 2023 / arXiv:2310.07188
- DA-MoE: Towards Dynamic Expert Allocation for MoE Models, arXiv:2409.06669
- Sparsity-Controllable Dynamic Top-p MoE, arXiv:2512.13996
Claude's rating too optimistic? NO — "完全撞" already fatal.
Differentiation: only as systems-side byte-ROI controller action inside D1-I4.

D1-I8 missed priors:
- LongCat-Flash Technical Report (ScMoE + SBO), arXiv:2509.01322 — explicit
  shortcut-connected MoE for inference overlap window.
- EPS-MoE: Expert Pipeline Scheduler, arXiv:2410.12247
- FlashDMoE: Fast Distributed MoE in a Single Kernel, arXiv:2506.04667
Claude's rating too optimistic? YES — LongCat-Flash ScMoE makes inference
cross-layer angle "much less thin; already claimed in near-explicit form".
Differentiation: only if restricted to unmodified vanilla Qwen3 EP=8 software
scheduling with formal dependency-safe schedule.
```

## Claude independent verification of GPT-cited papers

All 6 priors independently web_fetched:
- Adaptive Gating (Li et al., 2023, arXiv:2310.07188) — verified=YES.
- DA-MoE (Akhavan Aghdam et al., 2024, arXiv:2409.06669) — verified=YES.
- DTop-p MoE / Sparsity-Controllable (Jin et al., 2025, arXiv:2512.13996) — verified=YES.
- LongCat-Flash Technical Report (Meituan, 2025, arXiv:2509.01322) — verified=YES.
- EPS-MoE (Qian et al., 2024, arXiv:2410.12247) — verified=YES.
- FlashMoE / FlashDMoE (Aimuyo et al., 2025, arXiv:2506.04667) — verified=YES.

Zero verification failures.

## Claude verdict update post-GPT

- **D1-I7**: collision rating upgraded from "完全撞 / 接近" → "完全撞" with explicit
  3 missed priors. Conclusion strengthened from "放弃 / 降级" → "放弃" (only
  salvage = action in D1-I4 controller table).
- **D1-I8**: collision rating upgraded from "接近但有差异 (高风险)" → "完全撞"
  due to LongCat-Flash ScMoE explicit claim. Conclusion strengthened from
  "放弃 / 降级" → "放弃" (remove from D1 candidate set entirely).

## Codex usage estimate

- 1 substantive collision QC call this dimension. Cumulative across all
  dimensions = 11 substantive + 1 ping.
