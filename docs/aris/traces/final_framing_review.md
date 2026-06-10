# Trace — Final Framing Review (Joint D1-I10 + D2-I5)

- Date: 2026-05-30
- Reviewer: Codex (`mcp__codex__codex`), `model_reasoning_effort=xhigh`
- Thread ID: 019e780b-46f4-7590-a927-1d534242779b

## Codex bottom line (verbatim)
> "This is **not MLSys main-track ready** as an algorithmic systems paper. Both preregistered main claims failed, the strongest optimization claim is offline/proxy-based, and generalization is limited. It **is publishable as a scoped measurement/artifact paper**: MLSys workshop, efficient ML systems workshop, systems-for-LLMs workshop, or arXiv artifact report. The right contribution is not 'we propose a better MoE drop policy'; it is 'we falsify a plausible byte-centric hypothesis and identify what actually predicts latency and why common gating evaluations can be misleading.'"

## Recommended title-level claim
> **"Received-token load, not worst-peer bytes, predicts MoE all-to-all latency; prefill-dominated traces make many drop gates equivalent."**

## Venue assessment
- **MLSys main: NO** unless expanded substantially (need online validation + task accuracy + workload diversity + cross-system replication).
- **MLSys workshop / Efficient ML systems workshop / Systems-for-LLMs workshop: YES** — strong fit as measurement + negative-results + artifact paper.
- **arXiv**: YES, with trace schema + recorder + validation appendix + pre-registration table.

## What would move it toward tier-1 (Codex priority order)
1. **Stretch: LongBench strict_accuracy** — DO IT. Directly addresses the "router mass is only a proxy" objection. Either outcome strengthens the paper.
2. Online latency validation (real serving with the policies).
3. Decode-heavy or mixed workload to test whether gate redundancy disappears.
4. Cross-system / cross-model replication.
5. Non-tautological selector baselines beyond random (entropy, top-1 margin, layer-aware).
6. L_recv-driven online controller (only after the above shows quality+latency signal worth optimizing).

## Codex's recommendation on stretch goal: DO IT
> "Even though C2.main failed, this is still worth it because it directly addresses the biggest reviewer objection: 'router mass is only a proxy.' If EXP/static/tail_weight are accuracy-equivalent, that supports the methodological framing. If accuracy diverges despite similar mass/bytes, that is an even stronger finding: the proxy is insufficient. Either outcome improves the paper."

## Claude note: stretch goal complication
The original stretch as described in PLAN requires running EXP (L_recv-gated drop) in production inference. Reviewing dispatch_ep_ht.py: the L_recv (`total_recv` at line 383) is computed AFTER the drop decision (line 299). Implementing online EXP would require non-trivial code restructuring, which violates Stage D2/D3 "严禁修改 dispatch/combine/expert/drop 算法代码" constraint.

**Workaround**: substitute B-PG (phase-gated tail_weight r=0.3) as proxy for EXP. The D2-I5 replay showed B-PG ≈ EXP on aggregate metrics in this distribution (gate-trigger redundancy), so B-PG online would be a faithful stand-in. B-PG requires only harness-layer changes (turn drop_policy on/off based on `Context.is_prefill` between batches), no dispatcher edits.

User decision needed at GATE D3: run stretch with B-PG substitution (≤ 2h GPU, no algorithm-code changes) OR skip stretch and accept the proxy limitation.
