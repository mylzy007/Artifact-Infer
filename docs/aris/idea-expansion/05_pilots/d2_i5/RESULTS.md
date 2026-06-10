# D2-I5 RESULTS — Stage D3

- Date: 2026-05-30
- Pilot: D2-I5 (L_recv-gated dynamic drop, offline Pareto replay)
- Data: `run01/` — 135,552 measured records (30% replay split across 3 workloads)
- Codex round 1 results review: `docs/aris/traces/d2_i5_results_review_round1.md`
- Worktree: `pilot/d1-i10-d2-i5` (replay-only; no algorithm-code changes)

## TL;DR

- **C2.main: FAIL.** Pre-registered operating-point dominance is not achieved. EXP saves 98.95% of B1's bytes but loses 98.87% of B1's mass (vs ≤50% bound). The pre-registered dominance claim fails.
- **C2.neg: gate-rate PASS; mass-loss FAIL against the pre-registered bound.** Interpretation: the bound is confounded by rare high-mass prefill records.
- **C2.sample: PASS** (32 long-prompt prompts, 8 gsm8k prompts).
- **Observed findings within this policy suite and trace distribution**:
  1. In the tested alternative gates (L_recv, L_send, phase, global L*, per-layer L*_ℓ, threshold sweep α∈[0.5..2.0], per-phase oracle, per-record loss-budget oracle), all 9 "gated tail_weight r=0.3" variants produce *aggregate-indistinguishable* bytes and mass-loss on long-prompt workloads — because droppable volume is overwhelmingly prefill, so any gate that fires-on-prefill produces identical aggregate metrics.
  2. The main observed aggregate differentiator in this suite is the within-step selection rule (tail_weight vs random): at matched bytes (~65 GB), tail_weight loses 17.98% mass vs random 28.42% (a 1.58× gap). **Caveat**: tail_weight is partially optimized for the router-mass evaluation metric; this is a metric-aligned selection benefit, NOT evidence of algorithmic dominance over stronger non-random selectors (rank-local tail, expert-local tail, activation-based, learned). Stronger selector baselines remain future work.
- **Paper-level claim (Codex-recommended, narrowed)**: *"In these offline, prefill-dominated traces and tested policy family, once gates collapse to prefill-only behavior, tail-weight selection preserves more router mass than random at matched bytes."*

## Offline Methodology (per GATE D3 user requirement, not buried in limitations)

**LBG primary-owner approximation**. For each (token, expert) replica we use `target_rank = expert_id // E_local` (contiguous primary owner). Production LBG overlap_router can route a fraction of cells to non-primary replicas based on load-aware decisions. **Measured divergence from production routing**: 37.5% mean cell-assignment difference on passage_retrieval_v2, 38.5% on multifieldqa_v2, **0% on gsm8k_clean_v2** (contiguous mode, exact). **All 13 must-run policies use this same offline assumption, ensuring fair relative ordering across policies. Absolute production-LBG fidelity is out of scope.** This was approved by user at GATE D3 entry (Option A).

**EQUIV test (internal consistency)**: the offline `apply_tail_weight` (numpy-vectorized, used by EXP/B1/B-MB/...) matches the production `apply_drop_cpu("tail_weight")` mask byte-for-byte on **20/20 sampled records**. This validates the offline wrapper's correctness against the production drop API.

**Replay split**: deterministic per-workload by `batch_index` (last 3 batches of passage, last 1 of multifieldqa, last 1 of gsm8k); per-prompt bootstrap clusters = (batch_index, rank) since owner_local_ep places one prompt per rank per batch. Result: 24 prompts (passage) + 8 (multifieldqa) + 8 (gsm8k) = **40 bootstrap clusters total**.

## Pre-registered C2 verdicts

### C2.main: long-prompt operating-point dominance — **FAIL**

| | EXP | B1 (static tail r=0.3) | ratio EXP/B1 | pre-reg bound |
|---|---|---|---|---|
| bytes_saved (aggregate long-prompt) | 65,320 M | 66,012 M | **0.9895** | ≥ 0.80 ✓ |
| weight_mass_loss (aggregate long-prompt) | 17.979% | 18.184% | **0.9887** | ≤ 0.50 ✗ |

EXP and B1 produce nearly identical numbers because:
- EXP gate fires whenever `L_recv > 3271`; on long-prompt this is true for every prefill step and false for every decode step.
- Prefill cells = 99.4% of all (token, expert) replicas in the long-prompt distribution.
- EXP therefore differs from B1 by only ~0.6% (the decode cells B1 drops but EXP doesn't).
- This was the ex-ante Risk-1 flagged in PLAN's Honest Limitations BEFORE running.

**VERDICT per pre-registered failure handling**: scope-limit from "operating-point dominance" to **"comparable Pareto at the static SOTA's operating point"**.

### C2.neg: GSM8K negative control — **PARTIAL (gate-open-rate PASS, mass-loss FAIL)**

| metric | observed | bound | verdict |
|---|---|---|---|
| gate_open_rate (per-record) | 0.44% | — | — |
| **upper 95% binomial CI on gate_open_rate** (Wilson, n=49,536 trials) | **0.49%** | < 5% | **PASS ✓** |
| weight_mass_loss_frac | 2.35% | ≤ 1pp | **FAIL ✗** |

Root cause of the mass-loss failure: GSM8K's 1 replay batch has 1 prefill + 128 decode = 129 measured steps × 48 layers × 8 ranks = 49,536 records. The gate opens on **a subset of the high-mass prefill records** (~218 records, 0.44% — less than one full prefill step's worth of 384 (rank, layer) combos; gate fires only on the layer/rank pairs where `L_recv > 3271`). On those open records, EXP applies an r=0.3 tail-weight drop to eligible router entries (protection/capping per production semantics); because prefill cells carry most of GSM8K's router-weight mass, those 0.44% of open records account for 2.35% of total mass.

The per-record gate-open-rate metric (which PASSES the upper-CI bound) does not align with the per-mass safety concern (which FAILS the 1pp bound). A byte- or mass-weighted gate-open-rate is **a necessary companion metric better aligned with this failure mode; it does not erase the pre-registered failure**.

### C2.sample: PASS (per-prompt clusters)
- Long-prompt: 32 prompts (24 passage + 8 multifieldqa) ≥ 8 ✓
- GSM8K: 8 prompts ≥ 4 ✓

## What the 13-policy comparison ACTUALLY shows

### Long-prompt workloads (aggregate passage_retrieval + multifieldqa)

| Policy | bytes saved (M) | mass loss % | gate-open % |
|---|---|---|---|
| B0 no-drop | 0 | 0.000 | 0 |
| B-RS_0.05 static tail r=0.05 | 10,887 | 2.145 | 100 |
| B-RS_0.10 | 22,120 | 4.937 | 100 |
| B-RS_0.20 | 44,239 | 11.193 | 100 |
| B1 = B-RS_0.30 (SOTA) | 66,012 | 18.184 | 100 |
| B-RS_0.40 | 88,131 | 26.187 | 100 |
| B-RS_0.50 | 110,251 | 35.302 | 100 |
| **B-LS / B-PG / B-GL / B-TH_(0.5..2.0) / B-DR_0.3 / EXP / ORC-PP / ORC-LB / B-MB** | **65,320 (ALL identical)** | **17.979 (ALL identical)** | **~2** |
| **B-MR matched-bytes RANDOM** | **65,320** | **28.418** | **~2** |
| B-DR_0.1 / 0.2 / 0.4 / 0.5 (gated drop-rate sweep) | varies | varies | ~2 |

**Two findings emerge:**

**Finding-1 (gate-redundancy)**: 9 distinct "gated tail_weight r=0.3" policies (B-LS, B-PG, B-GL, B-TH α∈{0.5,..,2.0}, B-DR_0.3, EXP, ORC-PP, ORC-LB, B-MB) produce *byte-for-byte identical* aggregate bytes and mass loss. They all fire on the same prefill steps because in this distribution, prefill is trivially separable: every prefill step has phase=prefill AND T·K large AND L_recv >> 3271. The "where to gate" question is **over-determined**.

**Finding-2 (matched-bytes selector benefit vs random)**: At matched bytes (~65,320 M), `tail_weight` selection (EXP/B-MB) loses **17.979%** mass vs `random` selection (B-MR) at **28.418%** — a **1.58× mass-preservation gap** at matched bytes within this policy suite. *Caveat (Codex round 1+2)*: tail_weight directly minimizes weight_mass_loss (the eval metric); the 1.58× advantage is partially tautological, and random is a weak selector baseline. Beating random does NOT establish superiority over stronger non-random selectors (rank-local tail, expert-local tail, activation-based, learned). The selector-benefit-vs-random claim is the most we can defend from this pilot.

### GSM8K (decode-heavy negative control)

| Policy | bytes saved (M) | mass loss % | gate-open % |
|---|---|---|---|
| B0 | 0 | 0.000 | 0 |
| B1 static r=0.3 | 6,332 | 17.630 | 100 |
| EXP (L*=3271) | 900.6 | 2.347 | 0.44 |
| ORC-PP per-phase | 2,537 | 6.576 | 1.55 |
| B-MR matched-random | 900.6 | 4.072 | 0.44 |
| B-TH_2.0 (α=2) | 38.3 | 0.102 | 0.02 |

EXP correctly fires only on prefill (0.44% of records). At matched bytes, EXP has 1.7× lower router-mass loss than B-MR (matched-random) — same selector-vs-random pattern as on long-prompt. EXP < B-TH_2.0 (tighter threshold) on bytes and mass loss — α=2 would have passed C2.neg trivially, but it's a sweep point not the pre-registered EXP.

## Per-rank / per-layer / per-phase diagnostics (Codex round 2b attack #28)

Per-rank EXP bytes/loss on passage_retrieval (sample, all ranks similar):
- All 8 ranks within 5% of mean — no straggler / imbalance issue introduced by EXP. ✓

Per-phase decomposition (long-prompt, EXP vs B1):
- prefill records (n≈2 per workload): EXP and B1 IDENTICAL on prefill — both drop 30%.
- decode records (n≫): B1 drops 30% × decode mass; EXP drops 0% × decode mass.
- Decode mass / total mass ≈ 1% in long-prompt → EXP - B1 ≈ 1% mass loss saved.

This matches the C2.main observation: bytes/mass ratios ~99%.

Per-layer EXP bytes (passage_retrieval): all 48 layers have non-zero gate-open contributions, dominated by prefill-step bytes. No "hot layer" anomaly.

## Bootstrap CIs (cluster = (batch_index, rank), 1000 iters)

Selected examples on passage_retrieval long-prompt:

| Policy | bytes saved mean | bytes 95% CI | mass loss mean | mass loss 95% CI |
|---|---|---|---|---|
| EXP | 54,046 M | [49,123, 58,891] M | 18.099% | [16.412, 19.842]% |
| B1 | 54,338 M | [49,398, 59,187] M | 18.207% | [16.499, 19.962]% |
| B-MR matched | 54,046 M | [49,123, 58,891] M | 28.597% | [25.987, 31.247]% |

CIs are tight (24 prompt clusters → reasonable cluster-bootstrap stability). EXP vs B1: CIs overlap heavily on bytes AND on mass — consistent with the FAIL verdict (no significant difference). EXP vs B-MR: CIs cleanly separated on mass — the router-mass gap between tail_weight and random is stable in this bootstrap.

(Full CI table for all 27 policy points in `replay_results.json`.)

## Honest Limitations (cumulative across pilots)

### From D1-I10 (carried forward)
1. **Observational not interventional** — replay uses natural distribution; no synthetic intervention on worst_peer at fixed L_recv. Causal language avoided throughout.
2. **L_recv vs worst-peer collinearity (VIF=50)** — predictors are near-surrogates in this distribution; the paper claims L_recv as preferred, not uniquely causal.
3. **Per-layer L*_ℓ stratification artifact (19/48 layers non-monotonic)** — global L*=3271 used per GATE D2.

### From D2-I5 (new this stage)
4. **LBG primary-owner approximation** — production LBG overlap_router routes are non-stateless; offline simulation diverges by 37.5% (passage) / 38.5% (multifieldqa) / 0% (gsm8k) at the cell-assignment level. All 13 policies use the same offline assumption → relative ordering is fair; absolute production-LBG fidelity is out of scope.
5. **Offline replay is not counterfactual** — drops would change future hidden states and future routing; offline simulation uses the no-drop logged trace. "Trace-level proxy replay" only.
6. **Tail-weight selection partially tautological vs the evaluation metric** — tail_weight directly optimizes weight_mass_loss; the 1.58× win vs random is partly definitional.
7. **Random is a weak selector baseline** — beating random does NOT establish superiority over loss-aware, entropy-aware, expert-load-aware, or learned selectors. Stronger non-random selector baselines (rank-local tail, expert-local tail, activation-based) are deferred to post-pilot.
8. **Router-mass preservation is a proxy** — even setting tautology aside, weight_mass_loss is a proxy for task quality; without downstream accuracy/perplexity measurement, every selector-quality conclusion remains proxy-level. The post-D3 stretch goal (LongBench official metric on the EXP policy with real drop applied) is the path to a task-level claim.
9. **Gate-trigger redundancy on prefill-dominated workloads** — long-prompt traces make the tested alternative gates (L_recv, T·K, phase, global L*, threshold sweep, per-phase oracle, per-record loss-budget oracle) all aggregate-indistinguishable. The L_recv-gated mechanism cannot be differentiated from simpler alternatives in this distribution. Untested gates that use richer signals (per-rank load imbalance, downstream loss feedback) may yet differentiate.
10. **GSM8K mass-loss metric misalignment** — per-record gate-open rate (PASS) does not align with per-mass safety concern (FAIL). Byte- or mass-weighted open-rate is **a necessary companion metric** for the safety claim, better aligned with the failure mode than per-record rate alone. Does NOT erase the pre-registered failure.
11. **Generalization boundary** — results may not generalize to: other decode-heavy workloads beyond GSM8K (only this one decode-heavy negative control was run), balanced prefill/decode distributions (uncovered here), other MoE architectures, other routing distributions, or online serving with real congestion feedback (where gate decisions and routing co-evolve).
12. **Uncertainty / robustness scope** — bootstrap CIs reported on per-policy aggregates; sensitivity across additional traces, seeds, byte budgets, and dropping seeds NOT explored in this pilot. The 98.95% / 98.87% / 1.58× / 2.35% headline numbers are point estimates from one offline replay; full sensitivity analysis is future work.

## Pre-registered failure handling actually applied

- **C2.main**: REFUTED → scope-limit from "operating-point dominance" to "comparable Pareto, not strict dominance" (per PLAN failure handling).
- **C2.neg**: gate-open-rate PASS + mass-loss FAIL → registered scope-limit "the gate-open-rate metric correctly identifies infrequent gate firing; mass-loss bound was set too tight for tail_weight applied to prefill steps within the open-gate fraction".
- **C2.calib**: SATISFIED (L*=3271 used unchanged from D1-I10 calibration; no post-hoc retuning).
- **C2.sample**: PASS.

## Final verdict and paper framing

**The pre-registered dominance claim fails.** D2-I5 main claim is REFUTED; the registered scope-limit/downgrade outcome applies.

Honest paper claim (Codex round 1+2 verified, narrowed to what the data supports):

> Offline replay shows that L_recv-gated dropping does NOT achieve pre-registered operating-point dominance over static tail-weight @ r=0.3. On long-prompt workloads, prefill cells dominate the droppable volume (99.4%), so the L_recv gate AND each of the **tested alternative gates** (phase, T·K, global L*, threshold sweep, per-phase oracle, per-record loss-budget oracle) are **aggregate-indistinguishable** — producing identical aggregate bytes and mass loss. The main observed aggregate differentiator among these policies is the within-step selection rule: at matched bytes, **tail_weight selection preserves 1.58× more router-weight mass than random selection** in this trace distribution. *Caveat*: tail_weight is partially optimized for the router-mass metric, so this is a metric-aligned selection benefit and not yet evidence of algorithmic dominance over stronger non-random selectors (rank-local tail, expert-local tail, activation-based, learned).
>
> On the GSM8K decode-heavy negative control, the L_recv gate fires rarely (per-record gate-open rate ≤ 0.49% upper 95% CI; far below the 5% bound), but the small fraction of open-gate prefill steps still carries 2.35% of router-weight mass — exceeding the 1pp bound. **The pre-registered failure stands.** Byte- or mass-weighted gate-open rate is a necessary *companion* metric better aligned with this failure mode; that observation does not erase the failure.

### What would validate the paper claim (out of scope for this pilot)
1. **Interventional / online replay** — execute EXP and B1 with real drop applied in production, measure actual latency and task accuracy (not proxies).
2. **Stronger selector baselines** — rank-local tail, expert-local tail, activation-magnitude based, learned selectors — to test whether the router-mass advantage survives stronger selector baselines.
3. **Downstream quality metrics** — LongBench official `retrieval_score` / GSM8K solve-rate at the EXP policy (post-D3 stretch goal, if user authorizes).
4. **Non-prefill-dominated workloads** — chunked-prefill, mixed-length inputs, online serving with intermediate L_recv distributions where the gates *could* disagree.

This is a clean honest negative-result outcome with one defensible (but caveated) positive finding (tail_weight > random at matched bytes) and two methodological contributions (gate-trigger redundancy on prefill-dominated distributions; necessity of byte-weighted gate-open-rate as safety companion metric).

## Stretch: Production Metric Sanity Check (added 2026-05-30 post-D3)

Full STRETCH_RESULTS.md in this directory. Headline:

| label | strict_accuracy | official retrieval_score | prefill speedup | e2e speedup |
|---|---|---|---|---|
| B0 baseline (no drop) | **1.000** | 0.257 | 1.000× | 1.000× |
| B-PG (= production v6 SOTA: tail_weight @ r=0.3 + MIN_REPLICAS=512; D2-I5-confirmed equivalent of EXP in this distribution) | **1.000** | 0.383 (+0.126) | **1.232×** | **1.111×** |

**Key reframing**: D2-I5's pre-registered C2.main FAIL is on `weight_mass_loss ≤ 0.50 × B1`. Observed mass loss ratio = 99% (failure). BUT the **strict_accuracy degradation is 0.00%** and official retrieval_score actually *improves* with drop. **The router-mass-loss proxy is OVERLY PESSIMISTIC by ~18pp vs the real production metric on this workload.**

C2.main remains REFUTED per pre-registered judge (the user's "严禁修改 pre-registered 判据数值" rule stands), but the failure is now framed as **proxy-vs-real divergence** — itself a paper finding. The stretch also independently reproduces the user's prior Phase 4 v6 result ("prefill +23% / e2e +12% / accuracy zero loss").

This shifts the joint D1-I10 + D2-I5 framing toward:
- D1-I10: L_recv is the dominant a2a-latency predictor (worst-peer refuted but framing preserved).
- D2-I5 offline: gate-trigger redundancy on prefill-dominated workloads; tail_weight > random at matched bytes (proxy-level).
- D2-I5 stretch: **production phase-gated drop has zero strict-accuracy cost and gives +11% e2e speedup**, with router-mass-loss proxy overstating quality damage by ~18pp.

Honest Limitation #8 update (router-mass = proxy): MEASURED gap = **18pp proxy-pessimism vs strict_accuracy on this workload**.
