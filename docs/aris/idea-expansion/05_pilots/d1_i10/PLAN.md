# D1-I10 Pilot PLAN — Worst-Peer Bytes → Latency Causal Regression

- Date: 2026-05-29
- Worktree: `pilot/d1-i10-d2-i5` @ `/home/lzy/Artifact-Infer-pilot`
- Data dependency: Stage D0 v2 collection (`raw_collection_v2/`) — jsonl + sidecar
- Constraints: 禁修 dispatch/combine/expert/drop 算法代码,只能写新的 microbench harness 和分析脚本
- Methodology: `experiment-plan` skill (Phases 1-4) + `analyze-results` skill (4-tuple)

## Research Question
> Under owner_local_ep + ep_ht runtime, does the *worst-peer (dispatch+combine) bytes* of a step explain its segment latency strictly better than L_recv alone, when measured on real routing patterns under a controlled replay?

If yes, the 97% segment-attribution finding (Tier 1) is upgraded from *correlational* to a *predictive cost-model* that future optimizations can target directly.

> **Scope statement (Codex round 2 attack #48)**: This pilot makes a claim about *one* dispatcher implementation (`owner_local_ep + ep_ht`) on *one* model (Qwen3-30B-A3B, K=8) on *one* hardware (8×RTX 4090). Any "worst-peer is the right cost-model" generalization beyond this configuration is **out of scope** of this pilot.

> **Causality scope (Codex attack #1)**: "Controlled replay regression" — we control hidden-state random seed, drop-rate, and routing pattern source, but do NOT independently intervene on worst-peer bytes while holding L_recv fixed (true counterfactual). The regression is observational on real patterns under a synthetic-input replay. The word "causal" is dropped from the title.

## Definitions (Codex attacks #41–43, made explicit)
- **L_recv** (per-rank, per-step, per-layer) = `total_recv = sum(recv_counts)` in `dispatch_ep_ht.py` line 383. This is the number of token-replica rows received by THIS rank's experts after the a2a from peers (post-drop if any; drop is OFF in our v2 collection).
- **worst_peer_bytes** = `(max(send_counts) + max(recv_counts)) × H × elem_size` where the max is over all 8 peers (including local). Captures the slowest peer on each side of a2a.
- **skew** = `max(send_counts) / mean(send_counts)`; 1.0 = uniform, > 1.0 = peer hotspot.
- **segment_latency** (= `total_us` in JSONL row) = CUDA-event interval from `tot_t.__enter__()` (immediately before `dispatch(...)`) to `tot_t.__exit__()` (immediately after `combine(...)`), measured rank-max across 8 ranks via all-gather of local CUDA-event readouts.
- **FLOPs_proxy** = `L_recv × H × MOE_INTERMEDIATE_SIZE × 2` (one fused GEMM per recv row). **NOTE (Codex attack #3)**: this is `L_recv × const` → numerically equivalent to L_recv as a regressor. Single-variable R²(FLOPs_proxy) ≡ R²(L_recv); they are NOT separately informative. The pre-registered C1.main therefore reduces to "worst_peer_bytes beats L_recv by ≥ 20pp" — we report it under both labels for transparency but do not pretend FLOPs adds distinct evidence.
- **cell** = one (workload, rank, step_id, layer_id) sampled record put through the microbench.
- **tier1 baseline** (used in fail-fast) = the median `total_us` of tier1 microbench at matched (T_local, drop_rate=0.0) from `eval_results/prefill_drop_l_sweep_tier1/tier1_rows.jsonl`, after L_recv bucketing.

## Pre-registered Claims (immutable; numeric thresholds frozen at this date)

### C1.main — bytes vs FLOPs single-variable R²
> Single-variable regression of total segment latency on worst-peer (dispatch+combine) bytes achieves R² ≥ FLOPs single-variable R² + **20pp** in a controlled microbench whose routing patterns are sampled from real v2 data.

- **Failure handling**:
  - **R²Δ < 20pp** → C1 **REFUTED**. Action: *scope-limit* the communication-centric framing in the paper from "bytes are the dominant predictor of latency" to "bytes contribute a substantial but non-dominant share" (specific share to be the measured value); revisit D2-I5 motivation.
  - **R²Δ ≥ 20pp** → C1 **SUPPORTED** (pending anti-claims).

### C1.anti1 — incremental R² after controlling for FLOPs + L_recv
> Adding worst-peer bytes to a model already containing {FLOPs, L_recv} as predictors yields an incremental R² of **≥ 10pp**.

- **Failure handling (R²Δ < 10pp)** → anti-claim 1 **stands**. Action: *downgrade* C1.main to "bytes are a useful but redundant predictor relative to L_recv"; paper claim becomes "L_recv (a quantity that send-side never sees pre-collective) is the dominant predictor"; D2-I5 framing benefits (it gates on L_recv directly).

### C1.anti2 — coefficient stability across L_recv segments
> The fitted coefficient on worst-peer bytes in the multi-var model is consistent (sign-stable AND within 50% relative magnitude) across three L_recv tertiles {low, mid, high} of the sampled distribution.

- **Failure handling (coefficient unstable)** → anti-claim 2 **stands**. Action: *scope-limit* C1.main to the L_recv regime where coefficient is stable; document the regime explicitly. Paper claim becomes regime-conditional.

### C1.sample — statistical power (tightened per Codex attacks #39, #40)
> N (microbench cells) ≥ **300** after filtering;
> AND coverage of **≥ 40 of 48 layers** with ≥ 5 cells each;
> AND ≥ **3 distinct batch_index clusters** represented in each of the 3 L_recv tertiles (independence requirement for clustered bootstrap).

- **Failure handling** → INCONCLUSIVE; pilot must extend sampling before claiming any other.

## Method

### Inputs (NO new GPU data collection — uses Stage D0 v2)
- `raw_collection_v2/per_step_trace_{run_id}_rank{r}.jsonl` (metadata)
- `raw_collection_v2/per_step_trace_{run_id}_rank{r}_topk.bin` (topk_ids + topk_weights, format v1)
- Three workloads pooled: passage_retrieval_v2, multifieldqa_v2, gsm8k_clean_v2 — measured-only (`is_warmup==False`)

### Pipeline — `eval/d1_i10_bytes_latency_microbench.py` (NEW, in worktree)
Fork of `eval/drop/run_owner_local_ep_phase4_drop_tier1_microbench.py`. Key changes:

1. **Sampling stage (CPU pre-pass)**:
   - Build the full v2 measured-record pool keyed by `(workload, rank, step_id, layer_id)` with derived fields `L_recv`, `worst_peer_send = max(send_counts)`, `worst_peer_recv = max(recv_counts)`, `skew = max(send_counts) / mean(send_counts)`.
   - **Pre-registered 70/30 split (revised per Codex attack #13)**: deterministic by `(workload, batch_index) mod 10 < 7` → calibration set (70%, used by D1-I10); else **D2-I5 replay set** (30%). Splitting by `batch_index` (= one LongBench prompt or one GSM8K sub-batch of 8 problems) makes the holdout PROMPT-grouped rather than step-grouped, eliminating the periodic-block leak. Seed = **42** (placeholder; the split itself is deterministic, the seed is used by stochastic-baseline matched-random in D2-I5 only).
   - From the 70% calibration pool, sample cells by **stratified random**:
     - 48 layers (each layer represented)
     - 3 L_recv tertiles per layer (bottom/mid/top of the layer's L_recv distribution)
     - 2 skew buckets per (layer, tertile) (above/below median skew within the cell)
     - 3 samples per (layer, tertile, skew) → **48×3×2×3 = 864 cells** (after filtering, ≥ 400 expected; > 300 sample floor).
   - Save sampling manifest `samples.jsonl` with full provenance.

2. **Per-cell microbench (GPU)**: for each sampled cell, look up the rank's topk_ids and topk_weights from the v2 sidecar; build the same EP-HT MoE block as tier1 (`DispatchEPHT/ExpertsEPHT/CombineEPHT`); **logit-encode replay (revised per Codex attack #11, #12)**:
   - Hidden state: `torch.randn(T, H, dtype=bf16, generator=seeded)` — random but deterministic per (workload, step, layer). Caveat: latency is dominated by communication and grouped GEMM shape, not hidden-state values, so randomized inputs should not bias latency. We sanity-check by re-running 8 cells with recorded hidden states (replayed offline via a separate harness) and verifying segment latency agrees within 5%.
   - **Logit construction preserving order AND weights**: `logits[:, :] = -1000.0`; then `logits.scatter_(1, topk_ids.long(), 1000.0 + torch.log(topk_weights.clamp_min(1e-10)))`. This forces `torch.topk` to return exactly the recorded experts (their logits ≫ -1000) AND `torch.softmax(logits.gather(1, topk_ids))` equals the recorded `topk_weights` exactly (since `softmax(C + log w) = w` when C is per-row constant and w sums to 1). `norm_topk_prob` becomes a no-op.
   - **Drop OFF** for the main measurement (no policy bias).
   - Measure `dispatch_us`, `experts_us`, `combine_us`, `total_us` with `CUDAEventTimer` (reused from `eval/drop/shared.py`) over `--warmup-iters 8 --iters 30` (bumped from 5/15 per Codex attack #34), rank-max aggregated, repeated **3 times from fresh process starts** (Codex attack #47) and aggregated with median per cell.
   - **Additionally**: for each cell, run a paired drop=0.3 (tail_weight) measurement → Δlatency. **Drop-rate sweep** (Codex attack #21): also sample 60 cells at drop_rate ∈ {0.1, 0.2, 0.4} to derive per-layer Δlatency curves rather than a single point. This is what D2-I5's L*_ℓ calibration consumes.

3. **Sanity validation** (must PASS before regression):
   - For each cell: assert the dispatcher's runtime `send_counts` (from `tok_meta`) equals the v2-recorded `send_counts` for that (rank, step, layer). If mismatch on >1% of cells, **STOP** — logit encoding is broken.
   - Assert `L_recv_measured = L_recv_v2` per cell.

4. **Regression analysis** — `eval/d1_i10_regression_analysis.py` (NEW; revised per Codex attacks #3, #7, #24, #29, #44, #46):
   - Output JSONL rows joined with v2 ground-truth structural fields and microbench timing.
   - **Three response variables, regressed separately (priority finalized at GATE D1)**:
     - `dispatch_us + combine_us` = **PRIMARY** (a2a-only — the cleanest test of the brief's "97% from a2a payload" claim; the pre-registered C1 thresholds are evaluated on this response).
     - `total_us` = **SECONDARY** (whether the bytes-driven story generalizes to whole-segment latency; failure mode is well-known: experts_us may dominate total_us in long-prompt prefill, scoping the claim to a2a).
     - `experts_us` = **TERTIARY** (sanity / control — L_recv MUST predict this strongly since it IS the GEMM row count up to a constant; a failure here would imply timing-instrumentation error, not a hypothesis test).
   - **Models** (predictor side; same for each response):
     - M1: `y ~ worst_peer_bytes`
     - M2: `y ~ L_recv`   *(FLOPs_proxy = const × L_recv, so M2 covers both — see Definitions)*
     - M4: `y ~ worst_peer_bytes + L_recv + C(layer)` *(adds **layer fixed effects** per Codex attack #7; drops collinear FLOPs_proxy per attack #3)*
     - M5 (sanity / control): `y ~ T (num_tokens) + sum(send_counts) + C(layer)` — tests if cruder predictors already explain latency well.
   - **All predictors standardized (z-scored)** before fitting (Codex attack #30) — coefficients in M4 are then comparable across predictors. Raw-units OLS reported separately for interpretability.
   - **In-sample R² + 5-fold cross-validated R²** (groups by `batch_index` — prompt-grouped CV, Codex attack #46). The pre-registered C1 thresholds are evaluated using the **5-fold CV R²** (held-out), not in-sample.
   - **Clustered bootstrap CIs (Codex attack #24, #29)**: 1000 bootstrap iterations, clusters = `batch_index`. Report R², ΔR², coefficients with 95% CI for every reported number.
   - **Diagnostics**: VIF for M4 (must be < 5 to declare the multi-var fit interpretable; ≥ 10 = block); condition number of design matrix; per-tertile coefficient bootstrap CI; residual-vs-fitted plots.
   - **Censored cells (Codex attack #38)**: cells that fail at any stage (logit-encoding validation, wall-time blow-up, CUDA error) are NOT silently dropped from the analysis; they are recorded in a `censored_cells.jsonl` with the failure mode and the cell's predictor values. The primary analysis reports the rate and predictor-distribution of censored cells; sensitivity is reported with/without imputation (carry-forward median by stratum).

## Setup
- Hardware: 8×RTX 4090 (existing setup)
- Runtime: ep_ht + owner_local_ep (matches collection)
- Software: same conda env (`vllm`)
- Worktree: `/home/lzy/Artifact-Infer-pilot` @ `pilot/d1-i10-d2-i5` (no algorithm-code edits; only new files under `eval/`)
- Random seeds: sampling seed = 42 (fixed), hidden-state seed = `hash((workload, step_id, layer_id)) & 0xFFFFFFFF` per cell

## Metrics

### Primary (decides C1.main / anti1 / anti2)
- R² of M1 (worst_peer_bytes)
- R² of M2 (FLOPs_proxy)
- R²Δ = R²(M1) − R²(M2)
- Incremental R² (partial F-test): (R²(M4) − R²(M3∪M2)) attributing to worst_peer_bytes
- Per-tertile β̂(worst_peer_bytes) + 95% CI

### Secondary (paper context)
- M3 R² (L_recv alone)
- M4 R² + adjusted R² + VIF + condition number
- Per-segment regressions: dispatch_us-only and combine_us-only as M1 sub-fits
- Δlatency(drop=0.3 vs drop=0) per layer × L_recv bucket → **L*_ℓ output table** (CONSUMED BY D2-I5)

### Provenance
- `samples.jsonl` schema: cell_id, workload, rank, step_id, layer_id, source_offset, sampled_at, T, K, L_recv, worst_peer_send, worst_peer_recv, skew
- `regression_results.json` schema: model_id, R², adj_R², coefficients with SE, VIF, condition_num, n, tertile

## Fail-fast Conditions (any → STOP, no auto-recovery)
1. **logit-encoding validation fails on >1% of cells** → encoder broken; do not regress on garbage.
2. **microbench cell wall-time > 5× tier1 baseline** (i.e., > 1s/cell mean) → host overhead pathological; investigate.
3. **N usable < 300 after sanity filter** → insufficient power; do not declare any verdict.
4. **GPU OOM or process crash** → stop, report.
5. **Total wall time exceeds estimate × 1.5** → stop, report (per user's standing rule).

## Risks + Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Multicollinearity (anti1 test invalid)**: worst_peer ≈ L_recv × const in real data | high | high (C1.anti1 untestable) | Stratify by skew (max/mean ratio); ensure each skew bucket has > 1.5× spread in (worst_peer / L_recv); flag VIF > 5 as warning, > 10 as block |
| **Logit-encoding doesn't preserve send_counts exactly** for ties / overlap routing | med | high (replay invalid) | Validation step at every cell; the LBG overlap router is stateful, expect partial mismatches on overlap workloads — accept up to 1% mismatch for LBG, 0% for contiguous (gsm8k) |
| **R²Δ between 15–20pp** (close to threshold) | med | med (INCONCLUSIVE) | Pre-register: no fudging; report exact value + 95% bootstrap CI; INCONCLUSIVE if CI straddles 20pp |
| **Per-layer L*_ℓ is too noisy** (3 samples per cell may be thin for D2-I5 consumption) | med | med (D2-I5 calibration weak) | Report per-layer L*_ℓ as point estimate + uncertainty; D2-I5 must propagate uncertainty into Pareto bands |
| **Latency floor noise dominates at small L_recv** | low | low | Drop cells with `total_us < 2× CUDAEventTimer noise floor` from M1; report drop count |
| **Per-layer mode of all 48 layers in a single process** | low | high | tier1 already runs single-layer per-cell with `layer_id` arg; replicate; mute concurrency |

## Expected RESULTS schema (`analyze-results` 4-tuple per finding)

For each of C1.main / anti1 / anti2:
```
Observation:     R² values and Δ, with bootstrap CIs (exact numbers from regression_results.json)
Interpretation: which predictor mechanistically drives the latency signal, given dispatcher internals
Implication:    what this means for D2-I5 gate viability and for paper framing
Next-step:      either advance to D2-I5 with the calibrated L*_ℓ, or revise framing per fallback above
```

## Resource Estimate
- Sampling pass (CPU): < 1 min
- Microbench (GPU, EP=8, 864 cells × 2 drop-rates × ~20 iters × 5 ms): **≈ 30–60 min** wall (incl. framework warmup, sync barriers, cooldown). Memory headroom OK (single 1-layer block fits in <2 GiB / GPU).
- Regression + diagnostics (CPU): < 5 min
- **GPU budget**: ≤ 1 hour wall, hard fail-fast at 1.5 hours.

## Known Limitations (Stage D0 carryover)
- send_counts reconstruction from `topk_ids` was directly validated only in **contiguous** mode (gsm8k_clean_v2). For LBG overlap mode (passage_retrieval_v2 + multifieldqa_v2), stateful overlap_router replay was not run; instead, **v1↔v2 byte-identical evidence** (52,224 + 73,728 records per rank, 100% match under greedy decoding) establishes that the topk-recording patch did not perturb LBG routing.
- **Implication for D2-I5 calibration debugging**: if D2-I5's L_recv-gated replay diverges from microbench predictions, first suspect calibration (L*_ℓ noise, regression mis-specification, or the 70/30 split's representativity), not the v2 data source.

## Honest Limitations (ex-ante risks identified BEFORE running, per GATE D1 user request)

These are risks we KNOW upfront the pre-registered thresholds may not be able to clear; we run the pre-registered judges anyway and accept the registered scope-limit outcomes. Reviewer transparency:

### Risk-1 (C1.main on `total_us` may structurally underperform)
- Brief established "97% drop savings come from a2a payload" at the SEGMENT level (`dispatch_us + combine_us`). But `total_us = dispatch + experts + combine`.
- In long-prompt prefill (large L_recv), `experts_us` is dominated by the per-row GEMM cost = `L_recv × H × N_intermediate × 2` (linear in L_recv).
- If `experts_us / total_us` is large in the sampled cell distribution, then L_recv alone may explain `total_us` very well, leaving little room for `worst_peer_bytes` to add ≥ 20pp on `total_us`.
- **Ex-ante mitigation (already in PLAN)**: PRIMARY response is `dispatch_us + combine_us` (where the brief's claim lives cleanly); `total_us` is SECONDARY. A `total_us` failure with a `dispatch_us+combine_us` pass produces the scope-limit outcome "bytes dominate the communication segment; a2a-cost ≪ total-cost generalization needs further work" — which is honest and paper-publishable.

### Risk-2 (C2.main 50% mass-loss bound may be structurally hard on long-prompt workloads)
- (This is D2-I5's risk but D1-I10's L*_ℓ output flows into it; recording here for cross-pilot transparency.)
- Long-prompt workloads have ~99.4% of (token, expert) cells concentrated in prefill steps (one prefill step with T~5000, then ~31 decode steps with T=1).
- EXP's gate closes most aggressively on decode steps (low L_recv); but those steps contribute < 1% of total cells.
- B1 static drops 30% of cells uniformly. If most of B1's weight-mass loss comes from prefill (the bulk), then EXP can save at most ~0.6% of cells from being dropped → that's the maximum mass-loss reduction available → reducing mass-loss by 50% requires that the "saved" 0.6% of cells carry ≥ 50% of B1's mass loss, which would require severely concentrated low-weight branches in decode (possible but not guaranteed).
- **Ex-ante mitigation**: pre-registered scope-limit text says "if either condition (bytes ≥ 80% / mass-loss ≤ 50%) fails, C2.main is REFUTED; paper claim becomes 'comparable Pareto' instead of 'strict operating-point dominance'". This is the registered fallback.
- *We commit to NOT retuning C2.main thresholds post-hoc to manufacture a pass.*

### Risk-3 (cross-layer dependence in L*_ℓ, Codex attack #23, deferred)
- Per-layer L*_ℓ is calibrated in single-layer isolation; production runs all 48 layers in sequence with cumulative scheduling effects.
- This pilot cannot test the cross-layer composition. Composition risk is logged for the post-pilot stage (online validation).

## Devil's Advocate Review

Source: Codex round 2a, `traces/d1_i10_plan_attack.md` (48 attacks total). Below: the 20 most material attacks + Claude response + applied PLAN edit. Numbers reference Codex's attack list.

### Accepted in full (PLAN body edited)

| # | Codex attack | Claude response + edit |
|---|---|---|
| 1 | "Causal" is overstated — observational replay, not interventional | **Accept**. Word "causal" removed from title and research question. Added explicit "Causality scope" disclaimer. |
| 3 | FLOPs_proxy ≡ L_recv × const → M2 and M3 are the SAME regression | **Accept (deal-breaker)**. The plan now defines FLOPs_proxy mathematically as `L_recv × const`, explicitly says they are *not* separately informative as predictors, and merges them into a single M2 baseline. C1.main is now honestly "worst_peer_bytes beats L_recv by ≥ 20pp" — same threshold, cleaner statement. |
| 7 | No layer fixed effects | **Accept**. M4 now includes `C(layer)` term. |
| 11 | Equal +1000 logits destroy top-K ordering | **Accept**. New logit encoding `scatter_(1, topk_ids, 1000 + log(topk_weights))` preserves the recorded order AND yields softmax = topk_weights exactly. |
| 12 | topk_weights looked up but ignored | **Accept (same fix as #11)**. Replay now reproduces both routes AND weights. |
| 13 | Calibration/replay split groups nearby steps → leak | **Accept**. Split rule changed to `(workload, batch_index) mod 10 < 7` — prompt-grouped holdout. |
| 24 | N cells not independent — clustered by workload/step/layer | **Accept**. Cross-validated R² + clustered bootstrap CI (clusters = `batch_index`). |
| 29 | No confidence intervals | **Accept**. Bootstrap CI required for all reported R², ΔR², coefficients. |
| 30 | Predictors not standardized | **Accept**. All predictors z-scored before M4; coefficients comparable. |
| 34 | Warmup 5 / iters 15 looks optimistic | **Accept**. Bumped to 8 warmup / 30 iters, plus 3 fresh-process repeats with median aggregation. |
| 38 | STOP cells silently bias dataset | **Accept**. Censored-cell tracking + sensitivity analysis. |
| 39 | N < 300 too low for stratification | **Accept**. New: ≥ 40 layers × ≥ 5 cells/layer, ≥ 3 batch_index clusters per L_recv tertile. |
| 41-43 | L_recv / skew / segment_latency undefined | **Accept**. Explicit Definitions block added at top. |
| 44 | Dispatch and combine bytes collapsed prematurely | **Accept (partial)**. We now regress all three responses separately: total_us, (dispatch_us + combine_us), experts_us. |
| 46 | No held-out validation | **Accept**. Pre-registered thresholds evaluated on 5-fold CV R² with batch_index grouping. |
| 47 | One pilot run is hardware artifact | **Accept**. 3 fresh-process repeats per cell. |
| 48 | Generality exceeds experiment | **Accept**. Explicit scope statement added. |

### Accepted with scope caveat

| # | Codex attack | Claude response |
|---|---|---|
| 21 | drop_rate=0.3 arbitrary in L*_ℓ derivation | **Partial accept**. Added drop-rate sub-sweep on 60 cells {0.1, 0.2, 0.4} so L*_ℓ is derived from a per-layer Δlatency *curve* rather than a single point. Still pre-registers r=0.3 as the EXP rate for D2-I5 because user pre-registered it; the sub-sweep is for sensitivity diagnostics, not for D2-I5 retuning. |
| 33 | Overlap not properly accounted for | **Partial accept**. Single-cell measurement with `expert_overlap_strategy="disjoint"` (no overlap) for the regression; an additional confirmatory sub-experiment with the v6 LBG/greedy_balance overlap plan on 60 cells, reported separately. Overlap is *not* a confound in the primary regression because every cell uses the same disjoint setting. |

### Rejected (with stated reason)

| # | Codex attack | Claude response |
|---|---|---|
| 4, 5 | 20pp / 10pp thresholds arbitrary | **Reject**. The user pre-registered these numerically and pre-registered the rule "严禁修改 pre-registered 判据数值". The pilot reports the exact ΔR² with CI; a hostile reviewer can disagree with the threshold but we honor the registration. Bootstrap CI lets the reader judge. |
| 16, 17, 18 | Random hidden states / upstream runtime state / replay isolation | **Reject as pilot blockers** but **accept as scope caveats**: this is an *isolated single-MoE-block* microbench — that is *by design* (the tier1 template). Real production state confound exists; sensitivity sub-run on 8 cells with recorded hidden states will quantify it. End-to-end production-latency claims are explicitly out of scope. |
| 19 | Calibration split may leak workload structure | **Partial accept**. Edit #13 (batch_index split) addresses periodic leak. Splitting *across workloads* would mean training L*_ℓ on one workload and applying on another — that's a generalization test, not a calibration test. Reported as a future ablation if time permits. |
| 23 | Per-layer L*_ℓ ignores cross-layer dependence | **Reject for pilot**. The tier1 template is *isolated single-layer*. Cross-layer dependence is a real concern at production but cannot be tested in a single-layer microbench. Honestly noted in Known Limitations. |
| 31 | 8×RTX 4090 is fragile platform | **Accept as scope statement, reject as pilot blocker**. The whole research project is on this hardware; the pilot inherits this. Topology / clocks / NCCL version will be reported in the RESULTS appendix per skill conventions. |
| 36, 37 | "Scope-limit framing" too soft / anti-claim language confusing | **Partial accept**. Failure-handling text rewritten to use pass/fail/scope-limit/REFUTED labels; verdicts are explicit. The user's standing rule allows scope-limit as a legitimate outcome, not an escape hatch — Codex's complaint is fair but the alternative ("REFUTED, paper dead") is too brittle for an exploratory pilot. |

### Unresolved (will be revisited at GATE D2 RESULTS review)
- Codex #25 "R² dominated by between-bucket variance" — flagged for diagnostic; will report within-layer R² alongside global R². If within-layer R² is much weaker, the headline number will be scope-limited.
- Codex #27 "Linear model assumption unjustified" — diagnostic; will report residual vs fitted plots and run a spline fit on the same data. If linear is dramatically wrong, headline switches to "monotone predictor" framing.

