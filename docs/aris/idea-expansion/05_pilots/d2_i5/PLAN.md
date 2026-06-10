# D2-I5 Pilot PLAN — L_recv-gated Drop, Offline Pareto Replay

- Date: 2026-05-29 (last edit 2026-05-30)
- Worktree: `pilot/d1-i10-d2-i5` @ `/home/lzy/Artifact-Infer-pilot`
- Data dependency: Stage D0 v2 collection (`raw_collection_v2/`); L* = **3271** (tier1-derived, D1-I10 confirmed) — see GATE D2 decision
- Constraints: 禁修核心算法代码;只允许 offline replay 脚本和分析
- Methodology: `experiment-plan` + `ablation-planner` + `analyze-results`

## Changelog (edits to method, NOT to pre-registered judgement thresholds)
- **2026-05-30 (GATE D3 entry)**: L* calibration source frozen at L*=3271 global (GATE D2 decision); per-layer L*_ℓ relegated to B-TH α-sweep ablation. Background section updated.
- **2026-05-30 (GATE D3 entry, fail-fast #6 amendment)**: Original fail-fast "LBG offline reconstruction >1% mismatch → STOP" was triggered (measured: 37.5% mean cell-assignment divergence on passage_retrieval_v2, 38.5% on multifieldqa_v2; 0% on gsm8k_clean_v2 contiguous). After user authorization (Option A), this is RE-CLASSIFIED from a hard fail-fast to a documented Honest Limitation. Rationale: (1) all 13 policies use the same offline primary-owner assumption, ensuring Pareto ordering comparability; (2) Codex round 2b already accepted "offline replay is not counterfactual" — primary-owner LBG approximation is the same epistemic category. **Pre-registered C2 threshold values (50% mass-loss, 80% bytes) remain frozen.**

## Research Question
> Under offline replay on logged routing traces, does applying token-replica drop conditioned on `g(step, layer, rank) = (L_recv_bench(step, layer, rank) > L*)` with **L* = 3271** *dominate the static SOTA `tail_weight @ r=0.3`* on a pre-registered operating-point comparison (≥80% bytes saved AND ≤50% router-weight-mass-loss-proxy)?

## Background (post-D1-I10 framing, updated 2026-05-30)
D1-I10 controlled regression found L_recv (per-rank received row count) is the dominant single-scalar predictor of a2a latency (CV R² 0.87, in-sample 0.99 in load-dominated regime). The "worst-peer bytes" sub-thesis was REFUTED (CV ΔR² = −0.005 vs +0.20 threshold) and downgraded by the registered fallback. **L_recv-gated drop (D2-I5) gates on the right variable.** D1-I10 also empirically confirmed tier1's L*=3271 break-even (20/20 sampled cells with L_recv ≥ 3271 had drop helping; median Δa2a = −20.7 ms). Per-layer L*_ℓ derivation was too noisy (19/48 non-monotonic — stratification artifact); see B-TH for sensitivity reporting.

## L* calibration source (frozen, GATE D2 decision)
- **Primary**: L* = **3271** (tier1 break-even, D1-I10 confirmed). Used by EXP gate and the C2.main verdict.
- **Ablation only** (B-TH α-sweep): per-layer L*_ℓ from `run01/lstar_per_layer.json` reported as a sensitivity reference; NOT used for the pre-registered verdict.

If yes, the gate becomes a candidate central mechanism for the paper's communication-centric thesis (L_recv as the per-step decision variable), pending downstream online validation.

> **Scope statement (Codex round 2b)**:
> - The pilot makes a **proxy-quality replay claim** ("operating-point dominance at the registered point"), NOT "strict Pareto dominance" (Codex attack #3).
> - The pilot does **not** validate accuracy — it uses a router-weight-mass *proxy* (Codex #6). Online task-metric validation is a stretch goal, not a verdict gate.
> - The L_recv gate uses the **logged** L_recv (taken from a no-drop execution). It does NOT model the feedback effect (dropping → changed future L_recv at later steps); this is acknowledged as "trace-level proxy replay" (Codex attacks #1, #2).
> - Bytes saved = dropped-cell count × per-cell bytes. Real NCCL byte savings may differ due to padding/alignment/headers; this is NOT a latency claim. Latency comes from D1-I10's microbench.

## Pre-registered Claims (immutable; numeric thresholds frozen at this date)

### C2.main — Pareto dominance over static SOTA on long-prompt workloads
> Aggregated over passage_retrieval_v2 + multifieldqa_v2 (held-out 30% replay split, see §Method), the L_recv-gated policy achieves:
> - **bytes saved ≥ 80%** of static tail_weight @ r=0.3's bytes saved, AND
> - **weight-mass-loss proxy ≤ 50%** of static tail_weight @ r=0.3's weight-mass-loss proxy.

- **Failure handling**:
  - Either condition fails → C2.main **REFUTED**. Action: *scope-limit* the gate from "L_recv-gated dominates static" to "L_recv-gated provides comparable Pareto", or *downgrade* to a regime-conditional claim (specific regime to be reported).
  - Both conditions pass → C2.main **SUPPORTED** (pending negative-control C2.neg).

### C2.neg — decode-heavy negative control behaviour (tightened per Codex attack #25)
> On gsm8k_clean_v2 held-out 30% replay split, the L_recv-gated policy:
> - **upper 95% one-sided binomial CI** on gate-open rate (per measured step) < **5%** (with ≥ 50 steps × 8 ranks × 48 layers ≈ ≥ 19,200 trials, a true rate of 0% / 1% gives a UCB < 0.02% / < 1.6% respectively — well within the bound), AND
> - **weight-mass-loss proxy ≤ 1pp** of total measured-step mass.

- **Failure handling (gate opens too often on GSM8K)** → calibration of `L*_ℓ` is too low for the regime; *scope-limit* the gate's applicability to long-prompt workloads only; document the regime gap. Does NOT refute C2.main but tightens the deployment story.

### C2.calib — calibration robustness
> The `L*_ℓ` values used in this pilot are the values **output by D1-I10's microbench** (no post-hoc retuning). The calibration set is the 70% pre-registered split; the replay set is the held-out 30%.

- **Failure handling (calibration is post-hoc-tuned)** → ❌ the pilot is invalid; rerun with frozen calibration before any verdict.

### C2.sample — statistical scope
> The replay set contains ≥ **150 unique measured steps** across the 2 long-prompt workloads AND ≥ 50 measured steps in the gsm8k negative control after the 30% split.

- **Failure handling** → INCONCLUSIVE; extend split or augment data before claiming.

## Method

### Inputs
- v2 jsonl + sidecar across 3 workloads (measured records: `is_warmup==False`)
- `L*_ℓ` table from `eval/d1_i10_bytes_latency_microbench.py` output (one scalar per layer ∈ {0..47})

### Pre-registered data split (consistent with D1-I10, revised per Codex round 2a/b attack #13)
- Deterministic by `(workload, batch_index) mod 10 < 7` → **calibration set** (used by D1-I10 only — for L*_ℓ derivation).
- else → **D2-I5 replay set** (30%, used here for Pareto evaluation).
- Splitting by `batch_index` (= LongBench prompt or GSM8K sub-batch) holds out entire prompts, not nearby steps. Eliminates the periodic-block leak Codex flagged.
- **Phase stratification check** (Codex attack #14): after the split, the replay set must contain ≥ 20% prefill steps in long-prompt workloads (if not, the split is rebalanced by adding the lowest-batch_index prefill prompts to replay until 20%). Reported in `split_diagnostics.json`.
- Seed = **42** — used ONLY by stochastic baselines (B-MR random selection, B-RS rate sweep tie-breaking). The split itself is deterministic.

### Pipeline — `eval/d2_i5_replay.py` (NEW, in worktree)
1. **Load** all measured v2 records from the 30% replay split. For each record, load `topk_ids` and `topk_weights` from sidecar via `topk_offset`.
2. **Per policy**, simulate drop decisions per (step, layer, rank) record:
   - For each policy and each record, determine the kept-mask over the T·K replicas, then compute:
     - `bytes_saved` = (T·K − Σ kept) × H × elem_size (cells dropped × per-cell bytes)
     - `weight_mass_loss` = Σ_{dropped (t,k)} `topk_weights[t,k]`
     - `total_weight_mass` = Σ_{all (t,k)} `topk_weights[t,k]`
     - `weight_mass_loss_frac` = weight_mass_loss / total_weight_mass
3. **Aggregate** per workload and globally: total_bytes_saved, total_weight_mass_loss, and the (bytes, mass-loss) Pareto point per policy.
4. **Decision rule for each policy** (the kept-mask computation):
   - Reuse the *existing* `expert_drop.apply_drop_cpu` for `tail_weight` and `random` policies — feed it the topk_ids / topk_weights arrays plus a synthesized `target_rank` (computed offline: contiguous → `e // E_local`; LBG → load from overlap plan + re-route; same overlap_router class instantiated read-only). This keeps the simulation faithful to the production drop semantics.
   - **L_recv-gated**: a thin Python wrapper that calls `apply_drop_cpu("tail_weight", r=0.3, ...)` iff `L_recv_step_ℓ > L*_ℓ`; else returns identity mask.
   - **random @ matched-bytes**: per step, count L_recv-gated's dropped cells; randomly drop the same count from droppable cells (matched per (step, layer, rank); seeded run-wide).
   - **oracle per-phase**: prefill records → tail_weight@r=0.3 mask; decode records → identity mask.

### Comparison Groups (FINALIZED after Codex round 1; see `traces/d1_d2_ablation_design_codex.md`)

**MUST-RUN (in pre-registered RESULTS table):**
| ID | Mechanism | Defends against |
|---|---|---|
| **B0** | no-drop (identity mask everywhere) | sanity / Pareto origin |
| **B1** | static tail_weight @ r=0.3 (production SOTA) | the comparison target |
| **B-RS** | static tail_weight rate sweep r ∈ {0.05, 0.10, 0.20, 0.30, 0.40, 0.50} | "r=0.3 cherry-picked"; subsumes original B2 |
| **B-LS** | L_send-gated tail_weight @ r=0.3 (gate on prompt-token count, NOT recv) | "it's just long prompts, not recv-side pressure" |
| **B-PG** | phase-gated tail_weight @ r=0.3 (prefill yes / decode no) | "it's just phase awareness" |
| **B-GL** | global L* = 3271 gated tail_weight @ r=0.3 (vs per-layer L*_ℓ in EXP) | "per-layer L*_ℓ tuning is overfit" |
| **B-MB** | matched-bytes static tail_weight (per-workload static rate calibrated to match EXP's bytes_saved) | "you only win because you save fewer bytes" |
| **B-MR** | matched-bytes random drop (user's original B3 spec) | "any sparse drop would work; nothing special about L_recv" |
| **B-TH** | L_recv-gated × threshold sweep α ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0} (gate fires when L_recv > α·L*_ℓ) | "L* cherry-picked" |
| **B-DR** | L_recv-gated × drop-rate sweep r ∈ {0.1, 0.2, 0.3, 0.4, 0.5} | "r=0.3 cherry-picked WITHIN gated policy" |
| **EXP** | **L_recv-gated dynamic, per-layer L*_ℓ from D1-I10, r=0.3** | central candidate |
| **ORC-PP** | per-phase static (prefill: tail_weight r=0.3; decode: identity) | user-requested upper bound (phase-aware ceiling) |
| **ORC-LB** | loss-budget oracle (drop minimum router-weight-mass per byte, matched to EXP bytes) | absolute headroom upper bound |

**STRETCH (priority 2-3; run only if main verdict survives):**
- top-N L_recv budget gate (gate by ranking, not threshold)
- L_recv-gated × random drop selection (separates gate from heuristic)
- L_recv-gated × other drop policies (weighted_tail, hot_expert_relief, hotspot_relief)
- send/recv imbalance gate (alternative trigger feature)

Diff vs author's original 6: B0 / B1 / EXP / ORC-PP retained; B2 replaced by B-RS rate sweep (denser, no loss); B3 split into two distinct controls (B-MB for matched-bytes static, B-MR for matched-bytes random — Codex's #8 was a cleaner test than CC's original B3); 5 new must-runs (B-LS, B-PG framing, B-GL, B-TH, B-DR) added to defend specific reviewer attacks; ORC-LB added as a true oracle distinct from ORC-PP.

## Setup
- Hardware: CPU only (no GPU). Should fit in a single Python process.
- Software: numpy + the existing `expert_drop` module (read-only import)
- Worktree: same as D1-I10
- Random seeds: split seed = 42; random-policy seed per record = `hash((rank, step_id, layer_id)) & 0xFFFFFFFF`

## Metrics (decides C2.main / neg / calib)

### Primary
- Pareto point (bytes_saved, weight_mass_loss_frac) per policy, aggregated per workload
- C2.main check: `EXP.bytes_saved ≥ 0.8 × B1.bytes_saved` AND `EXP.weight_mass_loss_frac ≤ 0.5 × B1.weight_mass_loss_frac`
- C2.neg check on gsm8k: gate-open fraction, total weight_mass_loss_frac

### Secondary (Codex attacks #8, #28: aggregate hides worst cases)
- per-layer gate-open rate (which layers' gates fire when)
- per-rank bytes_saved and per-rank weight_mass_loss (Codex #28: rank-aggregate hides stragglers / imbalance)
- per-phase (prefill / decode) bytes_saved and weight_mass_loss per policy
- distribution of `L_recv − L*_ℓ` (signed gap) — shows how often the gate is near the boundary
- **per-step weight_mass_loss percentiles (p50 / p90 / p99 / max)** for EXP vs B1 (Codex #8: average hides catastrophic steps)
- per-batch_index clustered bootstrap CIs (1000 iters, clusters = batch_index)

### Stretch goal (NOT part of pre-registered verdict)
- Independent **dataset GT** evaluation via LongBench official metric: run a separate GPU job with `L_recv-gated` drop enabled in DispatchEPHT (would require small algorithm edit — out of D2-I5 pilot scope; deferred to post-pilot if time allows).

## Fail-fast Conditions
1. **`L*_ℓ` table from D1-I10 missing or includes layer ℓ with NaN** → STOP; rerun D1-I10 calibration first.
2. **Replay set ends up < 8 distinct prompts** in either long-prompt workload, or < 4 prompts in gsm8k after the split → STOP; extend (Codex attack #16: step counts are correlated; the right unit is *prompts*).
3. **EXP policy yields negative bytes_saved or quality_proxy > 1.0** → bug, STOP.
4. **expert_drop API breaks under offline use** → STOP.
5. **EXP-vs-online drop-mask equivalence test fails** (Codex attack #17): on a random sample of 20 records, the offline EXP kept-mask must agree byte-for-byte with the production `apply_drop_cpu(tail_weight, r=0.3, ...)` output when fed the same (topk_ids, topk_weights, target_rank) — covers the "protected branch" semantics. Any disagreement → STOP, fix the offline wrapper.
6. ~~**LBG offline target_rank reconstruction mismatch > 1%**~~ **AMENDED 2026-05-30** (see Changelog): LBG offline simulation uses primary-owner approximation. Mean cell-assignment divergence from production LBG measured at 37.5% (passage_retrieval_v2) / 38.5% (multifieldqa_v2) / 0% (gsm8k_clean_v2). All 13 policies use the same offline assumption, ensuring fair relative ordering. Documented as Honest Limitation. NOT a fail-fast.

## Risks + Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **weight-mass-loss proxy ≠ true accuracy loss** | high | high (paper claim ceiling) | Pre-register and ALWAYS report as "weight-mass-loss proxy"; never as accuracy. Stretch-goal GT eval addresses it but is NOT required for C2 verdict |
| **L*_ℓ noisy from thin D1-I10 sampling** → gate decisions flip-flop on near-boundary steps | med | med | Apply hysteresis or report sensitivity sweep (L*_ℓ × {0.8, 1.0, 1.2}) |
| **Offline target_rank computation for LBG drifts from production** | med | high | Verify on first 100 records that offline `bincount(target_rank) == jsonl.send_counts` exactly (it MUST for contiguous; for LBG accept ≤ 1% mismatch and document) |
| **70/30 split puts most prefill steps in calibration** | low | med | Stratify split by phase; pre-register stratified split |
| **B3 random @ matched-bytes shows EXP is no better** | medium | high (kills "L_recv-gated is meaningful" claim) | If this happens, paper conclusion shifts: the value is in the byte budget, NOT the gate's location. This is a legitimate negative result — document, don't fudge. |
| **expert_drop.apply_drop_cpu's "protected branch" logic** filters which (t,k) pairs are droppable; offline must replicate exactly | high | med | Read existing implementation and replicate; assert kept-mask matches one round-trip with the actual function |

## Expected RESULTS schema (`analyze-results` 4-tuple per finding)

For each of C2.main / C2.neg:
```
Observation:    Pareto points table + relative ratios vs B1, with 95% bootstrap CIs on aggregates
Interpretation: which mechanism is responsible — gate location vs byte budget vs phase awareness
Implication:    paper's main mechanism claim status (supported / scope-limited / refuted)
Next-step:      stretch-goal GT eval if SUPPORTED; framing revision if REFUTED
```

## Resource Estimate
- Loading + replay (CPU): ≈ 5–15 min for full 30% replay × 6 policies on a single core
- Analysis + plots: < 5 min
- **No GPU needed.**

## Known Limitations (Stage D0 carryover)
- Quality is a **router-weight-mass proxy**, not measured task accuracy. Stretch-goal LongBench GT eval is a separate post-pilot run if time allows.
- send_counts reconstruction from topk_ids was directly validated only in **contiguous** mode (gsm8k_clean_v2 → 0 mismatches). LBG mode (passage_retrieval_v2 + multifieldqa_v2) relies on the **v1↔v2 byte-identical** evidence (52,224 + 73,728 records on rank0, 100% match under greedy decode) as indirect proof that the topk-recording patch did not perturb LBG routing.
- **Implication for debugging**: if D2-I5's L_recv-gated replay produces routing inconsistent with the recorded send_counts, suspect (a) calibration noise or (b) offline overlap_router state drift — NOT the v2 data source.

## Honest Limitations (ex-ante risks identified BEFORE running, per GATE D1 user request)

### Risk-1 (C2.main 50% mass-loss bound may be structurally hard on long-prompt)
- Long-prompt workloads concentrate ~99.4% of (token, expert) cells in prefill steps. EXP's gate effectively reduces drop applications only on decode steps (low L_recv) — a tiny fraction of total cells.
- B1 (static tail_weight @ r=0.3) drops 30% of cells UNIFORMLY across all steps. The bulk of B1's weight-mass loss comes from prefill cells (the ones EXP also drops).
- Therefore, EXP can only "save" the mass loss of the ~0.6% decoded cells from being dropped. For EXP's loss to be ≤ 50% of B1's, the saved 0.6% of cells would need to carry ≥ 50% of B1's mass-loss — implausible unless decode router weights are extremely concentrated.
- **Mathematically plausible scope-limit outcome**: C2.main bytes condition (≥ 80%) passes easily; mass-loss condition (≤ 50%) fails on long-prompt → registered scope-limit to "comparable Pareto at fixed bytes".
- **We commit to NOT retuning the 50% threshold post-hoc.** Failure goes to scope-limit per registration.

### Risk-2 (router-weight-mass is a proxy, NOT accuracy)
- Acknowledged throughout the PLAN; stretch goal is the only path to accuracy validation.
- A reviewer may demand accuracy as a verdict gate — pre-registration says no. Stretch goal addresses it post-verdict.

### Risk-3 (offline replay does NOT model feedback)
- Dropping in EXP changes future hidden states → future L_recv → potentially a different gate decision; the offline replay uses LOGGED L_recv. The PLAN scope-limits to "trace-level proxy replay" explicitly. Honest reviewers may push for short-horizon recomputation; logged for post-pilot online validation.

### Risk-4 (D2-I5 result depends on D1-I10's L*_ℓ stability)
- If D1-I10 outputs noisy per-layer L*_ℓ, D2-I5's gate decisions on near-boundary steps could be unstable. B-TH threshold sweep (α ∈ {0.5, ..., 2.0}) lets us quantify sensitivity post-hoc.
- Threshold-sensitivity report is informational; EXP at α=1.0 is the pre-registered point.

## Devil's Advocate Review

### Round 1 — Ablation design (Codex independent proposal vs Claude's draft)
See `traces/d1_d2_ablation_design_codex.md`. Codex proposed 14 comparison groups; Claude originally had 6 (B0, B1, B2, B3, EXP, ORC). Result: Claude integrated 7 of Codex's groups into MUST-RUN (B-RS, B-LS, B-PG, B-GL, B-MB, B-TH, B-DR), kept ORC-PP from original, added ORC-LB. Final must-run = 13 groups. Stretch (priority 2-3) = 4 more.

### Round 2b — PLAN attack (30 attacks; key responses below)
Source: `traces/d2_i5_plan_attack.md`. Numbers reference Codex's attack list.

#### Accepted (PLAN body edited)

| # | Codex attack | Claude response + edit |
|---|---|---|
| 1 | Offline replay is not counterfactual (drops change future routing) | **Accept**. Research Question rewritten as "trace-level proxy replay" with explicit scope statement. NOT a policy evaluation. |
| 2 | L_recv gate is endogenous | **Accept**. Scope statement says we use *logged* L_recv from no-drop run; do not model feedback. Stretch goal would address via online runs. |
| 3 | "Strictly better Pareto" overclaimed | **Accept**. Research Question now says "operating-point dominance at the pre-registered point", not Pareto dominance. |
| 8 | Aggregate hides catastrophic steps | **Accept**. Added per-step p50/p90/p99/max weight_mass_loss reporting. |
| 13 | Split (step_id//5) leaks | **Accept**. Split changed to (workload, batch_index) mod 10. |
| 14 | 70/30 not stratified by phase | **Accept**. Added phase-stratification check with rebalance rule. |
| 17 | EXP protected-branch semantics may not be replicated | **Accept (now hard fail-fast)**. New fail-fast #5: byte-for-byte mask equivalence test vs `apply_drop_cpu`. |
| 18 | LBG reconstruction weak (now flagged as risk) | **Accept (now hard fail-fast)**. New fail-fast #6: > 1% LBG target_rank mismatch → STOP + exclude LBG or fix. |
| 19 | Bytes saved ≠ NCCL bytes | **Accept**. Scope statement now says bytes_saved is dropped-cell-count × per-cell bytes, NOT a latency or NCCL-byte claim. |
| 25 | <1% with 50 steps ≈ exact zero | **Accept**. C2.neg now uses upper 95% binomial CI < 5% on per-(step,rank,layer) trial (≥ 19,200 trials). |
| 27 | No statistical uncertainty | **Accept**. Per-batch_index clustered bootstrap CIs (1000 iters) on every reported aggregate. |
| 28 | Rank/layer aggregation hides regressions | **Accept**. Per-rank and per-layer breakdowns required in Secondary metrics. |

#### Accepted with scope caveat

| # | Codex attack | Claude response |
|---|---|---|
| 6 | Weight-mass-loss is weak proxy for accuracy | **Partial accept**. Pre-registered as a *proxy* (already in the user's original framing), every Claim explicitly says "router-weight-mass-loss proxy" not "accuracy". Stretch goal: small online task-metric validation on a subset (not a verdict gate; takes 1-2 GPU-hours). |
| 22 | B-TH risks post-hoc rescue | **Partial accept**. Diagnostic-only label on B-TH (already implied; made explicit). EXP point is α=1.0; B-TH is sensitivity. |
| 4 | Static r=0.3 may be strawman | **Accept via B-RS rate sweep**. EXP must beat the **best** static-tail point at matched bytes; B-MB matched-budget static directly tests this. |
| 23 | ORC-LB framing may confuse achievable vs hindsight | **Accept**. ORC-LB labeled as "absolute headroom upper bound" — not a baseline. |

#### Rejected (with stated reason)

| # | Codex attack | Claude response |
|---|---|---|
| 5 | 80% / 50% thresholds arbitrary | **Reject**. User pre-registered these and pre-registered "严禁修改 pre-registered 判据数值". Bootstrap CIs let reviewers judge the operating point themselves. |
| 26 | GSM8K is too easy a negative control | **Reject as blocker, accept as caveat**. GSM8K is the *user's* requested negative control for "decode-heavy structurally no benefit"; it does NOT need to be hard. Adding a long-context reasoning negative control (e.g. LongBench narrativeqa) is logged as stretch. |
| 30 | No latency claim despite byte motivation | **Reject as gap, accept as scope**. Latency is D1-I10's domain (separate pilot). D2-I5 explicitly does NOT claim latency improvements — only proxy-quality and bytes-saved on logged traces. |

#### Unresolved (will be reported at GATE D3 RESULTS review)
- Codex #11 "L*_ℓ may not transfer from microbench to workload" — diagnostic: report distribution of `L_recv − L*_ℓ` and the gate-open rate as a function of how close to L*_ℓ each step lands. If most steps are far from the threshold, transferability concern is mostly moot.
- Codex #12 "Per-layer L*_ℓ may encode layer scale, not congestion" — diagnostic: compare per-layer L*_ℓ values against per-layer L_recv distribution shape; if L*_ℓ correlates strongly with per-layer L_recv mean (rather than tail), report as a finding.
