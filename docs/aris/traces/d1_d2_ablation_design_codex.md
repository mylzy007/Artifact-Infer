# Trace — Stage D1 Round 1: Codex Independent D2-I5 Ablation Design

- Date: 2026-05-29
- Reviewer: Codex (`mcp__codex__codex`), `model_reasoning_effort=xhigh`, sandbox=read-only
- Thread ID: 019e7427-4aff-7d82-98d2-ae17ea9b79ec
- cwd: `/home/lzy/Artifact-Infer-pilot` (worktree pilot/d1-i10-d2-i5)
- 依据 skill: `ablation-planner` (Codex leads design, CC does NOT pre-filter)

## What Codex was given
Pilot D2-I5's research question, the trace fields available, success criteria (NOT the author's comparison groups). Asked to propose 4-8+ groups designed as reviewer-attack defenses.

## What Codex returned (14 groups, summarized)

| # | Codex name | Priority | Reviewer attack it defends |
|---|---|---|---|
| 1 | static_tail_r03 | 1 | baseline reference (= author's B1) |
| 2 | static_tail_rate_sweep (r∈{0.05,0.10,0.20,0.30,0.40}) | 1 | "you only beat over-aggressive static" |
| 3 | Lsend_gated_tail_r03 | 1 | "it's just long prompts (T·K), not recv-side pressure" |
| 4 | phase_gated_tail_r03 | 1 | "it's just phase awareness" |
| 5 | global_Lstar_gated_tail_r03 | 1 | "per-layer L*_ℓ tuning is overfit; global L* is enough" |
| 6 | threshold_sweep_Lrecv_tail_r03 (α∈{0.5,...,2.0}) | 1 | "L* was cherry-picked" |
| 7 | drop_rate_sweep_Lrecv | 1 | "r=0.3 was cherry-picked" |
| 8 | matched_budget_static_tail | 1 | "you only win because you save fewer bytes" |
| 9 | matched_open_random_gate_tail | 2 | "any sparse application would work, not specifically L_recv" |
| 10 | top_Lrecv_budget_gate_tail | 2 | "break-even L* unnecessary; top-N L_recv enough" |
| 11 | Lrecv_gated_random_r03 | 2 | "result is from tail_weight selection, not gate" |
| 12 | Lrecv_gated_policy_family | 2 | "hidden interaction with one drop policy" |
| 13 | send_recv_imbalance_gate | 3 | "mechanism is a2a imbalance, not total recv volume" |
| 14 | oracle_loss_budget_gate | 3 | upper-bound headroom |

## Diff: Claude's original 6 vs Codex's proposal

| Claude's group | Status in Codex list | Comment |
|---|---|---|
| B0 no-drop | (implicit baseline) | Codex assumes; CC should keep |
| B1 static tail r=0.3 | = Codex #1 | overlap, keep |
| B2 static tail r=0.5 | subsumed by Codex #2 rate sweep | redundant; replace with rate sweep |
| B3 random @ matched-bytes | ≈ Codex #9 (different semantics) | Codex's #9 = random gate + tail_weight; CC's = random cells; Codex's #8 matched_budget_static_tail is the cleaner "matched-bytes" test |
| EXP L_recv-gated | = Codex's central candidate | overlap, keep |
| ORC per-phase static | ≈ Codex #4 phase_gated_tail_r03 | overlap; CC mixed phase + bypass, Codex isolates phase as gate |

| Codex extras (priority 1) | CC originally had it? | CC's verdict |
|---|---|---|
| #2 static_tail_rate_sweep | partial (B1, B2 only) | **add** — denser grid replaces B2 |
| #3 Lsend_gated_tail_r03 | ❌ | **add** — critical confound (long-prompt vs recv pressure) |
| #5 global_Lstar_gated_tail_r03 | ❌ | **add** — directly relevant since user picked per-layer pipeline (decision C) |
| #6 threshold_sweep_Lrecv_tail_r03 | ❌ | **add** — cherry-pick defense |
| #7 drop_rate_sweep_Lrecv | ❌ | **add** — cherry-pick defense |
| #8 matched_budget_static_tail | partial (B3 different) | **add** — Codex's framing is cleaner than B3 random-cell |

| Codex extras (priority 2-3) | CC originally had it? | CC's verdict |
|---|---|---|
| #9 matched_open_random_gate_tail | ≈ B3 | **keep as B3 spirit**: implements user's "random @ matched-bytes" precisely |
| #10 top_Lrecv_budget_gate_tail | ❌ | **defer to stretch**: useful diagnostic, P2 |
| #11 Lrecv_gated_random_r03 | ❌ | **defer to stretch** |
| #12 Lrecv_gated_policy_family | ❌ | **defer to stretch**: large compute, narrower attack |
| #13 send_recv_imbalance_gate | ❌ | **defer to stretch** |
| #14 oracle_loss_budget_gate | ≈ ORC differently | **add ORC variant**: keep user's per-phase ORC AND add Codex's loss-budget oracle as an upper-bound point |

## Claude's proposed final comparison group set (for user confirmation)

**MUST-RUN (priority 1, in pre-registered RESULTS):**
| ID | Mechanism | Defends against |
|---|---|---|
| **B0** | no-drop | sanity / Pareto origin |
| **B1** | static tail_weight @ r=0.3 | SOTA reference, the comparison target |
| **B-RS** | static tail_weight rate sweep r∈{0.05,0.1,0.2,0.3,0.4,0.5} | "r=0.3 cherry-picked" / Pareto of static |
| **B-LS** | L_send-gated tail_weight @ r=0.3 | "it's just long prompts (T·K), not recv-side pressure" |
| **B-PG** | phase-gated tail_weight @ r=0.3 (prefill drop / decode skip) | "it's just phase awareness" |
| **B-GL** | global L*=3271 gated tail_weight @ r=0.3 (vs per-layer L*_ℓ in EXP) | "per-layer L*_ℓ tuning is overfit" |
| **B-MB** | matched-bytes static tail_weight (per-workload rate matches EXP's bytes_saved) | "you only win because you save fewer bytes" |
| **B-MR** | matched-bytes random drop (user's original B3) | "any sparse drop would work, not specifically L_recv" |
| **B-TH** | L_recv-gated threshold sweep α∈{0.5,0.75,1.0,1.25,1.5,2.0} | "L* cherry-picked" |
| **B-DR** | L_recv-gated drop-rate sweep r∈{0.1,0.2,0.3,0.4,0.5} | "r=0.3 cherry-picked WITHIN gated policy" |
| **EXP** | L_recv-gated dynamic (per-layer L*_ℓ, r=0.3) | central candidate |
| **ORC-PP** | per-phase static (prefill: tail_weight r=0.3, decode: identity) | user-requested upper bound |
| **ORC-LB** | loss-budget oracle (drop minimum mass per byte budget, matched to EXP bytes) | absolute headroom upper bound |

= 13 groups (B0..ORC-LB). User's original 6 (B0/B1/B2/B3/EXP/ORC) all preserved or replaced with cleaner equivalents (B2 → B-RS; B3 → B-MR; ORC → ORC-PP).

**STRETCH (priority 2-3, run only if main verdict survives):**
- top-N L_recv budget gate
- L_recv-gated × random drop selection
- L_recv-gated × other policies (weighted_tail, hot_expert_relief, hotspot_relief)
- send/recv imbalance gate

**Open questions for user confirmation**:
1. Accept the diff above? B-LS, B-GL, B-MB, B-TH, B-DR are the substantive additions that defend against specific reviewer attacks Claude's original 6 did NOT cover.
2. Anything Codex listed that the user wants moved between priority tiers?

## Compute estimate for must-run set
~13 policies × ~1 CPU-min per full replay = **15-25 min CPU**. Sweep policies (B-RS, B-TH, B-DR) are multi-point but cheap; still under 30 min total.
