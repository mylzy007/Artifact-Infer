# Pre-Token-Owner MoE Exploration Plan

This document is a detailed plan for the exploration phase **before** refactoring the runtime to an explicit token-owner / source-owner architecture.

The target audience is an engineering agent or researcher who needs a concrete execution plan that can be used directly as a working prompt.

---

## 0. Goal

Before rewriting the runtime around token ownership, we want to answer a narrower question:

> Under the current runtime, which MoE optimization directions show enough signal to justify a token-owner refactor?

The idea is to spend a small amount of engineering effort to identify which of the following directions are promising:

- dynamic top-k
- decode-only routing simplification
- routing reuse / prediction
- token drop / expert drop
- static placement
- load-aware placement
- expert replication
- alternate dispatch styles

We do **not** want to redesign the whole runtime yet. We want to cheaply measure:

1. which metrics are trustworthy in the current runtime,
2. which optimization directions still show signal despite current duplication artifacts,
3. which directions are blocked by the lack of a token owner.

---

## 1. Scope and non-goals

### In scope

- add diagnostics to understand how bad current duplication is
- improve profiling granularity
- run small to medium sweeps over current configurations
- build "oracle" style offline analyses to estimate upper bounds for future methods
- produce a decision about whether a direction needs token-owner semantics

### Out of scope

- fully implementing token-owner runtime
- redesigning scheduler ownership semantics
- integrating DP/SP/CP/PP end-to-end
- claiming final production-quality MoE performance

---

## 2. Current runtime facts to assume

These facts should be treated as the baseline model of the current repo:

- `pure EP` with `tp_size=1` replicates non-expert layers across EP ranks.
- under some `ws8` runs, routing profiles can degenerate because multiple EP ranks observe identical token batches before dispatch.
- `EP-HT` currently uses eager-mode variable-size all-to-all dispatch.
- `TP×EP` is already supported and should be used as an important exploration axis.
- current GSM stage01 runs are good enough to compare relative runtime trends, but answer accuracy is partially confounded by generation truncation and format issues.

Relevant files:

- [workshop/nanovllm_moe/services/engine/llm_engine.py](/home/lzy/Artifact-Infer/workshop/nanovllm_moe/services/engine/llm_engine.py:1)
- [workshop/nanovllm_moe/services/model_runner/model_runner.py](/home/lzy/Artifact-Infer/workshop/nanovllm_moe/services/model_runner/model_runner.py:1)
- [workshop/nanovllm_moe/services/utils/routing_profile.py](/home/lzy/Artifact-Infer/workshop/nanovllm_moe/services/utils/routing_profile.py:1)
- [workshop/nanovllm_moe/services/utils/expert_placement.py](/home/lzy/Artifact-Infer/workshop/nanovllm_moe/services/utils/expert_placement.py:1)
- [workshop/nanovllm_moe/artifacts/modeling/layers/moe/dispatch_ep_ht.py](/home/lzy/Artifact-Infer/workshop/nanovllm_moe/artifacts/modeling/layers/moe/dispatch_ep_ht.py:1)
- [eval/run_moe_stage01_gsm.py](/home/lzy/Artifact-Infer/eval/run_moe_stage01_gsm.py:1)
- [eval/run_moe_placement_experiments.py](/home/lzy/Artifact-Infer/eval/run_moe_placement_experiments.py:1)

---

## 3. Exploration philosophy

The exploration should be divided into two classes.

### Class A: current-runtime-safe explorations

These can be explored meaningfully **without** token-owner refactor:

- top-k reduction / dynamic top-k approximations
- decode-only top-k / routing heuristics
- simple drop strategies
- TP×EP geometry tradeoffs
- placement baselines
- richer per-layer / per-step metrics

### Class B: token-owner-sensitive explorations

These can be probed offline or via proxies, but should **not** be fully trusted until token-owner exists:

- expert replica placement
- replica selection policy
- routing prediction conditioned on source locality
- source-to-expert communication optimization
- owner-aware drop policies

The pre-token-owner phase should still gather evidence for Class B, but should label all such evidence as:

> directional / proxy / oracle-only

---

## 4. Deliverables of the exploration phase

At the end of this phase we should have:

1. a trustworthy baseline table over several runtime geometries
2. a profile quality report explaining which metrics are reliable and which are contaminated
3. a small number of promising optimization candidates
4. a clear recommendation: which direction should motivate token-owner refactor

Concrete artifacts to produce:

- `docs/moe/pre_token_owner_findings_*.md`
- `eval_results/pre_token_owner_exploration_*/`
- CSV/JSON result tables with:
  - runtime metrics
  - score metrics
  - routing-profile quality metrics
  - placement metrics
  - top-k/drop ablation metrics

---

## 5. Phase breakdown

### Phase 1. Profile quality audit

Objective:

- quantify how distorted the current routing profile is under different geometries

Questions:

- are `traffic[layer][src][expert]` rows identical across sources?
- if not identical, how different are they?
- does TP reduce profile degeneracy?
- are decode and prefill behaving differently?

Required code work:

1. extend `routing_profile.py` to export richer diagnostics
2. extend analysis scripts to summarize those diagnostics

Metrics to add:

- `row_identical_fraction_per_layer`
- `row_pairwise_l1_distance_per_layer`
- `row_pairwise_cosine_similarity_per_layer`
- `row_entropy_per_src_per_layer`
- `prefill_vs_decode_traffic` split if feasible

Recommended file changes:

- [workshop/nanovllm_moe/services/utils/routing_profile.py](/home/lzy/Artifact-Infer/workshop/nanovllm_moe/services/utils/routing_profile.py:1)
- add a new analysis script under `eval/`, for example:
  - `eval/analyze_routing_profile_quality.py`

Minimal implementation requirement:

- do not change runtime semantics
- only add counters / exports
- avoid large GPU syncs beyond what is already present

Suggested experiment matrix:

- `EP=8, TP=1`
- `EP=4, TP=2`
- `EP=2, TP=4`

For each geometry:

- run 1 short GSM smoke
- run 1 medium GSM run
- compare row identity metrics

Success criterion:

- prove whether profile degeneracy disappears, weakens, or persists when TP is enabled

---

### Phase 2. TP×EP baseline geometry sweep

Objective:

- identify which geometry should serve as the pre-token-owner baseline

Questions:

- how much repeated dense/attention compute disappears when `tp_size` increases?
- how much placement headroom remains when `ep_size` shrinks?
- which geometry gives the best balance between efficiency and MoE-control signal?

Required code work:

- mostly scripting, not core runtime changes

Suggested geometries on 8 GPUs:

- `TP=1, EP=8`
- `TP=2, EP=4`
- `TP=4, EP=2`

Recommended workload settings:

- `moe_impl=ep_ht`
- `enforce_eager=1`
- GSM stage01
- `num_problems=112` for smoke-medium
- `num_problems=256` for full comparison
- `max_tokens=128`
- `max_num_seqs=8`
- `max_model_len=512`
- `max_num_batched_tokens=512`

Metrics to collect:

- `e2e_total_time_s`
- `prefill_throughput_tok_s`
- `decode_throughput_tok_s`
- `total_generated_tokens`
- `average_score`
- profile quality metrics from Phase 1
- `gpu_cv`
- `gpu_cv_layer_mean`
- `cross_traffic_ratio`
- `cross_traffic_ratio_layer_mean`

Success criterion:

- choose one geometry for future method development

Recommended decision rule:

- if `TP=2, EP=4` materially reduces duplication while retaining placement spread, choose it
- if `TP=4, EP=2` collapses placement differences too much, reject it as main research geometry

---

### Phase 3. Cheap top-k exploration

Objective:

- determine whether routing budget reduction has real signal before doing prediction or owner-aware methods

Questions:

- how much speedup do we get from reducing active experts?
- how sensitive are score and runtime to lower `K`?
- is decode more robust to top-k reduction than prefill?

Required code work:

Add runtime flags to support:

- force `K=1`
- force `K=2`
- force `K=4`
- baseline `K=8`

Possible implementation style:

- add an inference-time override in config, e.g. `moe_top_k_override`
- plumb it into `FusedMoE` / dispatch modules without changing model weights

Relevant files:

- [workshop/nanovllm_moe/services/config.py](/home/lzy/Artifact-Infer/workshop/nanovllm_moe/services/config.py:1)
- [workshop/nanovllm_moe/artifacts/modeling/layers/moe/fused_moe.py](/home/lzy/Artifact-Infer/workshop/nanovllm_moe/artifacts/modeling/layers/moe/fused_moe.py:1)

Suggested runs:

- chosen geometry from Phase 2
- placements:
  - contiguous
  - round_robin
  - load_balanced_greedy_with_locality_tiebreak
- `K in {1,2,4,8}`

Success criterion:

- identify whether reduced top-k is viable
- determine whether any future routing-prediction work should target rank prediction, expert-group prediction, or exact top-k prediction

---

### Phase 4. Drop exploration

Objective:

- quantify whether selective dropping can buy meaningful decode speedup

Questions:

- what happens with random drop, low-weight drop, or capacity-style drop?
- is there a regime where tiny quality loss buys large throughput gains?

Required code work:

Introduce experimental drop policies at dispatch time:

- `none`
- `drop_low_weight_tail`
- `drop_if_rank_over_capacity`
- `drop_if_global_budget_exceeded`

Important:

- this phase does **not** need token owner yet if drop is purely local to current duplicated runtime
- but all conclusions should be labeled approximate

Potential implementation point:

- after `topk_weights/topk_ids` are computed in:
  - [dispatch_ep_ht.py](/home/lzy/Artifact-Infer/workshop/nanovllm_moe/artifacts/modeling/layers/moe/dispatch_ep_ht.py:147)
  - and/or EP-LL dispatch if also explored later

Metrics:

- dropped replica count
- dropped-token fraction
- score degradation
- e2e / decode gain

Success criterion:

- determine whether "drop" is a viable optimization family at all for this stack

---

### Phase 5. Routing reuse / prediction oracle

Objective:

- test whether routing is temporally stable enough to justify a predictor or reuse mechanism

This is the highest-value pre-token-owner proxy for future `predict expert` work.

Questions:

- for decode, how often does a token's routed rank set stay the same from step to step?
- how often does top-1 stay the same?
- how often does top-k set stay the same?
- can we predict next-step route from previous-step route with a simple heuristic?

Required code work:

Extend `routing_profile.py` or add a separate optional logger to capture per-step decode routing summaries:

- per-layer top-k ids histogram by decode step
- optionally per-token trace for a tiny debug run only

Do **not** log full per-token traces for large runs by default.

Use two modes:

- aggregate mode: safe for medium/full runs
- token-trace mode: tiny smoke only

Suggested cheap predictors to evaluate offline:

- reuse previous step's exact top-k
- reuse previous step's top-1 and fill rest by current gate
- predict only destination rank, not exact expert
- predict only coarse group then refine locally

Success criterion:

- decide whether routing prediction is promising enough to justify token-owner refactor

---

### Phase 6. Placement signal audit

Objective:

- determine whether current runtime is still useful for placement studies before token-owner refactor

Questions:

- do placements still measurably change:
  - `gpu_cv`
  - `gpu_cv_layer_mean`
  - decode throughput
  - score
- is the runtime mainly reacting to compute imbalance or communication?

Required code work:

- mostly analysis
- keep using current `estimated_metrics(...)`
- ensure per-layer metrics are included in placement JSON

Suggested analysis:

- compare rank-load balance vs runtime
- compare global cross vs runtime
- compare per-layer cross vs runtime
- compare score vs runtime

Success criterion:

- if placement still matters after TP is enabled, continue placement line
- if placement signal collapses, de-prioritize placement-only work and move toward routing simplification / prediction / replica

---

## 6. Prioritized experiment order

Do not do everything at once. Run in this order:

1. Phase 1: profile quality audit
2. Phase 2: TP×EP geometry sweep
3. Phase 3: top-k exploration
4. Phase 5: routing reuse / prediction oracle
5. Phase 4: drop exploration
6. Phase 6: placement signal audit

Why this order:

- Phase 1 tells us whether metrics are trustworthy
- Phase 2 chooses the right geometry
- Phase 3 and 5 expose the highest-potential method directions
- Phase 4 is cheap and useful but lower-priority than route reuse
- Phase 6 tells us whether placement remains worth deep investment

---

## 7. Concrete code changes before token-owner

### 7.1 Add profile-quality metrics

Files:

- [workshop/nanovllm_moe/services/utils/routing_profile.py](/home/lzy/Artifact-Infer/workshop/nanovllm_moe/services/utils/routing_profile.py:1)

Changes:

- keep current traffic export
- add optional analysis payloads at save time or in a post-processing script
- do not change inference semantics

### 7.2 Add richer placement metrics

Files:

- [workshop/nanovllm_moe/services/utils/expert_placement.py](/home/lzy/Artifact-Infer/workshop/nanovllm_moe/services/utils/expert_placement.py:1)
- [eval/generate_moe_placement.py](/home/lzy/Artifact-Infer/eval/generate_moe_placement.py:1)

Changes:

- already support per-layer cross/gpu_cv metrics
- keep both:
  - global aggregate metrics
  - per-layer mean metrics

### 7.3 Add top-k override

Files:

- [workshop/nanovllm_moe/services/config.py](/home/lzy/Artifact-Infer/workshop/nanovllm_moe/services/config.py:1)
- [workshop/nanovllm_moe/artifacts/modeling/layers/moe/fused_moe.py](/home/lzy/Artifact-Infer/workshop/nanovllm_moe/artifacts/modeling/layers/moe/fused_moe.py:1)
- dispatch modules if needed

Changes:

- add `moe_top_k_override: int = -1`
- if set, use it at inference instead of checkpoint default

### 7.4 Add drop-policy flags

Files:

- config
- dispatch modules

Suggested flags:

- `moe_drop_policy`
- `moe_drop_budget`
- `moe_drop_min_weight`

### 7.5 Add per-step routing trace mode

Files:

- `routing_profile.py`
- potentially `eval/` analysis helpers

Flags:

- `MOE_TRACE_ROUTING=1`
- `MOE_TRACE_MAX_STEPS=...`
- `MOE_TRACE_MAX_TOKENS=...`

### 7.6 Add a single experiment runner

Create one top-level script to orchestrate all pre-owner sweeps, for example:

- `eval/run_pre_token_owner_exploration.py`

Responsibilities:

- choose geometry
- run top-k sweeps
- run drop sweeps
- run profile audit
- write one unified summary

---

## 8. Recommended settings

### Geometry

Primary exploration geometry:

- `world_size=8`
- `tp_size=2`
- `ep_size=4`
- `moe_impl=ep_ht`
- `enforce_eager=1`

Rationale:

- reduces worst dense/attention duplication vs `TP=1, EP=8`
- keeps enough EP width for placement / routing experiments
- already supported by local code and docs

Secondary comparison geometries:

- `TP=1, EP=8`
- `TP=4, EP=2`

### Workload

For fast method exploration:

- GSM stage01
- `num_problems=112`
- `max_tokens=128`
- `max_model_len=512`
- `max_num_batched_tokens=512`
- `max_num_seqs=8`

For final comparison:

- `num_problems=256`

### Placement set

Always include:

- `contiguous`
- `round_robin`
- `fixed_random_shuffle`
- `load_balanced_greedy_with_locality_tiebreak`
- `communication_aware_greedy`

---

## 9. Decision gates

At the end of the exploration, use these gates.

### Gate A: Should we refactor token owner at all?

Refactor if **any** of the following is true:

- routing prediction oracle looks strong
- replica placement needs true source semantics
- placement conclusions are heavily profile-contaminated
- drop policies need owner-aware accounting

Do **not** refactor yet if:

- all promising gains come from simple top-k reduction alone
- placement signal is already weak after TP is enabled
- score/runtime tradeoff is too poor to justify deeper runtime work

### Gate B: Which main optimization family to pursue?

Prefer:

- **routing reuse / prediction** if next-step route stability is high
- **drop** if small quality loss buys large decode speedup
- **placement / replica** if TP-enabled geometry still shows meaningful rank-load / throughput spread
- **alternate dispatch runtime** if profile contamination remains the main blocker

---

## 10. Recommended final baseline after exploration

Unless exploration strongly contradicts it, the default baseline for future work should be:

- `world_size=8`
- `tp_size=2`
- `ep_size=4`
- `moe_impl=ep_ht`
- `enforce_eager=1`

Why:

- more realistic than `TP=1, EP=8`
- still exposes MoE placement/routing effects
- avoids the worst current-runtime pathologies
- does not force immediate token-owner refactor

The explicit token-owner runtime should be treated as a **phase-2 infrastructure upgrade**, not phase-1 baseline.

---

## 11. Suggested prompt for an engineering agent

Use the following prompt verbatim or adapt it.

```text
You are working in /home/lzy/Artifact-Infer.

Goal:
Run the pre-token-owner MoE exploration plan before redesigning the runtime.

Primary objective:
Identify which MoE optimization directions show enough signal to justify a later token-owner refactor.

Constraints:
- Do NOT implement token-owner runtime yet.
- Prefer minimal, local, low-risk changes.
- Preserve existing behavior by default; gate all new behavior behind flags.
- Keep all new profiling/analysis outputs explicit and reproducible.

Priority order:
1. Add routing-profile quality diagnostics.
2. Sweep TP×EP geometries: TP=1/EP=8, TP=2/EP=4, TP=4/EP=2.
3. Add top-k override and run K in {1,2,4,8}.
4. Add per-step decode routing trace mode and evaluate simple routing-reuse oracles.
5. Add simple drop-policy ablations.
6. Produce a unified comparison report.

Files likely to modify:
- workshop/nanovllm_moe/services/utils/routing_profile.py
- workshop/nanovllm_moe/services/utils/expert_placement.py
- workshop/nanovllm_moe/services/config.py
- workshop/nanovllm_moe/artifacts/modeling/layers/moe/fused_moe.py
- workshop/nanovllm_moe/artifacts/modeling/layers/moe/dispatch_ep_ht.py
- eval/generate_moe_placement.py
- eval/run_moe_stage01_gsm.py
- add eval/run_pre_token_owner_exploration.py
- add any small analysis helpers under eval/

Required outputs:
- a summary markdown report
- a CSV/JSON table of all experiments
- explicit labels for which conclusions are trustworthy under the current runtime and which are only directional proxies

Metrics to report:
- average_score
- e2e_total_time_s
- prefill_throughput_tok_s
- decode_throughput_tok_s
- estimated_gpu_cv
- estimated_gpu_cv_layer_mean
- estimated_cross_traffic_ratio
- estimated_cross_traffic_ratio_layer_mean
- profile row-identity / row-distance diagnostics
- drop fractions if drop is enabled
- routing reuse stability metrics if route tracing is enabled

Important interpretation rule:
If a direction depends on true source/owner semantics (replica selection, owner-aware placement, routing prediction by source locality), mark its pre-token-owner evidence as proxy-only and do not overclaim.
```

---

## 12. Final recommendation

Before touching token-owner runtime, the most valuable explorations are:

- profile-quality audit
- TP×EP geometry selection
- top-k reduction
- routing reuse oracle
- simple drop ablation

These are the cheapest ways to find out whether future work should focus on:

- prediction,
- drop,
- placement/replica,
- or a deeper dispatch/runtime redesign.

