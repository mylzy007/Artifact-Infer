# Feasibility + Self-Consistency Check — All "可做" / "需调整" Ideas

- Date: 2026-05-29
- Scope: 25 idea cards rated "可做" or "需调整" in `03_collision_summary.md`.
- Method:
  - **Check A (self-consistency)**: each idea cross-referenced against the user's existing conclusions in `docs/research/2026-05-28_communication-centric/00_brief.md` — Phase 3 (LBG / round_robin × overlap ∈ {0, 0.25, 0.5} → static placement converged ~115s e2e on GSM8K; "单换 placement 不再有显著加速"), Phase 4 (v1/v2 → GPU drop → P1 K_eff hard truncation: 三轮都失败 ±2% e2e), v6 best (tail_weight @ r=0.3 + MOE_DROP_MIN_REPLICAS=512), and the decode-structural-zero-benefit result (decode L_recv≈8 far below L*).
  - **Check B (pilot feasibility)**: read-only inspection of `workshop/nanovllm_moe/` (the codebase that hosts ep_ht / owner_local_ep / drop / placement / routing infrastructure). Identify files / modules touched, leverage existing API, estimate actual implementation cost.

## Codebase map (one pass)

- `workshop/nanovllm_moe/artifacts/modeling/layers/moe/`
  - `dispatch.py`, `dispatch_ep_ht.py`, `dispatch_ep_ll.py` — already parameterized by `drop_policy`, `drop_rate`, `drop_seed`; `total_recv` (= `L_recv`) is already a runtime variable.
  - `combine.py`, `combine_ep_ht.py`, `combine_ep_ll.py`.
  - `experts.py`, `experts_ep_ht.py`, `experts_ep_ll.py`, `fused_moe.py`.
- `workshop/nanovllm_moe/artifacts/modeling/models/qwen3_moe.py`.
- `workshop/nanovllm_moe/artifacts/moe_backend/` — Triton / DeepGEMM / torch backends.
- `workshop/nanovllm_moe/services/utils/`
  - `expert_drop.py` — drop policies (`apply_drop`, `ALL_DROP_POLICIES`).
  - `expert_placement.py`, `expert_replica_placement.py` — Phase 3 placement code lives here.
  - `expert_overlap.py`, `overlap_runtime_stats.py`.
  - `routing_profile.py`, `routing_profile_quality.py` — router histogram / quality utilities (existing!).
- `workshop/nanovllm_moe/services/engine/` — scheduler.
- `src/services/nanovllm_chunked/` — chunked-prefill service (separate from MoE workshop).

Most ideas can reuse existing scaffolding; the big-cost ones cross service boundaries.

## Master table

| idea_id | 自我一致性 | 与最相关旧结论的本质区别 | 改动量 | 改动量理由 (files / modules / traps) | 可行性结论 |
|---------|-----------|------------------------|--------|--------------------------------------|------------|
| D1-I1 | 独立创新 | Phase 3 placement 是 *部署期* 决策，本 idea 是 *运行时 per-token routing* 上加 `λ·byte_cost`；不与 Phase 3 重叠 | 中 (1–2d) | `dispatch_ep_ht.py` router scoring + `routing_profile.py` per-peer pending bytes counter；routing 路径要小心改动 | 可行 |
| D1-I3 | 独立创新 | 与 Phase 4 drop 正交：drop 减 token，FP8 减 per-token bits；不与现有 lever 重叠 | **大 (>3d)** | `dispatch_ep_ht.py` + `combine_ep_ht.py` + `moe_backend.py`；FP8 encode/decode 内核虽然 DeepEP 已支持但需要集成，4090 Ada 的 E4M3 tensor-core 路径要确认 | 可行但实施量大；建议借用 DeepEP FP8 路径而非自己写 |
| D1-I4 | 独立创新 | 与 Phase 4 区别：Phase 4 把 drop 当部署常量；D1-I4 在 D1-I10 metric 上做运行时 byte budget controller | 中 (1–2d) | 新建 `services/utils/byte_budget_controller.py` + `services/engine/scheduler.py` 集成 hook + `expert_drop.py` action set 扩展；离散查找表，不需要 RL/MPC | 可行 |
| D1-I5 | **危险标** — 是 Phase 3 placement 类 idea | Phase 3 已结论 "单换 placement 不再有显著加速"；D1-I5 区别：Phase 3 用 *sum-bytes / balance* 目标，D1-I5 用 *min-max bottleneck-bytes* 目标。**若两个目标在本硬件上选出 Jaccard≤0.1 的相同 placement，则 Phase 3 结果会复现，本 idea 无新内容**。Jaccard ≥0.2 falsification 必须先做。 | 中 (1–2d) | `expert_replica_placement.py` 增加 min-max ILP；先做离线模拟（v6 logs），不动 runtime；Phase 3 已有 placement infra | 可行 (但 Phase 3 复现风险显著)；**先跑 Jaccard 测试** |
| D1-I6 | 独立创新 | 与 Phase 4 区别：Phase 4 用 *symmetric drop with router score*；D1-I6 用 *combine-only drop with post-expert ‖g·o‖*。决策单位 + 时机 + criterion 三个都不同 | 中 (1–2d) | `expert_drop.py` 加 combine-only policy + `combine_ep_ht.py` 在 combine 前过滤 + `experts_ep_ht.py` 输出 ‖g·o‖。drop infra 存在；新增 combine-side hook | 可行 |
| D1-I10 | 独立创新 (建立 cost model) | Brief 里 "97% / 2.8%" 是叙述；D1-I10 是 *measured cost model* + R² 比较，把 narrative 升级为 artifact | **小 (<半天)** | 纯离线 — 在 `eval_results/owner_local_ep_phase4_drop_longbench_v6` 上重算 worst-peer bytes / L_recv / FLOPs 三个回归；扩展 `routing_profile.py` 收 per-peer 而非 per-rank | 可行 (零运行时改动) |
| D2-I4 | **危险标** — Phase 3 placement 类 | 与 Phase 3 区别：D2-I4 是 *两套 plan + atomic swap*，Phase 3 是单 plan sweep。两 plan ILP 解的 Jaccard ≥0.2 是 falsification。**若解 Jaccard ≤0.1 → Phase 3 结论复现**。 | 中-大 (2–3d) | `expert_placement.py` 增加 decode-hits 目标 ILP + `services/engine/scheduler.py` phase-tag swap + routing 表 indirection。Routing 表 indirection 不简单 (qwen3_moe.py 内 routing dict 是 per-layer 常量) | 可行 (Phase 3 复现风险 + 实现量中) |
| D2-I5 | 独立创新 | 与 Phase 4 区别：Phase 4 是单 drop policy 全局；D2-I5 是 *per-layer per-step L_recv-gated* 切换；L_recv 已是运行时量 | 中 (1–2d) | `dispatch_ep_ht.py` 在每层取 `total_recv` (已有) → 查 per-layer `L*_ℓ` 表 → 选 drop branch；`services/utils/context.py` 加 calibration table；offline 校准用 v6 + GSM8K traces | 可行 (top D2 candidate) |
| D2-I8 | 独立创新 (与 decode 旧结论 *并不矛盾*) | Brief 里 "decode L_recv≈8 结构性无收益" 是 *byte-reduction 操作器* 无收益；D2-I8 不是 byte reducer，是 *消除 a2a 本身* (集中到 host rank)。机制类别不同 | **大 (>3d)** | `expert_replica_placement.py` 加迁移 + `dispatch_ep_ht.py` per-session routing override + session state 跟踪 (`services/engine/scheduler.py`)。多组件改动；session-level routing 表是新概念 | 可行但实施量大 |
| D3-I1 | **是 Phase 4 旧结论的变体** | Phase 4 把 drop 当部署常量；D3-I1 *预测 workload → 选 policy bundle* 本质是 per-class Phase-4 sweep。Predictor 是新的，*policy actions 是旧的*。需要证明 per-class binning 比 single global 多拿多少 e2e | 中 (1–2d) | 新建 `services/utils/request_classifier.py` (sklearn 类分类器) + `services/engine/scheduler.py` 入口分流；offline 训练在 v6 logs | 可行但 Phase 4 变体风险 |
| D3-I2 | 独立创新 | 与 Phase 4 区别：Phase 4 静态 policy；D3-I2 在 request 内 *中途切换*；L_recv 已是运行时量；confidence gating 是新的 | 中 (1–2d) | `dispatch_ep_ht.py` 收前 k 层 L_recv samples → `services/utils/context.py` 中 confidence gating → `expert_drop.py` 切换 rate；scheduler 无需改动 | 可行 |
| D3-I3 | 独立创新 | 与 Phase 4 区别：D3-I3 在 *continuous batching microbatch* 级别选 policy；Phase 4 是 deploy-time | 中-大 (2–3d) | `services/engine/scheduler.py` per-step policy + `expert_drop.py` per-step branch + `dispatch_ep_ht.py` accept per-step policy。**陷阱**：当前 scheduler 把 policy 当 batch-level 常量，要支持 per-step 需要 plumbing；output invariance 承诺要严格审 | 可行 (但要小心 scheduler API 改动) |
| D3-I8 | 独立创新 | 与 Phase 4 区别：Phase 4 不动 chunk size；D3-I8 把 chunk size 由 stall-free 目标改为 MoE L* 目标 | **大 (>3d)** | `src/services/nanovllm_chunked/` 是独立 service，不与 `workshop/nanovllm_moe/` 共享。**陷阱：cross-service integration** — 把 chunked-prefill 与 MoE EP 跑在同一 runtime 上需要重新 wire `nanovllm_chunked` 调度器到 ep_ht backend；预计 ≥3 天 | 可行但实施量大 |
| D4-I2 | 独立创新 | Phase 4 用 post-hoc L_recv；D4-I2 *在 dispatch 前预测* L_recv → 避开 critical-path collective | 中 (1–2d) | `routing_profile.py` 已有 router histogram 缓存接口 → 扩展为 layer 级；新建 `services/utils/l_recv_estimator.py`；`dispatch_ep_ht.py` 加 pre-dispatch hook | 可行 |
| D4-I3 | 独立创新 | Phase 4 是二值 on/off；D4-I3 是 4-regime；分类器只是 lookup，不动 drop op | 小-中 (~1d) | 新建 `services/utils/regime_classifier.py` (small decision tree); 依赖 D4-I1 calibration | 可行 |
| D4-I4 | 独立创新 | Phase 4 没有 boundary safety；D4-I4 加 asymmetric-loss 二值 guard。**falsifiable**: boundary vs fixed margin 差 ≥0.10·L*_ℓ | 小-中 (~1d) | 新建 `services/utils/safety_guard.py`；纯封装现有 drop；离线 asymmetric-loss 训练 | 可行 (cost-model-only 风险最低，best D4 candidate) |
| D4-I7 | 独立创新 | Phase 4 单次校准 L*；D4-I7 BOCPD on joint (L_recv, latency) slope；需要 canary microprobes 才能与 workload-mix 漂移区分 | 中 (1–2d) | 新建 `services/utils/lstar_recalibration.py` (BOCPD lib `bayesian-changepoint-detection`) + `services/utils/overlap_runtime_stats.py` 中 (L_recv, latency) 滚动 buffer + canary microprobe schedule | 可行 |
| D4-I8 | 独立创新 | Phase 4 用硬阈值；D4-I8 用 conformal 下界保证 drop-decision precision | 中 (1–2d) | 新建 `services/utils/conformal_decision.py` + `expert_drop.py` 集成；conformal calibration on v6 traces | 可行 |
| D5-I1 | 独立创新 (方法论) | Phase 4 segment ablation 是 *correlational*；D5-I1 用 padding 固定 L_recv / GEMM / routing 同时单独变 bytes → *causal* | **小 (<半天)** | 新建 test `workshop/nanovllm_moe/_test_causal_byte_microbench.py`；复用 `_test_dispatch_ep_ht.py` 框架 | 可行 (lowest effort) |
| D5-I3 | 独立创新 (理论贡献) | Brief 给的是经验 ≥10× gap；D5-I3 给的是 rate-distortion 下界 derivation；reciever-side L2-ε 重建 bits | **小 (<半天)** | 纯 analytical Python script + v6 byte 统计；无生产代码改动 | 可行 |
| D5-I5 | **危险标** — 可能已在数据中 | Brief 显示 GPU drop + MOE_DROP_MIN_REPLICAS=128 zero-overhead，v6 best 用 =512。**若 128 与 512 e2e 差 ≤2% 上方 L\* → D5-I5 主张已在数据中**；只是没人显式拎出来 | 小 (<半天) | 纯离线分析 `eval_results/owner_local_ep_phase4_drop_longbench_v6/` MOE_DROP_MIN_REPLICAS sweep | 可行；**但贡献可能只是 explicit framing** of existing data，需 honest scope |
| D5-I7 | **是 Phase 4 P1 K_eff 旧结论的解释** | Phase 4 P1 已证 K_eff 硬截负杠杆；D5-I7 是 *为何* 的分析模型。不矛盾，是 retrospective explanation。需说服读者 explanation 本身有价值 | 小 (<半天) | 纯分析 + 已有数据 | 可行 (low-cost framing) |
| D5-I8 | 独立创新 | Phase 4 比较 drop 变体经验；D5-I8 partition 为 byte-equivalence 类，分离 latency 轴 vs accuracy 轴 | 小-中 (~1d) | offline 分析 + residual-term latency 模型推导 | 可行 |
| D5-I9 | **是 decode 旧结论的形式化** | Brief 已写 "decode L_recv≈8 结构性无收益"；D5-I9 *把它形式化为条件定理 + speculative decoding regime-flip 预测*。**前半（理论 statement）只是把 brief 形式化；后半（regime flip prediction）才是新内容**。如果 SD/MTP regime flip prediction 不成 → 实质重复 brief | 小 (<半天) | 纯分析；speculative regime flip 验证可用现有 decode trace + 简单 SD 估算 | 可行；但 *standalone 贡献小*；SD/MTP 部分受 MoESD / Utility-Driven SD 撞车 (见 collision) |
| D5-I10 | 独立创新 (但 API 大改) | 当前 dispatch/combine API 不暴露 byte budget；D5-I10 加 budget API。GPT collision verdict 要求 reframe 为 "budgeted semantic collective" layer 而非新 collective | **大 (>3d)** | `dispatch.py` + `combine.py` 增加 `BudgetedAllToAllV` API + `moe_backend.py` + `fused_moe.py` + 所有 caller 改动；与 NCCL EP / DeepEP 兼容性要 handle | **可行但实施量最大**；**实际改动量易被严重低估**：调用方多处 + backward compat |

## 高危标记 (与旧结论可能重复 / 矛盾 / 改动量低估)

### 撞自己旧结论 (Phase 3 placement / Phase 4 K_eff / decode 结论)
- **D1-I5** (placement) — Phase 3 复现风险：若 min-max 与 sum-bytes 在本硬件上选出 Jaccard ≤0.1 的同一布局，Phase 3 "placement 已收敛" 结果会复现。**必须先做 Jaccard ≥0.2 离线测试**作为 idea 存亡判据。
- **D2-I4** (bi-modal placement) — 同 Phase 3 复现风险：两 plan ILP 解需 Jaccard ≥0.2 才能不重叠到 Phase 3 单 plan 结论。
- **D5-I5** (replica-minimality above L*) — Phase 4 sweep 中已经跑过 MOE_DROP_MIN_REPLICAS=128 vs 512，若两者 e2e 差 ≤2% (above L\*)，D5-I5 主张 *已经在数据中*，只是没显式拎出。**预测可能成为 "我们重新解释 brief 中已有的数字"**。
- **D5-I7** (K_eff failure model) — Phase 4 P1 K_eff 硬截负杠杆是 *已知结论*；D5-I7 是 *retrospective explanation*。不矛盾但属于 "解释" 类贡献，价值取决于是否能预测 K_eff *何时* 不负杠杆 (e.g., 未来硬件)。
- **D5-I9** (decode-hopelessness theorem) — Brief 已经写了 "decode L_recv≈8 结构性无收益"；D5-I9 把它形式化。SD/MTP regime-flip 才是新增 *预测* 内容；如果它不成立，剩下的是 brief 重述。
- **D3-I1** (request-class router) — 实质是 *per-class Phase 4 sweep*；predictor 是新的但 *policy actions 是旧的*。

### 改动量严重低估 (听起来简单实际要大改)
- **D1-I3** (FP8 a2a) —— Listed "中" 但实质 **大**：要稳定打通 4090 Ada FP8 tensor-core 路径 + 集成 DeepEP FP8 + stochastic round + 在 dispatch+combine 同时换 dtype，不只是 "加个 quantize/dequant call"。
- **D2-I8** (decode expert re-co-location) —— 看起来 "迁移 32 个 expert"，实质需要 session state + per-session routing override + a2a bypass 路径，多文件协调。
- **D3-I8** (adaptive chunk size for MoE L\*) —— **`nanovllm_chunked` 和 `nanovllm_moe` 是两个独立 service**；要让 chunked-prefill 与 MoE EP 跑在同一 runtime 是 cross-service integration，cost ≥3 天保守估计。
- **D5-I10** (BudgetedDispatchCombine primitive) —— 新增 API + 所有 caller 修改 + NCCL EP/DeepEP 兼容；caller 多处分散，实际改动量易被严重低估。

### 独立创新且改动量合理 (短期可 pilot)
- **D1-I10** (perf model)、**D5-I1** (causal microbench)、**D5-I3** (info-theoretic bound)、**D5-I8** (byte-equivalence classes) — 全部 ≤1 天纯离线分析。
- **D4-I4** (don't-drop guard)、**D4-I3** (regime classifier)、**D2-I5** (L_recv-gated micro-switch) — ≤1.5 天，复用现有 drop infra + L_recv 已是运行时量。
- **D1-I4** (byte-budget controller)、**D1-I6** (combine drop)、**D3-I2** (early-layer feedback)、**D4-I2** (L_recv estimator)、**D4-I7** (BOCPD recalibration)、**D4-I8** (conformal decision) — 1-2 天，主要是 services/utils/ 下新 file + dispatch/combine hook。

## 推荐 pilot 排序 (综合 collision + 自我一致性 + 改动量)

**Tier 1 (低风险高收益, ≤1 天 + 独立创新)**:
1. **D1-I10** — perf model (offline only, validates whole D1 theory)
2. **D5-I1** — causal microbench (methodology)
3. **D5-I3** — info-theoretic lower bound (theory)
4. **D5-I8** — byte-equivalence classes (analytical)

**Tier 2 (中风险中收益, 1-2 天, 独立创新)**:
5. **D2-I5** — L_recv-gated per-layer micro-switch (top D2)
6. **D4-I4** — don't-drop safety guard (top D4 by cost-model-only risk)
7. **D3-I2** — online L_recv early-layer feedback (top D3)
8. **D4-I2** — L_recv estimator (D4 substrate)
9. **D1-I4** — byte-budget controller
10. **D1-I6** — combine-asymmetric drop

**Tier 3 (高 placement-复现风险, 必须先做 Jaccard ≥0.2 测试)**:
11. **D1-I5** — min-max bottleneck-cut placement
12. **D2-I4** — bi-modal placement

**Tier 4 (大改动 / cross-service / 须 reframe)**:
13. **D1-I3** — FP8 a2a payload (借 DeepEP)
14. **D2-I8** — decode expert re-co-location
15. **D3-I8** — adaptive chunk size (cross-service)
16. **D5-I10** — BudgetedDispatchCombine primitive (大改)

**Tier 5 (低优先级 / 多半已在数据中或 brief 中)**:
17. **D5-I5** — replica-minimality above L* (可能已在数据中)
18. **D5-I7** — K_eff failure explanation (retrospective)
19. **D5-I9** — decode-hopelessness theorem (formalization of brief)
20. **D3-I1** — request-class router (per-class Phase 4 variant)

Remaining (medium priority, single-knob 风险): D4-I3, D4-I7, D4-I8.
