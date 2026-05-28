# Research Brief — MoE Inference Optimization

**Date**: 2026-05-28
**Author**: lzy
**Source**: `docs/claude-moe/drop/{background.md, 0526-long-bench/FINAL_REPORT.md, SUMMARY.md}`

---

## Part 1 — 我的核心工作

- **Setting**: Qwen3-30B-A3B (48 层, E=128, K=8) on 8×RTX 4090 24GB, `ep_ht` runtime, `owner_local_ep` mode, Phase 3 LBG overlap=0.25 静态 placement。
- **Phase 3 (static placement)**：LBG vs round_robin × overlap ∈ {0, 0.25, 0.5} 已扫完；静态 placement 在 GSM8K 上收敛到 ~115s e2e，单换 placement 不再有显著加速。
- **Phase 4 v1/v2 → GPU drop → P1 (K_eff hard truncation)**：CPU drop 因 host-sync 慢 10-25%；GPU drop + `MOE_DROP_MIN_REPLICAS=128` 把 drop 算子做到零开销但 e2e 仍 ±2%；K_eff 硬截是负杠杆。**三轮都失败，但都是 ±2% e2e**，提示瓶颈定位错误。
- **Tier 1/2 prefill L-sweep**：用隔离 microbench 找到 MoE-block 的 break-even `L*`，再用全模型 prefill 实测复现，**第一次量化拐点**。
- **v1-v6 LongBench**：5 GPU drop policy × 4 rate × 2 Phase 3 plan × 3 dataset × 3 metric，~5,120 prompt-cell。最终在 `tail_weight @ r=0.3` + `MOE_DROP_MIN_REPLICAS=512` 上**首次拿到 e2e 正收益且 accuracy 零损失**。

## Part 2 — 关键 Finding（重点突出）

1. **97% drop 收益来自 a2a payload reduction**：dispatch −47% + combine −50%（Tier 1 segment ablation, T=2048 cell）。
2. **只有 2.8% 来自 expert GEMM**：直接推翻"drop 通过减少 expert 计算量加速"的 naive 假设，也解释了为什么 K_eff 硬截（专砍 expert GEMM）是负杠杆 — 杠杆错位。
3. **Break-even `L* ≈ 3271 rows/rank`**（Tier 1 −5% 阈值插值）。低于 L\* drop 因 launch/sync 开销反而变慢；高于 L\* 收益快速饱和，甜点在 `L_recv ≈ 4k`。
4. **生产配置**：`tail_weight @ r=0.3` + `MOE_DROP_MIN_REPLICAS=512` 在 LongBench long-prompt 上 prefill **+23%** / e2e **+12%** / accuracy **零损失**（v6 官方 binary metric）。r=0.5 性能极限 +46%/+20% 但带 6pp accuracy 代价。
5. **GSM8K decode-heavy 结构性无收益**：decode `L_recv ≈ 8`，远低于 L\*；GSM8K prefill 段 `L_recv ≈ 250`，仍差 13×。所有 Phase 4 历史 ±2% e2e 都由此解释。

## Part 3 — 三个候选 Research Direction

### Direction A — Communication-centric MoE Optimization

- **Hypothesis**：a2a 通信（dispatch + combine 字节数 + scatter 工作量）才是 sparse MoE 推理的真瓶颈；我的数据证明 97% drop 收益来自 a2a 通信侧。因此 expert replica / placement / token routing / load prediction 都应当**围绕 minimize a2a payload bytes & cross-node traffic** 重新设计，而不是围绕 expert workload balancing。
- **Core claim**：现有 MoE 系统的优化范式是 "compute-centric"（看 expert GEMM、看 GPU 利用率、看 load balance），但 sparse MoE inference 在中等以上 batch 上实际是 **communication-bound**。需要从 "balance expert compute" 范式转向 "minimize a2a bytes" 范式。
- **与传统工作区别**：FasterMoE / Tutel / DeepSpeed-MoE / MegaBlocks 都以 expert workload balancing 为优化目标，allreduce/a2a 被当作执行细节；本方向把 **a2a payload bytes 作为一等优化指标**，placement/drop/routing 全部围绕它重写。

### Direction B — Phase-asymmetric MoE Optimization

- **Hypothesis**：MoE prefill 和 decode 阶段对优化策略的响应**完全不同**：prefill 阶段 `L_recv` ≈ 1000s rows/rank，bandwidth-bound，drop / aggressive token reduction 有效；decode 阶段 `L_recv ≈ K·T/EP ≈ 8`，latency-bound，drop 反而引入 launch 开销。沿用统一策略（同一 placement、同一 drop policy、同一 K）必然在一边妥协。
- **Core claim**：MoE serving 应该是 **phase-aware framework** — drop policy、replica budget、placement、router K_eff、prediction 都按 phase 分别优化、运行时切换。
- **与 DistServe / SplitWise 区别**：DistServe / SplitWise 把 dense LLM 的 prefill 和 decode 分到不同 instance 上以隔离 latency 干扰；但它们完全没触及 MoE 特有的 phase 不对称性（`L_recv` 差两个数量级 → a2a 字节差两个数量级 → 优化杠杆完全不同）。本方向把 MoE 的 phase 不对称性升级为一等设计原则。

### Direction C — Workload-adaptive MoE Serving

- **Hypothesis**：最优 MoE 配置严重依赖 workload 形状 — long-prompt + short-decode (RAG / 长文档 QA / 检索) 上 drop 显著正向 (e2e +12%)，short-prompt + long-decode (GSM8K / reasoning / agent) 上 drop 零收益。生产 serving 系统的流量混合两种形态，**静态配置必然在一类 workload 上踩坑**。
- **Core claim**：提出 adaptive framework，**运行时**根据 incoming request 的 prompt length / 预期 decode length / batch shape 动态调整 drop rate / replica budget / K_eff / placement variant；甚至按 request 分流到不同的"模式"。
- **与现有工作区别**：vLLM / SGLang / TensorRT-LLM 的 MoE 实现把 expert placement、router top-K、replica 数都当作**部署时常量**；continuous batching 只调度 token 不调度 MoE 策略。本方向把 MoE 策略本身放进调度循环。

## Part 4 — 一句话 Hypothesis

- **A (Communication-centric)**：把"a2a payload bytes"提升为 MoE inference 的一等优化指标，并据此重新设计 replica / placement / drop / routing，能在 communication-bound 工况下拿到现有 compute-centric 方法摸不到的 e2e 收益。
- **B (Phase-asymmetric)**：MoE 的 prefill 和 decode 在 `L_recv` 上相差 ~100×，必须用 phase-aware 的 drop policy + replica budget + K_eff 才能同时优化两端，而不是在统一策略下二选一妥协。
- **C (Workload-adaptive)**：单一静态 MoE 配置无法同时服务 long-prompt-short-decode 和 short-prompt-long-decode 两类 workload；运行时按 request 形状自适应调整 MoE 策略（drop rate / K / replica）能在异构混合流量上拿到显著 e2e 收益。
