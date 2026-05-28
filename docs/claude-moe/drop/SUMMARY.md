# Drop 实验总结（快速上手）

> 一页纸快速了解 Phase 4 token-replica drop 的全部进展。完整细节见各子目录。
>
> **结论**：drop 是 **prefill-only 加速器**。在 long-prompt + short-decode workload 上 prefill +23%/e2e +12% 零 accuracy 损失（`tail_weight @ r=0.3`，`MOE_DROP_MIN_REPLICAS=512`）；在 GSM8K decode-heavy 上结构性无收益。
>
> **日期**：2026-05-28　**硬件**：8×RTX 4090 24GB　**模型**：Qwen3-30B-A3B (E=128, K=8)

---

## 1. 实验链路（按时间）

| 阶段 | 文档 | 核心问题 | 关键产出 |
|---|---|---|---|
| **背景** | `background.md` | 为什么需要 sweep | Phase 3/4 历史全是 ±2% e2e；诊断指向 workload 不匹配 |
| **Tier 1** microbench | `0526-plan/0526-final_report.md` §2 | 隔离 MoE-block 的拐点 L\* | **L\* ≈ 3.3k rows/rank** |
| **Tier 2** synthetic e2e | 同上 §3 | 全模型 prefill 是否复现拐点 | T=512: prefill +8.8%；T=2048: +4.5% |
| **v1-v3** LongBench 性能 sweep | `0526-long-bench/README.md` `v2_*.md` `v3_*.md` | 真实 prompt + policy/rate/bypass 矩阵 | 5 GPU policy 性能等价；bypass 是 e2e 开关 |
| **v4-v5** accuracy 重打分 | `v4_FINAL.md` `v5_cross_dataset.md` | token-F1/recall 跨数据集 | tail_weight 唯一在 r=0.5 不掉 recall |
| **v6** 官方 binary metric | `FINAL_REPORT.md`（权威） | LongBench `passage_retrieval_en` strict accuracy | r=0.3 是真正 Pareto 甜点；r=0.5 上 random/cross_numa_* accuracy 崩盘 |

**权威文档**：`0526-long-bench/FINAL_REPORT.md`（整合 v1-v6）。

---

## 2. Tier 1/2 — 拐点定位（按你的提问回答）

数据来源：`0526-plan/0526-final_report.md` Tier 1 main table + Tier 2 实测。

| 问题 | 答 |
|---|---|
| **拐点 L\* 落在哪？** | **L\* ≈ 3271 rows/rank**（介于 T=64 / L≈524 和 T=512 / L≈4160 之间，−5% 阈值线性插值）。对应全 batch ≈ 3.3k prefill tokens 跨 8 ranks。 |
| **拐点处 speedup？** | 跨过拐点的第一个采样点 T=512 (L=4160)：**MoE-block −13.3% (≈1.15×)**；Tier 2 全模型 prefill **+8.8% (1.096×)**。 |
| **拐点后是否饱和？** | **快速饱和**。T=512 → T=2048 MoE-block 收益只多 2.3pp（−13.3% → −15.6%），且 Tier 2 全模型 prefill 反而从 +8.8% 退到 +4.5%（attention O(N²) 稀释）。**甜点在 L_recv≈4k**，再往大走 e2e 收益反而下降。 |
| **Tier 2 e2e 验证跑了吗？** | **跑了**。环境阻塞（nvcc sm_89）通过 `CUDA_HOME=/usr/local/cuda-12.8` + 新 flashinfer JIT cache 修复。T=512 和 T=2048 两点都有实测，每点 3 reps，std <3%。但 Tier 2 是 prefill-only timing，不含 decode/sampling 的 e2e — 那部分由后续 v1-v6 在真实 LongBench 上补。 |
| **GSM8K 真实 L_recv？拐点离它多远？** | GSM8K (T=8) prefill 段 batch=4 下 **per-rank L_recv ≈ 250**（远低于 L\*=3271，差约 13×）。decode 段 L_recv≈8（差 400×）。这定量解释了 Phase 4 v1/v2/GPU/K_eff 四轮历史实验都在 ±2% e2e — GSM8K 整个 workload 都在 drop 负收益区。 |

**Tier 1 段归因（H2 推翻）**：drop 收益 = dispatch −47% + combine −50% = **97%**，experts GEMM 只占 2.8%。意义：drop 不是省"算 expert"，而是省"a2a payload bytes + combine scatter"。这反过来解释了 K_eff 硬截为什么是负杠杆（杠杆错位）。

---

## 3. 完整模型 / LongBench 真实数据（v1-v6）

跑在 Qwen3-30B-A3B 48 层完整模型 + flashinfer + scheduler 上，5 个 GPU policy × 4 个 rate × 2 个 Phase 3 plan。**总评估 ~5,120 prompt-cell，机时 ~8 小时**。

### 性能（policy 间 spread <2%，由 rate 决定）

| rate | prefill speedup | e2e speedup（short decode, max_new=32） | system tok/s |
|:-:|:-:|:-:|:-:|
| 0.0 baseline | 1.00× | 1.00× | ~6.0k |
| 0.1 | 1.06× | 1.03× | ~6.4k |
| **0.3** ⭐ | **1.23×** | **1.12×** | ~7.4k |
| 0.5 | 1.46× | 1.20× | ~8.8k |

### Accuracy（v6 LongBench `passage_retrieval_en` 官方 binary metric, n=64/cell）

| policy | r=0.3 acc_strict | r=0.5 acc_strict |
|---|:-:|:-:|
| **`tail_weight`** ⭐ | **1.000**（零损失）| 0.938 (LBG) / 1.000 (RR) |
| `weighted_tail` | 1.000 | 0.922 |
| `cross_numa_first` | 1.000 | **0.422** ✗ |
| `cross_numa_uniform` | 0.922 | **0.234** ✗ |
| `random` | 0.906 | **0.234** ✗ |

baseline acc_strict = 1.000。**v4/v5 的 token-F1 metric 看不出 r=0.5 上的 accuracy 崩盘**（system bias by length-mismatch），v6 binary metric 才暴露 −58~−77pp 的真实差距。跨 3 个 dataset × 3 个 metric 验证：`tail_weight` 是唯一稳健的 policy。

---

## 4. 当前的生产推荐

```yaml
moe_drop_policy:        tail_weight
moe_drop_rate:          0.3                       # 零 acc 损失甜点
moe_expert_overlap:     LBG / greedy_balance      # 或 RR/min_comm，两者 prefill <1% 差
MOE_DROP_IMPL:          auto
MOE_DROP_MIN_REPLICAS:  512                       # 必须 bypass decode；=0 时 e2e 反慢 2.5%
```

**适用 workload**：long prompt + short decode（RAG、长文档 QA、检索）；per-rank prompt ≥ ~500 tokens。
**不适用**：GSM8K decode-heavy / reasoning / agent loops（drop 在 decode 上被 bypass，e2e 收益 ≈ 0）。

---

## 5. 待办（按 ROI）

1. 把 `eval/run_owner_local_ep_phase4_drop.py:84` 默认 `--drop-min-replicas` 从 `0` 改 `512`（修 Phase 4 历史 GSM8K 负增长根因）。
2. 把 binary metric（passage_retrieval / GSM8K exact-match）加入 Phase 4 eval pipeline。
3. Receive-side drop（Tier 1 显示 combine 占 50% 收益，combine 之前砍接收行可能再压一刀）。
4. `router_keff × drop` 联合 / chunked prefill + drop（Phase 5）。

---

## 6. 文件索引

```
docs/claude-moe/drop/
├── background.md                    研究脉络与 Phase 3/4 历史
├── SUMMARY.md                       ← 本文件
├── 0526-plan/
│   ├── 0526-prefill_l_sweep_plan.md      Tier 1/2 设计
│   ├── 0526-tasks.md                     任务分解
│   ├── 0526-audit_and_long_e2e.md        audit 与 long e2e 计划
│   └── 0526-final_report.md              Tier 1/2 最终报告（L* 定位 + segment 归因）
├── 0526-superpowers/
│   └── 0526-prefill_l_sweep_socratic.md  苏格拉底追问
└── 0526-long-bench/
    ├── README.md                          v1 sweep
    ├── v2_all_policies_with_score.md      v2 (7 policy + token F1)
    ├── v3_two_baselines_with_new_policies.md  v3 (2 plans + 5 policy)
    ├── v4_FINAL.md                        v4 (token recall)
    ├── v5_cross_dataset.md                v5 (multifieldqa_en)
    └── FINAL_REPORT.md                    ⭐ 权威整合 (含 v6 官方 binary)
```
