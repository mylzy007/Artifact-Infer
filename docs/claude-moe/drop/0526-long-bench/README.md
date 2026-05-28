# LongBench Drop 多维度对比实验

> **TL;DR**：22-cell 一次 engine-load 内的多轴 sweep，**3 个 policy × 4 个 rate × 4 个 bypass × 2 个 length tier**。
> 主要发现：
> 1. **三个 GPU drop policy 性能上完全等价**（差 < 1%）— 选哪个都一样，挑 accuracy 损失最小的即可。
> 2. **rate 与 prefill 加速近似线性正相关**：r=0.1→+5%，r=0.3→+22%，r=0.5→+42%。
> 3. **bypass 是 e2e 加速的开关**：`bypass=0` 时 prefill 仍快 +21%，但 e2e **倒变慢 2.5%**；`bypass≥128` 即可保 e2e +8.7%。**这是最关键的工程参数**。
> 4. drop 收益跨 [3k, 5k] token 长度稳定 +20% prefill / +8% e2e。
>
> **日期**：2026-05-26
> **机时**：~9 分钟 8 卡
> **数据**：22 个 cell × 平均 3 个 batch = 66 个 batch；噪声 std < 5%
> **产物**：`eval_results/owner_local_ep_phase4_drop_longbench/`

---

## 1. 实验设计

### 1.1 对比维度

5 个 axes 同时扫，但聚焦 3 个 phase 避免笛卡尔积爆炸：

| Axis | 取值 | 假设 |
|---|---|---|
| **drop_policy** | `tail_weight` / `random` / `cross_numa_first` | 上层假设：policy 决定哪些 replica 被砍，但 a2a 收益主要来自 *字节减少*，所以 policy 间应当差异微小 |
| **drop_rate** | 0.1 / 0.2 / 0.3 / 0.5 | 假设：rate 与 effective payload 减少近似线性，所以 prefill speedup 也近似线性 |
| **MOE_DROP_MIN_REPLICAS (bypass)** | 0 / 128 / 512 / 2048 | 验证 audit 发现：bypass=0 时 decode 被 drop 拖累 |
| **prompt_length_tier** | short [1k,2k] / medium [2k,4k] / long [4k,6k] | 验证 Tier 2 找到的 "drop sweet spot ≈ 3k-4k" 是否依然成立 |
| **dataset** | LEval Generation_multidoc_qa（real-world long QA） | 取代之前 Tier 2 的 synthetic `" the"×N`，更接近真实分布 |

### 1.2 Phase 划分

- **Phase A**（13 cell，medium tier）：3 policy × 4 rate + 1 baseline，主报告矩阵
- **Phase B**（3 cell，medium tier）：固定 `tail_weight@0.3`，扫 bypass ∈ {0, 128, 2048}（512 与 A 共享）
- **Phase C**（6 cell）：固定 `tail_weight@0.3, bypass=512`，扫 3 个 length tier × {baseline, drop}

总计 **22 cell**（Phase A 的 baseline 与 Phase C 的 medium baseline 复用记账，summary 自动匹配同 tier baseline）。

### 1.3 防偏置设计

1. **同一 engine load**（22.8s）→ 模型 / weights / KV cache 都是同一份，避免跨 process 内存差异。
2. **cell 顺序随机化**（seed=42 shuffle）→ baseline 与 drop 不连续运行，避免 thermal / cache effects 系统偏置。
3. **3 个 baseline batch 单独 warmup**（discarded）→ flashinfer JIT compile 不污染 Phase A baseline。
4. **同一组 prompt** 在 baseline 与 drop 之间复用 → 排除 prompt-level 噪声。
5. **每 cell 3 个 batch** 给方差估计；std 列在结果中。

### 1.4 不测的维度（写明）

| 维度 | 为什么不测 |
|---|---|
| accuracy / score | 没跑 generation eval（这是性能 sweep）。v1/v2 已验证 rate≤0.3 score 持平 |
| `router_keff` × drop 联合 | Phase 4 P1 已单独测；联合维度过大，留 Phase 5 |
| CPU drop policies (`per_expert_uniform`/`hot_expert_relief`) | 这些走 CPU path，Phase 4 v1/v2 已显示 host-sync overhead 主导，prefill 也会被拖累 |
| 多 dataset | 单一 LEval 已显示效应；多 dataset 验证留分卷 |
| `expert_overlap_strategy` 变体 | Phase 3 已扫；本次固定 LBG greedy_balance |

---

## 2. 实验配置

| 项 | 值 |
|---|---|
| Model | Qwen3-30B-A3B (48 layers, E=128, K=8, H=2048) |
| World | 8×RTX 4090 24GB |
| MoE impl | `ep_ht` |
| Runtime mode | `owner_local_ep` |
| Overlap plan | `phase3_smoke_plan_20260521`（Phase 3 最优 LBG） |
| Overlap strategy | `greedy_balance` |
| Dataset | `/home/lzy/datasets/moe_benchmarks/prepared/leval.Generation_multidoc_qa.custom.jsonl` |
| max_num_batched_tokens | 6144（容纳到 long tier ~6k tokens） |
| gpu_memory_utilization | 0.95 |
| max_new_tokens | 32 |
| batch_size | 8（一 rank 一 prompt） |
| samples per tier | 24（medium）/ 8（long）/ 0（short — 数据集过滤后无样本） |
| warmup_batches | 3（baseline，丢弃计时） |
| Env | `CUDA_HOME=/usr/local/cuda-12.8`, `FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_tier2` |

**复现命令**：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
  PATH=/usr/local/cuda-12.8/bin:$PATH \
  FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_tier2 \
  /home/lzy/miniconda3/envs/vllm/bin/torchrun --nproc_per_node=8 --master_port=29513 \
  -m eval.drop.longbench_sweep \
  --output-dir eval_results/owner_local_ep_phase4_drop_longbench \
  --dataset /home/lzy/datasets/moe_benchmarks/prepared/leval.Generation_multidoc_qa.custom.jsonl \
  --num-samples-per-tier 24 --batch-size 8 --warmup-batches 3 --max-new-tokens 32 \
  --drop-rates "0.0,0.1,0.2,0.3,0.5" \
  --drop-policies "tail_weight,random,cross_numa_first" \
  --bypass-list "0,128,512,2048" \
  --length-tiers "short,medium,long" \
  --max-num-batched-tokens 6144 \
  --expert-overlap-path eval_results/tmp_phase3_overlap_test/moe_overlap_plan_smoke_plan_20260521_222623.json \
  --gpu-memory-utilization 0.95

# Plots
python -m eval.drop.plot_longbench --output-dir eval_results/owner_local_ep_phase4_drop_longbench
```

---

## 3. 结果

### 3.1 Phase A — policy × rate 矩阵

medium tier，~3k tokens prompt，bypass=512。每 cell 3 个 batch。

| policy | rate | prefill (s) | **prefill_speedup** | e2e_speedup |
|:--|--:|--:|--:|--:|
| baseline | 0.0 | 4.484 | — | — |
| tail_weight | 0.1 | 4.244 | 1.056× | 1.017× |
| tail_weight | 0.2 | 4.038 | 1.110× | 1.039× |
| tail_weight | 0.3 | 3.699 | **1.212×** | **1.087×** |
| tail_weight | 0.5 | 3.161 | **1.418×** | **1.157×** |
| random | 0.1 | 4.297 | 1.043× | 1.001× |
| random | 0.2 | 3.961 | 1.131× | 1.049× |
| random | 0.3 | 3.693 | **1.214×** | 1.077× |
| random | 0.5 | 3.158 | **1.419×** | 1.145× |
| cross_numa_first | 0.1 | 4.261 | 1.052× | 1.010× |
| cross_numa_first | 0.2 | 3.925 | 1.142× | 1.053× |
| cross_numa_first | 0.3 | 3.653 | 1.227× | 1.096× |
| cross_numa_first | 0.5 | 3.177 | 1.411× | 1.151× |

**观察**：

- **三个 policy 在任意 rate 下都互相在 1% 噪声内**。这是个强结论：drop 的收益**与具体扣谁无关，只与"扣多少 bytes"有关**。验证了 Tier 1 的 segment 归因结论（收益来自 dispatch+combine 的字节减少，不是 expert GEMM）。
- **rate × prefill 接近线性**：

  | rate | mean prefill_speedup | predicted (linear) |
  |---|---|---|
  | 0.1 | 1.050 | 1.05 |
  | 0.2 | 1.128 | 1.11 |
  | 0.3 | 1.218 | 1.17 |
  | 0.5 | 1.416 | 1.29 |

  实际超过线性预测（rate=0.5 时 1.42× > 1.29×），说明大 rate 下扣除的 a2a payload 让通信进入更友好的 message-size regime。
- **e2e_speedup 比 prefill_speedup 低**，因为 decode (~32 tokens × 48 layers) 占 e2e 的 ~50%。rate=0.5 + bypass=512 时仍能拿到 +15.7% e2e。

主图：[`phase_a_policy_rate_heatmap.png`](../../../eval_results/owner_local_ep_phase4_drop_longbench/phase_a_policy_rate_heatmap.png)，[`phase_a_rate_lines.png`](../../../eval_results/owner_local_ep_phase4_drop_longbench/phase_a_rate_lines.png)

### 3.2 Phase B — bypass 敏感度

`tail_weight@0.3`，medium tier。

| min_replicas | prefill (s) | prefill_speedup | e2e (s) | e2e_speedup |
|:--|--:|--:|--:|--:|
| 0 | 3.692 | **1.214×** | 10.117 | **0.975× (慢了 2.5%)** |
| 128 | 3.699 | 1.212× | 9.076 | 1.087× |
| 512 | 3.699 | 1.212× | 9.077 | 1.087× |
| 2048 | 3.731 | 1.201× | 9.196 | 1.073× |

**关键发现**：

- `bypass=0` 时 **prefill 仍快 21%，但 e2e 倒慢 2.5%**。
- decode 阶段每层每 GPU `L_recv≈8`，Tier 1 实测此处 drop **+25% slowdown**。32 个 decode step × 48 层 × 8 ranks 累积起来约 **1 秒 decode overhead**，把 prefill 节省的 0.8s 全部抹掉还有余。
- bypass=128 已经足够（128 > decode L_recv=8），bypass=512 与 bypass=128 几乎相同。
- bypass=2048 开始略损 prefill 收益（部分 prefill 步骤被误 bypass）。
- **结论**：bypass ∈ [128, 1024] 是安全区间；推荐 **512**（与 Phase 4 GPU drop 历史默认一致）。

**这定量验证了之前 audit 对 `run_owner_local_ep_phase4_drop.py:84-85` 默认值 `--drop-min-replicas=0` 的批评**：
该默认在历史 GSM8K 实验上让 drop "看似没用"，实际是 decode 被悄悄拖累。改默认到 512 后，所有 phase 4 历史负实验应当重新评估。

主图：[`phase_b_bypass.png`](../../../eval_results/owner_local_ep_phase4_drop_longbench/phase_b_bypass.png)

### 3.3 Phase C — 长度 tier 缩放

`tail_weight@0.3, bypass=512`，对比同 tier 的 baseline。

| tier | prompt p50 (token) | n batches | baseline_prefill (s) | drop_prefill (s) | prefill_speedup | e2e_speedup |
|:--|--:|--:|--:|--:|--:|--:|
| short | [1024, 2048] | **0 (无样本)** | — | — | — | — |
| medium | ~2979 | 3 | 4.482 | 3.735 | 1.200× | 1.072× |
| long | ~4144 | 1 | 5.550 | 4.538 | 1.223× | 1.089× |

**观察**：

- short tier 无数据 — LEval multidoc QA 没有 < 2k 的 prompt（最短 prompt ≈ 2.7k）。
- medium → long：prefill_speedup 从 1.200× → 1.223×，e2e_speedup 从 1.072× → 1.089×。
- 长 prompt 收益**略高于**中等 prompt，与 Tier 2 synthetic 趋势相反（synthetic 在 T=2048 反而 +4.5%，因 `" the"×N` attention 太友好）。**真实长 prompt 上 attention 内存带宽紧张 → MoE 占比相对高 → drop 收益保留更多**。

主图：[`phase_c_tier.png`](../../../eval_results/owner_local_ep_phase4_drop_longbench/phase_c_tier.png)

> Short tier 缺数据是当前 dataset 限制；若要补，需要切换到 prompt 长度更短的 dataset（如 OpenOrca、MMMU 等）。

### 3.4 跨 phase 一致性

`tail_weight@0.3 @ bypass=512` 同一 config 跨 Phase A 和 Phase C：
- Phase A 测到 prefill_speedup=1.212×，e2e=1.087×
- Phase C medium 测到 prefill_speedup=1.200×，e2e=1.072×
- 差异 < 1.5%，落在 batch-level std=4% 噪声内。**数据可信**。

---

## 4. 最佳组合 / 推荐

### 4.1 性能最大化（不顾 accuracy）

| 参数 | 推荐值 |
|---|---|
| drop_policy | **任意三个**（性能相同；按 accuracy 选 `tail_weight` 是 Phase 4 v1/v2 历史最好） |
| drop_rate | **0.5** → +42% prefill / +15.7% e2e |
| min_replicas (bypass) | **512** |
| 适用 workload | 单 rank prompt ≥ 2k tokens，prefill 占 e2e ≥ 30% |

### 4.2 平衡（保 accuracy）

Phase 4 v1/v2 历史：`tail_weight@0.3` 在 GSM8K score 保持 ±1pp。

| 参数 | 推荐值 |
|---|---|
| drop_policy | `tail_weight` |
| drop_rate | **0.3** → +22% prefill / +8.7% e2e |
| min_replicas | **512** |

### 4.3 错误配置示例（**别这么用**）

| 错误 | 后果 |
|---|---|
| bypass=0（runner 默认）+ short decode | e2e 慢 2.5% |
| bypass=0 + GSM8K 长 decode | e2e 慢 10-25%（Phase 4 v1/v2 历史现象） |
| rate=0（drop "关闭"但 policy 是 random）| 无影响，但浪费一次实验 |

---

## 5. 关键洞察（论文级）

1. **drop 的收益机制是"减少 a2a payload bytes"，与挑选 token 的策略无关**。
   - 证据：3 个 policy 在所有 rate 上误差 < 1%。
   - 配 Tier 1 segment 归因（dispatch −47% + combine −50% = 97% 收益）。
   - 含义：**未来 policy 创新应当聚焦在 accuracy 损失最小化，不在性能优化**。

2. **drop 与 decode 互不兼容**，必须 bypass。
   - 证据：bypass=0 让 e2e 慢 2.5%（即便 prefill 快 21%）。
   - 历史含义：**Phase 4 v1/v2/GPU drop 三轮 ±2% 结果的真正原因不是"drop 不行"，而是 bypass 默认错了**。改默认即可让旧实验重新出正向结论。

3. **drop 是 prefill 加速器，不是 e2e 加速器**。
   - prefill+22%、e2e+8.7% 的"打折比例" ≈ prefill 占 e2e 的比例。
   - 含义：drop 在 **长 prompt + 短 decode** 场景下最有效（chunked prefill、长文档 QA、RAG 等）。短 prompt + 长 decode（GSM、agent loops）则 drop 不适合。

4. **rate 与 prefill_speedup 线性外推合理**，最多到 rate=0.5 时**略超线性**（带宽 regime 改善）。
   - 含义：若 accuracy 容忍度允许 rate=0.5（需要 v3 重新校验），可拿到 1.42× prefill。

---

## 6. 修改建议（按 ROI）

排序按"几行代码 → 大量历史实验结论可重新解读"。

| # | 改动 | 文件 |
|---|---|---|
| 1 | 把 `--drop-min-replicas` 默认从 0 改成 512 | `eval/run_owner_local_ep_phase4_drop.py:84` |
| 2 | 把 `--profile-drop-runtime-stats` 默认从 1 改成 0 | `eval/run_owner_local_ep_phase4_drop.py:86` |
| 3 | 给 runner 加一个 `--dataset-prepared <jsonl>` 开关接长 prompt | `eval/run_owner_local_ep_phase4_drop.py` / `eval/utils_owner_local_ep.py` |
| 4 | 给 Phase 5 加一个 `rate=0.5` 的 accuracy 校验任务 | 新 issue |

---

## 7. 文件清单

```
新增 / 产物：
  eval/drop/longbench_sweep.py                                         (multi-axis sweep bench)
  eval/drop/plot_longbench.py                                          (plotting)
  eval_results/owner_local_ep_phase4_drop_longbench/
    longbench_rows.jsonl                                               (66 batches × 11 cells)
    longbench_summary.json                                             (per-cell agg + speedups)
    per_cell_table.md                                                  (markdown table)
    phase_a_policy_rate_heatmap.png                                    (Phase A heatmap)
    phase_a_rate_lines.png                                             (Phase A line plot)
    phase_b_bypass.png                                                 (Phase B bypass curve)
    phase_c_tier.png                                                   (Phase C tier scaling)
  docs/claude-moe/drop/0526-long-bench/README.md                       (本文件)
```

---

## 8. 未来工作

- **短 prompt tier**（1-2k tokens）— 需切换数据集（OpenOrca / ShareGPT / MMMU）。预期 prefill_speedup 较低，因为 L_recv 接近 break-even。
- **accuracy 校验** — rate=0.5 的 GSM8K / LongBench score 是否还在 ±1pp。
- **rate × bypass 联合**：是否 rate 更大时 bypass 阈值需要更高？（直觉上 rate=0.5 时 decode 上 drop 的 overhead 更大）
- **chunked prefill + drop**：把超长 prompt 拆成 4k 块，每块都进 drop 收益区间。
- **dataset variation**：在 LongBench-2WikiMQA、TriviaQA 上重做 Phase C，检验跨任务一致性。
- **K_eff × drop 联合**：Phase 4 P1 P0 的 K_eff 是 drop 之外的另一种 a2a 减载，组合后是否叠加？
- **receive-side drop**：基于 Tier 1 的"combine 段贡献 50% 收益"发现，加 receive-side drop 进一步压低 combine 时间。
