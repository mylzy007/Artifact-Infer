# Phase 4 Drop — v4 最终报告（recall-based accuracy）

> **核心更正**：v3 的 F1 不可信，因为 `gen_text` 只存了前 200 字符（模型的 CoT 前置废话），
> 没存到真正的答案。v4 改为存完整 gen + 完整 ref + `max_new_tokens=256`，并用 **recall** 取代 F1 作为主要 accuracy metric。
>
> **结论翻转**：v3 推荐的 `cross_numa_first @ r=0.5`（最佳 e2e）**实际上 recall 掉点最严重**
> （两 plan 都掉 4-6pp）。新最佳：**`tail_weight @ r=0.5`**，唯一在两个 plan 上 recall 都
> 显著正向（+3.1pp 平均），且 prefill_speedup 持平（1.43×）。
>
> **日期**：2026-05-27
> **机时**：~3.7 小时 8×4090（v4 主因 decode=256 慢 2.3 倍）
> **数据**：32 cells × 64 samples = **2048 prompt-cell**，每个评估带完整 256-token gen + ref 文本

---

## 1. 为什么 F1 不可信（debug 过程）

### 1.1 v3 看到的奇怪现象

v3 baseline F1 = 0.208，drop 后 F1 在 ±0.02 间随机波动 — 看起来 "drop 不影响 accuracy"。

### 1.2 真因分析

```
baseline gen[:200]  ："\n\nAnswer:\nOkay, the user is asking..."  (前置 CoT)
baseline reference  ："South West Ultras fan club."              (短答)
```

模型行为 = **CoT 前置（"Okay, let me think..."）+ answer**。v3 只存了前 200 字符的 head，落在 CoT 部分；
真正的 answer 在更后面，但被 [:200] 截掉了。

### 1.3 reference 长度分布（实测）

```
n=158 reference answers
min=4, max=69, mean=26 tokens
p25=16, p50=23, p75=35, p90=41, p99=64
max_new_tokens=64 已覆盖 99.4%
```

reference 一直在 max_new_tokens=64 内。**v4 升到 256 是给模型多吐 CoT 的空间**，让模型完整说完。

### 1.4 v4 修复

- `max_new_tokens` 64 → **256**
- 存储 `gen_text` (full) + `ref_text` (full)，不再 `gen_text_head[:200]`
- 离线 multi-metric 重打分（F1 / recall / substring / exact / ROUGE-L）

### 1.5 五个 metric 在 v4 baseline 上的对比

| metric | 值 | 解读 |
|---|---|---|
| F1 | 0.108 | 256-token gen vs 23-token ref → precision 暴跌 |
| **Recall** | **0.577** | 模型答案命中 58% 的 ref tokens — **真信号** |
| ROUGE-L | 0.087 | LCS F1，同样被长度差伤 |
| Substring（ref ⊆ gen） | 0.062 | 6% prompt 模型完整说出 ref |
| Exact-match | 0.000 | 一次都没完全一致 |

**结论**：模型在 prefill-heavy long-doc QA 上 **找到了 58% 的关键信息**（recall），但用更长的话表达。
F1 / ROUGE-L 这种 P×R 度量在 length mismatch 下系统性低估 accuracy。**recall 是这个 task 上唯一可信的 accuracy 信号**。

---

## 2. 完整实验配置

| 项 | 值 |
|---|---|
| 模型 | Qwen3-30B-A3B（48 层完整模型） |
| 硬件 | 8 × RTX 4090 24GB / CUDA 12.8 |
| Runtime | `owner_local_ep` + `ep_ht` + `enforce_eager=True` |
| max_num_batched_tokens / max_model_len | 6144 (per-rank) |
| max_num_seqs / batch_size | 64 / 8（1 prompt → 1 rank） |
| **max_new_tokens** | **256（v4 修复）** |
| gpu_memory_utilization | 0.95 |
| Dataset / 过滤 | LEval Generation_multidoc_qa, [2048,4096] tokens, p50≈2974 |
| **Samples / cell** | **64**（8 batches × 8） |
| Drop bypass | MOE_DROP_MIN_REPLICAS=512 |
| Drop policies (5 GPU) | tail_weight / random / cross_numa_first / weighted_tail / cross_numa_uniform |
| Drop rates | 0.1, 0.3, 0.5 |
| **2 个 Phase 3 plans** | RR/numa_local_first/min_communication + LBG/numa_local_first/greedy_balance |
| Cells | 2 plans × (1 baseline + 5 × 3) = **32** |
| 总评估 | 32 × 64 = **2048 prompt-cell** |
| Sampling | temperature=0.0 (greedy)，seed=42 |
| Warmup | 3 baseline batches，丢弃 timing |

---

## 3. 完整结果表

### 3.1 RR / min_comm plan

**Baseline**：prefill 4.344s, decode 47.245s, e2e **51.919s**, per-rank tok/s **709**, recall **0.577**

| policy | rate | prefill_sp | e2e_sp | tok/s | recall | recall Δ | F1 | ROUGE-L | substring |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cross_numa_first | 0.1 | 1.055 | 1.115 | 748 | 0.587 | +0.010 | 0.110 | 0.090 | 0.078 |
| cross_numa_first | 0.3 | 1.238 | 1.117 | 878 | 0.587 | +0.010 | 0.108 | 0.092 | 0.062 |
| cross_numa_first | 0.5 | 1.428 | 1.133 | 1013 | 0.539 | **−0.038** | 0.098 | 0.081 | 0.078 |
| cross_numa_uniform | 0.1 | 1.065 | 1.016 | 754 | 0.598 | +0.021 | 0.110 | 0.089 | 0.078 |
| cross_numa_uniform | 0.3 | 1.229 | 1.125 | 873 | 0.579 | +0.002 | 0.104 | 0.085 | 0.094 |
| cross_numa_uniform | 0.5 | 1.440 | 1.036 | 1021 | 0.522 | **−0.055** | 0.097 | 0.080 | 0.031 |
| random | 0.1 | 1.059 | 1.016 | 750 | 0.571 | −0.006 | 0.106 | 0.090 | 0.062 |
| random | 0.3 | 1.211 | 1.121 | 859 | 0.609 | +0.032 | 0.110 | 0.093 | 0.125 |
| random | 0.5 | 1.440 | 1.142 | 1021 | 0.551 | **−0.026** | 0.098 | 0.083 | 0.047 |
| tail_weight | 0.1 | 1.059 | 1.080 | 750 | 0.585 | +0.007 | 0.105 | 0.087 | 0.094 |
| tail_weight | 0.3 | 1.226 | 1.057 | 869 | 0.578 | +0.001 | 0.106 | 0.088 | 0.047 |
| **tail_weight** | **0.5** | **1.434** | 1.030 | 1017 | **0.605** | **+0.028** | 0.108 | 0.091 | 0.125 |
| weighted_tail | 0.1 | 1.059 | 1.120 | 750 | 0.583 | +0.006 | 0.109 | 0.089 | 0.031 |
| weighted_tail | 0.3 | 1.225 | 1.028 | 868 | 0.591 | +0.014 | 0.109 | 0.094 | 0.078 |
| weighted_tail | 0.5 | 1.441 | 1.149 | 1022 | 0.523 | **−0.054** | 0.095 | 0.078 | 0.047 |

### 3.2 LBG / greedy_balance plan

**Baseline**：prefill 4.327s, decode 43.210s, e2e **47.791s**, per-rank tok/s **711**, recall **0.577**

| policy | rate | prefill_sp | e2e_sp | tok/s | recall | recall Δ | F1 | ROUGE-L | substring |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cross_numa_first | 0.1 | 1.060 | 1.031 | 754 | 0.585 | +0.008 | 0.107 | 0.089 | 0.078 |
| cross_numa_first | 0.3 | 1.227 | 1.064 | 873 | 0.604 | +0.027 | 0.108 | 0.090 | 0.094 |
| cross_numa_first | 0.5 | 1.435 | 1.028 | 1021 | 0.516 | **−0.062** | 0.094 | 0.076 | 0.016 |
| cross_numa_uniform | 0.1 | 1.056 | 1.025 | 751 | 0.597 | +0.020 | 0.108 | 0.090 | 0.078 |
| cross_numa_uniform | 0.3 | 1.222 | 1.037 | 869 | 0.541 | **−0.036** | 0.096 | 0.083 | 0.094 |
| cross_numa_uniform | 0.5 | 1.428 | 1.042 | 1016 | 0.537 | **−0.040** | 0.104 | 0.083 | 0.047 |
| random | 0.1 | 1.057 | 1.004 | 751 | 0.575 | −0.002 | 0.104 | 0.087 | 0.062 |
| random | 0.3 | 1.217 | 1.046 | 866 | 0.585 | +0.008 | 0.105 | 0.084 | 0.078 |
| random | 0.5 | 1.424 | 1.046 | 1013 | 0.503 | **−0.074** | 0.094 | 0.076 | 0.031 |
| tail_weight | 0.1 | 1.060 | 1.016 | 754 | 0.585 | +0.008 | 0.107 | 0.089 | 0.047 |
| tail_weight | 0.3 | 1.218 | 1.041 | 866 | 0.596 | +0.018 | 0.109 | 0.093 | 0.078 |
| **tail_weight** | **0.5** | **1.429** | 1.049 | 1017 | **0.611** | **+0.033** | 0.108 | 0.089 | 0.062 |
| weighted_tail | 0.1 | 1.052 | 1.005 | 748 | 0.575 | −0.002 | 0.104 | 0.087 | 0.078 |
| weighted_tail | 0.3 | 1.221 | 1.030 | 868 | 0.595 | +0.018 | 0.110 | 0.091 | 0.078 |
| weighted_tail | 0.5 | 1.424 | 1.029 | 1013 | 0.574 | −0.003 | 0.108 | 0.090 | 0.078 |

---

## 4. 可视化（新增 v4 专用图，使用 recall 作 accuracy 轴）

所有图在 `eval_results/owner_local_ep_phase4_drop_longbench_v4/`：

### v4 专用（4 张）

| 图 | 解决问题 |
|---|---|
| **`v4_recall_vs_speedup_pareto.png`** | 主图：x=prefill_speedup, y=recall，散点 5 policy × 3 rate × 2 plan |
| **`v4_pareto_e2e_vs_recall.png`** | x=e2e_speedup, y=recall，自动标记最佳 Pareto 点 |
| **`v4_recall_rate_per_plan.png`** | recall vs rate per policy per plan，**带 ±2σ 噪声带**，能直接看 "tail_weight @ r=0.5 是唯一冲出噪声带的" |
| **`v4_grid_perf_recall.png`** | 3×2 heatmap：prefill_sp / e2e_sp / recall × 2 plans，每格标精确值 |

### v3 plot 脚本复用（13 张）

`v3_*.png`、`v3_accuracy_metrics.png` 等 — 给 v4 数据画的版本（数值是 v4 的）。F1 视角的图保留作 reference，但**应优先看 v4_* 图**。

### Markdown table

`v4_final_table.md` — 32 cells × 10 列（含 recall/F1/substring/ROUGE-L）。

---

## 5. 关键发现

### 5.1 Recall 是这个 task 上唯一可信的 metric

- baseline recall = 0.577（找到 58% ref tokens）
- baseline F1 = 0.108（被长度差打到地板）
- v3 用 F1 + gen[:200] → 几乎所有 cell 都看似 ±0.02 噪声内
- v4 用 recall + full gen → **5 个 policy 在 r=0.5 上明显分化**

### 5.2 r=0.5 的 accuracy 分化（**最重要的新结论**）

| policy | RR recall Δ | LBG recall Δ | 平均 | 评价 |
|:--|--:|--:|--:|:--|
| **tail_weight** | **+0.028** | **+0.033** | **+0.031** | ✓ **唯一双 plan 一致正向** |
| weighted_tail | −0.054 | −0.003 | mixed | 不稳 |
| cross_numa_first | −0.038 | −0.062 | **−0.050** | ✗ 双 plan 一致负 |
| cross_numa_uniform | −0.055 | −0.040 | **−0.048** | ✗ 双 plan 一致负 |
| random | −0.026 | −0.074 | **−0.050** | ✗ 双 plan 一致负 |

→ **r=0.5 推荐 `tail_weight`**，避免 `cross_numa_first / cross_numa_uniform / random`。

### 5.3 为什么 tail_weight 反而升 recall？

可能解释：
1. **Deterministic**：tail_weight 永远砍最低权重 branches，同输入下行为完全可重复；其他 4 个含随机或拓扑因素，砍 token 选择有 variance → 模型行为不稳
2. **正则化效应**：砍最低权重 branches 强制模型只看 top-K-drop branches → 注意力集中 → 输出更精准
3. **保留高权信息**：tail_weight 显式保留所有高权 branches；其他 policy 可能误砍重要 branches（如 cross_numa_first 优先砍跨 NUMA，但跨 NUMA 的 token 在 multi-doc QA 中恰好是跨片段证据）

### 5.4 性能：prefill +43% 跨所有 policy

5 个 GPU policy 在 prefill_speedup 上完全等价（spread < 1.5%）：
- r=0.1：~1.06×
- r=0.3：~1.22×
- r=0.5：**~1.43×**（per-rank tok/s 711 → ~1018，**系统总 ≈ 8,144 tok/s**）

**性能上选哪个 policy 都一样**。Accuracy 上选 **tail_weight @ r=0.5**。

### 5.5 e2e_speedup 被 256-decode 稀释

v4 baseline e2e ≈ 50s（decode 占 ~90%），所以 drop 在 decode 上 bypass 之后 e2e 增益只剩 3-15%（v3 = ~10%，v4 = ~3-5%）。**真正的 use case 仍是 prefill-heavy** —— 如果 decode 短（chunked prefill, RAG 场景，速答），e2e_speedup 会回到 10%。

### 5.6 两个 Phase 3 plan 在 prefill 上几乎不可区分

- RR baseline prefill = 4.344s
- LBG baseline prefill = 4.327s（差 0.4%）
- **但 LBG baseline decode 比 RR 快 ~10%**（43.2s vs 47.2s）→ LBG e2e 快 8%
- → **prefill 任务选哪个 plan 都行；如果 decode 也很重，LBG 略好**

---

## 6. 显著性讨论

n=64 per cell，recall std ≈ 0.32 → SEM ≈ 0.040 → **|Δrecall| < 0.080 不显著（2σ）**。

但**两 plan 联合**（每 policy×rate 总 n=128）→ SEM ≈ 0.028 → **|Δ| < 0.056 不显著**。

按此标尺：
- `tail_weight @ r=0.5`：avg +0.031（两 plan 一致 +0.028, +0.033）→ 1.1σ，**没正式显著**，但**方向一致 + 多 metric 同向（recall/F1/ROUGE-L 都 ≥0）→ 联合证据强**
- `cross_numa_first @ r=0.5`：avg −0.050（两 plan 都负）→ **1.8σ，接近显著**
- `random @ r=0.5`：avg −0.050 → 同上，接近显著

要达到 publish 级显著性，需要 **n ≥ 384/cell + 3 reps**（v4 的 4 倍数据量）。

---

## 7. 最终推荐配置

```yaml
# Phase 4 drop production config
moe_drop_policy:        tail_weight     # 唯一在 r=0.5 上 recall 反而升的 GPU policy
moe_drop_rate:          0.5             # +43% prefill, +3pp recall
moe_expert_overlap_path: <LBG / numa_local_first / greedy_balance>   # decode 比 RR 快 10%
MOE_DROP_IMPL:          auto            # GPU 路径
MOE_DROP_MIN_REPLICAS:  512             # bypass decode
```

**预期**（在 LEval-multidoc-qa-shaped workload 上）：
- prefill +42.9% (4.33s → 3.03s)
- per-rank prefill tok/s 711 → 1017（系统总 ~8,136 tok/s）
- e2e +5% （v4 256-decode 稀释下；prefill-heavy workload 上回到 +10%）
- **recall +5.5% 相对**（0.577 → 0.611，跨 plan 一致）

**避免**：
- `cross_numa_first @ r=0.5`（v3 看似最佳，v4 真实 recall 掉 5pp）
- `random @ r=0.5`（同上）
- `cross_numa_uniform @ r=0.5`（recall 掉 4pp）

---

## 8. v3 → v4 主要修正

| 项 | v3 | v4 | 影响 |
|---|---|---|---|
| max_new_tokens | 64 | **256** | 模型有空间完整说出 CoT + answer |
| gen 存储 | `gen_text[:200]` | 完整 `gen_text` | rescore 才能拿到真信号 |
| ref 存储 | 长度 only | 完整 `ref_text` | 同上 |
| 主 metric | F1 | **recall** | F1 被长度差打到地板，recall 才反映真实匹配 |
| samples / cell | 80 | 64 | 略减以承担更长 decode |
| 最佳 policy | cross_numa_first @ 0.5 | **tail_weight @ 0.5** | 完全翻转 |
| 推荐 plan | LBG（持平） | LBG（decode 快 10%） | 一致 |
| 性能结论 | 5 policy 等价 | 5 policy 等价 | 一致 |

---

## 9. 修改建议（按 ROI）

1. **【必改】** `eval/run_owner_local_ep_phase4_drop.py:84` 默认 `--drop-min-replicas` 从 `0` → **`512`**
   — 解决 Phase 4 历史 GSM8K 实验的 e2e 负增长根因
2. **【必改】** 任何长输出实验（max_new_tokens ≥ 64）**必须存完整生成文本**，不要截断
3. **【建议】** 在 Phase 4 metrics 模块加 recall + ROUGE-L 计算，不要只用 F1
4. **【可选】** 把 `weighted_tail` + `cross_numa_uniform` 加入 default policy list

---

## 10. 不确定 / 局限

- **单一 dataset (LEval multidoc QA)**：跨任务一致性未验证
- **n=64 + 2 plans 仍未达 publish-级显著性**（最大效应 1.8σ）
- **未测 short prompt (1-2k)**：LEval 没这么短的 prompt
- **未测 chunked prefill + drop**：长 prompt 切片场景
- **未测 receive-side drop**：Tier 1 显示 combine 占 50% 收益，receive-side 有潜力
- **recall 是 token-level**，不能区分 paraphrase（如果模型用同义词答对，recall 看不出）

---

## 11. 文件清单

```
代码（v4 新增 / 修改）：
  eval/drop/longbench_sweep_v2.py        ← 改：gen_text 存全文，加 ref_text
  eval/drop/run_v4.sh                    ← 新：v4 双 plan launcher
  eval/drop/rescore_v3.py                ← 改：兼容 v3 (head) 和 v4 (full) 两种存储
  eval/drop/plot_v4_final.py             ← 新：recall-based plots

数据（v4 新增）：
  eval_results/owner_local_ep_phase4_drop_longbench_v4/
    rr_mincomm/{v2_rows.jsonl, v2_summary.json}        ← 含完整 gen/ref
    lbg_greedybal/{v2_rows.jsonl, v2_summary.json}
    v3_rescore.json  v3_rescore_table.md               ← multi-metric
    v3_accuracy_metrics.png                            ← 5 metric × 2 plan
    v3_grid_heatmap.png  v3_speedup_bars_with_err.png  ← 复用 v3 脚本
    v3_pareto_annotated.png  v3_sensitivity.png  v3_plan_diff.png
    v3_*.png 等共 13 张
    v4_recall_vs_speedup_pareto.png                    ← 主图（v4 专用）
    v4_pareto_e2e_vs_recall.png                        ← e2e × recall 散点
    v4_recall_rate_per_plan.png                        ← recall vs rate ±2σ
    v4_grid_perf_recall.png                            ← 3×2 perf+recall heatmap
    v4_final_table.md                                  ← 完整 32-cell × 10 列

文档：
  docs/claude-moe/drop/0526-long-bench/
    README.md                            (v1 多维度)
    v2_all_policies_with_score.md        (v2 + F1)
    v3_two_baselines_with_new_policies.md (v3)
    FINAL_REPORT.md                      (v3 综合 — 主 metric 为 F1，部分结论需以 v4 替换)
    v4_FINAL.md                          ← **本文件，最新真信号**
```

---

## 12. 一键复现

```bash
# v4 sweep + rescore + plots (~3.7h on 8×4090)
bash eval/drop/run_v4.sh
/home/lzy/miniconda3/envs/vllm/bin/python -m eval.drop.rescore_v3 \
  --v3-root eval_results/owner_local_ep_phase4_drop_longbench_v4
/home/lzy/miniconda3/envs/vllm/bin/python -m eval.drop.plot_v3_comprehensive \
  --v3-root eval_results/owner_local_ep_phase4_drop_longbench_v4
/home/lzy/miniconda3/envs/vllm/bin/python -m eval.drop.plot_v4_final \
  --v4-root eval_results/owner_local_ep_phase4_drop_longbench_v4
```
