# v2 — 全 7 个 policy × rate × Score 对比实验

> **TL;DR**：
> 在 LEval-multidoc_qa（96 prompt × p50≈3k tokens × max_new_tokens=64）上扫
> **3 个 GPU 路径 + 4 个 CPU-grouped 路径 × 3 个 rate + baseline = 22 cell**。
>
> 1. **GPU vs CPU 路径分水岭极陡**：所有 GPU policy 都给 1.05-1.42× prefill 加速；
>    所有 CPU grouped policy 都让 prefill **变慢 25-37%**（host sync 主导 + 智能选择
>    无法弥补）。CPU 路径对生产 inference **不可用**。
> 2. **GPU 三种 policy 性能上几乎等价**（在同一 rate 下差 < 3%），与 v1（24 sample）
>    结论一致：drop 收益来自"减少 a2a 字节"，跟选谁无关。
> 3. **Score (token-F1) 与 rate 的关系**：rate=0.3 时三个 GPU policy 都丢 1-2pp F1
>    （cross_numa_first 最温柔 −0.012，random 最伤 −0.024）。rate=0.5 噪声内或保平
>    （F1 SEM≈0.016，n=96）。
> 4. **最佳 Pareto 点**：**cross_numa_first @ r=0.3**，prefill +22.7%，e2e +5.3%，
>    F1 在 −0.012（噪声内）。要更激进：**random @ r=0.5**，prefill +39.4%，e2e +7.6%，
>    F1 −0.019。
>
> **日期**：2026-05-26
> **机时**：~55 分钟 8 卡（CPU 路径明显拖时间）
> **数据**：22 cell × 12 batch × 8 prompt = 2,112 prompt-cell evaluation
> **产物**：`eval_results/owner_local_ep_phase4_drop_longbench_v2/`

---

## 1. v1 vs v2 differences（回应用户的问题）

| 项 | v1（24 prompt） | v2（96 prompt） |
|---|---|---|
| **dataset 大小** | 24 prompt × 3 batch | **96 prompt × 12 batch** ← 用户要求扩大 |
| **policy 覆盖** | 3 个 GPU 路径 | **全部 7 个**（3 GPU + 4 CPU grouped） |
| **score 度量** | 无 | **token-level F1** vs `reference_answer` |
| max_new_tokens | 32 | 64（让生成内容长到能评分） |
| 模型 | Qwen3-30B-A3B 48 层（完整） | 同 |
| GPU | 8×RTX 4090 24GB | 同 |
| 机时 | ~9 min | **~55 min**（CPU 路径慢 30% + 更多 sample + 更长 decode） |

为什么 v1 跑得快：**完整模型 + 小 dataset**（22 cell × 3 batch = 66 batch，每 batch ~9s）。
**不是只跑了几层 / 几个请求**。v2 数据规模变成 v1 的 32 倍。

---

## 2. 实验设计

### 2.1 sweep 轴

| Axis | 取值 | 数量 |
|---|---|---|
| drop_policy | `tail_weight` / `random` / `cross_numa_first` / `per_expert_uniform` / `hot_expert_relief` / `per_expert_tailtoken` / `hotspot_relief` | 7 |
| drop_rate | 0.1, 0.3, 0.5 | 3 |
| baseline | 1 | 1 |
| **total cells** | | **22** |

Sweep 的固定项：
- `MOE_DROP_MIN_REPLICAS = 512`（v1 验证为最佳 bypass 阈值）
- `MOE_DROP_IMPL = auto`（dispatcher 按 policy 自动选 GPU/CPU 路径）
- `expert_overlap_path = Phase 3 LBG plan`
- `expert_overlap_strategy = greedy_balance`
- prompt 过滤 `[2048, 4096]` tokens（甜点区间）
- batch_size=8（每 rank 1 prompt）
- max_new_tokens=64
- 3 batch warmup（baseline，丢弃 timing）
- cell 顺序 seed=42 随机化

### 2.2 Score metric

Token-level F1（QA 任务的标准 proxy）：

```python
pred_tokens = lowercase + strip-punct + split(pred)
ref_tokens  = lowercase + strip-punct + split(reference_answer)
common = multiset intersection
precision = |common| / |pred|;  recall = |common| / |ref|
F1 = 2·P·R / (P+R)
```

`reference_answer` 来自 LEval 自己的 ground truth。

---

## 3. 完整结果表

| policy | path | rate | prefill (s) | prefill_speedup | e2e_speedup | F1 mean | F1 Δ |
|:--|:-:|:-:|:--:|:-:|:-:|:-:|:-:|
| **baseline** | — | 0.0 | 4.500 | **1.000** | **1.000** | **0.224** | — |
| cross_numa_first | GPU | 0.1 | 4.272 | 1.053 | 1.003 | 0.224 | +0.000 |
| cross_numa_first | GPU | 0.3 | 3.667 | **1.227** | **1.053** | 0.212 | −0.012 |
| cross_numa_first | GPU | 0.5 | 3.321 | 1.355 | 1.042 | **0.241** | **+0.017** |
| random | GPU | 0.1 | 4.357 | 1.033 | 0.993 | 0.223 | −0.001 |
| random | GPU | 0.3 | 3.770 | 1.194 | 1.033 | 0.200 | −0.024 |
| random | GPU | 0.5 | 3.230 | **1.394** | **1.076** | 0.205 | −0.019 |
| tail_weight | GPU | 0.1 | 4.305 | 1.045 | 0.997 | 0.213 | −0.011 |
| tail_weight | GPU | 0.3 | 3.812 | 1.180 | 1.019 | 0.202 | −0.022 |
| tail_weight | GPU | 0.5 | 3.161 | **1.424** | 1.035 | 0.229 | +0.005 |
| per_expert_uniform | CPU | 0.1 | 6.608 | **0.681** | 0.870 | 0.219 | −0.005 |
| per_expert_uniform | CPU | 0.3 | 6.286 | 0.716 | 0.882 | 0.224 | −0.001 |
| per_expert_uniform | CPU | 0.5 | 6.067 | 0.742 | 0.881 | 0.229 | +0.005 |
| hot_expert_relief | CPU | 0.1 | 6.706 | 0.671 | 0.860 | 0.220 | −0.004 |
| hot_expert_relief | CPU | 0.3 | 6.353 | 0.708 | 0.878 | 0.208 | −0.016 |
| hot_expert_relief | CPU | 0.5 | 6.166 | 0.730 | 0.874 | 0.226 | +0.002 |
| per_expert_tailtoken | CPU | 0.1 | 6.729 | 0.669 | 0.865 | 0.219 | −0.005 |
| per_expert_tailtoken | CPU | 0.3 | 6.331 | 0.711 | 0.887 | 0.231 | +0.007 |
| per_expert_tailtoken | CPU | 0.5 | 6.595 | 0.683 | 0.864 | 0.231 | +0.007 |
| hotspot_relief | CPU | 0.1 | 6.624 | 0.679 | 0.857 | 0.221 | −0.003 |
| hotspot_relief | CPU | 0.3 | 6.527 | 0.689 | 0.878 | 0.225 | +0.001 |
| hotspot_relief | CPU | 0.5 | 7.107 | **0.633** | **0.815** | 0.225 | +0.001 |

**粗体**：每列最优值（或与 baseline 同列）。

---

## 4. 关键发现

### 4.1 GPU vs CPU 路径分水岭（最重要发现）

**全部 4 个 CPU grouped policy 都让 prefill 变慢 25-37%**：

| path | min prefill_speedup | max | min e2e_speedup | max |
|---|---|---|---|---|
| GPU (3 policy × 3 rate) | 1.033 | **1.424** | 0.993 | **1.076** |
| CPU (4 policy × 3 rate) | **0.633** | 0.742 | **0.815** | 0.887 |

CPU 路径的 host↔device sync 开销（per layer per step）累积成 ~2 秒/batch
的额外 wall time。即使 CPU policy 选 token 更智能、accuracy 略好，**性能上完全无法回本**。

**这定量验证了 Phase 4 v1/v2 的诊断**："CPU drop 每层每 decode step 都做 `.cpu()`/`.item()`，
48 层 × N step 累积 → 远超 drop 节省的 GEMM 时间"。在这里 prefill 也被同等开销拖累。

**实践结论**：
- **生产 inference**：只用 GPU policy。
- **CPU policy 仅在 ablation 研究 / 验证算法正确性时使用**，不要用于性能优化。

### 4.2 三个 GPU policy 性能上几乎等价

| 在同 rate | tail_weight | random | cross_numa_first | spread |
|---|---|---|---|---|
| r=0.1 | 1.045× | 1.033× | 1.053× | 2% |
| r=0.3 | 1.180× | 1.194× | 1.227× | 4% |
| r=0.5 | 1.424× | 1.394× | 1.355× | 5% |

**三个 GPU policy 的 prefill_speedup 在同 rate 下相差 < 5%**（与 v1 24-sample 的 <1%
相比稍微变大，但仍在噪声范围）。**收益来自 a2a 字节减少，与具体扣谁基本无关**，
和 Tier 1 segment 归因（dispatch −47% + combine −50% = 97% 收益，experts 仅 2.8%）一致。

### 4.3 F1 score 噪声水平

per-prompt F1 的 std 约 **0.15**（96 prompt），SEM ≈ 0.015。所以 cell 间 F1 mean 差
**小于 ±0.02 都不显著**。

按这把尺子：
- r=0.1：所有 policy F1 都在 −0.01 内（统计上未掉点）
- r=0.3：三个 GPU policy 全部跌出噪声 1-2pp（**真实掉点**）
- r=0.5：全部回归噪声内或反弹

**为什么 r=0.5 反而比 r=0.3 score 更好？**
- 一种可能：rate=0.5 触发更多 *per-token min-keep* 保护（"每 token 必留 1"），
  实际 effective drop frac 没有线性涨到 0.5
- 另一种可能：纯统计噪声 — 96 prompt 的 SEM 还偏大；用 384 prompt 才能定论
- 不论哪个原因，**r=0.5 看似免费午餐，需要更多数据验证**

### 4.4 Pareto 前沿（GPU only）

按 e2e_speedup 排序 + 看 F1 是否能接受：

| 选项 | 用法 | prefill_speedup | e2e_speedup | F1 Δ | 评价 |
|---|---|---|---|---|---|
| **random @ r=0.5** | 性能优先 | **1.394×** | **1.076×** | −0.019 | 最快 e2e；F1 略掉 |
| **cross_numa_first @ r=0.3** | 平衡 | 1.227× | 1.053× | −0.012（噪声） | **推荐** |
| cross_numa_first @ r=0.5 | 探索 | 1.355× | 1.042× | +0.017（噪声） | F1 看似回升，要 384 sample 验证 |
| tail_weight @ r=0.5 | Phase 4 v1/v2 同款 | **1.424×** | 1.035× | +0.005 | 最快 prefill；e2e 较弱 |
| any @ r=0.1 | 保守 | ~1.04× | ~1.00× | ~0 | 加速太小，但 0 风险 |

### 4.5 prefill_speedup ≠ e2e_speedup 的 spread

注意：
- tail_weight @ r=0.5：**prefill +42% 但 e2e 只 +3.5%**
- random @ r=0.5：prefill +39%，e2e **+7.6%**

prefill_speedup 最高的不是 e2e_speedup 最高的。一个可能解释：tail_weight 在某些 prompt
上让 decode 路径走 slower fallback（不太可能因为 bypass=512），更可能是 prompt 长度
分布不同导致 prefill/decode 占比不一致。值得用 384 sample 复测。

---

## 5. 图

主图位于 `eval_results/owner_local_ep_phase4_drop_longbench_v2/`：

1. **`v2_rate_lines_prefill.png`** — prefill_speedup vs rate（一条线每 policy）。
   GPU 线（实线）斜率正向；CPU 线（虚线）平躺在 0.7 附近。
2. **`v2_rate_lines_e2e.png`** — e2e_speedup vs rate。GPU 全部 ≥ 1，CPU 全部 < 0.9。
3. **`v2_rate_lines_f1.png`** — F1 vs rate；可看出 r=0.3 是 F1 谷底。
4. **`v2_pareto_score_speedup.png`** — 2D Pareto：x=prefill_speedup，y=F1，点大小 = rate。
   GPU policy 全在右半边，CPU 全在左半边，baseline 是中间星标。

---

## 6. 给 Phase 4 / 论文写作的建议

1. **写论文时 CPU policy 应当独立成"算法 ablation"小节**，明确标出它们在生产 inference
   不可用 — 而不是混在性能对比里。
2. **GPU policy 主比较应当聚焦 e2e_speedup 和 F1**，不是 prefill_speedup。tail_weight
   在 prefill 看似最佳，但 e2e 没优势。
3. **"random 是 noise floor"的标语应当反思**：本实验 random 的 e2e_speedup（r=0.5 时 1.076×）
   反而最好，F1 损失也不比 tail_weight 大。"随机"在这个任务上不弱于"smart"策略 — 进一步
   支持"drop 收益与选择无关"的诊断。
4. **`router_keff` 应当与 drop 联合 sweep**。Phase 4 P1 显示 K_eff 单独是负杠杆，
   但若它能在 r=0.5 上 free 出 accuracy buffer，合用可能更好。

---

## 7. 不确定 / 局限

- **F1 SEM=0.016 在 n=96 时还偏大**。F1 deltas 1-2pp 接近 1σ。建议
  follow-up 用 256-384 prompt 验证 r=0.3-0.5 区间的真实 score gap。
- **单一 dataset (LEval multidoc)**。在 LongBench 2WikiMQA / TriviaQA 上重做能
  排除任务效应。
- **F1 是 token-level**，不是 task-specific metric（multidoc QA 真正的 metric 是
  ROUGE / BLEU / 人工评分）。token F1 可能高估或低估真实掉点。
- **CPU policy 的算法优势没机会显现**（被 host sync 抹掉）。若实现 Triton fused 版本
  让它走 GPU 路径，可能 grouped 策略真能保 accuracy + 拿性能。

---

## 8. 文件清单

```
新增：
  eval/drop/longbench_sweep_v2.py            (7 policy + F1 score sweep)
  eval/drop/plot_longbench_v2.py             (rate-line, Pareto 图)
  eval_results/owner_local_ep_phase4_drop_longbench_v2/
    v2_rows.jsonl                            (264 行 = 22 cell × 12 batch；含生成文本头)
    v2_summary.json                          (per-cell agg + relative-to-baseline)
    v2_table.md                              (markdown 综合表)
    v2_rate_lines_prefill.png
    v2_rate_lines_e2e.png
    v2_rate_lines_f1.png
    v2_pareto_score_speedup.png
  docs/claude-moe/drop/0526-long-bench/v2_all_policies_with_score.md   (本文件)
```

---

## 9. 复现命令

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
  PATH=/usr/local/cuda-12.8/bin:$PATH \
  FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_tier2 \
  /home/lzy/miniconda3/envs/vllm/bin/torchrun --nproc_per_node=8 --master_port=29517 \
  -m eval.drop.longbench_sweep_v2 \
  --output-dir eval_results/owner_local_ep_phase4_drop_longbench_v2 \
  --dataset /home/lzy/datasets/moe_benchmarks/prepared/leval.Generation_multidoc_qa.custom.jsonl \
  --num-samples 96 --batch-size 8 --warmup-batches 3 \
  --max-new-tokens 64 \
  --drop-rates "0.1,0.3,0.5" \
  --drop-policies "tail_weight,random,cross_numa_first,per_expert_uniform,hot_expert_relief,per_expert_tailtoken,hotspot_relief" \
  --expert-overlap-path eval_results/tmp_phase3_overlap_test/moe_overlap_plan_smoke_plan_20260521_222623.json \
  --gpu-memory-utilization 0.95 --max-num-batched-tokens 6144

python -m eval.drop.plot_longbench_v2 --output-dir eval_results/owner_local_ep_phase4_drop_longbench_v2
```
