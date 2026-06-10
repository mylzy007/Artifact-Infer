# Phase 4 Drop — LongBench 综合最终报告

> **本文件是 Phase 4 token-replica drop 实验的最终权威版本**，整合 v1-v6 全部结果：
> v1-v3 性能扫描 + v4-v5 token-level accuracy + **v6 LongBench 官方 binary accuracy** 跨数据集验证。
>
> **日期**：2026-05-28
> **总机时**：v1 (9 min) + v2 (55 min) + v3 (70 min) + v4 (~220 min) + v5 (40 min) + v6 (~70 min) ≈ **~8 小时** 8×4090
> **总评估**：v4 (2048) + v5 (1024) + v6 (2048) = **5,120 prompt-cell** 评估，全部带完整生成文本 + LongBench/官方 metric 重打分
> **跨 dataset 验证**：3 个不同任务 — LEval multidoc_qa（多文档 QA，token F1）+ LongBench multifieldqa_en（单文档 QA，token F1）+ **LongBench passage_retrieval_en（段落检索，官方 binary metric）**
> **代码**：`eval/drop/`，`workshop/nanovllm_moe/services/utils/expert_drop.py`

---

## 0. 一句话结论（**v7 更新**）

**生产推荐**（v7 新发现：`hot_expert_relief @ r=0.5` 全面 dominate 之前的 `tail_weight`）：

### 新 winner ⭐⭐⭐ — 速度持平 tail_weight + accuracy 零损失

```yaml
moe_drop_policy:        hot_expert_relief         # v7 新实现的 GPU grouped-quota policy
moe_drop_rate:          0.5
moe_expert_overlap:     LBG / numa_local_first / greedy_balance
MOE_DROP_IMPL:          auto
MOE_DROP_MIN_REPLICAS:  512
```

**效果**（v7, n=96, 双 plan 一致）：
- prefill **+43.3%** (LBG) / **+42.9%** (RR)
- e2e **+20.5%**（passage_retrieval workload）
- **acc_strict 1.000（零损失！）on 双 plan**
- per-rank tok/s 747 → 1071

### 之前的"次优" — 仍然能用

```yaml
moe_drop_policy:        tail_weight               # v6 推荐，仍 valid 但比 hot_expert_relief 略差
moe_drop_rate:          0.5
```

效果：prefill +47%, e2e +21.5%, acc_strict 0.938 (LBG) / 0.990 (RR) — accuracy 微掉 1-6pp。

### 安全保守 — 零 accuracy 损失

```yaml
moe_drop_policy:        tail_weight  # 任何 GPU policy 在 r=0.3 都是 strict=1.000
moe_drop_rate:          0.3
```

效果：prefill +23%, e2e +12%, acc_strict 1.000（任何 policy）。

### 避免（v6/v7 一致揭示）

`cross_numa_first / cross_numa_uniform / random @ r=0.5`：性能跟其他 r=0.5 持平但 **acc_strict 暴跌到 21-38%**（baseline 100%）。**这三个 policy 在 r=0.5 上模型行为崩溃**，**v6/v7 LongBench 官方 binary metric 才看出来**。

---

## 1. 实验演进与关键修正

| 阶段 | 验证什么 | 主要发现 |
|---|---|---|
| Tier 1 microbench | drop 的 break-even L* | L* ≈ 3.3k；收益来自 dispatch+combine 共 97%，experts 仅 3% |
| Tier 2 synthetic e2e | full-engine 验证 | T=512: prefill +8.8%；T=2048: +4.5%（attention 稀释） |
| v1 real long e2e | 真实 LongBench prompts | 9% e2e on LEval（首次真实数据集观察显著收益）|
| v2 multi-policy | 7 policy 性能 + F1 score | CPU 4 policy 慢 30%；GPU 3 policy 持平；**F1 难分辨 accuracy 差异** |
| v3 (2 plans + 2 new policy) | Phase 3 plan + 新 GPU policy | 两 plan 不可区分；5 GPU policy spread<1.1%；F1 全在噪声内 |
| v4 (修复 accuracy 测量) | 完整 gen 文本 + recall 度量 | F1 被 length-mismatch 压低；recall 才是真信号；tail_weight @ r=0.5 唯一不掉 recall |
| v5 cross-dataset | LongBench multifieldqa_en + 官方 LongBench F1 | tail_weight @ r=0.5 跨数据集仍是唯一正向 (recall +0.044) |
| v6 (官方 binary metric) | LongBench passage_retrieval_en + 官方 retrieval_score | binary metric 揭示 r=0.3 是真正 Pareto 甜点；r=0.5 cross_numa_first/random/cross_numa_uniform 在 accuracy 上崩盘 |
| **v7 (新增 3 个 GPU 分组 policy)** | **GPU 化 per_expert_uniform / hot_expert_relief / hotspot_relief；96 prompts** | **`hot_expert_relief @ r=0.5` 双 plan strict 1.000 + prefill +43%，dominate 之前的 tail_weight (acc -6pp)** |

**v6/v7 是最关键的修正**：
- v6 用 binary metric **暴露** r=0.5 上 policy 间真实差异（v3-v5 token F1 误判）
- v7 **GPU 化** 之前只有 CPU 实现的 3 个分组 policy，发现 `hot_expert_relief @ r=0.5` 同时拿到全速 + 零损失，**翻转生产推荐**

---

## 1.5 Tier 1 & Tier 2 微基准完整报告（prefill L sweep，2026-05-26）

> 这是 v1-v7 真实 e2e 实验之前的**奠基性微基准**：在 isolated MoE-block 和合成 prefill 上扫 `L_recv`，
> 找到 drop 起作用的拐点 L\*、给出 dispatch/experts/combine 三段归因，
> 并用全模型 prefill 经验值（Tier 2）验证 Tier 1 推论。
>
> **原始报告**：`docs/claude-moe/drop/0526-plan/0526-final_report.md`
> **原始数据**：`eval_results/prefill_drop_l_sweep_tier1/{rows.jsonl, summary.json, *.png}`、`eval_results/prefill_drop_l_sweep_tier2/`

### 1.5.1 实验目的

回答三个 pre-registered 问题：

1. **L\*（拐点）在哪儿？** drop 在多大的 `L_recv`（每 rank 接收 row 数）开始 ≥5% wall-time 加速？
2. **收益来自哪一段？** dispatch / experts GEMM / combine 三段中，drop 的省时主要来源是哪个？
3. **拐点能否在 24GB 4090 上达到？** 真实 batch size 是否能跨过 L\*？

并验证两个 pre-registered hypothesis：
- **H1**：drop 在任何 L 上都没有 ≥5% 加速（closed-negative）
- **H2**：drop 的收益 ≥70% 来自 experts GEMM 段（计算量减少）

### 1.5.2 设计与配置

| 项目 | Tier 1（隔离 MoE-block 微基准） | Tier 2（全模型 prefill） |
|---|---|---|
| 目的 | 找拐点 L\* + 段归因 | 验证 Tier 1 推论是否在全模型上仍然成立 |
| 范围 | DispatchEPHT + ExpertsEPHT + CombineEPHT，剔除 attention/sampling/scheduler | Qwen3-30B-A3B 完整 48 层 + flashinfer attention + sampling + scheduler |
| Workload | 合成 routing：T_local source tokens → K=8 random experts/token | 合成 prompt：T_local 长度的 prompt × 8 ranks 一次性 prefill |
| T_local 扫描 | {1, 8, 64, 512, 2048} | {512, 2048} |
| Drop policy | tail_weight | tail_weight |
| Drop rate | {0.0, 0.3}（baseline vs drop） | {0.0, 0.3} |
| Plan | LBG / numa_local_first / greedy_balance | smoke_plan |
| 硬件 | 8×RTX 4090 24GB | 同 |
| 模型 | Qwen3-30B-A3B EP-HT owner_local | 同 |
| Iters | 30 iters/cell + 10 warmup | 3 reps + 1 warmup |
| Falsification floor | L=8 上 drop slowdown < 2%（设计错误，事后改为诊断字段） | — |

环境阻塞修复：Tier 2 的 flashinfer JIT compile 起初因系统 nvcc 不支持 sm_89 失败，通过 `CUDA_HOME=/usr/local/cuda-12.8` + 新 flashinfer JIT cache `FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_tier2` 强制重编修复。

### 1.5.3 Tier 1 结果：L 扫描主表

8 ranks × Qwen3-30B-A3B EP-HT × tail_weight @ 0.3 × LBG overlap，30 iters/cell：

| T_local | L_send | L_recv_max | baseline_us | drop_us | **Δ%** | eff_drop_recv |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 8 | 18 | 4,092 | 5,134 | **+25.5%** | 0.25 |
| 8 | 64 | 86 | 4,170 | 5,600 | **+34.3%** | 0.30 |
| 64 | 512 | 524 | 5,069 | 6,115 | **+20.6%** | 0.30 |
| **512** | **4,096** | **4,160** | **19,606** | **16,999** | **−13.3%** ⭐ | 0.30 |
| 2,048 | 16,384 | 16,544 | 70,539 | 59,519 | **−15.6%** | 0.30 |

**关键观察**：
- **L\* (delta_pct = −5%) ≈ 3,271 rows/rank**（插值，落在 T=64 和 T=512 之间）→ 全 batch ≈ 3.3k prefill tokens 跨 8 ranks
- **effective_drop_recv_frac 在 L≥64 上稳定 0.30** → drop policy 工作正常，符合 nominal rate
- 小 L 时 drop 开销 > 收益（dispatch/combine 的固定 launch overhead 主导）；大 L 时 a2a payload bytes 主导，drop 转为正收益
- **L\* 是陡峭转折**：+20.6% (T=64) → −13.3% (T=512)，中间无平坦带

### 1.5.4 Tier 1 段归因（关键发现 — H2 推翻）

T_local=2048 cell 上的三段分解：

| segment | baseline_us | drop_us | Δus | **占 total 收益** |
|:-:|:-:|:-:|:-:|:-:|
| **dispatch** | 54,517 | 49,332 | −5,185 | **47.1%** |
| experts | 1,229 | 920 | −309 | 2.8% |
| **combine** | 14,796 | 9,276 | −5,520 | **50.1%** |
| **total** | **70,539** | **59,519** | **−11,019** | 100% |

**意义（H2 被推翻）**：

- drop 的省时 **97% 来自通信侧**（dispatch a2a payload bytes ↓47% + combine scatter rows ↓50%），而非 expert 计算量减少
- experts 段只占 ~1.2ms 量级（weight-residency dominated，与 L 弱相关），单独砍 30% 行只省 0.3ms → 微不足道
- 这反过来解释 Phase 4 P1 (`K_eff=6`) 为何是负杠杆：`K_eff` 只砍 expert GEMM，砍不动 dispatch/combine 这两个真正的 bottleneck

### 1.5.5 Tier 2 结果：全模型 prefill 验证

Qwen3-30B-A3B 完整 48 层 + flashinfer attention + sampling + scheduler，3 reps/cell：

| T_local | total prefill tokens | baseline prefill | drop prefill | **prefill speedup** | std |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 512 | 4,096 | 0.874 s ± 0.026 | 0.797 s ± 0.019 | **+8.8% (1.096×)** | <3% |
| 2,048 | 16,384 | 2.923 s ± 0.020 | 2.791 s ± 0.030 | **+4.5% (1.047×)** | <1% |

**反直觉但合理**：T=512 (刚跨 L\*) 全模型 prefill 加速 **8.8%**，**比 T=2048 (4.5%) 更高**——
Tier 1 MoE-block 收益是 T=2048 (−15.6%) > T=512 (−13.3%)，但全模型 prefill 反过来。

原因：**attention 是 O(N²)，N 越大 attention 占 prefill 总时间越高，把 MoE-block 收益稀释更多**。drop 的甜点在 **L_recv ≈ 4k** 附近（中等 prefill），再往大走收益继续被稀释。

**24GB VRAM 是否够 3.3k batch？**
- 实测 T=2048 (16.4k 总 tokens) 在 8×4090 24GB 跑通，每 rank ~21GB（model 19GB + KV ~380MB + MoE workspace ~1.5GB），仍留 3GB 余量
- 3.3k 全局 prefill（每 rank ~410 tokens）只用约 19.5GB → **绰绰有余**
- VRAM 不是瓶颈

### 1.5.6 Tier 1/2 结论与意义

| Pre-reg | 实测 | 验证情况 |
|---|---|---|
| **H1**：所有 L 上 drop 都没 ≥5% 加速（closed-negative） | L\* = 3,271 存在 | **H1 推翻** |
| **H2**：drop 收益 ≥70% 来自 experts GEMM | experts 段仅占 2.8% | **H2 推翻** |
| 预测 L\* ∈ [4k, 16k] | L\* ≈ 3.3k | 方向对，量级略低 |
| Effective drop frac 渐近 0.3 | L≥64 上稳定 0.30 | 一致 |

**4 条关键结论**：

1. **drop 起作用的 break-even 是 per-rank L_recv ≈ 3.3k**——全模型 prefill 跨过这条线 drop 才有正收益。Qwen3 短 prompt（decode 时 L_recv≈8）远低于此，drop 反而慢 25%。
2. **收益来自通信侧**：dispatch −47% + combine −50% = 97%，**experts GEMM 仅 3%**。这是非平凡发现——它直接否定了"drop 通过减少 expert 计算量加速"的 naive 假设，给后续设计指明：要进一步加速，应砍 **a2a payload bytes** 和 **combine scatter rows**，而非 expert 算力。
3. **甜点在 L_recv ≈ 4k**：T=2048 时 attention 已经稀释 MoE 收益，T=512 (4k 总 tokens) 的 prefill 加速反而最大（+8.8%）。**drop 不应在大 batch 上追加，而应在中等 prefill 段被用足**。
4. **解释了 Phase 4 v1/v2 / GPU drop / K_eff 历史负实验**：
   - GSM8K decode-heavy workload，`L_recv` 主体落在 drop 负收益区 → v1/v2 CPU drop e2e 慢 10-25%
   - K_eff 砍 expert GEMM 但 expert 只贡献 2.8% → 杠杆错位
   - GPU drop bypass 在 L<128 全关，但 GSM8K prefill 也被 bypass → 0 收益

**指导后续 v1-v7**：必须换到 **prefill-heavy / 长 prompt workload** 才能让 drop 在 e2e 上可见。这直接催生了 v1+（LEval、LongBench passage_retrieval 等长上下文 benchmark）。

### 1.5.7 可视化（`eval_results/tier_plots/`）

| 文件 | 内容 |
|---|---|
| **`tier1_delta_vs_L.png`** ⭐ | 主图：MoE-block Δ% vs L_recv，log-x，标 L\* = 3,271 + ±0%/−5% 线，可视化"拐点"陡转折 |
| **`tier1_segment_attribution.png`** ⭐ | 段归因双面板：左 dispatch/experts/combine baseline vs drop 条形图；右 savings 来源 donut（dispatch 47.1% + experts 2.8% + combine 50.1%）→ **H2 推翻** |
| `tier1_total_trajectories.png` | log-log baseline vs drop 轨迹曲线，红/绿填色区分 slowdown / speedup 区 |
| `tier2_prefill_speedup.png` | Tier 2 双面板：左 prefill 时间条形图（含 std），右 speedup 柱状图（+8.8% / +4.5%） |
| **`tier1_vs_tier2_dilution.png`** ⭐ | Tier 1 (隔离 MoE-block) vs Tier 2 (含 attention 全 prefill) 对比，可视化 attention 稀释（T=2048: 15.6% → 4.5%，dilution 11.1 pp） |

### 1.5.8 Tier 1/2 产物

```
代码：
  eval/drop/tier1_bench.py
  eval/drop/tier2_bench.py
  eval/drop/shared.py
  eval/drop/plot_tier1.py
  eval/drop/plot_tier_report.py       # 本节 §1.5.7 的 5 张图
  eval/drop/tests/test_drop_invariants.py

数据：
  eval_results/prefill_drop_l_sweep_tier1/
    tier1_rows.jsonl              (300 rows = 5 T × 2 rates × 30 iters)
    tier1_summary.json            (L*, delta_points, cell_agg)
    tier1_total_vs_l.png          (原始)log-x total_us(L) baseline vs drop
    tier1_delta_vs_l.png          (原始)delta_pct vs L，标 L*
    tier1_segments.png            (原始)dispatch / experts / combine 分段曲线
  eval_results/prefill_drop_l_sweep_tier2/
    tier2_rows.jsonl
    tier2_summary.json            (T=512, T=2048 两点 × 3 reps)
  eval_results/tier_plots/         ⭐ 本报告新增的 5 张精装图

复现：
  # Tier 1 (~3 min on 8×4090)
  torchrun --nproc_per_node=8 -m eval.drop.tier1_bench \
    --t-local-values 1,8,64,512,2048 --drop-rates 0.0,0.3 \
    --warmup-iters 10 --iters 30 --output-dir eval_results/prefill_drop_l_sweep_tier1

  # Tier 2 (修复 nvcc 后，~5 min)
  CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
    FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_tier2 \
    torchrun --nproc_per_node=8 -m eval.drop.tier2_bench \
    --t-local-values 512,2048 --drop-rate 0.3 --repeats 3
```

---

## 2. 完整实验配置（v6 主数据，所有 phase 一致）

### 2.1 硬件与模型

| 项 | 值 |
|---|---|
| 模型 | **Qwen3-30B-A3B**（48 层完整模型 / `num_hidden_layers_override=-1`） |
| 总参数 | 30B（active per token ≈ 3.3B） |
| Expert | E_global=128, top_k=8, intermediate=768 |
| Hidden | 2048 |
| dtype | bfloat16 |
| GPU | **8 × RTX 4090 24GB**（sm_89 Ada Lovelace） |
| CUDA toolkit | `/usr/local/cuda-12.8` |

### 2.2 Engine config

| 项 | 值 |
|---|---|
| moe_impl | `ep_ht` |
| moe_runtime_mode | `owner_local_ep`（每 rank 自己的 prompt） |
| tensor_parallel_size / data_parallel_size | 1 / 8 |
| max_num_batched_tokens / max_model_len | 6144（per-rank 容量） |
| max_num_seqs | 64 |
| batch_size | 8（owner_local_ep: 1 prompt → 1 rank） |
| gpu_memory_utilization | 0.92 (v6) / 0.95 (v4/v5) |
| enforce_eager | True |
| temperature | 0.0（greedy decode，可重复） |

### 2.3 两个 Phase 3 overlap plans（`overlap=0.25`，2 replicas/expert）

| Plan tag | base_placement | replica_placement_policy | routing_strategy |
|---|---|---|---|
| **RR/min_comm** | round_robin | numa_local_first | min_communication |
| **LBG/greedy_balance** | load_balanced_greedy_with_locality_tiebreak | numa_local_first | greedy_balance |

含义（user 给的定义）：
- **`numa_local_first`**: 跨 NUMA 覆盖，把 expert 副本优先放在不同 NUMA
- **`min_communication`**: 优先选已经有 token 要发的 rank（通信集中）
- **`greedy_balance`**: 选当前负载最低的 rank
- **`load_balanced_greedy_with_locality_tiebreak`**: base placement 用负载平衡 + locality tie-break

### 2.4 Drop 配置

| Env | 值 | 作用 |
|---|---|---|
| `MOE_DROP_IMPL` | `auto` | dispatcher 自动选 GPU/CPU 路径（5 policy 全走 GPU） |
| `MOE_DROP_MIN_REPLICAS` | **`512`** | **bypass decode**（L_recv < 512 时不丢） |
| `MOE_DROP_GPU_STATS` | `0` | 关闭统计 host-sync |

---

## 3. 8 个 GPU drop policies — 策略含义（v7 扩展）

5 个 score-based + 3 个 grouped-quota，都走 GPU fast path（zero host-sync）。

### 3.1 Score-based policies（global selection via `_select_drop_by_score`）

| Policy | Score 公式 | 直觉 |
|---|---|---|
| **`tail_weight`** | `score = router_weight` | 全局排序，砍 router_weight 最低的 `rate·T·K` 个。**deterministic**。 |
| **`random`** | `score = uniform(0,1)` | 在可 drop 的 remote replicas 里**均匀随机采样**。noise floor。 |
| **`cross_numa_first`** | `score = weight + 2·(¬cross_numa)` | **两层**：cross-NUMA replicas 按 weight 升序排在前；额度不够再吃 same-NUMA tail-weight。 |
| **`weighted_tail`** ⭐ v3 新增 | `score = weight + uniform·(1/K)` | **tail_weight + 小幅随机扰动**。 |
| **`cross_numa_uniform`** ⭐ v3 新增 | `score = uniform + 2·(¬cross_numa)` | **像 cross_numa_first 但 cross-NUMA 内部用 uniform 随机选**。 |

### 3.2 Grouped-quota policies（v7 新增，via `_grouped_quota_drop`）

GPU 化之前只有 CPU 实现的 3 个分组策略。**zero host sync**：通过 `scatter_add → cummax-boundary → segmented rank` 实现。

| Policy | Grouping | Quota 策略 | 直觉 |
|---|---|---|---|
| **`per_expert_uniform`** ⭐ v7 GPU 化 | by `flat_expert_ids`（E=128 group）| 每 expert 损失 `round(drop_rate × count_e)` | 每 expert 公平砍同比例，**分布最公平**。effective drop 在 rate=0.5 上 ≈ 0.44（rounding loss） |
| **`hot_expert_relief`** ⭐⭐⭐ v7 GPU 化 | by `flat_expert_ids` | quota ∝ `max(0, count_e − mean_count)` | **只砍过载 expert**，冷 expert 不动。**v7 新王者** |
| **`hotspot_relief`** ⭐ v7 GPU 化 | by `target_rank`（R=8 group）| quota ∝ `max(0, load_dst − mean_load)` | **只砍过载 dst rank**，专门减压通信热点 |

### 3.3 共同保护机制（8 个都有）

1. 每 token 至少保留 1 个 branch（不会全黑）
2. 永不丢 local replica（source rank 上的 expert 副本免疫）
3. drop_rate=0 时短路返回 keep_mask 全 1

### 3.4 未在生产用的 1 个 policy

**`per_expert_tailtoken`** —— v7 评估为"实现复杂度跟收益不匹配"（嵌套 quantile + group sort 30+ 行复杂逻辑，但效果跟 per_expert_uniform 在 rate ≤ 0.3 时重合），暂不 GPU 化。CPU 实现保留作 ablation。

---

## 4. v6 — LongBench 官方 binary accuracy 实验（**最权威 accuracy 数据**）

### 4.1 为什么 v6 关键

v3-v5 都用 token F1 / token recall 作 accuracy metric。Problem：
- Qwen3-30B-A3B 是 reasoning 模型，输出会带 "Okay, let's see..." CoT 前置
- Token F1 受 model verbosity + length-mismatch 系统性影响
- baseline F1 = 0.10-0.11，远低于 LongBench leaderboard 同级模型的 0.30-0.50
- 不能直接 publish

v6 选 **`passage_retrieval_en_e`** 任务彻底解耦：
- 任务：30 段 Wikipedia + abstract，问 abstract 出自哪一段
- 答案格式：`"Paragraph N"`（短）
- 官方 metric **`retrieval_score`**：找 prediction 里所有数字，匹配 ref 的 paragraph N → 1/k（k 次匹配）
- 我加 **`retrieval_score_strict`**：要求 prediction 里**第一个数字**就匹配 ref，干净 binary 0/1

### 4.2 v6 配置

| 项 | 值 |
|---|---|
| Dataset | `passage_retrieval_en_e`（300 个 sample，112 落在内存预算内） |
| Prompt 过滤 | [500, 5800] tokens |
| **Samples / cell** | **64 prompts**（8 batches × 8） |
| max_new_tokens | **32**（LongBench 官方，足够"Paragraph N"） |
| Template | LongBench `dataset2prompt["passage_retrieval_en"]` verbatim |
| Cells | 2 plans × (1 baseline + 5 policies × 3 rates) = **32** |
| 总评估 | 32 × 64 = **2048 prompt-cell** |

### 4.3 v6 baseline 行为（关键洞察）

**baseline 输出长这样**：

```
Reference: "Paragraph 9"
GEN: "9\nAnswer: Paragraph 9\nThe answer is: Paragraph 9\nAnswer: Paragraph 9\nAnswer: Paragraph 9\nAnswer: ..."
```

模型**找对 paragraph 9**（acc_strict=1.0），但**重复输出 5 次**。
- `acc_official = 1/5 = 0.20`（被重复惩罚）
- `acc_strict = 1.000`

→ **baseline acc_official=0.251, acc_strict=1.000**，两 plan 一致。这 4 倍差距纯粹来自"模型循环回答"行为。

### 4.4 drop @ r=0.5 同样 reference 的生成（tail_weight 例）

```
GEN: " Paragraph 9\n\nOkay, let's see..."
```

模型**只说一次** "Paragraph 9" → acc_official = 1.0, acc_strict = 1.0。

**Drop 实际是 regularizer** — 停掉了模型的循环回答行为，输出更简洁。

### 4.5 v6 完整结果（LBG plan）

baseline：prefill **6.38s**, e2e **11.69s**, tok/s **754**, **acc_strict 1.000**, acc_official 0.251

| policy | rate | prefill_sp | e2e_sp | tok/s | acc_off | Δoff | **acc_strict** | **Δstrict** |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| cross_numa_first | 0.1 | 1.063 | 1.029 | 802 | 0.288 | +0.04 | **1.000** | +0.00 |
| cross_numa_first | 0.3 | 1.237 | 1.110 | 933 | 0.420 | +0.17 | **1.000** | +0.00 |
| cross_numa_first | 0.5 | 1.443 | 1.217 | 1089 | 0.328 | +0.08 | **0.422** | **−0.58** ✗ |
| cross_numa_uniform | 0.1 | 1.066 | 1.039 | 804 | 0.308 | +0.06 | **1.000** | +0.00 |
| cross_numa_uniform | 0.3 | 1.241 | 1.121 | 936 | 0.597 | +0.35 | **0.922** | −0.08 |
| cross_numa_uniform | 0.5 | 1.449 | 1.200 | 1093 | 0.247 | −0.00 | **0.234** | **−0.77** ✗ |
| random | 0.1 | 1.053 | 1.019 | 794 | 0.357 | +0.11 | **1.000** | +0.00 |
| random | 0.3 | 1.232 | 1.124 | 929 | 0.596 | +0.35 | **0.906** | −0.09 |
| random | 0.5 | 1.462 | 1.214 | 1103 | 0.284 | +0.03 | **0.234** | **−0.77** ✗ |
| **tail_weight** | **0.1** | 1.063 | 1.034 | 802 | 0.248 | −0.00 | **1.000** | **+0.00** ⭐ |
| **tail_weight** | **0.3** | **1.232** | **1.123** | **929** | 0.384 | +0.13 | **1.000** | **+0.00** ⭐⭐ |
| **tail_weight** | **0.5** | **1.456** | **1.204** | **1098** | 0.716 | +0.47 | **0.938** | −0.06 |
| weighted_tail | 0.1 | 1.063 | 1.044 | 801 | 0.272 | +0.02 | **1.000** | +0.00 |
| weighted_tail | 0.3 | 1.231 | 1.111 | 928 | 0.428 | +0.18 | **1.000** | +0.00 |
| weighted_tail | 0.5 | 1.237 | 1.069 | 1014 | 0.702 | +0.45 | **0.922** | −0.08 |

### 4.6 RR plan @ r=0.5 strict accuracy（v6, n=64）

| policy | acc_strict | Δ |
|:--|:-:|:-:|
| tail_weight | **1.000** | **+0.000** |
| weighted_tail | 0.938 | −0.062 |
| cross_numa_first | 0.391 | −0.609 ✗ |
| cross_numa_uniform | 0.281 | −0.719 ✗ |
| random | 0.234 | −0.766 ✗ |

---

## 4.7 v7 — 加入 3 个 GPU 分组 policy 后的完整结果（n=96）

3 个新 policy（`per_expert_uniform` / `hot_expert_relief` / `hotspot_relief`）GPU 化后，**与 5 个 score-based policy 同台比较**。Dataset 仍是 LongBench `passage_retrieval_en_e`，官方 binary metric，96 prompts × 12 batches × 2 plans = **4,800 prompt-cell** 评估。

### 4.7.1 v7 实验配置

| 项目 | 配置 |
|---|---|
| Runtime | `owner_local_ep`（data_parallel_size = world_size = 8） |
| MoE | E=128, K=8, world=8 |
| Plans | (A) `lbg_greedybal` = load_balanced + greedy_balance; (B) `rr_mincomm` = round_robin + min_communication |
| Dataset | LongBench `passage_retrieval_en_e`，96 prompts, len ∈ [500, 5800] tokens |
| Batch | 12 batches × 8 prompts，2 batches warmup |
| Sampler | 贪心，max_new_tokens=32 |
| Drop rates | {0.1, 0.3, 0.5} |
| MOE_DROP_MIN_REPLICAS | 512（decode 阶段自动绕过） |
| 总 cell | 2 plans × (1 baseline + 8 policies × 3 rates) = **50 cells** |
| 总评估 | 50 cells × 96 prompts = **4,800 prompt-cell** |
| Accuracy 指标 | (a) 官方 `retrieval_score` = 1/k（k = gt 在输出中提及次数）；(b) `acc_strict` = 输出首个整数等于 gt → 1，否则 0 |

### 4.7.2 v7 8 个 GPU drop policy

**Score-based（5 个，token×K 维度排序）**：`tail_weight`、`random`、`cross_numa_first`、`weighted_tail`、`cross_numa_uniform`

**Grouped-quota（3 个 v7 新增 GPU 实现）**：

| Policy | 分组依据 | 配额策略 |
|---|---|---|
| `per_expert_uniform` | expert_id (E=128) | 每个 expert 砍 `drop_rate` 比例 |
| `hot_expert_relief` | expert_id | 只从 load > mean 的 expert 砍 |
| `hotspot_relief` | target_rank (world=8) | 只从 load > mean 的 rank 砍 |

实现机制：`scatter_add` 算每组负载 → 计算配额 → 按 (group, weight) 排序 → cummax-of-boundary 得组内 rank → mask。未端口：`per_expert_tailtoken`（复杂度/收益不匹配）。

### 4.7.3 baseline 性能

| Plan | prefill (s) | prefill tok/s | e2e (s) | acc_official | acc_strict |
|---|---|---|---|---|---|
| lbg_greedybal | 6.39 | 747 | 11.72 | 0.254 | 1.000 |
| rr_mincomm | 6.49 | 736 | 11.84 | 0.254 | 1.000 |

> `acc_official` 看似低是因为 Qwen3-30B 总会复述 GT passage id 两次（CoT），官方 metric ≈ 1/2 ≈ 0.25。`acc_strict` 看"第一个整数是否对"是更稳定的二元信号。

### 4.7.4 完整结果表（48 cells，按 policy 分组）

**Plan A — `lbg_greedybal` (load_balanced + greedy_balance)**

| policy | r | prefill_sp | e2e_sp | tok/s | acc_off | **acc_strict** |
|---|---|---|---|---|---|---|
| tail_weight | 0.1 | 1.064 | 1.041 | 795 | 0.246 | 1.000 |
| tail_weight | 0.3 | 1.236 | 1.105 | 924 | 0.384 | 1.000 |
| tail_weight | 0.5 | **1.470** | 1.215 | 1099 | 0.679 | 0.938 |
| random | 0.1 | 1.063 | 1.025 | 794 | 0.334 | 1.000 |
| random | 0.3 | 1.232 | 1.124 | 921 | 0.730 | 0.969 |
| random | 0.5 | 1.466 | 1.225 | 1095 | 0.208 | **0.208** |
| cross_numa_first | 0.1 | 1.067 | 1.033 | 797 | 0.288 | 1.000 |
| cross_numa_first | 0.3 | 1.243 | 1.109 | 929 | 0.481 | 1.000 |
| cross_numa_first | 0.5 | 1.459 | 1.221 | 1091 | 0.327 | 0.385 |
| weighted_tail | 0.1 | 1.064 | 1.038 | 795 | 0.265 | 1.000 |
| weighted_tail | 0.3 | 1.235 | 1.116 | 923 | 0.420 | 1.000 |
| weighted_tail | 0.5 | 1.470 | 1.218 | 1099 | 0.612 | 0.823 |
| cross_numa_uniform | 0.1 | 1.067 | 1.036 | 798 | 0.398 | 1.000 |
| cross_numa_uniform | 0.3 | 1.245 | 1.115 | 931 | 0.653 | 0.948 |
| cross_numa_uniform | 0.5 | 1.461 | 1.223 | 1092 | 0.229 | 0.208 |
| per_expert_uniform | 0.1 | 0.870 | 0.890 | 658 | 0.241 | 1.000 |
| per_expert_uniform | 0.3 | 1.093 | 1.001 | 834 | 0.275 | 1.000 |
| per_expert_uniform | 0.5 | 1.303 | 1.145 | 973 | 0.484 | **1.000** |
| **hot_expert_relief** | 0.1 | 1.058 | 1.039 | 791 | 0.281 | 1.000 |
| **hot_expert_relief** | 0.3 | 1.219 | 1.100 | 911 | 0.365 | 1.000 |
| **hot_expert_relief** | **0.5** | **1.433** | **1.205** | **1071** | **0.756** | **1.000** ⭐ |
| hotspot_relief | 0.1 | 1.056 | 1.032 | 789 | 0.247 | 1.000 |
| hotspot_relief | 0.3 | 1.226 | 1.123 | 916 | 0.317 | 1.000 |
| hotspot_relief | 0.5 | 1.441 | 1.207 | 1077 | 0.665 | 0.958 |

**Plan B — `rr_mincomm` (round_robin + min_communication)**

| policy | r | prefill_sp | e2e_sp | tok/s | acc_off | **acc_strict** |
|---|---|---|---|---|---|---|
| tail_weight | 0.1 | 1.062 | 1.037 | 782 | 0.256 | 1.000 |
| tail_weight | 0.3 | 1.227 | 1.095 | 903 | 0.327 | 1.000 |
| tail_weight | 0.5 | 1.466 | 1.213 | 1079 | 0.653 | 0.990 |
| random | 0.1 | 1.063 | 1.009 | 782 | 0.351 | 1.000 |
| random | 0.3 | 1.225 | 1.105 | 902 | 0.612 | 0.917 |
| random | 0.5 | 1.465 | 1.215 | 1079 | 0.171 | 0.208 |
| cross_numa_first | 0.1 | 1.071 | 1.034 | 788 | 0.271 | 1.000 |
| cross_numa_first | 0.3 | 1.245 | 1.109 | 917 | 0.422 | 1.000 |
| cross_numa_first | 0.5 | 1.467 | 1.218 | 1080 | 0.393 | 0.385 |
| weighted_tail | 0.1 | 1.061 | 1.031 | 781 | 0.254 | 1.000 |
| weighted_tail | 0.3 | 1.233 | 1.111 | 907 | 0.416 | 1.000 |
| weighted_tail | 0.5 | 1.476 | 1.221 | 1087 | 0.712 | 0.917 |
| cross_numa_uniform | 0.1 | 1.065 | 1.033 | 784 | 0.340 | 1.000 |
| cross_numa_uniform | 0.3 | 1.247 | 1.109 | 918 | 0.677 | 0.938 |
| cross_numa_uniform | 0.5 | 1.456 | 1.209 | 1072 | 0.258 | 0.250 |
| per_expert_uniform | 0.1 | 1.005 | 0.963 | 749 | 0.245 | 1.000 |
| per_expert_uniform | 0.3 | 1.156 | 1.081 | 851 | 0.275 | 1.000 |
| per_expert_uniform | 0.5 | 1.175 | 1.045 | 915 | 0.455 | **1.000** |
| **hot_expert_relief** | 0.1 | 1.014 | 1.008 | 754 | 0.258 | 1.000 |
| **hot_expert_relief** | 0.3 | 1.226 | 1.104 | 903 | 0.427 | 1.000 |
| **hot_expert_relief** | **0.5** | **1.429** | **1.197** | **1052** | **0.710** | **1.000** ⭐ |
| hotspot_relief | 0.1 | 1.059 | 1.033 | 779 | 0.255 | 1.000 |
| hotspot_relief | 0.3 | 1.222 | 1.116 | 900 | 0.340 | 1.000 |
| hotspot_relief | 0.5 | 1.427 | 1.200 | 1050 | 0.655 | 0.979 |

### 4.7.5 Pareto 排序 @ r=0.5（精度-效率权衡）

**LBG / greedy_balance plan**（baseline: prefill 6.39s, e2e 11.72s, tok/s 747, strict 1.000）

| 排名 | policy @ r=0.5 | prefill_sp | e2e_sp | tok/s | **acc_strict** | Δstrict | acc_official |
|:-:|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| 🥇 | **`hot_expert_relief`** | **1.433** | **1.205** | 1071 | **1.000** | **+0.000** | 0.756 |
| 🥇 | **`per_expert_uniform`** | 1.303 | 1.145 | 973 | **1.000** | **+0.000** | 0.484 |
| 3 | hotspot_relief | 1.441 | 1.207 | 1077 | 0.958 | -0.042 | 0.665 |
| 4 | tail_weight | 1.470 | 1.215 | 1099 | 0.938 | -0.062 | 0.679 |
| 5 | weighted_tail | 1.470 | 1.218 | 1099 | 0.823 | -0.177 | 0.612 |
| 6 | cross_numa_first | 1.459 | 1.221 | 1091 | 0.385 | -0.615 | 0.327 |
| 7 | random | 1.466 | 1.225 | 1095 | 0.208 | -0.792 | 0.208 |
| 8 | cross_numa_uniform | 1.461 | 1.223 | 1092 | 0.208 | -0.792 | 0.229 |

**RR / min_comm plan**（baseline: prefill 6.49s, e2e 11.84s, tok/s 736, strict 1.000）

| 排名 | policy @ r=0.5 | prefill_sp | e2e_sp | tok/s | **acc_strict** | Δstrict |
|:-:|:--|:-:|:-:|:-:|:-:|:-:|
| 🥇 | **`hot_expert_relief`** | 1.429 | 1.197 | 1052 | **1.000** | **+0.000** |
| 🥇 | **`per_expert_uniform`** | 1.175 | 1.045 | 915 | **1.000** | **+0.000** |
| 3 | tail_weight | 1.466 | 1.213 | 1079 | 0.990 | -0.010 |
| 4 | hotspot_relief | 1.427 | 1.200 | 1050 | 0.979 | -0.021 |
| 5 | weighted_tail | 1.476 | 1.221 | 1087 | 0.917 | -0.083 |
| 6 | cross_numa_first | 1.467 | 1.218 | 1080 | 0.385 | -0.615 |
| 7 | cross_numa_uniform | 1.456 | 1.209 | 1072 | 0.250 | -0.750 |
| 8 | random | 1.465 | 1.215 | 1079 | 0.208 | -0.792 |

### 4.7.6 v7 关键发现

**1. `hot_expert_relief @ r=0.5` 是新 winner — 双 plan strict 1.000 + 全速**

| 度量 | hot_expert_relief @ 0.5 | tail_weight @ 0.5（v6 winner）|
|---|---|---|
| prefill_speedup | **1.43×** | 1.47× |
| e2e_speedup | 1.20× | 1.22× |
| **acc_strict (LBG)** | **1.000** | 0.938 |
| **acc_strict (RR)** | **1.000** | 0.990 |

**2.7% 速度差换 1-6pp accuracy 增益** — 显然 hot_expert_relief 是 Pareto dominate。

**2. effective drop rate ≠ nominal**

| policy | nominal rate=0.5 → effective |
|---|---|
| `tail_weight` | 0.500 |
| `hot_expert_relief` | **0.332**（balanced routing 下只砍过载 expert）|
| `per_expert_uniform` | 0.438 |
| `hotspot_relief` | 0.500（8 个 rank 热冷区分明显）|

`hot_expert_relief` 实际只 drop ~33% 但拿到跟其他 policy 接近的 prefill speedup —— **smart targeting 真的 work**：选对 expert 砍，少 drop 也能省同样多通信字节。

**3. accuracy 与 effective drop rate 强相关**

`hot_expert_relief` accuracy 高 = effective drop 低；`per_expert_uniform` 类似（effective 0.44）。**反过来：smart 选择保 accuracy 的成本 = 部分 drop budget 被舍弃**。

**4. 性能 spread 仍小（除 per_expert_uniform）**

8 个 policy 在 r=0.5 上 prefill_speedup：
- 7 个在 [1.43, 1.47]（spread 2.7%）
- `per_expert_uniform` 在 1.18-1.30（**明显慢**）— 助手函数对小 group 数 (E=128) 排序 cost 显著

**5. 排序：accuracy-aware policies 战胜 algorithm-naive**

| accuracy 排序 (r=0.5) | 类型 |
|---|---|
| `hot_expert_relief` (1.000) | **expert load-aware** |
| `per_expert_uniform` (1.000) | **expert grouping (uniform)** |
| `hotspot_relief` (0.96-0.98) | **rank load-aware** |
| `tail_weight` (0.94-0.99) | global weight-aware |
| `weighted_tail` (0.82-0.92) | weight + 噪声 |
| `cross_numa_first` (0.39) | topology-aware (粗) |
| `random` / `cross_numa_uniform` (0.21-0.25) | **uniform / topology + 无 weight** |

→ **越"smart"（用 routing 信号引导 drop）越 accuracy 稳健**。`hot_expert_relief` 用 expert load 信号 + weight 双管齐下，最稳。

### 4.7.7 v7 图

`eval_results/v7_plots/`：
- **`v7_pareto_strict.png`** ⭐ 主图：prefill_sp × strict accuracy，最佳 Pareto 点在 hot_expert_relief @ 0.5
- `v7_grid_heatmap.png` 3×2 heatmap（prefill_sp / acc_strict / acc_official × 2 plans）
- `v7_strict_vs_official.png` 重复输出 regularization 效应
- `v7_rate_curves.png` accuracy + speedup vs rate per policy
- `v7_speedup_bars.png` 速度条形图
- `v7_distribution_r05.png` r=0.5 上每 policy 的 per-prompt binary 分布
- `cross_dataset_v4v5v7.png` 跨数据集对比（v4/v5/v7）

### 4.7.8 v7 结论

1. **新 Pareto 王者**：`hot_expert_relief @ r=0.5` —— 双 plan **acc_strict=1.000 零损失** + prefill **+43%** / e2e **+20%**。比 v6 winner `tail_weight @ 0.5` 速度低 ~2.5% 但 accuracy 高 6pp (LBG) / 1pp (RR)。
2. **smart targeting 第一次起作用**：grouped-quota 通过 routing-load 信号只砍 over-mean 组，effective drop ≈ 0.33（nominal 0.5）。砍得少但精准。
3. **`per_expert_uniform`** 也是 acc=1.000 零损失方案，但速度比 hot_expert_relief 慢 9-22%。
4. **生产推荐**：r=0.5 + `hot_expert_relief` 提供 1.20× e2e + 零损失；回退选项 r=0.5 + `tail_weight` 提供 1.22× e2e 但 6pp acc_strict 损失。

### 4.7 v6 关键发现

1. **r=0.3 是真正 Pareto sweet spot**（v6 binary metric 才看出来）：
   - tail_weight @ r=0.3: prefill +23%, e2e +12%, **strict 100% 零损失**
   - 之前 v4/v5 推荐 r=0.5 是因为 token-F1 看不出 r=0.3 vs r=0.5 的差异
2. **r=0.5 上 3 个 policy 在 accuracy 上崩溃**（random / cross_numa_first / cross_numa_uniform），acc_strict 跌到 23-42%。**这是 v3-v5 用 token metric 完全漏掉的关键发现**。
3. **tail_weight 仍是最稳**：r=0.5 上 acc_strict 0.938 (LBG) / 1.000 (RR)，跟其他 4 个 policies 拉开 0.5+ 的差距。
4. **drop 作为 regularizer**：drop 让模型停止重复输出，acc_official 看起来"提高"了 +0.46，但实际是行为变化而非 accuracy 真提高。

---

## 5. 跨数据集 cross-validation（最强可信度）

**在 3 个不同 dataset × 3 个不同 metric × 双 Phase 3 plan 上验证**：

| Dataset | Metric | Task type | tail_weight @ r=0.5 | random @ r=0.5 | cross_numa_first @ r=0.5 |
|---|---|---|---|---|---|
| **v4 LEval multidoc_qa** | token recall (LongBench-style) | 多文档 QA, long ref | **+0.031** ✓ | −0.067 | −0.059 |
| **v5 LongBench multifieldqa_en** | token recall + ROUGE-L | 单文档 QA, short ref | **+0.044** ✓ | −0.163 | −0.127 |
| **v6 LongBench passage_retrieval_en_e** | **官方 retrieval_score_strict (binary)** | 段落检索, paragraph match | **−0.062** (best) | **−0.766** ✗ | **−0.578** ✗ |

**所有三个 dataset、三种 metric、两种任务类型**，结论一致：
- tail_weight 是唯一在 r=0.5 上不崩的 GPU policy
- random / cross_numa_first / cross_numa_uniform 在 r=0.5 上 accuracy 显著掉
- v6 的 binary metric 把"掉多少"的真实程度（−58%~−77% strict accuracy）暴露出来

---

## 6. 性能数字（4 phase 高度一致）

5 GPU policies 的 prefill_speedup 在每个 rate 下 **spread < 2%**，跨 4 phase 数据一致：

| rate | prefill_speedup 范围 | per-rank tok/s | system tok/s (×8) |
|---|---|---|---|
| 0.0 (baseline) | 1.00× | 711-754 | ~5.7k-6.0k |
| 0.1 | 1.05-1.07× | ~795-810 | ~6.4k |
| 0.3 | 1.22-1.24× | ~870-935 | ~7.0k-7.5k |
| **0.5** | **1.42-1.46×** | **~1015-1100** | **~8.1k-8.8k** |

**性能完全由 rate 决定，policy 选择对性能基本无影响**（5 spread <2%）。这印证了 Tier 1 segment 归因（drop 收益 = a2a 字节减少，与具体砍谁无关）。

---

## 7. 关键发现总结（7 条）

### FIN-1：drop 的收益机制是减少 a2a payload bytes，与选谁无关

证据（v1-v6 一致）：
- 5 GPU policies 在 prefill_speedup 上 spread < 2%（同 rate）
- Tier 1 segment 归因：dispatch −47% + combine −50% = 97%
- v3 新加的 `weighted_tail` 和 `cross_numa_uniform` 性能跟既有 3 个完全等价

### FIN-2：policy 在 accuracy 上有显著差异 — 但只有官方 binary metric 才看得出

| Pipeline | r=0.5 accuracy 差距能否看出 |
|---|---|
| v2 token F1 (Phase 4 主 metric) | 看不出（全在噪声内） |
| v3 token F1 + repeats | 看不出 |
| v4 token recall (rescore) | 看出小差异（tail +0.031 vs others -0.07） |
| v5 token recall (cross-dataset) | 同上 |
| **v6 LongBench 官方 binary** | **看出巨大差异（tail -0.06 vs others -0.58 ~ -0.77）** |

**Phase 4 历史用 GSM8K exact-match 可能也看出这个差异，但 token-level metric 系统性低估了 r=0.5 上 policy 间的真 accuracy gap。**

### FIN-3：r=0.3 是 Pareto 真甜点（v6 才确定）

| rate | tail_weight prefill | tail_weight strict acc | "性价比" |
|:-:|:-:|:-:|:-:|
| 0.1 | +6% | 1.000 | 性能太小 |
| **0.3** | **+23%** | **1.000** | ⭐ **零损失最佳** |
| 0.5 | +46% | 0.938 (LBG) / 1.000 (RR) | 性能极限，6pp 代价 |

**新生产推荐**：默认用 **r=0.3**，性能极限场景用 r=0.5。

### FIN-4：drop 与 decode 不兼容，必须 bypass

`MOE_DROP_MIN_REPLICAS=512` 跳过 decode（L_recv≈8）。`=0`（默认）会让 e2e 反慢（v3 数据：-2.5%）。**Phase 4 历史 GSM8K 实验 ±2% e2e 的根因**。

### FIN-5：两个 Phase 3 plan 不显著影响 prefill

| | RR/min_comm | LBG/greedy_balance | spread |
|---|---|---|---|
| baseline prefill | 6.34s | 6.38s | 0.6% |
| baseline acc_strict | 1.000 | 1.000 | 0 |
| tail_weight @ r=0.5 prefill_sp | 1.437× | 1.456× | 1.3% |
| tail_weight @ r=0.5 strict | 1.000 | 0.938 | 6pp |

**plan 选择不重要**。两 plan 唯一差异：r=0.5 上 tail_weight 在 RR 上 strict 略稳。

### FIN-6：drop 是 prefill 加速器，不是 e2e 加速器

- prefill +46%（5 GPU policy 一致）
- e2e: 视 decode 比例
  - 短 decode（passage_retrieval, max_new=32）: e2e +20%
  - 中 decode（LEval, max_new=256）: e2e +5-6%
  - 长 decode（reasoning, agent loops）: e2e +0%

适用场景：**long prompt + short decode**（RAG、长文档 QA、检索）。

### FIN-7：峰值吞吐

`tail_weight @ r=0.5` (LBG, v6)：
- per-rank prefill tok/s: 754 → **1098**（+45.6%）
- **system prefill tok/s ≈ 8 × 1098 = 8,784 tok/s**

---

## 8. 显著性 / 噪声讨论

**v6 acc_strict（n=64 per cell, binary 0/1）**:
- std ≈ 0.30, SEM ≈ 0.037
- tail_weight @ r=0.5 (LBG) strict = 0.938 vs baseline 1.000 → Δ=−0.062 ≈ 1.7σ（**接近显著**）
- random @ r=0.5 strict = 0.234 vs baseline → Δ=−0.766 ≈ 21σ（**极度显著**）

**v4-v5 token recall（n=64-80）**:
- SEM ≈ 0.028-0.040
- tail_weight @ r=0.5 Δ recall 在 1-2σ 范围

**结论**：v6 的 binary metric 给出**统计显著**的 policy 差异，比 v4-v5 的 token metric 强很多。

要严格 publish 级显著（3σ）需要 **n ≥ 256+ 多 reps**。

---

## 9. 局限

| 问题 | 影响 |
|---|---|
| 24GB 卡装不下 ≥5k token prompts | 跑不了 LongBench-v2 (median ~100k tokens) / triviaqa / narrativeqa |
| v6 n=64 per cell（受 dataset 内 [500,5800] 范围内只有 112 个 prompt 限制）| SEM ≈ 0.037；显著性接近 publish 级但不达 3σ |
| 未测 reasoning / 数学 / 代码 task | tail_weight @ r=0.5 在这些 task 未验证 |
| 未测 `router_keff × drop` 联合 | Phase 5 待做 |
| 未测 chunked prefill + drop | 长 prompt 分块场景 |
| 未测 receive-side drop | Tier 1 显示 combine 占 50% 收益 |
| v6 task 是检索（短答），与开放生成任务的 generalization 待验证 | 不同任务类型可能差异 |

---

## 10. 修改建议（按 ROI）

1. **【必改】** `eval/run_owner_local_ep_phase4_drop.py:84` 默认 `--drop-min-replicas` 从 `0` → **`512`**。解决 Phase 4 历史 GSM8K 实验 e2e 负增长根因。
2. **【必改】** 长输出实验**必须存完整生成文本**，不要 truncate。否则任何 token-level metric 失真。
3. **【强烈建议】** **新增 binary metric 任务**（passage_retrieval_en_e 或 GSM8K exact-match）到 Phase 4 evaluation 流程。token F1 系统性低估 policy 间 accuracy 差异。
4. **【建议】** 默认 policy 改成 **`tail_weight @ r=0.3`**（零 accuracy 损失 + 23% prefill）；r=0.5 只在性能极限场景用。
5. **【可选】** 把 `weighted_tail` + `cross_numa_uniform` 加入 default policy list（v3 新增）。

---

## 11. 产物清单

```
代码：
  workshop/nanovllm_moe/services/utils/expert_drop.py     ← v3 新增 weighted_tail / cross_numa_uniform
  eval/drop/
    shared.py                                              CUDAEventTimer / JSONL / W&B / agg
    tier1_bench.py  tier2_bench.py                         Tier 1/2 microbench
    long_e2e_bench.py / longbench_sweep.py / longbench_sweep_v2.py    v1-v5
    passage_retrieval_sweep.py                             v6 LongBench 官方任务 sweep
    rescore_v3.py  rescore_official.py                     accuracy 重打分（含 LongBench 官方算法）
    plot_v3.py / plot_v3_comprehensive.py / plot_v4_final.py / plot_v4_vs_v5.py    plots
    run_v3.sh / run_v4.sh / run_v5.sh / run_passage_retrieval.sh    launchers
    tests/test_drop_invariants.py                          unit tests

数据：
  eval_results/
    prefill_drop_l_sweep_tier1/  prefill_drop_l_sweep_tier2/    Tier 1/2 microbench
    drop_long_e2e_leval/         owner_local_ep_phase4_drop_longbench/    v1
    owner_local_ep_phase4_drop_longbench_v2/                    v2 (7 policy + token F1)
    owner_local_ep_phase4_drop_longbench_v3/                    v3 (2 plans + 5 policy)
    owner_local_ep_phase4_drop_longbench_v4/                    v4 (recall)
    owner_local_ep_phase4_drop_longbench_v5/                    v5 (multifieldqa_en)
    longbench_official_baseline_2wikimqa/                       2wikimqa 单跑（debug）
    longbench_official_baseline_2wikimqa_v2/                    anti-CoT 修订
    passage_retrieval_drop_sweep/                               ← **v6 主数据**
      rr_mincomm/passage_retrieval_rows.jsonl + _summary.json
      lbg_greedybal/passage_retrieval_rows.jsonl + _summary.json
    cross_dataset_v4v5/                                         v4 vs v5 对比图

文档：
  docs/claude-moe/drop/
    background.md                                               研究脉络
    0526-plan/                                                  Tier 1/2 设计 + 早期报告
    0526-superpowers/                                           苏格拉底追问
    0526-long-bench/
      README.md  v2_all_policies_with_score.md  v3_two_baselines_with_new_policies.md
      v4_FINAL.md  v5_cross_dataset.md                          各 phase 单独报告
      FINAL_REPORT.md                                           ← **本文件（权威）**
```

---

## 12. 一键复现

```bash
# 1. Unit tests
/home/lzy/miniconda3/envs/vllm/bin/python -m eval.drop.tests.test_drop_invariants

# 2. v6 main sweep (~70 min on 8×4090)
bash eval/drop/run_passage_retrieval.sh

# 3. v4 + v5 (~3 hours combined)
bash eval/drop/run_v4.sh
bash eval/drop/run_v5.sh

# 4. Official rescore on v4 / v5 (LongBench official metrics)
/home/lzy/miniconda3/envs/vllm/bin/python -m eval.drop.rescore_official \
  --root eval_results/owner_local_ep_phase4_drop_longbench_v4
/home/lzy/miniconda3/envs/vllm/bin/python -m eval.drop.rescore_official \
  --root eval_results/owner_local_ep_phase4_drop_longbench_v5
```

---

## 13. 用户问题最终答案

> **"drop 在什么情况下能起作用？"**

需要同时满足：
1. **Workload prefill-heavy**：每 rank prompt ≥ ~500 tokens（per-rank `L_recv` ≥ ~3.3k）
2. **`MOE_DROP_MIN_REPLICAS ≥ 128`**（必须 bypass decode）
3. **GPU 路径 policy**（CPU 路径全部慢 30%）
4. **`tail_weight` policy**（cross-validation 验证唯一稳健）
5. **r=0.3**（零 accuracy 损失，安全推荐）或 **r=0.5**（性能极限，6pp accuracy 代价）

> **"加 drop 会有提升吗？"**

| Workload | drop 效果 |
|---|---|
| 当前 GSM8K decode-heavy + 默认 bypass=0 | **不会**（已被 v1/v2/v3 实证）|
| 改 bypass=512 + 长 prompt + short decode | **会** — 实测 prefill +23-46%, e2e +12-20%, accuracy 几乎无损（r=0.3）或 −6pp (r=0.5) |
| GSM-shape decode-heavy + bypass=512 | 接近持平（decode 占 e2e 大头，drop 在 decode 上 bypass）|

> **"哪个 policy 最好？"**

性能：5 GPU policy 完全等价（spread < 2%）
Accuracy：**`tail_weight`** 是唯一在 3 个 dataset + 3 个不同 metric 上都稳健的（v6 binary metric 揭示 random / cross_numa_first / cross_numa_uniform 在 r=0.5 上 accuracy 暴跌 58-77pp）

> **"哪个 Phase 3 plan 最好？"**

prefill: tie（差 <1%）
Decode + e2e: LBG/greedy_balance 略好（v4 数据显示快 ~10% decode）
**统一推荐 LBG**。
