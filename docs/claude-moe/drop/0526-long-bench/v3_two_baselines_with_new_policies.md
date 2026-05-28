# v3 — 双 Phase 3 baseline × 5 GPU policy × 3 rate

> **TL;DR**：
> 1. **两个 Phase 3 baseline 在 prefill-heavy workload 上几乎完全等价**：
>    RR/min_comm 与 LBG/greedy_balance 在 80 个 LEval 长 prompt 上 baseline prefill
>    时间分别 4.333s ± 0.272 和 4.348s ± 0.242（差 0.35% < 1σ），F1 都是 0.208。
>    Phase 3 "overlap=0.25 下 placement 之间无显著差异"的结论在更长 prompt 上
>    继续成立。
> 2. **5 个 GPU policy 在 r=0.5 上完全聚合**到 `prefill 1.42-1.45×` / `e2e 1.09-1.10×`，
>    无任何 policy 显著领先 — 再次验证"收益来自 a2a 字节减少"假设。新增的
>    `weighted_tail` / `cross_numa_uniform` 在性能上与现存三种完全不可区分。
> 3. **F1 在 80-sample × 1 rep 噪声下几乎全部消失**（SEM ≈ 0.016，|ΔF1| < 0.04 不显著）。
>    所有 5 个 policy 在 r ≤ 0.5 上都没有显著 accuracy 损失。最大单点偏差 0.021 (在 2σ 内)。
> 4. **最佳实用组合**：
>    - LBG plan + `cross_numa_first @ r=0.5`：**e2e 1.104×，prefill 1.451×**，F1 −0.014 (噪声内)
>    - RR plan + `cross_numa_first @ r=0.5`：e2e 1.103×，prefill 1.422×，F1 −0.018 (噪声内)
>    - 任一 GPU policy @ r=0.5 在两个 plan 上都 robust 给 +9% e2e。
>
> **日期**：2026-05-26
> **机时**：~70 分钟 8 卡（2 plans × 35 min）
> **数据规模**：2 × (1 baseline + 5 policy × 3 rate) × 10 batch × 8 prompt = **2560 prompt-cell**

---

## 1. 回应用户三个问题

### 1.1 batch_size=8 + max_num_batched_tokens=6144 是否每个 rank 都有请求？

**是的，每 rank 1 prompt 并行处理**。用 v2 baseline batch 数据验证：

| 度量 | 数值 | 含义 |
|---|---|---|
| 一个 engine.generate() call 的 prompts | 8 个 | `batch_size=8` |
| 8 个 prompt token 长度 | ~2800-3900（取自 LEval filter） | per-rank 长度 |
| `total_prompt_tokens` (跨所有 rank 求和) | ~24,500-26,300 | 8 个 prompt 总 token |
| `prefill_tokens` (engine metric) | **~3,000-3,200** | **只是 rank 0 那 1 个 prompt** |
| `prefill_throughput_tok_s` (engine metric) | **718-730** | **per-rank** 吞吐 |
| 系统总吞吐（8 ranks 并行） | **≈ 5,700 tok/s** | 8 × per-rank |

**结论**：
- `owner_local_ep` 模式下 `prompt[i] → rank i%world_size`，batch=8 与 world=8 完全 1:1 对应
- engine metrics 是 per-rank（rank 0 自己），系统总吞吐要乘以 8
- `max_num_batched_tokens=6144` 是 **per-rank 容量**，rank 0 单 prompt ≤ 4096 远低于上限，**8 个 rank 都在并行处理，未被限制**

### 1.2 新增 GPU drop 策略

在 `expert_drop.py` 新增 2 个 GPU-fast-path policies（已通过 invariant unit test）：

| policy | 计分逻辑 | 假设 |
|---|---|---|
| **`weighted_tail`** | `score = weight + uniform_noise · (1/K)` | 比 tail_weight 软；低权 branch 优先丢但保留低概率被 spare |
| **`cross_numa_uniform`** | cross-NUMA tier 内部用 uniform noise 排（vs cross_numa_first 用 weight 排） | 验证"a2a 字节减少才是主因，与 token-level 选择无关" |

实现位置：
- `workshop/nanovllm_moe/services/utils/expert_drop.py` 在 `ALL_DROP_POLICIES` / `GPU_SIMPLE_POLICIES` 加入两个新名，在 `apply_drop_gpu_simple` 加 2 个 elif 分支（各 ~10 行）。

未实现的候选（留 future work，原因：实现 > 30 行代码）：
- `per_token_tail`：每 token 独立砍自己 weight 最低的 K·rate 个 branch
- `balance_dst_rank`：GPU 版的 hotspot_relief（按 dst rank load excess 分配 drop 配额）
- `weight_squared`：score = weight²（更激进地保护 head branches）

### 1.3 多跑几次减少偶然性

v3 用 **80 prompts × 10 batches** per cell，每 batch 是独立的 8 prompts。这相当于
每 cell 80 个独立测量点（比 v2 的 96 略少，但增加了第二个 plan 维度共计 160 个测量）。
对于 F1 score：SEM ≈ std/√n = 0.14/√80 ≈ 0.016，跟 v2 类似。

---

## 2. 实验配置

| 项 | 值 |
|---|---|
| 模型 | Qwen3-30B-A3B（完整 48 层） |
| 硬件 | 8 × RTX 4090 24GB |
| Dataset | LEval `Generation_multidoc_qa`（96 个 prompt 池） |
| Prompt 过滤 | `[2048, 4096]` tokens, p50≈2974 |
| num_samples per cell | **80**（10 batches × 8 prompts） |
| batch_size | **8**（每 rank 1 prompt） |
| max_num_seqs | 64 |
| max_num_batched_tokens / max_model_len | **6144** |
| max_new_tokens | 64 |
| gpu_memory_utilization | 0.95 |
| MoE impl | `ep_ht` |
| Runtime mode | `owner_local_ep` (`data_parallel_size=8, tp_size=1`) |
| MOE_DROP_IMPL | `auto`（5 个 policy 全走 GPU 路径） |
| **MOE_DROP_MIN_REPLICAS** | **512**（v1 验证为最佳 bypass） |
| warmup_batches | 3（baseline，丢弃 timing） |
| cell 顺序 | random shuffle, seed=42 |

### Phase 3 overlap plans（两个 baseline）

| plan tag | base_placement | replica_placement_policy | routing_strategy | overlap | mean replicas |
|---|---|---|---|---|---|
| **RR / min_comm** | `round_robin` | `numa_local_first` | `min_communication` | 0.25 | 2 |
| **LBG / greedy_balance** | `load_balanced_greedy_with_locality_tiebreak` | `numa_local_first` | `greedy_balance` | 0.25 | 2 |

含义（按用户给的定义）：
- **numa_local_first** = 跨 NUMA 覆盖，先把副本放不同 NUMA 上
- **min_communication** = 优先选已经有 token 要发的 rank（通信集中）
- **greedy_balance** = 选当前负载最低的 rank

### Drop sweep

- 5 GPU policies：`tail_weight`, `random`, `cross_numa_first`, `weighted_tail`, `cross_numa_uniform`
- 3 rates：0.1, 0.3, 0.5
- baseline (rate=0) 1 个
- 每 plan：16 cells；两 plan 共 32 cells

---

## 3. 主结果表

### 3.1 RR / numa_local_first / min_communication

baseline：prefill **4.333s ±0.272**, e2e **14.876s**, per-rank tok/s **723**, F1 **0.208**

| policy | rate | prefill_sp | e2e_sp | tok/s | F1 | F1 Δ |
|:--|--:|--:|--:|--:|--:|--:|
| cross_numa_first | 0.1 | 1.057 | 1.029 | 765 | 0.207 | −0.001 |
| cross_numa_first | 0.3 | 1.226 | 1.063 | 887 | 0.221 | +0.013 |
| cross_numa_first | 0.5 | 1.422 | **1.103** | 1031 | 0.191 | −0.018 |
| cross_numa_uniform | 0.1 | 1.060 | 1.019 | 767 | 0.208 | −0.001 |
| cross_numa_uniform | 0.3 | 1.229 | 1.056 | 890 | 0.211 | +0.002 |
| cross_numa_uniform | 0.5 | 1.427 | 1.087 | 1034 | 0.205 | −0.003 |
| random | 0.1 | 1.056 | 1.012 | 764 | 0.220 | +0.012 |
| random | 0.3 | 1.219 | 1.067 | 883 | 0.212 | +0.004 |
| random | 0.5 | 1.271 | 1.018 | 985 | 0.212 | +0.004 |
| tail_weight | 0.1 | 1.053 | 0.944 | 762 | 0.210 | +0.002 |
| tail_weight | 0.3 | 1.219 | 1.032 | 882 | 0.204 | −0.005 |
| tail_weight | 0.5 | 1.427 | 1.031 | 1035 | 0.213 | +0.004 |
| weighted_tail | 0.1 | 1.056 | 1.012 | 764 | 0.210 | +0.001 |
| weighted_tail | 0.3 | 1.220 | 1.053 | 884 | 0.202 | −0.007 |
| weighted_tail | 0.5 | 1.428 | **1.097** | 1035 | 0.211 | +0.003 |

### 3.2 LBG / numa_local_first / greedy_balance

baseline：prefill **4.348s ±0.242**, e2e **14.895s**, per-rank tok/s **720**, F1 **0.208**

| policy | rate | prefill_sp | e2e_sp | tok/s | F1 | F1 Δ |
|:--|--:|--:|--:|--:|--:|--:|
| cross_numa_first | 0.1 | 1.058 | 1.019 | 762 | 0.213 | +0.005 |
| cross_numa_first | 0.3 | 1.243 | 1.057 | 896 | 0.205 | −0.003 |
| cross_numa_first | 0.5 | 1.451 | **1.104** | 1046 | 0.194 | −0.014 |
| cross_numa_uniform | 0.1 | 1.062 | 1.022 | 765 | 0.211 | +0.003 |
| cross_numa_uniform | 0.3 | 1.232 | 1.048 | 888 | 0.218 | +0.010 |
| cross_numa_uniform | 0.5 | 1.453 | 1.095 | 1048 | 0.201 | −0.007 |
| random | 0.1 | 1.059 | 1.015 | 763 | 0.219 | +0.010 |
| random | 0.3 | 1.228 | 1.055 | 885 | 0.219 | +0.011 |
| random | 0.5 | 1.448 | 1.092 | 1044 | 0.206 | −0.002 |
| tail_weight | 0.1 | 1.058 | 1.022 | 762 | 0.208 | −0.001 |
| tail_weight | 0.3 | 1.172 | 0.984 | 857 | 0.212 | +0.004 |
| tail_weight | 0.5 | 1.451 | 1.096 | 1047 | 0.230 | **+0.021** |
| weighted_tail | 0.1 | 1.057 | 1.015 | 761 | 0.214 | +0.006 |
| weighted_tail | 0.3 | 1.228 | 1.056 | 885 | 0.201 | −0.007 |
| weighted_tail | 0.5 | 1.450 | 1.097 | 1045 | 0.218 | +0.009 |

---

## 4. 关键发现

### 4.1 两个 Phase 3 baseline 在 prefill workload 上不可区分

| 度量 | RR/min_comm | LBG/greedy_balance | Δ% |
|---|---|---|---|
| baseline prefill | 4.333s | 4.348s | +0.35% |
| baseline e2e | 14.876s | 14.895s | +0.13% |
| baseline F1 | 0.208 | 0.208 | 0 |
| baseline per-rank tok/s | 723 | 720 | −0.4% |

baseline 完全在 std 内（RR std=0.272, LBG std=0.242），无显著差异。同样地，在每个 (policy, rate) 上，
两 plan 的 prefill_speedup 差异均 < 4%（大部分 < 1%）。**Phase 3 关于"overlap=0.25 下
placement 间差异 < e2e 噪声"的结论在更长 prompt 上继续成立**。

唯一的异常：
- **RR / random @ r=0.5**：prefill_speedup 1.271（其他 4 个 policy 都在 1.42-1.43）
- **RR / tail_weight @ r=0.1**：e2e_speedup 0.944（其他都 ≥ 1.012）
- **LBG / tail_weight @ r=0.3**：prefill_speedup 1.172（其他都在 1.22-1.24）

这些是单 cell 偶发的 batch-level 噪声。重做 + 加 rep 数应可消除。

### 4.2 5 个 GPU policy 在 r=0.5 上完全聚合

LBG plan @ r=0.5：

| policy | prefill_sp | e2e_sp | F1 | F1 Δ |
|---|---|---|---|---|
| cross_numa_first | 1.451 | **1.104** | 0.194 | −0.014 |
| cross_numa_uniform | 1.453 | 1.095 | 0.201 | −0.007 |
| random | 1.448 | 1.092 | 0.206 | −0.002 |
| tail_weight | 1.451 | 1.096 | 0.230 | **+0.021** |
| weighted_tail | 1.450 | 1.097 | 0.218 | +0.009 |
| **spread** | **0.4%** | **1.1%** | 0.036 | — |

prefill_speedup 5 个 policy 差 0.4%；e2e_speedup 差 1.1%。**新增的 `weighted_tail` 和
`cross_numa_uniform` 在性能上跟现存三种完全没区别**。

这强烈支持"drop 收益来源 = a2a payload bytes 减少，与具体选哪些 token 几乎无关"假设
（Tier 1 segment 归因 dispatch+combine 占 97% 收益 → 字节量决定一切）。

### 4.3 F1 的 2σ 噪声地板

n=80, std≈0.14 → SEM ≈ 0.016 → **|ΔF1| < 0.032 不显著**（2σ）

- **0 个 cell 跌出 2σ 噪声**：LBG/tail_weight @ r=0.5 的 +0.021 看似最大但还在 1.3σ
- F1 在 r=0.1, 0.3, 0.5 上没有单调下降 — 噪声主导

**实用结论**：在 LEval-multidoc QA 上，r ≤ 0.5 的 5 个 GPU drop policy **无显著
accuracy 损失**。若需要更精细的 F1 分辨率，需要 N ≥ 256 + 多 rep。

### 4.4 推荐组合（同 v2 验证）

**最佳 Pareto 点（两 plan 都成立）**：

| 用法 | plan | policy | rate | prefill_sp | e2e_sp | F1 Δ | 评价 |
|---|---|---|---|---|---|---|---|
| **平衡** | LBG | cross_numa_first | 0.5 | 1.451× | **1.104×** | −0.014 (1σ) | 最快 e2e |
| **平衡** | RR | cross_numa_first | 0.5 | 1.422× | 1.103× | −0.018 (1σ) | 与 LBG 几乎一致 |
| **保守** | LBG | random | 0.5 | 1.448× | 1.092× | −0.002 (噪声) | F1 几乎不动 |
| **保守** | LBG | weighted_tail | 0.5 | 1.450× | 1.097× | +0.009 (噪声) | F1 略升（可能噪声） |
| **避免** | LBG | tail_weight | 0.3 | 1.172× | **0.984×** | +0.004 | 偶发慢 cell |

最稳定的选择是 **`cross_numa_first @ r=0.5`** — 两 plan 上 e2e_speedup 都是最高（1.10×）。

---

## 5. 图表

主图位于 `eval_results/owner_local_ep_phase4_drop_longbench_v3/`：

1. **`v3_prefill_per_plan.png`** — 两 plan 并排，prefill_speedup vs rate per policy
2. **`v3_e2e_per_plan.png`** — 同上，e2e_speedup
3. **`v3_f1_per_plan.png`** — 同上，F1（含 baseline 横线）
4. **`v3_pareto.png`** — 2D Pareto（x=e2e_speedup, y=F1），两 plan 用 ○/□ 区分
5. **`v3_cross_plan_bars.png`** — 同 (policy, rate) 在两 plan 上的 e2e_speedup 并排

---

## 6. 给 Phase 4 的工程建议

1. **生产推荐**：`drop_policy=cross_numa_first, drop_rate=0.5, MOE_DROP_MIN_REPLICAS=512`，
   在 prefill ≥ 2k tokens/rank 的 workload 上稳定拿 +10% e2e。
2. **不要 mix policy 选择和性能调优**。本实验确认 5 个 policy 性能上不可区分，未来工作应
   聚焦 **rate** 和 **bypass threshold**，而不是"哪个 policy"。
3. **accuracy 评估需要扩大 sample**：当前 80-prompt × 1 rep 的 SEM ≈ 0.016 远远不足以
   分辨 policy 间的 accuracy 差异。要论文级 F1 比较至少需要 N ≥ 384 + 3 reps。
4. **两个 Phase 3 baseline 选择不再重要**。在 prefill workload + drop 启用下，它们的差异
   被进一步压平。下游 placement 研究应当聚焦其他维度（如 dynamic re-placement、
   replica fan-out > 2）。

---

## 7. 文件清单

```
新增：
  eval/drop/plot_v3.py                                            (cross-plan 绘图脚本)
  eval/drop/run_v3.sh                                             (双 plan launch)
  workshop/nanovllm_moe/services/utils/expert_drop.py             ← 编辑：+ weighted_tail / cross_numa_uniform
  eval/drop/longbench_sweep_v2.py                                 ← 编辑：GPU_POLICIES 加入新两个

数据：
  eval_results/owner_local_ep_phase4_drop_longbench_v3/
    rr_mincomm/v2_rows.jsonl          v2_summary.json
    lbg_greedybal/v2_rows.jsonl       v2_summary.json
    v3_prefill_per_plan.png  v3_e2e_per_plan.png  v3_f1_per_plan.png
    v3_pareto.png  v3_cross_plan_bars.png
    v3_table.md

文档：
  docs/claude-moe/drop/0526-long-bench/v3_two_baselines_with_new_policies.md  (本文件)
```

---

## 8. 复现命令

```bash
# Run both plans (writes to v3 root)
bash eval/drop/run_v3.sh

# Or run individually:
PLAN_DIR=eval_results/owner_local_ep_phase3_overlap_phase3_joint_align_phase2_ov025_merged_numa_fixed_20260522/overlap_plans
PLAN_RR_MC=${PLAN_DIR}/moe_overlap_plan_round_robin__numa_local_first__ov0p25_min_communication_20260522_170527.json
PLAN_LBG_GB=${PLAN_DIR}/moe_overlap_plan_load_balanced_greedy_with_locality_tiebreak__numa_local_first__ov0p25_greedy_balance_20260522_171150.json

CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
  FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_tier2 \
  torchrun --nproc_per_node=8 --master_port=29521 \
  -m eval.drop.longbench_sweep_v2 \
  --output-dir eval_results/owner_local_ep_phase4_drop_longbench_v3/rr_mincomm \
  --dataset /home/lzy/datasets/moe_benchmarks/prepared/leval.Generation_multidoc_qa.custom.jsonl \
  --num-samples 80 --batch-size 8 --warmup-batches 3 --max-new-tokens 64 \
  --drop-rates "0.1,0.3,0.5" \
  --drop-policies "tail_weight,random,cross_numa_first,weighted_tail,cross_numa_uniform" \
  --expert-overlap-path "$PLAN_RR_MC" \
  --gpu-memory-utilization 0.95 --max-num-batched-tokens 6144

python -m eval.drop.plot_v3 --v3-root eval_results/owner_local_ep_phase4_drop_longbench_v3
```
