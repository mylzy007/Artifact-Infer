# Prefill L Sweep 实验计划（压缩版，2 天）

> **目的**：48 小时内回答两个问题——
> (1) drop 在什么情况下能起作用？
> (2) 我当前的 drop 实验有没有可能取得 e2e 提升？
>
> **日期**：2026-05-26
> **作者**：lzy + Claude
> **预备读物**：
> - `docs/claude-moe/drop/background.md`
> - `docs/claude-moe/drop/0526-superpowers/0526-prefill_l_sweep_socratic.md`

---

## 0. 已决断的开放问题

| 问题 | 我的决断 | 理由（一行） |
|---|---|---|
| H1 阈值 | **−5%** | 0% 在 4090 boost 抖动下区分不出来；−5% 是 figure-resolution 阈值 |
| `T_local` 取值 | `{8, 64, 512, 4096, 16384}` 5 点 | 5 点几何级数足够定位 kink；从 anchor 8 一路到 ≥ Tier 2 上限 |
| Tier 2 sanity 必跑 | **否**（仅在 H1 推翻时跑） | 2 天 budget 没空跑无收益的 sanity 点 |
| Pre-registration | **嵌进本计划**，不另起文件 | 本计划本身就是预注册，节省 0.5 day |
| 是否扫 rate=0.5 / controlled mode | **不扫** | 单一 LBG + `tail_weight@0.3`；其他维度留 Phase 5 |

**砍掉**：`effective_drop_frac` 前置 profile、`nccl-tests` overlay、`controlled_balanced` 对照、GSM8K sanity（无条件）、anchor point 30-iter 加测、receive-side hook 预埋。

**保留**（任何情况都不能砍）：baseline-only 单独 sweep（cache artefact 消除前提）、segment 计时（H2 归因）、随机化 cell 顺序（降频检测）、`L_recv_max` 跨 rank 聚合。

---

## 1. Research Question

**(Q1)** `tail_weight@0.3` token-replica drop 在 8×4090 EP-HT 上需要多大的
`L_recv` 才能让 MoE-block wall time 加速 ≥5%？

**(Q2)** 该 L\* 是否落在 24GB 单机可达 prefill 体量内（`L_recv ≤ 32k`）？

回答 (Q2) = 直接回答用户的两个问题：
- (Q2) 答 yes → drop 在你够得着的 prefill workload 上有可能起作用。
- (Q2) 答 no → drop 在当前硬件 + workload 下不可能有 e2e 提升，写 closed-negative 收尾。

---

## 2. Pre-Registered Hypotheses & Predictions

**H1（主，可证伪）**：在 `L_recv ∈ [8, 16384·8/8 ≈ 16k]` 内，
`delta_pct(L) = (drop_total_us − baseline_total_us) / baseline_total_us > −0.05` 恒成立
（即找不到 L\*，drop 不可能有 ≥5% 加速）。

**H2（segment 归因）**：若 H1 推翻，收益的 ≥70% 来自 `experts_us` 段。

**预测曲线**（事前写死）：
- `baseline_total_us(L)`：L≤64 平台（latency-bound）；L≥4096 进入近线性；中间过渡。
- `delta_pct(L)`：从 0±2%（L=8）单调下降，最低点估计 −5% 到 −15%。
- L\* 数量级猜测：押 **L\*∈[4k, 16k]**（即 Tier 1 上界附近，边缘 case）。
- Effective drop send frac：L=8 时 ~0.05（几乎全 local 无 droppable），L=16k 渐近 0.25-0.3。

**Falsification floor**（数据卫生）：L_recv=8 上 `delta_pct > +2%` → GPU drop 没真零开销，停下查 implementation。

---

## 3. Tier 1 — MoE-Block 隔离微基准

### 3.1 隔离

Timed region 只包含：

```
DispatchEPHT.forward → ExpertsEPHT.forward → CombineEPHT.forward
```

**Precompute** 在 timed 前：`ExpertOverlapRouter.route()` 结果 cache，
`hidden_states` / `router_logits` 预生成。

**剥离**：attention / sampling / scheduler / KV cache。

### 3.2 Sweep 参数

| 参数 | 值 |
|---|---|
| `T_local` | `{1, 8, 64, 512, 2048}`（对应 nominal `L_send = T*K ∈ {8, 64, 512, 4096, 16384}`） |
| `K` | 8 |
| `world_size` | 8（torchrun --nproc_per_node=8） |
| `drop_policy` | `tail_weight` |
| `drop_rate` | `{0.0, 0.3}` |
| `routing_mode` | `random_logits_lbg`（单一，复用 Phase 3 LBG overlap plan） |
| `warmup_iters` | 5 |
| `iters` | 8 |
| `dtype` | bf16 |
| Cell 顺序 | **随机** |
| Cell 间隔 | sleep 3s |

**Env**：

```
MOE_DROP_IMPL=gpu
MOE_DROP_MIN_REPLICAS=0
MOE_DROP_GPU_STATS=0
MOE_PROFILE_OVERLAP_RUNTIME=0
MOE_RECORD_TIMING=0
```

### 3.3 跑两轮

**Round A：baseline-only**（drop_rate=0）—— 5 cell。
**Round B：drop**（drop_rate=0.3）—— 5 cell。

两轮 cell 顺序都随机化但**用同一份 random seed**，确保 cell 内 `hidden_states` / `router_logits` 一致。

**主曲线**：`delta_us(L) = round_B(L) − round_A(L)` —— 减掉所有共有 cache / Triton / NCCL artefact。

### 3.4 输出 metric

每 cell（median over 8 iters）：

```
# 主结论
total_us_rank_max
delta_us_median       = drop_total_us - baseline_total_us
delta_pct_median      = delta_us / baseline_total_us
speedup_ratio_median  = baseline / drop

# Segment（H2）
dispatch_us_rank_max, experts_us_rank_max, combine_us_rank_max

# Drop 实际效力
L_send_kept_mean, L_recv_mean, L_recv_max, L_recv_cv
effective_drop_send_frac, effective_drop_recv_frac
```

### 3.5 输出文件

```
eval_results/prefill_drop_l_sweep_tier1/
  tier1_rows.csv
  tier1_summary.json     # L*, decision, H1/H2 verdict
  tier1_total_vs_l.png   # log-x, baseline + drop 两条线
  tier1_delta_vs_l.png   # delta_pct 主图
  tier1_segments.png     # dispatch/experts/combine 三条线
```

---

## 4. L\* 判定（精确）

主轴 `L_recv` 用 `L_recv_max_median`（slowest rank 决定 step 时间）。

```python
def compute_L_star(rows):
    # rows sorted by L_recv_max asc
    pts = [(L, delta / base) for L, delta, base in rows]
    for i in range(len(pts) - 1):
        L0, p0 = pts[i]
        L1, p1 = pts[i+1]
        if p0 > -0.05 and p1 <= -0.05:
            return L0 + (-0.05 - p0) * (L1 - L0) / (p1 - p0), "crossing_found"
    if pts[0][1] <= -0.05:
        return pts[0][0], "crossing_below_min"
    return None, "no_crossing"
```

**判决**：

| compute_L_star 返回 | H1 | 行动 |
|---|---|---|
| `no_crossing` | 未推翻 | **closed-negative**：drop 不可能有 e2e 提升。结束，写报告。 |
| `crossing_found` 且 L\* ≤ 16384 | 推翻 | 进 Tier 2 跑 1 个验证点 |
| `crossing_below_min` | 推翻 | 进 Tier 2 跑 1 个验证点 |

**H2 归因**（仅 H1 推翻时）：

```
ratio_experts = (experts_us_base - experts_us_drop) / (total_us_base - total_us_drop)
```

`ratio_experts ≥ 0.7` → H2 成立（收益来自 GEMM 减少，符合直觉）。
`< 0.7` → 收益来自 dispatch/combine 通信侧 → 写一段诊断。

---

## 5. Tier 2 — 仅在 H1 推翻时触发

### 5.1 触发后跑什么

**1 个点**，synthetic prompt：

```
target_T_total = round_to_8(L_star)
batch=4, prompt_len = target_T_total / 4
max_new_tokens = 1
3 repeats, 1 warmup
```

跑 baseline + drop 各一次，比较：

```
prefill_speedup = baseline_prefill_time / drop_prefill_time
e2e_speedup     = baseline_e2e_time / drop_e2e_time
peak_memory_allocated_gb
```

**通过条件**：`prefill_speedup ≥ 1.05` → 正向，drop 在你的硬件上**有可能起作用**。
否则 → MoE-block 上的 speedup 被 attention 完全稀释，写"Tier 1 有效 / Tier 2 无效"诊断。

### 5.2 不做

- 不跑 GSM8K（accuracy 在 v1/v2 已测）
- 不跑多 batch 多 prompt_len 扫
- 不跑 controlled_balanced 对照
- 不重测 effective_drop_frac

---

## 6. Risk — L\* 不存在怎么办

H1 未被推翻（`no_crossing`），即 `T_local=2048 (L_recv≈16k)` 仍未到 −5%。

**结论可直接写**：

> 在 8×RTX 4090 24GB、Qwen3-30B-A3B、EP-HT owner-local 配置下，
> `tail_weight` policy 在 drop rate ≤ 30% 时，对 `L_recv ≤ 16k` 范围内
> MoE-block 无 ≥5% wall-time 加速。e2e prefill 体量在 24GB 单卡可达
> 范围内 (`prefill_tokens ≤ 16k`) 不可能取得净收益。与 CPU drop 和
> K_eff hard-cap ablations 一致。

**用户问题的直接答案**：
- "drop 有可能起作用吗？" → **不可能**在当前 (硬件, 模型, workload, policy, rate) 配置下。
- "在什么情况下能起作用？" → 需要至少满足下列之一：
  - `L_recv > 16k`（需要 >24GB VRAM / 多卡 / 更大模型 / 更长 prompt task）
  - 换 policy 让 `effective_drop_recv_frac` 显著大于当前 measure 值
  - 换 receive-side drop 跳过 dispatch 开销

---

## 7. Timeline（48 小时硬时间盒）

| 时段 | 工作 | 8 卡机时 |
|---|---|---|
| **Day 1 AM (4h)** | 实现 `_bench_prefill_drop_sweep.py`（脚本职责见 §8） | 0 |
| **Day 1 PM 1h** | 2 卡 smoke：`T_local=8,64`，`iters=2 warmup=1`；验证 CSV/plot pipeline | 0.1h |
| **Day 1 PM 1h** | 8 卡 baseline-only sweep（Round A） | 0.3h |
| **Day 1 PM 1h** | 8 卡 drop sweep（Round B） | 0.3h |
| **Day 1 PM 1h** | 自动分析、画图、写 `tier1_summary.json` | 0 |
| **Day 1 晚** | 看 L\* 结果，决定走 Tier 2 还是写收尾 | 0 |
| **Day 2 AM (4h)** | **路径 A**（H1 推翻）：实现 Tier 2 脚本 + 跑 1 点 + 分析 | 0.5h |
| | **路径 B**（H1 未推翻）：直接写 closed-negative 报告 | 0 |
| **Day 2 PM (4h)** | 写最终结论（路径 A: 正向 story / 路径 B: closed-negative + roofline 简述） | 0 |

**总卡时**：~1 h（路径 B）或 ~1.5h（路径 A）。
**总人工**：约 16 工作小时，分两天。

**Hard stop**：Day 2 22:00 之前必须出最终结论文档，不允许加点延期。

---

## 8. 实施 — Tier 1 脚本职责（不写代码，先列契约）

文件：`workshop/nanovllm_moe/_bench_prefill_drop_sweep.py`

**职责契约**（执行前需用户再确认一次）：

1. **分布式 init**：torchrun 环境变量 → `init_parallel_groups(tp=1, dp=world, runtime=owner_local_ep)`
2. **MoE block 构造**：
   - `T_cap = max(T_local_values)`
   - `MoeBackend(config, E=128, K=8, H=2048, N=768)` 一次最大容量
   - `DispatchEPHT / ExpertsEPHT / CombineEPHT` 单层，挂接 backend buffers
   - 权重 `normal_(std=0.02)`，避免 NaN
3. **CLI**（最小）：
   ```
   --output-dir
   --t-local-values 1,8,64,512,2048
   --drop-rates 0,0.3
   --expert-overlap-path <LBG json>
   --iters 8
   --warmup-iters 5
   --seed 42
   ```
4. **每 cell 流程**：
   - `dist.barrier()` → 5 warmup → `cuda.synchronize()` → 8 measured（4 段 CUDA event）
   - `all_gather` 收集 per-rank `(dispatch_us, experts_us, combine_us, total_us, L_recv, L_send_kept)`
   - 每 repeat 取 `rank_max`，cell 内 median over repeats
5. **Cell 顺序随机化**：固定 seed 的 `random.shuffle((T_local, rate) pairs)`
6. **rank0 输出**：CSV + JSON + 3 张 plot
7. **自动决策**：在 summary.json 中写：
   ```json
   {
     "L_star": null or float,
     "h1_status": "refuted" or "not_refuted",
     "h2_ratio_experts": float or null,
     "decision": "RUN_TIER2" or "STOP_CLOSED_NEGATIVE",
     "pre_reg_check": {
       "small_L_drop_overhead_pct": float,
       "falsification_floor_violated": bool
     }
   }
   ```
8. **门控**：若 `pre_reg_check.falsification_floor_violated == True`（L=8 上 drop 比 baseline 慢 ≥2%），**`decision = "STOP_INVESTIGATE_DROP_IMPL"`**，不继续。

---

## 9. 如果一切顺利的最终交付物

**路径 B（closed-negative，预期路径）**：

`docs/claude-moe/drop/0526-plan/0526-final_report.md` 包含：

1. RQ + H1 + 数据：`delta_pct vs L_recv` 主图
2. Verdict：H1 未被推翻
3. 用户问题直接回答（§6 那段）
4. 与 v1/v2、K_eff 三个负实验的并排对照表
5. 下一步建议（receive-side drop / 换 workload / pivot）

**路径 A（H1 推翻，意外路径）**：

`docs/claude-moe/drop/0526-plan/0526-final_report.md` 包含：

1. Tier 1 L\* + 曲线
2. Tier 2 单点 e2e 验证
3. 适用区间 statement
4. Phase 5 ablation 计划草稿（receive-side + 多 policy + 多 workload）

---

## 10. 自检

- ✅ 5 个开放问题全部决断
- ✅ 时间盒压到 2 天，机时 ~1.5h
- ✅ Hypotheses 在数据前 commit（本文件即 pre-reg）
- ✅ 不做的事明确（§5.2 / §0 砍掉清单）
- ✅ Falsification floor + 失败路径都有 fallback（§4 决策树 / §6 closed-negative claim）
- ⚠️ 风险：Tier 1 脚本 4h 实现窗口偏紧——若 Day 1 AM 写不完，砍掉 segment 计时只保 total（H2 归因放弃）

---

## 等用户最后一次确认

只需要回答 **是 / 不是**：

- 计划如上，可以开始写 pre-reg + Tier 1 脚本吗？

确认后我立即开工，不再追问。
