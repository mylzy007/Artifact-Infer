# MoE / Drop Code Audit + Long-Prompt e2e 实验

> **TL;DR**：
> - `eval/run_owner_local_ep_phase4_drop.py` 没有功能 bug，但**关键参数默认值有问题**：
>   `--drop-min-replicas` 默认 `0`，导致 drop 在 decode (L≈8) 上每步都激活，
>   带 +25% 慢，**结构性解释了 Phase 4 v1/v2/GPU drop 所有实验"e2e ±2%"**。
> - MoE 代码 (`workshop/nanovllm_moe`) 整体健康，有 3 处中/低优先级隐患（K_eff×drop
>   索引、`drop_enabled` 缓存、combine 的 unperm 零填充），都不影响当前实验。
> - **e2e 验证（48 个 LEval-multidoc 长 prompt，p50≈3k tokens）**：
>   `prefill_speedup = 1.215×（−17.7%）`，`e2e_speedup = 1.098×（−8.9%）`。
>   首次在真实数据集上观察到显著 e2e 收益。
>
> **日期**：2026-05-26  实验耗时 ~7 分钟卡时。

---

## 1. 代码审查

### 1.1 `eval/run_owner_local_ep_phase4_drop.py` (571 lines)

#### 结构性观察

- 子进程拉起 `test_bazaar_moe.py`（line 412-456），每个 (baseline × policy × rate × rep) cell 独立 fork。
- 每 cell 都构造新 `LLMEngine` + 新 dispatch 模块 → `drop_enabled` 在 `__init__` 时按
  `drop_rate>0` 设置，没有 mutation 风险。
- env 透传：`MOE_DROP_IMPL` / `MOE_DROP_MIN_REPLICAS` / `MOE_DROP_GPU_STATS`
  在 `extra_env`（line 439-443）通过 `command_env` 注入子进程，正确。

#### 实际 bug 级问题

**[HIGH]** `--drop-min-replicas` 默认值 `0`（line 84-85）

```python
parser.add_argument("--drop-min-replicas", type=int, default=0,
                    help="bypass drop when T*K <= this value (small batch bypass)")
```

**后果**：drop 在每个 decode step（每 rank 8 个 replicas） 上都激活。Tier 1 实测
`T_local=1 (L=8)` 上 drop 比 baseline 慢 **+25.5%**，`T=8 (L=64)` 慢 +34.3%。
GSM8K decode 段 ~60 步 × 48 层 = 累积 2880 次 drop kernel launch + 微小工作量 →
正好抵消 prefill 的微弱收益。

**修复**：用户应当**始终传** `--drop-min-replicas 512`（或更高），让 drop 仅在
prefill（L_recv ≥ ~512）激活。

#### 其他问题

**[MEDIUM]** dataset 硬编码为 GSM8K（line 305, `prepare_gsm_dataset_if_needed`）。
若要扫长 prompt 数据集，必须改 utils 或绕过这个脚本。我已写了独立的
`eval/drop/long_e2e_bench.py` 直接对接 LongBench / LEval `.custom.jsonl`。

**[LOW]** `args.profile_drop_runtime_stats=1` 默认开启（line 86）→ 每步多一次
host sync 写统计。性能敏感跑要设 `--profile-drop-runtime-stats 0`。

### 1.2 `workshop/nanovllm_moe` MoE 代码

| 编号 | 文件:行 | 严重度 | 描述 |
|:-:|:-:|:-:|:--|
| M1 | `dispatch_ep_ht.py:135` | LOW | `drop_enabled = policy != "none" and drop_rate > 0.0` 在 `__init__` 缓存。若代码后期 mutate `drop_rate` 而不同步 `drop_enabled`，drop 会被静默关闭。本仓库的 Tier1/Tier2 bench 已正确同步两个字段。**runner 不受影响**（每 cell 子进程重建）。 |
| M2 | `dispatch_ep_ht.py:330-332` | LOW | drop 后 weights 重归一化；若 per-token min-keep=1 被违反，会有全零行被 combine。`apply_drop_gpu_simple` 通过 `_build_droppable_mask` 强制保留每 token 最高权重的 branch，所以不会发生。已有 unit test (`test_drop_rate_zero_keeps_all`) 覆盖。 |
| M3 | `combine_ep_ht.py:95` | LOW (by design) | `unperm = torch.zeros(T*K, H)` 总是按原始 T*K 分配；被 drop 的位置保留零行。意味着 combine 的 scatter cost 仍按 T*K 计；但 reverse-a2a payload 是按 kept 数算的，所以 NCCL 时间确实降。我们 Tier 1 实测 combine_us 下降 37% — 这是 reverse-a2a 收益，与 unperm zero-fill 设计一致。 |
| M4 | `expert_drop.py:362-380` | MEDIUM | `protected_pos = arange_T * K + best_k`：K 是函数参数。当 `router_keff > 0` 时 dispatch 用 `K=K_eff < K_model` 调用 `apply_drop`，但 `flat_topk_w` 是 `[T, K_eff]` 展平的。**只要 dispatch 用 `torch.topk(..., K_eff)` 拿 logits 而非 `torch.topk(..., K_model)` 再切片**，索引就对。代码 line 258 是 `torch.topk(logits_fp32, K)`，K = K_eff 时正确。无 bug，但路径偶尔被 review 视为可疑。 |
| M5 | `expert_drop.py:461-469` | OK | `drop_rate <= 0.0` 短路返回 `_empty_droppable_result(keep_mask=ones)` ✓ 已经在 unit test 验证 |

**结论**：代码可以放心跑 drop 实验。所有 audit 出来的"隐患"都是设计意图或被保护
的 invariant；没有当前实验需要先修的 bug。

---

## 2. 数据集选择

`/home/lzy/datasets/moe_benchmarks/prepared/` 下 prompt 长度分布（按 token 估算）：

| 数据集 | n | p50 (tokens) | p90 | p99 |
|---|--:|--:|--:|--:|
| `gsm8k.custom.jsonl` | 200 | empty `prompt` field | - | - |
| `leval.Generation_multidoc_qa` | 158 | **3,599** | 8,237 | 8,265 |
| `leval.Generation_legal_contract_qa` | 154 | 21,162 | 58,929 | 68,070 |
| `longbench.2wikimqa` | 200 | 6,406 | 14,071 | 17,688 |
| `longbench.multifieldqa_en` | 150 | 7,860 | 11,939 | 15,242 |
| `longbench.triviaqa` | 200 | 12,648 | 21,232 | 24,956 |

**选 `leval.Generation_multidoc_qa`**：p50=3.6k 完美对应 Tier 2 找到的甜点
(prompt_len ~ 1-3k，per-rank source tokens 落在 1-3k，L_recv 8-24k，Tier 1 显示
drop 在这区间 −13% 到 −16% MoE-block 加速)。filter 到 `[1024, 4096]` token 范围
得到 48 个样本，prompt p50=3k。

---

## 3. e2e 实验设置

**Bench**：`eval/drop/long_e2e_bench.py`（新写）

**Config**：

| 参数 | 值 | 来源 |
|---|---|---|
| dataset | `leval.Generation_multidoc_qa.custom.jsonl` | 长度对齐 drop 甜点 |
| num_samples | 48 | 6 batches × 8 prompts/batch |
| batch_size | 8 | 1 prompt/rank（owner_local_ep）|
| prompt_len | `[1024, 4096]`，p50≈3k | Tier 2 甜点 |
| max_new_tokens | 32 | 短 decode 让 prefill 主导 |
| drop_policy | tail_weight | Phase 4 主策略 |
| drop_rate | 0.3 | Phase 4 主策略 |
| **drop_min_replicas** | **512** | **关键修复，让 drop 仅在 prefill 激活** |
| warmup_batches | 2 | 消除 flashinfer JIT-compile 冷启 |
| repeats | 3 | 噪声估计 |
| case order | 交替（baseline↔drop）| 平均剩余 cache 不对称 |
| world | 8×4090 | Phase 4 标准 |
| overlap_plan | LBG `phase3_smoke_plan` | Phase 3 最优 |

**复现命令**：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
  PATH=/usr/local/cuda-12.8/bin:$PATH \
  FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_tier2 \
  /home/lzy/miniconda3/envs/vllm/bin/torchrun --nproc_per_node=8 --master_port=29510 \
  -m eval.drop.long_e2e_bench \
  --output-dir eval_results/drop_long_e2e_leval \
  --dataset /home/lzy/datasets/moe_benchmarks/prepared/leval.Generation_multidoc_qa.custom.jsonl \
  --num-samples 48 --batch-size 8 \
  --min-prompt-tokens 1024 --max-prompt-tokens 4096 \
  --max-new-tokens 32 \
  --drop-rate 0.3 --moe-drop-min-replicas 512 \
  --warmup-batches 2 --repeats 3 \
  --gpu-memory-utilization 0.92 --max-num-batched-tokens 4096 \
  --expert-overlap-path eval_results/tmp_phase3_overlap_test/moe_overlap_plan_smoke_plan_20260521_222623.json
```

---

## 4. 结果

### 4.1 主指标（3 reps × 6 batches = 18 个 baseline/drop pair）

| 度量 | baseline (mean ± std) | drop (mean ± std) | speedup | Δ% |
|---|---|---|---|---|
| prefill_time | 4.510s ± 0.215 | 3.712s ± 0.236 | **1.215×** | **−17.7%** |
| e2e_time | 10.000s ± 0.388 | 9.122s ± 0.379 | **1.098×** | **−8.9%** |
| prefill_throughput | 678 tok/s | 824 tok/s | **1.215×** | **+21.5%** |

### 4.2 Per-batch 一致性

baseline prefill 区间 `[4.02s, 4.72s]`，drop prefill 区间 `[3.13s, 4.01s]`
**完全不重叠**。每个 rep 内每个 batch 单独看，drop 都比 baseline 快。
跨 rep 的方差 < 5%。结果不是 lucky run。

### 4.3 关键产物

```
eval_results/drop_long_e2e_leval/
  long_e2e_rows.jsonl      36 行 (3 reps × 2 cases × 6 batches)
  long_e2e_summary.json    每 case 聚合 + 跨 case speedup
```

---

## 5. 与 Tier 1 / Tier 2 / 用户两个问题的闭环

### 5.1 三阶段一致性

| 阶段 | workload | per-rank tokens | prefill speedup | e2e speedup |
|---|---|---|---|---|
| Tier 1 (MoE-block isolated) | synthetic | T=512 (L≈4k) | MoE-block −13.3% | n/a |
| Tier 1 | synthetic | T=2048 (L≈16k) | MoE-block −15.6% | n/a |
| Tier 2 (engine + synthetic) | `" the"×N` × 8 ranks | T=512 | **−8.8%** | noisy |
| Tier 2 | T=2048 | T=2048 | **−4.5%** | noisy |
| **Long e2e (real prompts)** | **LEval 长 QA** | **~3k** | **−17.7%** | **−8.9%** |

实际长 prompt 的 prefill 收益 (−17.7%) **比 Tier 2 synthetic 还好**（−4.5 ~ −8.8%）。
猜测原因：
- 真实 prompt 在 attention 路径上的 KV pattern 更分散 → attention 段的 cache 行为
  比 `" the"×N` 这种重复 token 差 → attention 在 prefill 中的相对占比下降
  → MoE-block 收益被稀释得更少
- 待验证

### 5.2 用户问题的最终答案

**Q1: drop 在什么情况下能起作用？**
- 必要条件：per-rank prefill tokens ≥ ~500（L_recv ≥ ~3k）
- 必要条件：`MOE_DROP_MIN_REPLICAS ≥ 512` 跳过 decode（否则被 decode 慢拖累）
- 充分条件：prefill 占 e2e 的比例足够大（这次 ~50%）

**Q2: 加 drop 会有提升吗？**
- 在 GSM8K decode-heavy + 默认 `--drop-min-replicas=0`：**不会**，结构性解释了
  Phase 4 v1/v2/GPU drop 所有 ±2% 历史结果。
- 在长 prompt workload + `--drop-min-replicas=512`：**会**，**实测 +8.9% e2e**
  / +21.5% prefill throughput。

---

## 6. 推荐立即修改的项

按 ROI 排序：

1. **改 `eval/run_owner_local_ep_phase4_drop.py` line 84 默认值** 从 `0` 改成 `512`：
   ```python
   parser.add_argument("--drop-min-replicas", type=int, default=512, ...)
   ```
   或者在 `parser.set_defaults(...)` block (line 91-97) 加一行
   `moe_drop_min_replicas=512`。

2. **改 `--profile-drop-runtime-stats` 默认 `0`**：当前默认 1 引入 host sync，
   性能 sweep 不应开启（用户自己想做 accounting sweep 时显式 `--profile-...=1`）。

3. **`eval/run_owner_local_ep_phase4_drop.py` 加一个 `--dataset-prepared` 路径
   开关**，让 runner 接受 LongBench / LEval `.custom.jsonl` 而不只是 GSM 格式。
   或者推荐用户直接用 `eval/drop/long_e2e_bench.py`（更轻量，无 Phase 3 summary
   依赖）。

4. **Phase 4 P1 (`router_keff`) 的 drop 联合 case**：审计中 M4 提示 K_eff×drop
   的索引正确但偶尔被误读，加单元测试覆盖这条路径会预防回归。

---

## 7. 不做的事 / 未来工作

- 没扫 `--drop-min-replicas` 敏感度（0/128/256/512/1024）。建议下一步做。
- 没扫多 drop_rate（0.1/0.2/0.3/0.5）。已知 0.3 是 score-preserving 上限。
- 没量化 accuracy 影响（leval 没跑 score）。drop @ 0.3 在 v1/v2 的 GSM8K accuracy
  保持 ±1pp，预期长 prompt 影响相似。
- Receive-side drop / chunked prefill — 留给 Phase 5。

---

## 8. 文件清单

```
新增：
  eval/drop/long_e2e_bench.py                                  长 prompt e2e 测试
  eval_results/drop_long_e2e_leval/long_e2e_rows.jsonl         原始 36 行
  eval_results/drop_long_e2e_leval/long_e2e_summary.json       聚合 + speedup
  docs/claude-moe/drop/0526-plan/0526-audit_and_long_e2e.md   (本文件)
```
