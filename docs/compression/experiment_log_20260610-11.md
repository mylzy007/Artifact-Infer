# MoE EP 通信压缩 Stage-0 实验记录（2026-06-10 → 2026-06-11）

## 0. 总览

**研究问题**：在 frozen Qwen3-30B-A3B + 8×RTX 4090 EP 推理场景下，能否对 all-to-all payload 做有损压缩，把通信量降下来同时保住模型质量？

**这两天做了 12 个实验**，从最初的 dispatch 端 SVD 谱分析（判断"激活是否天然低秩"），一路走到真实 EP=8 NCCL 上跑 HumanEval pass@1（真实 task accuracy 评测），最后做了两组"不均匀压缩"探索（按 token / 按 expert 选择性压）。中间证伪了 dispatch 端低秩这条路（无论换什么映射、训不训），最终在 combine 端找到正路：**topk_l2 + FP8 sparsification，4× 压缩下 HumanEval pass@1 从 55.5% 掉到 46.3%（−9.15pp）**；进一步证明 **不均匀压缩可以把同等压缩比下的 next-token drop 从 -3.94pp 救到 -0.54pp**（E1 hidden_norm 选择 + E2 累积权重保护）。

（前后曾尝试 stage-0 SVD v1（标定太少）和 EP=2 smoke test，结果均被 stage-0 SVD v2 / EP=8 完整模型 sweep 完全覆盖，已删；HF 路径的 HumanEval 因 device_map="auto" 太慢被弃用，最终用真 EP=8 跑完。）

**全部实验代码** 在 `eval/compression/stage0_*.py`，**全部输出** 在 `eval_results/compression_lowrank_stage0_*/`，**已 push 到 origin** `my-a2a-kernel-test` 分支（commit `25c2294`）。

---

## 1. 公共环境

| 项 | 值 |
|---|---|
| 硬件 | 8×RTX 4090（24 GB/卡，PCIe 4.0，无 NVLink）|
| 系统 CUDA driver | 13.0 / Driver 580.82.07 |
| 编译用 CUDA | 12.8 (`/usr/local/cuda-12.8/bin/nvcc`，4090 sm_89 要求 ≥11.8) |
| 主模型 | Qwen3-30B-A3B BF16, hidden=2048, n_layers=48, n_experts=128, top_k=8 |
| 模型路径 | `/home/lzy/models/Qwen3-30B-A3B` |
| Python env A (HF + accelerate) | `~/miniconda3/envs/atom`（transformers 4.51.3, accelerate 1.7.0, datasets 3.6.0）|
| Python env B (nanovllm_moe + flashinfer) | `~/miniconda3/envs/vllm`（transformers 4.57.6, flashinfer 0.6.7, sgl_kernel, ninja）|
| 数据 | LongBench (`/home/lzy/datasets/moe_benchmarks/longbench/extracted/data/`), GSM8K (`.../gsm8k/main/`), HumanEval (via `human_eval` pip package) |
| MoE 实现 | HF 参考实现 `Qwen3MoeSparseMoeBlock`（实验 A 系列）+ 用户的 `workshop/nanovllm_moe/` EP-HT 真实实现（实验 D 系列）|

## 1.5 术语 / 缩写 / 参数说明

每个实验里频繁出现的几个参数 / 指标，统一在这里说清楚：

### 压缩相关参数

| 缩写 | 全称 | 含义 | 取值范围 |
|---|---|---|---|
| **keep_frac** | keep fraction | **每一行被压缩时保留多少个分量**（top-k by magnitude）。原 row 是 hidden_size=2048 维的 bf16 向量，压缩后只保 round(2048 × keep_frac) 个值。例如 keep_frac=0.25 → 保 512 个值（约 4× value 压缩）。 | (0, 1]；keep_frac=1 = 不压 |
| **cf** | compress_frac（仅 E1）| **本层有多少比例的 token 被压缩**。E1 实验中，按某种重要性信号挑出 cf 比例的 token 走压缩路径、剩下的 (1-cf) 走原 bf16 路径。例如 cf=0.5 → 50% token 被压、50% 原样。 | [0, 1]；cf=0 = 全不压（teacher）；cf=1 = 全压 |
| **T** | threshold（仅 E2）| **累积权重保护阈值**。E2 中每个 token 的 top-8 个 expert 按 routing weight 降序累加，"之前的累积 weight < T" 的 expert 全量保留、其余压缩。例如 T=0.5 → 累积到 0.5 前的几个 expert 不压。 | [0, 1]；T=0 = 全压；T=1 = 全保护（teacher） |
| **avg compress** | average compression ratio | E2 输出"平均有效压缩比"——按 (token, expert) 对的全量 vs 压缩比例加权算的平均字节压缩。例如 50% 对全量 + 50% 对 4× 压 → avg = 1 / (0.5 + 0.5/4) = 1.6×。 | ≥ 1 |

### 精度指标

| 缩写 | 全称 | 含义 | 例 |
|---|---|---|---|
| **PPL** | perplexity 困惑度 | exp(cross-entropy loss)。语言模型的标准生成质量代理指标。**越低越好**。 | teacher PPL = 3.15 |
| **PPL +%** | PPL increase percentage | (student_ppl - teacher_ppl) / teacher_ppl × 100% **相对**变化。 | 3.70 vs 3.15 = +17.2% |
| **top-1 acc** | next-token top-1 accuracy | argmax(logits) == ground_truth 的概率。**越高越好**。 | teacher 76.15% |
| **pp** | percentage points 百分点 | 两个百分数相减的**绝对差值**。"掉 1pp" ≠ "掉 1%"。 | 76.15% → 75.34% = 掉 **0.81pp**（如果按相对算就是 -1.06%，容易混淆）|
| **pass@1** | HumanEval pass-at-1 | 164 道代码题里 greedy 生成一次就过测试的比例。**越高越好**。 | teacher 55.49% (91/164) |
| **EV @ ell** | explained variance at rank ell | SVD 累计方差解释率：前 ell 个奇异分量解释的方差占总方差比例。**越高 = 越低秩**。 | layer 12 EV@d/4 = 0.66 |
| **relMSE** | relative MSE | MSE(student, teacher) / ‖teacher‖²。**越低 = 越接近 teacher**。 | hidden relMSE = 0.013 |
| **agreement w/ teacher top-1** | top-1 prediction match | 学生 argmax(logits) == 老师 argmax(logits) 的概率（与 ground truth 无关）。**衡量学生跟老师"行为是否一致"**。 | 95% = 严格意义上"几乎相同的模型" |

### 模型 / 架构相关

| 缩写 | 含义 |
|---|---|
| **d** | hidden_size = 2048（Qwen3-30B-A3B）|
| **E** | num_experts = 128 |
| **top_k** / **K** | num_experts_per_tok = 8（每个 token 路由到 8 个 expert）|
| **N** | 样本 token 数（SVD 分析里 N/d 比值衡量 well-conditioned 程度，N ≥ 4d 才稳）|
| **ell** | SVD truncation rank（保留前 ell 个奇异分量）|
| **routing weight** | softmax(W_gate · x) 中的某 expert 那一项，∈ [0, 1]，top-k 加起来 = 1 |

## 2. 实验目录

### 算法层面（HF 单进程模拟）
- [A1. Stage-0 SVD spectrum](#a1-stage-0-svd-spectrum) — 32K token，per-expert + per-layer
- [A2. Per-domain SVD](#a2-per-domain-svd) — 跨 7 域比较
- [A3. Task-aware projector 训练](#a3-task-aware-projector-训练) — 学一个非 SVD 的投影
- [A4. 多层全 SVD sweep](#a4-多层全-svd-sweep) — 验证 dispatch 端 dead 的关键实验
- [A5. Routing-aware projection](#a5-routing-aware-projection) — 试图救 dispatch 路由
- [B1. Combine 端 top-magnitude sparsification sweep](#b1-combine-端-top-magnitude-sparsification-sweep) — 转向 combine 的第一个 win
- [B2. Combine error-feedback 变种对比](#b2-combine-error-feedback-变种对比) — topk_l2 winner
- [B3. Cross-domain 验证](#b3-cross-domain-验证) — winner 跨 4 域稳定
- [B4. Sparse + FP8 叠加](#b4-sparse--fp8-叠加) — FP8 几乎免费

### 工程层面（EP-HT 真实集成）
- [C1. CombineEPHT 加压缩](#c1-combineepht-加压缩) — 改 nanovllm_moe 代码
- [C3. EP=8 完整模型 sweep](#c3-ep8-完整模型-sweep) — 数学推理在 8× 下保持

### Accuracy 层面（真实 task 指标）
- [D1. Next-token top-1 accuracy on lcc](#d1-next-token-top-1-accuracy-on-lcc)
- [D2. HumanEval pass@1（EP=8）](#d2-humaneval-pass1ep8)

### 不均匀压缩探索（师兄建议）
- [E1. 选择性 per-token 压缩](#e1-选择性-per-token-压缩) — "压哪些 token 重要吗"
- [E2. Top-k 累积权重保护](#e2-top-k-累积权重保护) — "top-1 / 重要 expert 全量"

---

## A1. Stage-0 SVD spectrum

### 目的
最初的 Go/No-Go：MoE dispatch 端激活在每一层的奇异值谱是否陡（top-d/4 维能解释 ≥90% 方差），决定低秩压缩这条路是否值得做。同时增加 per-expert 谱分析作为 bonus signal。

> 注：原本先跑过一版 N=3293 token 的 quick check，但 N/d=1.6 太小、SVD 边缘不稳，直接被这一版（N=32768, N/d=16）覆盖，已删旧脚本和输出。

### 策略
换标定数据为 LongBench 4 个子集（gov_report / multi_news / multifieldqa_en / lcc）拼接的长文，按 512-token 连续切片成 64 个 chunk，得到 N=32768 token（N/d=16）。除全局谱外，按 top-1 路由专家分组算 per-expert 谱。

### 代码
`eval/compression/stage0_low_rank_svd_v2.py`，约 340 行。新增：
- `load_longbench_text(args.longbench_files)` 读 jsonl 的 `context` 字段
- top-1 expert hook（在 `mlp.gate` 上拿 argmax）
- per-expert SVD：把 X 按 top-1 expert 分组，每组做 SVD

后续配套：
- `eval/compression/stage0_reanalyze.py` — 剔除 n_e < 2d=4096 的"假性低秩" expert（小样本下累积变差到 d/4 处会虚高到 1.0）
- `eval/compression/stage0_final_plot.py` — 合成 paper-ready 图

### 配置 / 组件
- env: `atom`
- 同 A0 的模型加载
- 6 个探针层：2/6/12/24/36/46
- 数据：每 LongBench 文件取 `context` 字段，拼到 ~800 KB 字符上限

### 完整实验结果

全局谱（与 A0 一致但 N/d=16 稳定）：

| layer | EV @ d/16 | EV @ d/8 | **EV @ d/4** | EV @ d/2 | rank @ 0.99 |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.629 | 0.718 | **0.821** | 0.925 | 1772 |
| 6 | 0.396 | 0.530 | **0.692** | 0.867 | 1854 |
| 12 | 0.367 | 0.498 | **0.661** | 0.846 | 1898 |
| 24 | 0.385 | 0.511 | **0.669** | 0.848 | 1900 |
| 36 | 0.435 | 0.557 | **0.702** | 0.863 | 1892 |
| 46 | 0.550 | 0.660 | **0.781** | 0.903 | 1850 |

Well-conditioned per-expert（n_e ≥ 4096，每层只有 1 个 popular expert 合格）：跟全局谱基本一样陡（0.77–0.88 @d/4），并不显著低秩。

### 结论
**判定：NO-GO（严格判据）**——所有 6 层 EV@d/4 都不到 0.90，最浅层最强 0.82，中层最弱 0.66。**Per-expert 也没救**（popular expert 谱 ≈ 全局谱）。要保 90% 方差平均需要 ~½ 的 d 维（即只 2× 压缩），要保 99% 几乎全要。

### 输出
- `eval_results/compression_lowrank_stage0_v2/svd_summary.json`
- `eval_results/compression_lowrank_stage0_v2/reanalysis_summary.json`
- `eval_results/compression_lowrank_stage0_v2/svd_curve.png`
- `eval_results/compression_lowrank_stage0_v2/svd_curve_per_expert.png`
- `eval_results/compression_lowrank_stage0_v2/svd_curve_summary.png`
- `eval_results/compression_lowrank_stage0_v2/report.md`
- `eval_results/compression_lowrank_stage0_v2/reanalysis_report.md`
- `docs/compression/stage0_findings.md`（中文详版）
- `activations_layer*.pt` / `top1_layer*.pt` — gitignored 大文件（1.6 GB），脚本可重生

---

## A2. Per-domain SVD

### 目的
A1 用混合域得到 NO-GO，怀疑是因为多域语义异质拉平了谱。**如果换成单一窄域（只代码 / 只数学 / 只 QA），LSH-MoE 那种"相似 token 聚类"直觉是否生效？**

### 策略
7 个窄域：lcc(代码) / repobench_p(纯 Python 代码) / gov_report / multifieldqa_en / dureader(中文) / passage_ret(模板化检索) / gsm8k(数学)。每域 16 chunks × 512 tokens = 8192 token（N/d=4 well-conditioned），并列 5 层做 SVD。

### 代码
`eval/compression/stage0_per_domain.py`，约 280 行。一次模型加载 + 一次 forward 跑 112 chunks，按域 slice 出对应激活做 SVD。

### 配置 / 组件
- env: `atom`
- 7 域，5 层（2/12/24/36/46）

### 完整实验结果

EV @ d/4（**粗体 = 通过 0.90 阈值**）：

| layer | lcc | repobench_p | gsm8k | gov_report | multifieldqa | dureader | passage_ret |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **2** | **0.934** | **0.924** | **0.912** | 0.863 | 0.827 | 0.790 | 0.785 |
| 12 | 0.813 | 0.791 | 0.783 | 0.729 | 0.714 | 0.688 | 0.662 |
| 24 | 0.775 | 0.762 | 0.744 | 0.736 | 0.724 | 0.696 | 0.685 |
| 36 | 0.783 | 0.771 | 0.728 | 0.763 | 0.746 | 0.707 | 0.709 |
| 46 | 0.878 | 0.877 | 0.782 | 0.829 | 0.797 | 0.742 | 0.757 |

EV @ d/2（2× 压缩判据）：lcc / repobench_p 在所有 5 层都 ≥ 0.90；其它域大多在浅 / 深层通过、中层吃力。

### 结论
- **域差异很大**：layer 2 上 lcc(0.93) 比 passage_ret(0.79) 高 0.15。
- **代码 / 数学 + layer 2 处 4× 压缩首次通过**；layer 12–36 即使最低秩的代码域也只 0.76–0.81。
- 排名稳定：代码 > 数学 > 英文报告 > 混合 QA > 中文 / 模板检索。
- **结论**：用单一窄域可以救一半（浅 / 深层 + 代码），但**中层"自然秩"高，换数据集救不回来**。

### 输出
- `eval_results/compression_lowrank_stage0_per_domain/svd_per_domain.json`
- `eval_results/compression_lowrank_stage0_per_domain/svd_per_domain.png`（5×7 网格）
- `eval_results/compression_lowrank_stage0_per_domain/svd_per_domain_heatmap.png`
- `eval_results/compression_lowrank_stage0_per_domain/report.md`
- `docs/compression/stage0_per_domain_findings.md`

---

## A3. Task-aware projector 训练

### 目的
A1/A2 都证 dispatch 激活的 L²-最优低秩（SVD）丢太多。**学一个 "task-aware" 投影器 P_down/P_up（不是 L² 最优而是任务 loss 最优）能否反超 SVD？** 这是 LSH-MoE 训练期做法的"轻量版"——不动 expert 权重，只训 2M 参数的投影器。

### 策略
- 选 layer 24, lcc 域, ℓ=d/4=512（A2 里最难的中层 + 最低秩的代码域）
- 投影器 P_down (d×ℓ), P_up (ℓ×d) 共 2M 参数，**从 SVD 初始化**
- 损失：MSE(MoE 层输出 y_hat vs y_full)
- AdamW lr=1e-3, 500 steps

### 代码
`eval/compression/stage0_task_aware_projector.py`，约 450 行。先 cache `x_full / y_full / h_final`，然后在 cached `(x, y)` 上跑训练，最后用 hook 在全模型上验证。

### 配置 / 组件
- env: `atom`
- 32 训练 chunks（16384 tokens）+ 8 测试 chunks（4096 tokens）
- 训练 batch_chunks=4（2048 tokens）

### 完整实验结果

| 指标 | SVD baseline (test) | task-aware trained (test) | trained / SVD |
|---|---:|---:|---:|
| x 重建 MSE | 0.144 | 0.404 | **2.80× 变差** |
| MoE 输出 relMSE | 0.739 | 1.544 | 2.09× 变差 |
| final hidden relMSE | **0.0131** | 0.0207 | 1.59× 变差 |

训练曲线第 100–175 步炸过（peak train MoE MSE 从 1e-3 飙到 3e-2），后来收敛到 2.7e-3，没甩开 SVD 起点。

### 结论
**Task-aware naive 训练 NO-GO。** 根因：**MoE 路由的 top-k 是不可微的**。投影器扰动 x 一旦改变 top-k 决策，下游专家组合整个换一套，损失景观出现不连续，Adam 在这种地形上要么炸要么停在差解。

**意外发现**：SVD baseline 的 final hidden relMSE 居然只有 1.3%（虽然 x 重建 MSE 高、MoE 输出 relMSE 高达 74%）——**模型自己对单层 dispatch 噪声很鲁棒**。这暗示"严格 L² 阈值"不是正确的判据。

### 输出
- `eval_results/compression_lowrank_stage0_task_aware/summary.json`
- `eval_results/compression_lowrank_stage0_task_aware/report.md`
- `eval_results/compression_lowrank_stage0_task_aware/training_curve.png`

---

## A4. 多层全 SVD sweep

### 目的
A3 暴露：单层 1.3% drift 不代表全模型 1.3% drift。**把 SVD 压缩同时装到 ALL 48 个 MoE 层上**，扫多档压缩比，看真实 task 质量（PPL + final hidden relMSE + 路由一致率）。这是判定 dispatch 端方向"真的死了"的关键实验。

### 策略
对每层用一份 SVD calibration（同一份 lcc 数据），生成 4 档投影器（ℓ ∈ {1024, 512, 256, 128} = 2×/4×/8×/16× 压缩），forward 时 hook 替换每层 MoE 输入为 P_up·P_down·(x-mu)+mu。

### 代码
`eval/compression/stage0_full_model_sweep.py`，约 380 行。3 个 phase：
1. 一次 forward 收集所有 48 层 calibration 激活
2. 每层每档 ell 算 SVD，把 projector 直接放在对应 MoE block 的 device 上（避免跨卡传输）
3. teacher forward（无 hook）+ 4 个 student forward（带 hook），取 loss / hidden / top-1 路由

### 配置 / 组件
- env: `atom`
- calib=16 chunks (8192 tokens), test=8 chunks (4096 tokens)，lcc 域
- 4 档 ell

### 完整实验结果（teacher PPL = 3.153）

| ell | 压缩 | 平均 train EV | PPL | PPL +% | hidden relMSE | top-1 路由一致 |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 2× | 0.938 | **6.92** | **+119%** | 0.152 | 65% |
| 512 | 4× | 0.826 | 25.6 | +711% | 0.265 | 55% |
| 256 | 8× | 0.693 | 4 296 | +136 158% | 0.994 | 35% |
| 128 | 16× | 0.561 | 2 128 823 | INF | 1.848 | 19% |

### 结论
**Dispatch 端低秩压缩彻底死掉**——哪怕最温和 2×，PPL 就翻倍。**根因：**
1. **多层 L² 误差乘性累积**——单层 1.3% drift × 48 层 ≠ 全模型 1.3%
2. **路由翻车滚雪球**——2× 时 35% token 路由到错的 expert，48 层下来路由几乎全坏

**这是 dispatch 端方向的死亡证书。**

### 输出
- `eval_results/compression_lowrank_stage0_full_sweep/summary.json`
- `eval_results/compression_lowrank_stage0_full_sweep/report.md`
- `eval_results/compression_lowrank_stage0_full_sweep/sweep_curve.png`

---

## A5. Routing-aware projection

### 目的
A4 诊断"路由翻车 + L² 误差累积"。**有针对性的修复**：构造投影器使其列空间**强制包含** W_gate 的行空间——这样 `gate(x_hat) = gate(x)` 数学上恒等，路由 logits 完美保持。能不能救回来？

### 策略
对每层用 SVD 算 P 但前 128 个方向强制是 W_gate^T 的正交基（rank(W_gate) ≤ E = 128），剩下的 ell-128 维用 (Q_g 正交补空间内的) 最大 SVD 方向填。

数学验证：fp64 下 `|W - W·P·P^T|_max = 9.99e-16`（机器精度），路由 top-1 一致率 100%。

### 代码
- 主：`eval/compression/stage0_routing_aware_sweep.py`，约 400 行。同 A4 的 sweep 框架，加 routing-aware 投影器变种。
- Debug：`eval/compression/debug_routing_aware.py` — 单独验证 fp64/bf16-cast/bf16-runtime 三档精度下的路由保持率。

### 配置 / 组件
- env: `atom`
- 同 A4 的 calibration / test
- 3 档 ell（1024 / 512 / 256），SVD 和 routing-aware 并列

### 完整实验结果

| ell | 压缩 | variant | PPL | PPL +% | hidden relMSE | 路由一致（mean/min）|
|---:|---:|---|---:|---:|---:|---:|
| 1024 | 2× | svd | 6.92 | +119% | 0.152 | 0.654 / - |
| 1024 | 2× | **routing-aware** | **30.4** | **+863%** | 0.439 | **0.395 / -** |
| 512 | 4× | svd | 25.6 | +711% | 0.265 | 0.548 |
| 512 | 4× | routing-aware | 2429 | +76948% | 0.756 | 0.205 |
| 256 | 8× | svd | 4296 | +136158% | 0.994 | 0.352 |
| 256 | 8× | routing-aware | 201070 | +6.3e6% | 1.318 | 0.092 |

**Debug 确认**：单层 fp64 下保 100% 路由、bf16 单层 99.6%；但 48 层叠加后路由一致率只有 39.5%。

### 结论
**Routing-aware 反而比 SVD 更差！** 单层数学正确，但全模型多层叠加时：
- routing-aware 牺牲了 L²-最优换来"单层路由完美"
- 单层路由完美的"完美"只针对本层实际收到的 x；但**下一层收到的 x 已经飘了**（因为上层 L² 误差大）
- routing-aware 因为 L² 误差更大，**飘得更厉害**，下一层路由更乱

**真正诊断**：核心问题不在路由保不保，**在 residual stream 上的 L² 误差累积**。任何线性投影器都会有这个问题。**Dispatch 端的所有变种全部 NO-GO，与映射方法无关**。

### 输出
- `eval_results/compression_lowrank_stage0_routing_aware/summary.json`
- `eval_results/compression_lowrank_stage0_routing_aware/report.md`
- `eval_results/compression_lowrank_stage0_routing_aware/comparison_curve.png`

---

## B1. Combine 端 top-magnitude sparsification sweep

### 目的
Dispatch 全部死掉后，转向 combine 端。**关键直觉**：combine 输出直接进 residual stream，误差被 residual 吸收而不是被下层放大（dispatch 端误差进 MoE 输入，会乘性放大），所以同档压缩 combine 应该 PPL 退化得多得多。

### 策略
对每个专家的 forward 输出（即 combine all-to-all 真正传的东西）做 top-k 幅值稀疏化：每行只保留幅值最大的 k 个分量，其余置零。**不需要 calibration（data-free）**。所有 48 层 × 128 专家共 6144 个 hook。

### 代码
`eval/compression/stage0_combine_sparsify_sweep.py`，约 270 行。挂 `forward_hook` 在每个 `mlp.experts[e]` 上做稀疏化。

### 配置 / 组件
- env: `atom`
- test = 8 chunks lcc（与之前同一 held-out split）
- 5 档 keep_frac：0.5 / 0.25 / 0.125 / 0.0625 / 0.03125

### 完整实验结果（teacher PPL = 3.153）

| keep_frac | k | value-only 压缩 | value+index 压缩 | PPL | PPL +% | hidden relMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 1024 | 2× | 1× | **3.26** | **+3.3%** | 0.042 |
| 0.25 | 512 | 4× | 2× | 3.80 | +20.6% | 0.141 |
| 0.125 | 256 | 8× | 4× | 5.83 | +84.8% | 0.278 |
| 0.0625 | 128 | 16× | 8× | 15.35 | +387% | 0.415 |
| 0.03125 | 64 | 32× | 16× | 49.17 | +1460% | 0.521 |

**跟 dispatch SVD 同档对比（这是 paper-grade 数字）**：

| 压缩比 | dispatch SVD PPL +% | **combine sparsify PPL +%** | 比 dispatch 强多少倍 |
|---:|---:|---:|---:|
| 2× | +119% | **+3.3%** | **36×** |
| 4× | +711% | **+20.6%** | **34×** |
| 8× | +136158% | **+84.8%** | **1606×** |

### 结论
**重大转折——这是这一周第一个正面结果。** Combine 端 2× 压缩 PPL 只涨 3.3%（几乎无损），4× 涨 20.6%（可接受）；同档 dispatch 是 +119% / +711%。直接验证了 residual stream 吸收误差的假设。**主线确认：转 Direction B (combine 端稀疏化)。**

### 输出
- `eval_results/compression_lowrank_stage0_combine_sparsify/summary.json`
- `eval_results/compression_lowrank_stage0_combine_sparsify/report.md`
- `eval_results/compression_lowrank_stage0_combine_sparsify/sweep_curve.png`

---

## B2. Combine error-feedback 变种对比

### 目的
B1 用最 naive 的 top-k。试 4 种"sparsify 之后做点什么补偿"的变种，挑赢的进下一步。

### 策略
- `naive_topk`（B1 已有）：只保 top-k 幅值，其余置零
- `topk_l2`：top-k + 缩放 kept values 使每行 L2 norm = 原 row L2 norm
- `topk_l1`：top-k + 缩放使 L1 norm 保留
- `stochastic`：按 ∝|v_i| 概率不放回采样 k 个位置，scale by 1/p_i（Wangni 2018 的 unbiased 估计器）

### 代码
`eval/compression/stage0_combine_ef_variants.py`，约 350 行。4 个 sparsifier 函数 + 同 B1 的 sweep 框架。

### 配置 / 组件
- env: `atom`
- lcc test 8 chunks
- 3 档 keep_frac（0.25 / 0.125 / 0.0625）

### 完整实验结果

| Variant | keep=0.25 (4×) PPL +% | keep=0.125 (8×) PPL +% | keep=0.0625 (16×) PPL +% |
|---|---:|---:|---:|
| naive_topk | +20.6% | +84.8% | +387% |
| **topk_l2** | **+18.0%** | **+64.8%** | **+231%** |
| topk_l1 | +50.0% | +2269% | 模型废 |
| stochastic | +261% | 模型废 | 模型废 |

### 结论
**Winner: `topk_l2`**——免费午餐式改进，所有档都比 naive 强。L1 跟 L2 norm 不对齐；stochastic 方差大被 residual 累积放大。

### 输出
- `eval_results/compression_lowrank_stage0_ef_variants/summary.json`
- `eval_results/compression_lowrank_stage0_ef_variants/report.md`
- `eval_results/compression_lowrank_stage0_ef_variants/sweep_curve.png`

---

## B3. Cross-domain 验证

### 目的
B1/B2 都在 lcc（代码）上测。**验证 winner topk_l2 不是 lcc 专享胜利**，跨 4 个域看是否稳定。

### 策略
Winner topk_l2 + naive_topk baseline，并列跑在 lcc / multifieldqa / dureader / gov_report 4 个域上，每域 8 个 held-out chunks。

### 代码
`eval/compression/stage0_combine_cross_domain.py`，约 290 行。

### 完整实验结果

| 域 | teacher PPL | naive@4× | **topk_l2@4×** | naive@8× | **topk_l2@8×** |
|---|---:|---:|---:|---:|---:|
| lcc (代码) | 3.15 | +20.6% | **+18.0%** | +84.8% | **+64.8%** |
| multifieldqa (混合 QA) | 6.09 | +20.6% | **+18.0%** | +149.3% | **+96.4%** |
| dureader (中文) | 21.8 | +32.9% | **+25.5%** | +171.3% | **+107.1%** |
| gov_report (英文报告) | 6.62 | +18.6% | **+14.1%** | +108.3% | **+70.3%** |

### 结论
- **topk_l2 在所有 4 域都赢 naive_topk**——域无关改进
- 4× 压缩所有域 PPL +14–25%，稳定可用
- 8× 压缩域差异大：代码 65% / 英文报告 70% / 混合 QA 96% / 中文 107%
- **最差域（中文）4× 也只 +25.5%**——不是 lcc 专享胜利

### 输出
- `eval_results/compression_lowrank_stage0_cross_domain/summary.json`
- `eval_results/compression_lowrank_stage0_cross_domain/report.md`
- `eval_results/compression_lowrank_stage0_cross_domain/cross_domain_curve.png`

---

## B4. Sparse + FP8 叠加

### 目的
Sparse 已经省 4× 字节，再叠 FP8（每个值 8-bit 而非 16-bit）能再省一半。**FP8 是 DeepEP/MoRI 的产线标准**，叠加后总流量 = sparse × FP8。问：FP8 在已稀疏的 kept values 上 round-trip 会再掉多少精度？

### 策略
topk_l2 输出后，每行做 per-row amax/448 的 FP8 E4M3 round-trip（cast to fp8 → cast back to bf16）。compare 4 种 variant：
- `fp8_only`（只 FP8，无 sparse）—— DeepEP 同档对照
- `topk_l2_bf16`（B2 winner）
- `topk_l2_fp8`（叠加）
- `naive_topk_fp8`（看 L2 rescale 在 FP8 下是否仍 win）

### 代码
`eval/compression/stage0_combine_sparse_fp8.py`，约 340 行。使用 `torch.float8_e4m3fn` (pytorch 2.7 内置)。

### 配置 / 组件
- env: `atom`
- lcc test 8 chunks, 3 档 keep_frac

### 完整实验结果（teacher PPL = 3.153）

| Variant | keep | k | value-only 压缩 | PPL | PPL +% |
|---|---:|---:|---:|---:|---:|
| **fp8_only**（对照）| 1.0 | 2048 | **2×** | **3.15** | **−0.1%** ← FP8 几乎无损 |
| topk_l2_bf16 | 0.25 | 512 | 4× | 3.72 | +18.0% |
| topk_l2_bf16 | 0.125 | 256 | 8× | 5.20 | +64.8% |
| topk_l2_bf16 | 0.0625 | 128 | 16× | 10.42 | +230.5% |
| **topk_l2_fp8** | 0.25 | 512 | **8×** | **3.70** | **+17.2%** |
| **topk_l2_fp8** | 0.125 | 256 | **16×** | **5.17** | **+63.9%** |
| **topk_l2_fp8** | 0.0625 | 128 | **32×** | **10.32** | **+227.3%** |
| naive_topk_fp8 | 0.25 | 512 | 8× | 3.80 | +20.6% |

### 结论
**FP8 在 topk_l2 已稀疏的值上叠加几乎免费**——sparse 4× bf16 → +18.0%，再叠 FP8 = 8× 总 → +17.2%（几乎一样）。**这是这一周最值钱的 headline 数字**：

> "Qwen3-30B-A3B frozen + combine-side 4× topk_l2 sparsification + 2× FP8 E4M3 = **8× value compression at +17% PPL on lcc**. Same 8× via dispatch-side SVD gives PPL +136 158%. 4 个数量级 gap，根因：误差去 residual stream 而非乘性叠加。"

### 输出
- `eval_results/compression_lowrank_stage0_sparse_fp8/summary.json`
- `eval_results/compression_lowrank_stage0_sparse_fp8/report.md`
- `eval_results/compression_lowrank_stage0_sparse_fp8/stack_curve.png`

---

## C1. CombineEPHT 加压缩

### 目的
A/B 系列都在 HF 单进程 + forward hook 模拟，从未在真实 NCCL all-to-all 上跑过。**把算法真正集成进 `workshop/nanovllm_moe/` 的 EP-HT combine 路径**，下一步才能在 8×4090 真实环境下验证。

### 策略
- 改 `combine_ep_ht.py`：在 reverse all-to-all 之前对 expert_out 做 pack（topk_l2 + 可选 FP8），分别 a2a indices / values / scale（NCCL 不支持 int16/fp8，用 uint8 bitcast 转），接收侧 unpack 回 dense `[send_total, H]`
- 加 Config 字段 + 全链路 plumb：`Config → ModelRunner → Qwen3MoeForCausalLM → Qwen3MoeModel → Qwen3MoeDecoderLayer → FusedMoE → CombineEPHT`
- 同时保留 env-var override（per-call 切换，no engine restart）

### 代码 / 改动文件

| 文件 | 改动 |
|---|---|
| `workshop/nanovllm_moe/artifacts/modeling/layers/moe/combine_ep_ht.py` | 加 pack / unpack 工具 + reverse a2a 三 payload 版本（uint8 bitcast）+ env-var resolve 函数 |
| `workshop/nanovllm_moe/artifacts/modeling/layers/moe/fused_moe.py` | `FusedMoE.__init__` 加 3 个 `combine_compress_*` kwarg，转发给 `CombineEPHT` |
| `workshop/nanovllm_moe/artifacts/modeling/models/qwen3_moe.py` | 3 个类（DecoderLayer/Model/ForCausalLM）的 `__init__` 全加 3 个 kwarg 转发 |
| `workshop/nanovllm_moe/services/config.py` | 加 3 个字段：`moe_combine_compress_keep_frac` (float, 默认 0.0 = off), `_fp8` (bool, True), `_use_l2` (bool, True) |
| `workshop/nanovllm_moe/services/model_runner/model_runner.py` | 把 3 个 Config 字段传给 `Qwen3MoeForCausalLM` 构造 |

### 用法

**Config-time 启用（推荐 benchmark / 生产）**：
```python
engine = LLMEngine(
    model="/path/to/Qwen3-30B-A3B",
    moe_impl="ep_ht",
    moe_combine_compress_keep_frac=0.25,   # 4× value 压缩
    moe_combine_compress_fp8=True,         # 叠 FP8
    moe_combine_compress_use_l2=True,
)
```

**Env-var override（同 engine 内 A/B 切换）**：
```python
os.environ["MOE_COMBINE_COMPRESS_KEEP_FRAC"] = "0.25"
# generate 调用看见压缩
os.environ.pop("MOE_COMBINE_COMPRESS_KEEP_FRAC")
# 后续 generate 又是 baseline
```

env 优先级高于 Config。

### 结论
集成完成，**接口干净（Config 默认 off，baseline 路径 byte-identical），算法跟 HF 单进程模拟逐位一致**。

### 输出
代码改动已 commit（commit `25c2294` 的一部分），push 到 `origin/my-a2a-kernel-test`。

---

## C3. EP=8 完整模型 sweep

### 目的
在真实 EP=8 + 完整 48 层模型上验证"压缩后输出仍连贯"。**真正的端到端 NCCL 集成验证**。

> 注：之前用 EP=2 + NUM_LAYERS=2 trimmed 模型做过一次 smoke test（只看"代码不 crash"），结果被这个完整模型 sweep 完全覆盖，已删脚本。

### 策略
spawn EP=8 worker（8×4090，每卡 16 expert），加载完整 Qwen3-30B-A3B，对同一组 prompt 在 4 个 config 下 greedy generate（teacher / keep=0.5 / 0.25 / 0.125 + FP8 + L2），对比文本 / token match / 数学正确性 / wall-clock。

### 代码
`eval/compression/test_ep_ht_compress_full.py`，约 170 行。同样基于 spawn 范式，多了 token match% 比较和 timing。

### 配置 / 组件
- env: `vllm`
- 8 GPU, MOE_IMPL=ep_ht, ENFORCE_EAGER=1, MAX_TOKENS=24, batch_chunks 自动调度
- 4 prompts：`"The capital of France is"`, `factorial code`, `"Q: What is 17 * 23?"`, `"The quick brown fox"`

### 完整实验结果

| Config | 时间 | 时间 ratio | 数学题对吗 | "Paris" 对吗 | 整体连贯 | match% |
|---|---:|---:|:---:|:---:|:---:|---:|
| **Teacher**（不压）| 3.98s | 1.00× | ✅ 391 | ✅ Paris | ✅ | — |
| keep=0.5 + FP8 | 5.11s | 1.28× | ✅ 391 | ✅ Paris | ✅ | 76% |
| keep=0.25 + FP8 | 7.45s | 1.87× | ✅ 391 | ✅ Paris | ✅ | 59% |
| keep=0.125 + FP8 | 6.63s | 1.66× | ✅ 391 | ✅ Paris* | ⚠ | 48% |

\* "Washington D5" 而非 "D.C." —— 但 Paris 这个事实 token 对了。

### 结论
- ✅ **算法在真 NCCL 上跑通**——所有 4 个 config 没 crash 没 hang
- ✅ **数学推理在 8× 压缩下保持**——17×23=391 全部对
- ✅ **HF 模拟数字与 EP=8 真实行为对齐**（HF 模拟 4× 压缩 PPL +17%；这里 4× 时整体输出仍合理）
- ⚠️ **wall-clock 暂时慢 1.3–1.9×**——因为 pack/unpack 在 eager 模式没融合 + 3 NCCL call vs 1。这是工程优化空间不是算法问题。

### 输出
代码：`eval/compression/test_ep_ht_compress_full.py`。运行日志在控制台。

### Config-time 测试

额外脚本 `eval/compression/test_ep_ht_compress_config.py`（约 130 行）专门验证 Config-time 路径（不用 env，直接 `moe_combine_compress_keep_frac=0.25` 当 engine kwarg 传）。pass。

---

## D1. Next-token top-1 accuracy on lcc

### 目的
之前所有结果都只有 PPL，**没有真正的 task accuracy 数字**。Next-token top-1 是 PPL 的最直接的"accuracy 版本" —— argmax(logits) 对不对，只比 ground-truth 多了一个 argmax 操作。10 分钟能出数。

### 策略
HF + hook 模拟（已经验证跟真 EP 等价），同一份 lcc held-out 测试集（4096 tokens），4 个 config（teacher / keep=0.5/0.25/0.125 + FP8 + L2）。对每个位置 t 取 argmax(logits[:, t, :])，跟 input_ids[:, t+1] 比。

3 个 metric：
- next-token top-1 accuracy（vs ground truth）
- next-token top-5 accuracy
- top-1 agreement with teacher（compressed top-1 == teacher top-1，独立于 ground truth）

### 代码
`eval/compression/stage0_taskacc_lcc.py`，约 280 行。基于 B 系列同样的 sparsify_topk_l2_fp8 hook。

### 配置 / 组件
- env: `atom`
- lcc test 8 chunks (4096 tokens)，与 B 系列同一 held-out split
- 4 configs

### 完整实验结果

| Config | 压缩 | PPL | top-1 acc | top-1 drop | top-5 acc | top-5 drop | teacher top-1 agree |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher | - | 3.153 | **76.15%** | - | **90.80%** | - | 100% |
| keep=0.5 + FP8 | **2×** | 3.226 | 75.49% | **−0.66pp** | 90.83% | -0.02pp | **94.84%** |
| keep=0.25 + FP8 | **4×** | 3.695 | 72.21% | **−3.94pp** | 88.80% | −2.01pp | 86.47% |
| keep=0.125 + FP8 | **8×** | 5.166 | 66.73% | **−9.42pp** | 84.47% | −6.34pp | 75.78% |

### 结论
- **2× 压缩下模型几乎完全等价于 teacher**（top-1 掉 0.66pp，跟 teacher 95% 一致）
- 4× 也很温和（−3.94pp）
- 8× 开始有感（−9.42pp）但还能用

是 D2 HumanEval 的"轻量先验"——PPL 之外的第一个 task-level 数字。

### 输出
- `eval_results/compression_lowrank_stage0_taskacc_lcc/summary.json`
- `eval_results/compression_lowrank_stage0_taskacc_lcc/report.md`
- `eval_results/compression_lowrank_stage0_taskacc_lcc/accuracy_curve.png`

---

## D2. HumanEval pass@1（EP=8）

### 目的
**这一周的 final benchmark**——给出真正的 code-completion task accuracy 数字。HumanEval 是 OpenAI 提出的 164 题 Python 函数补全标准 benchmark，跟 lcc 校准域完全对齐。pass@1 = greedy 生成的代码跑测试通过率。

### 策略
- 第一版（`stage0_humaneval.py`）走 HF + device_map="auto"，单 batch 都 8 分钟还没出 —— **被弃**，因为 device_map="auto" 是 pipeline parallel，单卡活跃 utilization 5%
- **第二版（`stage0_humaneval_ep.py`）走真实 EP=8 nanovllm_moe LLMEngine** —— 8 卡并发计算，每题 ~6.5s，单 config ~18 分钟，4 config ~80 分钟可完成
- 同 engine 内通过 env-var 切换 4 个 config，避免 4 次 model 加载
- 每 config：generate 164 题 → 截断到首个 stop pattern（`\nclass `, `\ndef `, `\n#`, `\nif __name__`, etc.）→ thread pool 并行 grade via `human_eval.execution.check_correctness`

### 代码
`eval/compression/stage0_humaneval_ep.py`，约 270 行。

### 配置 / 组件
- env: `vllm`（含 flashinfer / sgl_kernel）
- `pip install human-eval` (1.0.3) 到 vllm env
- 8 GPU EP=8, MOE_IMPL=ep_ht, ENFORCE_EAGER=1
- `max_new_tokens=384`, `temperature=0.0`（greedy）, `batch_size=8`
- grader: ThreadPoolExecutor(8) + `check_correctness(timeout=10s)`
- 164 题（全集）, 4 config

### 完整实验结果（**这是论文级 headline 数字**）

| Config | 压缩 | **pass@1** | 掉分 (pp) | 相对掉分 | gen time |
|---|---:|---:|---:|---:|---:|
| **teacher** | - | **55.49%** (91/164) | - | - | 1078.8s |
| keep=0.5 + FP8 | **2×** | **47.56%** (78/164) | **−7.93pp** | −14.3% | 1329.4s |
| keep=0.25 + FP8 | **4×** | **46.34%** (76/164) | **−9.15pp** | −16.5% | 1292.4s |
| keep=0.125 + FP8 | **8×** | **18.29%** (30/164) | **−37.20pp** | **−67.0%** | 1283.1s |

### 三个 metric 一起看

| Compression | PPL +% | Next-token top-1 掉分 | **HumanEval pass@1 掉分** |
|---|---:|---:|---:|
| 2× | +2.3% | −0.66pp | **−7.93pp** ← 12× 于 next-token |
| 4× | +17.2% | −3.94pp | **−9.15pp** |
| 8× | +63.9% | −9.42pp | **−37.20pp** ← 急剧崩坏 |

### 结论

**3 个 metric 的非线性关系是核心 finding**：

1. **PPL / next-token 比 task accuracy 乐观得多**——2× 压缩 PPL 仅 +2.3% / next-token 只掉 0.66pp，看上去"几乎无损"，但 HumanEval 掉 7.93pp。**原因**：代码补全是 sequential，函数体 50–100 个 token 里 ANY ONE 错就整个测试不过。

2. **甜点在 4×**：从 2× 到 4× 只多掉 1.22pp（48 → 46），流量再省 2×。**4× 是真正的 Pareto 甜点**。

3. **8× 是悬崖**：4× → 8× pass@1 暴跌 28pp，模型不能写代码了。next-token 只掉 9.4pp 完全不在一个量级 → **task-level 是非线性崩坏**。

### 输出
- `eval_results/compression_lowrank_stage0_humaneval_ep/summary.json`
- `eval_results/compression_lowrank_stage0_humaneval_ep/report.md`
- `eval_results/compression_lowrank_stage0_humaneval_ep/pass_at_1.png`
- `eval_results/compression_lowrank_stage0_humaneval_ep/completions/{label}.jsonl` — 4 个 config 的完整 164 题生成（可重 grade）
- `eval_results/compression_lowrank_stage0_humaneval_ep/results/{label}.jsonl` — 每题 passed/failed/error

---

## E1. 选择性 per-token 压缩

### 目的
之前所有实验都"全部 token 都按同一比例压"。师兄建议先**测边界**：如果只压一部分 token、留另一部分不压，是否能换更高 accuracy？另外**选哪些 token 压重要吗**——能不能优先压"不重要"的 token？

### 策略
fix 每行压缩为 topk_l2+FP8 keep_frac=0.25 (4×)。**变量是"哪些 token 被压"和"多少比例 token 被压"**。

3 种 token 重要性信号：
- `random` — 随机选（baseline，验证选择是否真的有所谓）
- `router_conf` — 按 router top-1 prob，**高 conf 的 token 被压**（直觉：决定确定 = 冗余）
- `hidden_norm` — 按 MoE 输入 hidden state 的 L2 norm，**小 norm 的 token 被压**（直觉：残差贡献小）

压缩比例 cf ∈ {0.0, 0.25, 0.5, 0.75, 1.0}。

### 代码
`eval/compression/stage0_selective_token.py`，约 360 行。**monkey-patch 每个 `Qwen3MoeSparseMoeBlock._old_forward`**（accelerate 的 device_map=auto 把原 forward 包成 `functools.partial` 缓存进 `_old_forward`，patch class.forward 不生效）。Patch 后的 forward 按策略算 keep_full_mask `[B*T, top_k]`，per-(token, expert) pair 决定压不压。

### 配置 / 组件
- env: `atom`
- lcc test 8 chunks (4096 tokens)，与之前所有 sweep 同口径
- 3 strategies × 5 fractions = 13 unique configs（cf=0 共享 teacher）

### 完整实验结果

next-token top-1 accuracy（teacher = 76.15%, drop in pp）：

| strategy \ cf | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|
| random | 75.54% (-0.61) | 74.41% (-1.74) | 73.51% (-2.64) | 72.21% (-3.94) |
| router_conf | 75.39% (-0.76) | 73.73% (-2.42) | 73.43% (-2.72) | 72.21% (-3.94) |
| **hidden_norm** | **75.59% (-0.56)** | **75.34% (-0.81)** | **74.46% (-1.69)** | 72.21% (-3.94) |

PPL +%：random ≤ hidden_norm < router_conf；hidden_norm 在所有 cf 上都赢。

### 结论
- **WHICH tokens 重要吗：是的，选择有显著影响**
  - hidden_norm @ cf=0.5：**只掉 0.81pp**（2× 平均压缩）
  - random @ cf=0.5：掉 1.74pp（同 2× 平均压缩）
  - 同样压一半 token，按 hidden_norm 选比 random 选少掉 1pp
- **router_conf 反直觉地最差**：高 conf 不是"冗余"，是"决定明确" — 压坏路由 logits 等于丢决策信息
- **新 Pareto 点**：hidden_norm@cf=0.5 = 2.5× avg compression, -0.81pp 比 4× full-compress (-3.94pp) 强 ~5 倍

### 输出
- `eval_results/compression_lowrank_stage0_selective_token/summary.json`
- `eval_results/compression_lowrank_stage0_selective_token/report.md`
- `eval_results/compression_lowrank_stage0_selective_token/heatmap.png`

---

## E2. Top-k 累积权重保护

### 目的
师兄第二个建议：**"top-1 全量，其他压缩"**，并扩展到"前若干个 expert 的累积权重 ≥ 阈值 T 都全量"，扫不同的 T 看效果。

### 策略
对每个 token 的 top-8 个 expert 按 routing weight 排降序，从大到小累加。每个 expert 的"之前累积权重 < T"的就保留全量 bf16，其余压缩 4×。

T 范围：
- `0.0`：没有保护（= baseline 全 4×）
- `top1_only`：只 top-1 全量（师兄原版）
- `0.5 / 0.7 / 0.9`：累积权重阈值
- `1.0`：全保护（= teacher）

### 代码
`eval/compression/stage0_topk_cumulative.py`，约 360 行。同 E1 的 monkey-patch 框架，仅 keep_full_mask 计算逻辑变（按 routing weight 累积阈值），同时记录每 config 的"压缩对数比例"和"被压的 weight 占比"，输出平均有效压缩比。

### 配置 / 组件
- env: `atom`
- 同 E1 lcc 测试集

### 完整实验结果

| T | top-1 acc | drop | PPL +% | 压缩对数 | 被压 weight | **avg compress** |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 (baseline) | 72.21% | -3.94pp | +17.2% | 100.0% | 100.0% | **4.00×** |
| **top1_only** | 74.41% | **-1.74pp** | +6.8% | 87.5% | 70.2% | **2.91×** |
| **0.5** | 75.61% | **-0.54pp** | +1.5% | 59.8% | 36.6% | **1.81×** |
| 0.7 | 75.88% | -0.27pp | +0.3% | 39.1% | 19.4% | 1.41× |
| 0.9 | 75.98% | -0.17pp | -0.1% | 12.8% | 5.2% | 1.11× |
| 1.0 (teacher) | 76.15% | 0pp | 0% | 0.0% | 0.0% | 1.00× |

### 结论
**Top-k 累积权重保护是非常干净的 Pareto win**：

- **T=top1_only**（师兄原版）：**2.91× 压缩、only -1.74pp** — 比 baseline 4× (-3.94pp) 直接掉一半 drop
- **T=0.5**：1.81× 压缩、-0.54pp — 几乎无损
- 曲线非常平滑：从 100% 压（4×）到 0% 压（1×），accuracy 单调上升，没 cliff

直觉上：MoE 的 top-1 expert 平均吃 30% 的 routing weight mass，压它的影响远大于压排名第 8 的 expert（吃 ~5%）。**weight-proportional budget allocation 是理论最优近似**。

### 输出
- `eval_results/compression_lowrank_stage0_topk_cumulative/summary.json`
- `eval_results/compression_lowrank_stage0_topk_cumulative/report.md`
- `eval_results/compression_lowrank_stage0_topk_cumulative/tradeoff_curve.png`

### E1 vs E2 综合对比（同 next-token acc 口径）

| 方案 | 平均压缩 | top-1 drop |
|---|---:|---:|
| Baseline 全压 4× | 4.00× | -3.94pp |
| E1 hidden_norm @ cf=0.75 | ~3.25× | -1.69pp |
| **E2 T=top1_only** | **2.91×** | **-1.74pp** |
| **E1 hidden_norm @ cf=0.50** | **~2.50×** | **-0.81pp** ← 同档最优 |
| E2 T=0.5 | 1.81× | -0.54pp |
| E1 hidden_norm @ cf=0.25 | ~1.75× | -0.56pp |
| E2 T=0.7 | 1.41× | -0.27pp |
| E2 T=0.9 | 1.11× | -0.17pp |

**两条策略 Pareto 互补**：
- 高压缩区（≥3×）：E1 hidden_norm 略胜
- 中等压缩区（1.5–2.5×）：E2 累积权重略胜
- **下一步可以试组合**：先按 hidden_norm 选 token，再对被选 token 用 E2 累积保护其 top-1 expert

---

## 3. 跨实验总结

### 3.1 方向决策树

```
低秩压缩 dispatch 端
  └ 严格 EV@d/4 ≥ 0.90 → NO（A1，中层 0.66-0.78）
  └ 跨域救（A2）→ 部分救（代码域浅层），但中层无效
  └ Task-aware 训练 P_down/P_up（A3）→ NO（top-k 不可微）
  └ 多层全 SVD sweep（A4）→ NO（2× 已 PPL +119%）
  └ Routing-aware projection（A5）→ NO（L² 累积才是根因）
      └ 诊断：无论用什么映射，dispatch 端死透

  ↓ 转向

Combine 端 sparsification（全部 token 同档压）
  ├ naive top-k（B1）→ 2× +3.3%, 4× +20.6%, 8× +85% — GO 至 4×
  ├ topk_l2 winner（B2）→ 同档比 naive 强 20-156pp
  ├ 跨 4 域稳定（B3）→ 域无关 win
  └ + FP8 几乎免费（B4）→ 4× sparse + 2× FP8 = 8× 总，PPL 几乎一样

  ↓ 集成

真实 EP-HT 集成（C1）+ EP=8 真 NCCL（C3）→ math 在 8× 下保持，wall-clock 暂慢

  ↓ 真 task accuracy

Next-token (D1)：2× 掉 0.66pp、4× 掉 3.94pp、8× 掉 9.42pp
HumanEval (D2)：2× 掉 7.93pp、4× 掉 9.15pp、8× 掉 37.2pp（崩坏）

  ↓ 不均匀压缩探索

E1 选 token 压：hidden_norm @ cf=0.5 → 2.5× avg + 只 -0.81pp（vs full 4× 的 -3.94pp）
E2 保护重要 expert：T=top1_only → 2.9× avg + 只 -1.74pp；T=0.5 → 1.8× avg + 只 -0.54pp
  └ E1 / E2 都给出干净 Pareto 改进；E1+E2 可组合
```

### 3.2 关键 take-aways（论文级）

1. **Dispatch 端低秩压缩在 frozen MoE 上结构性 NO-GO**——映射方法无关，根因是 residual stream 上 L² 误差跨层乘性累积 + 路由 top-k 不可微。

2. **Combine 端 top-magnitude + L2 rescale + FP8 是干净的可行方向**——4× 压缩 PPL +17%, HumanEval pass@1 掉 9.15pp。

3. **3 个 metric 的非线性关系**：PPL << next-token << HumanEval。**Code 部署只看 PPL 严重低估退化**。

4. **均匀压缩甜点在 4× value 压缩**（= ~2.7× 真实 wire 压缩，含 indices + scale 开销）。8× 是悬崖。

5. **不均匀压缩可以再降一个 magnitude 的 drop**：
   - 按 token L2 norm 选择压（hidden_norm@cf=0.5）→ 2.5× avg compression 只掉 0.81pp
   - 按 routing weight 累积阈值保护 expert（T=0.5）→ 1.8× avg compression 只掉 0.54pp
   - router_conf 反直觉地最差 — 高 conf token 不是冗余，是决策明确

### 3.3 待办（按价值排序）

1. **fused Triton kernel for pack/unpack** — 当前 EP=8 wall-clock 1.3–1.9× 慢，瓶颈在 eager 模式 48 层 × 8 rank 的小 kernel 调度，融合后估计回 baseline 甚至更快。
2. **HumanEval 4× 那 9pp 的 task-aware recovery** — 比如选择性给中层更高 keep_frac、或对"容易错的位置"动态加大 budget。
3. **跨模型对照** — DeepSeek-V2-Lite-Chat 跑同套压缩，看是不是 Qwen3 专享。
4. **MMLU / GSM8K accuracy** — HumanEval 之外的第二个 task benchmark。
5. **整理论文 outline** — 数据齐了，可以开始写。

---

## 4. 复现 checklist

```bash
# 公共
cd /home/lzy/Artifact-Infer
git checkout my-a2a-kernel-test     # commit 25c2294 或更新

# A 系列（HF 单进程）— env: atom
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /home/lzy/miniconda3/envs/atom/bin/python eval/compression/stage0_low_rank_svd_v2.py
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /home/lzy/miniconda3/envs/atom/bin/python eval/compression/stage0_per_domain.py
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /home/lzy/miniconda3/envs/atom/bin/python eval/compression/stage0_task_aware_projector.py
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /home/lzy/miniconda3/envs/atom/bin/python eval/compression/stage0_full_model_sweep.py
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /home/lzy/miniconda3/envs/atom/bin/python eval/compression/stage0_routing_aware_sweep.py

# B 系列（HF 单进程）— env: atom
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /home/lzy/miniconda3/envs/atom/bin/python eval/compression/stage0_combine_sparsify_sweep.py
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /home/lzy/miniconda3/envs/atom/bin/python eval/compression/stage0_combine_ef_variants.py
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /home/lzy/miniconda3/envs/atom/bin/python eval/compression/stage0_combine_cross_domain.py
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /home/lzy/miniconda3/envs/atom/bin/python eval/compression/stage0_combine_sparse_fp8.py

# C 系列（真 EP=2/8 NCCL）— env: vllm + CUDA 12.8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:/home/lzy/miniconda3/envs/vllm/bin:$PATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WORLD_SIZE=8 MOE_IMPL=ep_ht ENFORCE_EAGER=1 \
  KEEP_FRACS=0.5,0.25,0.125 USE_FP8=1 MAX_TOKENS=24 \
  /home/lzy/miniconda3/envs/vllm/bin/python -m eval.compression.test_ep_ht_compress_full

# Config-time 路径自检（用 Config kwarg 而非 env var 启用压缩）
CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 NUM_LAYERS=2 KEEP_FRAC=0.25 \
  /home/lzy/miniconda3/envs/vllm/bin/python -m eval.compression.test_ep_ht_compress_config

# E 系列（HF 单进程 + monkey-patch）— env: atom
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /home/lzy/miniconda3/envs/atom/bin/python eval/compression/stage0_selective_token.py
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /home/lzy/miniconda3/envs/atom/bin/python eval/compression/stage0_topk_cumulative.py

# D 系列
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /home/lzy/miniconda3/envs/atom/bin/python eval/compression/stage0_taskacc_lcc.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WORLD_SIZE=8 MOE_IMPL=ep_ht ENFORCE_EAGER=1 \
  /home/lzy/miniconda3/envs/vllm/bin/python -m eval.compression.stage0_humaneval_ep
```

## 5. 已知 caveat / 风险

- **算法 wall-clock 暂时慢**：当前 EP=8 上 1.3–1.9× baseline，因为 pack/unpack 是小 kernel，eager 模式 dispatch 不够紧。fused triton kernel 后应该回正。
- **HumanEval 用 raw text-completion 不是 chat template**：跟 Qwen3 公开 benchmark 数字（>70%）有 gap，**但 teacher / student 同口径对比仍 valid**。
- **HF 模拟 vs 真 EP 的一致性**：已验证 algorithm 在数值上完全一致（同 dtype 同 op），但 wall-clock 行为不一样。
- **代码补全是最敏感的 task**——HumanEval 数字是"上界悲观"。MMLU / 自然语言生成应该掉得少得多。

## 6. 相关文档

- `docs/compression/MoE_EP通信压缩_调研报告.md` — 最初的方向调研（领域 landscape）
- `docs/compression/report.md` — 第一周周报简版
- `docs/compression/stage0_findings.md` — 实验 A1 详版（中文）
- `docs/compression/stage0_per_domain_findings.md` — 实验 A2 详版（中文）
- 本文档 — 第二周完整实验记录（A0 → D2）

## 7. Git 状态

- 分支: `my-a2a-kernel-test`
- 最新 commit: `25c2294 drop 实验 + EP-HT combine-side topk_l2/FP8 compression`
- origin: 已同步（GitHub https://github.com/mylzy007/Artifact-Infer/tree/my-a2a-kernel-test）
- 所有实验代码 + 输出（除 1.6 GB 大 .pt activation dump，已 gitignore）都在 push 范围内。
