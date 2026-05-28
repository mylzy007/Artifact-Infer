# `eval/drop/` — Phase 4 token-replica drop 实验脚本

跟 `eval/run_owner_local_ep_phase4_drop.py`（Phase 4 v1/v2 历史 GSM8K runner）配套，本目录是 v3+ 的扩展工作。命名规则：

- `run_owner_local_ep_phase4_drop_<descriptor>_<version>.py` — **主实验运行脚本**（torchrun launchable）
- `launch_phase4_drop_<descriptor>_<version>.sh` — **shell launcher**（一键起两 plan、设环境变量）
- `plot_phase4_drop_<descriptor>.py` — **离线绘图脚本**
- `rescore_phase4_drop_<descriptor>.py` — **离线 accuracy 重打分**

## 主实验运行脚本（runners）

| 文件 | 干啥的 | 输出 |
|---|---|---|
| `run_owner_local_ep_phase4_drop_tier1_microbench.py` | **Tier 1**：1 层 EP-HT MoE block 隔离 microbench，扫 (T_local, drop_rate)，分段 CUDA event 计时 | `eval_results/prefill_drop_l_sweep_tier1/` |
| `run_owner_local_ep_phase4_drop_tier2_synthetic_e2e.py` | **Tier 2**：full LLMEngine + synthetic prompts，验证 Tier 1 break-even L* 在 e2e 上 | `eval_results/prefill_drop_l_sweep_tier2/` |
| `run_owner_local_ep_phase4_drop_long_e2e_v1.py` | **v1**：单 policy `tail_weight@0.3` 在 LEval 长 prompt e2e | `eval_results/drop_long_e2e_leval/` |
| `run_owner_local_ep_phase4_drop_multiaxis_v1.py` | **v1**：多轴 sweep（policy × rate × bypass × length tier） | `eval_results/owner_local_ep_phase4_drop_longbench/` |
| `run_owner_local_ep_phase4_drop_multipolicy_v2.py` | **v2-v5 主 bench**：5 GPU policies × 3 rates，按 `--num-samples / --dataset / --max-new-tokens` 跑不同变体 | `eval_results/owner_local_ep_phase4_drop_longbench_v{2,3,4,5}/` |
| `run_owner_local_ep_phase4_drop_passage_retrieval_v6.py` | **v6**：LongBench `passage_retrieval_en_e` + 官方 binary metric (`retrieval_score`)。最权威的 accuracy 实验 | `eval_results/owner_local_ep_phase4_drop_passage_retrieval_v6/` |
| `run_owner_local_ep_phase4_drop_official_template_test.py` | LongBench 官方 template + max_new_tokens 单独 baseline 验证（debug 用，验证 pipeline 是否对齐 leaderboard） | `eval_results/longbench_official_baseline_*/` |

## Shell launchers（一键跑双 plan）

| 文件 | 调用谁 | 用途 |
|---|---|---|
| `launch_phase4_drop_multipolicy_v3.sh` | `run_owner_local_ep_phase4_drop_multipolicy_v2.py` | v3：2 plans × 5 GPU policies × 3 rates × 80 prompts，max_new=64 |
| `launch_phase4_drop_multipolicy_v4.sh` | 同上 | v4：max_new=256, 存完整 gen + ref 文本，accuracy 重测修复 |
| `launch_phase4_drop_multipolicy_v5_cross_dataset.sh` | 同上 | v5：换 multifieldqa_en 数据集，cross-dataset 验证 |
| `launch_phase4_drop_passage_retrieval_v6.sh` | `run_owner_local_ep_phase4_drop_passage_retrieval_v6.py` | v6：passage_retrieval + 官方 binary metric |

## 重打分（offline accuracy）

| 文件 | 干啥的 |
|---|---|
| `rescore_phase4_drop_multimetric.py` | 拿 v4/v5 jsonl 的 `gen_text` + `ref_text`，重算 F1 / recall / substring / ROUGE-L / exact-match 5 个 metric（手写实现） |
| `rescore_phase4_drop_longbench_official.py` | 用 LongBench 官方 `qa_f1_score` + Google `rouge_score` 包重打分。**对齐 leaderboard 必备** |

## Plots

| 文件 | 画什么 |
|---|---|
| `plot_phase4_drop_tier1.py` | Tier 1：total_us vs L_recv，delta 曲线，per-segment 分解 |
| `plot_phase4_drop_multiaxis_v1.py` | v1：3 phase（policy×rate / bypass / length-tier） |
| `plot_phase4_drop_multipolicy_v2.py` | v2：5 policy × rate heatmap + Pareto |
| `plot_phase4_drop_multipolicy_v3.py` | v3：跨 plan 对比 |
| `plot_phase4_drop_multipolicy_v3_comprehensive.py` | v3：7 张综合图（heatmap / 误差棒 / F1 噪声带 / 绝对吞吐 / Pareto / sensitivity / plan diff） |
| `plot_phase4_drop_multipolicy_v4_recall.py` | v4：recall-based plots（取代 F1） |
| `plot_phase4_drop_cross_dataset_v4_vs_v5.py` | v4 vs v5 cross-dataset 对比 |
| `plot_phase4_drop_passage_retrieval_v6.py` | v6：7 张图（heatmap / Pareto / strict vs official / rate 曲线 / 跨 3 dataset 总汇 / per-prompt 分布） |

## Utilities / tests（不重命名）

| 文件 | 干啥的 |
|---|---|
| `shared.py` | `CUDAEventTimer`, `write_jsonl`, `agg_stats`, optional `WandbLogger` |
| `__init__.py` | 包标记 |
| `tests/test_drop_invariants.py` | unit tests：drop_rate=0 等价于无 drop / effective_drop_send_frac 跟 nominal 一致 / 新 policy（weighted_tail, cross_numa_uniform）正常工作 |

## 典型工作流

```bash
# 1. 跑 unit test 确保 drop kernel 正确
/home/lzy/miniconda3/envs/vllm/bin/python -m eval.drop.tests.test_drop_invariants

# 2. 跑 Tier 1 microbench 找 break-even L*（~3 min on 8×4090）
/home/lzy/miniconda3/envs/vllm/bin/torchrun --nproc_per_node=8 --master_port=29503 \
    -m eval.drop.run_owner_local_ep_phase4_drop_tier1_microbench \
    --output-dir eval_results/prefill_drop_l_sweep_tier1 \
    --t-local-values 1,8,64,512,2048 --drop-rates 0.0,0.3 \
    --warmup-iters 10 --iters 30 \
    --expert-overlap-path <Phase 3 plan>

# 3. v6 主实验（~70 min）
bash eval/drop/launch_phase4_drop_passage_retrieval_v6.sh

# 4. 画 v6 图
/home/lzy/miniconda3/envs/vllm/bin/python -m eval.drop.plot_phase4_drop_passage_retrieval_v6
```

## 相关代码（不在本目录但配合使用）

- `workshop/nanovllm_moe/services/utils/expert_drop.py` — drop kernel 本体（含 v3 新增的 `weighted_tail` / `cross_numa_uniform`）
- `eval/run_owner_local_ep_phase4_drop.py` — Phase 4 v1/v2 历史 GSM8K runner（本目录是它的延续）

## 文档

`docs/claude-moe/drop/0526-long-bench/FINAL_REPORT.md` — Phase 4 drop 综合最终报告（v1-v6 全部实验整合）
