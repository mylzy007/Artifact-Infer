# perf_ws8_ws4_20260516_0228 实验统计与对比分析

## 1. 本次实验配置

这份报告只统计磁盘上这批真正同时包含 `ws=8` 和 `ws=4` 的凌晨实验结果：

- Top-level run id: `perf_ws8_ws4_20260516_0228`
- 批次目录: `/home/lzy/Artifact-Infer/eval_results/moe_placement_batches_20260516_1609/early_perf_ws8_ws4_0228`
- ws8 summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_batches_20260516_1609/early_perf_ws8_ws4_0228/moe_placement_run_perf_ws8_ws4_20260516_0228_p1_t32_ws8/summary.json`
- ws4 summary: `/home/lzy/Artifact-Infer/eval_results/moe_placement_batches_20260516_1609/early_perf_ws8_ws4_0228/moe_placement_run_perf_ws8_ws4_20260516_0228_p1_t32_ws4/summary.json`

原始 scale sweep 配置里包含：

| 配置项 | 值 |
| --- | --- |
| `world_sizes` | `8,4` |
| `scale_steps` | `1x32,4x64,8x64` |
| `model_path` | `/home/lzy/models/Qwen3-30B-A3B` |
| `dataset_path` | `/home/lzy/Artifact-Infer/datasets/gsm8k_moe_smoke.parquet` |
| `tp_size` | `1` |
| `num_layers` | `-1`，即 full layers |
| `max_model_len` | `512` |
| `max_num_batched_tokens` | `1024` |
| `max_num_seqs` | `4` |
| `gpu_memory_utilization` | `0.85` |
| `enforce_eager` | `1` |
| `moe_ll_overflow_policy` | `drop` |
| `timeout_s` | `5400` |

需要特别说明：这次 scale sweep 计划继续跑 `4x64` 和 `8x64`，但实际只完成了 `1x32` 的 `ws=8` 和 `ws=4`。原因是 `p1_t32_ws4` 里 `ep_ht + round_robin` 失败，scale sweep 因 `all_pass=false` 停止，后面的 scale step 没有继续执行。所以本报告的对比范围是：

| 实际完成项 | 值 |
| --- | --- |
| `num_problems` | `1` |
| `max_tokens` | `32` |
| `world_size` | `8` 和 `4` |
| generated tokens | 每个成功 eval 为 `32` |

## 2. 并行与 GPU 设置

脚本会按 world size 改变 GPU 分组：

| world size | GPU 分组 | 并发方式 |
| --- | --- | --- |
| ws8 | `0,1,2,3,4,5,6,7` | 一次占满 8 卡，`parallel_jobs=1` |
| ws4 | `0,1,2,3` 和 `4,5,6,7` | 两组 4 卡 job 并发，`parallel_jobs=2` |

`tp_size=1`，所以 EP size 等于 world size：ws8 是 EP=8，ws4 是 EP=4。

## 3. 实验组合

每个 world size 的流程是：

1. 先用 `ep_ll_triton + contiguous` 跑一次 routing profile。
2. 基于 routing profile 生成需要 JSON 文件的 placement。
3. 对所有 `impl x placement` 组合做正式 eval。

本次 impl：

| impl | 含义 |
| --- | --- |
| `ep_ll_triton` | EP low-latency 路径，dispatch 用 triton 版本 |
| `ep_ht` | EP high-throughput 路径 |

本次 expert placement：

| placement | 含义 |
| --- | --- |
| `contiguous` | 连续 expert 分配，作为 baseline |
| `round_robin` | 按 rank 轮转分配 expert |
| `fixed_random_shuffle` | 固定随机 shuffle，使用相同 seed 可复现 |
| `load_balanced_greedy_with_locality_tiebreak` | 贪心负载均衡，locality 作为 tie-break |
| `communication_aware_greedy` | 贪心降低跨 rank traffic，偏通信优化 |

每个 world size 应有 `2 impls x 5 placements = 10` 个正式 eval。ws8 全部成功；ws4 成功 9 个，失败 1 个。

## 4. 这份目录里每个文件是干嘛的

| 文件 | 用途 |
| --- | --- |
| `analysis.md` | 当前这份中文说明、统计和结论 |
| `runtime_results.csv` | 正式 eval 的 runtime 明细，包含 e2e、prefill、decode、是否成功、log 路径 |
| `placement_compute_comm_metrics.csv` | 基于 routing profile 估计的 placement compute/comm 负载指标，不是真实 kernel 耗时 |
| `delta_vs_contiguous.csv` | 同一 `ws + impl` 下，每个 placement 相对 contiguous 的变化 |
| `ws4_vs_ws8.csv` | 同一 `impl + placement` 下，ws4 相对 ws8 的变化 |
| `ep_ht_vs_ep_ll_speedup.csv` | 同一 `ws + placement` 下，EP-HT 相对 EP-LL 的加速比 |

## 5. 指标说明

`runtime_results.csv` 里的 runtime 指标含义：

| 指标 | 含义 |
| --- | --- |
| `e2e_s` | 单个 eval 从 generate 开始到结束的端到端时间，越低越好 |
| `prefill_time_s` | prefill 阶段累计时间，越低越好 |
| `prefill_tok_s` | prefill token throughput，越高越好 |
| `decode_time_s` | decode 阶段累计时间，越低越好 |
| `decode_tok_s` | decode token throughput，越高越好 |
| `generated_tokens` | 本次 eval 实际生成 token 数 |

`placement_compute_comm_metrics.csv` 里的指标来自 routing profile 估算：

| 指标 | 含义 |
| --- | --- |
| `estimated_compute_total` | routing profile 中所有 routed expert replica 的总量 |
| `estimated_compute_max/min` | 各 GPU/rank 估计 compute load 的最大值和最小值 |
| `estimated_gpu_cv` | GPU/rank 负载变异系数，越低表示越均衡 |
| `estimated_expert_cv` | expert 热度变异系数，由 workload 决定，placement 不会改变它 |
| `estimated_local_traffic` | 估计留在本 rank 的 routed traffic |
| `estimated_cross_traffic` | 估计跨 rank 的 routed traffic |
| `estimated_cross_traffic_ratio` | 跨 rank traffic 占比，越低表示理论通信压力越小 |

注意：当前脚本没有保存真实 `dispatch_time / combine_time / expert_compute_time`。这里的 compute/comm 是基于 routing profile 的 placement 估计指标，不是 CUDA kernel 或阶段计时。

## 6. 执行状态

| world size | profile | placement generation | formal eval |
| --- | --- | --- | --- |
| ws8 | pass | 3/3 pass | 10/10 pass |
| ws4 | pass | 3/3 pass | 9/10 pass |

ws4 唯一失败项：

| ws | impl | placement | 状态 |
| --- | --- | --- | --- |
| 4 | `ep_ht` | `round_robin` | fail，log 显示属于 master port `EADDRINUSE`/端口冲突类问题，不是该 placement 的性能结果 |

## 7. Runtime 结果

完整 runtime 明细如下。`e2e_s` 越低越好，`prefill_tok_s` 和 `decode_tok_s` 越高越好。

| ws | impl | placement | status | e2e_s | prefill_tok_s | decode_tok_s | generated |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 8 | ep_ll_triton | contiguous | pass | 116.676 | 7.018 | 0.292 | 32 |
| 8 | ep_ll_triton | round_robin | pass | 115.228 | 8.042 | 0.292 | 32 |
| 8 | ep_ll_triton | fixed_random_shuffle | pass | 115.640 | 7.687 | 0.292 | 32 |
| 8 | ep_ll_triton | load_balanced_greedy_with_locality_tiebreak | pass | 113.969 | 7.719 | 0.297 | 32 |
| 8 | ep_ll_triton | communication_aware_greedy | pass | 118.520 | 7.176 | 0.287 | 32 |
| 8 | ep_ht | contiguous | pass | 13.700 | 11.879 | 4.155 | 32 |
| 8 | ep_ht | round_robin | pass | 13.205 | 12.314 | 4.316 | 32 |
| 8 | ep_ht | fixed_random_shuffle | pass | 11.897 | 15.455 | 4.371 | 32 |
| 8 | ep_ht | load_balanced_greedy_with_locality_tiebreak | pass | 13.464 | 11.466 | 4.431 | 32 |
| 8 | ep_ht | communication_aware_greedy | pass | 13.448 | 11.929 | 4.288 | 32 |
| 4 | ep_ll_triton | contiguous | pass | 94.317 | 11.172 | 0.354 | 32 |
| 4 | ep_ll_triton | round_robin | pass | 89.942 | 9.401 | 0.378 | 32 |
| 4 | ep_ll_triton | fixed_random_shuffle | pass | 94.220 | 9.786 | 0.358 | 32 |
| 4 | ep_ll_triton | load_balanced_greedy_with_locality_tiebreak | pass | 89.777 | 9.360 | 0.379 | 32 |
| 4 | ep_ll_triton | communication_aware_greedy | pass | 93.695 | 11.464 | 0.355 | 32 |
| 4 | ep_ht | contiguous | pass | 13.254 | 13.453 | 4.006 | 32 |
| 4 | ep_ht | round_robin | fail |  |  |  |  |
| 4 | ep_ht | fixed_random_shuffle | pass | 10.797 | 16.380 | 4.946 | 32 |
| 4 | ep_ht | load_balanced_greedy_with_locality_tiebreak | pass | 10.787 | 17.921 | 4.664 | 32 |
| 4 | ep_ht | communication_aware_greedy | pass | 23.345 | 11.473 | 1.839 | 32 |

每个 `ws + impl` 的 e2e 最优项：

| ws | impl | best placement | best e2e_s | decode_tok_s |
| --- | --- | --- | ---: | ---: |
| 8 | ep_ll_triton | load_balanced_greedy_with_locality_tiebreak | 113.969 | 0.297 |
| 8 | ep_ht | fixed_random_shuffle | 11.897 | 4.371 |
| 4 | ep_ll_triton | load_balanced_greedy_with_locality_tiebreak | 89.777 | 0.379 |
| 4 | ep_ht | load_balanced_greedy_with_locality_tiebreak | 10.787 | 4.664 |

## 8. Placement compute/comm 估计结果

这部分只看 placement 本身的 routing 负载和跨 rank traffic，不区分 `ep_ll_triton` 和 `ep_ht`。同一个 world size 下，不同 impl 共用同一份 placement profile 估计指标。

| ws | placement | compute_total | compute_max | compute_min | gpu_cv | cross_ratio | cross_traffic |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | contiguous | 319488 | 44766 | 37351 | 0.0570 | 0.8750 | 279556 |
| 8 | round_robin | 319488 | 44953 | 35835 | 0.0830 | 0.8749 | 279526 |
| 8 | fixed_random_shuffle | 319488 | 46486 | 31555 | 0.1167 | 0.8750 | 279566 |
| 8 | load_balanced_greedy_with_locality_tiebreak | 319488 | 40022 | 39720 | 0.0024 | 0.8748 | 279479 |
| 8 | communication_aware_greedy | 319488 | 44061 | 35245 | 0.0739 | 0.8736 | 279109 |
| 4 | contiguous | 159744 | 43450 | 37768 | 0.0539 | 0.7497 | 119761 |
| 4 | round_robin | 159744 | 40815 | 39195 | 0.0181 | 0.7499 | 119797 |
| 4 | fixed_random_shuffle | 159744 | 42384 | 37694 | 0.0435 | 0.7501 | 119827 |
| 4 | load_balanced_greedy_with_locality_tiebreak | 159744 | 39966 | 39878 | 0.0009 | 0.7500 | 119806 |
| 4 | communication_aware_greedy | 159744 | 46010 | 36218 | 0.0922 | 0.7479 | 119477 |

从这个表看：

- `load_balanced_greedy_with_locality_tiebreak` 在 ws8 和 ws4 都把 `gpu_cv` 压到最低，说明 rank 间 compute load 最均衡。
- `communication_aware_greedy` 确实降低了一点 `cross_ratio`，但幅度很小；同时它在 ws4 的 `gpu_cv=0.0922`，负载不均衡最明显。
- ws8 的 `cross_ratio` 约 `0.875`，ws4 的 `cross_ratio` 约 `0.75`，这符合 EP rank 越多、本地命中概率越低的直觉。

## 9. 相对 contiguous 的变化

这部分回答：同一个 `ws + impl` 下，换 placement 相比 baseline `contiguous` 是变快还是变慢。

| ws | impl | placement | status | e2e_delta | prefill_delta | decode_delta |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 8 | ep_ll_triton | round_robin | pass | -1.2% | +14.6% | +0.1% |
| 8 | ep_ll_triton | fixed_random_shuffle | pass | -0.9% | +9.5% | +0.1% |
| 8 | ep_ll_triton | load_balanced_greedy_with_locality_tiebreak | pass | -2.3% | +10.0% | +1.7% |
| 8 | ep_ll_triton | communication_aware_greedy | pass | +1.6% | +2.3% | -1.9% |
| 8 | ep_ht | round_robin | pass | -3.6% | +3.7% | +3.9% |
| 8 | ep_ht | fixed_random_shuffle | pass | -13.2% | +30.1% | +5.2% |
| 8 | ep_ht | load_balanced_greedy_with_locality_tiebreak | pass | -1.7% | -3.5% | +6.6% |
| 8 | ep_ht | communication_aware_greedy | pass | -1.8% | +0.4% | +3.2% |
| 4 | ep_ll_triton | round_robin | pass | -4.6% | -15.8% | +6.9% |
| 4 | ep_ll_triton | fixed_random_shuffle | pass | -0.1% | -12.4% | +1.2% |
| 4 | ep_ll_triton | load_balanced_greedy_with_locality_tiebreak | pass | -4.8% | -16.2% | +7.1% |
| 4 | ep_ll_triton | communication_aware_greedy | pass | -0.7% | +2.6% | +0.5% |
| 4 | ep_ht | round_robin | fail |  |  |  |
| 4 | ep_ht | fixed_random_shuffle | pass | -18.5% | +21.8% | +23.5% |
| 4 | ep_ht | load_balanced_greedy_with_locality_tiebreak | pass | -18.6% | +33.2% | +16.4% |
| 4 | ep_ht | communication_aware_greedy | pass | +76.1% | -14.7% | -54.1% |

## 10. ws4 vs ws8

这部分回答：同一个 `impl + placement` 下，四卡和八卡谁更快。`ws4_e2e_delta` 为负表示 ws4 e2e 更低。

| impl | placement | ws4_status | ws4_e2e | ws8_e2e | ws4_e2e_delta | ws4_decode_delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ep_ll_triton | contiguous | pass | 94.317 | 116.676 | -19.2% | +21.0% |
| ep_ll_triton | round_robin | pass | 89.942 | 115.228 | -21.9% | +29.2% |
| ep_ll_triton | fixed_random_shuffle | pass | 94.220 | 115.640 | -18.5% | +22.3% |
| ep_ll_triton | load_balanced_greedy_with_locality_tiebreak | pass | 89.777 | 113.969 | -21.2% | +27.5% |
| ep_ll_triton | communication_aware_greedy | pass | 93.695 | 118.520 | -20.9% | +24.0% |
| ep_ht | contiguous | pass | 13.254 | 13.700 | -3.3% | -3.6% |
| ep_ht | round_robin | fail |  | 13.205 |  |  |
| ep_ht | fixed_random_shuffle | pass | 10.797 | 11.897 | -9.3% | +13.2% |
| ep_ht | load_balanced_greedy_with_locality_tiebreak | pass | 10.787 | 13.464 | -19.9% | +5.3% |
| ep_ht | communication_aware_greedy | pass | 23.345 | 13.448 | +73.6% | -57.1% |

## 11. EP-HT vs EP-LL

这部分回答：同一个 `ws + placement` 下，`ep_ht` 相比 `ep_ll_triton` 快多少。`e2e_speedup` 是 `ep_ll_e2e / ep_ht_e2e`。

| ws | placement | ep_ht_status | ep_ll_e2e | ep_ht_e2e | e2e_speedup | decode_speedup |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 8 | contiguous | pass | 116.676 | 13.700 | 8.52 | 14.22 |
| 8 | round_robin | pass | 115.228 | 13.205 | 8.73 | 14.76 |
| 8 | fixed_random_shuffle | pass | 115.640 | 11.897 | 9.72 | 14.94 |
| 8 | load_balanced_greedy_with_locality_tiebreak | pass | 113.969 | 13.464 | 8.46 | 14.92 |
| 8 | communication_aware_greedy | pass | 118.520 | 13.448 | 8.81 | 14.96 |
| 4 | contiguous | pass | 94.317 | 13.254 | 7.12 | 11.33 |
| 4 | round_robin | fail | 89.942 |  |  |  |
| 4 | fixed_random_shuffle | pass | 94.220 | 10.797 | 8.73 | 13.82 |
| 4 | load_balanced_greedy_with_locality_tiebreak | pass | 89.777 | 10.787 | 8.32 | 12.32 |
| 4 | communication_aware_greedy | pass | 93.695 | 23.345 | 4.01 | 5.18 |

## 12. 分析结论

1. 这批结果只能代表 `num_problems=1, max_tokens=32` 的短 decode、小 batch 场景。原计划的 `4x64` 和 `8x64` 没有完成，不能从这批数据推出更大 scale step 的结论。

2. EP-HT 明显快于 EP-LL。ws8 下 EP-HT 的 e2e 加速约 `8.46x-9.72x`；ws4 下除 `communication_aware_greedy` 外也有约 `7.12x-8.73x`。主要差距体现在 decode throughput：EP-HT 大约是 EP-LL 的 `11x-15x`。

3. 对 EP-LL，`load_balanced_greedy_with_locality_tiebreak` 是最稳的 placement。它在 ws8 e2e 为 `113.969s`，ws4 e2e 为 `89.777s`，都是 EP-LL 内最佳；同时它的 `gpu_cv` 也是最低，说明负载均衡确实对 EP-LL 有帮助。

4. 对 EP-HT，`fixed_random_shuffle` 和 `load_balanced_greedy_with_locality_tiebreak` 表现最好。ws8 最快是 `fixed_random_shuffle`，e2e `11.897s`；ws4 最快是 `load_balanced_greedy_with_locality_tiebreak`，e2e `10.787s`，但 `fixed_random_shuffle` 的 `10.797s` 几乎并列。

5. `communication_aware_greedy` 这批数据里不适合作为默认 placement。它确实小幅降低 cross traffic，但收益很小；ws4 下它的负载不均衡明显变差，导致 EP-HT e2e 从 contiguous 的 `13.254s` 恶化到 `23.345s`。

6. ws4 对 EP-LL 更友好。所有成功 placement 中，ws4 相比 ws8 的 EP-LL e2e 低约 `18.5%-21.9%`，decode throughput 高约 `21.0%-29.2%`。这说明在 4090 无 NVLink 环境里，EP-LL 增加到 8 卡后通信/同步压力更重。

7. ws4 和 ws8 对 EP-HT 没有单调结论。`fixed_random_shuffle` 和 load-balanced 在 ws4 更快，但 contiguous 只略快，`communication_aware_greedy` 在 ws4 反而明显更差。EP-HT 更容易受 placement 的负载形状影响。

8. 后续如果继续做主线实验，建议优先保留 `load_balanced_greedy_with_locality_tiebreak` 和 `fixed_random_shuffle`。`communication_aware_greedy` 需要加负载均衡约束后再评估；另外应补跑 `ws4 / ep_ht / round_robin`，否则这一个组合没有有效性能点。
