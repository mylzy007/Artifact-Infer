# vLLM E2E Bench Artifact

根据vllm的main分支修改，详细修改可见 https://github.com/mylzy007/vllm.git，branch: e2e-bench
diff: e2e_artifact.diff
diff baseline: origin/main

其中我新增的脚本都在vllm/scripts下

## 相关脚本

- `scripts/run_deepseek_v2_lite_vllm_longbench_multi.sh`
  - E2E benchmark 主脚本，负责启动 `vllm serve`、运行 `vllm bench serve`，并组织不同实验 case。

- `scripts/run_q2_selected_vllm_matrix4.sh`
  - 批量调度脚本，用于批量运行挑选好的数据集和实验设置。

- `scripts/prepare_bench_custom_jsonl.py`
  - 数据准备脚本，将原始数据集转换为 `vllm bench serve` 可直接使用的 custom JSONL 格式。

- `scripts/q2_collect_unified_cv_existing_runs.py`
  - case 级 load balancing 统计脚本，从 `expert_distribution_recorder*.pt` 中汇总整体 CV 指标。

- `scripts/export_layer_gpu_share_stats.py`
  - layer 级 load balancing 统计脚本，输出每层的 CV、分位数和 routed tokens 信息。

- `scripts/plot_expert_distribution_heatmap.py`
  - 可视化脚本，用于绘制 expert / rank 的负载 heatmap。

- `scripts/run_moe_kernel_cv_sweep.py`
  - kernel microbenchmark 脚本，用 synthetic workload 分析不同 CV 下的 kernel 性能变化。

## 运行示例

准备 benchmark 输入数据：
```bash
python scripts/prepare_bench_custom_jsonl.py \
  --dataset longbench \
  --source /path/to/longbench \
  --subset triviaqa \
  --out /path/to/prepared/longbench.triviaqa.custom.jsonl
```

运行单个 E2E case：
```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
BENCHMARK_DATASET=longbench \
BENCHMARK_SUBSET=triviaqa \
RESULT_ROOT=/home/lzy/eval/vllm_deepseek_v2_lite_matrix4 \
CASE_FILTER=nochunk_eplb \
bash scripts/run_deepseek_v2_lite_vllm_longbench_multi.sh
```

批量运行选定 case：
```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
BENCHMARK_NUM_PROMPTS=128 \
BENCHMARK_MAX_CONCURRENCY=8 \
bash scripts/run_q2_selected_vllm_matrix4.sh
```

统计 case 级 CV：
```bash
python scripts/q2_collect_unified_cv_existing_runs.py \
  --vllm-root /home/lzy/eval/vllm_deepseek_v2_lite_matrix4 \
  --output /home/lzy/vllm/refine-logs/q2_unified_cv_existing_runs.csv
```

导出 layer 级统计：
```bash
python scripts/export_layer_gpu_share_stats.py \
  --root /home/lzy/eval/vllm_deepseek_v2_lite_matrix4/longbench_triviaqa
```

绘制 heatmap：
```bash
python scripts/plot_expert_distribution_heatmap.py \
  --manifest /home/lzy/eval/vllm_deepseek_v2_lite_matrix4/longbench_triviaqa/nochunk_eplb/expert_record_files.txt \
  --out-dir /home/lzy/eval/vllm_deepseek_v2_lite_matrix4/longbench_triviaqa/nochunk_eplb/heatmap \
  --model-name /home/lzy/models/DeepSeek-V2-Lite-Chat \
  --enable-eplb 1
```
运行 kernel CV sweep：

```bash
python scripts/run_moe_kernel_cv_sweep.py \
  --output-dir /home/lzy/eval/vllm_kernel_cv_sweep \
  --cv-values 0.05,0.10,0.15,0.20 \
  --world-size 8 \
  --total-tokens 131072
```

## vLLM 源码修改说明

对 vLLM 源码做了少量修改，使它支持 E2E benchmark 中的 expert routing 记录与 load balancing 分析:

增加 expert routing record 的导出；
保存 MoE / EP / EPLB 相关映射和元信息；
支持恢复 all / prefill / decode 三种粒度的 expert load；
为后续 CV 统计和 heatmap 可视化提供数据。