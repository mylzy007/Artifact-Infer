# Prefill L Sweep Experiment Plan

本文档定义 Phase 4 GPU drop 后续实验：在 prefill / MoE-heavy 场景中扫描
每 GPU 每层实际接收的 token-replica 数 `L_recv`，寻找 drop 开始真正加速的
拐点。

背景结论：

- GPU 化 `tail_weight/random/cross_numa_first` 后，decode-heavy workload 仍不理想。
- 这说明当前 workload 下 drop 减少的 expert rows / a2a payload 可能不足以抵过：
  - GPU drop 自身开销；
  - EP-HT variable-size split sync；
  - payload all-to-all 形状变化；
  - combine 端 scatter / zero-fill / reverse a2a；
  - overlap route 当前仍有 CPU path。
- 因此下一步问题不是“drop 实现是否同步”，而是：
  **当 `L_recv` 增大到多少时，drop 减少的 expert GEMM 和通信量能超过额外开销？**

## 1. 实验目标

找出 `L_recv` 拐点 `L*`：

```text
baseline_total_us(L_recv) == drop_total_us(L_recv, rate=0.3)
```

其中 `total_us` 是单层 MoE block 的 wall time：

```text
dispatch -> experts -> combine
```

最终输出：

- Tier 1 MoE-block 隔离曲线：
  - `L_recv_baseline`
  - `L_recv_drop`
  - `dispatch_us`
  - `experts_us`
  - `combine_us`
  - `total_us`
  - `effective_drop_fraction`
- 标出交叉点 `L*`。
- 自动判断是否进入 Tier 2 全模型 prefill 验证。
- 若全段无交叉，给出完整负结论：在当前 4090 24GB 可达 prefill 规模内，drop 不带来收益。

## 2. 关键修正

原始方案方向正确，但需要收紧四点。

### 2.1 x 轴必须用实际 `L_recv`

`T_local * K` 只是每个 source rank dispatch 前的本地 replica 数。EP-HT 中每个
rank 实际收到的 rows 是：

```python
L_recv_rank = tok_meta.recv_hidden.shape[0]
```

在 routing 完全均衡时：

```text
mean_rank(L_recv_rank) ~= T_local * K
```

但实际可能被以下因素打破：

- top-k expert 分布不均；
- overlap plan 的 local-first routing；
- drop 后每个 dst rank 被删除的 rows 不均；
- `tail_weight` 按 weight 而不是按 dst rank 均匀 drop；
- EP-HT ragged all-to-all 使用真实 split sizes。

因此 CSV 中必须同时记录：

```text
L_send_nominal = T_local * K
L_recv_mean
L_recv_p50
L_recv_max
L_recv_min
```

画图时主 x 轴使用 baseline 的 `L_recv_mean` 或 `L_recv_max`。如果判断“分布式 step
真实耗时”，更推荐用 `L_recv_max`，因为 slowest rank 决定 step 时间。

### 2.2 timing 必须取跨 rank max

每个 rank 的 local CUDA event 只能测本 rank 时间。分布式 MoE step 的可见耗时由
最慢 rank 决定。因此每次 repeat 应收集所有 rank 的 segment times，并记录：

```text
dispatch_us_rank_max
experts_us_rank_max
combine_us_rank_max
total_us_rank_max
```

也保留 mean/p50 作为负载不均衡诊断。

### 2.3 Tier 1 不应包含 attention / sampling

Tier 1 只测：

```text
DispatchEPHT.forward
ExpertsEPHT.forward
CombineEPHT.forward
```

不测：

- attention；
- logits sampling；
- tokenizer；
- scheduler；
- KV cache allocation；
- full model residual / norm。

Tier 1 的结论是 MoE block 内部的 break-even。Tier 2 再验证 full prefill 中
attention 会把收益稀释多少。

### 2.4 K_eff 是附加对照，不是主实验

`router_keff=6` 可顺带测，但不能混入主 `tail_weight@0.3` 拐点结论。

当前 standalone Tier 1 可以让 `CombineEPHT(top_k=dispatch.K)`，因此 K_eff
维度一致。若跑全模型 K_eff，需要确认 `FusedMoE` 中 `CombineEPHT` 的 `top_k`
与实际 dispatch K 一致，否则 `tok_meta.sort_perm` 和 `topk_weights` 的 K 维度
可能不匹配。

## 3. 实验分层

### Tier 1：MoE-block 隔离微基准

先做。新增：

```text
workshop/nanovllm_moe/_bench_prefill_drop_sweep.py
```

运行方式：

```bash
torchrun --nproc_per_node=8 \
  -m workshop.nanovllm_moe._bench_prefill_drop_sweep \
  --output-dir eval_results/prefill_drop_l_sweep_tier1 \
  --drop-policy tail_weight \
  --drop-rates 0,0.3 \
  --t-local-values 16,64,128,256,512,1024,2048,4096 \
  --iters 10 \
  --warmup-iters 5
```

目标：

- 快速扫描 `L_recv`。
- 分离 dispatch / experts / combine。
- 确定是否存在 MoE-block 内部交叉点。

### Tier 2：全模型 prefill 验证

只有 Tier 1 显示拐点在可达范围内才跑。新增或复用：

```text
eval/run_prefill_drop_len_sweep.py
```

运行真实 `LLMEngine.generate(max_tokens=1)`，用 synthetic prompt 控制
`batch * prompt_len`。读取 `engine.last_generation_metrics` 中的：

```text
prefill_tokens
prefill_time_s
prefill_throughput_tok_s
decode_time_s
e2e_total_time_s
```

目标：

- 验证 attention / scheduler / KV cache / gate 等 full model 开销是否稀释 Tier 1 收益。
- 只跑 `L*` 周围 3 到 5 个点。

## 4. Tier 1 详细实现

### 4.1 脚本职责

`_bench_prefill_drop_sweep.py` 负责：

1. 初始化 8-rank owner-local EP。
2. 构造一层 EP-HT MoE block：
   - `DispatchEPHT`
   - `ExpertsEPHT`
   - `CombineEPHT`
   - `MoeBackend`
3. 构造 synthetic `hidden_states` 和 `router_logits`。
4. 对每个 `(T_local, drop_rate)` 运行 warmup + measured repeats。
5. 用 CUDA event 分段计时。
6. 用 `all_gather` 收集每 rank 时间和 row counts。
7. rank 0 写 CSV / JSON / plot。
8. rank 0 自动判断是否建议进入 Tier 2。

### 4.2 分布式初始化

使用 `torchrun` 环境变量：

```python
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)
dist.init_process_group(
    backend="nccl",
    device_id=torch.device(f"cuda:{local_rank}"),
)
torch.set_default_device(f"cuda:{local_rank}")

init_parallel_groups(
    tp_size=1,
    world_size=world_size,
    data_parallel_size=world_size,
    runtime_mode="owner_local_ep",
)
```

约束：

- `world_size == 8` 是主实验。
- 可支持 `world_size=2/4` smoke，但结论只认 8 卡。

### 4.3 MoE block 初始化

模型 shape 使用 Qwen3-30B-A3B：

```text
H = 2048
N = 768
E_global = 128
K = 8
BLOCK_M = 64
dtype = bf16
```

初始化一次最大容量，而不是每个 shape 重建：

```text
max_T_local = max(t_local_values)
T_cap = max_T_local
```

`MoeBackend._init_ep_ht_buffers()` 会按 strict worst-case 分配：

```text
T_recv_cap = world_size * T_cap * K
```

对于 `T_cap=4096, world_size=8, K=8`：

```text
T_recv_cap = 262144
```

主要 workspace 约数 GB，单层 benchmark 在 24GB 卡上可承受。

建议脚本中用一个轻量 config：

```python
from types import SimpleNamespace

config = SimpleNamespace(
    max_num_batched_tokens=max_T_local,
    moe_block_size_m=64,
    moe_impl="ep_ht",
    enforce_eager=True,
    hf_config=SimpleNamespace(torch_dtype=torch.bfloat16),
    moe_expert_overlap_enabled=bool(args.expert_overlap_path),
    moe_expert_overlap_path=args.expert_overlap_path,
)
backend = MoeBackend(config, E_global, K, H, N)
```

实例化模块：

```python
dispatch = DispatchEPHT(
    num_experts_global=E_global,
    top_k=K,
    block_size_m=BLOCK_M,
    norm_topk_prob=True,
    expert_placement=args.expert_placement,
    expert_placement_seed=args.expert_placement_seed,
    expert_placement_path=args.expert_placement_path,
    expert_overlap_enabled=bool(args.expert_overlap_path),
    expert_overlap_path=args.expert_overlap_path,
    expert_overlap_strategy=args.expert_overlap_strategy,
    drop_policy=drop_policy,
    drop_rate=drop_rate,
    drop_seed=args.drop_seed,
    router_keff=args.router_keff,
    layer_id=0,
).cuda()

experts = ExpertsEPHT(...).cuda()
combine = CombineEPHT(hidden_size=H, top_k=dispatch.K).cuda()
```

手动挂接 backend：

```python
dispatch.sorted_token_ids_buf = backend.sorted_token_ids_buf
dispatch.expert_ids_buf = backend.expert_ids_buf
dispatch.num_tokens_post_padded = backend.num_tokens_post_padded
dispatch.cumsum_buffer = backend.cumsum_buffer

experts.run_experts = backend.run_experts
```

权重初始化：

```python
with torch.no_grad():
    experts.w1.normal_(mean=0.0, std=0.02)
    experts.w2.normal_(mean=0.0, std=0.02)
```

说明：

- 权重内容不影响 GEMM 算法路径，但不能使用未初始化内存，避免 NaN / inf 污染。
- 只初始化一层，不加载真实 checkpoint。

### 4.4 Drop 环境

Tier 1 performance path 必须设置：

```text
MOE_DROP_IMPL=gpu
MOE_DROP_MIN_REPLICAS=0
MOE_DROP_GPU_STATS=0
MOE_PROFILE_OVERLAP_RUNTIME=0
MOE_PROFILE_ROUTING=0
MOE_RECORD_TIMING=0
```

其中：

- `MOE_DROP_MIN_REPLICAS=0` 强制所有 L 都跑 drop，便于画完整曲线。
- `MOE_DROP_GPU_STATS=0` 避免 stats path 的 `.item()` / `.cpu()` 同步。
- `MOE_PROFILE_OVERLAP_RUNTIME=0` 避免 `record_drop` 和 `record_dispatch` 影响结果。

脚本内部应在构造 `DispatchEPHT` 前设置这些 env。

### 4.5 待扫变量

主 sweep：

| 参数 | 取值 |
| --- | --- |
| `T_local` | `16,64,128,256,512,1024,2048,4096` |
| `K` | `8` |
| nominal `L_send = T_local*K` | `128,512,1024,2048,4096,8192,16384,32768` |
| `drop_rate` | `0.0,0.3` |
| `drop_policy` | `tail_weight` |
| `router_keff` | `0` |

附加对照：

| 参数 | 取值 |
| --- | --- |
| `router_keff` | `6` |
| `drop_policy` | `none` |
| `drop_rate` | `0.0` |

K_eff 只作为额外曲线，不用于 `tail_weight@0.3` 的拐点判定。

### 4.6 Routing 输入构造

Tier 1 支持两种 routing mode。

#### Mode A：`controlled_balanced`（建议先跑 smoke / 下界）

目标：使每个 rank 的 send/recv 分布尽量均衡，变量主要是 `L`。

做法：

```python
router_logits = torch.full((T_local, E_global), -10000.0, device=device)
for t in range(T_local):
    for k in range(K):
        dst = (rank + 1 + ((t * K + k) % (world_size - 1))) % world_size
        local_e = (t * K + k) % E_per_rank
        global_e = dst * E_per_rank + local_e
        router_logits[t, global_e] = 10.0 - 0.01 * k
```

特点：

- 主要用于 disjoint contiguous placement。
- top-K 全 remote，drop 有充足 droppable pool。
- 每个 token 全 remote 时，per-token protection 会保留最大 weight branch。
- 可测出 “drop 最可能有收益” 的近似下界。

#### Mode B：`random_logits_lbg`（主报告推荐）

目标：匹配当前 Phase 4 的真实 routing/overlap 路径。

做法：

- 固定使用 Phase 3 选定的 LBG overlap plan。
- 传入：

```text
--expert-overlap-path <phase3_lbg_overlap_plan.json>
--expert-overlap-strategy greedy_balance
```

- `router_logits = torch.randn(T_local, E_global, dtype=torch.float32)`。
- 每个 shape 使用固定 seed，baseline/drop 共用同一组 logits。

特点：

- 包含当前 `ExpertOverlapRouter.route()` CPU path。
- `L_recv` 不再由 `T*K` 完全决定，所以必须记录实际 row counts。
- 是更接近真实 Phase 4 栈的 Tier 1 曲线。

推荐执行顺序：

1. 先跑 `controlled_balanced`，确认脚本、计时和曲线合理。
2. 再跑 `random_logits_lbg`，作为最终 Tier 1 结论。

### 4.7 输入张量

每个 `(T_local, routing_seed)`：

```python
hidden_states = torch.randn(
    (T_local, H),
    device=device,
    dtype=torch.bfloat16,
    generator=gen,
)
router_logits = make_router_logits(...).to(torch.float32)
```

baseline 和 drop 使用同一份 `hidden_states/router_logits`。

注意：

- 不要在 timed region 内生成随机输入。
- 不要在 timed region 内重建模块或初始化权重。
- 可以为每个 `T_local` 预生成输入，避免重复分配造成噪声。

### 4.8 计时方法

每个 cell：

```text
cell = (routing_mode, T_local, drop_policy, drop_rate, router_keff)
```

执行：

1. `dist.barrier()`
2. warmup `warmup_iters` 次完整 block
3. `torch.cuda.synchronize()`
4. measured repeats

每个 repeat 记录：

```python
dist.barrier()
total_start.record()

d0.record()
tok_meta = dispatch(hidden_states, router_logits)
d1.record()

e0.record()
expert_out = experts(tok_meta)
e1.record()

c0.record()
out = combine(expert_out, tok_meta)
c1.record()

total_end.record()
torch.cuda.synchronize()
```

本 rank 得到：

```python
dispatch_us = d0.elapsed_time(d1) * 1000
experts_us = e0.elapsed_time(e1) * 1000
combine_us = c0.elapsed_time(c1) * 1000
total_us = total_start.elapsed_time(total_end) * 1000
L_recv = tok_meta.recv_hidden.shape[0]
L_send_kept = sum(tok_meta.send_counts)
```

跨 rank 聚合：

```python
local = torch.tensor([
    dispatch_us,
    experts_us,
    combine_us,
    total_us,
    float(L_recv),
    float(L_send_kept),
], device=device, dtype=torch.float64)
gathered = [torch.empty_like(local) for _ in range(world_size)]
dist.all_gather(gathered, local)
matrix = torch.stack(gathered)
```

每次 repeat 的分布式 step 指标：

```text
dispatch_us_rank_max = max_r dispatch_us[r]
experts_us_rank_max  = max_r experts_us[r]
combine_us_rank_max  = max_r combine_us[r]
total_us_rank_max    = max_r total_us[r]
L_recv_mean          = mean_r L_recv[r]
L_recv_max           = max_r L_recv[r]
L_recv_min           = min_r L_recv[r]
L_recv_cv            = std_r(L_recv) / mean_r(L_recv)
```

最终 cell 指标：

```text
median over repeats
p10/p90 over repeats
std over repeats
```

为什么保留 `total_us`：

- segment max 之和不一定等于 total max，因为每段最慢 rank 可能不同。
- 判断拐点以 `total_us_rank_max` 为准。
- segment 指标用于解释原因。

### 4.9 输出 CSV

输出文件：

```text
eval_results/prefill_drop_l_sweep_tier1/
  tier1_rows.csv
  tier1_rows.json
  tier1_summary.json
  tier1_total_us_vs_l_recv.png
  tier1_segments_vs_l_recv.png
```

CSV 字段：

```text
routing_mode
world_size
T_local
K
router_keff
L_send_nominal
drop_policy
drop_rate
drop_impl
drop_min_replicas
iters
warmup_iters
L_recv_mean_median
L_recv_max_median
L_recv_min_median
L_recv_cv_median
L_send_kept_mean_median
effective_drop_send_frac
dispatch_us_median
dispatch_us_p10
dispatch_us_p90
experts_us_median
experts_us_p10
experts_us_p90
combine_us_median
combine_us_p10
combine_us_p90
total_us_median
total_us_p10
total_us_p90
total_us_std
notes
```

`effective_drop_send_frac` 定义：

```text
1 - L_send_kept_mean(drop) / L_send_nominal
```

可额外计算：

```text
effective_drop_recv_frac =
  1 - L_recv_mean(drop) / L_recv_mean(baseline_same_T)
```

### 4.10 图

主图：

```text
x = L_recv_max_median of baseline
y = total_us_median
line 1 = baseline drop_rate=0.0
line 2 = tail_weight drop_rate=0.3
```

辅助图：

- `dispatch_us` vs `L_recv`
- `experts_us` vs `L_recv`
- `combine_us` vs `L_recv`
- `effective_drop_recv_frac` vs `L_recv`
- `L_recv_cv` vs `L_recv`

图上标出：

- 交叉点 `L*`；
- drop 比 baseline 慢的区域；
- drop 比 baseline 快的区域。

### 4.11 交叉点计算

对每个 `T_local` 找 baseline/drop pair：

```text
delta_us = total_us_drop - total_us_baseline
```

判定：

- 若存在相邻点 `delta_us[i] > 0` 且 `delta_us[i+1] <= 0`，线性插值得到 `L*`。
- 若所有点 `delta_us > 0`，无交叉。
- 若最小点已经 `delta_us <= 0`，交叉点小于 sweep 下界。

线性插值：

```python
x0, y0 = L[i], delta[i]
x1, y1 = L[i + 1], delta[i + 1]
L_star = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
```

保存：

```json
{
  "crossing_found": true,
  "L_star": 12345,
  "decision": "run_tier2_3_to_5_points",
  "baseline_points": [...],
  "drop_points": [...]
}
```

## 5. Tier 1 自动门控

按 `L*` 做决策：

| Tier 1 结果 | 决策 |
| --- | --- |
| `L* < 8192` | 拐点在易达区，跑 Tier 2，点位取 `{0.5x, 1x, 2x, 4x}` |
| `8192 <= L* < 32768` | 拐点在大 prefill 区，跑 Tier 2，但只跑 `{1x, 2x}` 或 VRAM 允许点 |
| `L* >= 32768` | 不跑完整 Tier 2，最多跑一个最大 prompt sanity |
| 全段无交叉 | 不跑 Tier 2，写负结论 |
| `L* < sweep_min` | drop 在 MoE block 内已有效，直接跑 Tier 2 多点验证 |

注意：

- 用 `L_recv_max` 判断保守拐点。
- 用 `L_recv_mean` 作为辅助报告。

## 6. Tier 2 详细实现

### 6.1 触发条件

Tier 2 仅在 Tier 1 证明 MoE block 内部有可达拐点时启动。

启动前需要得到：

```text
L_star_recv_per_rank
world_size
K
target T_local ~= L_star_recv_per_rank / K
target total prefill tokens ~= T_local * world_size
```

如果 Tier 1 使用 `L_recv_max`：

```text
target_total_tokens ~= L_star_recv_max
```

因为在 owner-local EP 中每 rank 处理约 `total_tokens / world_size` 个 source tokens，
每 rank send 约 `(total_tokens / world_size) * K` 个 replicas，均衡时每 rank recv
也约这个量。

### 6.2 推荐脚本

新增：

```text
eval/run_prefill_drop_len_sweep.py
```

不要强行复用 AIME/GSM runner，因为该 runner 的 dataset / scoring / prompt template
会引入无关变量。Tier 2 只需要 synthetic prompts 和 engine metrics。

脚本职责：

1. 初始化 SPMD `LLMEngine`。
2. 构造指定 batch / prompt_len 的 synthetic prompts。
3. 跑 baseline 和 `tail_weight@0.3`。
4. 每个点 warmup 1 次、repeat 3 次。
5. rank 0 写 CSV / JSON。

### 6.3 Prompt 构造

目标是控制 token 数，不关心语义。

推荐用 tokenizer 直接构造 token ids，避免自然语言长度不稳定：

```python
base_token = tokenizer.encode(" the", add_special_tokens=False)[0]
prompt_ids = [base_token] * prompt_len
engine.generate([prompt_ids] * batch, SamplingParams(max_tokens=1, temperature=0.0))
```

`LLMEngine.add_request` 已支持 `prompt: str | list[int]`，因此可以直接传 token ids。

### 6.4 点位选择

从 Tier 1 的 `L*` 得到：

```text
target_T_total = round_to_multiple(world_size, L_star / K * world_size)
```

实际扫描 token totals：

```text
{0.5x, 1x, 2x, 4x} * target_T_total
```

转换成 `(batch, prompt_len)`：

优先固定 batch 小一点，减少 scheduler 和 per-sequence overhead：

| total tokens | 推荐组合 |
| --- | --- |
| <= 4096 | batch=1, prompt_len=total |
| 4096 - 8192 | batch=2, prompt_len=total/2 |
| 8192 - 16384 | batch=4, prompt_len=total/4 |
| 16384 - 32768 | batch=4 或 8, prompt_len=total/batch |

约束：

```text
batch <= max_num_seqs
batch * prompt_len <= max_num_batched_tokens
prompt_len <= max_model_len
```

Tier 2 需要把 `max_num_batched_tokens` 和 `max_model_len` 设置到目标值以上。

### 6.5 VRAM 预算

粗略估计：

```text
KV per token ~= 48 layers * 2(K,V) * 8 kv heads * 128 dim * 2 bytes
             ~= 196 KB/token
```

24GB 卡中权重和 workspace 之外剩余可用量不稳定。建议上限：

```text
batch * prompt_len <= 32768
```

如果遇到 OOM：

1. 降低 batch，保持 total tokens 接近目标。
2. 降低 `gpu_memory_utilization` 反而可能减少可用 KV block，不优先。
3. 降低最大点位，保留 Tier 1 负/正结论。

### 6.6 Tier 2 命令模板

Baseline：

```bash
torchrun --nproc_per_node=8 \
  -m eval.run_prefill_drop_len_sweep \
  --model ~/models/Qwen3-30B-A3B \
  --moe-impl ep_ht \
  --runtime-mode owner_local_ep \
  --expert-overlap-path <LBG_OVERLAP_PLAN_JSON> \
  --expert-overlap-strategy greedy_balance \
  --drop-policy none \
  --drop-rate 0 \
  --batch-sizes 1,2,4 \
  --prompt-lens 2048,4096,8192 \
  --max-new-tokens 1 \
  --repeats 3 \
  --warmup 1
```

Drop：

```bash
MOE_DROP_IMPL=gpu MOE_DROP_MIN_REPLICAS=0 MOE_DROP_GPU_STATS=0 \
torchrun --nproc_per_node=8 \
  -m eval.run_prefill_drop_len_sweep \
  --model ~/models/Qwen3-30B-A3B \
  --moe-impl ep_ht \
  --runtime-mode owner_local_ep \
  --expert-overlap-path <LBG_OVERLAP_PLAN_JSON> \
  --expert-overlap-strategy greedy_balance \
  --drop-policy tail_weight \
  --drop-rate 0.3 \
  --batch-sizes 1,2,4 \
  --prompt-lens 2048,4096,8192 \
  --max-new-tokens 1 \
  --repeats 3 \
  --warmup 1
```

实际 CLI 名称可沿用现有 config：

```text
--moe-drop-policy
--moe-drop-rate
--moe-router-keff
```

但新脚本的外层参数建议保持简短，内部映射到 `LLMEngine` config。

### 6.7 Tier 2 输出字段

CSV：

```text
case
drop_policy
drop_rate
drop_impl
batch
prompt_len
total_prompt_tokens
max_new_tokens
repeat
prefill_tokens
prefill_time_s
prefill_tok_s
decode_tokens
decode_time_s
e2e_total_time_s
peak_memory_allocated_gb
peak_memory_reserved_gb
pass
error
```

聚合：

```text
prefill_time_s_median
prefill_time_s_p10
prefill_time_s_p90
prefill_tok_s_median
drop_speedup = baseline_prefill_time / drop_prefill_time
```

## 7. 风险与诊断

### 7.1 Tier 1 有收益，Tier 2 无收益

可能原因：

- attention 占 prefill 大头，MoE block 节省被稀释；
- full model 中 routing 分布比 synthetic 更不利；
- KV cache / scheduler / tokenizer overhead 掩盖；
- full model overlap route CPU path 随 token 数增长明显。

处理：

- 报告 MoE-block speedup 和 full-prefill speedup 的稀释比例。
- 如果 Tier 1 `experts_us` 明显下降但 Tier 2 无收益，说明 drop 不是当前端到端瓶颈。

### 7.2 dispatch_us 变慢抵消 experts_us

可能原因：

- drop 后 ragged split 更不均；
- `torch.nonzero + argsort(tr_kept)` 成为瓶颈；
- payload all-to-all 小消息碎片化；
- count a2a 固定 latency 主导。

处理：

- 看 `dispatch_us` segment。
- 看 `L_recv_cv`。
- 如果 drop 让 `L_recv_cv` 增大很多，说明 tail_weight 不是通信友好策略。

### 7.3 combine_us 变慢

可能原因：

- reverse a2a shape 更不均；
- `unperm = torch.zeros(T*K,H)` 仍按原始 K 分配和清零，drop 后没有完全减少 combine 成本；
- `unperm[tok_meta.sort_perm] = rev` scatter 成本随原始 `T*K` 或 memory bandwidth 走。

处理：

- 在 Tier 1 中同时记录 `combine_us`。
- 如 combine 变成瓶颈，后续可考虑 sparse combine 或直接 compact reduce。

### 7.4 effective drop fraction 不到 0.3

原因：

- local replica 永不 drop；
- all-remote token 要保留最大 branch；
- droppable pool 小于 `round(0.3*T*K)`。

处理：

- CSV 中必须记录 `effective_drop_send_frac` 和 `effective_drop_recv_frac`。
- 如果 effective drop 只有 0.1，不能用它解释 `drop_rate=0.3` 的预期收益。

### 7.5 K_eff 对照跑不通

原因：

- full model `FusedMoE` 中 `CombineEPHT` 的 top_k 可能仍是 model K，而
  `DispatchEPHT` 使用 K_eff。

处理：

- Tier 1 standalone 用 `CombineEPHT(top_k=dispatch.K)`。
- Tier 2 K_eff 前先修 full model 维度：
  - 或让 `CombineEPHT` 从 `tok_meta.topk_weights.shape[1]` 推断 K；
  - 或 `FusedMoE` 构造 combine 时传 `self.dispatch.K`。

## 8. 不做的事

本轮不做：

- 不扫多 placement，主报告固定当前 Phase 4 最优 LBG overlap plan。
- 不扫多 drop policy，主策略固定 `tail_weight`。
- 不跑 GSM/AIME accuracy，prefill L sweep 是性能边界实验。
- 不引入 Triton fused drop kernel。
- 不 GPU 化 overlap route。
- 不改 EP-LL。

可选对照：

- `router_keff=6`，作为“直接减少 K 是否更有效”的 side curve。
- `controlled_balanced`，作为干净 routing 的 lower-bound curve。

## 9. 实施步骤

### Step 0：确认当前 GPU drop 快路径

运行：

```bash
/home/lzy/miniconda3/envs/vllm/bin/python \
  -m workshop.nanovllm_moe._test_expert_drop_gpu
```

确认：

- `tail_weight` GPU/CPU parity 通过。
- `cross_numa_first` GPU/CPU parity 通过。
- `random` reproducibility 通过。
- small batch bypass 通过。

### Step 1：实现 Tier 1 脚本

新增：

```text
workshop/nanovllm_moe/_bench_prefill_drop_sweep.py
```

最小 CLI：

```text
--output-dir
--t-local-values
--drop-policy
--drop-rates
--routing-mode controlled_balanced|random_logits_lbg
--expert-placement
--expert-placement-path
--expert-overlap-path
--expert-overlap-strategy
--router-keff
--warmup-iters
--iters
--seed
```

### Step 2：Tier 1 smoke

2 卡 smoke：

```bash
torchrun --nproc_per_node=2 \
  -m workshop.nanovllm_moe._bench_prefill_drop_sweep \
  --t-local-values 16,64 \
  --drop-rates 0,0.3 \
  --iters 2 \
  --warmup-iters 1 \
  --routing-mode controlled_balanced \
  --output-dir /tmp/prefill_drop_sweep_smoke
```

通过条件：

- 所有 rank 正常退出。
- CSV 有 baseline/drop rows。
- `L_recv_*` 非零。
- drop row 的 `L_send_kept_mean` 小于 baseline。

### Step 3：Tier 1 主 sweep

8 卡 controlled：

```bash
torchrun --nproc_per_node=8 \
  -m workshop.nanovllm_moe._bench_prefill_drop_sweep \
  --routing-mode controlled_balanced \
  --t-local-values 16,64,128,256,512,1024,2048,4096 \
  --drop-policy tail_weight \
  --drop-rates 0,0.3 \
  --iters 10 \
  --warmup-iters 5 \
  --output-dir eval_results/prefill_drop_l_sweep_controlled
```

8 卡 LBG：

```bash
torchrun --nproc_per_node=8 \
  -m workshop.nanovllm_moe._bench_prefill_drop_sweep \
  --routing-mode random_logits_lbg \
  --expert-overlap-path <LBG_OVERLAP_PLAN_JSON> \
  --expert-overlap-strategy greedy_balance \
  --t-local-values 16,64,128,256,512,1024,2048,4096 \
  --drop-policy tail_weight \
  --drop-rates 0,0.3 \
  --iters 10 \
  --warmup-iters 5 \
  --output-dir eval_results/prefill_drop_l_sweep_lbg
```

主结论以 LBG sweep 为准。

### Step 4：自动分析

Tier 1 脚本结束时 rank 0 自动：

- 生成图。
- 计算 `delta_us = drop - baseline`。
- 计算 `L*`。
- 写 `tier1_summary.json`。
- 打印建议：

```text
DECISION: RUN_TIER2_FULL_PREFILL
reason: L*=6144 < 8192
suggested points: ...
```

或：

```text
DECISION: STOP_NEGATIVE
reason: no crossing up to L_recv_max=32768
```

### Step 5：实现 Tier 2 脚本

仅当 Tier 1 触发。

新增：

```text
eval/run_prefill_drop_len_sweep.py
```

输入 Tier 1 summary：

```text
--tier1-summary eval_results/prefill_drop_l_sweep_lbg/tier1_summary.json
```

脚本根据 `L*` 生成 `(batch,prompt_len)` 点位，也允许手动覆盖：

```text
--points 1x,2x,4x
--batch-choices 1,2,4,8
--max-total-tokens 32768
```

### Step 6：Tier 2 主 sweep

只跑必要点位，baseline/drop 成对。

通过条件：

- 每个点 3 reps。
- 没有 OOM。
- `prefill_time_s_median` 可比较。
- 结论说明 Tier 1 MoE speedup 到 Tier 2 prefill speedup 的稀释比例。

## 10. 最终报告结构

写入主报告时按以下结构：

1. 问题：
   - GPU drop 在 decode-heavy workload 不理想。
   - 需要确定 drop 的适用规模。
2. Tier 1 设置：
   - shape、routing mode、drop policy、rate、EP-HT、world size。
3. Tier 1 曲线：
   - total vs `L_recv_max`。
   - dispatch / experts / combine 分段。
4. 拐点：
   - `L*` 是否存在。
   - 若不存在，最大测到哪里。
5. Tier 2 设置和结果：
   - 若运行，列 full prefill speedup。
   - 若未运行，说明门控原因。
6. 结论：
   - drop 是否只适合大 prefill。
   - 当前 24GB / decode-heavy 负结果是否完整。
   - 下一步是否转向 K_eff、receive-side drop 或 route GPU 化。

