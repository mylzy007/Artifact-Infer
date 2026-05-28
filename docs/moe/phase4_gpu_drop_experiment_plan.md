# Phase 4 GPU Drop Experiment Plan

本文档定义 Phase 4 token-replica drop 的 GPU 化实验方案。目标是把当前
`apply_drop` 的 device-to-host 同步和 Python 排序移出 decode critical path，
验证 drop 在纯 device-side 决策下是否能真正转化为 e2e 收益。

## 1. 当前结论

当前 EP-HT drop 的主要瓶颈不是 drop 比例本身，而是每层每次 forward 的
host-side 固定开销。

代码路径：

- `DispatchEPHT.forward` 在 `topk + route` 后进入 Phase 4 drop：
  - `workshop/nanovllm_moe/artifacts/modeling/layers/moe/dispatch_ep_ht.py`
- drop 策略实现：
  - `workshop/nanovllm_moe/services/utils/expert_drop.py`
- drop 后 Combine 不需要改：
  - `workshop/nanovllm_moe/artifacts/modeling/layers/moe/combine_ep_ht.py`

当前 `apply_drop` 的同步点：

- `flat_topk_w.sum().item()`，一次 device-to-host sync。
- `flat_expert_ids.to("cpu").tolist()`，一次 device-to-host sync。
- `target_rank.to("cpu").tolist()`，一次 device-to-host sync。
- `flat_topk_w.to("cpu").tolist()`，一次 device-to-host sync。
- Python 中完成 per-token 保护、分组、排序、抽样。
- `torch.tensor(keep_mask_list, device=device)`，一次 host-to-device copy。

需要区分两类 host sync：

- drop 引入的额外同步，本文档 P0 要消除。
- EP-HT variable-size dispatch 固有同步：
  `send_counts_t.tolist()` 和 `recv_counts_t.tolist()` 仍然存在，因为
  `all_to_all_single` 的 split sizes 需要 Python int list。GPU drop 不会消除
  这一点。

如果启用了 expert overlap plan，当前 `ExpertOverlapRouter.route()` 也会把
`flat_expert_ids` 搬到 CPU 并用 Python 选择 target rank。这不是 drop 的增量
开销，但会继续影响绝对性能。本文档 P0 先不处理 route GPU 化。

## 2. 实验目标

P0 目标：

- 将 `tail_weight`、`random`、`cross_numa_first` 三个策略改成 GPU-only drop。
- 保留当前语义：
  - `target_rank == source_rank` 的 local replica 永不 drop。
  - 如果某个 token 的 top-K 全是 remote replica，保留该 token 中
    `router_weight` 最大的一个 replica。
  - drop budget 仍按 `round(drop_rate * T * K)` 定义。
- drop 决策路径不使用 `.tolist()`、`.cpu()`、`.item()`。
- profiling 关闭时，drop 不产生任何 host sync。
- 保持 `CombineEPHT` 不变。
- 增加 small batch bypass，避免 decode 小 batch 下固定开销大于节省。

P1 目标：

- 增加 threshold-based drop，作为 `tail_weight` 的低开销近似替代。
- 增加 router 内部 per-token K_eff drop 作为对照实验。

P2 目标：

- GPU 化 grouped policies：
  - `per_expert_uniform`
  - `hot_expert_relief`
  - `per_expert_tailtoken`
  - `hotspot_relief`
- 只在 P0 证明 device-side drop 有收益后投入。

## 3. 非目标

本轮不处理：

- EP-HT split-size list 的 host sync。
- `ExpertOverlapRouter.route()` 的 CPU 路径。
- EP-LL drop 路径。
- Triton fused kernel。
- 真正 distributed global 的 hot expert 或 hot rank drop。

当前策略名字里的“全局”只表示本 rank 当前 forward 内的全局预算或全局排序，
不是跨 rank 收集所有 token 后再决策。P0 只做 source-rank local drop。

## 4. P0 设计

### 4.1 数据定义

设：

```text
T = hidden_states.shape[0]
K = self.K
L = T * K
flat_expert_ids: [L]
target_rank:     [L]
flat_topk_w:     [L]
source_rank:     int
world_size:      int
```

device-side mask：

```python
is_remote = target_rank != source_rank
is_local = ~is_remote
```

per-token 保护：

```python
weights_TK = flat_topk_w.view(T, K)
remote_TK = is_remote.view(T, K)
fully_remote_token = remote_TK.all(dim=1)
best_k = weights_TK.argmax(dim=1)
protected_idx = torch.arange(T, device=device) * K + best_k

protected = torch.zeros(L, dtype=torch.bool, device=device)
protected[protected_idx[fully_remote_token]] = True
```

可 drop 集合：

```python
droppable = is_remote & ~protected
```

注意：不能为了 cap budget 读取 `droppable.sum().item()`。`torch.topk` 的 `k`
必须是 Python int，因此只使用 `round(drop_rate * L)` 得到预算，然后把不可 drop
位置的 score 设为 `inf`。如果预算大于实际 droppable 数，topk 可能选到 `inf`
位置，最后用 `torch.isfinite(score[idx])` 过滤即可。这样不需要 device-to-host
同步。

### 4.2 通用 topk drop helper

建议新增 helper：

```python
def _select_drop_by_score(
    score: torch.Tensor,       # [L], lower means dropped first
    droppable: torch.Tensor,   # [L] bool
    budget: int,
) -> torch.Tensor:
    keep_mask = torch.ones_like(droppable)
    if budget <= 0 or score.numel() == 0:
        return keep_mask

    k = min(int(budget), int(score.numel()))
    masked_score = score.float().masked_fill(~droppable, float("inf"))
    vals, idx = torch.topk(masked_score, k=k, largest=False, sorted=False)
    idx = idx[torch.isfinite(vals)]
    keep_mask[idx] = False
    return keep_mask
```

这个 helper 是 P0 三个策略共用路径。

### 4.3 `tail_weight`

语义：本 rank 内所有 droppable remote replica 按 router weight 升序，丢最小的
`round(drop_rate * L)` 个。

GPU 实现：

```python
score = flat_topk_w.float()
keep_mask = _select_drop_by_score(score, droppable, budget)
```

### 4.4 `random`

语义：本 rank 内从 droppable remote replica 中均匀采样。

GPU 实现：

```python
rand = torch.rand((L,), device=device, dtype=torch.float32, generator=generator)
keep_mask = _select_drop_by_score(rand, droppable, budget)
```

随机性处理：

- 在 `DispatchEPHT.__init__` 中保留当前 Python RNG 给 CPU fallback。
- 新增 per-layer CUDA generator，例如 `_drop_torch_generator`。
- seed 使用 `drop_seed + layer_id * 1009`，和当前层间独立的设计一致。
- GPU random 不要求和旧 CPU random 逐元素一致，但同 seed 下应可复现。

### 4.5 `cross_numa_first`

语义：优先 drop 跨 NUMA remote replica，额度不够时再 drop 同 NUMA remote replica。
每个组内仍按 router weight 升序。

GPU 实现：

```python
split = max(1, world_size // 2)
src_numa = min(source_rank // split, 1)
dst_numa = torch.div(target_rank, split, rounding_mode="floor").clamp_max(1)
is_cross_numa = is_remote & (dst_numa != src_numa)

bias = 1.0e6
score = flat_topk_w.float() + (~is_cross_numa).to(torch.float32) * bias
keep_mask = _select_drop_by_score(score, droppable, budget)
```

注意：如果使用最小 score 优先 drop，跨 NUMA 不能加正 bias。应当给同 NUMA
remote 加大 bias，或者给跨 NUMA 减 bias。上面的写法是“跨 NUMA 先吃”。

### 4.6 Small Batch Bypass

decode 下常见 `T <= max_num_seqs`，例如 `T=8, K=8, L=64`。这种规模下即使
GPU topk 很快，也可能不如直接 bypass。

建议新增参数：

```text
MOE_DROP_MIN_REPLICAS=128
```

行为：

```python
if L <= min_replicas:
    skip drop
```

实验中需要同时跑：

- `MOE_DROP_MIN_REPLICAS=0`，验证纯 GPU drop 开销。
- `MOE_DROP_MIN_REPLICAS=64`，跳过极小 decode batch。
- `MOE_DROP_MIN_REPLICAS=128`，更保守地跳过小 batch。

默认建议先用 `128`，因为当前 decode-heavy workload 下 `L=64` 是常见情况。

## 5. 统计路径设计

这是 P0 最容易误判的地方。

当前 Phase 4 sweep 在 `run_owner_local_ep_phase4_drop.py` 中对 drop case 设置
`overlap_runtime_enabled=True`。这会调用 `overlap_runtime_stats.record_drop`，
而当前 `DropResult` 包含大量 Python int、float、dict 统计。

如果 GPU drop 为了填这些字段继续 `.item()` 或 `.cpu()`，性能 sweep 会重新
引入同步，掩盖 P0 收益。

建议分成两种运行模式：

### 5.1 Fast path

用于真实性能实验：

- `MOE_PROFILE_OVERLAP_RUNTIME=0`
- `DropResult` 只需要携带 `keep_mask`。
- 其他统计字段可以填 0 或空 dict，因为 `record_drop` 在 profiling disabled 时
  不读取这些字段。
- `apply_drop_gpu` 中禁止 `.item()`、`.cpu()`、`.tolist()`。

### 5.2 Stats path

用于 drop fraction、remote fraction、weight mass 等统计：

- `MOE_PROFILE_OVERLAP_RUNTIME=1`
- 可以在 drop 后用 `drop_mask` 计算统计。
- 允许少量同步，因为这是统计 sweep，不作为 e2e 性能结论。

建议新增实验脚本开关：

```text
--profile-drop-runtime-stats 0|1
```

或者在 Phase 4 sweep 中分两遍：

- performance sweep：overlap runtime stats off。
- accounting sweep：overlap runtime stats on，只跑少量问题。

## 6. 代码改动计划

### 6.1 `expert_drop.py`

新增：

- `apply_drop_gpu_simple(...)`
- `_build_droppable_mask(...)`
- `_select_drop_by_score(...)`
- `_drop_result_fast(...)`
- `_drop_result_with_stats(...)`

保留：

- 当前 `apply_drop(...)` CPU 实现作为 fallback。
- grouped policies 仍使用 CPU fallback，直到 P2。

建议入口：

```python
def apply_drop(..., impl: str = "auto") -> DropResult:
    if impl in ("gpu", "auto") and drop_policy in GPU_SIMPLE_POLICIES:
        return apply_drop_gpu_simple(...)
    return apply_drop_cpu(...)
```

环境变量：

```text
MOE_DROP_IMPL=auto|gpu|cpu
MOE_DROP_MIN_REPLICAS=128
MOE_DROP_GPU_STATS=0|1
```

`auto` 语义：

- P0 simple policies 用 GPU。
- grouped policies 走 CPU fallback。
- `drop_policy=none` 直接 no-op。

### 6.2 `dispatch_ep_ht.py`

改动点：

- 初始化 `self.drop_impl` 和 `self.drop_min_replicas`。
- 初始化 CUDA generator 或延迟到第一次 forward 时按 device 初始化。
- 调 `apply_drop` 时传入 `impl`、`min_replicas`、`torch_generator`。
- 保留现有后处理：
  - `keep_mask.view(T, K)`
  - topk weight renorm
  - `torch.nonzero(keep_mask)`
  - `argsort(tr_kept)`
  - `record_drop(...)`

不改：

- payload all-to-all。
- reverse all-to-all。
- `CombineEPHT`。

### 6.3 Phase 4 runner

建议对 `eval/run_owner_local_ep_phase4_drop.py` 增加：

```text
--drop-impl auto|gpu|cpu
--drop-min-replicas 128
--profile-drop-runtime-stats 0|1
```

并通过 `extra_env` 写入：

```python
{
    "MOE_DROP_IMPL": args.drop_impl,
    "MOE_DROP_MIN_REPLICAS": str(args.drop_min_replicas),
}
```

性能 sweep 不应默认开启 overlap runtime stats。统计 sweep 可以开启。

## 7. 测试计划

### 7.1 Unit tests

新增测试文件：

```text
workshop/nanovllm_moe/_test_expert_drop_gpu.py
```

测试矩阵：

| case | 目标 |
| --- | --- |
| all local | 不 drop 任何 replica |
| all remote, K=1 | 每个 token 至少保留 1 个，因此不能 drop |
| all remote, K=8 | 每个 token 保护 weight 最大 replica |
| mixed local/remote | local replica 永不 drop |
| budget > droppable | 只 drop droppable，不误伤 protected/local |
| tail_weight deterministic | GPU keep_mask 与 CPU keep_mask 一致 |
| cross_numa_first deterministic | GPU keep_mask 与 CPU keep_mask 一致 |
| random constraints | drop 数、local/protected 约束正确，同 seed 可复现 |

`tail_weight` 和 `cross_numa_first` 应要求 CPU/GPU exact mask parity。
`random` 不要求和旧 CPU RNG parity，只要求同 seed GPU 可复现和约束正确。

### 7.2 Dispatch smoke tests

使用现有 EP-HT smoke：

```bash
WORLD_SIZE=2 NUM_LAYERS=2 MOE_IMPL=ep_ht ENFORCE_EAGER=1 \
MOE_DROP_IMPL=gpu MOE_DROP_POLICY=tail_weight MOE_DROP_RATE=0.1 \
python -m workshop.nanovllm_moe._test_engine_ep_ll
```

8 GPU smoke：

```bash
WORLD_SIZE=8 NUM_LAYERS=2 MOE_IMPL=ep_ht ENFORCE_EAGER=1 \
MOE_DROP_IMPL=gpu MOE_DROP_POLICY=tail_weight MOE_DROP_RATE=0.1 \
python -m workshop.nanovllm_moe._test_engine_ep_ll
```

如果环境变量没有被该 test 读取，则改用 `eval/test_bazaar_moe.py` 的
`--moe-drop-policy` 和 `--moe-drop-rate` 参数。

## 8. Microbenchmark 计划

新增脚本：

```text
eval/bench_expert_drop_gpu.py
```

输入维度：

| T | K | L |
| --- | --- | --- |
| 1 | 8 | 8 |
| 8 | 8 | 64 |
| 16 | 8 | 128 |
| 64 | 8 | 512 |
| 256 | 8 | 2048 |
| 1024 | 8 | 8192 |

策略：

- `tail_weight`
- `random`
- `cross_numa_first`

drop rate：

- `0.05`
- `0.10`
- `0.20`

实现：

- CPU current
- GPU P0
- GPU P0 with stats enabled
- GPU P0 with small batch bypass

指标：

- median latency in microseconds，使用 CUDA event。
- p90 latency。
- 是否出现 host sync，使用 Nsight Systems 抽样确认。
- keep_mask parity 或约束检查。

预期：

- `L <= 64` 时 GPU topk 未必明显优于 no-op，因此 small batch bypass 很重要。
- `L >= 128` 时 GPU P0 应明显快于 CPU current。
- stats enabled 版本不作为性能结论。

## 9. E2E 实验计划

### 9.1 性能 sweep

目的：测 GPU drop 是否消除当前 +10% 到 +25% 的 e2e 退化。

关键设置：

```text
MOE_DROP_IMPL=gpu
MOE_PROFILE_OVERLAP_RUNTIME=0
MOE_PROFILE_ROUTING=0
```

推荐命令模板：

```bash
python eval/run_owner_local_ep_phase4_drop.py \
  --impl ep_ht \
  --overlap 0.25 \
  --drop-policies tail_weight,random,cross_numa_first \
  --drop-rates 0.05,0.1,0.2 \
  --num-repeats 3 \
  --warmup-cases 1 \
  --prompt-style reasoning_brief \
  --score-mode answer_first \
  --max-num-seqs 8 \
  --max-tokens 96
```

如果 runner 尚未支持关闭 overlap stats，需要先加开关，否则该 sweep 不是纯性能
结果。

建议 case 组合：

| case | drop impl | min replicas | 目的 |
| --- | --- | --- | --- |
| baseline | none | n/a | 原始 EP-HT |
| CPU drop | cpu | 0 | 复现实验中的慢路径 |
| GPU drop | gpu | 0 | 验证 GPU 化本身 |
| GPU drop + bypass64 | gpu | 64 | 验证极小 batch bypass |
| GPU drop + bypass128 | gpu | 128 | 当前 decode-heavy 推荐配置 |

成功标准：

- `GPU drop + bypass` 的 e2e 不应慢于 baseline 超过 2%。
- 若 drop 真的节省 compute/a2a，decode tok/s 应高于 baseline。
- `tail_weight` 和 `cross_numa_first` 的 score 不应明显低于当前 CPU 策略同 rate。
- `random` 主要作为 noise floor，不要求质量好。

### 9.2 统计 sweep

目的：记录 drop fraction、remote fraction、cross NUMA fraction、weight mass。

关键设置：

```text
MOE_DROP_IMPL=gpu
MOE_PROFILE_OVERLAP_RUNTIME=1
```

只跑少量问题，例如：

```bash
python eval/run_owner_local_ep_phase4_drop.py \
  --impl ep_ht \
  --overlap 0.25 \
  --drop-policies tail_weight,cross_numa_first \
  --drop-rates 0.05,0.1,0.2 \
  --num-repeats 1 \
  --num-problems 32 \
  --max-tokens 96 \
  --max-num-seqs 8
```

统计 sweep 的 e2e 只用于 sanity check，不纳入性能结论。

## 10. P1 实验

### 10.1 Threshold-based drop

目标：避免 `topk` 排序开销，用阈值比较近似 drop。

版本 A：runtime quantile。

```python
score = flat_topk_w.float().masked_fill(~droppable, float("inf"))
threshold = torch.quantile(score[droppable], drop_rate)
keep_mask = (score > threshold) | ~droppable
```

注意：`score[droppable]` 会产生 dynamic compact tensor，`torch.quantile` 也未必比
`topk` 快。它是实验项，不应先替代 P0。

版本 B：calibrated threshold。

- warmup 阶段按 layer 记录低权重分布。
- runtime 使用固定或 EMA threshold。
- 优点是 runtime 只需要一次比较。
- 缺点是 drop ratio 不严格，且不同 layer、prefill、decode 的分布可能不同。

验收：

- drop fraction 允许相对目标有小误差，例如 ±2 个百分点。
- e2e 需要优于 GPU topk 版本。

### 10.2 Router K_eff drop

目标：直接减少每 token 的有效 top-K，绕开 keep_mask 路径。

实现方向：

```python
topk_vals, topk_ids = torch.topk(logits_fp32, K, dim=-1)
topk_weights = torch.softmax(topk_vals, dim=-1)

K_eff = 6
active = torch.arange(K, device=device).view(1, K) < K_eff
topk_weights = topk_weights * active.to(topk_weights.dtype)
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
```

更彻底的版本是直接只 route `K_eff` 个 branch，这样才能减少 route、sort、
bincount、a2a、expert GEMM 和 combine 的行数。单纯把 weight 置 0 但仍发送
`K=8` 个 branch，不会节省 dispatch。

语义差异：

- 这是 per-token drop，不是按 expert 或 rank 减压。
- 不能表达 hot expert relief。
- 适合作为“减少有效专家数是否有收益”的强对照。

## 11. P2 实验

### 11.1 Grouped topk GPU 化

适用：

- `per_expert_uniform`
- `hot_expert_relief`
- `per_expert_tailtoken`
- `hotspot_relief`

推荐先实现通用 segmented selection：

1. 用 group id 构造 key：
   - expert group：`flat_expert_ids`
   - dst rank group：`target_rank`
2. 对 key 和 score 做 lexicographic sort。
3. `bincount` 得到每组长度。
4. 根据每组 quota 构造组内 rank。
5. `group_rank < quota[group]` 的位置 drop。

风险：

- `torch.argsort` 全量排序的 kernel launch 和 O(L log L) 成本可能不低。
- quota rounding 要和 CPU 语义对齐。
- grouped policies 在当前 decode 小 batch 下可能没有足够规模摊销复杂度。

### 11.2 Fused Triton kernel

只建议在 P0/P1 有正收益后考虑。

优先对象：

- `tail_weight`
- `random`
- `cross_numa_first`

不建议第一版 fused grouped policies，因为 grouped topk 在 Triton 中复杂度高，
调试成本大。

## 12. 结果记录模板

每次实验记录：

```text
date:
git commit:
dirty diff summary:
model:
gpu:
world_size:
runtime mode:
moe impl:
overlap plan:
drop impl:
drop policy:
drop rate:
drop min replicas:
profile overlap runtime:
num problems:
max tokens:
max num seqs:
score:
e2e_s:
prefill_tok_s:
decode_tok_s:
drop_fraction_total:
drop_fraction_remote:
drop_fraction_cross_numa:
dropped_weight_mass_fraction:
notes:
```

性能结论必须标明 profiling 是否开启。profiling on 的结果只能说明统计，不说明
真实 e2e 性能。

## 13. 判定标准

P0 继续推进的条件：

- GPU `tail_weight` 和 `cross_numa_first` unit parity 通过。
- profiling off 时，microbenchmark 中 GPU drop 明显快于 CPU drop。
- e2e 中 `GPU drop + bypass` 不再出现当前 CPU drop 的固定退化。
- 至少一个策略在 score 可接受的情况下提升 decode tok/s 或降低 e2e。

P0 停止或转向的条件：

- 即使 GPU drop 和 bypass 后，e2e 仍然慢于 baseline 超过 5%。
- Nsight 显示瓶颈转移到 EP-HT split sync、route CPU path 或 NCCL latency。
- drop 后减少的 rows 太少，expert GEMM/a2a 不在当前 workload 的主瓶颈上。

若 P0 不带来收益，下一步优先级应改为：

1. route GPU 化或 route 策略简化。
2. router K_eff drop。
3. receive-side drop，仅验证 compute-bound prefill 场景。
4. EP-LL 或 fixed bucket 路径，只作为单独路线评估。

## 14. 推荐执行顺序

1. 写 unit tests，先覆盖 CPU current 语义。
2. 实现 GPU helper 和 P0 三个策略。
3. 跑 unit parity。
4. 跑 microbenchmark，确认 sync 已消除。
5. 给 Phase 4 runner 增加 drop impl、small batch bypass、profiling 分离开关。
6. 跑 performance sweep，profiling off。
7. 跑 accounting sweep，profiling on，少量问题。
8. 汇总 P0 结论。
9. 决定是否进入 threshold 或 router K_eff。

