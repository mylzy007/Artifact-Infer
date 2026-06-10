# 采集 Schema — per-step / per-layer / per-rank 结构数据(无时序)

- Date: 2026-05-29
- Stage: 1(GATE A 前)
- 决策上下文: GATE Pre-Stage Q2 = **方案 B** —— 生产采集**只出结构数据**,时序通道交给受控 tier1-style microbench(本任务范围外)。本 schema 据此**移除所有时序字段**(`dispatch_time_ms / combine_time_ms / expert_time_ms`),但保留 D1-I10 后续配 microbench 所需的全部 byte / L_recv 原语。
- 方法论依据: `system-profile` skill —— 落在 "Interconnect & communication" 维度;遵守 Step 4.4 "minimize observer effect"(只记录**已经 host-sync 的量**,零新增 GPU 同步)。
- 输出格式: jsonl,每 rank 一个文件 `per_step_trace_<run_id>_rank<r>.jsonl`,每条记录 = 一个 (step, layer, rank) 三元组。
- 默认输出目录: `docs/aris/idea-expansion/raw_collection/`(env `MOE_COLLECT_OUT_DIR` 可覆盖)。

---

## 设计原则:为什么这些字段是"零新增同步"

`dispatch_ep_ht.py` 的关键事实(已在 03_codebase_check.md 核验):
- `send_counts` / `recv_counts` 在 `:365-366` **已经 `.tolist()` host-sync** 成 Python list —— 记录它们不引入任何新同步。
- `total_recv = sum(recv_counts)`(`:383`)、`L_send = sum(send_counts)` 同理纯 host 端。
- `T`(num_tokens)= `hidden_states.shape[0]`、`H` = `hidden_states.shape[1]`、`elem_size` = `element_size()` —— 纯 shape 读取,无同步。
- `phase` 来自 `get_context().is_prefill` —— 纯 Python bool,无同步。

→ **整条记录的全部字段都来自已 host-sync 或纯 CPU 的量,record 调用对 GPU 关键路径零新增 `cuda.synchronize()`。** 这正是方案 B + system-profile 4.4 的要求。

`routing entropy` 不在循环内计算(那需要对 128-expert 直方图做 bincount + `.item()`,会引入同步)。改为**离线从 `send_counts` 派生**(per-peer dispatch 集中度熵),零成本。若日后需要 expert 级 per-step 熵,需单独加同步字段 —— 已在末尾"已知局限"标出。

---

## 字段表

| 字段 | 类型 | 来源 | system-profile 标准? | 用途 |
|---|---|---|---|---|
| `run_id` | str | env `MOE_COLLECT_RUN_ID` | 标准(provenance) | 区分 workload |
| `step_id` | int | recorder 内 layer-rollover 推导 | **D2-I5 专用** | per-step gate replay 的时间轴 |
| `layer_id` | int | `self.layer_id` | 标准 | per-layer L*_ℓ 校准 |
| `rank_id` | int | `self.rank` | 标准 | per-rank 聚合 |
| `phase` | str `"prefill"`/`"decode"`/`"unknown"` | `get_context().is_prefill` | **D2-I5 专用** | 分相;验证 decode 段 L_recv≪L* |
| `is_warmup` | bool | prefill-onset 计数 vs `MOE_COLLECT_WARMUP_BATCHES` | **下游过滤(GATE A/Stage2 加)** | 干净剔除 warmup step,不留给下游猜 |
| `batch_index` | int (1-based) | prefill-onset 计数 | provenance | 第几个 batch;0 = 首个 prefill 前 |
| `num_tokens` | int | `T = hidden.shape[0]` | 标准 | 本 rank 本 step token 数 |
| `L_send` | int | `sum(send_counts)` | **D1-I10/D2-I5** | 本 rank 发出的 kept 副本行数 |
| `L_recv` | int | `total_recv = sum(recv_counts)` | **D1-I10/D2-I5 核心** | gate 判据 + 回归自变量基底 |
| `send_counts` | list[int] (len=N) | `:365` 已 host-sync | **D1-I10 核心** | per-peer dispatch 行数 → worst-peer dispatch bytes |
| `recv_counts` | list[int] (len=N) | `:366` 已 host-sync | **D1-I10 核心** | per-peer 行数 → worst-peer **combine** bytes(combine 是 reverse a2a,用 recv_counts) |
| `hidden_size` | int | `H = hidden.shape[1]` | **D1-I10** | bytes = rows × H × elem_size |
| `elem_size` | int | `hidden.element_size()`(bf16=2) | **D1-I10** | 字节换算,自包含不需外部 join |
| `K_eff` | int | `self.K` | 标准 | 实际 routing fan-out(K_eff 截断后) |
| `K_model` | int | `self.K_model` | 标准 | 模型 top-k(=8),核对是否被 K_eff 改 |
| `E_global` | int | `self.E_global` | 标准(provenance) | =128 |
| `world_size` | int | `self.world_size` | 标准(provenance) | =8 (EP) |
| `drop_policy` | str | `self.drop_policy` | 标准(provenance) | 采集时 = `"none"` |
| `drop_rate` | float | `self.drop_rate` | 标准(provenance) | 采集时 = `0.0` |
| `is_source_leader` | bool | `self.is_source_leader` | 标准 | owner_local_ep 恒 True;非 leader 行可过滤 |

**记录条件**:仅当 `MOE_COLLECT_PER_STEP=1`(默认关闭,关闭时 `record_step` 立即 return,零开销)。

**warmup 标注机制(K)**:`is_warmup = (prefill_onset_count <= K)`,`K = MOE_COLLECT_WARMUP_BATCHES`(默认 0 = 无 warmup)。一个 "prefill onset" = 一个 prefill step,其上一个**已知** phase 非 prefill(= 一个新 batch / sequence-group 开始)。harness 用 `--warmup-batches N` 跑 N 个丢弃 batch 再进测量循环,故设 `K = N`(v6: 2,multipolicy: 3,gsm: 见 run_plan)。**harness 无关、精确到 batch、零 harness 改动。** 各 workload 在启动命令里设 `MOE_COLLECT_WARMUP_BATCHES`。

**持久化**:line-buffered 流式 append;**每个 step 边界 fsync 一次**(crash 最多丢 ≤1 step,比 per-sample 更细;cost 可忽略,off GPU 关键路径)。reset/run 结束再 fsync 一次。

---

## 离线可派生量(不入 jsonl,分析时算)

- **worst-peer dispatch bytes** = `max_j(send_counts[j]) × H × elem_size`(j≠self 可选)。
- **worst-peer combine bytes** = `max_j(recv_counts[j]) × H × elem_size`(combine reverse a2a 把 expert_out 按 recv_counts 发回)。
- **worst-peer (dispatch+combine) bytes** = D1-I10 主自变量。
- **total dispatch bytes** = `L_send × H × elem_size`;**total combine bytes** = `L_recv × H × elem_size`。
- **expert FLOPs 代理**(D1-I10 反 claim 对照)= `L_recv × (GEMM flops per row)`,常数因子离线补。
- **dispatch-traffic entropy** = `H(send_counts / sum)` —— "routing entropy" 的零成本代理。
- **L*_ℓ 跨越判定**(D2-I5)= 对每 (layer, step) 比较 `L_recv` 与校准 `L*_ℓ`(或全局 `L*=3271`)。

---

## 字段对 D1-I10 / D2-I5 的覆盖自检(GATE A 第 7 问)

- **D1-I10 回归(bytes vs FLOPs)**:需要 per-观测的 worst-peer bytes、total bytes、L_recv、FLOPs 代理 → `send_counts/recv_counts/L_recv/H/elem_size/K_eff` **全覆盖**。注意:**时序(因变量 latency)不在本采集**,由 microbench 配对提供(方案 B);本采集保证 byte/L_recv 侧自变量齐全且能与 microbench 的 (L_recv, bytes) 网格对齐。
- **D2-I5 replay(L_recv-gated)**:需要 per-step per-layer L_recv + phase + step 轴 → `step_id/layer_id/phase/L_recv/num_tokens` **全覆盖**。

**已知缺口(诚实标注)**:
1. expert 级 per-step routing 熵未采(避免同步);如需,聚合版在现有 `routing_profile` 里有,per-step 需另加同步字段。
2. 时序字段全部缺失 = **有意为之**(方案 B)。D1-I10 的 latency 来自 microbench,不来自本采集。
3. ~~`step_id` 含 warmup~~ → **已解决**:新增 `is_warmup` 字段,采集时按 prefill-onset 精确打标,下游直接 `filter(is_warmup == False)`,无需猜。
