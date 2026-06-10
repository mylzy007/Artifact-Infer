# Codebase 实测可行性核验 — D2-I5 / D1-I6 / D5-I1

- Date: 2026-05-29
- 任务: 对 SELECTION.md 推荐的 3 个 pilot idea,打开实际代码(只读)核验 agent 的可行性推测。
- 约束: 禁实验、禁改代码、禁占 GPU;所有判断基于实际 grep / read。
- 核验对象代码(均已 read 全文):
  - `workshop/nanovllm_moe/artifacts/modeling/layers/moe/dispatch_ep_ht.py`
  - `workshop/nanovllm_moe/artifacts/modeling/layers/moe/combine_ep_ht.py`
  - `workshop/nanovllm_moe/artifacts/modeling/layers/moe/experts_ep_ht.py`
  - `workshop/nanovllm_moe/services/utils/expert_drop.py`
  - `workshop/nanovllm_moe/services/utils/routing_profile.py`
  - `workshop/nanovllm_moe/services/utils/overlap_runtime_stats.py`
  - `workshop/nanovllm_moe/services/utils/context.py`
  - `eval/drop/run_owner_local_ep_phase4_drop_tier1_microbench.py`
  - 数据: `eval_results/prefill_drop_l_sweep_tier1/{tier1_rows.jsonl, tier1_summary.json}`,
    `eval_results/.../profiles/moe_routing_profile_*.json`

> 先更正一个引用名:SELECTION/feasibility 多处写 `..._longbench_v6`,实际目录是
> `eval_results/owner_local_ep_phase4_drop_passage_retrieval_v6`(及 `_v7`)。`longbench_v6` 不存在。

---

## 全局结论(先讲,后给证据)

1. **三个 idea 的代码落点全部真实存在**,SELECTION 的文件级地图基本准确。
2. **但三个 idea 的"复用度"和"改动量"都被乐观估计了**,根因是同一个被忽略的事实:
   **现有 profiling 基础设施只存"聚合直方图",不存"per-step 时序",且把 prefill+decode 合并**
   (`routing_profile.py:155 prefill_decode_split="combined"`;`overlap_runtime_stats` 的 send
   matrix 跨 forward 累加)。三个 idea 的 pilot 都需要"按 step / 按 phase"的量,而这个量当前
   **没有任何 artifact 持久化**。
3. **D5-I1 和 D2-I5 都不是"纯离线"** — 它们需要先做一次带新埋点的 GPU 采集run。真正"纯离线、零 GPU"
   的是 SELECTION 排在它们前面的 **D1-I10**(可直接吃 `tier1_rows.jsonl`)。
4. **D2-I5 有一个 SELECTION 完全没提到的运行时顺序障碍**:drop 决策发生在
   `dispatch_ep_ht.py:299`,而 `total_recv`(= L_recv)要到 `:383` 才算出来。**运行时无法在 drop
   之前读到本 rank 的 L_recv**。离线 replay 不受此影响,但"runtime gate on L_recv"需要额外机制。
5. **D1-I6 的运行时实现被显著低估**:combine-only drop 改变 reverse-a2a 的 per-rank counts,需要
   在 combine 路径**新增一次 counts 交换 + host-sync**,并把"哪些行被丢"回传给 source 重建
   `sort_perm` / 零填充。这是 SELECTION "改动中等" 没算进去的分布式记账。

最终建议:**先做 D1-I10(纯离线,零风险)打底,再做 D2-I5 作为主机制 pilot;D5-I1 作为 D1-I10
的因果加固(需 GPU);D1-I6 第三,且其 pilot 只该锁定"质量/Pareto"那半边。** 详见末尾"重新裁决"。

---

## Idea 1 — D2-I5: L_recv-gated continuous phase detection

### 1.1 涉及文件清单(已验证)

| 文件 | 作用 | 状态 |
|---|---|---|
| `dispatch_ep_ht.py` | drop 决策 + `total_recv` 计算所在 | 关键路径,已读全文 |
| `services/utils/expert_drop.py` | drop policy 选择器 | 成熟,可复用选择器 |
| `services/utils/context.py` | `Context.is_prefill` / `num_tokens` | per-step 上下文,新发现 |
| `services/utils/routing_profile.py` | 路由直方图(聚合) | 只有聚合,缺 per-step |
| `services/utils/overlap_runtime_stats.py` | send matrix → per-layer L_recv(聚合) | 只有聚合 |

### 1.2 agent 推测逐条核对

- **"`total_recv`/`L_recv` 已经在 `dispatch_ep_ht.py` 路径里存在"** → **经验证正确(但有重大附加条件)**。
  `total_recv = int(sum(recv_counts))` 在 `dispatch_ep_ht.py:383`。**但它在 drop 之后才算出**:
  drop 在 `:294-339`(step 2.5),counts 交换在 `:358`,`total_recv` 在 `:383`。
  → **运行时:本 rank 在做 drop 决策时拿不到本 rank 的 `total_recv`。** 这是 SELECTION 没提到的顺序约束。
- **"复用现有 drop infra"** → **部分正确**。drop 选择器(`expert_drop.apply_drop` / `_select_drop_by_score`)
  确实可直接复用;`drop_policy/drop_rate/drop_seed` 已是 `DispatchEPHT.__init__` 入参(`:107-109`)。
  缺的是"每层一个 gate 阈值 + per-step 决定开关"这一层逻辑,需新写。
- **"先离线 replay,不需要先改 runtime"** → **经验证不准确**。离线 replay 需要 **per-step、按层的
  L_recv 时序**。核查所有 artifact:
  - `routing_profile.py` 存的是 `traffic[layer][src][expert]` 聚合直方图,且 `prefill_decode_split="combined"`(`:155`),跨 step 累加(`:57-59`)。
  - `overlap_runtime_stats` 的 `per_dst_incoming`(= L_recv)同样按层跨 forward **累加**(`:94-100`),
    且 `record_dispatch` 仅在 `overlap_plan.enabled` 时调用(`dispatch_ep_ht.py:367`)。
  - `tier1_rows.jsonl` 是**合成均匀 logits**(`run_...tier1_microbench.py:193-196` 用 `torch.randn`),
    不是真实 LongBench/GSM8K 逐 step 路由。
  → **结论:真实 per-step L_recv 时序当前不存在于任何文件**。要做忠实的离线 replay,必须先加 per-step
  L_recv 埋点(改 runtime)+ 跑一次 GPU 采集。
- **"按 L_recv 是否越过 L* 决定 drop,而不是 prefill/decode 标签"** → 机制成立,但发现一个**更便宜的可实现变体**:
  `Context` 里已有 `is_prefill` 和 `num_tokens`(`context.py`),它们在 drop **之前**就已知。
  `L_send_nominal = num_tokens·K` 是 send 侧的可得代理,据此 gate 是当前就能写、绕开 `:383` 顺序障碍的。
  真正按 recv 侧 `L_recv` gate 才撞顺序障碍。

### 1.3 真实改动量

- **离线 replay pilot(忠实版)**:
  - 新增 per-step L_recv 埋点:`dispatch_ep_ht.py` 在 `:383` 后写 `(layer_id, step_id, total_recv)` →
    新增一个 per-step recorder(类比 `overlap_runtime_stats` 但不累加),~40–60 LOC。
  - 采集脚本:复用 `eval/drop/run_*` 模板跑 LongBench + GSM8K 各一次(**需 8×GPU**)。
  - 离线 sim:新脚本,读 per-step L_recv + 校准 `L*_ℓ`,模拟 binary gate vs static `tail_weight@r=0.3` vs oracle,~150–250 LOC。
  - **真实 LOC ≈ 250–350;真实工期 ≈ 2–3 天(含 1 次 GPU 采集),不是 1–2 天纯离线。**
- **运行时版(若 pilot 通过)**:额外要解 `:383` 顺序障碍 —— 选项:(a) drop 前加一次 counts-only a2a 预探 L_recv(多一次 collective + host-sync);(b) 用 D4-I2 estimator 预测;(c) 用上一 step 的 L_recv。每条都是独立的非平凡工作。

### 1.4 复用度评级:**改一点 + 新写**(混合)

- 直接复用:drop 选择器、`drop_*` 入参、`Context.is_prefill/num_tokens`。
- 需新写:per-step L_recv 埋点、`L*_ℓ` 校准表、gate 逻辑、离线 sim 脚本。

### 1.5 隐藏风险

1. **顺序障碍(高)**:运行时 gate on 本 rank L_recv 不可直接实现(见 1.2)。pilot 用离线/或 send 侧代理可绕开,但 paper 的"runtime gate"主张要诚实交代这点。
2. **send/recv 侧不对称(中)**:drop 是 send 侧动作(砍本 rank 发出的 replica),`L_recv` 是 recv 侧量(本 rank 收到的行)。"本 rank 按自己 L_recv 决定砍自己 send" 是对全局量的局部启发式 —— 每个 rank 砍 send 会改变**别的 rank** 的 recv。离线可模拟,但机制叙事需澄清。
3. **prefill+decode 合并(中)**:现有 profile 合并两相,拿不到 decode 段的 L_recv 分布,而 decode(L_recv≈8)正是 gate 该关掉的区。必须新埋点才能分相。
4. **失败可区分性**:若 pilot 失败,**能区分**机制失败 vs 实现踩坑 —— 因为离线 sim 是确定性的,gate 阈值/L* 是显式输入,失败会直接表现为"binary gate 不优于 static",属机制信号,不是 kernel bug。这是 D2-I5 相对干净的地方。

---

## Idea 2 — D1-I6: Combine-asymmetric drop(post-expert ‖g·o‖)

### 2.1 涉及文件清单(已验证)

| 文件 | 作用 | 关键行 |
|---|---|---|
| `experts_ep_ht.py` | 产出 `expert_out`(= o,raw,未加权) | `:116-147` |
| `combine_ep_ht.py` | reverse-a2a + 加权归约 | `:74-99` |
| `dispatch_ep_ht.py` | `recv_topk_w`(= g per recv-row)在此被算出**但丢弃** | `:394, :408-412, :425` |
| `expert_drop.py` | percentile 选择器可复用 | `_select_drop_by_score :322` |

### 2.2 agent 推测逐条核对

- **"需要 combine-side hook 和 ‖g·o‖ 统计"** → **经验证正确,且信号比想象的更近**:
  - `o`(expert output):在 `combine_ep_ht.py` 入口 `expert_out`(`:52`,reverse-a2a 之前)即是,可直接取 ‖o‖₂。
  - `g`(per-recv-row router weight):**已经被 dispatch 发到 expert rank**(`recv_topk_w`,`:394` 算、`:408-412` a2a 收),
    **但没有放进 `TokMetaEPHT`**(对比 `:447-462` 返回项,只回传了恒为 1.0 的 `recv_topk_weights`,见 `:425`),
    所以现在被丢弃。要算 ‖g·o‖ 只需在 tok_meta 里**保留 `recv_topk_w`**(小改)。
- **"复用现有 drop infra"** → **部分正确,且方向不同**。现有 **所有** drop policy 都在 **dispatch 侧、expert 之前**,
  信号是 `flat_topk_w`(router weight)(`expert_drop.py:166-168` tail_weight 用 `w_list`)。
  D1-I6 的信号 ‖g·o‖ **要等 expert 跑完才存在**,落点在 **combine 侧**。**选择器逻辑可复用,信号与落点全新。**
- **"改动中等;最小 pilot 先做离线 replay / cosine proxy"** → **离线 pilot 估计合理;但 runtime 改动被低估**(见 2.3)。

### 2.3 真实改动量

- **离线 pilot(只验质量/Pareto)**:
  - 需录 `expert_out` 激活(per layer per step)—— **无任何现有 hook 录激活**(profiler 只录路由计数)。
    新增激活录制 hook + **磁盘问题**:`expert_out` 是 `[total_recv, H]` bf16,逐层逐 step 录全量是 GB 级,
    必须子采样/抽层。~60–100 LOC + 需 GPU 采集。
  - 离线:对录到的 o 算 ‖g·o‖₂,按 percentile 做 combine-only drop,比较等 combine-byte 下与 symmetric `tail_weight` 的 layer-output cosine / 任务 proxy。~150–200 LOC。
  - **真实 LOC ≈ 250–300;工期 ≈ 1.5–2.5 天(含 GPU 采集 + 子采样设计)。**
- **运行时实现(被严重低估)**:
  - 在 combine 的 reverse-a2a(`combine_ep_ht.py:74-82`)**之前**丢行 → reverse 的 per-destination counts 改变。
    现在 combine 直接复用 `tok_meta.send_counts/recv_counts`(`:78-80`)对称回传;丢行后必须**重算回传 counts +
    新增一次 counts a2a + host-sync**(等于把 combine 变得像 dispatch 一样有 host-sync)。
  - source 侧 un-permute `unperm[sort_perm] = rev`(`:96`)依赖 `sort_perm` 覆盖固定行;combine 侧丢行后,
    source 必须知道**哪些 (t,k) 被丢**才能正确零填充 → 需把 drop mask/索引随 reverse-a2a 回传。
  - → 真实是 **大(>3d)**,不是"中"。

### 2.4 复用度评级:**改一点(信号)+ 大改(runtime 回传记账)**

- 直接复用:`_select_drop_by_score` percentile 选择;`recv_topk_w` 已在 wire 上(只需保留)。
- 大改:combine 侧 counts 重算 + 额外 a2a + dropped-index 回传。

### 2.5 隐藏风险

1. **字节上限单边且更低(高,机制层面)**:combine-only drop **不省 dispatch、不省 expert GEMM**(都已发生),只省 combine(~50% 那半)。在**相同丢弃行数**下,它省的字节**严格少于** symmetric dispatch+combine drop。它唯一的赌注是:‖g·o‖ 是更好的信号 → 同质量下能丢更多 / 丢得更准,从而在"质量 vs combine-byte"Pareto 上反超 symmetric。这是真问题但天花板被钉死在 combine 半边。
2. **录激活的磁盘/采样陷阱(中)**:全量 expert_out 落盘不现实,子采样方案设计本身是 pilot 的一部分。
3. **失败可区分性(中)**:离线 pilot 失败 = 质量信号不够好(机制),可区分;但 runtime 若实现,combine 额外 host-sync 可能吃掉 combine 省下的时间,**latency 失败会混入实现开销**,不易与机制失败区分 → 故 pilot 应**只锁离线质量问题**,不碰 runtime latency。

---

## Idea 3 — D5-I1: pad-only causal bytes microbench

### 3.1 涉及文件清单(已验证)

| 文件 | 作用 | 状态 |
|---|---|---|
| `eval/drop/run_owner_local_ep_phase4_drop_tier1_microbench.py` | 段计时 + L* + JSONL 的完整 harness | 极佳脚手架 |
| `eval/drop/shared.py` | `CUDAEventTimer/agg_stats/write_jsonl` | 直接复用 |
| `eval_results/prefill_drop_l_sweep_tier1/tier1_rows.jsonl` | 现有段计时数据(300 行) | 关联性证据已在 |
| `dispatch_ep_ht.py` | 被 harness 实例化的真实 dispatcher | 复用 |

### 3.2 agent 推测逐条核对

- **"复用已有 dispatch/bench scaffold"** → **经验证正确(plumbing 层)**。tier1 harness 已经构造真实
  Dispatch/Experts/Combine、用 CUDA event 分段计时、rank-max 聚合、出 JSONL + 算 L*。计时/聚合/输出可整段复用。
- **"固定 L_recv/routing/expert compute,只通过 padding 改 wire bytes"** → **机制是新的,现有数据不满足**。
  tier1 sweep 变的是 `T_local`(token 数),`L_recv`、GEMM 行数、a2a 字节**三者同时变**
  (`tier1_summary.json` 里 `L_recv` 随 `T_local` 单调涨)。**这正是 D5-I1 要打破的关联性混淆。**
  现有数据**没有**"固定行数、只变每行字节"的点 → 必须新写 padding 隔离。
- **"最小改动,半天级别"** → **编码量估计基本对,但漏了"必须跑 GPU"**。padding-isolation 最干净的写法是
  一个**只测 a2a collective 本身**的 standalone bench(固定 split sizes,变每行字节宽度),复用 harness 的计时/输出。
  ~120–200 LOC。**但它本质要测延迟,必须 8×4090 torchrun**,不是纯离线。

### 3.3 真实改动量

- 新增 padding-isolation bench:~120–200 LOC(复用 `shared.py` + tier1 setup)。
- **必须一次 8-GPU 采集run**(测 per-byte latency slope)。
- **真实工期 ≈ 0.5–1 天编码 + GPU 排队**。编码是"小",但**不是零 GPU、不是纯离线**。

### 3.4 复用度评级:**改一点(plumbing 直接用,核心 padding 逻辑新写)**

### 3.5 隐藏风险

1. **需要 GPU(中)**:与 D1-I10/D5-I3(纯离线吃现有 jsonl)不同,D5-I1 必须上机。作为"支撑 pilot",
   它比 D1-I10 贵且需排队。
2. **padding 实现位置(中)**:在 DispatchEPHT 内 padding hidden 宽度会动到 dispatch 契约;干净做法是
   绕过 DispatchEPHT、直接 bench `dist.all_to_all_single`,但那样测的是"裸 collective"而非"生产 dispatch",
   需在 paper 里说明这层抽象差。
3. **失败可区分性(高,好)**:slope 拟合是纯计量,失败 = 斜率不稳/字节非主因(机制结论),不会与实现 bug 混淆。

---

## 重新裁决

### 与 SELECTION 的一致性

| idea | SELECTION 排序/定性 | codebase 实测后 | 变化 |
|---|---|---|---|
| D2-I5 | 第1,"复用 infra,先离线,改动中" | 仍是最佳**机制** pilot,但改动 **2–3d(含 GPU 采集)**,且有 `:383` 运行时顺序障碍 | **降速不降级**:仍首选机制,但 cost 上修、风险点新增 |
| D1-I6 | 第2,"改动中,可离线" | 离线质量 pilot 可行(1.5–2.5d);**runtime 实现 >3d**,字节天花板单边 | **运行时改动升级为"大";pilot 范围收窄到质量** |
| D5-I1 | 第3,"半天,纯支撑" | 编码小但**需 GPU、非纯离线**;现有数据不满足因果隔离 | **澄清:非零-GPU**;作为支撑被 D1-I10 比下去 |

### 有没有 idea 被实测劝退 / 升降级

- **没有被劝退的**:三者机制都成立,代码落点都真实。
- **D1-I6 运行时部分升级为"大改动"**:combine 侧额外 counts 交换 + host-sync + dropped-index 回传,是 SELECTION 漏算的分布式记账。建议**其 pilot 明确只做离线质量/Pareto**,把 latency 主张推迟到机制被验证之后。
- **D5-I1 相对降一格(在支撑层内)**:因为它需要 GPU,而同属"支撑证据"的 **D1-I10 是纯离线、可直接吃
  `tier1_rows.jsonl`**,zero-GPU、零风险。先做 D1-I10 拿到 cost model,再用 D5-I1 做因果加固更合理。
- **D2-I5 不降级,但务必把"runtime gate on L_recv"的顺序障碍写进 pilot 设计**;或先采用 send 侧
  `num_tokens·K` 代理(`Context.num_tokens` 已可得)作为可立即实现的弱化版。

### 建议立刻进入 pilot 设计的是哪一个

**D2-I5**,但**建议先用半天插一个 D1-I10(纯离线 cost model)打底**,再正式做 D2-I5。理由:

1. D2-I5 仍是唯一把 `L*`/`L_recv` 从 finding 变成 **runtime decision** 的主机制,撞车/自一致性最干净(SELECTION 四条标准),失败信号可与实现 bug 区分(1.5.4)。
2. 但它的真实前置依赖(per-step L_recv 埋点 + 1 次 GPU 采集)与 D1-I10 的采集需求高度重叠 —— **同一次 routing/L_recv 采集既喂 D1-I10 的回归,又喂 D2-I5 的 replay**,合并采集省一次 GPU run。
3. D1-I10 纯离线、零 GPU、当天可出 R²,先确立"bytes/L_recv 是主因"的 cost model,能在 D2-I5 真正上机前就给 go/no-go 信号,降低 D2-I5 的采集浪费风险。

> 即:**D1-I10(纯离线打底)→ D2-I5(主机制,共享采集)→ D5-I1(因果加固)→ D1-I6(离线质量,第三)**。
> 这与 SELECTION 的精神一致(D2-I5 仍是机制主线),只是把"纯离线、零成本"的 D1-I10 提到 D2-I5 之前作为
> 共享采集的前置,并据实修正了三者的 cost 与风险标注。

**未自动进入 pilot 设计阶段,停在此处等指示。**

---

## 实现状态更新 (2026-05-29, Stage 1-2 数据采集)

D2-I5 / D1-I10 所需的 per-step 结构埋点已实现(worktree `collection/per-step-routing`,logging-only,opt-in `MOE_COLLECT_PER_STEP=1`):
- `services/utils/per_step_trace.py`(新)— 结构 trace recorder(L_send/L_recv/send_counts/recv_counts/phase/is_warmup/...)。
- `dispatch_ep_ht.py` `:447` 前 — record_step 调用点(零新增 GPU 同步;send/recv counts 已 host-sync)。
- `model_runner.run()` — guarded MAX all-reduce 求 **rank-consistent global phase**(修复本报告指出的 per-owner phase 问题:idle rank 把 global-prefill 误标 decode)。
- 2 个 LongBench harness — warmup `set_warmup()` 标注。
- patch: `04_collection_patch.diff`(6 文件,388 insertions);schema: `04_collection_schema.md`;dry-run: `04_dryrun_report.md`。
- 关于本报告 §"total_recv 在 drop 之后" 的顺序约束:埋点只读已算出的 counts,故不受影响(Codex review 已确认,见 traces/gate_a_patch_review.md)。
