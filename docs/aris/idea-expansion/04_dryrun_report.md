# Dry-run 报告 (GATE B)

- Date: 2026-05-29 | Stage 2 | 方法论: run-experiment / monitor-experiment
- 代码: worktree `collection/per-step-routing`(埋点),从 worktree 启动,trace 输出 `raw_collection/dryrun/`
- GPU 预检: 8×4090 全空闲(0 MiB)→ 通过。

## 执行情况
- **Block A (passage_retrieval_en_e)**: 4 samples, batch-size 2, warmup-batches 1, drop OFF, `MOE_COLLECT_PER_STEP=1`。**完成 (exit 0)**。
- **env 传播验证**: `MOE_COLLECT_PER_STEP` / `MOE_COLLECT_OUT_DIR` 经 torchrun 传到 8 个 worker → trace 文件按预期写入 dryrun 子目录。✓(GSM8K 的 subprocess 传播待 Block 完成后单独验证,因下方 Finding 暂缓。)
- **Block B (multifieldqa)**: **未跑** —— 故意暂缓,见下方决策点(用 buggy warmup 标注再跑一个 load 是浪费)。
- 墙钟: ~2.5 min(含 50s warmup,首次 framework cold-start + 模型载入)。注:50s warmup 主要是 Artifact 框架的 per-method 日志开销(日志 9475 行 "Propagating..."),全量前应关掉该 verbose。

## Block A — L_recv 分布(全 8 rank,global-phase aware)

| 集合 | n (records) | L_recv min | median | max | mean | >L*(3271) |
|---|---|---|---|---|---|---|
| 全部 measured | 49152 | 0 | 2 | 19469 | 284 | **3.1%** |
| global-PREFILL step | 1536 | 5013 | 7996 | 19469 | 9044 | **100%** |
| global-DECODE step | 47616 | 0 | 2 | 9 | 2 | **0%** |

- **L_recv 完美双峰、完全分离**:prefill 步 L_recv 5k–19k 全部 >L*;decode 步 L_recv 0–9 全部 <L*。gate 的"开/关"两态信号干净。
- **是否跨越 L\*=3271**:✅ 是,`max=19469 > L* > min=0`。
- 非 owner rank 的 L_recv 也正确:rank2(本地 num_tokens=0)在 global-prefill 步仍收到 routed token,L_recv 最高 8915 → **L_recv 信号对 D1-I10/D2-I5 在所有 rank 上都有效**。
- trace 大小: **26.6 MB / 61440 records(433 B/rec)**。

## 🔴 Finding 1 — `is_warmup` / `phase` 是"per-OWNER"而非"global",非 owner rank 标错

- **现象**: rank0/1 标注正常(batch_index 1–5,warmup=batch1);**rank2–7 全部 `batch_index=0`、`is_warmup=True`、`phase` 恒为 `decode`**。
- **根因**: `phase` 取自 `get_context().is_prefill`,这是**本 rank 是否在 prefill 自己拥有的序列**。dry-run batch-size=2 < 8 DP rank → 只有 rank0/1 拥有序列,rank2–7 `num_tokens=0`、本地 is_prefill 恒 False → 我的 prefill-onset(靠 `phase=="prefill"` 跳变)在 rank2–7 永不触发 → onset 计数停在 0 → 全标 warmup。
- **影响**:
  - L_recv **不受影响**(正确)。
  - `is_warmup` 在 idle rank 上不可靠;`phase` 在 idle rank 上把 global-prefill 步误标 decode。
  - 全量用 batch-size=8(每 rank 1 序列)时,绝大多数步每 rank 都有 token → phase 基本正确,但**末尾不满批 / 分配不均仍会重现**。
- **建议修法(需你批准)**:把 warmup 和 global-phase 改为 **harness/engine 显式 toggle**,rank 无关:
  - `per_step_trace.set_warmup(bool)`:harness 在 warmup 循环前后各调一次(所有 rank 同步调用)。
  - `per_step_trace.set_global_phase(is_prefill)`:从 `llm_engine.step()`(schedule 已知 is_prefill)或 harness 注入,使 `phase` 是**全局计算相位**而非 per-owner。
  - 代价:3 个 harness(+ 可能 llm_engine)各加 1–2 行 guarded logging,worktree 内,仍 logging-only。
  - 修后用 **batch-size 8 快速重跑 Block A 复核**,再跑 Block B + 全量。

## 🟡 Finding 2 — 跨越点比例 ≈ 3.12% < C2 预注册的 5%

- **C2 step-level**:rank-step 中 >L* 占 **3.12%**(global-step 同为 3.12%,= 4/128 prefill 步)。
- **原因**:passage_retrieval 官方 `max_new_tokens=32` → 每 batch 1 prefill 步 + 31 decode 步 → prefill 仅占 1/32 ≈ 3.1%。**这是 long-prompt-short-decode 形态的固有结果**(decode 步数压倒)。multifieldqa `max_new_tokens=128` 会**更低**(~0.8%)。
- **按 C2 字面判据(step 数比例 ≥5%):passage_retrieval 低于阈值。** 我**不会**事后改阈值(你已规定不得事后改)。
- **需要你判断的关键 nuance**:3.12% 是按 **step 计数**;按 **a2a 字节量 / 时间**,prefill 步携带 ~99.9% 的 received rows(prefill mean L_recv 9044 vs decode 2)。gate 的全部价值集中在这 ~3% 的步里。**5%-step-count 阈值对 long-prompt 工况可能量错了轴** —— 但是否重释/调整 C2,是你的决定,不是我的。

## 全量推算(基于 Block A)
- 单 workload(batch-size 8、单 baseline cell、64 samples、warmup 2):
  - records ≈ steps × 48 × 8。passage_retrieval ~8 batch ×(1 prefill+31 decode)=~256 步 → ~98k records ≈ **~43 MB**。
  - multifieldqa(128 new tokens)~4 batch ×129 步 ≈ 516 步 → ~200k records ≈ **~86 MB**。
  - GSM8K 小。**三 workload 合计 ≈ 130–180 MB**,**串行墙钟 ≈ 15–25 min**(三次模型载入 + verbose 日志关掉后更快)。

## 决策点(等你定,不自行推进)
1. **Finding 1 修法**:是否批准 harness/engine 显式 toggle(set_warmup + set_global_phase)?修完用 batch-size 8 重跑 Block A 复核。
2. **Finding 2 / C2**:3.12% < 5%(step 计数)。是否:(a) 维持 C2 字面判据并据此判定 passage_retrieval"step 比例不足"、(b) 你重释 C2 的"跨越点"轴(如按字节/时间)、还是 (c) 其它?**我不动预注册阈值,等你指令。**
3. **Block B (multifieldqa)** 是否在修完 Finding 1 后再跑(我建议:先修再跑,避免重现 bug + 浪费 load)?

---

# GATE B' — Finding 1 修复后复核 + Block B (2026-05-29)

## Finding 1 修复实现(global phase + global warmup)
- `model_runner.run()`:guarded MAX all-reduce 本地 is_prefill → rank-consistent **global phase**,经 `per_step_trace.set_global_phase()` 注入(opt-in,关闭时零开销)。
- 2 个 LongBench harness:warmup 循环前后 `set_warmup(True/False)`(SPMD → rank 一致,无需 broadcast);GSM8K 走 onset+env fallback(现也 rank 一致,因 phase 已 global)。
- `phase`/`is_warmup` 不再取自 per-owner local context。patch 现含 6 文件(388 insertions),见 04_collection_patch.diff。

## 复核标准:rank 0-7 同一 step 的 phase 一致性
- **Block A(post-fix,batch-size 2,故意制造 idle rank 2-7)**:**PHASE-inconsistent steps across ranks = 0 / 96 steps**。idle rank(num_tokens=0)在 global-prefill step 0 全部正确标 `prefill`。✅
- **Block B(multifieldqa)**:**PHASE-inconsistent = 0 / 256 steps**。✅
- 样例(Block A step0,8 rank):`phase=[prefill×8]`,`num_tokens=[5003,4530,0,0,0,0,0,0]` —— idle rank 标注正确。

## 两个 Block 的 L_recv 分布 + 跨越点字节量(精确计算,非估算)

| Block | measured records | L_recv(prefill) | L_recv(decode) | **C2 字面**(step≥5%) | **C2'**(字节≥70%) | trace |
|---|---|---|---|---|---|---|
| A passage_retrieval | 24576 | 5013–19469 (med 7996) | 0–9 (med 2) | **3.12%** ❌ | **99.32%** ✅ | 15.9MB |
| B multifieldqa | 98304 | 3974–16391 (med 6780) | 0–9 (med 2) | **0.78%** ❌ | **96.64%** ✅ | 62.9MB |

- a2a 字节量 = `(L_send+L_recv)×hidden_size×elem_size`,跨越点 = `L_recv>L*` 的记录;分子分母全量精确求和。
- Block A total a2a=57.29GB,跨越点 56.90GB(99.32%);Block B total=47.59GB,跨越点 45.99GB(96.64%)。

## C2 / C2' 双判据下两 workload 状态
| | C2(字面 step≥5%) | C2'(字节≥70%) |
|---|---|---|
| passage_retrieval | 3.12% → **字面失败** | 99.32% → **通过** |
| multifieldqa | 0.78% → **字面失败** | 96.64% → **通过** |

- **ablation 信号(保留,不筛 prompt)**:passage_retrieval(longer prompt)step 跨越比例 3.12% > multifieldqa 0.78%,印证"gate 价值随 prompt 长度变化";按字节量两者都被 prefill 主导(99.3% vs 96.6%),细微差异同向。
- multifieldqa **故意保留**较低跨越比例作对比组(用户指示:不为达标筛长 prompt)。

## 全量预估(更新版,基于 post-fix dry-run 432 B/rec)
- passage_retrieval(64 样本,batch-size 8,warmup 2,单 baseline cell):~98k measured records ≈ **~50 MB**。
- multifieldqa(32 样本,batch-size 8,128 new tokens):~200k records ≈ **~85 MB**。
- GSM8K(16–32 样本,512 ctx):小,**~10–20 MB**。
- **合计 ≈ 145–155 MB,串行墙钟 ≈ 15–22 min**(三次模型载入主导;关掉框架 verbose "Propagating" 日志可更快)。

## GSM8K env 传播(Q7 ⚠ 复核)
- Block A/B 已证 `MOE_COLLECT_*` 经 torchrun 传到 8 worker。GSM8K launcher 用 `subprocess.run(cmd)` 无 `env=` → 继承父 env。**仍按计划在 GSM8K 全量首个 step 后实测确认 trace 文件落盘**(若没落盘则停下问)。

## 结论
- Finding 1 **已修复并验证**(0 跨-rank 不一致)。
- L_recv 数据干净、双峰、跨越 L*;byte-volume 判据(C2')两 workload 均通过。
- 数据**可用于 D1-I10 / D2-I5**。建议进入 Stage 3 全量。
