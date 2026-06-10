# Stage 3 全量采集 + 完整性审计 (GATE C)

- Date: 2026-05-29 | 方法论: run-experiment / monitor-experiment / experiment-audit
- 代码: worktree `collection/per-step-routing`(logging-only),输出 `raw_collection/`
- **Codex 独立审计 Overall = FAIL**(GSM8K 重复 append)。**按用户规则:停下,不进入下游 D1-I10/D2-I5 分析,等用户决定。**

## (1) 三 workload 执行汇报

| workload | run_id | exit | 墙钟 (实际 / 预估) | files | records | size | unexpected |
|---|---|---|---|---|---|---|---|
| passage_retrieval | full_passage_retrieval | 0 | 295s / ~5min ✓ | 8 | 122880 | 52.9MB | — clean |
| multifieldqa | full_multifieldqa | 0 | 301s / ~6-8min ✓ | 8 | 294912 | 125.1MB | — clean |
| GSM8K (neg ctrl) | full_gsm8k | 0 | 472s / ~5min(超) | 8 | 297216 | 126.1MB | **2× 重复 + size 超估** |

- 总墙钟 ~18 min(含三次模型载入),在 15-25min 预估内。
- **总大小 ~304MB vs 预估 ~150MB(~2×)** —— 触发"size 超 2×"标记。根因:GSM8K `--max-tokens 128` decode 量被低估(GATE B' 估"~10-20MB",实 126MB)。三个 run **均正常完成(bounded,exit 0),非 runaway**;GSM8K 在我察觉时已自行跑完。

### unexpected behaviors(完整披露)
1. **GSM8K 2× 重复 append(MAJOR,Codex FAIL)** —— 见 (2)/(3)。
2. **size 超估 2×** —— GSM8K decode 量低估;非 runaway,数据有效。
3. **GSM8K 5-placement 默认** —— 首次启动用默认 `--placements`(5 个),我**主动 kill 重启**为单 placement(contiguous),避免 5× 浪费 + 混合数据(kill 时无 trace 写入)。kill 残留 orphan(multiprocessing.spawn worker)需按 PID 清理,已确认 GPU 归零后重启。
4. **框架 verbose 日志** —— 已用 `grep -v` 流过滤(零代码改动)。

## (2) Codex 独立 integrity 审计(trace: traces/gate_c_integrity_audit.md)
- A/B:正确 **N/A**(不打分)。
- **C(行数公式):FAIL** —— GSM8K `records=297216` 实为 **2× 重复**(应 ~148608);文件第 18576 行重启 step_id=0。passage_retrieval / multifieldqa **算术一致、无重复**(2560×48=122880,6144×48=294912)。*(注:我在 _gate_c_facts.md 写的 "3096×48=297216" 是错的,Codex 抓出 → 实为数据 doubled。)*
- **D(dead-code):WARN** —— 代码路径正确,每层每 step 调用,**不漏层**(全 48 层 0..47,100% 覆盖);WARN 仅因 append+复用 run_id 的记账 hazard。
- **E(scope):PASS** —— D1-I10 / D2-I5 所需字段齐全;~295k measured LongBench records 足够。

## (3) GSM8K 重复 root cause + 影响

- **机制**:GSM8K 经 `run_moe_placement_experiments` → `test_bazaar_moe`,该路径让模型**跑了不止一遍**(routing-profile pass + eval pass,`--moe-profile-routing 1`),两遍都经 `dispatch_ep_ht` 且 append 到同一 `run_id` 文件 → 2× 重复。两遍 routing **不同**(仅 966/18576 (step,layer) 的 L_recv 相同)→ 是两次不同 model pass,**在同一文件里 step_id 撞车**,可按第 18576 行物理切分。
- **LongBench harness(v6 / multipolicy)单 generate 循环 → 无此问题(已确认 0 restart)。**
- **影响分级**:
  - **passage_retrieval / multifieldqa(D1-I10 / D2-I5 的主数据):CLEAN,可用。**
  - **GSM8K(负对照):2× 重复**。但负对照结论(decode L_recv≪L*、字节集中度 19.44% 低)在精确 doubling 下比例不变,**结论稳健**;只是 raw 文件需切分/重跑才能干净计数。

## (4) 数据可用性判断(与 dry-run 对比)

| workload | 全量 L_recv (measured) | C2 字面 step% | C2' 字节% | vs dry-run | phase 跨-rank 一致 |
|---|---|---|---|---|---|
| passage_retrieval | 0 / 8 / 49787 | 3.12% | **99.35%** | 一致(dry 99.32%) | 0/320 ✓ |
| multifieldqa | 0 / 8 / 36100 | 0.78% | **96.24%** | 一致(dry 96.64%) | 0/768 ✓ |
| GSM8K (neg ctrl) | 0 / 64 / 9296 | 1.55% | **19.44%** | 新(负对照) | 0/387 ✓ |

- **C2' 在全量上仍通过**(两 LongBench 96-99% ≫ 70%);与 dry-run 高度一致。
- **跨 rank phase 一致性**:三 workload 全 0 不一致(Finding 1 修复在全量上稳固)。
- **GSM8K 负对照符合预期**:字节集中度仅 19.44% → decode-heavy 工况 gate 杠杆小,印证 brief。
- **dead-code**:三 workload 100% (rank,step) 组含全 48 层。

## (5) 修复建议(等用户定)
- **GSM8K 去重**,二选一:
  - (a) **离线切分**:每 rank 取第 18576 行之前(pass 1)或之后(pass 2),保留一遍。需先确认哪遍是 eval pass(对负对照结论无差别)。零重跑。
  - (b) **重跑**:给两遍不同 run_id,或用下方硬化后重跑。
- **硬化 recorder(防此类 bug)**:run_id 文件已存在则**报错而非 append**,或每 run 写独立子目录。属代码改动,需你批准纳入 patch。
- LongBench 数据无需处理。

## 裁决(原始 GATE C)
**Codex Overall = FAIL(GSM8K 重复)。按用户硬性规则,停下,不进入下游分析,等用户决定。** LongBench 主数据 clean、C2' 通过、phase 一致,本身可用。

---

# GATE C 问题处理记录(方法论透明性)

用户裁决:(b) 重跑 GSM8K 关掉 profiling + 批准 recorder 硬化。处理全流程留痕如下。

## 1. recorder 硬化(防此类 bug 复发)
- `per_step_trace._handle`:**(run_id, rank) 文件已存在 → `raise FileExistsError`(明确提示加时间戳/换 run_id),而非静默 append**。`MOE_COLLECT_FORCE_APPEND=1` 可显式 override(默认 raise)。
- 新增 smoke test `test_no_append_collision`:同 (run_id, rank) 写两次 → 第二次 raise;force-append → 追加。**7/7 smoke 全过**。
- patch 更新:`04_collection_patch.diff` 现 6 文件 / 436 insertions(含硬化 + test)。

## 2. GSM8K 重复 root cause(精确)
- `run_moe_placement_experiments` 先跑 **profile job**(`run_id=..._profile`,生成 routing profile,PASS 1),再跑 **eval job**(PASS 2);两个独立进程都继承 `MOE_COLLECT_RUN_ID=full_gsm8k` → 写同一文件 → 旧 recorder append → 2× 重复(两遍 routing 不同,仅 966/18576 匹配)。
- 仅关 `--moe-profile-routing` **不够**(profile *pass* 仍跑)。故改为**绕过 placement pipeline,直接跑 `test_bazaar_moe`(单 eval pass)+ `--moe-profile-routing 0`**。

## 3. 重跑(run_id=gsm8k_clean,单 eval pass)
- 旧 2× 文件改名 `*.raw_2x_DEPRECATED`(8 个,审计留痕,未删)。
- `--prepare-only` 先正常化数据集 parquet,再 `test_bazaar_moe` 直跑(world 8,contiguous,profiling off)。
- 硬化 recorder 在重跑中写 fresh('w'),无 collision crash。

## 4. 重跑完整性检查(基本,未走完整 Codex audit —— 按用户"看一眼"即可)
| 指标 | 旧 2× (full_gsm8k) | 新 clean (gsm8k_clean) |
|---|---|---|
| total records | 297216 (2×) | **148608(恰好一半)** ✓ |
| step0/layer0 restarts (rank0) | 1(重复) | **0(单 pass)** ✓ |
| 行数公式 (rank,step)×48 | 3096×48≠297216 ❌ | **3096×48=148608 ✓** |
| layers/group | — | **全 48** ✓ |
| phase 跨-rank 不一致 | 0/387 | **0/387** ✓ |
| L_recv measured (min/med/max) | 0/64/9296 | **0/64/9296**(一致) |
| C2' 字节% | 19.44% | **19.44%**(一致,负对照) |
| size | 126.1MB | **63.2MB** |
| 墙钟 | 472s | 240s |

- **GSM8K 现 clean、单 pass、行数公式通过、phase 一致**;负对照结论(decode-heavy 字节集中度 19.44% ≪ 70%)在干净数据上确认。
- 此即 Codex check C 的 FAIL 已**解决**(新数据通过)。

## 5. 最终数据状态(三 workload 全 clean)
| workload | run_id | records | size | 状态 |
|---|---|---|---|---|
| passage_retrieval | full_passage_retrieval | 122880 | 52.9MB | CLEAN ✓ |
| multifieldqa | full_multifieldqa | 294912 | 125.1MB | CLEAN ✓ |
| GSM8K (neg ctrl) | **gsm8k_clean** | 148608 | 63.2MB | CLEAN ✓(单 pass) |
| ~~GSM8K 旧~~ | ~~full_gsm8k~~ | ~~297216~~ | — | DEPRECATED(`.raw_2x_DEPRECATED`) |

**三 workload 数据现全部 clean,行数公式/dead-code/phase 一致性/C2' 全通过。** D1-I10(LongBench bytes)+ D2-I5(per-step L_recv replay)+ GSM8K 负对照数据齐备。(下游分析等用户下一条 prompt,不主动开始。)
