# Trace — GATE C 独立 integrity 审计 (Codex)

- Date: 2026-05-29
- Reviewer: Codex (mcp__codex__codex), model_reasoning_effort=high, sandbox=read-only
- Thread ID: 019e7272-27b4-78e0-bc49-5ced4d6b6214
- cwd: /home/lzy/Artifact-Infer-collection
- 依据 skill: experiment-audit (reviewer-independence:executor 收集路径,reviewer 判定)
- 审计范围: A/B 标 N/A(采集不打分);重点 C(行数公式)/ D(dead-code)/ E(scope)

## Codex 裁决摘要

**A/B:正确 N/A** —— 仅记录路由/结构 trace,无 label、无 accuracy 分数。

**C — Result existence & row-count:FAIL**
- GSM8K facts 声称 `records=297216, groups=3096, 3096×48=297216` —— **算术错误**:`3096×48=148608 ≠ 297216`。
- raw GSM8K rank0 = 37152 行 = 774×48,但 max step_id=386(387 distinct steps);**文件在第 18576 行重启,出现重复 step_id=0,layer_id=0** → **数据 2× 重复 append**。
- passage_retrieval(2560×48=122880 ✓)、multifieldqa(6144×48=294912 ✓)算术一致,**无重复**。

**D — Dead-code / coverage:WARN**
- 代码路径正确:`record_step()` 在 `DispatchEPHT.forward()` 每层每 step 调用(dispatch_ep_ht.py:447-471);Qwen3-MoE 全 48 层 MoE(qwen3_moe.py:233/241/261);step rollover 逻辑合理(per_step_trace.py:136);global phase 经 MAX all-reduce rank 一致(model_runner.py:608/617)。**不会漏层。**
- WARN 原因:**append 模式 + 复用 run_id** 会在 dataset 级重复 step_id(GSM8K 已演示)。非漏层 bug,但是 step/记账 hazard。
- Action:run_id 已存在文件时应让采集失败,或写入全新 run 目录。

**E — Scope sufficiency:PASS**
- D1-I10 自变量齐全(send_counts/recv_counts/L_recv/L_send/hidden_size/elem_size/K_eff);D2-I5 字段齐全(step_id/layer_id/phase/is_warmup/L_recv)。timing 缺失为设计(microbench 供)。

**Limitations 评估**:1 GSM8K warmup 未标→对 LongBench 主分析 benign;2 timing 缺→设计内 benign;3 size~304MB→本身 benign,但 GSM8K 重复对其统计 material;4 expert 熵缺→benign,可从 send_counts 离线导。

**GSM8K 负对照**:C2' 字节集中度 19.44%(vs long-prompt 96-99%)**作为负对照概念上合理**,不是 routing 采集 bug;真正的 bug 信号是 GSM8K 的重复 append,不是低集中度。

**Overall:FAIL** —— C 有硬性算术/一致性失败 + GSM8K raw 证据显示同 run_id 下重复 append。D/E 结构上可接受(D 降 WARN)。

## Action items(Codex)
1. GSM8K:去重或重跑到干净/截断输出路径(recorder append 模式:per_step_trace.py:107)。
2. 硬化:run_id 已有文件则采集失败,或每 run 写新目录。
