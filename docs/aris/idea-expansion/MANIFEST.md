# ARIS idea-expansion 输出清单 (MANIFEST)

## Stage 1 (data collection) — 2026-05-29
- [04_collection_schema.md](04_collection_schema.md) — 结构-only 采集 schema(无时序,方案 B)
- [04_collection_patch.diff](04_collection_patch.diff) — 埋点 patch(worktree collection/per-step-routing,294 insertions)
- [04_collection_run_plan.md](04_collection_run_plan.md) — 运行计划 + Pre-registered Claims C1/C2
- [../traces/gate_a_patch_review.md](../traces/gate_a_patch_review.md) — Codex 跨模型 patch 审查(全 OK)
- worktree code: workshop/nanovllm_moe/services/utils/per_step_trace.py (new, 150L) + _test_per_step_trace.py (new, 115L) + dispatch_ep_ht.py (+29L logging-only)

## Stage 2 (dry-run, GATE B) — 2026-05-29
- [04_dryrun_report.md](04_dryrun_report.md) — Block A L_recv 分布 + 2 findings(warmup/phase per-owner bug;crossing 3.12%<5%)
- raw_collection/dryrun/per_step_trace_dryrun_passage_retrieval_rank{0..7}.jsonl — Block A 原始 trace (26.6MB)

## Stage 2 (GATE B') — 2026-05-29
- 04_dryrun_report.md 追加 GATE B' 节(Finding 1 修复验证 + Block B + byte-volume C2')
- 04_collection_run_plan.md 追加 C2' + Mid-experiment Re-registration
- 04_collection_patch.diff 更新(6 文件:per_step_trace, dispatch_ep_ht, model_runner, 2 harness, smoke test)
- raw_collection/dryrun/per_step_trace_dryrunA2_*.jsonl (post-fix Block A), per_step_trace_dryrunB_*.jsonl (Block B)

## Stage 3 (GATE C) — 2026-05-29
- [04_collection_audit.md](04_collection_audit.md) — 全量执行 + 完整性审计(Codex FAIL: GSM8K 2× 重复)
- [../traces/gate_c_integrity_audit.md](../traces/gate_c_integrity_audit.md) — Codex 独立 integrity 审计 trace
- raw_collection/per_step_trace_full_passage_retrieval_rank{0..7}.jsonl (52.9MB, CLEAN)
- raw_collection/per_step_trace_full_multifieldqa_rank{0..7}.jsonl (125.1MB, CLEAN)
- raw_collection/per_step_trace_full_gsm8k_rank{0..7}.jsonl (126.1MB, 2× DUP — 待去重)
- raw_collection/_gate_c_facts.md — 审计输入(注:GSM8K 行数算术含 doubling,Codex 已纠)

## GATE C 问题处理 (GSM8K 重跑 + recorder 硬化) — 2026-05-29
- recorder 硬化: per_step_trace._handle 同 (run_id,rank) 已存在则 raise FileExistsError (MOE_COLLECT_FORCE_APPEND=1 override); 新增 test_no_append_collision (7/7 smoke pass)
- 04_collection_patch.diff 重新生成 (6 文件, 436 insertions, 含硬化+test)
- 04_collection_audit.md 追加 "GATE C 问题处理记录" (2x→重跑→硬化 全流程)
- raw_collection/per_step_trace_gsm8k_clean_rank{0..7}.jsonl (63.2MB, CLEAN 单pass) <- GSM8K 负对照最终数据
- raw_collection/per_step_trace_full_gsm8k_rank{0..7}.jsonl.raw_2x_DEPRECATED (旧 2x, 留痕未删)

## Stage D0 (topk 补采集 + Sidecar binary) — 2026-05-29
- [topk_sidecar_format_v1.md](topk_sidecar_format_v1.md) — Sidecar v1 binary format 契约
- [05_d0_audit.md](05_d0_audit.md) — Stage D0 完整审计 (size / 验证 / 跨版本无扰动)
- [04_collection_patch.diff](04_collection_patch.diff) — 6 files / 639 insertions (含 Sidecar + 9 smoke tests)
- raw_collection_v2/per_step_trace_passage_retrieval_v2_rank{0..7}.{jsonl,topk.bin} — 497 MB, 122,880 records (LBG)
- raw_collection_v2/per_step_trace_multifieldqa_v2_rank{0..7}.{jsonl,topk.bin} — 315 MB, 294,912 records (LBG)
- raw_collection_v2/per_step_trace_gsm8k_clean_v2_rank{0..7}.{jsonl,topk.bin} — 112 MB, 148,608 records (contiguous, send_counts 重建 100% 一致)
- raw_collection_v2/_d0_validate.py — D0 验证 harness (round-trip + consistency + sanity + contiguous reconstruction)

## Stage D1 (PLAN 设计 + Codex Devil's Advocate) — 2026-05-29
- [05_pilots/d1_i10/PLAN.md](05_pilots/d1_i10/PLAN.md) — D1-I10 PLAN (212L, 含 Devil's Advocate Review)
- [05_pilots/d2_i5/PLAN.md](05_pilots/d2_i5/PLAN.md) — D2-I5 PLAN (210L, 含 13 comparison groups + Devil's Advocate Review)
- [../traces/d1_d2_ablation_design_codex.md](../traces/d1_d2_ablation_design_codex.md) — Codex round 1 独立 ablation 设计 (14 groups)
- [../traces/d1_i10_plan_attack.md](../traces/d1_i10_plan_attack.md) — Codex round 2a PLAN attack (48 attacks)
- [../traces/d2_i5_plan_attack.md](../traces/d2_i5_plan_attack.md) — Codex round 2b PLAN attack (30 attacks)
- worktree pilot/d1-i10-d2-i5 created (no code yet — Stage D1 is design-only)

## Stage D2 (D1-I10 execution) — 2026-05-30
- [05_pilots/d1_i10/RESULTS.md](05_pilots/d1_i10/RESULTS.md) — D1-I10 full results (C1.main FAIL, anti1/2 ANTI_HOLD, scope-limit/downgrade per pre-registered fallback)
- [../traces/d1_i10_results_review_round1.md](../traces/d1_i10_results_review_round1.md) — Codex round 1 results attack (verdict accepted; reframe to "L_recv preferred")
- 05_pilots/d1_i10/run01/microbench_rows_pid{0,1,2}.jsonl — 5,514 raw rows from 3 fresh-process repeats (919 cells × 2 drop rates)
- 05_pilots/d1_i10/run01/samples.jsonl + split_diagnostics.json
- 05_pilots/d1_i10/run01/regression_results.json + lstar_per_layer.json
- worktree pilot/d1-i10-d2-i5: new eval/d1_i10_{sample,microbench,regression}.py (no algorithm-code changes)

## Stage D3 (D2-I5 execution) — 2026-05-30
- [05_pilots/d2_i5/RESULTS.md](05_pilots/d2_i5/RESULTS.md) — D2-I5 full results (C2.main FAIL by structural prefill dominance; selector-vs-random benefit measured with tautology caveat)
- 05_pilots/d2_i5/run01/replay_results.json + run.log
- [../traces/d2_i5_results_review_round1.md](../traces/d2_i5_results_review_round1.md) — Codex round 1 attack (FAIL verdict accepted; reframe to "selector matters; gate doesn't")
- [../traces/d2_i5_results_review_round2.md](../traces/d2_i5_results_review_round2.md) — Codex round 2 (12 overclaim fixes applied)
- [../traces/d2_i5_results_review_round3.md](../traces/d2_i5_results_review_round3.md) — Codex round 3 (5 wording + numeric fixes applied)
- [../traces/final_framing_review.md](../traces/final_framing_review.md) — Codex final venue/paper-framing assessment (workshop tier; "L_recv predicts latency, not worst-peer" title)
- worktree pilot/d1-i10-d2-i5: new eval/d2_i5_replay.py (replay-only, no algorithm-code edits)

## Stage D3 Stretch (production strict_accuracy validation) — 2026-05-30
- [05_pilots/d2_i5/STRETCH_RESULTS.md](05_pilots/d2_i5/STRETCH_RESULTS.md) — B-PG (prod v6 SOTA = phase-gated equivalent of EXP) on LongBench passage_retrieval_en, 64 samples
- 05_pilots/d2_i5/stretch/passage_retrieval_summary.json + passage_retrieval_rows.jsonl + stretch.log
- 05_pilots/d2_i5/RESULTS.md updated with "Stretch: Production Metric Sanity Check" section
- [../traces/final_framing_review_post_stretch.md](../traces/final_framing_review_post_stretch.md) — Codex revised venue/title/abstract after stretch ("borderline MLSys main")
- Key result: strict_accuracy 1.000 vs 1.000 (zero degradation); official retrieval_score +0.126 (B-PG vs B0); +11.1% e2e / +23.2% prefill speedup; **router-mass proxy off by ~18pp vs measured strict_accuracy**.
