# Trace — GATE A 埋点 patch 跨模型审查

- Date: 2026-05-29
- Reviewer: Codex (mcp__codex__codex), model_reasoning_effort=high, sandbox=read-only
- Thread ID: 019e71ae-0929-7e40-a428-9f0457e8dc40
- cwd: /home/lzy/Artifact-Infer-collection (worktree, branch collection/per-step-routing)
- Patch reviewed: docs/aris/idea-expansion/04_collection_patch.diff (294 insertions)
- 依据 skill: experiment-bridge Phase 2.5 (CODE_REVIEW=true), reviewer-independence

## 四个审查点 + Codex 裁决

1. **total_recv 读取顺序 — OK**。recorder 在 `:447` 之后,total_recv 在 `:383` 已算;send_counts/recv_counts 在 `:365-366` 已 `.tolist()`。记录的 counts 是 **post-drop kept replicas**(send_counts_t 来自 `target_rank[sort_perm]`,sort_perm 是 drop 后的 kept 排列),与真实 payload a2a (`:398-411`) 用的是同一对 list。`L_recv = sum(recv_counts)` == `total_recv`。无"读取早于计算"。

2. **recv_topk_w / combine 1.0 语义 — OK**。diff 只增加 recorder block + 新文件,未碰 `recv_topk_w` / `recv_topk_weights` / ExpertsEPHT / CombineEPHT。kernel 仍收 1.0 权重(experts_ep_ht.py:140),combine 仍用 `tok_meta.topk_weights` 加权(combine_ep_ht.py:89)。加权路径 byte-for-byte 不变。

3. **隐藏 CUDA 同步 — OK**。recorder 路径无任何新增 `.tolist()/.item()/.cpu()/synchronize()`/kernel launch。唯一的 `.tolist()` 是上游既有的 `:365-366`。recorder 输入全是 host/Python 值(counts lists、T/H shape int、element_size 元数据、self.* 标量、is_prefill bool);内部仅 Python sum/int/json/write。

4. **flag-off 零开销 — OK**。关闭时热路径只多:module import(sys.modules 缓存,可接受)+ `enabled()`(一次缓存 bool)。`_ENABLED` 初始 None,首次 `enabled()` 调用才读 env(:45),import 本身不读 env。缓存正确。**有意的 caveat**:首次 `enabled()` 后改 env 不生效,除非 `reset()` —— 对采集无影响(env 在 import 前设好)。

## 其他扫描

- 无算法正确性问题。
- step rollover 逻辑对"每 forward 内 layer_id 单调递增"成立。
- 文件生命周期:lazy open + line-buffered append + reset 关闭,适合 opt-in trace。
- **唯一 minor 措辞 caveat**:"crash-safe" 指流式逐行写,**非 fsync 持久化**,断电/宕机可能丢未刷盘行。→ 已知,pilot 采集可接受(进程正常结束会 flush);如需强持久可加周期 fsync。

## 结论

**无 CRITICAL / MAJOR 问题,无需返工。** 仅 1 个 minor 文档措辞 caveat(fsync),不影响采集正确性。可进入 Stage 2(GATE A 通过后)。
