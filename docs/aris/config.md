# ARIS Project Config

## Paths
SKILL_ROOT = skill/Auto-claude-code-research-in-sleep/skills
PROJECT_ROOT = /home/lzy/Artifact-Infer
DOCS_ROOT = /home/lzy/Artifact-Infer/docs
OUTPUT_ROOT = /home/lzy/Artifact-Infer/docs/aris/idea-expansion
TRACE_ROOT = /home/lzy/Artifact-Infer/docs/aris/traces
RUNS_ROOT = /home/lzy/Artifact-Infer/runs
BRIEF_COMMUNICATION_CENTRIC = /home/lzy/Artifact-Infer/docs/research/2026-05-28_communication-centric/00_brief.md
BRIEF_PHASE_ASYMMETRIC = /home/lzy/Artifact-Infer/docs/research/2026-05-28_phase-asymmetric/00_brief.md
BRIEF_WORKLOAD_ADAPTIVE = /home/lzy/Artifact-Infer/docs/research/2026-05-28_workload-adaptive/00_brief.md

## Models
EXECUTOR = claude
REVIEWER_BACKEND = codex
REVIEWER_MODEL = gpt-5.5
REVIEWER_REASONING = high
MAX_INTERNAL_DRAFT_ROUNDS = 2

## Venue
VENUE = MLSys 2026
ASSURANCE_LEVEL = draft

## GPU 资源约束
TOTAL_GPUS_AVAILABLE = 8
MAX_GPUS_PER_PILOT = 8
MAX_PARALLEL_PILOTS = 1
RESERVED_GPUS = []
EP_SIZE = 8                       # 所有 pilot 必须 EP=8,和历史 baseline 可比

## Experiment 约束
PILOT_VALIDATION = enabled
EXPERIMENT_ISOLATION = git_worktree
MAX_PILOT_HOURS = 4
FAIL_FAST = enabled

## Codex 调用约束(省额度)
- 只对综合评分 top 5 的 idea 调 GPT review
- 每个 idea 的 GPT review 限 1 轮
- 撞车检查只对最终推荐的 3 个 idea 调 GPT 二次核验

## === 以下为补充约束 ===

## 流程控制(最重要)
AUTO_PROCEED = false              # 每个阶段结束必须停下等人工确认
STAGE_GATE = manual               # 5 个阶段之间是人工 gate,不可自动跨越
NO_SCOPE_CREEP = true             # 严禁超出当前阶段任务范围,"顺便做X"一律禁止
STOP_ON_UNEXPECTED = true         # 遇到任何意外/报错/歧义,停下问人,不自行决定

## 实验权限分级(按阶段)
- 阶段 A (idea生成): 禁止任何 GPU 实验、禁止改源代码
- 阶段 B (撞车检查): 禁止任何 GPU 实验、禁止改源代码
- 阶段 C (pilot设计): 禁止跑实验、禁止写实际代码(只写文字 code plan)
- 阶段 D (pilot执行): 仅在人工逐个审批后才能跑;每个 pilot 单独审批
- 阶段 E (综合推荐): 禁止任何 GPU 实验

## 代码安全
- 禁止修改 main 分支;所有 pilot 改动走 git worktree + pilot/{idea_id} 分支
- 实验前必须: 跑现有 test suite 确认 baseline 通过 + git status 确认工作区干净
- pilot 改动只 commit 到 pilot 分支,禁止 merge 回 main
- 禁止修改 SKILL_ROOT 下的任何 skill 文件
- 禁止删除或覆盖已有的实验数据/结果文件

## 文献真实性(防幻觉)
- 所有 paper 引用必须经 web_fetch 验证存在,验证失败标记 verified=NO
- GPT(Codex) 提到的任何 paper,Claude 必须独立用 web_fetch 二次验证
- 严禁基于训练记忆引用论文;凭记忆写的标 [MEMORY-NEEDS-VERIFY]
- 撞车判断("已被发表"/"没撞车")必须有具体 paper + 具体差异支撑,不接受空泛结论

## Idea 质量门槛
- 每个 idea 必须可证伪,有明确 yes/no hypothesis
- 严禁 "探索X的可能性" / "未来可以研究Y" 这类无法验证的空话
- 严禁同一 idea 换措辞重复计数
- novelty 自评必须给具体理由,不接受裸分

## 产出规范
- 所有输出文件写入 OUTPUT_ROOT 或 TRACE_ROOT 或 RUNS_ROOT,不得散落到项目其他位置
- 每个阶段结束生成一份该阶段的 summary,列: 做了什么 / 产出在哪 / 下一步建议
- GPT review 的完整 trace 必须落盘到 TRACE_ROOT,不能只在对话里说
- 中间产物(idea_pool/scoring/lit_check)用 markdown,结果数据用 jsonl

## 汇报规范
- 每个阶段结束停下,用中文汇报,包含可量化的结果(数量/评分/消耗)
- 汇报中明确指出: 哪些地方需要人工判断、哪些地方 Claude 不确定
- 不夸大结果;pilot 结论必须如实标 YES/NO/INCONCLUSIVE,不 over-claim