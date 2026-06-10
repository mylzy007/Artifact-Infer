# ARIS Idea 池

在 ARIS 阶段 A（idea 生成）下产出。仅追加，不覆盖。

- 阶段范围：只做 idea 生成；不跑任何 GPU 实验；不改源码；不下"已撞车 / 没撞车"的正式结论。
- 所有论文引用均经过 Claude 独立 `web_fetch` 验证（`verified=YES/NO`）。
- 自评分规则（1–5，Overall 1–10）：Novelty potential、MLSys fit、Mechanism clarity、Feasibility on 8×RTX4090 EP=8、Risk of being just engineering（分数越高越偏纯工程 / 越不像研究）、Overall。

---

## 维度 1 — Communication-Centric MoE Framework

本维度的核心 user finding：在 Qwen3-30B-A3B / 8×RTX4090 / EP=8 / owner_local_ep / ep_ht runtime 上，**MoE drop 的端到端收益中有 97% 来自 dispatch+combine a2a payload 字节减少**（dispatch −47% bytes + combine −50% bytes，Tier-1 segment ablation，T=2048 cell），只有 **2.8% 来自 expert GEMM**。Break-even `L* ≈ 3271` rows/rank；甜点 `L_recv ≈ 4k`。生产配置 `tail_weight @ r=0.3` + `MOE_DROP_MIN_REPLICAS=512` 在 LongBench long-prompt 上拿到 prefill +23% / e2e +12% / 精度零损失；decode（`L_recv ≈ 8`）无任何收益，确认瓶颈是通信字节，不是计算。→ 本维度所有 idea 都把 **a2a payload bytes / 跨 GPU 流量 / dispatch-combine 通信** 当作一等优化目标，而不是 expert workload balance。

### D1-Idea-1：Payload-aware top-K 路由（byte-cost-augmented router）

- 维度：1 — Communication-Centric MoE Framework
- 一句话主张：在每 token top-K 路由器上加一个 byte-cost 正则项，使得当多个 expert 分数相近时优先选已经在 token owner rank 本地的、或目标端已经被批量聚集的那个，从而无需重训即可降低 dispatch+combine 总字节。
- Yes/No 假设：一个 drop-in 的 byte-cost-augmented router（在 router logit 上加 λ·byte_cost，λ 事后调）能在 LongBench long-prompt 上把 dispatch+combine 总字节降 ≥10%，精度 delta ≤0.5pp。
- 核心机制：每个 rank 按 layer 维护一个 per-peer pending dispatch bytes 的滚动估计。路由器分数变为 `logit_e − λ·estimated_bytes(e | this_token's_rank)`；expert `e` 的副本按最小代价选。无任何权重更新。
- 与 user finding 的关系：bytes 是瓶颈（97%）；router 当前只看 expert 分数，不看它自身诱发的通信代价。
- 相对现有 drop 结果的新意：drop 通过*移除 token* 减字节；本 idea 通过*把 token 引向更便宜的 rank-pair* 减字节，与 drop 兼容。
- 最小 pilot sketch，不执行：纯离线 sketch —— 重放 v6 LongBench grid 的 router 输出，在 `λ ∈ {0, 0.05, 0.1, 0.2}` 下重算 permutation，统计字节，计算结果 expert 集合与原始的 Jaccard。无 GPU runtime 改动。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：在 Jaccard(expert-set) ≥ 0.9 的前提下字节减少 ≥10%。
- 证伪准则：当 λ ≥ 0.2 时，要么字节减少 <5%，要么 top-1 expert 重合度 <0.8。
- 主要 baseline：`ep_ht` owner_local_ep 模式下的 stock top-K router，不开 drop。
- 主要风险：当 λ 足够有意义时，精度可能因 "便宜 expert" 与 "分数最佳 expert" 结构不同而退化。
- 文献参考：
  - DeepSpeed-MoE (Rajbhandari et al., 2022), verified=YES, https://arxiv.org/abs/2201.05596, 相关性：把 a2a 当作固定开销的端到端 MoE 推理框架；本 idea 把这部分开销变成 router 变量。
  - Tutel: Adaptive Mixture-of-Experts at Scale (Hwang et al., 2022/2023), verified=YES, https://arxiv.org/abs/2206.03382, 相关性：MoE 自适应并行；不按 per-peer byte cost 改 router。
- Claude 自评：
  - Novelty potential: 3.5
  - MLSys fit: 4
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4
  - Risk of being just engineering: 3
  - Overall: 7.5
- GPT/Codex critique：本轮未入选 GPT review（Overall 未进 top 5）。
- Claude 回应 GPT：n/a
- 决策：保留

### D1-Idea-2：跨层 dispatch permutation 复用

- 维度：1 — Communication-Centric MoE Framework
- 一句话主张：prefill 中相邻 MoE 层之间，非平凡比例的 token 选择强重叠的 expert 集合；缓存 layer ℓ 的 dispatch permutation 并在 ℓ+1 对这部分 token 复用，从而跨层省一次 dispatch a2a。
- Yes/No 假设：在 per-token Jaccard ≥ 0.7（层 ℓ 与 ℓ+1 的 router top-K）下，LongBench prefill 上 ≥25% 的 token 允许 permutation 复用，整体 dispatch 字节降 ≥10%，精度退化 ≤0.5pp。
- 核心机制：每层 router 输出按 expert 集合做 hash；permutation cache 以 (token-id, hash) 为 key。ℓ+1 命中时，该 token 走缓存路径，本地路由。
- 与 user finding 的关系：bytes 是瓶颈；跨层 routing 相似性是已观察现象（ExFlow），但被用于 placement 而非 runtime 字节消除。
- 相对现有 drop 结果的新意：drop 减少*哪些 token* dispatch；本 idea 减少*每 token 的 dispatch 次数*。
- 最小 pilot sketch，不执行：离线重放 —— 给定 v6 LongBench prompt-cell，记录每层 router 输出，测量 Jaccard(ℓ, ℓ+1) 分布，再统计如果复用 permutation *原则上* 能省多少 dispatch 字节。无 runtime 改动。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：Jaccard ≥ 0.7 覆盖 ≥25% token 且对应字节 ≥10% of total。
- 证伪准则：<10% 的 token 达到 Jaccard ≥ 0.7；或即使达到，ℓ+1 的 fresh activation 依然必须重新跨 wire 传输（机制失败）。
- 主要 baseline：标准 per-layer dispatch（无复用）。
- 主要风险：**机制可能崩塌** —— 即便 router top-K 重叠，ℓ+1 的*隐藏状态*仍是 fresh 的，必须跨 wire 抵达 expert。GPT critique 明确点出（见下）。
- 文献参考：
  - Exploiting Inter-Layer Expert Affinity for Accelerating MoE Model Inference (ExFlow) (Yao et al., 2024), verified=YES, https://arxiv.org/abs/2401.08383, 相关性：直接研究跨层 expert affinity 以减少 all-to-all routing —— 与本 idea 最强解释下高度重叠。
  - Path-Constrained Mixture-of-Experts (Gu et al., 2026), verified=YES, https://arxiv.org/abs/2603.18297, 相关性：架构层面研究跨层 expert path consistency。
- Claude 自评：
  - Novelty potential: 4
  - MLSys fit: 4.5
  - Mechanism clarity: 3（GPT 之后下调）
  - Feasibility on 8×RTX4090 EP=8: 3
  - Risk of being just engineering: 3
  - Overall: 6（从 7.5 下调）
- GPT/Codex critique：
  - 机制是否明确？**否** —— 复用 permutation 本身并不省 dispatch 字节；ℓ+1 的 fresh activation 仍需抵达 ℓ+1 自己的 expert。
  - 最强 objection：「如果下一层仍需要把 fresh hidden state 路由到自己的 expert，到底是哪段 tensor 传输被消除了？」
  - 初步撞车风险：ExFlow (IPDPS'24, arXiv 2401.08383)；Path-Constrained MoE (arXiv 2603.18297)。
- Claude 回应 GPT：我同意此机制如所写是站不住的。"复用" 真正能省 wire bytes 的唯一路径要么是 (a) 把 ℓ 与 ℓ+1 的 expert 在同一 rank 上为 affinity 高的 token 共置（这就是 ExFlow），要么是 (b) ℓ+1 的 expert 输入能由 ℓ 的本地 residual 推导出而无需再传——这需要我无法辩护的架构假设。在最强解释下，本 idea 直接退化为 ExFlow。我看不出既能与 ExFlow 区分、又在标准 transformer residual 下机械上成立的可挽救表述。
- 决策：**drop**（机制塌进 ExFlow，无可辩护的字节节省路径）

### D1-Idea-3：FP8 / INT8 dispatch+combine payload 量化

- 维度：1 — Communication-Centric MoE Framework
- 一句话主张：仅在 a2a hop 段把 dispatch payload（token 隐藏状态）与 combine payload（per-token 加权 expert 输出）量化到 FP8 / INT8（带 stochastic round），接收端立即反量化，使 wire bytes 在任何 token-side drop 之前就下降 ~50%。
- Yes/No 假设：FP8 dispatch+combine（E4M3 + per-token scale + stochastic round）在 LongBench long-prompt 上 wire 字节减少 ≥35%，精度 delta ≤0.5pp，且与 `tail_weight @ r=0.3` 加法叠加到 wire 字节总减少 ≥55%。
- 核心机制：发送端的 dispatch packing kernel 输出 FP8 而非 BF16/FP16（RTX 4090 Ada FP8 tensor core 支持 E4M3）；接收端在 expert GEMM 之前立即反量化。combine 路径对称。其它任何地方 expert 权重 / 激活的精度不变。
- 与 user finding 的关系：bytes 主导；最便宜的 per-bit 减字节就是对跨 wire 的那段 tensor 做精度收缩。
- 相对现有 drop 结果的新意：drop 是*token 侧*的减字节；本 idea 是 *payload 精度侧*的减字节；两者应可叠加。
- 最小 pilot sketch，不执行：离线 calibration —— 让记录的 dispatch/combine tensor 走软件 FP8 编解码，测量 per-token 输出漂移与 aggregate 精度代理（如 logit cosine vs BF16 baseline）。无 runtime 改动。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：E4M3 + per-token scale 下 per-token logit cosine ≥ 0.999。
- 证伪准则：即使加 stochastic round，per-token cosine <0.997；或与 drop 叠加后字节额外节省 ≤25%（即不可叠加）。
- 主要 baseline：BF16 dispatch/combine + `tail_weight @ r=0.3`。
- 主要风险：看起来像 "applied FP8" —— 差异化贡献必须落在与 token-level drop 的*组合性*和 byte-budget framing 上，而非 FP8 本身。
- 文献参考：
  - LSH-MoE: Communication-efficient MoE Training via Locality-Sensitive Hashing (Nie et al., NeurIPS 2024), verified=YES, https://arxiv.org/abs/2411.08446, 相关性：训练时通过 payload 重新表示（hash）减 a2a；本 idea 针对推理且 operator 不同。
- Claude 自评：
  - Novelty potential: 3
  - MLSys fit: 4
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4
  - Risk of being just engineering: 4
  - Overall: 6.5
- GPT/Codex critique：本轮未入选 GPT review。
- Claude 回应 GPT：n/a
- 决策：保留（作为 D1-I4 / D1-I6 的组合性研究候选）

### D1-Idea-4：Wire-byte budget runtime controller

- 维度：1 — Communication-Centric MoE Framework
- 一句话主张：把 per-rank dispatch+combine 字节 per layer-step 当作受控变量；一个在线 controller 在 `B*`（latency 侧目标字节预算）+ `m*`（quality 侧 router-margin 下限）约束下，每步选 (drop policy, drop rate, 可选量化 step) 这套动作。
- Yes/No 假设：一个目标为 (`B*`, `m*`) 的两变量 controller 在 LongBench long-prompt、GSM8K、混合形态 stream 上能持平或击败 best per-workload-static (`drop policy`, `r`) 配置，精度落在 static optimum ±0.5pp 内。
- 核心机制：在 layer ℓ 结束时 controller 读取 `(L_recv_ℓ, bytes_ℓ, router_margin_ℓ)`，从一张小的离散表（按 `L_recv` 桶 × `B*` 档 × `m*` margin 索引）选 layer ℓ+1 的动作。closed-loop；byte budget 是 latency 侧目标，router margin 是 quality 侧 guard。
- 与 user finding 的关系：bytes —— 不是 L_recv，也不是 rate —— 是真正的 lever（97%）；现有 drop policy 把 `r` 当部署常量。
- 相对现有 drop 结果的新意：drop 是静态旋钮；这里把它升级为带显式字节目标的 runtime 受控变量。
- 最小 pilot sketch，不执行：离线模拟器 —— 用记录的 `L_recv` 与 router-margin 重放 v6 LongBench grid，在 held-out 子集上评估 controller，对比每个 static 配置的字节节省与精度代理。无 GPU runtime 改动。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：simulated controller 在至少一个 workload 类上击败 per-workload best static，且其它类不输。
- 证伪准则：所有 controller law 下没有任何 workload 类显示 ≥3% 字节节省优势；或 router-margin guard 无法阻止 GSM8K 混合 stream 上 >0.5pp 的精度下降。
- 主要 baseline：static `tail_weight @ r=0.3, MOE_DROP_MIN_REPLICAS=512`（生产配置）。
- 主要风险：bytes 在 per-step 粒度上是被混淆的 latency 信号；controller 在异构混合上可能震荡。
- 文献参考：
  - Toward Efficient Inference for Mixture of Experts (Huang et al., NeurIPS 2024), verified=YES, https://proceedings.neurips.cc/paper_files/paper/2024/hash/98bf3b8505c611ac21055dd9d355c66e-Abstract-Conference.html, 相关性：刻画 MoE 推理 inefficiency（含通信）——邻近的诊断 baseline；差别在它们不提 runtime byte-budget controller。
  - Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts (He et al., 2025, ICLR 2026), verified=YES, https://arxiv.org/abs/2503.05066, 相关性：推理期 token drop/reroute，但目标是 straggler，不是 byte budget。
- Claude 自评：
  - Novelty potential: 4
  - MLSys fit: 5
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4
  - Risk of being just engineering: 3
  - Overall: 8.0
- GPT/Codex critique：
  - 最强 objection：单一 byte budget 不是 quality-aware；同样字节的两步可能精度敏感性完全不同。
  - 最可能失败模式：controller 在语义脆弱的 layer 上震荡或过 drop，因为 `L_recv` 是滞后的、纯形状代理。
  - 机制：部分指定 —— controller law 与 quality guard 缺失。
  - 锋利问题：为什么 bytes 一个变量就够同时控住 latency 和 accuracy？
  - 初步撞车风险：NeurIPS'24 "Toward Efficient Inference for MoE"；ICLR'26 Capacity-Aware Inference。
- Claude 回应 GPT：我同意 quality-blindness 这个 objection。我把 thesis 改写为 **两变量 controller**：`B*`（latency）+ `m*`（quality guard，router-margin floor），并显式把 controller law 写成小的离散查找表（不是学得的 policy），避免越出阶段 A 的范围。撞车论文是真邻近但不是重复：Capacity-Aware Inference 目标是 expert imbalance 的 straggler latency，不是 dispatch/combine 字节 budget；NeurIPS'24 是 characterization，不是 controller。后续 pilot 要把区别讲清楚。
- 决策：**revise**（升级为双控变量 + 显式 quality guard；继续推进）

### D1-Idea-5：Payload-bottleneck-cut expert replica placement

- 维度：1 — Communication-Centric MoE Framework
- 一句话主张：把 MoE replica placement 重新表述为**最小化最坏 per-peer dispatch+combine 字节负载**（每层 expert-to-rank 分配图上的 min-max-flow / bottleneck-cut），而不是 expert workload balance；目标是把最慢的那条 a2a 边压扁，而不是平均的那条。
- Yes/No 假设：bottleneck-cut placement（目标 = 对 rank-pair 字节取 min-max，约束 = per-rank VRAM 容量）相比 LBG-overlap=0.25 在同显存预算下把*最坏 per-peer* dispatch+combine 字节降 ≥15%，并在同一 Qwen3-30B-A3B / 8×RTX4090 setup、`L_recv ≥ 4k` 时把 prefill 提速 ≥6%；router 与权重均不动。
- 核心机制：构建 bipartite graph（token-source-rank × expert-replica-slot），边权 = calibration 集上 router 输出 histogram 推得的期望字节；在 VRAM 与 replica budget 约束下求小规模 min-max-flow / ILP，得到 per-expert replica 数 + per-rank 分配。静态 placement，不做 runtime 控制。
- 与 user finding 的关系：现有 placement（LBG-overlap）优化 expert-compute load balance；97%/2.8% 的拆分说明 compute balance 是错的 loss —— wire bytes、尤其是 gate 着 collective 的 worst-rank bytes，才是对的 loss。
- 相对现有 drop 结果的新意：drop 是在各 rank-pair 上均匀减字节；本 idea 改变*字节流向*，使最慢边更小 —— 这正是 a2a collective 实际在等的量。
- 最小 pilot sketch，不执行：纯离线 —— 用 v6 LongBench 记录的 router 输出，在 (i) LBG-overlap=0.25、(ii) round-robin、(iii) 本方案 bottleneck-cut placement 三种下重算 per-rank dispatch byte histogram；比较 `max-per-peer-bytes` 与 `total-bytes`。无 runtime 改动。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：bottleneck-cut placement 在至少一个主要 LongBench 数据集上同时把 `max-per-peer-bytes` 与 `total-bytes` 降到 LBG-overlap=0.25 之下。
- 证伪准则：bottleneck-cut placement 让 `total-bytes` 降但 `max-per-peer-bytes` 升；或两者都没降。
- 主要 baseline：LBG-overlap=0.25（user 在 Phase-3 的最佳静态 placement）。
- 主要风险：bottleneck-cut 目标只在 collective 真正瓶颈在最慢链路时严格更优；在 launch 开销主导的拓扑上两个指标可能都饱和。
- 文献参考：
  - MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing (Go & Mahajan, 2025), verified=YES, https://arxiv.org/abs/2502.06643, 相关性：ILP 做 placement，但 loss 是 balance 而非 bottleneck-bytes —— 直接对比对象。
  - Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling (Li et al., ICLR 2026), verified=YES, https://arxiv.org/abs/2503.04398, 相关性：通过 token/expert 协同调度减 a2a —— 目标邻近；区别是它们 co-schedule data 与 model，本方案在 static 部署期替换 placement 目标。
  - Scaling Multi-Node MoE Inference Using Expert Activation Patterns (Bambhaniya et al., 2026), verified=YES, https://arxiv.org/abs/2604.23150, 相关性：workload-aware micro-batch 分组 + expert placement 优化 locality；单位（token 分组）和指标（locality 而非 bottleneck-bytes）不同。
  - FasterMoE (He et al., PPoPP 2022), verified=YES, https://dl.acm.org/doi/10.1145/3503221.3508418, 相关性：topology-aware gate / dynamic shadowing；loss 是 balance + locality，不是 bottleneck a2a bytes。
- Claude 自评：
  - Novelty potential: 4.5
  - MLSys fit: 5
  - Mechanism clarity: 4（从 3.5 上调，因为已显式落到 bottleneck-cut）
  - Feasibility on 8×RTX4090 EP=8: 3.5
  - Risk of being just engineering: 2
  - Overall: 8.5
- GPT/Codex critique：
  - 最强 objection：co-activation 图上的 cut 与真实 A2A 字节不等价；后者还取决于 token 源 rank、副本选择、per-peer skew、collective 实现。
  - 最可能失败模式：理论上减少了 remote-expert 命中，但没减少 critical-path A2A latency，因为字节集中在少数 rank-pair 上。
  - 机制：部分指定 —— 副本路由、per-peer byte 模型、拓扑、负载约束都未给。
  - 锋利问题：在真实 EP collective 下，co-activation graph 的 cut 与 dispatch+combine 字节最小化的等价性从何而来？
  - 初步撞车风险：Semantic Parallelism (ICLR 2026)；Scaling Multi-Node MoE Inference via Activation Patterns (arXiv 2604.23150)。
- Claude 回应 GPT：完全同意 "graph-cut ≠ critical-path bytes" 这一 objection —— 这正是我把 thesis 从平 min-sum-cut 改写为 **min-max-flow / bottleneck-cut** 的根本原因。collective 等的是最慢 rank-pair，所以唯一在因果上正确的目标就是 `max-per-peer-bytes`，不是 sum-bytes。我现在在证伪准则里同时显式带两个 metric。撞车风险方面：MoETuner 在*机制*上最近（ILP placement）但优化 balance，不是 bytes；Semantic Parallelism 在*目标*上最近（减 a2a）但 co-schedule data flow，没替换 placement 目标；两者都是真邻近但都未把 "bottleneck-bytes per rank-pair" 当 loss。我和 GPT 的分歧：阶段 A 我不认为这个 idea 被撞死。
- 决策：**revise**（目标从 min-sum-cut 改写为 min-max bottleneck-cut；显式把 MoETuner 和 LBG-overlap 加进 baseline；作为 D1 最强候选继续推进）

### D1-Idea-6：Combine-asymmetric drop

- 维度：1 — Communication-Centric MoE Framework
- 一句话主张：只在 combine 路径上 drop（在 expert compute 返回加权输出之后），保留 dispatch 完整，让 drop 决策基于 post-expert 幅度而不是 pre-expert router score。
- Yes/No 假设：在等量*combine 字节*节省下，combine-only drop 在 LongBench long-prompt 上以等 combine bytes 优于 symmetric (dispatch+combine) drop 的精度；并在精度零退化下，单独通过 combine drop 拿回 97% a2a 收益里 combine 那一半的 ≥50%。
- 核心机制：在 expert compute 之后，每个 expert 持有 per-token 输出向量 `o_e,t`（被 `g_e,t` 加权）。combine sender 丢掉 `‖g_e,t · o_e,t‖_2` 低于自适应百分位的 token；receiver 对存活贡献做标准 combine。
- 与 user finding 的关系：dispatch 和 combine 大致各贡献 50%（−47% / −50%）于 97% e2e 收益；combine-side drop 用的是*post-expert 信息*（实际贡献幅度），所以在等量 combine 字节节省下应严格更好地保留精度。
- 相对现有 drop 结果的新意：现有 drop policy（v1–v6）在 dispatch 和 combine 两侧对称丢；本 idea 首次利用两侧之间的*信息不对称* —— 是 quality-asymmetry，不是 latency-asymmetry。
- 最小 pilot sketch，不执行：离线重放 —— 拿 v6 记录的 dispatch tensor，在一个子集上重新跑 expert GEMM（数据已记录，无新 GPU 运行），算 `‖g·o‖` 分布，在 r∈{0.1,0.2,0.3} 下模拟 combine-side percentile drop，比较 per-layer 输出 cosine vs reference 与 vs 等 combine 字节下的 symmetric drop。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：在等 combine 字节节省下，≥80% 的 layer 上 combine-only drop 的 per-layer 输出 cosine 严格高于 symmetric drop。
- 证伪准则：combine-only drop 在精度上无法支配 symmetric drop；或低 `‖g·o‖` 与 task-critical direction 高相关（drop 它们比 drop dispatch 端低 `g` 的 token 还差）。
- 主要 baseline：`tail_weight @ r=0.3`（symmetric drop 家族，user 生产配置）。
- 主要风险：combine-side drop 在拿到节省字节之前*同时付*了 dispatch 与 expert GEMM —— 所以如果有 latency 收益，也只能来自 97% 里 combine 那一半；在纯 latency 上不可能击败 symmetric drop，只能在 quality-vs-bytes Pareto 前沿上赢。
- 文献参考：
  - Not All Experts are Equal: Efficient Expert Pruning and Skipping for MoE LLMs (Lu et al., 2024, ACL 2024), verified=YES, https://arxiv.org/abs/2402.14800, 相关性：整 expert 跳过（决策单位不同），但 importance criterion 上有重叠。
  - Finding Fantastic Experts in MoEs: A Unified Study (Jaiswal et al., 2025), verified=YES, https://arxiv.org/abs/2504.05586, 相关性：benchmarks expert-dropping criteria；importance-estimation 方法论重叠。
- Claude 自评：
  - Novelty potential: 4
  - MLSys fit: 4
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4
  - Risk of being just engineering: 2
  - Overall: 8.0
- GPT/Codex critique：
  - 最强 objection：combine-only drop 先付了 dispatch 与 expert compute，所以 latency 主张在结构上比 symmetric drop 弱，除非 combine 独立主导。
  - 最可能失败模式：低 router weight ≠ 低 post-expert 贡献；高范数但低权重的贡献可能 task-critical。
  - 机制：部分指定 —— post-expert importance criterion 未给。
  - 锋利问题：在 dispatch + GEMM 已经付了的前提下，什么 post-expert 信号能比 pre-expert router score *更好* 地决定丢谁？
  - 初步撞车风险："Not All Experts are Equal" (ACL 2024, 2402.14800)；"Finding Fantastic Experts" (arXiv 2504.05586)。
- Claude 回应 GPT：部分同意。(a) latency：GPT 对的，*latency* 比较结构上更弱 —— 但 user 的 segment ablation 已经证明 combine 在 97% 里贡献 ≈50%（约 48.5%），所以 combine-bytes 这个 lever 本身是有量的。thesis 必须重定位为 **等 combine 字节下的 quality 优势**，而不是与 symmetric drop 的 latency 持平。(b) post-expert 信号：我把 criterion 钉死在 `‖g_e,t · o_e,t‖_2`（gate × expert output 的合并幅度），它严格比 `g_e,t` 信息更多，因为反映了 expert 对 residual stream 的实际贡献。撞车风险方面：两篇都是关于*expert-level* 跳过（对一个 expert 跨所有它的 token 丢掉），不是 per-(expert, token) combine-bytes drop —— 决策单位不同。真邻近，但不是重复。
- 决策：**revise**（重定位为 dispatch 侧与 combine 侧 drop 在等 combine 字节下的 quality 不对称；importance criterion 钉到 `‖g·o‖_2`；继续推进）

### D1-Idea-7：基于期望 byte ROI 的 per-token K_eff thinning

- 维度：1 — Communication-Centric MoE Framework
- 一句话主张：自适应选 per-token `K_eff ∈ {1, …, K}`，使第 k 个 expert 只有在其期望边际贡献（对 layer 输出）超过其期望边际 dispatch+combine 字节代价乘上全局价 `p` 时才被纳入。
- Yes/No 假设：基于 score-gap 的 per-token `K_eff`（当 `score_k / score_1 < θ` 时丢 expert `k`）在 LongBench long-prompt 上让 a2a 总字节降 ≥20%，精度 delta ≤0.5pp，`θ` 按数据集调。
- 核心机制：router 之后按分数排序，保留满足 `score_k / score_1 ≥ θ` 的 k；约束 `K_eff ∈ [1, K]`。对保留的 token 不施加任何 drop。
- 与 user finding 的关系：字节随 K 线性扩张（dispatch 路径上 K·tokens-on-wire）；降平均有效 K 是不与 replica 选择交互的 byte lever。
- 相对现有 drop 结果的新意：v1–v6 drop 家族在 expert 内均匀丢 token；per-token K_eff 在 token 内丢 expert，互补；user 之前的 K_eff 硬截负杠杆是因为它瞄准的是 compute 而不是 bytes。
- 最小 pilot sketch，不执行：离线 —— 在记录的 router 输出上 sweep `θ ∈ {0.3, 0.5, 0.7}`，假设只剩存活 expert 时重算 dispatch 字节，与固定 K=8 对比。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：`θ=0.5` 时平均 K_eff 降到 ≤4，字节减少 ≥20%，per-layer 输出 cosine ≥ 0.998。
- 证伪准则：`θ=0.5` 时输出 cosine <0.99（重现 K_eff 硬截失败的形态）。
- 主要 baseline：固定 K=8（Qwen3-30B-A3B 默认）。
- 主要风险：本质就是 user 已经文档化过为负杠杆的 K_eff 硬截；必须证明 *per-token 自适应* K_eff 与 fixed truncation 行为本质不同。
- 文献参考：
  - Capacity-Aware Inference: Mitigating the Straggler Effect in MoE (He et al., 2025, ICLR 2026), verified=YES, https://arxiv.org/abs/2503.05066, 相关性：也在推理期动态调路由；目标是 straggler latency，不是 byte ROI。
- Claude 自评：
  - Novelty potential: 3.5
  - MLSys fit: 4
  - Mechanism clarity: 3.5
  - Feasibility on 8×RTX4090 EP=8: 3
  - Risk of being just engineering: 3
  - Overall: 6.5
- GPT/Codex critique：本轮未入选 GPT review。
- Claude 回应 GPT：n/a
- 决策：保留（作为 D1-I4 的互补 —— 可作为 controller 的动作之一）

### D1-Idea-8：跨层 dispatch/combine 软件流水

- 维度：1 — Communication-Centric MoE Framework
- 一句话主张：prefill 中，把 layer ℓ+1 的 dispatch a2a 与 layer ℓ 的 combine a2a + 本地 expert GEMM 跨 MoE 层做软件流水，使 critical path 上*暴露*的 a2a latency 近似于 1 次 a2a，而不是 2L 次。
- Yes/No 假设：两层滚动流水（next-layer dispatch 在 current-layer combine 返回之前就 issue）在 LongBench prefill 上隐藏 ≥40% 的累计 dispatch+combine 墙钟时间，*不改变*精度。
- 核心机制：把每个 MoE layer 拆成 (i) attention + router（compute-bound）、(ii) dispatch a2a、(iii) expert GEMM、(iv) combine a2a、(v) residual+MLP-tail（compute-bound）；重排使 ℓ+1 的 (i)+(ii) 与 ℓ 的 (iii)+(iv) 并发。仅需 scheduler 层面改动 owner_local_ep contract。
- 与 user finding 的关系：bytes 是瓶颈 → 减 critical path 上*暴露*的通信；这是 scheduling lever，不是 bytes lever。
- 相对现有 drop 结果的新意：drop 减*字节量*；流水减*单位字节量的暴露时间*。与 drop 可叠加。
- 最小 pilot sketch，不执行：离线 timeline 仿真 —— 用 v6 grid 已有的 per-stage 时长，在 serial vs 两段流水下构造 Gantt 图，报告 long-prompt prefill 上的预期加速。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：`L_recv ≥ 4k` 时 dispatch+combine 部分仿真加速 ≥1.4×。
- 证伪准则：仿真加速 ≤1.1×，意味着 compute stage 太短，吸不住 a2a hop。
- 主要 baseline：serial MoE-layer 执行（user 当前 schedule）。
- 主要风险：与 MoE 训练侧流水工作（Lancet、Comet）重度重叠；推理期框架必须证明相对 prefill-decode 专属调度的非平凡性。
- 文献参考：
  - Lancet: Accelerating MoE Training via Whole Graph Computation-Communication Overlapping (Jiang et al., MLSys 2024), verified=YES, https://arxiv.org/abs/2404.19429, 相关性：训练侧 whole-graph overlap；本 idea 是推理期且跨层专属。
  - Comet: Fine-grained Computation-Communication Overlapping for MoE (Zhang et al., 2025), verified=YES, https://arxiv.org/abs/2502.19811, 相关性：细粒度 MoE overlap；"有没有人做过了" 的主要对比对象。
- Claude 自评：
  - Novelty potential: 3.5（识别到 Comet/Lancet 密度后下调）
  - MLSys fit: 5
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 3.5
  - Risk of being just engineering: 4
  - Overall: 7.0
- GPT/Codex critique：本轮未入选 GPT review。
- Claude 回应 GPT：n/a
- 决策：保留（低优先级；只作为 vs Comet/Lancet 的差异化候选携带）

### D1-Idea-9：异构链路下的 topology-aware 副本注入

- 维度：1 — Communication-Centric MoE Framework
- 一句话主张：在异构带宽 EP 拓扑（如 PCIe-only 8×4090 box）上，把最热 expert 在带宽最便宜的 rank 上额外加副本，使最慢 a2a 链路承载更少流量。
- Yes/No 假设：在 8×RTX4090 PCIe 拓扑上，把 calibration 集上 activation 频率最高的 top-`R` expert 复制到 bandwidth-cheapest rank 上，在 ≤5% VRAM 开销下把 worst-per-peer dispatch 字节降 ≥10%。
- 核心机制：profile 目标机器的 pairwise PCIe 带宽；构造 per-pair effective-cost matrix；placement 层把复制作为离散决策，按 marginal worst-pair byte reduction 最大处加副本。静态。
- 与 user finding 的关系：bytes 瓶颈 → 异构链路下*最慢*链路主导；副本放置以短路这条链路是直接 lever。
- 相对现有 drop 结果的新意：drop 均匀减字节量；本 idea 把字节重定向到更快链路。
- 最小 pilot sketch，不执行：离线 —— 用系统 log 中已有的 PCIe pairwise 带宽测试输出；按 cost matrix 重算 placement；用 v6 logs 估新 placement 下 worst-per-peer 字节。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：仿真中 worst-per-peer 字节降 ≥10%，total 字节同时降或持平。
- 证伪准则：仿真中 worst-per-peer 字节降但 total 字节升；或任意可行 VRAM 开销下两者都不改善。
- 主要 baseline：LBG-overlap=0.25（homogeneous-cost placement）。
- 主要风险：与 D1-I5 显著重叠；差异化只靠 (a) 显式异构 cost matrix 和 (b) 用 replication 而非 cut 作为 operator。后续可能并入 D1-I5。
- 文献参考：
  - FasterMoE (He et al., PPoPP 2022), verified=YES, https://dl.acm.org/doi/10.1145/3503221.3508418, 相关性：topology-aware gate / dynamic shadowing —— 精神上最近；差别是 static-vs-dynamic 和 inference-vs-training。
- Claude 自评：
  - Novelty potential: 3.5
  - MLSys fit: 4.5
  - Mechanism clarity: 3.5
  - Feasibility on 8×RTX4090 EP=8: 3.5
  - Risk of being just engineering: 3.5
  - Overall: 7.0
- GPT/Codex critique：本轮未入选 GPT review。
- Claude 回应 GPT：n/a
- 决策：保留（下一阶段可能并入 D1-I5）

### D1-Idea-10：把 a2a 字节作为 MoE 一等性能模型

- 维度：1 — Communication-Centric MoE Framework
- 一句话主张：把 **per-rank worst-peer dispatch+combine wire bytes per layer-step** 定义为 MoE prefill latency 的主解释变量，并在本硬件 / runtime 上跨 drop policy 与 rate 显式证明：它对 per-layer 延迟的预测显著优于 `L_recv` rows 与 expert-GEMM FLOPs。
- Yes/No 假设：在已有 v1–v6 LongBench grid（~5,120 prompt-cell）上，`per-rank worst-peer wire bytes` 解释 ≥75% 的 per-layer prefill latency 方差（R²≥0.75），而 `L_recv` rows 单变量 R²≤0.55，`expert-GEMM FLOPs` 单变量 R²≤0.30。
- 核心机制：从现有 trace 文件重提 per-step `(worst-peer bytes, total bytes, L_recv, GEMM FLOPs, 实测 latency)` 元组（无新 GPU 运行）；拟三个线性回归；报告 R²、残差结构、和一个校准过的 cost model。
- 与 user finding 的关系：97% / 2.8% 说明 MoE 推理性能建模一直建在错的主特征上；建立这个指标本身就是贡献，并解锁本维度其它所有 idea。
- 相对现有 drop 结果的新意：当前 drop 工作把 bytes 当成*叙事*；本 idea 把它升级为可应用于其它 policy 的*实测 cost model*，并把范围诚实限定到单机 8×RTX4090 + `ep_ht`。
- 最小 pilot sketch，不执行：纯事后分析 —— 在已收集的 v6 trace 上做，加一两次额外日志 pass 收 per-peer（而非仅 per-rank）字节计数。无新 GPU run。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：`worst-peer wire bytes` 模型 R² ≥ 0.75，残差与 `L_recv` 无相关。
- 证伪准则：`L_recv` 单变量 R² ≥ 0.7；或 `worst-peer wire bytes` 作为 predictor 并不优于 `total bytes`。
- 主要 baseline：在 `L_recv` rows 上的回归；在 expert-GEMM FLOPs 上的回归。
- 主要风险：没有机制贡献的 characterization 论文 MLSys 审稿人不爱过；framing 必须把 metric 落到（至少一个）D1-I4 / D1-I5 / D1-I6 上。
- 文献参考：
  - Toward Efficient Inference for Mixture of Experts (Huang et al., NeurIPS 2024), verified=YES, https://proceedings.neurips.cc/paper_files/paper/2024/hash/98bf3b8505c611ac21055dd9d355c66e-Abstract-Conference.html, 相关性：也刻画 MoE 推理 inefficiency，但把通信当多个因素之一处理，没把 worst-peer wire bytes 单独拎出来作为主解释变量。
  - MixServe (Zhou et al., 2026), verified=YES, https://arxiv.org/abs/2601.08800, 相关性：fused AR-A2A 的 hybrid-parallel MoE serving；通信被建模为 scheduling cost function 的一部分，而非一等指标。
- Claude 自评：
  - Novelty potential: 4
  - MLSys fit: 4
  - Mechanism clarity: 4.5
  - Feasibility on 8×RTX4090 EP=8: 4.5
  - Risk of being just engineering: 2
  - Overall: 8.0
- GPT/Codex critique：
  - 最强 objection："bytes 解释 latency" 可能是被混淆的 characterization 结果，而非 systems contribution；a2a latency 还取决于 max per-peer bytes、不均衡、launch overhead、overlap。
  - 最可能失败模式：模型在一个 Qwen3/4090/drop 格子上 fit 但跨 decode、不同 EP layout、NCCL/DeepEP backend、多节点拓扑就崩。
  - 机制：部分指定 —— latency target、aggregation 粒度、硬件/backend 不变量未给。
  - 锋利问题：total wire bytes 是*因果*特征，还是 worst-rank/per-peer A2A 压力的代理？
  - 初步撞车风险：NeurIPS'24 "Toward Efficient Inference for MoE"；MixServe (arXiv 2601.08800)。
- Claude 回应 GPT：强同意 per-peer 这一点 —— 这正是修订后 thesis 用 **worst-peer** wire bytes 而非 total wire bytes 的原因。我显式把 per-peer 特征作为主回归量，total wire bytes 作为对照。关于泛化：我同意本模型诚实地只限定到单机 8×RTX4090 + `ep_ht` runtime；跨硬件外推超出本论文范围。关于 framing 风险：贡献必须打包为 "metric + mechanism"，不能 "metric alone" —— 即与 D1-I4 或 D1-I5 耦合以证明 metric 可操作。撞车论文是真邻近但都没把 worst-peer wire bytes 当主特征。
- 决策：**revise**（主回归量改为 worst-peer bytes；诚实限定单机范围；承诺与某个 mechanism idea 耦合以证 actionability）

---

### 维度 1 Summary

- 生成 idea 数：去重后 10 个（D1-I1 … D1-I10）。早期草稿合并："drop + payload-compression co-design" 并入 D1-I3；"general min-cut placement" 与 "topology-aware replica injection" 拆成 D1-I5 + D1-I9 两个候选并显式标注差异（cut objective vs replication operator）。
- 提交 GPT 的 top 5（一次 codex 调用打包，遵守每 idea 1 轮的预算）：
  - D1-I5（payload-bottleneck-cut placement）
  - D1-I4（wire-byte budget controller）
  - D1-I6（combine-asymmetric drop）
  - D1-I10（a2a-bytes as first-class metric）
  - D1-I2（cross-layer dispatch permutation reuse）
- 保留（无重大改动）：D1-I1, D1-I3, D1-I7, D1-I8, D1-I9
- 修订（按 GPT 回应做显式改动后继续推进）：
  - D1-I4：加 router-margin floor `m*` 作为第二控制变量，回应 quality-blindness。
  - D1-I5：目标从 min-sum-cut 改写为 **min-max bottleneck-cut**；显式把 MoETuner 加入 baseline。
  - D1-I6：贡献从 "latency parity" 重定位为 "quality-at-fixed-combine-bytes"；importance criterion 钉到 `‖g_e,t · o_e,t‖_2`。
  - D1-I10：主回归量从 total wire bytes 改为 **worst-peer wire bytes**；范围限定到单机；承诺与一个 mechanism idea 耦合。
- 淘汰：D1-I2（cross-layer dispatch permutation reuse）—— GPT 正确指出其唯一机械上成立的表述会塌进 ExFlow；在标准 transformer residual 下没有既新于 ExFlow 又机械成立的字节节省路径。
- GPT 主要批判：
  1. 单特征优化目标（光看 bytes / 光看 co-activation / 光看 router score）因果欠定 —— collective 等的是**最慢**那条边，不是平均，准确率也不止 bytes 一个因子。
  2. 多个 idea 机制欠规范：controller 怎么动作、post-expert importance 怎么定义、permutation reuse 究竟省的是哪段 tensor。
  3. 初步撞车风险存在但对保留的 idea 都不致命 —— 最近的 prior（Semantic Parallelism、MoETuner、ExFlow、Capacity-Aware Inference、NeurIPS'24 efficient-inference characterization、Comet、Lancet）在 *motivation* 上重叠但在 *objective* / *decision unit* / *runtime vs static* 上有真差异。唯一实际撞死 D1-I2 的是 ExFlow。
- Claude 与 GPT 的主要分歧：
  - D1-I5：GPT 说是 "错的抽象"；Claude 视为 "错的损失函数 (sum vs max)" 并修订而非废弃。真分歧：我认为 min-max-cut 是可发表的，min-sum-cut 不是。
  - D1-I6：GPT 强调 latency claim 弱；Claude 强调等 combine 字节下的*quality*不对称贡献 —— 我重新表述而非放弃。
  - D1-I10：GPT 当作被混淆的 characterization；Claude 同意 per-peer 但不同意 "characterization + actionable 耦合" 不可发表。
- Codex 用量估计：本维度共 2 次 codex 调用（1 次连通性 ping、1 次打包 critique）。远低于 15 次告警阈值。
- 下一推荐维度（不要执行）：**维度 2 — Phase-asymmetric MoE Optimization**（brief `BRIEF_PHASE_ASYMMETRIC`）。理由：D1 立了 "通信是 lever"；D2 把这点压在已经解释 decode 为何零收益的 phase asymmetry 上。两维度互补，D1 的 idea（特别是 I4 controller、I10 metric）能干净继承 phase-aware 结构。**等人工 gate。**

---

### 维度 1 — 质检结果（QC pass, 2026-05-28）

QC 范围：对 D1 的 10 个 idea 重新做 (1) 硬性门 —— 可证伪、具体性、与 finding 关系、重复性；(2) 修订后的质量评分 —— novelty / story_fit（severity：能 work 但不推进主故事必须低分）/ pilot_feasibility（离线或 8×RTX4090 EP=8 下 1–4 小时）/ pilot_cost（反向，5 = 最便宜）；(3) 每个 idea 一句话赌注；(4) 对 top 3 调 GPT 挑战。不做撞车结论，不进入阶段 B。

GPT 挑战 trace：`docs/aris/traces/dimension1_qc_review.md`。下表中分数为 *GPT 之后*，与 GPT 的分歧记录在 trace 里。

#### QC 表

| idea_id | 硬性检查 | novelty | story_fit | feasibility | cost | 一句话赌注 | 标记 |
|---------|----------|---------|-----------|-------------|------|------------|------|
| D1-I1 | PASS | 3 | 4.5 | 4.5 | 5 | router 在分数接近的多个 expert 之间存在 "byte-cheap 等价 expert"，且这种等价性稳定到可以用 `λ·bytes` 的 post-hoc 微调实现几乎无损偏置 | PASS |
| D1-I2 | REJECT — 机制崩塌 | — | — | — | — | layer ℓ+1 复用 ℓ 的 permutation 能省 dispatch bytes | REJECT（机制塌进 ExFlow；layer ℓ+1 的 fresh hidden state 仍需跨 wire 抵达自己的 expert） |
| D1-I3 | PASS | 2.5 | 4 | 4 | 5 | 在 a2a hop 这段距离上对 dispatch/combine payload 做 FP8 量化，能在不动专家权重的前提下安全减半字节，并与 token-side drop 加法叠加 | PASS（borderline；FP8/FP4 a2a 已在 air，撞 LSH-MoE / FP8-Flow-MoE / FP4 MoE 系列） |
| D1-I4 | PASS | 3.5 | 4.5 | 4 | 4 | 把 dispatch+combine 字节当作一阶可控变量后，一张简单的离散决策表（无需 RL/MPC）就能跨 workload 击败最优的 per-workload-static 配置，同时仅靠 router-margin 这一轻量信号就能阻止精度退化 | PASS（top 3；GPT 警告 story_fit 偏高 → 5 降到 4.5；router-margin 部分有把故事拉向 "byte-labeled adaptive drop" 的风险） |
| D1-I5 | PASS | 3.5 | 5 | 4 | 4 | a2a collective 等的是最慢 rank-pair；把 placement objective 从 sum-bytes 改成 max-bytes 会在同 VRAM 预算下选出 *实际不同* 的 placement，并在 Qwen3 真实 routing 分布上同时降低 max 和 total bytes | PASS（top 1；GPT 把 novelty 4 拉到 3.5；story_fit=5 保留，因为 "worst-peer" 是 finding 的结构形式） |
| D1-I6 | PASS | 3.5 | 4 | 4 | 3.5 | post-expert `‖g·o‖₂` 是比 pre-expert router score 更准确的 contribution 排序信号，因此在等 combine 字节预算下 combine-only 丢弃能严格 Pareto-击败 symmetric drop 的精度 | PASS（latency 维度结构性弱于 symmetric drop，已在 thesis 中重定位为 quality-asymmetry 贡献） |
| D1-I7 | PASS | 3 | 3.5 | 4.5 | 5 | user 的 K_eff hard truncation 失败是因为 uniform 截断，而 *per-token score-gap* K_eff 会保留 task-critical 的 high-K tokens，因此能在等 byte 节省下保持精度 | PASS（borderline；SMIDT/SERE 已覆盖 dynamic Top-K，差异主要在 byte-ROI framing） |
| D1-I8 | PASS（story_fit 危险低） | 2.5 | 2.5 | 4 | 4.5 | 在 prefill 长序列上，compute stages 足以吸收一次 a2a hop，跨层 (ℓ+1 dispatch) ∥ (ℓ combine+GEMM) 双层流水能减少 40%+ 的暴露 a2a 延迟，且不需要新的 byte-reduction operator | PASS-WEAK（能 work 但 *并不推进* "bytes 作为一等目标" 的主故事 —— 是 latency-hiding 而非 byte-reduction；同时被 Lancet/Comet 密度高度挤压） |
| D1-I9 | DUP（与 D1-I5 实质重叠） | 2.5 | 3.5 | 4 | 4 | 8×4090 PCIe 的 pairwise 带宽差异足够大以使 "homogeneous-cost optimal placement" ≠ "measured-cost optimal placement"，在 bandwidth-cheapest rank 上追加少量 replicas 能拿到 ≥10% worst-peer bytes 节省 | DUP（与 D1-I5 合并；作者本人已注 "may fold into D1-I5"） |
| D1-I10 | PASS（characterization-risk 标注） | 3 | 5 | 5 | 5 | MoE inference latency 在 single-node EP=8 上由 worst-peer dispatch+combine bytes 单变量主导，R² 比 L_recv-rows / GEMM-FLOPs 模型高 20+ pp，足以让它成为 MoE 系统设计的一阶 cost model | PASS-CONDITIONAL（GPT 把它挑为 "loser / empty rhetoric danger"；Claude 同意它必须与 D1-I4 或 D1-I5 耦合才能脱离 "只是看 MoE 不一样" 的标签） |

REJECT / DUP 的具体原因：
- **D1-I2 REJECT**：机制崩塌 —— 即便层 ℓ 与 ℓ+1 的 expert top-K Jaccard 高，layer ℓ+1 的 hidden state 依然是 fresh 的且必须按 ℓ+1 的 expert 分布跨 wire 抵达；唯一能真正省 bytes 的 reformulation 直接退化为 ExFlow。作者已自行 drop。
- **D1-I9 DUP**：与 D1-I5 共用 "worst-peer bytes as placement objective" 同一损失函数；唯一差异是引入 measured pairwise bandwidth 与 replication-as-operator，这两点应作为 D1-I5 的扩展项处理，不再独立计数。

#### Post-GPT top 2–3（按 novelty + story_fit 排序，PASS 项）

- **Top 1 — D1-I5 (8.5)**：worst-peer bottleneck-cut placement。GPT 同意 "not empty"，承认 incremental novelty；Claude 接受 novelty 降级但 story_fit 不让步。
  - 一句话赌注：a2a collective 等的是最慢 rank-pair；把 placement objective 从 sum-bytes 改成 max-bytes 会在同 VRAM 预算下选出 *实际不同* 的 placement，并同时降低 max 和 total bytes。
- **Top 2 — D1-I4 (8.0)**：wire-byte budget runtime controller (`B*` × `m*`)。GPT 指出最强结构风险（"closed-loop 不一定需要"），Claude 接受 story_fit 降级但保留为 top，因为它是把 D1-I10 的 metric 落到运行时的最直接 mechanism。
  - 一句话赌注：把 dispatch+combine 字节当作一阶可控变量后，一张简单的离散决策表就能跨 workload 击败最优的 per-workload-static 配置，且 router-margin 单信号足以阻止精度退化。
- **Top 3 — D1-I10 (8.0, conditional)**：a2a-bytes first-class performance model。GPT 选它当 "loser"。Claude 部分同意：作为 standalone 是 characterization 文章，必须 *显式* 与 D1-I4 或 D1-I5 打包才有 system contribution；满足这个条件后保留为 top 3，不满足则降为支撑性 ablation 章节。
  - 一句话赌注：MoE inference latency 在 single-node EP=8 上由 worst-peer dispatch+combine bytes 单变量主导，R² 比 L_recv-rows / GEMM-FLOPs 模型高 20+ pp。

#### Claude–GPT 评分分歧总结

| idea | Claude pre-GPT (novelty, story_fit) | GPT 挑战方向 | Claude post-GPT (novelty, story_fit) | 是否仍分歧 |
|------|-------------------------------------|--------------|--------------------------------------|------------|
| D1-I5 | 4, 5 | 双双偏高；story 可能塌成 "better placement using histograms" | 3.5, 5 | story_fit 仍有分歧（Claude 坚持 worst-peer 是 finding 的结构形式） |
| D1-I10 | 3.5, 5 | "empty-rhetoric danger"；标为 loser | 3, 5 | 是否 publishable 仍有分歧（Claude 认为 metric + 耦合 mechanism 可发表；GPT 倾向 "纯 characterization 不够"） |
| D1-I4 | 3.5, 5 | story_fit 偏高（router-margin 会拉走故事）；"beats every" 太强 | 3.5, 4.5 | "every"-claim 表述上有分歧（Claude 指明落到 ≥1 workload class，不是 every）；其余已对齐 |

#### QC 结论（维度 1）

- 总数：10
- PASS：8（D1-I1, I3, I4, I5, I6, I7, I8, I10）；其中 I8 story_fit=2.5 是 "能 work 但不推进主故事" 的边缘 PASS，I10 PASS-CONDITIONAL on coupling。
- REJECT：1（D1-I2，机制崩塌）
- DUP：1（D1-I9，与 D1-I5 合并）
- Top 2-3：D1-I5 → D1-I4 → D1-I10（conditional）
- Codex 本 QC 轮额外消耗：1 次（GPT 挑战 top 3 一并打包）。D1 累计：idea critique 1 + QC 1 = 2 substantive Codex calls。
- 维度结束，等待人工 gate。

---

## 维度 2 — Phase-Asymmetric MoE Serving

本维度的核心 user finding：prefill `L_recv ≈ 1000s` rows/rank，带宽受限，drop / token reduction 有正向效应；decode `L_recv ≈ K·T/EP ≈ 8`，launch 受限，drop 零收益（Phase-4 v1/v2 + GPU drop + K_eff truncation 都是 ±2% e2e）。`L_recv` 大约 100× 的不对称对应 ~100× 的 a2a 字节不对称。单一部署期 static policy（drop、replica budget、K、placement）不可避免地在一个 phase 上让步给另一个。→ 本维度所有 idea 把 **prefill 与 decode 视为同一模型实例内结构不同的优化区域**，区别于 DistServe/SplitWise 的 instance-level 拆分。

维度 2 自评分规则：Novelty potential、MLSys fit、Mechanism clarity、Feasibility on 8×RTX4090 EP=8、**Risk of being too close to DistServe/SplitWise**（分数越高越像 "DistServe-for-MoE"）、Overall (1–10)。

### D2-Idea-1：带运行时切换的 phase-bifurcated drop policy

- 维度：2 — Phase-Asymmetric MoE Serving
- 一句话主张：对同一模型跑两套不同的 drop policy —— prefill 用 `tail_weight @ r=0.3`，decode 关 drop —— 在第一个生成 token（或 chunked prefill 下的 chunk 边界）翻转。
- Yes/No 假设：phase 切换的 policy 在异构请求混合（LongBench long-prompt + GSM8K）上端到端持平 per-phase static optimum，精度 ±0.5pp，每 phase 比较好的那个 static 配置在 1% 内。
- 核心机制：一个布尔位在 prefill→decode 边界翻转，门控 drop kernel。
- 与 user finding 的关系：prefill 通信受限（97% 收益来自 drop），decode launch 受限（drop 反杠杆）。一套不可能同时对。
- 相对现有 drop 结果的新意：现有 drop 配置是部署期静态；这是 phase-awareness 的最小一步。
- 与 DistServe/SplitWise 的区别：留在同一 instance；不拆分 instance；不传 KV。
- 最小 pilot sketch，不执行：离线仿真 —— 重放 LongBench + GSM8K trace，在 decode 边界翻转标志位下计算 user 已测过的 e2e 延迟；对比两套 static。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：每个 phase 的延迟与其各自 static 最优在 1% 内；无精度回退。
- 证伪准则：切换引入的 transition overhead >2% e2e；或 KV 复用与 drop 的交互导致精度下降 >0.5pp。
- 主要 baseline：处处 `tail_weight @ r=0.3` 的 static（user 生产配置）。
- 主要风险：太显然；审稿人会说工程不算研究。主要作为 D2-I5/I9/I4 的对比 baseline 携带。
- 文献参考：
  - DistServe (Zhong et al., OSDI 2024), verified=YES, https://arxiv.org/abs/2401.09670, 相关性：跨 instance 的 phase 分离 —— "留在一个 instance" 版本的显式对比。
  - SplitWise (Patel et al., 2023), verified=YES, https://arxiv.org/abs/2311.18677, 相关性：dense LLM 的 phase splitting —— 直接对比。
- Claude 自评：
  - Novelty potential: 2.5
  - MLSys fit: 4
  - Mechanism clarity: 4.5
  - Feasibility on 8×RTX4090 EP=8: 4.5
  - Risk of being too close to DistServe/SplitWise: 4.5
  - Overall: 6.0
- GPT/Codex critique：本轮未入选 GPT review。
- Claude 回应 GPT：n/a
- 决策：保留（作为 D2 其它 idea 的必要对照 baseline）

### D2-Idea-2：Phase-specific expert replica 预算

- 维度：2 — Phase-Asymmetric MoE Serving
- 一句话主张：能在 prefill 上 pay off 的副本数（高，用来铺带宽）在 decode 上是浪费 VRAM（低流量，无争用）；按 phase 调整 per-layer replica budget，把 decode 释放的 VRAM 捐给该会话的 KV cache。
- Yes/No 假设：phase 化的 budget（例如 prefill `MOE_DROP_MIN_REPLICAS=512`，decode `=128`，释放的 VRAM 分给 per-session KV）在 LongBench long-prompt 上把 sustained decode batch 提升 ≥10%，prefill 延迟和精度不退化。
- 核心机制：在 prefill→decode 边界，runtime 在会话的 rank 集上释放底部 X% 的冷副本；per-rank allocator 把 slot 交给 KV-cache 池。若未来 decode token 路由到一个已释放的副本，回退到 remote rank（与 cold expert 同样路径）。
- 与 user finding 的关系：decode 没有 drop 收益是因为 `L_recv ≈ 8`；同一逻辑表明 decode 的副本分散在 wire 侧没买到东西，只占 VRAM。
- 相对现有 drop 结果的新意：drop 与 replica budget 一直被当作固定部署常量；本 idea 让它们 phase-dependent 且与 KV *内存可互换*。
- 与 DistServe/SplitWise 的区别：intra-instance 跨 phase 的内存重分配；DistServe 反而复制模型，没有 fungibility 税但冗余浪费 VRAM。
- 最小 pilot sketch，不执行：离线 —— 用 v6 log，按 decode 时的 per-rank per-layer 副本利用率分布找出冷副本候选，仿真释放的 VRAM；按已知 KV-page 尺寸估并发 decode 容量。无 GPU run。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：副本占的 per-rank VRAM 在 decode 时有 ≥15% 可释放，且实际路由到 decode token 的 miss rate <1%。
- 证伪准则：任何释放策略下 miss rate >5%；或释放的 VRAM 不与 KV-page 对齐、allocator 不做 compaction 就回收不了。
- 主要 baseline：部署期 static replica budget（生产：`MOE_DROP_MIN_REPLICAS=512`）。
- 主要风险：内存 fungibility —— 释放的 VRAM 可能落在非 KV 形状的 allocator slab 里，永远变不成可用 KV。是真实的工程风险，slot 对齐 allocator 可缓解但阶段 A 无法完全打消。
- 文献参考：
  - CRAFT: Cost-aware Expert Replica Allocation with Fine-Grained Layerwise Estimations (Zhao et al., 2026), verified=YES, https://arxiv.org/abs/2603.28768, 相关性：在固定内存预算下做 per-layer 复制；区别在轴（static layerwise vs runtime phase-wise）和目标（load balance vs phase-fungibility）。
  - SMIDT: High-Performance Inference Framework for MoE Models with Dynamic Top-K Routing (AAAI-26), verified=YES, https://ojs.aaai.org/index.php/AAAI/article/view/39403, 相关性：含 DPMoE memory-aware replication 组件并拆分 prefill/decode；D2 里最近的邻近。
- Claude 自评：
  - Novelty potential: 4
  - MLSys fit: 4.5
  - Mechanism clarity: 3.5（GPT 指出 mixed-batch 模糊后下调）
  - Feasibility on 8×RTX4090 EP=8: 3.5
  - Risk of being too close to DistServe/SplitWise: 3
  - Overall: 7.5
- GPT/Codex critique：
  - (a) 只是 DistServe-for-MoE 吗？否。
  - (b) 只是一个工程切换吗？部分 —— 需要把 memory fungibility / 争用论证放在中心。
  - (c) 机制：部分指定；mixed prefill/decode batch 的处理未说明。
  - 最强 objection：副本是全局共享资源；per-session decode 回收可能与并发 prefill 冲突；释放的 VRAM 在没有 allocator/residency 成本的情况下变不成可用 KV。
  - 锋利问题：在混合 workload 下，某一层同时服务 prefill 和 decode 流量时，谁的 phase 决定 replica budget？
  - 撞车风险：CRAFT (2603.28768)；SMIDT (AAAI-26 39403)。
- Claude 回应 GPT：我同意 *mixed-batch* 是核心未解决的机制问题，必须先回答。两个候选答案：(i) **per-rank, per-layer slot accounting** —— 副本 slot 在部署期划分为 "prefill-pool" 与 "elastic-pool"，后者只在该层当前服务的批纯是 decode 时才回收；或 (ii) **shadow-replica** —— 物理上从不释放，只标 decode-cold，把它的 VRAM 与 KV page 通过 paging policy 时分复用。(ii) 绕过 allocator-residency 成本。撞车：CRAFT 是静态、load-balance 导向；SMIDT 同时做 Top-K 和 DPMoE 但跨 instance 拆 phase，所以 *memory-fungibility* 角度是我的。真邻近，不重复。
- 决策：**revise**（承诺 shadow-replica 表述；显式回答 mixed-batch 归属；继续推进）

### D2-Idea-3：Phase-asymmetric top-K (K_eff)

- 维度：2 — Phase-Asymmetric MoE Serving
- 一句话主张：user 的 K_eff 硬截实验把 prefill 和 decode 合在一起做，结论是 "负杠杆"；拆开看，decode launch 受限、受益于小 K，prefill 字节受限、倾向 K=8。
- Yes/No 假设：在同一 Qwen3-30B-A3B 上 `K_decode = 4`（对比默认 8）把 GSM8K 上 decode 每步延迟降 ≥5%，精度 delta ≤0.3pp，且 `K_prefill = 8` 不动。
- 核心机制：per-phase top-K 计数受同一个 prefill/decode 标志位（与 D2-I1 同）门控，在 router 评分*之后*应用；当第 K 个 expert 落在 per-token routing softmax 的底 (K−K_eff) 时丢弃。
- 与 user finding 的关系：`L_recv≈8` 时 decode 延迟由 per-expert kernel launch 主导；K 减半就把 per token launch 减半，不动 byte 路径（在 decode 里本就微不足道）。
- 相对现有 drop 结果的新意：之前的 K_eff sweep 是 one-size-fits-all；这个 lever 在不同 phase 含义不同。
- 与 DistServe/SplitWise 的区别：改变*模型 per phase 的有效路由语义*，不动基础设施布局。
- 最小 pilot sketch，不执行：离线 —— 在 GSM8K 记录的 router 输出上，算 per-token 上 expert 5–8 vs 1–4 的 softmax-mass；估截到 top-4 的精度损失；估 launch 折半带来的延迟收益（用已知 kernel-launch 成本）。无 GPU run。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：top-4 在 ≥80% 的 decode token 上吃掉 ≥0.92 softmax mass；估计延迟下降 ≥5%。
- 证伪准则：在非平凡比例 token 上 top-4 softmax mass <0.85；或估的 launch 减少 <3% decode per-step time。
- 主要 baseline：两阶段都 `K=8`（Qwen3-30B-A3B 默认）。
- 主要风险：SMIDT-style "dynamic Top-K routing" 论文已存；差异化必须承诺一个*结构性* phase 不对称（`K_decode < K_prefill` 恒成立，不是数据驱动），并通过 launch-overhead 这一对 user 硬件/runtime 特有的论证展示 lever。
- 文献参考：
  - SMIDT: Dynamic Top-K Routing for MoE Inference (AAAI-26), verified=YES, https://ojs.aaai.org/index.php/AAAI/article/view/39403, 相关性：题目就叫 dynamic Top-K；区别在于是 workload-dynamic 而不是 phase-structural。
  - SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE (2026), verified=YES, https://arxiv.org/abs/2602.07616, 相关性：decode-only 路由变更；机制（re-route 到相似 expert）和目标不同。
  - Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining (2025), verified=YES, https://arxiv.org/abs/2511.02237, 相关性：decode 侧路由变更；目标（load-resident experts）不同于我们的 K-reduction。
- Claude 自评：
  - Novelty potential: 3.5（SMIDT 撞车后下调）
  - MLSys fit: 4
  - Mechanism clarity: 3.5（renormalization / 精度 delta 范围欠定）
  - Feasibility on 8×RTX4090 EP=8: 4
  - Risk of being too close to DistServe/SplitWise: 2
  - Overall: 7.0（从 7.5 下调）
- GPT/Codex critique：
  - (a) 只是 DistServe？否 —— 改路由，不改基础设施。
  - (b) 工程切换？否 —— 模型-系统语义 tradeoff。
  - (c) 机制：部分指定；renormalization、router 校准、精度 delta 定义欠规范。
  - 最强 objection：如果全局 K reduction 是负杠杆，主张 decode-only 在精度上安全需要比 launch overhead 更强的模型侧理由。
  - 锋利问题：decode logit 直接决定生成内容，为什么 rank 5–8 的 expert 在 decode 时是可丢的？
  - 撞车风险：SMIDT (AAAI-26 39403)；SERE (2602.07616)。
- Claude 回应 GPT：在模型侧理由上部分同意 —— 必须把 "对此特定模型 (Qwen3-30B-A3B, K=8)，decode token 上 expert 5–8 的 *gated* mass 低" 当作 pilot 前必须离线验证的经验主张，而不是靠泛泛的 launch-overhead 论证。renormalization 问题真实：K 从 8 降到 4 时存活的 expert weight 必须 re-normalize 到原始 mass（成熟做法），精度 delta 与未 re-normalize 的 top-8 baseline 对比。撞车：SMIDT 做的是 workload-dynamic K，不是 phase-structural K；SERE 是 re-route 而非丢第 K 个 expert；两者都不包含本 idea，但在它们之后做新 K-eff 工作的门槛更高。
- 决策：**revise**（承诺 phase-structural-K + 离线 softmax-mass 证据；显式区分 SMIDT/SERE）

### D2-Idea-4：Phase-asymmetric expert placement（双 plan）

- 维度：2 — Phase-Asymmetric MoE Serving
- 一句话主张：维持两套 placement plan —— "prefill plan" 最小化 worst-peer dispatch+combine 字节（D1-I5 目标），"decode plan" 最小化 per top-K 上的预期 remote-rank 命中数 —— 在 phase 边界通过预加载副本的 pointer-only re-tagging 切换。
- Yes/No 假设：双 plan placement（prefill-plan + decode-plan，通过共享物理副本在一份合并 VRAM 预算下都可行）在异构请求混合上，prefill 延迟比单 plan LBG-overlap=0.25 baseline 快 ≥5% *且* decode 延迟快 ≥3%，不超 expert-replica 给的 per-rank VRAM 预算。
- 核心机制：两个 plan 共享 disk/GPU 内存里的物理 expert 副本；只在 per-expert 所有权 tag 和 router 目的图上不同。phase 边界上一次原子指针交换改变 routing。
- 与 user finding 的关系：prefill 带宽受限（cost = worst-peer 字节）；decode launch 受限（cost ≈ remote-rank kernel launch 计数）。两套 cost 结构一般会选出不同布局。
- 相对现有 drop 结果的新意：现有 placement 是单目标静态；本 idea 引入 phase 依赖的 placement，且不需要内存翻倍。
- 与 DistServe/SplitWise 的区别：同一模型实例，同一组物理副本；只换*路由目的表*，不换权重。
- 最小 pilot sketch，不执行：离线 —— 给定 LongBench 记录的 router 输出，在同副本集上求两个 ILP（prefill-bytes、decode-hits）；检查 (i) 是否选出不同 placement，(ii) 合并是否在 VRAM 预算下可行。无 GPU run。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：两个求解器在 expert-to-rank 映射上 Jaccard 距离 ≥0.2，且合并 VRAM footprint ≤1.05× 最差单 plan footprint。
- 证伪准则：两个布局重合（Jaccard 距离 ≤0.1）或合并 VRAM footprint 超预算；任一都使贡献蒸发。
- 主要 baseline：单 plan LBG-overlap=0.25。
- 主要风险：两个目标可能选出几乎相同的 placement，贡献蒸发。还有：collective 是按 layer 而非按 hit launch 时，decode 的 "remote-rank hits" 未必是对的 cost —— 这是 fundamental 的测量问题。
- 文献参考：
  - Cluster Topology-Driven Placement of Experts Reduces Network Traffic in MoE Inference (2025), verified=YES, https://arxiv.org/abs/2508.09229, 相关性：基于 ILP 的 topology-aware placement；机器最近；区别是单目标、phase 不可知。
  - MoETuner (Go & Mahajan, 2025), verified=YES, https://arxiv.org/abs/2502.06643, 相关性：placement + routing 的 ILP；区别是 balance 目标、无 phase 不对称性。
- Claude 自评：
  - Novelty potential: 4
  - MLSys fit: 4.5
  - Mechanism clarity: 3（下调 —— 双 plan 切换力学欠规范）
  - Feasibility on 8×RTX4090 EP=8: 3
  - Risk of being too close to DistServe/SplitWise: 2
  - Overall: 7.0
- GPT/Codex critique：
  - (a) 只是 DistServe？否。
  - (b) 工程切换？部分 —— 两 plan 必须可证地选出不同布局才超出标志切换。
  - (c) 机制：部分指定 —— pointer-only re-tagging 假设副本已存；混合 phase batch 同时需要两个 plan 的情况未说明。
  - 最强 objection：decode 的 "remote-rank hits" 仍走 collective / a2a，bytes、sync、launch overhead 不可分。
  - 锋利问题：在同副本预算下，是怎样的具体路由分布会使 byte-minimizing placement 与 hit-minimizing placement 分歧？
  - 撞车风险：Cluster Topology-Driven Placement (2508.09229)；MoETuner (2502.06643)。
- Claude 回应 GPT：强同意贡献完全依赖于两个目标可证地选出可观测不同的布局。这现在就是证伪准则：两个 ILP 解之间的 Jaccard 距离 ≥0.2。关于 "collective 路径" 这一 objection：我承认在每层一次 a2a collective 时，per-hit launch 成本不能单独观测 —— 但 user 的 `ep_ht` runtime 上 decode dispatch 实际是按 active expert per point-to-point kernel 发的（已在现有 trace log 里验证），所以本栈上 per-hit 成本是可单独测量的。这是一个 hardware/runtime 条件性贡献，必须诚实限定。撞车：两篇都是单目标静态 placement；phase 不对称性是不同的。
- 决策：**revise**（加 Jaccard 距离证伪准则；把贡献限到 ep_ht-style point-to-point decode runtime；继续推进）

### D2-Idea-5：L_recv-gated continuous phase detection（per-layer 微切换）

- 维度：2 — Phase-Asymmetric MoE Serving
- 一句话主张：把 binary 的 prefill/decode phase 标签换成 per-layer、per-step 条件 `L_recv_ℓ > L*_ℓ`（带小的 per-layer 校准 break-even）；每层 MoE runtime 独立选 drop 开/关，使得 chunked-prefill、speculative decoding、异构混合 batch 自动继承 per-chunk phase 行为。
- Yes/No 假设：layer-local L_recv-gated drop 在每个 phase 上 e2e 持平 per-phase static optima（延迟 ±1% 内，精度 ±0.5pp 内），并在 binary phase 切换无能为力的 chunked-prefill workload 上额外拿到 ≥3% e2e。
- 核心机制：在每层 MoE 入口，runtime 读取刚成形的 dispatch metadata，算此步的 `L_recv_ℓ`，与一个 layer 特定的离线校准 `L*_ℓ`（per topology 校准一次）比较，选 drop kernel 分支。
- 与 user finding 的关系：break-even 是 *`L_recv` 的属性*，不是 phase 标签的；二值 phase tag 是同一底层条件的粗代理。
- 相对现有 drop 结果的新意：现有 phase-aware 工作（DuoServe-MoE, SMIDT）粒度在 request 级；这是 layer-local 的且 runtime 廉价。
- 与 DistServe/SplitWise 的区别：这是 *instance 内* 且 *请求内* 的；DistServe/SplitWise 跨 instance 操作。
- 最小 pilot sketch，不执行：离线 —— 重放 v6 LongBench + GSM8K trace，按 step 算 per-layer `L_recv_ℓ` 分布，仿真 layer-local 切换决策，对照 Tier-1 ablation 观测到的实际 break-even。无 GPU run。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：layer-local 切换在每个 phase 上对 per-phase static 都在 1% 内 *且* 在 `L_recv` 跨 chunk 边界横跨 `L*` 的 chunked-prefill 上额外拿到 ≥3% e2e。
- 证伪准则：`L*_ℓ` 在各层间变动 >2× 且没有稳定的 per-layer 校准；或 per-layer 切换开销超过 drop policy 的收益。
- 主要 baseline：binary phase 切换（D2-I1）。
- 主要风险：GPT 正确指出 `L*≈3271` 可能是硬件 / runtime artifact 而非结构性阈值。校准纪律是全部胜负所在。
- 文献参考：
  - DuoServe-MoE: Dual-Phase Expert Prefetch and Cache Scheduling (Zhang et al., 2025), verified=YES, https://arxiv.org/abs/2509.07379, 相关性：dual-phase MoE 调度；粒度（request-level prefetch vs layer-local drop）和 lever（prefetch vs drop）不同。
  - SMIDT: Dynamic Top-K Routing for MoE Inference (AAAI-26), verified=YES, https://ojs.aaai.org/index.php/AAAI/article/view/39403, 相关性：拆分 prefill/decode 性能；区别在 instance 级且用 Top-K 作为 lever。
  - Nexus: Proactive Intra-GPU Disaggregation of Prefill and Decode (2025), verified=YES, https://arxiv.org/abs/2507.06608, 相关性：intra-GPU phase 切分，但 dense LLM（非 MoE）；展示了 intra-instance phase-awareness 的趋势但 lever 不同。
- Claude 自评：
  - Novelty potential: 4.5
  - MLSys fit: 5
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4.5
  - Risk of being too close to DistServe/SplitWise: 1.5
  - Overall: 8.5
- GPT/Codex critique：
  - (a) 只是 DistServe？否。
  - (b) 工程切换？部分 —— 但用实测 L_recv 代替 phase 标签是真因果主张。
  - (c) 机制：是、可证伪；薄弱点在单一 L* 是否泛化。
  - 最强 objection：`L*≈3271` 可能是硬件 / runtime artifact，而非结构性阈值。
  - 锋利问题：当 routing skew、peer 不均衡、kernel fusion、overlap 都会移动 break-even 时，单看 `L_recv` 凭什么够？
  - 撞车风险：DuoServe-MoE (2509.07379)；SMIDT (AAAI-26 39403)。
- Claude 回应 GPT：强同意单一全局 `L*` 是薄弱点。我现在修订为 *per-layer* `L*_ℓ`，per (模型 × 拓扑 × runtime) 离线校准一次，校准流程本身作为贡献的一部分（不当作给定）。`L*≈3271` 这一 user finding 因此被定位为本硬件上 layer ~24/48 的校准常数，不是普适物理阈值。撞车：DuoServe-MoE 是基于 phase 做 prefetch（机制 = prefetch），本 idea 基于 L_recv 做 drop（机制 = drop，运行在 layer 内）；SMIDT 在 request 级跑 dynamic Top-K。都不以 per-layer-per-step 粒度用 L_recv 当门控变量。**与 GPT 的分歧**：我不认为校准问题杀掉这个 idea —— MLSys 里每个 break-even 调度器都依赖校准常数，贡献是抽象（`L_recv > L*_ℓ` 作为门控），不是常数。
- 决策：**revise**（承诺 per-layer 校准；把校准流程作为贡献文档化；显式区分 vs DuoServe-MoE 的 lever）

### D2-Idea-6：Decode fast-path kernel scheduler

- 维度：2 — Phase-Asymmetric MoE Serving
- 一句话主张：decode 时把通用 (drop-aware) MoE dispatch 路径替换为预烘焙的参数注入 fast path —— per token 发 `K·1` 最小 dispatch 任务，host 端零分支。
- Yes/No 假设：fast-path decode dispatcher 相对保留 drop-policy 分支的通用路径，per-step decode launch overhead 降 ≥10%，无精度影响。
- 核心机制：decode-only kernel-launch 图，把 K、drop=off、replica-map 烘进静态 CUDA graph；在 phase 边界切换。
- 与 user finding 的关系：decode launch 受限；per-launch overhead 是主导成本；prefill 路径的通用性是浪费。
- 相对现有 drop 结果的新意：user 没追过 decode 的 kernel-scheduler 专化；这是 phase 不对称性被命名后顺势的结构性 framing。
- 与 DistServe/SplitWise 的区别：留在一个 instance；一份模型；减 launch overhead，不分区。
- 最小 pilot sketch，不执行：离线 —— 从现有 decode trace 数 per-step host-side 分支和可 CUDA-graph-eligible 区域，估可消除延迟。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：仅靠消除 host-side 分支，per-decode-step 延迟降 ≥10%。
- 证伪准则：估的可消除 launch-time 成本 <5% per-step。
- 主要 baseline：通用 MoE 路径（当前生产）。
- 主要风险：滑向纯工程；必须与 D2-I3 / D2-I5 组合才有研究贡献。
- 文献参考：
  - DuoServe-MoE (2509.07379), verified=YES, 相关性：prefill prefetch 的 two-stream CUDA pipeline —— MoE 推理 kernel 调度的邻近工作。
- Claude 自评：
  - Novelty potential: 3.5
  - MLSys fit: 4.5
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4
  - Risk of being too close to DistServe/SplitWise: 2.5
  - Overall: 7.0
- GPT/Codex critique：本轮未入选 GPT review。
- Claude 回应 GPT：n/a
- 决策：保留（与 D2-I3 与 D2-I5 组合；单独优先级低）

### D2-Idea-7：基于 prefill 活动的跨 phase expert prefetch

- 维度：2 — Phase-Asymmetric MoE Serving
- 一句话主张：在 prefill→decode 边界，用 prefill 期间收集的 per-expert activation histogram 预测 decode 活跃 expert 集合，把它们 prefetch 到最快内存层。
- Yes/No 假设：prefill 推得的 expert-activity histogram 在第一个 decode step 预测 ≥80% 的 decode 活跃 expert；prefetch 预测集让 decode TTFT 降 ≥10%。
- 核心机制：prefill 期间维护 per-(rank, expert) activation 计数器；在边界，按计数取 top-K' 个 expert 钉/prefetch；未覆盖的 expert 回退标准加载。
- 与 user finding 的关系：decode `L_recv≈8` 意味着每步活跃 expert 极少；prefetch 对的 ~8 个 expert 把 load 延迟移出 decode critical path。
- 相对现有 drop 结果的新意：与 drop 正交；用 phase 边界作为信息传递点。
- 与 DistServe/SplitWise 的区别：信息在 *同一 request* 跨 phase 传递，而不是跨 instance 分区。
- 最小 pilot sketch，不执行：离线 —— 从 v6 trace 数 prefill 期间 per-expert 激活与 decode 期间 per-expert 激活的对比；测预测召回。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：prefill→decode expert-activation 重叠召回 ≥0.8。
- 证伪准则：召回 ≤0.5（decode expert 集本质独立于 prefill 集），prefetch 无用。
- 主要 baseline：无跨 phase 信息传递。
- 主要风险：**DuoServe-MoE（2025 年 9 月）已经实现了带 layer-level predictor 的 dual-phase expert prefetch+caching**。机制非常接近。挽救需要可辩护的差异化角度（如 per-rank co-location vs cache，或不同的 predictor 变量），阶段 A 内难以建立。
- 文献参考：
  - DuoServe-MoE: Dual-Phase Expert Prefetch and Cache Scheduling (Zhang et al., 2025), verified=YES, https://arxiv.org/abs/2509.07379, 相关性：机制非常接近（dual-phase prefetch+cache for MoE，decode 端 predictor）。
  - Fate: Accurate Expert Predictions in MoE Inference via Cross-Layer Gate (2025), verified=YES, https://arxiv.org/abs/2502.12224, 相关性：从上游 gate 推 expert 激活预测。
- Claude 自评：
  - Novelty potential: 2.5（DuoServe 撞车后下调）
  - MLSys fit: 4
  - Mechanism clarity: 3.5
  - Feasibility on 8×RTX4090 EP=8: 3.5
  - Risk of being too close to DistServe/SplitWise: 2.5
  - Overall: 5.5
- GPT/Codex critique：本轮未入选 GPT review（未进 top 5）。
- Claude 回应 GPT：n/a
- 决策：**drop** —— DuoServe-MoE 已发表机制就是 prefill 信息 → decode 端的 dual-phase MoE expert prefetch；本 idea 塌入其中，且我没有阶段 A 内能站得住的差异化角度对抗真实审稿人对比。

### D2-Idea-8：Decode 阶段 expert 重共置

- 维度：2 — Phase-Asymmetric MoE Serving
- 一句话主张：decode 时流量足够轻，系统可以承担把少数活跃 expert 迁到单一 rank（或小 island）上为该 session 服务，使 a2a 塌成单 rank gather，完全消除 dispatch a2a。
- Yes/No 假设：把每活跃 session 的 top-32 最活跃 expert 迁到一个指定 "decode-host" rank，decode dispatch a2a 调用降 ≥80%，LongBench long-decode 上精度无回退。
- 核心机制：tracker 数 per-expert decode 激活；当某 expert 超过 migrate-cost 阈值，runtime 在 decode-host rank 上一次性创建副本并重写 routing 表偏好该 rank。
- 与 user finding 的关系：`L_recv≈8` 时 a2a 协调开销主导 wire bytes；完全消除 a2a 是 decode 上的最大杠杆操作。
- 相对现有 drop 结果的新意：不丢任何东西；把杠杆从 "更少字节" 移到 "更少目的端"。
- 与 DistServe/SplitWise 的区别：同 instance，副本创建而非 instance 分区；模型本身没复制，只有 ~32 个 expert。
- 最小 pilot sketch，不执行：离线 —— 数典型 LongBench 回复在所有 decode token 上活跃的 distinct expert；检查复制后是否能装入 per-rank VRAM。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：per-session 的 distinct decode-active expert ≤32（或某个可管理数），VRAM 余量足够。
- 证伪准则：per-session distinct expert >50（装不下）或 migration 成本按典型 decode 长度摊销下来净负。
- 主要 baseline：标准 EP=8 a2a-based decode。
- 主要风险：decode-host rank 的副本创建延迟可能抹掉 decode 延迟收益；阈值逻辑必须紧。
- 文献参考：
  - MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache (Xue et al., 2024), verified=YES, https://arxiv.org/abs/2401.14361, 相关性：sequence-level activation 稀疏性用于 offload/cache；推理邻近、operator 不同。
- Claude 自评：
  - Novelty potential: 4
  - MLSys fit: 4
  - Mechanism clarity: 3
  - Feasibility on 8×RTX4090 EP=8: 3
  - Risk of being too close to DistServe/SplitWise: 2.5
  - Overall: 7.0
- GPT/Codex critique：本轮未入选 GPT review。
- Claude 回应 GPT：n/a
- 决策：保留（与 D2-I9 区别清楚 —— I8 迁移活跃 expert，I9 释放不活跃的）

### D2-Idea-9：Phase-transition 内存重平衡（shadow-replica）

- 维度：2 — Phase-Asymmetric MoE Serving
- 一句话主张：把 prefill→decode 边界用作内存重平衡事件；不是释放 decode-冷的 expert 副本，而是 *shadow 它们* —— 把其物理页标为 elastic，与 per-session KV cache 时分复用，使未来的 routing miss 能以低成本 re-page。
- Yes/No 假设：在 transition 处 shadow-replica re-tagging 把 ≥15% 的 expert-replica VRAM 回收为 KV-eligible 页，在代表性硬件占用上把并发 decode session 提升 ≥1，触发 ≤2% 的副本 miss-recall 事件，实测 re-page 成本 <500 µs。
- 核心机制：prefill→decode 边界把激活计数底 X% 的低活跃副本标为 "shadow"；其物理页移到 KV allocator 可申领的软分页区域；shadow expert 的 routing miss 触发立即 re-page（若另一 rank 有 resident 副本则同步读，否则从 CPU 拉）。
- 与 user finding 的关系：drop 的 97% 收益来自 prefill 减去 dispatch+combine 字节；同一逻辑说 decode 不需要大多数副本驻留。transition 是天然的提交点。
- 相对现有 drop 结果的新意：把 drop 的 framing 从 "跳过传输" 扩展到 "跨 phase 边界释放内存"。
- 与 DistServe/SplitWise 的区别：同 request 同 instance 跨 phase 动态内存预算；DistServe 在分离 instance 同时持有两阶段，冗余浪费 VRAM。
- 最小 pilot sketch，不执行：离线 —— 按请求类型算 per-expert prefill 激活计数；挑 prefill 期间计数 = 0 的候选；估释放的 VRAM；从独立 decode trace 估 miss 概率。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：VRAM 回收 ≥15%，预测 miss rate ≤2%；离线 benchmark 测得的 re-page 成本 ≤500 µs。
- 证伪准则：miss rate >5%；或 re-page 成本 >2 ms（会抹掉任何 decode 延迟收益）。
- 主要 baseline：static 副本驻留（无 phase 重平衡）。
- 主要风险：GPT 的 "future use 不可知" —— shadow-replica 方案靠保留 remote-rank fallback 缓解但不能完全消除最坏 miss 成本。
- 文献参考：
  - MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache (Xue et al., 2024), verified=YES, https://arxiv.org/abs/2401.14361, 相关性：带驱逐的 sparsity-aware caching；operator 最近（cache+evict），但是 single-phase。
  - DuoServe-MoE (Zhang et al., 2025), verified=YES, https://arxiv.org/abs/2509.07379, 相关性：dual-phase expert cache 调度；lever 不同（prefetch vs free + shadow）。
- Claude 自评：
  - Novelty potential: 4
  - MLSys fit: 4.5
  - Mechanism clarity: 4（改写为 shadow-replica 后上调）
  - Feasibility on 8×RTX4090 EP=8: 3.5
  - Risk of being too close to DistServe/SplitWise: 2.5
  - Overall: 7.5
- GPT/Codex critique：
  - (a) 只是 DistServe？否。
  - (b) 工程切换？部分 —— 除非 trajectory 可预测性表达精确，否则释放冷副本就是 cache/offload policy。
  - (c) 机制：部分指定 —— "不太可能激活" 与 miss 处理欠规范。
  - 最强 objection：transition 时未来的 decode expert 使用不可知；冷 expert miss 要么靠 reload 损延迟，要么改模型行为。
  - 锋利问题：若一个已释放副本之后被路由，token 是远程服务、stall 等 reload、还是被近似？
  - 撞车风险：MoE-Infinity (2401.14361)；DuoServe-MoE (2509.07379)。
- Claude 回应 GPT：强同意。把 operator 从 "释放" 改为 **shadow-replica**，使 miss 可恢复：物理副本不被删除，只改驻留状态，该 expert 的 routing 表指向 (i) in-rank shadow 区域，(ii) 远程 rank resident 副本，(iii) CPU 侧备份。miss 处理变成分层 fallback，不是模型近似。撞车：MoE-Infinity 是推理时 *single-phase* sparsity-aware caching；DuoServe-MoE 基于 per-decode-layer 预测做 prefetch。两者都不把 phase 边界用作 memory budget 的提交点且带 shadow 语义。邻近真实，机制不同。
- 决策：**revise**（承诺 shadow-replica + 分层 miss-fallback 表述；继续推进）

### D2-Idea-10：Phase-asymmetric 重叠目标（通信重叠 vs 注意力重叠）

- 维度：2 — Phase-Asymmetric MoE Serving
- 一句话主张：prefill（通信受限）时把 expert GEMM 与下一次 dispatch a2a 做软件重叠；decode（注意力侧 compute-bound）时把 expert GEMM 与下一次 attention 预投影重叠。
- Yes/No 假设：phase 条件化的重叠 policy 在 LongBench long-prompt + long-decode 上相对 user 的 serial baseline 隐藏 ≥30% prefill a2a 延迟 *且* ≥10% decode attention 延迟。
- 核心机制：phase tag 选哪个下游 stage 作为重叠目标；scheduler 用已有 stream 并发发射所选对。
- 与 user finding 的关系：prefill 瓶颈是通信字节；decode 瓶颈是 per-step attention compute（per-step a2a 本就小）。不同 phase 适合不同目标。
- 相对现有 drop 结果的新意：drop 减字节量；本 idea 在不同 phase 减不同重叠目标的暴露时间。
- 与 DistServe/SplitWise 的区别：无 instance 分区；只在 scheduler 级改重叠目标。
- 最小 pilot sketch，不执行：离线 —— 用现有 per-stage 计时 log，仿真 serial vs 每 phase 各自重叠对的 Gantt 图；报告估的加速。
- 若以后做 pilot 所需 EP 设置：EP=8
- 期望正向信号：prefill 的 dispatch+combine 部分仿真加速 ≥1.3×，decode 的 attention 部分 ≥1.1×。
- 证伪准则：任一仿真加速 ≤1.05×。
- 主要 baseline：serial MoE-layer 执行（生产 schedule）。
- 主要风险：与已有 MoE comm/comp 重叠工作（Lancet, Comet）重叠；decode 侧不那么成熟但与 dense LLM serving 中的 attention-overlap 模式相似（Sarathi）。
- 文献参考：
  - Lancet (Jiang et al., MLSys 2024), verified=YES, https://arxiv.org/abs/2404.19429, 相关性：训练侧 whole-graph overlap。
  - Comet (Zhang et al., 2025), verified=YES, https://arxiv.org/abs/2502.19811, 相关性：细粒度 MoE comm-comp overlap。
- Claude 自评：
  - Novelty potential: 3.5
  - MLSys fit: 4.5
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4
  - Risk of being too close to DistServe/SplitWise: 2.5
  - Overall: 7.0
- GPT/Codex critique：本轮未入选 GPT review。
- Claude 回应 GPT：n/a
- 决策：保留（低优先级；作为组合 lever 有价值）

---

### 维度 2 Summary

- 生成 idea 数：去重后 10 个（D2-I1 … D2-I10）。早期草稿合并："Phase-specific replica budget" + "Decode-phase memory reclaim" → D2-I2（budget）+ D2-I9（带 shadow-replica 的 transition event）两个独立 lever。
- 提交 GPT 的 top 5（一次 codex 调用打包）：D2-I5、D2-I2、D2-I3、D2-I9、D2-I4。
- 保留（无重大改动）：D2-I1、D2-I6、D2-I8、D2-I10。
- 修订（按 GPT 显式改动后继续推进）：
  - D2-I2：承诺 **shadow-replica** 来回答 mixed-phase batch 归属；显式回应 allocator-residency 成本。
  - D2-I3：承诺 **phase-structural** `K_decode < K_prefill` + 离线 softmax-mass 证据；显式区分 SMIDT/SERE。
  - D2-I4：加 **Jaccard 距离 ≥0.2** 作为两 ILP 解的证伪准则；把贡献限定到 point-to-point decode runtime（ep_ht-style）。
  - D2-I5：把 per-layer `L*_ℓ` 校准作为贡献的一部分；显式区分 DuoServe-MoE（prefetch）和 SMIDT（Top-K）。
  - D2-I9：operator 从 "释放副本" 切换到 **shadow-replica + 分层 miss-fallback**。
- 淘汰：D2-I7（cross-phase expert prefetch）—— DuoServe-MoE (Zhang et al., 2025, arXiv:2509.07379) 实现了同样的 dual-phase MoE expert prefetch+caching 机制 + layer-level decode predictor。阶段 A 内无可在真实审稿人对比下站住的差异化角度。
- GPT 主要批判：
  1. 多个 idea 是 "包装成研究的 policy switch"，除非 *fungibility* 或 *causal-feature* 论证成为主干 —— 适用于 D2-I2、D2-I4、D2-I9。
  2. 混合 prefill/decode batch 会打破朴素的 per-phase policy；D2-I2 和 D2-I4 必须说清楚同层同时跑两阶段时怎么处理。
  3. `L*≈3271` break-even 可能是硬件/runtime artifact，而非结构性阈值（D2-I5）。
  4. Phase-asymmetric K（D2-I3）直接撞 SMIDT (AAAI-26 dynamic Top-K) 和 SERE —— 必须证明是 phase-*结构性* 而非数据驱动的 K 选择。
  5. transition 处未来 decode 行为不可知；D2-I9 必须说清 miss 语义，不能光说 "释放副本"。
- Claude–GPT 主要分歧：
  - D2-I5：GPT 把 "L* 可能是 artifact" 当致命；Claude 视为校准问题并主张抽象（每层 `L_recv > L*_ℓ`）即使常数硬件相关也可发表。
  - D2-I3：GPT 把 "experts 5–8 是可丢的吗？" 提为模型侧关切；Claude 承诺以离线 softmax-mass 实证作答，而不放弃。
  - D2-I9：GPT 把 "future use 不可知" 当 kill question；Claude 用 **shadow-replica** 语义（分层 fallback，不做模型近似）重写并不同意致命。
- Codex 用量估计：本维度 1 次 substantive codex 调用（5 idea 打包）。D1+D2 累计 = 3 次 substantive + 1 次连通性 ping；低于 15 次告警阈值。
- 下一推荐维度（不要执行）：**维度 3 — Workload-Adaptive MoE Serving**（brief `BRIEF_WORKLOAD_ADAPTIVE`）。理由：D1 = "通信是 lever"，D2 = "prefill ≠ decode"，D3 = "long-prompt-short-decode ≠ short-prompt-long-decode at request granularity"。D2 和 D3 互补：D2 按 phase 切，D3 按请求形状切；许多 D2 机制（replica budget、K-eff、placement）当 "per phase" 提升为 "per request class" 时即变成 D3 的动作。**等人工 gate。**

---

### 维度 2 — 质检结果（QC pass, 2026-05-29）

QC 范围：对 D2 的 10 个 idea 重新做 (1) 硬性门 —— 可证伪、具体性、与 finding 关系、重复性；(2) 修订后的质量评分 —— novelty / story_fit（severity：能 work 但 *不* 推进 phase-aware policy 主故事必须低分，被 engineering / scheduling / characterization 包裹的也算）/ pilot_feasibility / pilot_cost（反向）；(3) 每个 idea 一句话赌注；(4) 对 top 3 调 GPT 挑战，重点问 "是否只是 DistServe-for-MoE / 只是 engineering policy switch / bet 是否站得住"。不做撞车结论，不进入阶段 B。

GPT 挑战 trace：`docs/aris/traces/dimension2_qc_review.md`。下表中分数为 *GPT 之后*，与 GPT 的分歧记录在 trace 里。

#### QC 表

| idea_id | 硬性检查 | novelty | story_fit | feasibility | cost | 一句话赌注 | 标记 |
|---------|----------|---------|-----------|-------------|------|------------|------|
| D2-I1 | PASS | 2 | 4 | 4.5 | 5 | prefill→decode 边界翻转 drop 标志位的 transition overhead 不会超过 2% e2e，KV 复用与 drop 的交互不会造成 >0.5pp 精度退化，因此 phase 切换在每 phase 端到端持平相应 static optimum | PASS-BASELINE（作者本人定位为对比 baseline；novelty 低但完整可证伪） |
| D2-I2 | PASS | 3 | 3.5 | 3.5 | 4 | shadow-replica 把 decode-冷副本 VRAM 与 KV 时分复用，可在 ≤2% miss rate 下回收 ≥15% 副本 VRAM，且 freed VRAM 在 page 对齐下能被 KV allocator 申领增加 ≥1 并发 decode session | PASS-WEAK（GPT 挑为 loser；最长的脆弱主张链 "冷副本→可回收 VRAM→allocator 可用→额外 decode session"，每一环都能技术上成立但意义打折；shadow-replica 本质偏 cache residency 语言；novelty/story_fit 双双下调） |
| D2-I3 | PASS | 2.5 | 4 | 4 | 5 | Qwen3-30B-A3B 在 decode token 上 rank 5-8 expert 的 gated softmax mass 低到可丢，因此 K_decode=4 + renormalization 在 ≤0.3pp 精度代价下省 ≥5% decode 延迟 | PASS-NARROW（GPT 警告 "narrow"、可能塌成 "decode approximation works" 而非 phase-aware policy；必须把 launch-bound 诊断显式与 K 绑定才能脱离 "two static configs" 形态） |
| D2-I4 | PASS | 3.5 | 4 | 3 | 3 | 同副本预算下，prefill bytes 目标与 decode hits 目标会选出 Jaccard 距离 ≥0.2 的不同 placement，且合并 VRAM 在预算内可行，从而相对单 plan baseline 在两 phase 都拿到加速 | PASS（cross-dimension 风险：prefill plan 复用 D1-I5 目标 —— 必须显式说清 D2-I4 = D1-I5 + decode-plan + 切换机制，否则会被读成 D1-I5 的 phase 化扩展） |
| D2-I5 | PASS | 3.5 | 5 | 4.5 | 5 | user's break-even L*≈3271 不是全局常数而是一组可一次性离线校准的 per-layer L*_ℓ；用 L_recv_ℓ > L*_ℓ 替代 phase 标签作门控，会让 chunked-prefill 等混合形态自动拿到 binary 切换拿不到的额外 ≥3% e2e | PASS（top 1；GPT 确认 story_fit 真实、结构上安全、与 DistServe-for-MoE 可区分；novelty 4→3.5） |
| D2-I6 | PASS（story_fit 低） | 2.5 | 3 | 4 | 4.5 | decode path 的 host-side 分支与 dispatch 通用性是延迟主要可消除部分，烘进 CUDA graph 能省 ≥10% per-step 延迟且零精度影响 | PASS-WEAK（kernel specialization，本质是工程优化而非 phase-aware policy 创新；与 D1-I8 性质类似，能 work 但不推进主故事） |
| D2-I7 | REJECT — 已被 DuoServe-MoE 覆盖 | — | — | — | — | prefill 期 expert-activity histogram 能预测 decode 活跃 expert 集合并 prefetch 命中以减 TTFT | REJECT（作者本人已 drop；DuoServe-MoE arXiv:2509.07379 已发表同机制 dual-phase prefetch+caching + layer-level predictor，无可辩护的差异化角度） |
| D2-I8 | PASS | 3.5 | 4 | 3 | 3.5 | 典型 LongBench 回复在所有 decode token 上的 distinct 活跃 expert 数 ≤32，可装入单 rank VRAM，因此把它们集中迁到 decode-host rank 后 decode dispatch a2a 调用降 ≥80% 且精度无回退 | PASS（与 I9 区别清楚 —— I8 *迁移* 活跃 expert 而非 *释放* 不活跃的；与 MoE-Infinity 的 cache/offload 不同 operator） |
| D2-I9 | DUP（与 D2-I2 实质重叠） | 3.5 | 4.5 | 3.5 | 4 | decode-冷副本 shadow + 分层 miss-fallback 能在 ≤2% miss rate 与 <500 µs re-page 成本下回收 ≥15% VRAM 给 KV，无模型近似 | DUP（与 D2-I2 共用 shadow-replica + 给 KV 的同一 mechanism；I2 是 "phase-specific budget" framing、I9 是 "transition event" framing，但 operator 和动作时机完全相同；应作为 D2-I2 的扩展项处理） |
| D2-I10 | PASS（story_fit 低） | 3 | 3 | 4 | 4.5 | prefill 与 decode 的瓶颈对应不同 overlap 对（dispatch a2a vs attention 预投影），phase 条件化重叠 policy 能同时拿到 ≥30% / ≥10% 的延迟隐藏 | PASS-WEAK（scheduling 优化而非 phase-aware policy；撞 Comet/Lancet 训练侧重叠 + dense LLM Sarathi 解码侧重叠） |

REJECT / DUP 的具体原因：
- **D2-I7 REJECT**：DuoServe-MoE (arXiv:2509.07379) 已发表完全同机制 —— prefill 期 expert 活动驱动 decode prefetch，带 layer-level predictor。作者已自行 drop；阶段 A 内没有能在真实审稿人对比下站住的差异化角度。
- **D2-I9 DUP**：与 D2-I2 共用同一 operator（shadow-replica 把 decode-冷副本 VRAM 与 KV 时分复用）和同一动作时机（prefill→decode 边界）。D2-I2 的 framing 是 "phase-specific budget 变量"；D2-I9 的 framing 是 "transition memory rebalancing event"；但两者的机制本体、数据需求、证伪准则、风险类别全部同源。应合并为 D2-I2 的扩展项处理，不再独立计数。

#### Post-GPT top 2–3（按 novelty + story_fit 排序，PASS 项）

GPT 挑战之后 D2-I2 跌出 top 3。重新排序后：

- **Top 1 — D2-I5 (8.5)**：L_recv-gated continuous phase detection。GPT 确认 "not empty"、story_fit 真实、结构上安全、与 DistServe-for-MoE 可区分。Claude 接受 novelty 4→3.5，但 story_fit=5 GPT 也认可保留。
  - 一句话赌注：user's break-even L*≈3271 不是全局常数而是一组可一次性离线校准的 per-layer L*_ℓ；用 L_recv_ℓ > L*_ℓ 替代 phase 标签作门控会让 chunked-prefill 等混合形态自动拿到 ≥3% 额外 e2e。
- **Top 2 — D2-I4 (7.5)**：phase-asymmetric bi-modal placement。未经 GPT 单独挑战；保持 Claude 原评。注意 cross-dimension 组合性风险：prefill 端目标直接复用 D1-I5，需要显式把 D2-I4 定位为 "D1-I5 + decode-plan + Jaccard ≥0.2 切换证据"，否则会被读成 D1-I5 的 phase 化扩展。
  - 一句话赌注：同副本预算下 prefill bytes 目标与 decode hits 目标会选出 Jaccard 距离 ≥0.2 的不同 placement，合并 VRAM 在预算内可行 —— 这意味着两 phase 上都能相对单 plan baseline 拿到加速。
- **Top 3 — D2-I8 (7.5)**：decode-phase expert re-co-location。未经 GPT 挑战；mechanism clarity 偏低（迁移触发阈值与平摊条件 underspecified），但 operator 与文献清晰区分（既不是 prefetch、也不是 cache eviction，是 active-expert 迁移到 host rank）。
  - 一句话赌注：典型 LongBench 回复在所有 decode token 上的 distinct 活跃 expert ≤32，可装入单 rank VRAM；把它们迁到 decode-host rank 后 decode dispatch a2a 调用降 ≥80% 且精度无回退。

候选 #4 — D2-I3 (6.5)：phase-asymmetric K_eff。GPT 警告 "narrow" 与 "two static configs" 风险，但 yes/no hypothesis 仍最干净、pilot 最便宜；如果 top 3 中某个无法 pilot，I3 是替补。

#### Claude–GPT 评分分歧总结

| idea | Claude pre-GPT (novelty, story_fit) | GPT 挑战方向 | Claude post-GPT (novelty, story_fit) | 是否仍分歧 |
|------|-------------------------------------|--------------|--------------------------------------|------------|
| D2-I5 | 4, 5 | novelty 略高；story 真实；安全 | 3.5, 5 | 基本对齐；novelty 让步、story_fit 双方认可 |
| D2-I2 | 3.5, 4.5 | novelty 太高（CRAFT/DPMoE/MoE-Infinity）；story_fit 太高（本质是 memory management，phase 仅作 justification）；claim 链最脆弱；挑为 loser | 3, 3.5 | 大体认输；分歧点：Claude 主张如果显式承诺 *online arbitration* 仍可挽救成 PASS，GPT 倾向直接降级 |
| D2-I3 | 3, 4.5 | novelty 太高（SMIDT/SERE/Opportunistic Expert Activation）；story_fit narrow，可能塌成 "decode approximation works" | 2.5, 4 | 部分对齐；分歧点：Claude 认为如果把 launch-bound 诊断显式与 K 绑定就脱离 "two static configs" 形态、story_fit 可保留 4，GPT 倾向更低 |

#### QC 结论（维度 2）

- 总数：10
- PASS：8（D2-I1, I2, I3, I4, I5, I6, I8, I10）；其中 I2 因 GPT 挑战降级 PASS-WEAK（loser），I6 与 I10 story_fit 偏低（scheduling/engineering 而非 phase-aware policy），I1 定位为 baseline。
- REJECT：1（D2-I7，DuoServe-MoE 覆盖；作者已自行 drop）
- DUP：1（D2-I9，与 D2-I2 共用 shadow-replica + 给 KV 同一 mechanism）
- Top 2-3：D2-I5 → D2-I4 → D2-I8（D2-I3 候补 #4；D2-I2 被 GPT 挑战后跌出 top 3）
- Codex 本 QC 轮额外消耗：1 次（GPT 挑战 top 3 一并打包，high reasoning）。D2 累计：idea critique 1 + QC 1 = 2 substantive Codex calls。
- 维度结束，等待人工 gate。

---

## Dimension 3 — Workload-Adaptive MoE Scheduling

本维度的核心 user finding：long-prompt+short-decode（RAG / 长文档 QA / 检索）在 LongBench 上 drop 显著正向（prefill +23% / e2e +12% / 精度零损失）；short-prompt+long-decode（GSM8K / reasoning / agent）结构性零收益（decode `L_recv≈8` 远低于 `L*≈3271`）。生产 serving 系统的流量混合两类，**任何单一静态 MoE 配置必然在一类 workload 上踩坑**。vLLM / SGLang / TensorRT-LLM 把 expert placement、router top-K、replica 数都当作部署期常量；continuous batching 只调度 token 不调度 MoE 策略。→ 本维度所有 idea 在 **运行时按 request 形状 / batch 形状 / 观测到的 L_recv** 动态选 MoE 策略；决策单位是 *request shape* 或 *microbatch*，区别于 D2 的 *within-request phase*，也区别于 DistServe/SplitWise 的 *cross-instance partition*。

维度 3 自评分规则：Novelty potential、MLSys fit、Mechanism clarity、Feasibility on 8×RTX4090 EP=8、**Risk of being just heuristic scheduling**（分数越高越像 if-else）、Overall (1–10)。

### D3-Idea-1: Request-class router with offline-binned MoE policy lookup

- Dimension: 3 — Workload-Adaptive MoE Scheduling
- One-sentence thesis: 在请求入口按 (prompt length, max_tokens, sampling params) 算 predicted L_recv 轨迹，把 request 分到 K 个离线训好的 class（每 class 有预定的 (drop policy, drop rate, K_eff)），无需在线学习。
- Yes/No hypothesis: 离线训的 predicted-L_recv 分类器在 mixed LongBench+GSM8K stream 上 ≥85% 的请求被正确划入应得 policy bracket，整体 e2e 比 single-static 改善 ≥3%。
- Runtime input signals: prompt length, max_tokens, sampling params (temperature, top-p), 历史回复模式。
- Runtime decision variables: 给该 request 分配一个 class id ∈ {1..K}，对应一组预烘焙 MoE 策略。
- Core mechanism: 离线 regressor `ŷ = g(prompt_len, max_tokens, sampling)` 输出 predicted-L_recv envelope；按 quantile 分 bin；每 bin 在 v6 LongBench grid 上已确定最优 (drop policy, r)。
- Why this follows from the user finding: long-prompt 与 short-prompt 在 L_recv 上差 ~13×，predictable from prompt-len 单变量。
- What is new beyond current drop result: drop 配置当前是部署期常量；本 idea 让 *入口决策* 与 *MoE 策略* 一体化。
- Minimal pilot sketch, no execution: 离线 —— 用 v6 LongBench + GSM8K 已有日志构造 (prompt_len, observed L_recv envelope, optimal-policy-id) 三元组；fit shallow regressor + bin；hold-out 上测正确率。无 GPU。
- Required EP setting if later piloted: EP=8
- Expected positive signal: 4-class binning 下 ≥85% requests 正确分类。
- Falsification criterion: 单变量 (prompt_len) 单独 R²≤0.4 to predicted L_recv envelope；或 hold-out 分类准确率 ≤0.7。
- Main baseline: single static (`tail_weight @ r=0.3`, MOE_DROP_MIN_REPLICAS=512)。
- Main risk: pure heuristic if-else 包装；novelty 完全靠 MoE-specific 的 class 含义。
- Literature references:
  - LYNX: Efficient MoE Inference Through Dynamic, Workload-Adaptive Serving (2024), verified=YES (题目+作者确认；abstract 详尽机制不可见), https://arxiv.org/abs/2411.08982, 相关性：题目最接近；实际机制是 batch 内 token-expert remapping (AffinityBinning) 而非 per-request shape binning，但需要仔细对比。
- Claude self-score:
  - Novelty potential: 3
  - MLSys fit: 4
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4.5
  - Risk of being just heuristic scheduling: 4
  - Overall: 6.5
- GPT/Codex critique: 本轮未入选 GPT review。
- Claude response to GPT: n/a
- Decision: keep (作为 D3-I2 的 baseline)

### D3-Idea-2: Online L_recv early-layer feedback with mid-request policy switch

- Dimension: 3 — Workload-Adaptive MoE Scheduling
- One-sentence thesis: 用 request 前 k 个 MoE 层（k ∈ {2, 3}）的实测 L_recv 作为反馈信号预测后续层的 L_recv envelope；若实测穿过 break-even 桶边界，就为该 request 余下的层切换 drop policy；无需离线预测器。
- Yes/No hypothesis: online switching 在 mixed LongBench+GSM8K stream 上 e2e 比 best-per-workload-static 改善 ≥3%，无 per-request ≥0.5pp 精度回退。
- Runtime input signals: 层 0..k 的实测 per-layer L_recv。
- Runtime decision variables: 层 k+1..L 的 drop 开关 + drop rate。
- Core mechanism: 在 layer k 末，runtime 比较 `mean(L_recv_0..k)` 与离线校准的 per-layer break-even band；若稳定超出某一侧、置信度（用 k 步方差测）足够高，置策略；否则保守不切。
- Why this follows from the user finding: 全 request 平均 L_recv 是预测良好的，可以由前几层 sample 估计；GSM8K 的 prefill L_recv ≈ 250 与 LongBench 的 ≥ 4k 差远，2 层 sample 就能分清。
- What is new beyond current drop result: 不依赖 offline predictor；用 in-stream 实测信号决策；与 D3-I1 形成对照（offline 训练 vs online 反馈）。
- Minimal pilot sketch, no execution: 离线 replay —— 取 v6 traces 的 per-layer L_recv 序列，模拟 layer-2 决策与 layer-L 真值的一致性；测「错切率」与「错切代价」。
- Required EP setting if later piloted: EP=8
- Expected positive signal: 前 2 层的 L_recv 均值与全 request L_recv 均值 R²≥0.7；策略「错切率」≤10%。
- Falsification criterion: 前 2 层均值 R²≤0.4；或错切代价（错切回退到 worse-than-static）>2% e2e。
- Main baseline: best per-workload-static configuration。
- Main risk: 早期层 L_recv 是弱/不稳定 predictor；"opportunistic" switching；错切静默损害精度或延迟。
- Literature references:
  - Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling (Li et al., ICLR 2026), verified=YES, https://arxiv.org/abs/2503.04398, 相关性：从 routing 预测 MoE 通信/expert 活动；和用部分 routing 信号减通信邻近。
  - Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts (He et al., 2025, ICLR 2026), verified=YES, https://arxiv.org/abs/2503.05066, 相关性：动态 drop policy 控制；机制邻近但目标是 straggler 而非 workload shape。
- Claude self-score:
  - Novelty potential: 4
  - MLSys fit: 5
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4
  - Risk of being just heuristic scheduling: 3
  - Overall: 8.0
- GPT/Codex critique:
  - (a) 是否只是 if-else 启发式？部分 —— 边界比较本质是 if-else，除非把"前 k 层信号 → 余下层 L_recv envelope"形式化为带误差模型的校准 predictor。
  - (b) 机制 systems-paper 级别？部分 —— signals/knobs 清楚，但 mapping 仍 hand-waved。
  - (c) 与现有 engines 区分？部分 —— MoE-specific 层粒度反馈不属于 vLLM/Sarathi 普通调度，但"switch a flag mid-request"可能塌成 engine knob。
  - 最强 objection：早期层 L_recv 可能是跨 layer/prompt/phase 不稳定的预测；审稿人会说是 opportunistic。
  - 锋利问题：什么 invariant 让前 2-3 层 L_recv 足够 predictive，使错切不会静默损害质量或延迟？
  - 撞车风险：Semantic Parallelism (2503.04398)；Capacity-Aware Inference (2503.05066)。
- Claude response to GPT: 同意"前 k 层是否稳定预测"是核心论证负担。修订承诺：把决策机制从 raw threshold 改为 **置信度门控** —— 若前 k 层的 L_recv 估计标准差超过 calibrated band 一半，*不切*（保守）；只在 high-confidence 区段切换。这样错切率可控、最坏退化是回退到 static。撞车：Semantic Parallelism 预测 expert routing 用于 offline scheduling，不是 online L_recv 反馈；Capacity-Aware Inference 的反馈量是 expert load imbalance（straggler），不是 L_recv envelope。**与 GPT 分歧**：GPT 倾向把 mapping 当作未完成的承诺；我认为「前 k 层 L_recv 均值 ↔ 全 request 均值」的稳定性可在阶段 A 离线 replay 就实证，承诺写在 falsification criterion 里。
- Decision: **revise** (加置信度门控；把 mapping 校准纳入贡献；继续推进)

### D3-Idea-3: Continuous-batching microbatch-level MoE policy

- Dimension: 3 — Workload-Adaptive MoE Scheduling
- One-sentence thesis: continuous batching 每步混合多个 request 的 prefill chunk 和 decode token；每步 *有效 per-layer L_recv* 是当前 microbatch 组成的确定函数；按 microbatch（而非 request）设 MoE 策略：policy(t) = f(Σ active_prefill_chunk_len + N_decode_tokens)。
- Yes/No hypothesis: per-microbatch 策略在异构 continuous-batching stream 上 mean e2e 比 single-static 改善 ≥5%，无精度回退。
- Runtime input signals: live microbatch 组成（每步总 token 数、prefill/decode 拆分）。
- Runtime decision variables: 每 microbatch 的 drop kernel 分支；分支内 drop rate。
- Core mechanism: 在 batch packer 输出后、kernel launch 前，按当前 step 的 prefill_tokens + decode_tokens 计算 *估计 L_recv*；查小表选 drop kernel；该 step 内所有 request 用同一 MoE comm policy（router top-K 与 expert 选择保持不变 → 不影响 per-request 精度）。
- Why this follows from the user finding: per-step L_recv 主要由 batch 组成决定，且 drop 在 L_recv > L* 时才有收益 —— 每步决策一次就好，不必预测 request 形状。
- What is new beyond current drop result: 把 MoE 策略放进 continuous batching 的 *step level* —— 既不是部署期常量，也不是 per-request offline binning。
- Minimal pilot sketch, no execution: 离线 —— 用 v6 traces 重构 per-step microbatch 组成，模拟 microbatch policy 的 e2e；与 single-static 对比。
- Required EP setting if later piloted: EP=8
- Expected positive signal: 异构 stream 上 ≥5% mean e2e 改善；per-request 精度 0 改变。
- Falsification criterion: per-microbatch policy 与 single-static 的 e2e 差 ≤1%；或 microbatch 内 request consistency 在 KV 层面被破坏。
- Main baseline: Sarathi-Serve / vLLM 的 single-MoE-policy continuous batching。
- Main risk: 若 router top-K 与 expert 选择没变，所有 byte-saving 都来自 drop policy 的开关 → 与 D3-I2 等价（差别只是 sample 粒度）；mechanism 必须显式定位 "step-as-decision-unit"。
- Literature references:
  - Orca: A Distributed Serving System for Transformer-Based Generative Models (Yu et al., OSDI 2022), verified=YES, https://www.usenix.org/conference/osdi22/presentation/yu, 相关性：iteration-level scheduling 是 per-step 控制的祖先；本 idea 把 MoE 策略加进同一粒度。
  - SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills (Agrawal et al., 2023), verified=YES, https://arxiv.org/abs/2308.16369, 相关性：同 iteration 内 prefill chunk + decode 复合；本 idea 在同 iteration 内加 MoE policy 决策。
  - Sarathi-Serve (Agrawal et al., OSDI 2024), verified=YES, https://arxiv.org/abs/2403.02310, 相关性：stall-free 调度；本 idea 兼容并在其之上加 MoE byte-policy 层。
- Claude self-score:
  - Novelty potential: 4
  - MLSys fit: 5
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 3.5
  - Risk of being just heuristic scheduling: 3
  - Overall: 8.0
- GPT/Codex critique:
  - (a) 部分 —— `f(prefill_tokens + decode_tokens)` 风险像阈值启发式，但 microbatch 级控制是合法 systems 单位。
  - (b) 部分 —— composition/decisions 清楚，但 mapping 需要超过"确定函数"才到 systems-paper 级。
  - (c) 与现有 engines 撞车风险高 —— 可能被读成 "add MoE-aware per-step policy to vLLM/SGLang/TensorRT-LLM continuous batching"，除非 MoE byte 模型是核心。
  - 最强 objection：microbatch policy 可能优化 step 但破坏 per-request 一致性（当同 batch 内不同 request 想要不同 MoE 行为）。
  - 锋利问题：当 MoE 通信字节、per-request 延迟、global throughput 不一致时，决策在优化哪一个？
  - 撞车风险：Orca (OSDI'22)；SARATHI (2308.16369)。
- Claude response to GPT: 关于 per-request 一致性：mechanism 必须显式承诺 —— **microbatch policy 只影响 MoE comm path（drop kernel 分支与字节量），不动 router top-K、不动 expert 选择、不动每 token 的输出语义**；因此同 batch 内 request 拿到 byte-字节不同但 *expert output 完全相同*，per-request 精度严格不变。撞车：Orca/SARATHI 是 dense LLM step 调度；MoE byte model 不在它们考虑里。**与 GPT 分歧**：GPT 担心 mapping = "确定函数" 太弱；我认为只要 byte model 是 D1-I10 metric 的运行时实例化，mapping 就有论证背书。决策目标问题：明确为 **MoE 通信字节优化** —— per-request 延迟与 throughput 是下游指标，靠 byte 减少同时改善。
- Decision: **revise** (显式承诺 expert output invariance；mapping 与 D1-I10 metric 耦合；继续推进)

### D3-Idea-4: Workload-conditioned elastic replica pool with per-session claim

- Dimension: 3 — Workload-Adaptive MoE Scheduling
- One-sentence thesis: 部署期把 expert replica slot 拆为 "anchored"（恒驻）与 "elastic"（per-session 可申领）；入口处按 predicted L_recv envelope 给 session 分配 elastic replica budget；长 prompt RAG 多拿、短 prompt long-decode 让出给 KV。
- Yes/No hypothesis: elastic claim 机制能在并发的 LongBench long-prompt 上维持 e2e +12%，同时不损害并发 short-prompt-long-decode 的 p99 latency。
- Runtime input signals: predicted L_recv trajectory + 当前 free VRAM + 并发 session map。
- Runtime decision variables: per-session elastic replica count + KV reservation。
- Core mechanism: 入口 admission 后给 session 一个 routing tag（"replica-rich" vs "replica-light"）；MoE dispatch 时该 session 的 token 优先去到 elastic 副本（仅 logical pointer update，无 weight 拷贝）；session 退出释放 tag。
- Why this follows from the user finding: long-prompt 的 prefill 阶段实际从更多 replica 中拿到 byte-spread 收益，short-prompt 短-decode 用不上，且当前 VRAM 都被恒驻 replica 占满限制 KV。
- What is new beyond current drop result: drop 是 per-step 字节级操作；本 idea 是 per-session 资源分配。
- Minimal pilot sketch, no execution: 离线 —— 在 v6 trace 上模拟 session 级 elastic claim 与 KV reservation 的合并图；测 long-prompt e2e 与 short-prompt p99 latency。
- Required EP setting if later piloted: EP=8
- Expected positive signal: long-prompt e2e 不变，short-prompt p99 降 ≥10%；总 KV 容量净增 ≥1 个并发 session。
- Falsification criterion: claim 切换的 routing 延迟 >50 µs，或同 layer 内 anchored vs elastic 副本路径切换的 kernel 拆分让 dispatch 变慢。
- Main baseline: 部署期静态 replica 全恒驻。
- Main risk: GPT "unit of elasticity" 问题 —— 必须承诺是 **logical routing only**，不是物理 weight 拷贝。
- Literature references:
  - ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models (Singh et al., 2025), verified=YES, https://arxiv.org/abs/2510.02613, 相关性：直接是 elastic MoE scaling + HBM/KV remap；机制最近 (zero-copy HBM remap)；区别在我们的 elasticity 单位只是 routing tag。
  - MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool (Hu et al., 2024), verified=YES, https://arxiv.org/abs/2406.17565, 相关性：elastic memory pool + KV 管理；区别在 MemServe 只管 KV，不管 MoE replica。
  - CRAFT (Zhao et al., 2026), verified=YES, https://arxiv.org/abs/2603.28768, 相关性：cost-aware replica allocation；区别在 CRAFT 静态、本 idea 是 per-session runtime claim。
- Claude self-score:
  - Novelty potential: 3.5
  - MLSys fit: 4.5
  - Mechanism clarity: 3 (claim 仲裁与 anchored/elastic 拆分粒度需要明确)
  - Feasibility on 8×RTX4090 EP=8: 3
  - Risk of being just heuristic scheduling: 3.5
  - Overall: 7.0
- GPT/Codex critique:
  - (a) 否（若 replica/KV tradeoff 是真实时资源分配）；部分（若 "long prompt 得 replica，decode 得 KV" 就是全部机制）。
  - (b) 部分 —— "spawn/release replicas" 昂贵且欠规范，除非 materialization/remapping 成本被建模。
  - (c) 部分 —— MoE expert replica 是特定的，但 elastic GPU/KV 资源池化已经拥挤；可能塌成 MoE-aware 内存管理器。
  - 最强 objection：per-session 时间尺度的 replica elasticity 在 4090-class PCIe 系统上可能太慢或太破坏 online serving。
  - 锋利问题：elasticity 的单位到底是物理 weight 拷贝、对已有 replica 的 logical routing、还是 remap HBM 预留？
  - 撞车风险：ElasticMoE (2510.02613)；MemServe (2406.17565)。
- Claude response to GPT: GPT 锋利问题 (f) 是本 idea 的核心 —— 我承诺 **elasticity 单位 = logical routing tag**：物理副本在部署期全部加载到位（含 anchored + elastic），运行时只通过 routing 表的 (session_id → replica_set) 映射改变某 session 用哪些 replicas；anchored 副本所有 session 都用，elastic 副本只当 session 持有 tag 时用。零物理拷贝，O(1) routing 表更新。撞车：ElasticMoE 是 HBM-remap 级（重；Ascend NPU 特性），我的 idea 是 routing-table 级（轻；纯 software）；MemServe 是 KV 池化（不管 MoE replicas）；CRAFT 静态分配。
- Decision: **revise** (承诺 logical-routing-only 单位；与 ElasticMoE/MemServe/CRAFT 显式区分；继续推进)

### D3-Idea-5: MoE-aware admission control with byte-budget SLO

- Dimension: 3 — Workload-Adaptive MoE Scheduling
- One-sentence thesis: serving loop 前置 admission controller，按 incoming request 的 *predicted worst-peer wire bytes 总载 + KV demand* 决定 admit / queue / redirect；预测公式直接用 D1-I10 metric 在该 request 的 predicted L_recv envelope 上外推。
- Yes/No hypothesis: MoE-byte-budget admission 在突发混合流量上把 p99 latency 降 ≥15%，总 throughput 不降。
- Runtime input signals: per-request predicted byte load + 当前 rolling byte budget + p99 latency 余量。
- Runtime decision variables: admit / queue / redirect。
- Core mechanism: D1-I10 metric `latency = α·worst_peer_bytes + β` 在 candidate request 的 (prompt_len, predicted_decode_len) 上外推得 *该 request 在每步贡献的 byte 增量*；若叠加超过 rolling 预算就 queue 或 redirect。
- Why this follows from the user finding: 字节是因果信号；现有 admission 用 token-count 或队列长度，是 dense-LLM 代理。
- What is new beyond current drop result: 把 D1-I10 的 cost model 从 *观察* 升级为 *admission 决策依据*。
- Minimal pilot sketch, no execution: 离线 —— 在 v6 trace 重构 request-level byte 累加；模拟 admission controller；测 p99 latency 与 throughput。
- Required EP setting if later piloted: EP=8
- Expected positive signal: 突发负载下 p99 降 ≥15%；throughput 不降。
- Falsification criterion: byte-budget admission 与 token-count baseline 比 ≤5% p99 改善；或不显著优于 SCORPIO-style heterogeneous-SLO admission。
- Main baseline: vLLM/SGLang token-count admission；SCORPIO heterogeneous-SLO admission。
- Main risk: GPT 警告 "byte 是 proxy 不是 control law" —— admission 决策应包括 queueing/KV/skew 等多因子，byte 单独不够。
- Literature references:
  - SCORPIO: Serving the Right Requests at the Right Time for Heterogeneous SLOs in LLM Inference (Tang et al., 2025), verified=YES, https://arxiv.org/abs/2505.23022, 相关性：admission control + queue management + batch selection for heterogeneous SLO；本 idea 把 MoE 字节作为新的压力信号注入。
  - BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs (Hu et al., 2025), verified=YES (existence), https://arxiv.org/abs/2507.17133, 相关性：MoE-specific SLO-aware serving under bursty workload + dynamic brownout token处理；adjacency 最强；机制详情未在 abstract，需要 pilot 前 deep read。
  - DistServe (Zhong et al., OSDI 2024), verified=YES, https://arxiv.org/abs/2401.09670, 相关性：goodput-oriented serving；本 idea 在 admission 而非 instance partition 上做。
- Claude self-score:
  - Novelty potential: 4 (BrownoutServe 撞车后下调候选)
  - MLSys fit: 5
  - Mechanism clarity: 3.5
  - Feasibility on 8×RTX4090 EP=8: 3
  - Risk of being just heuristic scheduling: 3
  - Overall: 7.5
- GPT/Codex critique:
  - (a) 部分 —— admission control 经典；若只是 "字节超阈值则 queue" 就是 if-else，MoE-specific worst-peer bytes 给它更锋利的资源 metric。
  - (b) 部分 —— p99 与 byte budget 的映射是 hand-waved 部分。
  - (c) 多半塌掉 —— 除非框为 *新 MoE 网络 / EP 压力模型*；vLLM/SGLang 入口 admission 已有自然 hooks。
  - 最强 objection："byte budget" 可能是代理而非 control law；p99 还受 queueing、KV 压力、routing skew、kernel contention 影响。
  - 锋利问题：为何 worst-peer MoE wire bytes 是 admission 的 bottleneck，而不是 broader latency predictor 里的一个 feature？
  - 撞车风险：SCORPIO (2505.23022)；DistServe。
- Claude response to GPT: 大体同意。我把贡献重定位为 **MoE-byte pressure as a feature inside a broader admission latency predictor**，而非 standalone admission system —— 即与 SCORPIO-style heterogeneous SLO 控制器组合，把 D1-I10 metric 作为新的输入信号注入。结构上是 D3-I5 = SCORPIO + (D1-I10 byte feature)。撞车：BrownoutServe 是直接的 MoE SLO-aware serving with bursty + dynamic token brownout —— 需要 deep read 后才能判定差异；目前阶段只标 preliminary risk。
- Decision: **revise** (重定位为 feature/composition，不是 standalone system；BrownoutServe 列为 #1 collision risk 待 deep read；继续推进但优先级下调)

### D3-Idea-6: Per-request drop-rate by expected ROI

- Dimension: 3 — Workload-Adaptive MoE Scheduling
- One-sentence thesis: 在 admission 时按 (prompt_len, predicted decode_len, current batch shape) 估该 request 启用 drop 能带来的 *预期 e2e ROI*；按 ROI 从 {0, 0.3, 0.5} 选 per-request drop rate。
- Yes/No hypothesis: per-request r 比 uniform r=0.3 在 mixed stream 上 mean e2e 改善 ≥4%，无 ≥0.5pp 精度回退。
- Runtime input signals: prompt_len + predicted decode_len + current batch shape。
- Runtime decision variables: per-request drop rate r ∈ {0, 0.3, 0.5}。
- Core mechanism: 离线训的 ROI 表 (ŷ = h(prompt_len, decode_len, batch_density)) → 给 r 打分；选最高。
- Why this follows from the user finding: 同一 r 在 long-prompt 上 +12%、在 short-prompt-long-decode 上 ±2%；按 request 分流 r 是最直接 lever。
- What is new beyond current drop result: 把 drop rate 从部署常量改成 per-request 函数。
- Minimal pilot sketch, no execution: 离线 —— 用 v6 LongBench grid 的 (request_shape, r, e2e) 元组训 ROI 表；hold-out 上测决策正确率。
- Required EP setting if later piloted: EP=8
- Expected positive signal: ROI 表在 hold-out 上 ≥80% 决策匹配 oracle-optimal r。
- Falsification criterion: hold-out 决策准确率 ≤60%，或 mean e2e 改善 ≤2%。
- Main baseline: uniform r=0.3。
- Main risk: 形式上是 "per-request rate"，机制等价于一张 per-shape lookup —— 与 D3-I1 (request-class router) 等价，应作为后者的细化项。
- Literature references:
  - (本 idea 与 D3-I1 高度同源，无额外独立参考；继承 D3-I1 的 LYNX 引用。)
- Claude self-score:
  - Novelty potential: 3
  - MLSys fit: 4
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4.5
  - Risk of being just heuristic scheduling: 4
  - Overall: 6.5
- GPT/Codex critique: 本轮未入选 GPT review。
- Claude response to GPT: n/a
- Decision: keep（与 D3-I1 等价，作为细化项保留）

### D3-Idea-7: Decode-length predictor for adaptive routing

- Dimension: 3 — Workload-Adaptive MoE Scheduling
- One-sentence thesis: 训一个轻量 predictor（proxy LM 或 prompt-features classifier）预测该 request 的 decode_len；用 predicted decode_len 决定 MoE 策略变量（drop rate / K_eff / 副本 budget）。
- Yes/No hypothesis: simple decode-length predictor 在代表性 trace 上达到 ≥0.7 R²，使用它做 routing 比 random/uniform 改善 e2e ≥5%。
- Runtime input signals: prompt embedding features / proxy LM 输出。
- Runtime decision variables: routing tag → MoE 策略变量。
- Core mechanism: pretrained proxy（如 small LLM）在 admission 时 forward 一次估 decode_len；按估 bin 选策略。
- Why this follows from the user finding: GSM8K 与 LongBench 在 decode_len 分布上差异巨大，是 workload shape 的主要信号。
- What is new beyond current drop result: 把 decode-len prediction 用于 MoE 策略，而非 dense LLM 调度。
- Minimal pilot sketch, no execution: 离线 —— 在 v6 trace 上 fit prompt-feature 回归；测预测 R²。
- Required EP setting if later piloted: EP=8
- Expected positive signal: predictor R²≥0.7；下游策略 e2e 改善 ≥5%。
- Falsification criterion: R²≤0.4；或决策与 oracle 差距 ≤2% e2e。
- Main baseline: 无 prediction（uniform 策略）。
- Main risk: decode-length 预测在 dense LLM 已经很拥挤（S3 / LTR / TRAIL / SSJF / EGTP）；MoE-policy conditioning 是唯一新角度。
- Literature references:
  - S³: Increasing GPU Utilization during Generative Inference for Higher Throughput (NeurIPS 2023), verified=YES (via search), 题目和 NeurIPS'23 venue 可查；arxiv: https://arxiv.org/abs/2306.06000, 相关性：proxy-model length prediction 的代表作；本 idea 用同样思路但下游决策是 MoE 策略而非 dense scheduling。
  - Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction (Qiu et al., 2024), verified=YES, https://arxiv.org/abs/2404.08509, 相关性：proxy model length prediction；不针对 MoE。
- Claude self-score:
  - Novelty potential: 2.5 (被 S3 family 压)
  - MLSys fit: 4
  - Mechanism clarity: 3.5
  - Feasibility on 8×RTX4090 EP=8: 3.5
  - Risk of being just heuristic scheduling: 3
  - Overall: 6.0
- GPT/Codex critique: 本轮未入选 GPT review。
- Claude response to GPT: n/a
- Decision: keep（低优先级，价值在与 D3-I1/I6 组合）

### D3-Idea-8: Adaptive chunked-prefill chunk size targeting MoE break-even

- Dimension: 3 — Workload-Adaptive MoE Scheduling
- One-sentence thesis: chunked-prefill 在 Sarathi-Serve 与后继中 chunk size 是 offline-tuned 超参；让 chunk size 在线自适应，每 chunk 的 L_recv 落到 [L*, sweet_spot] = [3271, 4096] rows/rank —— 小到能塞进 continuous-batching slot，大到 drop 真正赚钱。
- Yes/No hypothesis: adaptive chunk size targeting `L_recv ≈ 4k` 在 long-prompt workload 上比 Sarathi-Serve 默认 chunk size 提升 prefill ≥6% 且保持 stall-free。
- Runtime input signals: candidate chunk size 下的 predicted L_recv + per-layer `L*_ℓ` calibration + 当前 step 的 decode slot 余量。
- Runtime decision variables: per-request per-step 的 chunk size（受 stall-free 约束）。
- Core mechanism: 在 batch packer 选 prefill chunk 长度时，对几个候选 chunk size 估 L_recv（用 batch-density 模型），选离 `L*_ℓ + 800` 最近且不破 stall-free 的那个。
- Why this follows from the user finding: L* 来自 user 的 Tier-1 ablation；目前 Sarathi-Serve 的 chunk size 对此盲。
- What is new beyond current drop result: 显式把 MoE break-even 加入 chunked-prefill scheduler 决策。
- Minimal pilot sketch, no execution: 离线 —— 在 v6 trace 上估各候选 chunk size 的 L_recv 分布；模拟 chunk 决策与 e2e。
- Required EP setting if later piloted: EP=8
- Expected positive signal: 长 prompt 上 adaptive 比固定 chunk size 提升 ≥6% prefill；stall-free 保持。
- Falsification criterion: L_recv 与 chunk size 的关系跨 layer 跨 routing skew 漂移超 30%；或 stall-free 约束让 chunk 总落在 L*-bound 之外。
- Main baseline: Sarathi-Serve 默认 chunk size；DeepSpeed-FastGen Dynamic SplitFuse。
- Main risk: GPT "moving target" —— chunk size 改变 L_recv 是间接的，可能与 continuous batching 约束打架。
- Literature references:
  - Sarathi-Serve (Agrawal et al., OSDI 2024), verified=YES, https://arxiv.org/abs/2403.02310, 相关性：chunked-prefill 直接 baseline；offline chunk-size tuning。
  - DeepSpeed-FastGen (Holmes et al., 2024), verified=YES, https://arxiv.org/abs/2401.08671, 相关性：Dynamic SplitFuse 也调 chunk 组合；目标是 throughput 而非 MoE break-even。
- Claude self-score:
  - Novelty potential: 4
  - MLSys fit: 5
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4
  - Risk of being just heuristic scheduling: 2.5
  - Overall: 8.0
- GPT/Codex critique:
  - (a) 部分 —— 目标 `[L*, sweet_spot]` 是阈值调谐，但把 chunk size 与 MoE a2a break-even 绑定是具体 systems 角度。
  - (b) 部分 —— signal/knob 清楚；"per request per layer" chunk 设置可能太复杂，除非约束清晰。
  - (c) 部分 —— 接近 Sarathi-Serve/vLLM chunked prefill；只有当 MoE `L_recv` break-even（而非通用 stall-free 调度）驱动设计时才不同。
  - 最强 objection：chunk size 间接改变 L_recv，可能与 continuous batching 约束打架；审稿人会说是 "tune Sarathi for MoE"。
  - 锋利问题：`L_recv ≈ 4k` 跨 layer 跨 routing skew 稳定吗，还是 chunk controller 在追移动目标？
  - 撞车风险：Sarathi-Serve (2403.02310)；DeepSpeed-FastGen (2401.08671)。
- Claude response to GPT: GPT 的 moving target 问题答案与 D2-I5 相同 —— `L*_ℓ` 是 per-layer 校准常量（不是单一全局 L*）。Sarathi-Serve 与 Dynamic SplitFuse 都不把 MoE break-even 作为 chunk 决策的 objective —— 它们用 stall-free / throughput 作为 objective，本 idea 用 MoE byte break-even。区分清楚。简化承诺：chunk size 是 *per-request*（不是 per-request-per-layer），降低复杂度。
- Decision: **revise** (chunk size 简化为 per-request；与 Sarathi/Dynamic SplitFuse 显式区分；继续推进)

### D3-Idea-9: Workload-aware expert prefetch budget

- Dimension: 3 — Workload-Adaptive MoE Scheduling
- One-sentence thesis: 对 predicted long-decode requests 启用 expert prefetch；对 predicted short-decode requests 关掉 prefetch（overhead 超过收益）。
- Yes/No hypothesis: prefetch-on 在 long-decode 上 TTFT 改善 ≥10%；prefetch-off 在 short-decode 上避免 ≥3% e2e 退化。
- Runtime input signals: predicted decode_len。
- Runtime decision variables: prefetch 开关。
- Core mechanism: D3-I7 predictor 给 decode_len → 二值决策。
- Why this follows from the user finding: decode_len 与 prefetch 收益单调相关。
- What is new beyond current drop result: prefetch 当前是部署常量；本 idea 让它 per-request。
- Minimal pilot sketch, no execution: 离线 —— 在 v6 trace 上估 prefetch overhead vs 收益的 break-even decode_len。
- Required EP setting if later piloted: EP=8
- Expected positive signal: break-even decode_len 明显（≥30 tokens），predictor 在 hold-out 上能正确两分。
- Falsification criterion: break-even 不存在或 ≤10 tokens（短 decode 也赚）。
- Main baseline: prefetch 全开（DuoServe-MoE 路线）。
- Main risk: **DuoServe-MoE (2509.07379) 已实现 dual-phase prefetch with predictor**；本 idea 撞它；与 D2-I7（已 drop）同源。
- Literature references:
  - DuoServe-MoE (Zhang et al., 2025), verified=YES, https://arxiv.org/abs/2509.07379, 相关性：直接同机制；已在 D2-I7 上对此撞车 drop 了。
- Claude self-score:
  - Novelty potential: 2
  - MLSys fit: 4
  - Mechanism clarity: 3
  - Feasibility on 8×RTX4090 EP=8: 3.5
  - Risk of being just heuristic scheduling: 4
  - Overall: 5.0
- GPT/Codex critique: 本轮未入选 GPT review。
- Claude response to GPT: n/a
- Decision: drop（DuoServe-MoE 已覆盖；与 D2-I7 drop 同理）

### D3-Idea-10: Heterogeneous worker fleet with workload-shape sharding

- Dimension: 3 — Workload-Adaptive MoE Scheduling
- One-sentence thesis: 多 instance fleet 中按 workload class 专门化：long-prompt-short-decode worker（带 aggressive drop）与 short-prompt-long-decode worker（无 drop）；前端 router 按 predicted shape 派发。
- Yes/No hypothesis: 异构 fleet 与 single-class baseline 比 mean e2e 提升 ≥10% 且 p99 不损。
- Runtime input signals: predicted workload shape。
- Runtime decision variables: 派发到哪个 worker class。
- Core mechanism: 前端 router + 两类专门化 worker。
- Why this follows from the user finding: 同一硬件下两类 workload 偏好 *不同* deployment config。
- What is new beyond current drop result: 当前 fleet 是同构。
- Minimal pilot sketch, no execution: 离线 —— 在 v6 trace 上模拟双 class fleet 的 cost/perf trade。
- Required EP setting if later piloted: EP=8（每 worker 上）
- Expected positive signal: 异构 fleet 的 *资源调度* 比同构 fleet 优 ≥10%。
- Falsification criterion: 资源调度差 ≤3%；或 router 错派率高导致 e2e 退化。
- Main baseline: 同构 fleet。
- Main risk: **本质是 DistServe-for-MoE-with-extra-mechanism**；GPT 会问 "为何不是 DistServe 的 MoE variant"。
- Literature references:
  - DistServe (Zhong et al., OSDI 2024), verified=YES, https://arxiv.org/abs/2401.09670, 相关性：跨 instance 拆 phase；本 idea 跨 instance 拆 workload shape；同一思路另一维度。
  - SplitWise (Patel et al., 2023), verified=YES, https://arxiv.org/abs/2311.18677, 相关性：同样的跨 instance 拆 phase。
- Claude self-score:
  - Novelty potential: 3
  - MLSys fit: 4.5
  - Mechanism clarity: 3.5
  - Feasibility on 8×RTX4090 EP=8: 2.5（多 instance）
  - Risk of being just heuristic scheduling: 3
  - Overall: 6.0
- GPT/Codex critique: 本轮未入选 GPT review。
- Claude response to GPT: n/a
- Decision: keep（作为 DistServe-for-MoE 风险的显式对照；低优先级）

---

### Dimension 3 Summary

- Generated ideas: 10 (D3-I1 … D3-I10) 去重后。早期草稿合并："speculative decoding × MoE drop coupling" 因与 MoE byte lever 关系太弱并入 D3-I9 思考栈后舍弃；"workload-shape sharding" 保留为 D3-I10。
- Top 5 reviewed by GPT (1 codex call, bundled): D3-I2, D3-I3, D3-I4, D3-I5, D3-I8。
- Kept (no major change): D3-I1, D3-I6, D3-I7, D3-I10。
- Revised (carried forward with explicit edits):
  - D3-I2: 加 **置信度门控** —— 早期层 L_recv 估计标准差超过 calibrated band 一半时不切；mapping 校准纳入贡献。
  - D3-I3: 显式承诺 **expert output invariance** —— microbatch policy 只动 MoE comm path 与字节量、不动 router top-K、不动 expert 选择；byte model 锚定到 D1-I10 metric。
  - D3-I4: elasticity 单位锁定为 **logical routing tag**（O(1) routing 表更新，零物理拷贝），与 ElasticMoE (HBM-remap) / MemServe (KV-only) / CRAFT (static) 显式区分。
  - D3-I5: 从 standalone admission 重定位为 **feature in broader admission predictor**，与 SCORPIO 组合；BrownoutServe 列为 #1 collision risk 需要 deep read；优先级下调。
  - D3-I8: chunk size 简化为 **per-request**（非 per-layer）；moving-target 问题用 `L*_ℓ` per-layer 校准回应（同 D2-I5）。
- Dropped: D3-I9 (workload-aware expert prefetch budget) —— DuoServe-MoE (arXiv:2509.07379) 已实现 dual-phase prefetch with decode-side predictor；与 D2-I7 同理 drop。
- Main GPT criticisms:
  1. 多个 idea 在 "if-else heuristic" 边缘 —— `f(prompt_len)` / 阈值比较 / 按 shape 分流，除非把 mapping 形式化为带误差模型的 calibrated predictor，都可能被审稿人当 heuristic scheduler。
  2. 与 vLLM / SGLang / Sarathi-Serve / DistServe 的差异化压力大 —— D3-I3 (microbatch) 最容易被读成 "engine knob"，D3-I5 (admission) 最容易塌入 SCORPIO，D3-I8 (chunk) 最容易被读成 "tune Sarathi for MoE"。
  3. 多个 idea 的中心 mapping 是 hand-waved："确定函数"、"早期层预测整 envelope"、"字节 → p99 latency" 都需要论证背书。
  4. per-microbatch policy 可能破坏 per-request 一致性（D3-I3）。
  5. per-session elasticity 的单位（D3-I4）必须明确，否则会与 "expensive replica materialization" 形象绑定。
- Main Claude–GPT disagreements:
  - On D3-I2: GPT 倾向把 "前 k 层 → 全 envelope" mapping 当未完成承诺；Claude 认为该 mapping 可在阶段 A 离线 replay 实证，并通过 falsification criterion 兜底。
  - On D3-I3: GPT 担心 mapping="确定函数"太弱；Claude 认为只要 byte model 是 D1-I10 metric 的运行时实例化，mapping 就有论证背书 —— 关键是 expert output invariance 承诺。
  - On D3-I5: 大体对齐 —— Claude 接受重定位为 "byte feature in broader predictor"，但保留 "MoE-specific worst-peer 字节是 dense systems 看不到的新压力源" 作为 contribution 的独立点。
- Estimated Codex usage: 1 substantive Codex call this dimension (5 ideas bundled). Cumulative across D1+D2+D3 = 5 substantive + 1 ping. User lifted budget mid-session but the "top 5 / 1 round each" protocol was kept as it is a content rule.
- Next recommended dimension (do not execute): **Dimension 4**（brief 用户未指定）。等待人工 gate 与下一维度的明确 brief。

---

### 维度 3 — 质检结果（QC pass, 2026-05-29）

QC 范围：对 D3 的 10 个 idea 重新做 (1) 硬性门 —— 可证伪、具体性、与 finding 关系（"a2a 是 MoE 推理真瓶颈"）、重复性；(2) 修订后的质量评分 —— novelty / story_fit（关键 severity：能 work 但 *不* 推进 "a2a-as-first-class" 主故事必须低分；heuristic scheduler / 通用调度优化 都算）/ pilot_feasibility / pilot_cost（反向）；(3) 每个 idea 一句话赌注；(4) 对 top 3 调 GPT 挑战。不做撞车结论，不进入阶段 B。

GPT 挑战 trace：`docs/aris/traces/dimension3_qc_review.md`。下表中分数为 *GPT 之后*，与 GPT 的分歧记录在 trace 里。

#### QC 表

| idea_id | 硬性检查 | novelty | story_fit | feasibility | cost | 一句话赌注 | 标记 |
|---------|----------|---------|-----------|-------------|------|------------|------|
| D3-I1 | PASS | 2.5 | 3.5 | 4.5 | 5 | (prompt_len, max_tokens, sampling) 单变量足以预测 L_recv envelope 到 ≥85% 分类精度，因此一张离线训的 4-class lookup 表能消除 long vs short workload 的 MoE 策略错配 | PASS（borderline；novelty 被 LYNX/length-prediction 系列压低；story_fit 间接 —— 走 predicted L_recv envelope 不直接走 byte lever） |
| D3-I2 | PASS | 3.5 | 4.5 | 4 | 5 | 前 2-3 层 L_recv 均值在 R²≥0.7 下预测整 request 的 L_recv envelope；加置信度门控后错切率可控、最坏退化到 static —— 因此 online 反馈切换能在 mixed stream 上击败 best-per-workload-static | PASS（top 2 post-GPT；GPT 指出 "mean 不保护 layer tails / expert skew"，要求 mitigation 升级；Claude 接受 novelty 4→3.5、story_fit 5→4.5） |
| D3-I3 | PASS | 3 | 5 | 3.5 | 3.5 | MoE drop kernel 分支在 microbatch 边界切换不破坏 per-request 一致性（router top-K 与 expert 选择不动），按 microbatch 的 L_recv 估计选 drop 分支能在异构 continuous-batching stream 上拿到 ≥5% mean e2e 优势 | PASS（top 1 post-GPT；GPT 评为 "best story fit / least empty / structurally safer / 唯一可成为 systems primitive 的候选"；Claude 接受 novelty 3.5→3 但 story_fit=5 GPT 也认可） |
| D3-I4 | PASS | 3 | 3.5 | 3 | 3 | 用 logical routing tag（零物理拷贝、O(1) 路由表更新）作 elasticity 单位，足以让 per-session replica budget 适应 workload shape，并在并发 long-prompt + short-prompt-long-decode 流量上同时维持 long e2e +12% 与 short p99 -10% | PASS-WEAK（cross-dimension 重叠：与 D2-I2 / D2-I9 共用 shadow-replica + 给 KV 的同一 mechanism，决策单位 phase vs session 不同；story_fit 间接 —— 走 memory fungibility 而非 byte 直接） |
| D3-I5 | PASS | 3 | 4 | 3 | 3.5 | MoE worst-peer wire bytes 是 dense SLO 系统看不到的新压力源；把它作为 SCORPIO-style admission predictor 的额外 feature 注入，能在 bursty mixed traffic 上把 p99 latency 多降 ≥15% | PASS（已重定位为 feature in broader predictor；与 SCORPIO 组合而非 standalone；BrownoutServe 列为 #1 collision，需 deep read） |
| D3-I6 | DUP（与 D3-I1 实质同源） | — | — | — | — | drop rate 的 e2e ROI 在 (prompt_len, decode_len, batch_density) 上离线训得可达 ≥80% 决策匹配 oracle，因此 per-request r ∈ {0, 0.3, 0.5} 能比 uniform r=0.3 多拿 ≥4% mean e2e | DUP（作者本人已注 "与 D3-I1 高度同源、无独立参考"；D3-I1 = 预测 L_recv envelope → 选 policy bundle；D3-I6 = 预测 ROI → 选 r；都是 per-request offline predictor → lookup，机制等价） |
| D3-I7 | PASS | 2 | 3 | 3.5 | 4 | decode_len 在 prompt features 上可被预测到 R²≥0.7，且此预测能改进 MoE 策略选择 → e2e ≥5% | PASS-WEAK（novelty 被 S3/LTR/TRAIL/SSJF/EGTP length-prediction 系列严重压低；机制路径与 D3-I1 高度重叠 —— predictor → policy bundle；只是预测变量不同；story_fit 间接） |
| D3-I8 | PASS | 3 | 3 | 4 | 4 | per-request chunk size 目标 L_recv ≈ L*_ℓ + 800 行/rank 跨 layer 经 per-layer 校准稳定，能比 Sarathi-Serve 默认 chunk size 多拿 ≥6% long-prompt prefill 速度并保持 stall-free | PASS-WEAK（GPT 挑为本轮 loser："Sarathi-Serve + Dynamic SplitFuse 已占 chunked-prefill 适配空间、MoE 主题是 bolted on、最像 'Sarathi-extension 经验研究' 而非新 mechanism"；novelty 4→3、story_fit 5→3 双双下调） |
| D3-I9 | REJECT — DuoServe-MoE 覆盖 | — | — | — | — | predicted long-decode 启用 prefetch、short-decode 关掉能在两类 workload 上同时正向 | REJECT（作者本人已 drop；DuoServe-MoE arXiv:2509.07379 已发表 dual-phase prefetch with decode-side predictor 同机制；与 D2-I7 drop 同理） |
| D3-I10 | PASS（story_fit 显著低） | 2.5 | 2.5 | 2.5 | 2 | long-prompt-short-decode 与 short-prompt-long-decode 在硬件上偏好 *不同* deployment config，分到两类专门化 worker 能让资源调度比同构 fleet 优 ≥10% | PASS-WEAK（本质是 DistServe-for-MoE-with-extra-mechanism；作者本人已承认；story_fit 间接 —— 走 instance partition 而非 byte lever；feasibility 也低，需要 multi-instance setup） |

REJECT / DUP 的具体原因：
- **D3-I9 REJECT**：DuoServe-MoE (arXiv:2509.07379) 已发表完全同机制 —— prefill 期 expert 活动驱动 decode prefetch with layer-level predictor。作者已自行 drop；与 D2-I7 drop 同源。
- **D3-I6 DUP**：作者本人已注 "本 idea 与 D3-I1 高度同源、无额外独立参考"。D3-I1 与 D3-I6 都是 "per-request offline predictor → policy lookup" 形态：I1 predict L_recv envelope，I6 predict ROI；输出层都是离散 policy bundle。应作为 I1 的细化项处理，不独立计数。

#### Post-GPT top 2–3（按 novelty + story_fit 排序，PASS 项）

GPT 挑战之后 D3-I8 跌出 top 3（5→3 → 6 总分），D3-I3 从 8.5 升为 #1，D3-I2 与之并列。Re-rank：

- **Top 1 — D3-I3 (8.0)**：continuous-batching microbatch-level MoE policy。GPT 独立评为三者中 "best story fit"、"least empty"、"structurally safer"、"唯一可成为 systems primitive 的候选"。
  - 一句话赌注：MoE drop kernel 分支在 microbatch 边界切换不破坏 per-request 一致性（router top-K 与 expert 选择不动），按 microbatch 的 L_recv 估计选 drop 分支能在异构 continuous-batching stream 上拿到 ≥5% mean e2e 优势。
- **Top 2 — D3-I2 (8.0)**：online L_recv early-layer feedback with mid-request switch。与 I3 并列；GPT 指出最强 mitigation 缺口（"mean prediction 不保护 layer tails / expert skew"），Claude 接受并承诺把 falsification 升级为 per-layer 尾部方差监控，不只是 mean 上的置信度门控。
  - 一句话赌注：前 2-3 层 L_recv 均值在 R²≥0.7 下预测整 request 的 L_recv envelope；加置信度门控后错切率可控、最坏退化到 static —— 因此 online 反馈切换能在 mixed stream 上击败 best-per-workload-static。
- **Top 3 — D3-I5 (7.0)**：MoE-aware admission control with byte-budget SLO（已重定位为 SCORPIO-style admission 的 byte-feature 注入）。BrownoutServe (arXiv:2507.17133) 是 #1 collision，必须 deep read 后才能定 standalone 与否。
  - 一句话赌注：MoE worst-peer wire bytes 是 dense SLO 系统看不到的新压力源；把它作为 SCORPIO-style admission predictor 的额外 feature 注入，能在 bursty mixed traffic 上把 p99 latency 多降 ≥15%。

候选 #4 — D3-I8 (6.0)：adaptive chunked-prefill chunk size。GPT 挑为 loser；可作 "Sarathi-extension 经验研究" 形态保留，不作 standalone 系统贡献。
候选 #5 — D3-I4 (6.5)：elastic replica pool。cross-dimension 与 D2-I2 / D2-I9 重叠 → 建议作为 D2 idea 的 D3 视角细化项处理。

#### Claude–GPT 评分分歧总结

| idea | Claude pre-GPT (novelty, story_fit) | GPT 挑战方向 | Claude post-GPT (novelty, story_fit) | 是否仍分歧 |
|------|-------------------------------------|--------------|--------------------------------------|------------|
| D3-I2 | 4, 5 | novelty 高（Capacity-Aware Inference）；story_fit "still a feedback controller for a drop knob"；mean 预测不保护 layer tails | 3.5, 4.5 | 部分对齐；分歧点：Claude 主张通过 *per-layer 尾部方差监控* 把 mean-only 升级为多尺度信号，story_fit 4.5 不再下调 |
| D3-I3 | 3.5, 5 | novelty 高（Orca/SARATHI 已是 iteration-level scheduling 的祖先）；story_fit GPT 独立评为 "best of three" | 3, 5 | 基本对齐；novelty 让步、story_fit 双方认可 |
| D3-I8 | 4, 5 | 双双过高；Sarathi + Dynamic SplitFuse 已占 chunked-prefill 空间；"MoE 主题是 bolted on" / "最像 Sarathi-extension 经验研究而非新 mechanism" | 3, 3 | 大体认输；分歧点：Claude 认为如显式包装为 "MoE break-even 重新定义 chunk objective" 的对比研究仍可发表，GPT 倾向不视为独立系统贡献 |

#### QC 结论（维度 3）

- 总数：10
- PASS：8（D3-I1, I2, I3, I4, I5, I7, I8, I10）；其中：
  - I3 是 GPT 唯一肯定 "可成 systems primitive" 的候选；
  - I4 cross-dimension 与 D2-I2 / D2-I9 重叠，应统筹处理；
  - I7 与 I1 机制路径高度重叠（都是 predictor → policy bundle）、novelty 被 length-prediction 系列严重压；
  - I8 被 GPT 挑为本轮 loser；I10 story_fit 显著低；I1 borderline。
- REJECT：1（D3-I9 expert prefetch budget，DuoServe-MoE 覆盖）
- DUP：1（D3-I6 per-request drop-rate by ROI，与 D3-I1 实质同源；作者本人已承认）
- Top 2-3：D3-I3 → D3-I2 → D3-I5（D3-I8 从 tied-#1 跌出 top 3；候选 #4 = D3-I8，候选 #5 = D3-I4）
- Codex 本 QC 轮额外消耗：1 次（GPT 挑战 top 3 一并打包，high reasoning）。D3 累计：idea critique 1 + QC 1 = 2 substantive Codex calls。
- 跨 D1+D2+D3 累计：6 次 substantive + 1 次连通性 ping，远低于 15 次告警阈值。用户解除 GPT 调用限制的消息已收到，但 "top 3 / 1 round each" 是内容规则继续保留。
- 维度结束，等待人工 gate。

---

## Dimension 4 — L*-Aware Cost Model and Control Plane

本维度的核心 user finding：break-even `L* ≈ 3271 rows/rank`（Tier-1 −5% 阈值插值）是 prefill MoE 块的拐点 —— 在它之下 drop 因 launch/sync 开销反而变慢，在它之上收益快速饱和、甜点 `L_recv ≈ 4k`。这是经验阈值，但尚未被系统化为 *cost model* 与 *control plane*。所有 phase-aware / workload-adaptive / placement / drop-policy 的设计都隐式依赖 "我知道现在 L_recv 在哪侧"。本维度目标：把 `L*` 升级为可预测、可决策、可监测的系统 primitive，避免在 latency-bound 区误启动 drop。**不**再提出一个新的 drop policy；要做的是 *围绕已有 lever 的 cost-model + control-plane*。

维度 4 自评分规则：Novelty potential、MLSys fit、Mechanism clarity、Feasibility on 8×RTX4090 EP=8、**Risk of being only a cost model, not a system**（分数越高越像 "光建一个模型不落到系统"）、Overall (1–10)。

### D4-Idea-1: Per-layer `L*_ℓ` calibration as deployable artifact

- Dimension: 4 — L*-Aware Cost Model and Control Plane
- One-sentence thesis: 把校准过的 48-entry `L*_ℓ` 表与 model checkpoint 一同发布，作为 (model × hardware × runtime) 三元组的部署期一等 artifact —— 与 quantization scales / KV-page sizes 同级 —— 所有下游 MoE 调度决策（drop 开关、replica budget、K_eff）查表。
- Yes/No hypothesis: 在一个 workload class（如 LongBench long-prompt）上校准的 `L*_ℓ` 在 hold-out class（如 GSM8K prefill）上 ≥80% 的层相对误差 ≤±15%。
- L*/L_recv role in the mechanism: `L*_ℓ` 就是部署 artifact 的本体；运行时的所有 lever 选择都是 "比较实测 L_recv_ℓ 与查表得 L*_ℓ"。
- Control-plane inputs: 部署期 (model, hardware, runtime) 三元组；校准期观测的 (L_recv, latency) 对。
- Control-plane decision: per-layer `L*_ℓ` 数值 + 其置信区间。
- Error/failure mode: 校准偏高 → drop 永不触发，把 +12% 留在桌上；偏低 → drop 在 latency-bound 区触发，e2e 退化至 −5%。
- Core mechanism: 在校准期跑 L-sweep microbench（user 已有），在每层独立插值出 −5% 阈值；以 (L*_ℓ, σ_ℓ) 元组形式持久化。
- Why this follows from the user finding: user 已经在一个 layer 上测出 L*≈3271；本 idea 是这一发现的部署期通用化。
- What is new beyond current drop result: drop 现在是部署期常量；本 idea 把"何时启用 drop"升级为 artifact 查表，且加入 per-layer 维度。
- Minimal pilot sketch, no execution: 离线 —— 用 v6 LongBench grid 估每层 break-even；在 GSM8K prefill 上做 hold-out 验证。
- Required EP setting if later piloted: EP=8
- Expected positive signal: 跨 workload class 的 hold-out 上 ≥80% 层相对误差 ≤±15%。
- Falsification criterion: 仅 ≤50% 层达到此精度；或 `L*_ℓ` 跨 layer 方差 >3× 难以稳定校准。
- Main baseline: 部署期单一 global `L*=3271`（user 当前隐式假设）。
- Main risk: GPT 说"看起来像 hardware-specific autotuning metadata"。需要 transferability claim 保护：`L*_ℓ` 主要依赖 bandwidth / launch-overhead 比，跨硬件代际变化慢。
- Literature references:
  - Vidur: A Large-Scale Simulation Framework For LLM Inference (Agrawal et al., MLSys 2024), verified=YES, https://arxiv.org/abs/2405.05465, 相关性：LLM 推理性能 cost model；以 dense LLM 为主，未把 MoE break-even 作为 artifact 单独抽取。
  - MoE-CAP: Benchmarking Cost, Accuracy and Performance of Sparse MoE Systems (Jiang et al., 2024), verified=YES, https://arxiv.org/abs/2412.07067, 相关性：MoE 系统的 cost-accuracy-perf benchmark；是 benchmark 而不是 cost model + decision boundary。
- Claude self-score:
  - Novelty potential: 3.5
  - MLSys fit: 5
  - Mechanism clarity: 4.5
  - Feasibility on 8×RTX4090 EP=8: 5
  - Risk of being only a cost model: 3
  - Overall: 8.0
- GPT/Codex critique:
  - (a) Marginal —— shipped threshold table 看起来像 profiled config 而不是系统机制。
  - (b) 部分 —— 能避免坏 drop 激活，但多半只是把已有的 +12% / −5% 边界换个说法。
  - (c) 部分 —— inputs / failure direction 清楚；confidence interval 语义和 transfer axis 仍 hand-waved。
  - 最强 objection："这就是硬件专属 autotuning metadata，不是研究贡献。"
  - 锋利问题：什么 invariant 让 `L*_ℓ` 在这一 (model × hardware × runtime) 三元组之外仍有科学意义？
  - 撞车风险：Vidur；MoE-CAP。
- Claude response to GPT: 强同意 GPT 的"autotuning metadata"objection。修订承诺：必须把贡献从"shipped table"升级为 **"shipped table + 可验证的 transferability claim"** —— 即 `L*_ℓ` 主要由两个机器结构量决定（per-layer activation byte 量 / launch overhead），这两个量跨硬件代际变化慢于 3 个月（H100 → H200 → B100 实际带宽差异 <2×）。带这个 invariant 主张就脱离 metadata 性质。撞车：Vidur 不专门处理 MoE break-even；MoE-CAP 是 benchmark 不是 decision boundary。
- Decision: **revise**（升级为 artifact + transferability invariant claim；继续推进）

### D4-Idea-2: `L_recv` estimator from microbatch composition + router histogram

- Dimension: 4 — L*-Aware Cost Model and Control Plane
- One-sentence thesis: 给定 live microbatch 组成（per-request prefill tokens、decode tokens）+ 离线校准的 per-layer router skew histogram，**在 dispatch 之前**预测 per-layer `L_recv`；该估计是所有 D4 控制决策的运行时输入。
- Yes/No hypothesis: `L_recv` 估计器在 hold-out v6 traces 上达到 R²≥0.85；mixed stream 上的错估代价 ≤1% e2e vs oracle。
- L*/L_recv role in the mechanism: `L_recv` 估计是 control plane 的核心信号；与 `L*_ℓ`（D4-I1）比较得到 drop/no-drop 决策。
- Control-plane inputs: microbatch composition (prefill_tokens, decode_tokens) per step + per-layer router histogram + `L*_ℓ` 表。
- Control-plane decision: 下一步的 predicted L_recv_ℓ → 通过查表选 drop 开关 + drop rate。
- Error/failure mode: 错估 (L̂_recv > L* 但实际 L_recv ≤ L*) → drop kernel 在 latency-bound 区被选中 → e2e 慢 ≤2%。
- Core mechanism: dispatch 前一步，runtime 取微批 token 计数 + 缓存的 router skew histogram（按 expert 的相对热度），近似计算 per-layer per-rank token 到达数（L_recv 的定义量）。
- Why this follows from the user finding: 一切 drop 决策都隐式依赖"我知道 L_recv 在哪侧"；user 当前用观测后回看，本 idea 改成 dispatch 前预测。
- What is new beyond current drop result: drop 现在是事后决策；本 idea 把决策前置到 dispatch 前。
- Minimal pilot sketch, no execution: 离线 —— 在 v6 traces 上 fit 估计器；测 R² 与错估代价。
- Required EP setting if later piloted: EP=8
- Expected positive signal: R²≥0.85；错估代价 ≤1%。
- Falsification criterion: R²≤0.6；或错估代价 >3% e2e。
- Main baseline: 实测 L_recv 后调度（即 D2-I5 / D3-I2 用的反馈式控制）。
- Main risk: GPT 的锋利问题 —— 如果 router 输出在 dispatch 前就可得，为何要从 stale histogram 预测？答案：在 `ep_ht` runtime 里 router 输出可得但 *跨 rank L_recv reduction* 仍需 a2a；估计器避免这次 reduction。
- Literature references:
  - MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache (Xue et al., 2024), verified=YES, https://arxiv.org/abs/2401.14361, 相关性：sequence-level activation 预测；不是 per-step L_recv 估计。
  - Fast MoE Inference via Predictive Prefetching and Expert Replication (Jyothish et al., 2026), verified=YES, https://arxiv.org/abs/2605.11537, 相关性：predictive prefetching for overloaded experts；目标是 prefetch 而非 control-plane decision。
- Claude self-score:
  - Novelty potential: 4
  - MLSys fit: 5
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4.5
  - Risk of being only a cost model: 3.5
  - Overall: 8.0
- GPT/Codex critique:
  - (a) 可能足够 —— prediction-before-dispatch 是真正的 control-plane primitive。
  - (b) 部分 —— 新收益只在预测击败 live routed-token 测量或 token-count heuristic 时存在。
  - (c) 部分 —— inputs 具体；router histogram 平稳性、prompt mix、layer/expert 相关性欠规范。
  - 最强 objection：「如果 router 输出在 dispatch 前可得，为何要从 stale histogram 预测 L_recv？」
  - 锋利问题：什么决策必须在实际 routed counts 可观测之前做？
  - 撞车风险：MoE-Infinity；Fast MoE Inference via Predictive Prefetching。
- Claude response to GPT: GPT 的锋利问题是本 idea 的中心论证负担。**答案**：router 输出（logits + 选中 expert 列表）在 dispatch 前确实可得；但 `L_recv_ℓ`（per-rank 收到的 token 行数）需要 **跨 rank reduction** —— 一次额外的 collective —— 才能精确测量。本 idea 的 estimator 用 *cached per-layer router skew histogram* 估这个 reduction 的结果，避免一次 collective 摆在调度决策的关键路径上。换句话说：control-plane 决策需要 dispatch-前 L_recv 估计，因为 dispatch-后 L_recv 实测意味着多一次 collective 延迟到关键路径。撞车：MoE-Infinity 预测的是 sequence-level expert 激活以做 caching/offload，不是 per-step L_recv；Fast MoE Inference (2605.11537) 预测 expert overload 以做 prefetch，不是 L_recv estimator。
- Decision: **revise**（把 GPT 的锋利问题答在 thesis 内显式化：predicted L_recv 价值在于避开 reduction-on-critical-path；继续推进）

### D4-Idea-3: Multi-dimensional regime classifier (compute / comm / idle / transition)

- Dimension: 4 — L*-Aware Cost Model and Control Plane
- One-sentence thesis: 一个轻量分类器把 per-layer 当前状态分到 {compute-bound, comm-bound, idle, transition} 四类 regime；regime 决定哪一根 lever 主导（drop / K_eff / placement variant / replica budget）。
- Yes/No hypothesis: 4-regime 分类器在 hold-out v6 traces 上 ≥85% 准确率；按 regime 选 lever 比单一 lever 多拿 ≥4% mean e2e。
- L*/L_recv role in the mechanism: `L_recv ≷ L*` 与其它信号（KV pressure、replica load）共同构成 regime 分类的 input。
- Control-plane inputs: per-layer (L_recv, batch density, KV pressure, replica utilization) 元组。
- Control-plane decision: regime label per step → lever 选择。
- Error/failure mode: 错分类 → 选错 lever → 控制动作浪费或 e2e 退化。
- Core mechanism: 简单决策树/线性分类器在离线 trace 上训练；按 regime 输出 lever id。
- Why this follows from the user finding: 同一观测下 lever 的有效性差几个量级 —— GSM8K decode 的 drop 是 anti-leverage，LongBench prefill 的 drop 是 +12%。Regime 分类显式编码这种差异。
- What is new beyond current drop result: drop 当前是按部署常量；本 idea 把 lever 选择 multiplexed 到 regime。
- Minimal pilot sketch, no execution: 离线 —— 标注 v6 traces 的 regime；fit 分类器；hold-out 测准。
- Required EP setting if later piloted: EP=8
- Expected positive signal: ≥85% 准确率；regime-guided 比单 lever 多 ≥4% e2e。
- Falsification criterion: ≤70% 准确率；或 regime-guided e2e ≤2% 改善。
- Main baseline: 单 lever（uniform drop）。
- Main risk: regime 数太少 → 不区分；太多 → overfit 校准集。
- Literature references:
  - Roofline: An Insightful Visual Performance Model (Williams et al., CACM 2009), verified=NO (作者引用经典但本 trace 内未独立 web_fetch 验证；标 [MEMORY-NEEDS-VERIFY])，相关性：roofline 把性能分到 compute-bound vs memory-bound regime 是本 idea 的 dense-LLM 类比。
- Claude self-score:
  - Novelty potential: 3.5
  - MLSys fit: 4.5
  - Mechanism clarity: 3.5（regime 数太敏感）
  - Feasibility on 8×RTX4090 EP=8: 4
  - Risk of being only a cost model: 4
  - Overall: 7.0
- GPT/Codex critique: 本轮未入选 GPT review。
- Claude response to GPT: n/a
- Decision: keep（作为 D4-I1/I2/I4 的下游组合候选）

### D4-Idea-4: One-way "don't-drop" safety guard

- Dimension: 4 — L*-Aware Cost Model and Control Plane
- One-sentence thesis: 在任何底层 drop scheduler（如 D3-I2、D2-I5）之上放一个 *严格精度的非对称分类器* —— "L_recv 远低于 L*，drop 会害人" —— 输出二值 `drop_allowed ∈ {yes, no}`，使 scheduler 不可能在 latency-bound 区启用 drop。调到 *低假阴性*（绝不漏放退化），容忍 *高假阳性*（偶尔错过 +12% 机会）。
- Yes/No hypothesis: 在 margin = 0.15·L*_ℓ 下，二值分类器达到 false-negative rate ≤1%，false-positive rate ≤20% —— 即 ≥80% 真 drop 机会被捕到、≤1% 真退化漏放。
- L*/L_recv role in the mechanism: 决策完全围绕 `L_recv vs L*_ℓ − margin`；本 idea 让该 boundary 成为强制硬约束。
- Control-plane inputs: per-layer L_recv_ℓ + `L*_ℓ − margin`。
- Control-plane decision: 二值 drop gate。
- Error/failure mode: 假阴性（允许 drop 但 L_recv<L*）→ e2e 退化；假阳性（屏蔽 drop 但 L_recv>L*）→ 把收益留在桌上。非对称。
- Core mechanism: asymmetric-loss 训练的二值分类器；用 Tier-1 ablation 数据校准 margin。
- Why this follows from the user finding: user 当前的策略选择隐式假设 "不在 latency-bound 区"；本 idea 把这个假设显式化为可证伪的硬约束。
- What is new beyond current drop result: drop scheduler 不再独立做 boundary 判断，必须经此 guard。
- Minimal pilot sketch, no execution: 离线 —— 在 v6 traces 上 fit asymmetric-loss 分类器；测 FN / FP rate。
- Required EP setting if later piloted: EP=8
- Expected positive signal: 在 margin=0.15·L*_ℓ 下达到 FN≤1%、FP≤20%。
- Falsification criterion: asymmetric-loss 训出的 boundary 与 fixed margin rule 差 ≤0.10·L*_ℓ —— 即 asymmetric loss 没有把 boundary 真正移动，本 idea 退化为 margin rule。
- Main baseline: 固定 0.15·L* margin rule。
- Main risk: GPT 担心 "what does the guard learn that 0.15·L* margin does not?" —— 必须落到 boundary 实际移动 ≥0.10·L* 才算系统贡献。
- Literature references:
  - Faster MoE LLM Inference for Extremely Large Models (Yang et al., 2025), verified=YES, https://arxiv.org/abs/2505.03531, 相关性：under varying service loads expert reduction；不是 safety-guard 形式。
  - MoE-CAP (Jiang et al., 2024), verified=YES, https://arxiv.org/abs/2412.07067, 相关性：MoE 系统 cost-accuracy benchmark；提供评估方法但不是 boundary guard。
- Claude self-score:
  - Novelty potential: 4
  - MLSys fit: 4.5
  - Mechanism clarity: 4.5
  - Feasibility on 8×RTX4090 EP=8: 4.5
  - Risk of being only a cost model: 2.5
  - Overall: 8.0
- GPT/Codex critique:
  - (a) Weak standalone —— 有用的工程 guard，但接近 `if L_recv < L* + margin: disable`。
  - (b) 部分 —— 产生 regression avoidance，不是新的正向加速来源。
  - (c) 部分 —— 决策与非对称失败代价清楚；"classifier" 不比确定性 margin 规则更清楚。
  - 最强 objection："这是给另一个 scheduler 加的 safety wrapper，不是 paper-level mechanism。"
  - 锋利问题：guard 学到 / 模型什么是 fixed 0.15·L* margin 没有的？
  - 撞车风险：Faster MoE LLM Inference；MoE-CAP。
- Claude response to GPT: GPT 的锋利问题是 idea 的 make-or-break 检验。如果 asymmetric-loss 训出的 boundary 与 fixed margin 一致，本 idea 塌成 engineering trick。我把这一点显式写入证伪准则：**asymmetric-loss boundary 与 symmetric / margin boundary 差 ≥0.10·L*** 才算成立。如果没差，drop。这把 GPT 的 worry 变成 yes/no 测试。撞车：两篇都不是 safety-guard 形式。
- Decision: **revise**（把 "boundary must materially differ from margin rule" 加入证伪；继续推进）

### D4-Idea-5: L*-aware admission queue with regime-conditional routing

- Dimension: 4 — L*-Aware Cost Model and Control Plane
- One-sentence thesis: serving 前端 admission queue 把请求按 *predicted regime* 分到两个队列 —— predicted comm-bound 的请求进 "drop-eligible" 队列，predicted latency-bound 进 "no-drop" 队列；queue 调度独立。
- Yes/No hypothesis: regime-routed admission 在异构 stream 上把 long-prompt e2e 维持 +12% 并把 short-prompt long-decode 的 p99 降 ≥10%。
- L*/L_recv role in the mechanism: regime 由 `predicted L_recv vs L*` 划分；前端用 D4-I1 + D4-I2 的输出。
- Control-plane inputs: prompt_len, max_tokens, current global byte budget, predicted L_recv envelope。
- Control-plane decision: 队列分配。
- Error/failure mode: 错路由 → 请求拿到错策略 → e2e 退化或 p99 增。
- Core mechanism: 入口分类器 → 两类队列 → 分别用 drop-on / drop-off 配置。
- Why this follows from the user finding: 长 prompt 与短 prompt 在 L_recv 分布上差 ~13×；前端分流避免 in-place 切换。
- What is new beyond current drop result: drop 决策从 dispatch 时点提前到 admission 时点。
- Minimal pilot sketch, no execution: 离线 —— 在 v6 traces 上模拟双队列；测 long e2e 与 short p99。
- Required EP setting if later piloted: EP=8
- Expected positive signal: 双队列分别维持 +12% 与 -10% p99。
- Falsification criterion: 错路由率 >10%；或 short queue 的 e2e 退化 ≥3%。
- Main baseline: 单队列 admission；与 D3-I5 (byte-budget admission) 区别在维度: D3-I5 是字节，本 idea 是 regime label。
- Main risk: 本 idea 与 D3-I5 部分重叠；差异化在 "regime classifier" 这一中间层。
- Literature references:
  - SCORPIO: Serving the Right Requests at the Right Time for Heterogeneous SLOs in LLM Inference (Tang et al., 2025), verified=YES, https://arxiv.org/abs/2505.23022, 相关性：heterogeneous-SLO admission；本 idea 在 regime classifier 上做 admission split。
- Claude self-score:
  - Novelty potential: 3
  - MLSys fit: 4.5
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 3.5
  - Risk of being only a cost model: 3
  - Overall: 7.0
- GPT/Codex critique: 本轮未入选 GPT review。
- Claude response to GPT: n/a
- Decision: keep（与 D3-I5 标差异化保留；低优先级）

### D4-Idea-6: `L*_ℓ`-anchored multi-lever policy bundle

- Dimension: 4 — L*-Aware Cost Model and Control Plane
- One-sentence thesis: 定义 4D policy 空间 (drop_r, K_eff, replica_budget, placement_variant)；对每个 `L*_ℓ` regime 离线算 Pareto-最优 bundle；运行时按 regime 选 bundle。
- Yes/No hypothesis: 离线 Pareto 表覆盖 ≥3 个 regime；运行时 bundle 选择 ≥85% 与 oracle 一致。
- L*/L_recv role in the mechanism: regime 由 `L_recv_ℓ vs L*_ℓ` 段划分；每段一组 bundle。
- Control-plane inputs: regime label per layer。
- Control-plane decision: per-layer (r, K, replica, placement_variant) bundle。
- Error/failure mode: bundle 不匹配 → 多 lever 同时次优。
- Core mechanism: 离线 Pareto-search；运行时查 bundle 表。
- Why this follows from the user finding: 多个 lever 在不同 regime 下贡献度差异巨大，bundle 选择必须 joint。
- What is new beyond current drop result: drop r 与其它 lever 现在独立调；本 idea 把它们绑定为 regime 条件下的联合最优。
- Minimal pilot sketch, no execution: 离线 —— 在 v6 grid 上 Pareto search；hold-out 验证 bundle 选择。
- Required EP setting if later piloted: EP=8
- Expected positive signal: ≥3 个 regime 各有显式不同的 bundle；hold-out 上 bundle 选择准确率 ≥85%。
- Falsification criterion: 所有 regime 的 Pareto 最优 bundle 都是同一个 —— bundle 思路无效。
- Main baseline: 独立调每个 lever。
- Main risk: Pareto 表组合爆炸；bundle 太多会 overfit。
- Literature references:
  - Vidur (MLSys 2024), verified=YES, https://arxiv.org/abs/2405.05465, 相关性：Vidur-Search 做 deployment config 搜索；不专门按 L* regime 分 bundle。
- Claude self-score:
  - Novelty potential: 3.5
  - MLSys fit: 5
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4
  - Risk of being only a cost model: 3
  - Overall: 7.5
- GPT/Codex critique: 本轮未入选 GPT review。
- Claude response to GPT: n/a
- Decision: keep（作为 D4-I1/I2 的下游应用候选）

### D4-Idea-7: Online `L*_ℓ` recalibration via Bayesian change-point detection

- Dimension: 4 — L*-Aware Cost Model and Control Plane
- One-sentence thesis: `L*_ℓ` 不平稳 —— KV 压力、NCCL backend 版本、副本数变化都让它漂；用 Bayesian online change-point detector 在滚动 (L_recv_ℓ, latency_ℓ) 窗口上更新 `L*_ℓ` 后验，posterior shift 概率超阈值时触发重校准。
- Yes/No hypothesis: change-point detector 在故意触发的 `L*_ℓ` shift（如改 replica 数）下 ≤50 步内检测到，平稳负载下假报率 ≤1 per 10⁴ steps。
- L*/L_recv role in the mechanism: `L*_ℓ` 是被 track 的非平稳变量；本 idea 提供其 online 更新机制。
- Control-plane inputs: 滚动窗口 (L_recv_ℓ, latency_ℓ) 对（最近 N 步）。
- Control-plane decision: 更新后的 `L*_ℓ` 后验 + change-point alert。
- Error/failure mode: 检测延迟 → controller 用 stale `L*_ℓ` K 步；假报 → 不必要的重校准 churn。
- Core mechanism: Adams-MacKay BOCPD 应用在 (L_recv, latency) 联合分布的斜率上。
- Why this follows from the user finding: user 当前的 `L*≈3271` 是一次性测出来的，没机制保证其在生产期不漂。
- What is new beyond current drop result: drop scheduler 当前在 stale `L*` 假设下跑；本 idea 显式追踪其漂移。
- Minimal pilot sketch, no execution: 离线 —— 在 v6 traces 上模拟 shift 注入；测检测延迟与假报率。
- Required EP setting if later piloted: EP=8
- Expected positive signal: 检测延迟 ≤50 步；假报率 ≤1 per 10⁴ steps。
- Falsification criterion: 检测延迟 >200 步；或假报率 ≥1 per 10³ steps。
- Main baseline: 静态 `L*_ℓ`（无在线追踪）。
- Main risk: GPT 锋利问题 —— 真 L* 漂移 vs workload-mix 漂移在 traces 上能否区分？
- Literature references:
  - Bayesian Online Changepoint Detection (Adams & MacKay 2007), verified=YES, https://arxiv.org/abs/0710.3742, 相关性：核心算法基础。
  - Vidur (MLSys 2024), verified=YES, https://arxiv.org/abs/2405.05465, 相关性：性能模型 simulation；不是 online 漂移追踪。
- Claude self-score:
  - Novelty potential: 4
  - MLSys fit: 4.5
  - Mechanism clarity: 3.5
  - Feasibility on 8×RTX4090 EP=8: 3.5
  - Risk of being only a cost model: 2.5
  - Overall: 7.5
- GPT/Codex critique:
  - (a) 更系统化 —— 非平稳 break-even 追踪是 legitimate control-plane 问题，但有沦为 "通用 changepoint detection 套在 MoE 上" 的风险。
  - (b) 部分 —— 收益是避免 stale-threshold 退化，不是改善稳态。
  - (c) 部分 —— rolling inputs / alerts 清楚；per-layer latency attribution 受 NCCL contention、batching、replicas、cross-layer overlap 混淆。
  - 最强 objection："已知的 online changepoint detection；MoE-specific novelty 只在监测的 scalar。"
  - 锋利问题：production traces 能否把真 `L*` 漂移与 workload-mix 漂移分开？
  - 撞车风险：Bayesian Online Changepoint Detection；Vidur。
- Claude response to GPT: GPT 的锋利问题是关键。**回答**：workload-mix 漂移会移动 L_recv 分布但不会移动 (L_recv → latency) 斜率；真 `L*` 漂移（如 NCCL 升级）会改这个斜率。本 idea 在 (L_recv, latency) **联合**分布上追踪斜率变化，而不是在 L_recv 边际上追踪 —— 这把 workload-mix 漂移过滤掉。我会把这一区分写进 mechanism 描述。撞车：BOCPD 是工具；本 idea 的贡献不是 BOCPD 本身，而是 "在 MoE 推理 (L_recv, latency) 斜率上应用它 + 与 workload mix 分离的 invariant"。
- Decision: **revise**（在联合 (L_recv, latency) 分布上追踪以分离 workload mix；继续推进）

### D4-Idea-8: `L_recv` uncertainty band — conformal-style decision

- Dimension: 4 — L*-Aware Cost Model and Control Plane
- One-sentence thesis: 不要把点估计与 `L*` 直接比；把 `L_recv` 在每层建模为分布，用 conformal prediction 导出 (mean, std-dev) 带；drop 仅在带的 *置信下界* > `L*_ℓ` 时启动。连接 D4-I1 (`L*_ℓ` 校准) 与 D4-I4 (don't-drop guard) 的不确定性量化器。
- Yes/No hypothesis: conformal-band 决策规则在 margin 0.1·L* 上达到 90% **drop-decision precision** —— Pr(L_recv > L*_ℓ | decided yes) ≥ 0.9 —— 且捕到 mixed stream 上 ≥80% 真 drop 机会。
- L*/L_recv role in the mechanism: `L_recv` 的分布与 `L*_ℓ` 的比较是核心决策；conformal band 量化决策的置信度。
- Control-plane inputs: per-step (L_recv_ℓ samples 跨层) → 经验带。
- Control-plane decision: 用下界 > `L*_ℓ` 来决定 drop 开关。
- Error/failure mode: 带太宽 → 保守度抹掉收益；带太窄 → 假阳性 → e2e 退化。
- Core mechanism: 在 calibration set 上拟合 conformal score；运行时用滑窗更新。
- Why this follows from the user finding: drop 决策的对错是非对称损失；点估计与硬阈值不能反映置信度。
- What is new beyond current drop result: drop 决策从确定性比较升级为概率保证。
- Minimal pilot sketch, no execution: 离线 —— 在 v6 上拟合 conformal；hold-out 上测 precision 与 recall。
- Required EP setting if later piloted: EP=8
- Expected positive signal: 90% precision；≥80% recall。
- Falsification criterion: 在任何带宽下都无法同时 ≥85% precision 和 ≥70% recall。
- Main baseline: 点估计与硬 `L*` 比较。
- Main risk: GPT 锋利问题 —— guarantee 是什么？我重新定位为 "drop-decision precision"，不是 L_recv coverage。
- Literature references:
  - Adaptive Conformal Inference Under Distribution Shift (Gibbs & Candès), verified=YES, https://arxiv.org/abs/2106.00170, 相关性：核心方法；本 idea 应用到 MoE break-even 决策。
  - Fast MoE Inference via Predictive Prefetching and Expert Replication (Jyothish et al., 2026), verified=YES, https://arxiv.org/abs/2605.11537, 相关性：predictive 角度邻近；不是 conformal。
- Claude self-score:
  - Novelty potential: 4
  - MLSys fit: 4.5
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4
  - Risk of being only a cost model: 3
  - Overall: 7.5
- GPT/Codex critique:
  - (a) 中等 —— uncertainty-aware boundary 比点阈值控制清楚，但仍是统计 wrapper 除非与 latency loss 绑定。
  - (b) 部分 —— 能减少坏激活；保守度可能抹掉 +12% 机会的大部分。
  - (c) 部分 —— 下界决策清楚；conformal score、calibration window、非平稳、per-layer 依赖未规范。
  - 最强 objection："L_recv 的 coverage 不是 latency-regret guarantee，特别是在自适应 serving stream 下。"
  - 锋利问题：到底保证什么 —— L_recv coverage、drop-decision precision、还是 bounded e2e latency regression？
  - 撞车风险：Adaptive Conformal Inference Under Distribution Shift；Fast MoE Inference via Predictive Prefetching。
- Claude response to GPT: 同意 GPT 的目标重定位。修订承诺把 guarantee 重定位为 **drop-decision precision**：Pr(L_recv > L*_ℓ | decided yes) ≥ 0.9，而不是 L_recv coverage 本身。这把 conformal 机器对准决策正确性，而不是底层变量准确性。如果有人想升级到 latency-regret guarantee，那是下一篇论文。撞车：Adaptive Conformal 是工具；本 idea 是它在 MoE break-even 决策上的应用。
- Decision: **revise**（guarantee 重定位为 drop-decision precision；继续推进）

### D4-Idea-9: Cost-model-as-API for downstream serving systems

- Dimension: 4 — L*-Aware Cost Model and Control Plane
- One-sentence thesis: 把 (`L*_ℓ`, byte cost model) 暴露为版本化 API，下游 serving system（vLLM、SGLang）在做调度决策前 query；贡献是 API surface 与 protocol，不只是模型本身。
- Yes/No hypothesis: API 设计可在 vLLM/SGLang 上以 ≤200 行 patch 接入，且改善 mixed workload e2e ≥3%。
- L*/L_recv role in the mechanism: `L*_ℓ` 是 API 输出之一；调用者用它做决策。
- Control-plane inputs: query (model, hardware, runtime, batch)。
- Control-plane decision: 预测的 (worst_peer_bytes, latency, regime)。
- Error/failure mode: API 误用 / 版本不一致 → 调用者拿到错信息。
- Core mechanism: REST/gRPC API + 版本化 schema + per-model `L*_ℓ` cache。
- Why this follows from the user finding: `L*_ℓ` artifact (D4-I1) 必须被消费才有用；API 是消费接口。
- What is new beyond current drop result: 没有任何当前 MoE serving system 暴露 L* / byte-cost 作为可 query 接口。
- Minimal pilot sketch, no execution: 离线 —— 设计 API spec；测 ≤200 行 patch 可接入 vLLM。
- Required EP setting if later piloted: EP=8
- Expected positive signal: vLLM/SGLang 上的接入 patch 极小；调用方拿到正确预测。
- Falsification criterion: 接入 patch >500 行；或 API surface 与各 serving system 内部假设冲突。
- Main baseline: 各 serving system 自己重写 cost model。
- Main risk: API 设计常被审稿人视为 engineering not research；需要 system-level integration story。
- Literature references:
  - Vidur (MLSys 2024), verified=YES, https://arxiv.org/abs/2405.05465, 相关性：通用 LLM perf simulator；可作为 reference API design。
- Claude self-score:
  - Novelty potential: 4
  - MLSys fit: 4
  - Mechanism clarity: 3.5
  - Feasibility on 8×RTX4090 EP=8: 3
  - Risk of being only a cost model: 4
  - Overall: 6.5
- GPT/Codex critique: 本轮未入选 GPT review。
- Claude response to GPT: n/a
- Decision: keep（作为 D4-I1 的接口层；低优先级）

### D4-Idea-10: `L*`-stationarity diagnostic for ops

- Dimension: 4 — L*-Aware Cost Model and Control Plane
- One-sentence thesis: 一个持续 monitor 检查实测 (L_recv, latency) 关系是否偏离校准曲线超出置信带；触发重校准或运维告警。
- Yes/No hypothesis: monitor 在 v6 traces 上对故意注入的 `L*` shift 的检测准确率 ≥90%、误报率 ≤5%。
- L*/L_recv role in the mechanism: `L*` 的平稳性是被监测对象。
- Control-plane inputs: telemetry 流。
- Control-plane decision: stationarity 判断。
- Error/failure mode: 漏检测 → silent regression。
- Core mechanism: 在 (L_recv, latency) 滚动窗口上做 KS-test 或 control chart。
- Why this follows from the user finding: 生产期 `L*` 不一定保持校准时的值。
- What is new beyond current drop result: 没有 MoE serving system 监测 `L*` 平稳性。
- Minimal pilot sketch, no execution: 离线 —— 模拟 shift 注入；测检测准确率。
- Required EP setting if later piloted: EP=8
- Expected positive signal: 检测准确率 ≥90%。
- Falsification criterion: ≤70%。
- Main baseline: 无监测。
- Main risk: 与 D4-I7 (online recalibration) 重叠 —— I10 是诊断，I7 是更新；可作为 I7 的可观测面。
- Literature references:
  - DriftBench: Measuring and Predicting Infrastructure Drift in LLM Serving Systems (Vitale, MLSys 2026), verified=YES, https://mlsys.org/virtual/2026/oral/3799, 相关性：generic LLM serving drift 监测；本 idea 是 MoE / L*-specific 应用。
- Claude self-score:
  - Novelty potential: 3.5
  - MLSys fit: 4
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4
  - Risk of being only a cost model: 3.5
  - Overall: 7.0
- GPT/Codex critique: 本轮未入选 GPT review。
- Claude response to GPT: n/a
- Decision: keep（作为 D4-I7 的可观测层；可能合并）

---

### Dimension 4 Summary

- Generated ideas: 10 (D4-I1 … D4-I10) 去重后。早期草稿合并："regime classifier" + "policy bundle" 拆为 D4-I3 + D4-I6 (分类器输出 vs Pareto 表)；I9/I10 拆为 API 暴露 vs 平稳性诊断。
- Top 5 reviewed by GPT (1 codex call, bundled): D4-I1, D4-I2, D4-I4, D4-I7, D4-I8。
- Kept (no major change): D4-I3, D4-I5, D4-I6, D4-I9, D4-I10。
- Revised (carried forward with explicit edits):
  - D4-I1: 升级为 **artifact + transferability invariant claim** —— `L*_ℓ` 跨硬件代际由 bandwidth / launch-overhead 比变化慢，作为科学 invariant 落实。
  - D4-I2: thesis 内显式回答 GPT 锋利问题 —— predicted `L_recv` 价值在于*避开 cross-rank reduction collective 进入决策关键路径*；router 输出可得但 reduction 仍需 a2a。
  - D4-I4: 加 falsification "asymmetric-loss boundary 与 fixed margin rule 差 ≥0.10·L*_ℓ" —— 不满足就 drop。
  - D4-I7: 在 (L_recv, latency) **联合**分布上追踪，分离 workload-mix 漂移 vs 真 L* 漂移。
  - D4-I8: guarantee 从 "L_recv coverage" 重定位为 **drop-decision precision** —— Pr(L_recv > L*_ℓ | decided yes) ≥ 0.9。
- Dropped: 无（D4-I9 / D4-I10 优先级低但保留）。
- Main GPT criticisms:
  1. 多个 idea 在 "包装阈值规则" 与 "系统 mechanism" 之间漂移；D4-I1 最像 metadata，D4-I4 最像 wrapper，D4-I8 最像统计帽子。
  2. 几条 idea 的 new measurable benefit 不清晰 —— 多半是 regression avoidance（避免坏 drop）而非新正向加速。
  3. 机制中心的 invariant / assumption hand-waved：D4-I1 的 transferability、D4-I2 的 router histogram 平稳性、D4-I7 的 latency attribution、D4-I8 的 conformal score 选择。
  4. 与现有 perf-modeling 工作的差异化压力：Vidur 占 dense LLM cost model 空间；MoE-CAP 占 MoE benchmark 空间；本维度必须把 MoE-specific `L*` 决策放在中心。
  5. D4-I2 的核心论证 ("dispatch-前为何需要预测") 必须显式写在 thesis 里。
- Main Claude–GPT disagreements:
  - On D4-I1: GPT 说"就是 autotuning metadata"；Claude 同意但反驳必须加 *transferability invariant* 才升级；分歧在 "是否可发表" —— Claude 认为带 invariant 可，GPT 倾向不可。
  - On D4-I2: GPT 锋利问题 "router 输出已可得为何预测"；Claude 答 "predicted L_recv 避开 reduction-on-critical-path"，已写入修订 thesis；分歧已对齐。
  - On D4-I7: GPT 担心 "MoE-specific novelty 只在监测的 scalar"；Claude 主张 (L_recv, latency) 联合分布的斜率追踪是 MoE-specific（dense LLM 没有这个量），不只是 BOCPD 套个壳；分歧仍存。
  - On D4-I8: GPT 问 "guarantee 是什么"；Claude 接受 reframe 为 drop-decision precision，分歧已对齐。
- Estimated Codex usage: 1 substantive Codex call this dimension (5 ideas bundled). Cumulative across D1+D2+D3+D4 = 7 substantive + 1 ping. User lifted budget mid-session; "top 5 / 1 round each" protocol kept as content rule.
- Next recommended dimension (do not execute): **Dimension 5**（brief 用户未指定）。本维度的产出（特别是 D4-I1 artifact + D4-I2 estimator + D4-I4 safety guard）是 D1（communication-centric）和 D2（phase-asymmetric）下游一切控制决策的基础；D5 若涉及 multi-node / 跨 EP 拓扑，会继承本维度的 L*_ℓ 概念。**等人工 gate 与下一维度 brief。**

---

### 维度 4 — 质检结果（QC pass, 2026-05-29）

QC 范围：对 D4 的 10 个 idea 重新做 (1) 硬性门 —— 可证伪、具体性、与 finding 关系（"a2a 是 MoE 推理真瓶颈"）、重复性；(2) 修订后的质量评分 —— novelty / story_fit（关键 severity：能 work 但 *不* 推进 "a2a-as-first-class" 主故事必须低分；纯 cost model 性质算）/ pilot_feasibility / pilot_cost（反向）；(3) 每个 idea 一句话赌注；(4) 对 top 3 调 GPT 挑战。不做撞车结论，不进入阶段 B。

GPT 挑战 trace：`docs/aris/traces/dimension4_qc_review.md`。下表中分数为 *GPT 之后*，与 GPT 的分歧记录在 trace 里。

#### QC 表

| idea_id | 硬性检查 | novelty | story_fit | feasibility | cost | 一句话赌注 | 标记 |
|---------|----------|---------|-----------|-------------|------|------------|------|
| D4-I1 | PASS（但 GPT loser） | 2 | 3 | 5 | 5 | L*_ℓ 跨 workload class 在 ≥80% 层上相对误差 ≤±15%，意味着 break-even 是可复用的硬件结构常量、可作为部署 artifact 而不是 workload-specific 偶然观察 | PASS-WEAK / LOSER（GPT 评为本轮 loser："just calibration metadata with inflated framing / 'first-class artifact' is branding, not a system idea"；transferability invariant 太弱、6 个 confounder 主导 "bandwidth differs <2×"；唯一价值是作为 D4-I2/I4/I7 的底层基础） |
| D4-I2 | PASS | 3 | 4 | 4.5 | 5 | router 输出虽在 dispatch 前可得但 L_recv 计算需要 cross-rank reduction（多一次 collective）；用 cached router skew histogram + microbatch composition 估计 L_recv 能在不引入 critical-path collective 的前提下 R²≥0.85，从而支撑 dispatch-前的所有 D4 控制决策 | PASS（top 1 post-GPT；GPT 提出最关键风险 "errors cluster near L* 时 R²=0.85 没用"，Claude 同意并升级证伪准则为 *near-boundary error rate*；需绑定到 D4-I4 作 actuator） |
| D4-I3 | PASS | 2.5 | 3.5 | 4 | 4.5 | MoE 推理的运行时状态可由 4 类 regime（compute / comm / idle / transition）充分描述，每类对应不同的主导 lever，按 regime 选 lever 比单 lever 多拿 ≥4% mean e2e | PASS（borderline；与 D4-I4 / D4-I6 在 within-D4 重叠 —— I4 是二值 regime guard、I6 = regime + bundle 查表；下阶段可考虑合并） |
| D4-I4 | PASS | 3 | 4 | 4.5 | 5 | asymmetric-loss 训出的 don't-drop 二值边界与对称 / fixed-margin 边界相差 ≥0.10·L*_ℓ —— 即非对称损失训出的边界本身就是研究贡献（不是 margin rule 的换装），并在 ≤1% FN 下捕到 ≥80% 真 drop 机会 | PASS（top 2 post-GPT；显式 falsification "boundary must materially differ from margin rule" 把 GPT 的核心 worry 变成 yes/no 测试；cost-model-only 风险最低） |
| D4-I5 | PASS-WEAK | 2.5 | 3.5 | 3.5 | 3.5 | 把请求按 predicted regime 分到两个独立 queue（drop-eligible vs no-drop），比单 queue + 内部 dispatch-time 决策能在异构 stream 上更稳定地避免 latency-bound 误激活 drop | PASS-WEAK（cross-dimension 与 D3-I5 byte-budget admission 重叠；本 idea 区别只在 "regime label 而非 byte budget"；story_fit 间接） |
| D4-I6 | PASS | 3 | 4 | 4 | 3.5 | MoE 4D policy 空间 (r, K_eff, replica, placement) 在 ≥3 个 L_recv regime 下的 Pareto 最优 bundle 互不相同，按 regime 查 bundle 与 oracle ≥85% 一致 | PASS（within-D4 与 D4-I3 重叠 —— I3 只给 regime label，I6 = I3 + bundle 表；下阶段可合并为 "regime → bundle" 单一 mechanism） |
| D4-I7 | PASS | 3 | 3.5 | 3.5 | 3.5 | L* 的真实漂移由 (L_recv, latency) 联合分布斜率变化捕获、可与 workload-mix 漂移（仅移动 L_recv 边际而不改斜率）分开；BOCPD 在联合斜率上 ≤50 步检出 shift、≤10⁻⁴ 假报率 | PASS（GPT 接受 "joint slope" 角度但批 "platform drift 不是唯一斜率移动者"；6 个 confounder 列入必做 ablation；novelty 3.5→3、story_fit 4→3.5） |
| D4-I8 | PASS | 3 | 4 | 4 | 4 | 把 drop 决策的 guarantee 从 "L_recv coverage" 重定位为 "drop-decision precision"（Pr(L_recv > L*_ℓ \| decided yes) ≥ 0.9），conformal 机器在保持 ≥80% recall 下把误激活 drop 概率约束在 ≤10% | PASS（top 3 post-GPT；guarantee 重定位是最清晰的回应；novelty 3 story_fit 4 双方对齐） |
| D4-I9 | PASS-WEAK | 3 | 3 | 3 | 2.5 | 把 L*_ℓ + byte cost model 暴露为版本化 API，vLLM/SGLang 能以 ≤200 行 patch 接入并改善 mixed workload e2e ≥3% —— 即 API surface 是值得做的系统层贡献 | PASS-WEAK（API 设计很少作 systems 主贡献；story_fit 间接；novelty 在接口而非 mechanism） |
| D4-I10 | DUP（与 D4-I7 重叠） | — | — | — | — | KS-test / control chart 在 (L_recv, latency) 滚动窗口能 ≥90% 准确检测到 L* shift —— 不需在线更新即可避免 silent regression | DUP（作者本人已注 "与 D4-I7 重叠 —— I10 是诊断、I7 是更新 + 告警；可作为 I7 的可观测层"；I10 ⊂ I7 的能力子集，应合并） |

REJECT / DUP 的具体原因：
- **D4-I10 DUP**：与 D4-I7 共用 "在 (L_recv, latency) 滚动窗口上检测 L* shift" 同一 mechanism。I7 = detection + 在线更新 + 告警；I10 = detection-only 诊断。作者本人已承认 "可能合并"。I10 是 I7 的能力子集，独立计数无意义。
- 无 REJECT（所有 idea 都满足可证伪 + 具体性 + 与 finding 关系；硬性门通过）。

#### Post-GPT top 2–3（按 novelty + story_fit 排序，PASS 项）

GPT 挑战后 D4-I1 跌出 top 3（被挑为 loser），D4-I2 仍 #1 但 score 从 8.5 降到 7。Re-rank：

- **Top 1 — D4-I2 (7.0)**：L_recv estimator from microbatch composition + router histogram。GPT 最强批判是 "errors clustering near L*"，Claude 接受并升级证伪准则为 *near-boundary error rate*；需与 D4-I4 绑定作 actuator。
  - 一句话赌注：router 输出虽在 dispatch 前可得但 L_recv 计算需要 cross-rank reduction（多一次 collective）；用 cached router skew histogram + microbatch composition 估计 L_recv 能在不引入 critical-path collective 的前提下 R²≥0.85，且 near-boundary error rate ≤5%，从而支撑 dispatch-前的所有 D4 控制决策。
- **Top 2 — D4-I4 (7.0)**：one-way "don't-drop" safety guard。Cost-model-only 风险最低（2.5）；falsification 把 GPT 的 "guard learns nothing" 担忧变成 yes/no 测试（asymmetric boundary ≥0.10·L* 远离 margin rule）。
  - 一句话赌注：asymmetric-loss 训出的 don't-drop 二值边界与对称 / fixed-margin 边界相差 ≥0.10·L*_ℓ —— 即非对称损失训出的边界本身就是研究贡献（不是 margin rule 的换装），并在 ≤1% FN 下捕到 ≥80% 真 drop 机会。
- **Top 3 — D4-I8 (7.0)**：conformal-band drop-decision。Guarantee 重定位为 drop-decision precision 是最清晰的回应；与 D4-I2 (估计) + D4-I4 (guard) 在统计-决策层互补。
  - 一句话赌注：把 drop 决策的 guarantee 从 "L_recv coverage" 重定位为 "drop-decision precision"（Pr(L_recv > L*_ℓ | decided yes) ≥ 0.9），conformal 机器在保持 ≥80% recall 下把误激活 drop 概率约束在 ≤10%。

候选 #4 = D4-I6 (7.0)：regime-anchored bundle；与 D4-I3 重叠待合并。
候选 #5 = D4-I7 (6.5)：online recalibration；GPT 6-confounder 列表需 ablation。
LOSER = D4-I1 (5.0)：被 GPT 挑出，"calibration metadata with inflated framing"；作为 D4-I2/I4 的底层基础保留但不作 standalone 贡献。

#### Claude–GPT 评分分歧总结

| idea | Claude pre-GPT (novelty, story_fit) | GPT 挑战方向 | Claude post-GPT (novelty, story_fit) | 是否仍分歧 |
|------|-------------------------------------|--------------|--------------------------------------|------------|
| D4-I2 | 3.5, 5 | novelty 高（MoE-Infinity + 通用 router-load estimation）；story_fit "borderline empty" 除非绑定 actuator；最强风险 "errors cluster near L*" 让 mean R² 无用 | 3, 4 | 部分对齐；分歧点：Claude 主张 near-boundary error rate ≤5% 的承诺可挽救 mean R² 的局限，GPT 仍倾向 "需要 actuator 实证才不算 instrumentation" |
| D4-I1 | 3, 4.5 | 双双过高；属"decades of profiling tables / TVM/Ansor / TensorRT metadata" 空间；transferability invariant 被 6 个 confounder 主导；挑为 loser | 2, 3 | 大体认输；分歧点：Claude 主张 L*_ℓ 作为 D4-I2/I4/I7 的底层 substrate 仍有保留价值；GPT 倾向 "不可作 standalone contribution" |
| D4-I7 | 4, 4 | novelty 高（BOCPD + 通用 drift detection）；workload-mix 不是唯一斜率移动者（overlap / queueing / kernel contention / token shape / cache pressure / expert imbalance）；新收益只在长期部署 | 3, 3.5 | 部分对齐；分歧点：Claude 主张 6-confounder ablation 进 pilot 可挽救 disambiguation 主张；GPT 倾向 "即使完美 drift detection 仍不指明哪根 lever" |

#### QC 结论（维度 4）

- 总数：10
- PASS：9（D4-I1, I2, I3, I4, I5, I6, I7, I8, I9）；其中：
  - **D4-I1 PASS-WEAK / LOSER** —— GPT 挑出 "branding not contribution"；novelty 3→2、story_fit 4.5→3 双降；仅作 D4-I2/I4/I7 的底层 substrate 保留；
  - **D4-I5 PASS-WEAK** —— cross-dimension 与 D3-I5 重叠；
  - **D4-I9 PASS-WEAK** —— API 设计不是 systems 主贡献；
  - **D4-I3 + D4-I6 within-D4 重叠** —— I3 是 regime label，I6 = I3 + bundle 查表；下阶段建议合并为 "regime → bundle" 单一 mechanism；
  - **D4-I7 GPT 批 confounder 多**，joint-slope 主张需 6-confounder ablation 才稳。
- REJECT：0
- DUP：1（D4-I10 与 D4-I7 共用 detection mechanism；I10 ⊂ I7）
- Top 2-3：D4-I2 → D4-I4 → D4-I8（D4-I1 被挑为 loser、跌出 top 3；D4-I7 候补 #5、D4-I6 候补 #4）
- Codex 本 QC 轮额外消耗：1 次（GPT 挑战 top 3 一并打包，high reasoning）。D4 累计：idea critique 1 + QC 1 = 2 substantive Codex calls。
- 跨 D1+D2+D3+D4 全部累计：8 次 substantive + 1 次连通性 ping，远低于 15 次告警阈值。用户解除 GPT 调用限制的消息已收到，"top 3 / 1 round each" 是内容规则继续保留。
- 维度结束，等待人工 gate。

---

## Dimension 5 — Minimal Deepening, Not Component Stacking

本维度纪律：**做减法，不做加法**。每个 idea 必须 (1) 围绕一个 dominant mechanism；(2) 显式声明 "this does NOT add" 以拒绝堆 drop+placement+predictor+replica 四件套；(3) 把已有 finding 升级成 *新概念 artifact*（proof / lower bound / theorem / abstraction），而不只是 "再做一次 +12% / ±2%"。重点是研究品味：少组件、强机制、强可证伪。

维度 5 自评分规则：Novelty potential、MLSys fit、Mechanism clarity、Feasibility on 8×RTX4090 EP=8、**Simplicity/depth**（分数越高越少组件越深）、**Risk of component stacking**（分数越高越像加法）、Overall (1–10)。

### D5-Idea-1: Causal-isolation microbench for "bytes ⇒ latency"

- Dimension: 5 — Minimal Deepening, Not Component Stacking
- One-sentence thesis: 建一个单一 microbench，通过 padding 单独变化 wire bytes，同时把 L_recv、expert compute、routing 分布固定为常量；干净建立 *bytes → latency* 的因果斜率，分离 drop 的所有混淆。
- Yes/No hypothesis: per-byte 斜率在 r ∈ {0, 0.1, 0.2, 0.3, 0.5} 下恒定 ±5%。
- Dominant mechanism: pad-only 单一 microbench。
- What this deliberately does NOT add: 不新增 drop 变体；不动 placement / replica / predictor。
- Why this follows from the user finding: 现有 +12% 与 segment ablation 是 *相关* 证据；本 idea 是 *因果* 证据。
- What is new beyond current drop result: 把 "97% 来自 bytes" 升级为因果断言。
- Minimal pilot sketch, no execution: 离线设计 microbench 协议；在已有 L-sweep 数据上拟合。
- Required EP setting if later piloted: EP=8
- Expected positive signal: 斜率 ±5%。
- Falsification criterion: 斜率方差 >20% → bytes 不是因果，存在其它支配 confound。
- Main baseline: 现有 segment ablation 的相关性证据。
- Main risk: microbench 设计难以完全固定 L_recv（拓扑实际触发的 reduce 不可同时不变）。
- Literature references:
  - Roofline (Williams et al., CACM 2009), verified=NO [MEMORY-NEEDS-VERIFY], 相关性：roofline 用 microbench 隔离 bandwidth-bound vs compute-bound 是本 idea 的方法论祖先。
- Claude self-score:
  - Novelty potential: 3
  - MLSys fit: 4
  - Mechanism clarity: 5
  - Feasibility on 8×RTX4090 EP=8: 5
  - Simplicity/depth: 4.5
  - Risk of component stacking: 1
  - Overall: 7.5
- GPT/Codex critique: 本轮未入选 GPT review。
- Claude response to GPT: n/a
- Decision: keep（作为 D5-I2/I3 因果前置）

### D5-Idea-2: Drop ↔ byte-conservative router-perturbation equivalence

- Dimension: 5 — Minimal Deepening, Not Component Stacking
- One-sentence thesis: 证明（并经验支持）所有 drop policy 等价于 router 上的某种随机扰动，该扰动在分布上保留 **加权 expert 输出期望** —— 整个 drop 家族塌成一个概念 "byte-conservative router perturbation"，drop 变体之间的差别只在扰动分布。
- Yes/No hypothesis: 在 v6 traces 上，按扰动分布预测的 per-layer 输出 L2 范数与实际 drop 输出 L2 范数 R²≥0.9。
- Dominant mechanism: 等价证明 + 经验验证。
- What this deliberately does NOT add: 不新增 drop 变体；不新增 placement；不新增 predictor；甚至不动 router 实现。
- Why this follows from the user finding: 97% 收益来自 bytes，说明 drop 行为可由 "byte 维度上的扰动" 完全刻画；本 idea 是这一直觉的形式化。
- What is new beyond current drop result: 把多个 drop 变体的实证比较升级为单一 equivalence class 的数学刻画。
- Minimal pilot sketch, no execution: 离线 —— 在 v6 traces 上对每个 drop 变体推导其扰动分布参数；预测 L2 范数；与实测比较。
- Required EP setting if later piloted: EP=8
- Expected positive signal: R²≥0.9。
- Falsification criterion: 等价在 >20% 层上失败 → drop 不能由单一扰动家族完整刻画。
- Main baseline: 经验 drop 变体比较（v6 grid）。
- Main risk: GPT 锋利问题：什么 invariant 真正保留？必须钉死 **加权 expert 输出期望**（residual stream contribution），而不是 expert identity 分布。
- Literature references:
  - Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer (Shazeer et al., 2017), verified=YES, https://arxiv.org/abs/1701.06538, 相关性：noisy top-k routing 是 router-perturbation 的训练侧版本；本 idea 是其推理侧 + drop 等价的形式化扩展。
  - Capacity-Aware Inference (He et al., 2025, ICLR 2026), verified=YES, https://arxiv.org/abs/2503.05066, 相关性：推理期 token drop；不涉及 router-perturbation 等价。
  - Turn Waste into Worth: Rectifying Top-k Router of MoE (Zeng et al., 2024), verified=YES, https://arxiv.org/abs/2402.12399, 相关性：drop/padding 的实践修正；不讨论等价。
- Claude self-score:
  - Novelty potential: 4
  - MLSys fit: 4
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4
  - Simplicity/depth: 5
  - Risk of component stacking: 1
  - Overall: 8.0（post-GPT 从 8.5 下调）
- GPT/Codex critique:
  - (a) 减法：YES，如果整篇论文就是 equivalence class。
  - (b) 新 artifact：部分 —— 真正的等价定理是新的；"drop = stochastic mask/noisy router" 接近 tautology 除非在 top-k renormalization 下保留 weighted expert output。
  - (c) MLSys-level：部分 —— 理论-only 可以，但 output L2 R² 是 weak systems artifact。
  - 最强 objection："每个 drop policy 都是 router 扰动" 要么对任意 policy 是假的，要么是 vacuously 成立。
  - 锋利问题：到底什么 invariant —— expert identity 分布、weighted expert output 期望、还是 final residual stream 分布？
  - 撞车风险：Shazeer (1701.06538)；Capacity-Aware Inference。
- Claude response to GPT: 接受 novelty 4.5→4 与 story_fit 5→4.5。锋利问题的答案：**weighted expert output expectation**（即 Σ_e g_e,t · o_e,t 的 distributional invariance），而不是 expert identity 分布或 final residual stream。这是 non-trivial 的：top-k renormalization 后 g 的分布会变，不是所有扰动都保留 g·o 期望，所以等价类是有界的。Output L2 R² 作为 *系统侧* 证据弱我同意 —— 必须升级到 *任务 metric*（accuracy / perplexity）级的 invariance test 才有 MLSys 力度。
- Decision: **revise**（钉死 invariant = weighted expert output expectation；把 R² 测试从 L2 升级到 task metric；继续推进）

### D5-Idea-3: Information-theoretic lower bound on per-step a2a bytes

- Dimension: 5 — Minimal Deepening, Not Component Stacking
- One-sentence thesis: 推导每层每步要在 receiver 侧把 expert 输出重建到 ε-精度所需的最小 a2a bits（rate-distortion 风格的下界）；与 v6 traces 上观测到的实际 bytes 比，量化系统的 "浪费带宽"。
- Yes/No hypothesis: 观测 bytes 在 K=8 EP=8 prefill, L_recv=4k 上超过下界 ≥10×。
- Dominant mechanism: 信息论下界。
- What this deliberately does NOT add: 不新增系统；不新增 lever。
- Why this follows from the user finding: drop 把 wire bytes 减到 47–50%，但与最优距离多远？下界回答这个。
- What is new beyond current drop result: 把 +12% / +20% 升级为相对绝对最优的距离表。
- Minimal pilot sketch, no execution: 推导 rate-distortion bound；在 v6 traces 上算实际 bytes；比较。
- Required EP setting if later piloted: EP=8
- Expected positive signal: 实际 / 下界 ≥10×。
- Falsification criterion: 比值 ≤2× → 系统已经接近最优，drop 收益接近 max；本 idea 的 "浪费" 主张不成立。
- Main baseline: 无 — 本 idea 提供 baseline。
- Main risk: GPT 锋利问题：被下界的随机变量是 dispatch activations / expert outputs / router decisions / task loss 中的哪个？必须明确 —— 我承诺是 **expert 输出在 receiver 端 L2-ε 重建所需 bits**。
- Literature references:
  - LLM Inference Unveiled: Survey and Roofline Model Insights (Yuan et al., 2024), verified=YES, https://arxiv.org/abs/2402.16363, 相关性：roofline-style 分析；不专门做 MoE a2a 下界。
  - Rate Distortion For Model Compression (Gao et al., 2018), verified=YES, https://arxiv.org/abs/1810.06401, 相关性：核心方法 (rate-distortion for compression)；本 idea 是其在 MoE a2a 上的应用。
  - LatentMoE (verified=YES), https://arxiv.org/abs/2601.18089, 相关性：MoE accuracy-vs-cost limits；不专门做 a2a 字节下界。
- Claude self-score:
  - Novelty potential: 3.5（post-GPT 从 4 下调，rate-distortion 已知）
  - MLSys fit: 4
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4
  - Simplicity/depth: 5
  - Risk of component stacking: 1
  - Overall: 7.5
- GPT/Codex critique:
  - (a) 减法：YES。
  - (b) 新 artifact：YES，若数学非平凡且 tight 到能解释 traces。
  - (c) MLSys-level：部分 —— 松的 Shannon 下界会被当 obvious gap theater。
  - 最强 objection：token / router 熵不决定保留非线性 expert 输出所需 bits；真正的对象是 remote activation transformation，不是 token identity。
  - 锋利问题：下界的对象是什么随机变量？
  - 撞车风险：Rate-distortion model compression (1810.06401)；LatentMoE (2601.18089)。
- Claude response to GPT: 完全同意 GPT 的 framing。被下界的对象明确为 **expert 输出在 receiver 端 L2-ε 重建所需的 bits**（不是 token entropy，不是 router entropy）。这是 rate-distortion 在 *跨 GPU 跨非线性变换* 上的应用，不只是经典 Shannon。撞车：Gao et al. 2018 是工具基础；LatentMoE 是 MoE cost-accuracy 但不专门做 a2a 字节下界。novelty 在 *把 rate-distortion 套在 MoE a2a 上 + 实证 traces 上的浪费量化*。
- Decision: **revise**（被下界对象钉死为 expert 输出重建 bits；与 rate-distortion 工具基础显式区分；继续推进）

### D5-Idea-4: Single-knob byte-budget runtime (subsumes drop / K_eff / quant)

- Dimension: 5 — Minimal Deepening, Not Component Stacking
- One-sentence thesis: 用 **一个** 控制变量 "target bytes per layer per rank" 替代 (drop %, K_eff, dispatch FP precision) 三个 lever；runtime 内部按 budget 选最便宜的组合。贡献是 **统一**，不是新 lever。
- Yes/No hypothesis: 单旋钮 runtime 在每个 budget 点上与 3-knob Pareto 前沿误差 ≤2% e2e。
- Dominant mechanism: 单一 byte-budget 控制变量。
- What this deliberately does NOT add: 不新增 drop policy；不新增 quantization；不新增 K_eff 变体。
- Why this follows from the user finding: 97% bytes 主导意味着这三个 lever 的 e2e 影响应该 *投影*到字节轴上 —— 如果真的，单旋钮就够。
- What is new beyond current drop result: drop / K_eff / quant 当前独立调；本 idea 主张它们是同一字节轴的投影。
- Minimal pilot sketch, no execution: 离线 —— 在 v6 grid 上模拟单旋钮的最优组合；与 3-knob Pareto 对比。
- Required EP setting if later piloted: EP=8
- Expected positive signal: 任何 budget 点误差 ≤2%。
- Falsification criterion: 任何 budget 点误差 >5% → 三 lever 不是字节轴的投影、抽象失败。
- Main baseline: 3 独立 lever 的 Pareto sweep。
- Main risk: GPT 锋利问题：移走 3 lever 之后，剩下的是 theorem 还是 packaging？我承诺 falsification ≤5% 严格 —— 如果 ≤5% 成立，*三个 lever 不独立* 才是 theorem。
- Literature references:
  - Dynamic top-K MoE routing (2024), verified=YES, https://arxiv.org/abs/2403.07652, 相关性：K_eff 动态调整；不与 quant + drop 联合统一。
  - FP8 post-training quantization (2023), verified=YES, https://arxiv.org/abs/2309.14592, 相关性：单 lever 工作；不与 drop + K_eff 联合。
  - Capacity-Aware Inference (2503.05066), verified=YES via D1 trace, 相关性：drop 角度。
- Claude self-score:
  - Novelty potential: 3（post-GPT 从 4 下调，GPT 说是 disguised stacking）
  - MLSys fit: 4
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 3.5
  - Simplicity/depth: 4（post-GPT 从 5 下调）
  - Risk of component stacking: 3.5（post-GPT 从 1.5 上调 —— 是 disguised stacking 风险高）
  - Overall: 6.5（post-GPT 从 8.0 下调）
- GPT/Codex critique:
  - (a) 减法：**否**。三 lever 套一个旋钮，是 disguised stacking。
  - (b) 新 artifact：部分 —— "byte budget" 是干净抽象，但 Pareto matching 是 controller 包装，不是概念。
  - (c) MLSys-level：否，如所写 —— 像 autotuner over 已知 knobs。
  - 最强 objection：dominant mechanism 不是 byte budget，是 drop/K/quant 在每点 winning 的那个。
  - 锋利问题：移走 3 lever 之后，剩下什么 theorem 或 abstraction？
  - 撞车风险：Dynamic Top-K MoE (2403.07652)；FP8 PTQ (2309.14592)。
- Claude response to GPT: 这是本轮 GPT 最锋利的批评。承认风险：如果 ≤2% Pareto 误差成立，本 idea 才有 theorem（三 lever 是字节轴的投影）；如果不成立，本 idea 就是 disguised stacking。我把这一点变成 **核心 yes/no** —— Pareto 误差 ≤2% 就保留 thesis，否则 drop。这是非常风险高的 thesis：可能 fail。考虑降级到 D5 候选 #5。
- Decision: **revise → 降级**（保留但降为低优先级；明确告知"Pareto-match 是 make-or-break"；如果初步实验 fail 就立即 drop）

### D5-Idea-5: Replica-minimality above L*

- Dimension: 5 — Minimal Deepening, Not Component Stacking
- One-sentence thesis: 一旦 L_recv > L*，再加副本对 a2a-bound 延迟无改善（瓶颈是字节而非争用）；comm-bound 区域内 replica budget 应该设为防 decode OOM 的最小值，而**不是**为 prefill 负载平衡的最大值。这把 MoE replica 设计直觉在 L* 上方**倒转**。
- Yes/No hypothesis: 在 v6 traces 上 L_recv > L* 的层段中，MOE_DROP_MIN_REPLICAS 从 128 改到 1024 的 e2e 差 ≤2%。
- Dominant mechanism: comm-bound regime 下副本不相关的经验断言。
- What this deliberately does NOT add: 不新增 placement 算法；不新增 replication 策略。
- Why this follows from the user finding: 97% 字节主导 → 副本只在 *contention bound* 时才有用 → 字节 bound 下副本就是占 VRAM。
- What is new beyond current drop result: 现有 placement 工作都假设副本越多越好；本 idea 在 comm-bound 区段提出反向规则。
- Minimal pilot sketch, no execution: 离线 —— 用 v6 grid 比较 MIN_REPLICAS=128 vs 1024 在 L_recv > L* 段的 e2e。
- Required EP setting if later piloted: EP=8
- Expected positive signal: e2e 差 ≤2%。
- Falsification criterion: 差 ≥5% → 副本在 comm-bound 段也有效，主张错。
- Main baseline: 默认 MIN_REPLICAS=512（user 生产）。
- Main risk: GPT 锋利问题 —— L* rules out replica benefit in topology/queueing terms or only payload-byte? 我承诺把主张窄化为 *payload-byte-only*：local NCCL contention / hotspot 在 comm-bound 段已经被 drop 缓解，因此进一步加副本不会显著降字节侧延迟。
- Literature references:
  - CRAFT: Cost-aware Expert Replica Allocation with Fine-Grained Layerwise Estimations (Zhao et al., 2026), verified=YES, https://arxiv.org/abs/2603.28768, 相关性：副本分配；本 idea 是其反向 regime law。
- Claude self-score:
  - Novelty potential: 3（post-GPT 从 3.5 下调）
  - MLSys fit: 4.5
  - Mechanism clarity: 4.5
  - Feasibility on 8×RTX4090 EP=8: 4.5
  - Simplicity/depth: 5
  - Risk of component stacking: 1
  - Overall: 7.0（post-GPT 从 8.0 下调）
- GPT/Codex critique:
  - (a) 减法：YES —— 把副本热情移除。
  - (b) 新 artifact：部分 —— "replica-minimality above L*" 是 regime law，但只在它不是单系统观察时才有意义。
  - (c) MLSys-level：部分 —— 品味强；硬件/模型依赖窄是危险。
  - 最强 objection：副本能改变 locality / queueing / hotspot contention；"bytes 主导" 不能证 "副本无关"。
  - 锋利问题：L* 排除副本收益是在拓扑/队列层面，还是在纯 payload-byte 层面？
  - 撞车风险：vLLM EPLB；CRAFT。
- Claude response to GPT: 完全同意。**把主张窄化** 为 "payload-byte-only" 而不是 "any benefit"：在 L_recv > L* 区段，drop 已经把字节侧瓶颈缓解到 communication-bound 的甜点；进一步加副本会再降字节但其边际不显著（这是可证伪的）。Locality/queueing 仍可能 benefit，但那是另一篇文章。撞车：vLLM EPLB / CRAFT 都是 "副本越多越好"；本 idea 提供反向 regime law。
- Decision: **revise**（窄化为 payload-byte-only regime law；继续推进）

### D5-Idea-6: Canonical config from L*-distance

- Dimension: 5 — Minimal Deepening, Not Component Stacking
- One-sentence thesis: v6 grid 的所有 production-positive 单元（5 drop policy × 4 rate × 2 plan × 3 dataset）能用 *单一* 由 L*-distance 索引的 canonical config 良好近似 —— 即所有 +12% 结果实际都是同一 (drop policy, r) 家族的不同 L*-距离投影。
- Yes/No hypothesis: 单参数家族（按 L*-distance 索引）解释 ≥90% v6 e2e 改善方差。
- Dominant mechanism: 单参数家族 + L*-distance 投影。
- What this deliberately does NOT add: 不新增配置；不新增 lever。
- Why this follows from the user finding: 97% 字节主导意味着配置空间应该塌成字节轴。
- What is new beyond current drop result: 把 v6 grid 实证升级为单参数刻画。
- Minimal pilot sketch, no execution: 离线 —— 在 v6 grid 上拟合 L*-distance 家族；测方差解释。
- Required EP setting if later piloted: EP=8
- Expected positive signal: R² ≥ 0.9。
- Falsification criterion: R² ≤ 0.6 → 配置不能塌成单轴。
- Main baseline: 多维 grid 实证。
- Main risk: 与 D5-I2 / D5-I4 部分重叠（都是字节轴的简化主张），但更经验、更窄。
- Literature references: 无独立 references（基于 user finding 内部分析）。
- Claude self-score:
  - Novelty potential: 3
  - MLSys fit: 4
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 5
  - Simplicity/depth: 4.5
  - Risk of component stacking: 1.5
  - Overall: 7.0
- GPT/Codex critique: 本轮未入选 GPT review。
- Claude response to GPT: n/a
- Decision: keep（与 D5-I2 / D5-I4 重叠风险高；作为后者的支撑实证）

### D5-Idea-7: K_eff failure model (analytical reason for negative leverage)

- Dimension: 5 — Minimal Deepening, Not Component Stacking
- One-sentence thesis: 用 *分析模型* 解释为何 K_eff 硬截在 user 的设置下是负杠杆 —— GEMM 减少同时 byte-per-token 因 renormalization 上升 / per-launch overhead 占比上升 —— 给 2.8% finding 一个干净解释，并预测何时 K reduction 会 *真正* 帮忙。
- Yes/No hypothesis: 分析模型在 ±20% 内预测 user 实测的 K_eff 硬截 e2e 退化幅度。
- Dominant mechanism: K_eff → byte-per-token + per-launch overhead 分析模型。
- What this deliberately does NOT add: 不新增 K 变体；不新增 routing。
- Why this follows from the user finding: K_eff 是失败的 lever；解释失败本身就是贡献。
- What is new beyond current drop result: K_eff 当前作为 "已知失败" 列在 finding 里；本 idea 给定量解释。
- Minimal pilot sketch, no execution: 离线 —— 用 v6 K_eff 硬截数据拟合分析模型。
- Required EP setting if later piloted: EP=8
- Expected positive signal: 预测误差 ±20%。
- Falsification criterion: 误差 >50% → 分析模型不抓住主因。
- Main baseline: 经验观察 "K_eff 硬截负杠杆"（无解释）。
- Main risk: 解释性论文 MLSys 接受度低于 mechanism 论文。
- Literature references: 无独立 references。
- Claude self-score:
  - Novelty potential: 3
  - MLSys fit: 4
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4.5
  - Simplicity/depth: 4
  - Risk of component stacking: 1.5
  - Overall: 7.0
- GPT/Codex critique: 本轮未入选 GPT review。
- Claude response to GPT: n/a
- Decision: keep（作为 D5-I3 信息下界的具体应用）

### D5-Idea-8: Byte-equivalence classes on drop variants

- Dimension: 5 — Minimal Deepening, Not Component Stacking
- One-sentence thesis: 在 drop policy 上定义 byte-equivalence 类：两 policy byte-equivalent ⇔ 它们在每层产生同分布的 wire bytes。证明同类内 e2e 延迟 ±2%、精度 ≥5pp 变化 —— 即 drop variant 的选择是 **纯精度-质量 lever**，延迟由字节决定。
- Yes/No hypothesis: 同类内 e2e 方差 ≤2%、精度方差 ≥5pp。
- Dominant mechanism: 等价类定义 + 经验证明。
- What this deliberately does NOT add: 不新增 drop；不新增 placement。
- Why this follows from the user finding: 97% 字节主导意味着延迟轴和精度轴在 drop 上正交。
- What is new beyond current drop result: 把 drop variant 比较升级为 "在等字节预算下的精度 Pareto"。
- Minimal pilot sketch, no execution: 离线 —— 在 v6 grid 上识别 byte-equivalent 单元；测延迟方差 vs 精度方差。
- Required EP setting if later piloted: EP=8
- Expected positive signal: 等字节预算下延迟方差 ≤2%、精度方差 ≥5pp。
- Falsification criterion: 延迟方差 >5% → drop variant 影响 latency 而不只是 bytes。
- Main baseline: v6 multi-variant grid。
- Main risk: 与 D5-I2 等价定理重叠（D5-I2 是 router-perturbation 等价、D5-I8 是 byte-distribution 等价）。
- Literature references: 与 D5-I2 共享。
- Claude self-score:
  - Novelty potential: 4
  - MLSys fit: 4
  - Mechanism clarity: 4.5
  - Feasibility on 8×RTX4090 EP=8: 4
  - Simplicity/depth: 5
  - Risk of component stacking: 1
  - Overall: 8.0
- GPT/Codex critique: 本轮未入选 GPT review。
- Claude response to GPT: n/a
- Decision: keep（与 D5-I2 互补 —— I2 抽象 router 侧，I8 抽象 byte 侧）

### D5-Idea-9: Decode-hopelessness theorem

- Dimension: 5 — Minimal Deepening, Not Component Stacking
- One-sentence thesis: 证明：在 *固定批次单元* (T per step) 下，给定 EP=8、K=8 的解码 L_recv ≈ K·T/EP 结构性低于 L*；因此任何 a2a 字节减少算子在解码上不可能产生正 e2e。把 user 的 GSM8K ±2% 从 "puzzling failure" 重写为 "predicted theorem"，并预测 speculative decoding / MTP 何时把 decode 推到 L* 之上。
- Yes/No hypothesis: 推导在所有 5 drop policy × 4 rate × Qwen3 + GSM8K 单元上 ±3% absolute 预测 ±2% 解码结果。
- Dominant mechanism: 结构不等式 + 硬件常量表。
- What this deliberately does NOT add: 不新增机制；不新增 lever；是关闭的定理。
- Why this follows from the user finding: GSM8K decode 0 收益是 user finding 的核心负面例；本 idea 把它从 "anomaly" 升级为 "structural"。
- What is new beyond current drop result: 给负面例一个可预测的形式化原因 + 边界条件（speculative decoding / MTP 何时打破）。
- Minimal pilot sketch, no execution: 离线 —— 推导不等式；在 v6 GSM8K 上验证。
- Required EP setting if later piloted: EP=8
- Expected positive signal: ±3% 预测。
- Falsification criterion: 任何单元预测误差 >5%。
- Main baseline: 经验观察（无解释）。
- Main risk: GPT 锋利问题 —— 定理是 Qwen3/8×4090/ep_ht 常量还是 model-independent? 我承诺以 *条件定理*：给定 (K, EP, T per step, hardware constants)，decode L_recv 上界可由结构推导；模型独立性是 future work。
- Literature references:
  - METRO: Decode-serving in memory-bound MoE (verified=YES), https://arxiv.org/abs/2512.09277, 相关性：decode-side MoE 优化；不专门做 hopelessness 定理。
  - Scaling Multi-Node MoE Inference via Activation Patterns (Bambhaniya et al., 2026), verified=YES via D1 trace, https://arxiv.org/abs/2604.23150, 相关性：MoE 推理 scaling；不专门 decode 不等式。
- Claude self-score:
  - Novelty potential: 4
  - MLSys fit: 4
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 4
  - Simplicity/depth: 5
  - Risk of component stacking: 1
  - Overall: 7.5（post-GPT 从 8.0 下调）
- GPT/Codex critique:
  - (a) 减法：YES。清洁定理尝试，无新机制。
  - (b) 新 artifact：YES —— 把 ±2% decode 失败转为结构不等式。
  - (c) MLSys-level：部分 —— 可能锋利但 "any byte-reduction operator" 是 overclaim。
  - 最强 objection：decode L_recv 不是结构固定的；batching、speculative verification、MTP、serving mix 都能把 decode 推到不同 regime。
  - 锋利问题：定理是 Qwen3/8×4090/ep_ht 常量还是 model-independent MoE decode law？
  - 撞车风险：METRO (2512.09277)；Scaling Multi-Node MoE Activation Patterns。
- Claude response to GPT: 接受 overclaim 批评。重新表述为 **条件定理**：给定 (K, EP, T per step ≤ T_max, hardware constants)，decode L_recv ≤ K·T_max/EP < L*。speculative decoding / MTP / 大 batch 能把 T per step ≥ T_threshold 抬到 L* 之上 —— 把 user 的负面 + 正面例子统一在一个不等式下，并预测何时反转。这其实是 idea 的 *最大 contribution* —— predicting the regime change。撞车：METRO 在 memory-bound regime 优化 decode，不证 hopelessness；Activation Patterns 是 placement。
- Decision: **revise**（重新表述为条件定理 + 预测 regime change；继续推进）

### D5-Idea-10: Fused byte-counted reduce-scatter primitive

- Dimension: 5 — Minimal Deepening, Not Component Stacking
- One-sentence thesis: 用 *一个* fused collective primitive 替代 dispatch + combine a2a 对；primitive 原生支持 byte budget 参数；所有上层 lever (drop / K_eff / dispatch FP) 都通过这个 primitive 编程。
- Yes/No hypothesis: 原型 primitive 达到 drop + FP8 叠加的同 byte 节省，但 API surface 严格更小（≤3 个参数 vs ≥6）。
- Dominant mechanism: 新 collective primitive。
- What this deliberately does NOT add: 不新增 drop policy；不新增 quantization；不新增 placement。
- Why this follows from the user finding: 97% 字节主导意味着 collective 本身应该 byte-aware，不是上层补丁。
- What is new beyond current drop result: 把 drop / FP / K_eff 都通过同一 primitive 编程，不再是分散补丁。
- Minimal pilot sketch, no execution: 离线设计 primitive API spec；离线模拟与 stacked drop+FP8 的 byte equivalence。
- Required EP setting if later piloted: EP=8
- Expected positive signal: API 严格更简 + byte 节省持平。
- Falsification criterion: API 不能比 stacked 更简 或 byte 节省 <90% of stacked → primitive 没达到统一目标。
- Main baseline: stacked drop + FP8 in NCCL EP / DeepEP。
- Main risk: kernel-level 实现重，feasibility 低；和 NCCL EP / DeepEP 重叠。
- Literature references:
  - NCCL EP: Towards a Unified Expert Parallel Communication API for NCCL (2024), verified=YES, https://arxiv.org/abs/2603.13606, 相关性：unified MoE 通信 API；本 idea 在其上加 byte budget。
  - DeepEP (DeepSeek), 相关性：现有 fused dispatch+combine kernels；不暴露 byte budget。GitHub 实现非论文。
- Claude self-score:
  - Novelty potential: 4
  - MLSys fit: 5
  - Mechanism clarity: 4
  - Feasibility on 8×RTX4090 EP=8: 3
  - Simplicity/depth: 4.5
  - Risk of component stacking: 2
  - Overall: 7.5
- GPT/Codex critique: 本轮未入选 GPT review。
- Claude response to GPT: n/a
- Decision: keep（最重的 systems 贡献候选；feasibility 低优先级低）

---

### Dimension 5 Summary

- Generated ideas: 10 (D5-I1 … D5-I10)。所有 idea 都强制 "ONE dominant mechanism" + "what this does NOT add"。
- Top 5 reviewed by GPT (1 codex call, bundled): D5-I2, D5-I3, D5-I4, D5-I5, D5-I9。
- Kept (no major change): D5-I1, D5-I6, D5-I7, D5-I8, D5-I10。
- Revised (carry forward with explicit edits):
  - D5-I2: invariant 钉死为 **weighted expert output expectation**（不是 expert identity / final residual）；R² 测试升级到 task metric。
  - D5-I3: 被下界对象明确为 **expert 输出在 receiver 端 L2-ε 重建所需 bits**（不是 token entropy）。
  - D5-I4: 降级 —— GPT 挑出 "disguised stacking"；Pareto ≤2% 是 make-or-break，可能 fail 就 drop。
  - D5-I5: 主张窄化为 **payload-byte-only** regime law；locality/queueing 是 future work。
  - D5-I9: 重新表述为 **条件定理** —— 给定 (K, EP, T per step ≤ T_max, hardware constants) 预测 decode hopelessness；speculative / MTP 推过 L* 是预测的 regime change。
- Dropped: 无。
- Main GPT criticisms:
  1. **Equivalence / theorem 类 idea 必须钉死 invariant**（D5-I2 invariant 是什么、D5-I3 下界对象是什么）—— 否则塌成 tautology。
  2. **Single-knob abstraction 可能是 disguised stacking**（D5-I4） —— 必须用严格 Pareto match 测试，否则就是 controller packaging。
  3. **Regime law 必须窄化**（D5-I5 不能说 "副本无关"，只能说 "payload-byte 上副本无关"）。
  4. **Theorem 不能 overclaim**（D5-I9 不能说 "any byte-reduction operator"，需要条件化）。
- Main Claude–GPT disagreements:
  - On D5-I2: GPT 说 output-L2 R² 是 weak systems artifact；Claude 接受并升级到 task-metric R²，但仍主张 equivalence 本身是 MLSys 力度（不只是 ML 力度）。
  - On D5-I3: GPT 说 token/router 熵不决定 bits；Claude 完全同意，重定位被下界对象为 expert 输出重建 bits；分歧已对齐。
  - On D5-I4: GPT 说 disguised stacking；Claude 接受并降级，等 Pareto ≤2% 经验测试决定生死。
  - On D5-I5: GPT 说不能说 "副本无关"；Claude 接受并窄化为 payload-byte-only；分歧已对齐。
  - On D5-I9: GPT 说 decode 不是结构固定；Claude 重新表述为条件定理 + regime-change 预测，把 GPT 担忧变成 idea 的核心 contribution；分歧已对齐。
- Estimated Codex usage: 1 substantive Codex call this dimension (5 ideas bundled). Cumulative across D1+D2+D3+D4+D5 = 9 substantive + 1 ping. 仍远低于 15 次告警阈值。
- Next recommended dimension (do not execute): **本维度是 5/5 的最后一维**。整个 idea pool 已覆盖 (1) Communication-Centric (D1), (2) Phase-Asymmetric (D2), (3) Workload-Adaptive (D3), (4) L*-aware Cost Model (D4), (5) Minimal Deepening (D5)。下一阶段建议进入 **阶段 B 撞车检查** —— 把每维度 top 2-3 的 idea 集中起来做正式 lit-check。**等人工 gate。**

---

### 维度 5 — 质检结果（QC pass, 2026-05-29）

QC 范围：对 D5 的 10 个 idea 重新做 (1) 硬性门 —— 可证伪、具体性、与 finding 关系（"a2a 是 MoE 推理真瓶颈"）、重复性；(2) 修订后的质量评分 —— novelty / story_fit（关键 severity：能 work 但 *不* 推进 "bytes-as-first-class" 主故事必须低分）/ pilot_feasibility / pilot_cost（反向）；(3) 每个 idea 一句话赌注；(4) 对 top 3 调 GPT 挑战。不做撞车结论，不进入阶段 B。

GPT 挑战 trace：`docs/aris/traces/dimension5_qc_review.md`。下表中分数为 *GPT 之后*，与 GPT 的分歧记录在 trace 里。

#### QC 表

| idea_id | 硬性检查 | novelty | story_fit | feasibility | cost | 一句话赌注 | 标记 |
|---------|----------|---------|-----------|-------------|------|------------|------|
| D5-I1 | PASS | 3 | 4.5 | 5 | 5 | padding-only microbench 在固定 L_recv / expert compute / routing 分布下变化 wire bytes，per-byte→latency 斜率在所有 drop rate 下恒定 ±5% —— 把 user 的 97% 关联性证据升级为因果证据 | PASS（方法论 microbench；feasibility 最强；作为 I2/I3 因果前置） |
| D5-I2 | PASS-WEAK / LOSER | 2.5 | 3 | 4 | 4.5 | 所有 drop policy 都等价于 router 上保留 weighted expert output expectation 的 stochastic perturbation；task-metric R²≥0.9 把整个 drop 家族塌成一个数学概念 | PASS-WEAK / LOSER（GPT 挑为本轮 loser："equivalence 在严格约束下假、在宽松约束下空；最 rhetorical 的三个之一；least aligned with bytes-as-first-class"；novelty 4→2.5、story_fit 4.5→3） |
| D5-I3 | PASS | 3.5 | 4.5 | 4 | 4.5 | 用 rate-distortion 推 expert 输出在 receiver 端 L2-ε 重建所需的最小 bits，与 v6 实测对比，会发现系统目前发送的 bytes 是下界的 ≥10× —— 这意味着 drop 后还有大量优化空间，且下界本身是 MLSys 论文级 artifact | PASS（被下界对象钉死为 expert 输出重建 bits；信息论工具基础 + MoE 应用） |
| D5-I4 | PASS-WEAK | 3 | 3.5 | 3.5 | 3.5 | drop / K_eff / dispatch FP precision 这三个 lever 不独立 —— 它们都是 byte 轴的投影；单旋钮 runtime 在每个 budget 点 Pareto error ≤2% 才有 theorem，否则就是 disguised stacking | PASS-WEAK（GPT 挑出 "disguised stacking" 风险高；Pareto ≤2% 是 make-or-break；novelty 3、story_fit 3.5） |
| D5-I5 | PASS | 3 | 4 | 4.5 | 5 | 在 payload-byte-only 角度，当 L_recv > L* 时 replica budget 从 128 到 1024 的 e2e 差 ≤2% —— 这意味着 comm-bound regime 里副本是浪费 VRAM 而非加速，MoE 副本设计直觉在 L* 上方倒转 | PASS（窄化为 payload-byte-only 后稳；locality/queueing 是 future work） |
| D5-I6 | DUP（与 D5-I4 实质同源） | — | — | — | — | v6 grid 的所有正向单元都能用一个由 L*-distance 索引的单参数家族 ≥90% R² 拟合 —— 这经验性证明字节轴塌缩，是 D5-I4 的 retrospective 实证 | DUP（与 D5-I4 主张同源 —— I4 是 prospective single-knob、I6 是 retrospective single-param fit；作者已注 "与 D5-I2 / D5-I4 重叠"） |
| D5-I7 | PASS | 3 | 4 | 4.5 | 5 | K_eff 硬截负杠杆的原因可由一个分析模型给定量描述 —— GEMM 减少 vs byte-per-token 上升 + per-launch overhead 占比，模型预测误差 ±20%；这给 2.8% finding 一个干净的形式化解释 | PASS（独立解释 K_eff 失败；与 D5-I3 互补） |
| D5-I8 | PASS-WEAK | 3 | 3.5 | 4 | 4.5 | 在 v6 grid 上同字节预算的不同 drop 变体的 e2e 延迟方差 ≤2%、精度方差 ≥5pp —— 这意味着 drop variant 选择是纯精度-质量 lever，延迟由字节单独决定 | PASS-WEAK（GPT："iso-traffic ablation 是已知 systems 传统、equivalence 'mathematically cheap'、'latency = bytes alone' 是 overstated"；需升级为带 residual-term 的 latency 模型；novelty 4→3、story_fit 5→3.5） |
| D5-I9 | PASS | 3.5 | 5 | 4 | 4.5 | 给定 (K, EP, T per step ≤ T_max, hardware constants)，decode L_recv ≤ K·T_max/EP < L* 是结构不等式 —— GSM8K ±2% 是被预测的定理而非异常；定理还预测 speculative decoding / MTP 何时把 decode 推过 L* | PASS（top 1 post-GPT；GPT 独立评为 "cleanest fit to umbrella thesis"；novelty 4→3.5、story_fit 5 GPT 也认可保留；speculative decoding sign flip 升为 core contribution） |
| D5-I10 | PASS | 4 | 4.5 | 3 | 2.5 | drop / K_eff / dispatch FP 都可以通过同一个 byte-counted reduce-scatter primitive 编程，API 严格更简（≤3 参数 vs ≥6）且 byte 节省与 stacked 等同 —— 上层 lever 都是 primitive 上的 syntactic sugar | PASS（未经 GPT review；最重的 systems 贡献候选；feasibility 低优先级低） |

REJECT / DUP 的具体原因：
- **D5-I6 DUP**：与 D5-I4 (single-knob byte-budget runtime) 共用 "字节轴塌缩" 同一论证。D5-I4 是 prospective single-knob design，D5-I6 是 retrospective single-param fit；作者本人已 risk-note "与 D5-I2 / D5-I4 重叠"。应作为 D5-I4 的支撑实证而不独立计数。
- 无 REJECT（所有 idea 都满足可证伪 + 具体性 + 与 finding 关系；硬性门通过）。

#### Post-GPT top 2–3（按 novelty + story_fit 排序，PASS 项）

GPT 挑战后 D5-I8 与 D5-I2 跌出 top 3（I2 被挑为 loser、I8 被 "overstated" 拉下）；D5-I9 升为清晰 #1（GPT 独立确认 "cleanest fit to umbrella thesis"）。Re-rank：

- **Top 1 — D5-I9 (8.5)**：decode-hopelessness conditional theorem。GPT 独立评为 "cleanest fit to umbrella thesis"。核心修订：speculative decoding sign flip 从 example 升为 **core contribution**（不是 post-hoc 算法围 L* 转，而是预测 unmeasured regime 翻转）。
  - 一句话赌注：给定 (K, EP, T per step ≤ T_max, hardware constants)，decode L_recv ≤ K·T_max/EP < L* 是结构不等式 —— GSM8K ±2% 是被预测的定理而非异常；定理还预测 speculative decoding / MTP 何时把 decode 推过 L*。
- **Top 2 — D5-I10 (8.5)**：fused byte-counted reduce-scatter primitive。未经 GPT 单独挑战；最重的 systems 贡献候选；feasibility 低（kernel-level 实现）。
  - 一句话赌注：drop / K_eff / dispatch FP 都可以通过同一个 byte-counted reduce-scatter primitive 编程，API 严格更简（≤3 参数 vs ≥6）且 byte 节省与 stacked 等同 —— 上层 lever 都是 primitive 上的 syntactic sugar。
- **Top 3 — D5-I3 (8.0)**：information-theoretic lower bound on a2a bytes。未经 GPT 单独挑战，但 D5 idea critique 已给 framing 提示（rate-distortion 工具基础）。
  - 一句话赌注：用 rate-distortion 推 expert 输出在 receiver 端 L2-ε 重建所需的最小 bits，与 v6 实测对比，会发现系统目前发送的 bytes 是下界的 ≥10× —— 这意味着 drop 后还有大量优化空间，且下界本身是 MLSys 论文级 artifact。

候选 #4 = D5-I1 (7.5)：causal-isolation microbench；feasibility 最强，方法论价值。
LOSER = D5-I2 (5.5)：被 GPT 挑出 "obvious mathematical reparameterization / least aligned with bytes-as-first-class / most rhetorical"；novelty 4→2.5、story_fit 4.5→3 双双下调。

#### Claude–GPT 评分分歧总结

| idea | Claude pre-GPT (novelty, story_fit) | GPT 挑战方向 | Claude post-GPT (novelty, story_fit) | 是否仍分歧 |
|------|-------------------------------------|--------------|--------------------------------------|------------|
| D5-I9 | 4, 5 | novelty 略高（METRO / activation patterns 邻近）；story_fit GPT 独立评为 "cleanest fit"；"any operator" overclaim 需 residual-term framing | 3.5, 5 | 基本对齐；novelty 让步、story_fit 双方认可 5；分歧已对齐 |
| D5-I8 | 4, 5 | novelty 高（iso-traffic ablation 已知）；story_fit overstated（"latency = bytes alone" 过强，依赖 burstiness / sync / overlap / 不只是 byte 分布）；需带 residual-term 的 latency 模型 | 3, 3.5 | 部分对齐；分歧点：Claude 主张 "byte-distribution equivalence + residual term" 仍可发表，GPT 倾向 "纯实证模式不是 artifact" |
| D5-I2 | 4, 4.5 | novelty 严重过高（"obvious mathematical reparameterization"）；story_fit "weak for bytes-as-first-class"（routing-quality framing 不是 communication-objective framing）；preservation paradox（Σ g·o 保留 ⇒ 为何 accuracy 移动？）；挑为 loser | 2.5, 3 | 大体认输；分歧点：Claude 主张在 restrictive perturbation family 下可救，GPT 倾向 "ML 理论论文不是 MoE-systems contribution" |

#### QC 结论（维度 5）

- 总数：10
- PASS：9（D5-I1, I2, I3, I4, I5, I7, I8, I9, I10）；其中：
  - **D5-I2 PASS-WEAK / LOSER** —— GPT 挑为本轮 loser；"obvious reparameterization / preservation paradox / 与 bytes-first 主故事 misaligned"；
  - **D5-I8 PASS-WEAK** —— "iso-traffic ablation 已知 systems 传统、equivalence cheap、'latency = bytes alone' overstated"；
  - **D5-I4 PASS-WEAK** —— GPT "disguised stacking" 挑战；Pareto ≤2% make-or-break；
  - **D5-I9 GPT 独立确认 cleanest fit to umbrella thesis** —— 升为 post-GPT top 1。
- REJECT：0
- DUP：1（D5-I6 与 D5-I4 实质同源；作者已注）
- Top 2-3：D5-I9 → D5-I10 → D5-I3（D5-I8 / D5-I2 跌出 top 3；D5-I1 候补 #4）
- Codex 本 QC 轮额外消耗：1 次（GPT 挑战 top 3 一并打包，high reasoning）。D5 累计：idea critique 1 + QC 1 = 2 substantive Codex calls。
- 跨 D1+D2+D3+D4+D5 全部累计：10 次 substantive + 1 次连通性 ping，远低于 15 次告警阈值。用户解除 GPT 调用限制的消息已收到，"top 3 / 1 round each" 是内容规则继续保留。
- 维度结束。**整个 ARIS 阶段 A idea pool（5/5 维度 + 全部质检）完成**，等待人工 gate 进入阶段 B。
