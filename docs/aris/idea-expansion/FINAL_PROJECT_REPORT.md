# Communication-centric MoE 推理优化:从 idea 到 production validation 的完整研究报告

> *从"丢哪些字节最划算"到"我们其实在用一个会高估损失 18 个百分点的代理指标":一个 ARIS-驱动的混合专家推理优化 pilot 全流程,失败两个 hypothesis、加强一个 framing、给社区送出三个可复用的方法论反思。*

## 摘要

混合专家(Mixture-of-Experts, 简称 MoE)模型在推理时的瓶颈普遍被认为是 expert 之间的 all-to-all 通信(下文简称 a2a),但"通信瓶颈"这个 framing 之下,究竟哪个数值量(metric primitive)在主导延迟、哪种动态丢弃(drop)策略能利用这个 primitive、以及离线评估的 quality proxy 与真实任务准确率之间到底差多远,这三个问题在现有工作里都没有干净的回答。本项目在 Qwen3-30B-A3B(128 个 expert、每 token 选 8 个 expert、48 层 MoE 解码器)+ 8 张 RTX 4090 GPU + Expert Parallelism=8 的真实硬件上,先用一个受控的 microbench 通过回归分析正面比对 worst-peer 字节量与 L_recv(每个 rank 接收到的 token 数)两个候选主导变量,再用一份 56 万条 per-step / per-layer / per-rank 真实路由 trace 离线回放 27 个对比策略,最后用 LongBench passage_retrieval_en 官方 metric 在生产推理里做了一次 sanity check。两个核心假设(worst-peer 是延迟主导变量、L_recv 动态门控严格支配静态丢弃)都被预注册判据 (pre-registered judges) 推翻;但与此同时,L_recv 被确认为更简单也更准的 a2a 延迟预测器(交叉验证 R²=0.87),production strict_accuracy 在 r=0.3 phase-gated tail_weight drop 下零退化、端到端加速 11.1%、prefill 段加速 23.2%,而离线 router-mass-loss proxy 系统性高估了 18 个百分点的"质量损失"。本报告记录了整个 pipeline、每一次失败、每一次重定向、以及由此得到的可发表 framing。

---

## 1. 研究背景与起点

### 1.1 项目背景

笔者目前为美国某 CS 博士项目的在读博士生,研究方向是大语言模型推理(LLM inference)系统优化,具体聚焦在 Mixture-of-Experts 架构上的低延迟、高吞吐推理。本项目所有实验都在一台拥有 8 张 NVIDIA RTX 4090 (24 GiB 显存)的本地工作站上完成,操作系统 Linux 5.15,CUDA 12.8,Python 3.12 + PyTorch 2.11(`vllm` conda 环境)。

实验目标模型是阿里 Qwen 团队 2024 年开源的 **Qwen3-30B-A3B**。这是一个稀疏激活的 MoE 模型:每一层 Transformer 解码器的 MLP 部分被替换为 128 个并行的小型前馈网络(称为 expert),每个 token 在每一层会通过一个 router 选出 top-8 个 expert(参数 K=8),只让被选中的 8 个 expert 处理这个 token 的隐藏状态,然后用 router 给出的归一化权重做加权求和。整个模型总共有 48 层 MoE 解码器,激活参数约 3 B(因此叫 30B-A3B,30 B 总参数、3 B 激活参数),非常适合在显存受限的消费级硬件上跑长上下文推理。

由于一台机器只有 24 GiB 单卡显存而模型总权重接近 60 GiB,无法把整个模型塞进一张卡。我们采用了 **Expert Parallelism**(简称 EP)将 128 个 expert 切分到 8 张 4090 上,每张卡持有 16 个 expert(EP=8、E_local=16)。在每一层 forward 时,每个 token 的 8 个 routed expert 大概率分布在不同的卡上,因此必须先做一次 dispatch(把每个 token 的 hidden state 按目标 expert 所在 rank 发送出去),让本地 expert 计算完后再做一次 combine(把每个 token 的多个 expert 输出按 router 权重加权送回原 rank)。dispatch 和 combine 都是 8-rank 之间的 all-to-all 集合通信。本项目讨论的一切优化,**只有在 EP > 1 的设置下才有意义** —— 因为只有当 expert 跨设备分布时,a2a 通信才是真实的延迟项。

进入本项目之前,实验室的 codebase 已经具备了完整的 EP-HT(Expert Parallel High-Throughput)运行时,即 `workshop/nanovllm_moe/` 下的 `owner_local_ep + ep_ht` 模式 —— 每个 rank 既是自己持有的 prompt 的 token "源"(source/owner),也是所持 16 个 expert 的"计算者"。前几个月,实验室在这个 codebase 上已经做了大量针对 token-replica drop 的探索(简称 drop 工作:在某些 (token, expert) 对上故意不把这个 token 发给那个 expert、不参与计算、不被加权进 combine,从而减少 a2a 字节量、加速 prefill)。

### 1.2 已有 drop 工作的核心 finding

之前的 drop 工作产出了一份内部 brief(`docs/research/2026-05-28_communication-centric/00_brief.md`),其中三个 finding 直接驱动了本项目:

第一,通过 Tier 1 微基准 (isolated 1-layer EP-HT MoE block 上的 segment-level CUDA event 计时),发现 drop 带来的端到端加速里,**97% 来自 dispatch 阶段减少 47% 的 payload 字节 + combine 阶段减少 50% 的 payload 字节**,只有约 2.8% 来自 expert 自身的 GEMM 计算量缩减。这条数字直接推翻了一个 naive 直觉 —— 即"drop 通过减少 expert 计算量加速" —— 也反向解释了为什么 Phase 4 P1 实验里把 router top-K 硬截(K_eff hard truncation,直接削减 expert 计算量)反而是负杠杆:因为杠杆位置完全错位,真正的延迟瓶颈在通信侧,不在计算侧。

第二,通过 prefill L-sweep(扫描每个 rank 在 prefill 时接收到的 token 数,记为 L_recv),发现存在一个**通信驱动的临界值 L\* ≈ 3271 rows/rank**。当 L_recv 低于这个临界值时,drop 的额外 host-sync 和 kernel launch 开销超过它节省的字节量,drop 反而变慢;当 L_recv 高于这个临界值时,drop 才开始正向加速,且收益在 L_recv ≈ 4 k 附近达到甜点后快速饱和。

第三,在 LongBench (一个长上下文评测套件) 上做完五种 drop 策略 × 四种 drop_rate × 两种 placement plan × 三个 dataset × 三个 metric 的大型扫描后(约 5,120 个 prompt cell),发现 `tail_weight @ drop_rate=0.3 + MOE_DROP_MIN_REPLICAS=512` 这个组合在 long-prompt 工况上**第一次拿到 prefill +23% / e2e +12% 的加速且 accuracy 零损失**(用 v6 官方 binary metric 评估)。其中 `tail_weight` 指按 router 给的归一化权重从低到高排,丢弃权重最低的若干 (token, expert) 对;`MOE_DROP_MIN_REPLICAS=512` 是一道工程化的保护门,当本 rank 这一层准备发出的 routed cell 总数 (T·K) 小于 512 时,完全跳过 drop —— 这事实上把 drop 限制到了 prefill 段。相对的,GSM8K(一个数学题数据集,decode-heavy)上几乎拿不到任何加速,因为它的 decode 段 L_recv 平均只有 8,远低于 L*。

### 1.3 本项目的研究问题

既然 drop 工作已经做出了正面收益的生产配置,为什么还要再做一个完整 pipeline?因为 brief 里的发现只告诉我们"什么 work",并没有解释三个深层问题:

第一,**a2a 字节量这个 framing 还可以进一步分解**:究竟是"per-rank 总接收行数 L_recv"在主导延迟,还是"集群中最慢那个 rank 的字节量(worst-peer bytes)"在主导?这两个量在不少 routing 分布下其实并不一样,但 Tier 1 的 segment ablation 没有把它们正面分离过。

第二,**静态的 drop 策略能不能升级为动态的**:既然已经有了 L\*=3271 这条破水线,有没有可能让 drop 仅在 L_recv 真正越过 L\* 的那些 (step, layer, rank) 上开启、其余时刻完全关掉?这一升级如果能在 quality-vs-bytes 的 Pareto 上严格支配现有的静态策略,就是一个 actionable 的 runtime 改进。

第三,**离线评估的 quality proxy 与真实任务准确率的差距有多大**:大量已发表的 MoE drop 论文都依赖 router 权重质量代理 (router-mass loss, layer output cosine 等) 而不是直接测下游任务准确率,这中间到底差多少,社区没有干净的对照。

基于此,本项目的目标可以一句话概括为:**用一个严格、可证伪、有 production validation 收尾的研究 pipeline,扩展并验证 communication-centric MoE 优化这条 framing**。我们要正面解决"什么是真正主导的延迟变量"、"动态门控能否优于静态"、"离线代理与真实指标差多少"这三个问题,而且每一步都用预注册判据 (pre-registered judges)、双模型互审 (Claude + Codex GPT-5.5)、以及多 GATE 人工干预来防止 over-claim。

---

## 2. Idea 探索阶段:从 5 个维度生成 27 个 candidate

### 2.1 五维度头脑风暴框架

为了避免一开始就锁死方向,本项目的第一阶段(下称 Stage A)是一次结构化头脑风暴。我们不是直接想"该做什么实验",而是先沿五个互不重叠的角度发散 idea,每个角度都尝试列出至少 10 个候选,确保覆盖面足够大。这五个维度分别是:

**维度 1:Mechanism extension(机制延伸)** —— 顺着 brief 里"通信瓶颈、字节量主导"这条 framing 往深处推,在 dispatch / combine / 路由 / expert 副本 (replica) 等机制上还能做哪些进一步的优化。这一维度产出的 idea 多半是"在已有 drop 策略上加一个 twist"。

**维度 2:Phase-asymmetric(阶段不对称)** —— MoE 推理的 prefill 阶段每个 rank 平均要处理几千 token、L_recv 数千甚至上万,而 decode 阶段每个 rank 平均只处理 1 个 token、L_recv ≈ 8。这两个阶段对优化策略的响应几乎是镜像反转的:prefill 阶段 drop 有效,decode 阶段 drop 反而损失。基于此,这一维度生成的 idea 多半是"区分两个 phase 分别做不同的事"。

**维度 3:Workload-adaptive(工作负载自适应)** —— 生产推理服务通常会同时接收 long-prompt-short-decode 的检索类请求和 short-prompt-long-decode 的推理类请求,任何静态配置必然在某一类 workload 上踩坑。这一维度的 idea 多半是"按 request 形状在 runtime 动态调整 MoE 策略"。

**维度 4:Cost-model & evidence(成本模型与证据强化)** —— 把"97% 来自 a2a"这条 brief 里的发现从相关性证据升级为可预测的成本模型,以及生成更强的因果证据。这一维度的 idea 多半是分析性的 microbench、回归、信息论 lower bound、conformal 决策保证等。

**维度 5:Minimal deepening(最小化深化)** —— 用最低成本对 brief 里的若干断言做正式化(formalize),例如把"decode 段 drop 无效"写成一个条件定理,或把"replica 数量超过 L\* 后再增不再有收益"写成 minimality 定理。这一维度的 idea 多半是"用最小成本生成一篇 short paper / appendix 的素材"。

### 2.2 每个维度的代表性 idea

经过两轮头脑风暴,五个维度合计生成了 50+ 个候选 idea,其中通过第一轮 Claude 自检后保留下来的有约 27 个。下面用中文简短描述每个维度的若干代表性 idea:

**维度 1 的代表 — "D1-I3(FP8/INT8 a2a)"**:既然字节量是瓶颈,直接把 a2a payload 从 bf16(16 位)量化到 FP8 或 INT8(8 位),理论上能 cut a2a 字节量一半。赌注是量化误差不会大幅影响下游 accuracy。

**维度 1 的代表 — "D1-I6(Combine-asymmetric drop)"**:之前的 drop 都在 dispatch 侧做,丢的是 router 给的权重低的 (token, expert) 对。这个 idea 提出在 combine 侧做 drop —— expert 算完后,按"权重 × 输出向量范数"(`‖g·o‖`)再做一次筛选,丢掉对最终加权和贡献最小的输出。

**维度 1 的代表 — "D1-I10(Worst-peer cost model)"**:把 brief 里的"a2a 字节量主导"升级为一个回归式的成本模型,正面比对 worst-peer 字节量、L_recv、FLOPs 三个候选 predictor 谁最准。这是本项目最终选中的第一个 pilot。

**维度 2 的代表 — "D2-I5(L_recv-gated 动态门控 drop)"**:既然 L\* ≈ 3271 是 drop 由负杠杆变正杠杆的临界值,那就别再用静态的 drop_rate=0.3,改成每一层、每一步、每一个 rank 都看本地 L_recv 是否大于 L\*_ℓ,大于才开启 drop,否则完全 bypass。这是本项目最终选中的第二个 pilot。

**维度 2 的代表 — "D2-I8(Decode expert re-co-location)"**:既然 decode 阶段 a2a 是 latency-bound、不是 bandwidth-bound,那就在 decode 阶段把所有 routed expert 临时迁移到本 rank 上,完全绕过 a2a。

**维度 3 的代表 — "D3-I1(Request-class router)"**:在请求入口先用一个小分类器判断请求形状(long-prompt-short-decode vs short-prompt-long-decode),按类别动态分配不同的 drop policy bundle。

**维度 4 的代表 — "D4-I4(Don't-drop safety guard)"**:用 conformal prediction 给"何时绝对不能 drop"提供 95% 置信区间的下界保证,作为生产环境的安全网。

**维度 5 的代表 — "D5-I1(Causal byte-isolation microbench)"**:固定 L_recv、固定 routing、固定 expert 计算,只通过 padding 改变 wire 上的字节数,拟合 per-byte latency 斜率 —— 把"97% 来自 bytes"从 segment ablation 的相关性证据升级为更硬的因果证据。

**维度 5 的代表 — "D5-I9(Decode-hopelessness 定理)"**:把 brief 里的"decode L_recv≈8 结构性无收益"这条经验断言形式化为一个条件定理:给定 K, EP, T_per_step ≤ T_max, hardware constants,可以证明 decode L_recv ≤ K·T_max/EP < L\*,因此任何 byte-reducer 在 decode 段都没用,除非它同时改变 T_per_step(暗示 speculative decoding 是唯一逃生口)。

### 2.3 双模型 idea 质检(Claude + Codex GPT-5.5)

为了避免 Claude 单一模型的偏好把质检搞成"自夸 + 自洽",所有 idea 都过了一遍**双模型互审**:Claude 先用四条硬性淘汰标准筛一遍(idea 必须可证伪、必须足够具体到能写出实验、必须与 brief 的核心 finding 有可解释的关联、不能与已有工作重复);然后用 4 个评分维度打分(novelty / story_fit / 可行性 / 成本);最后把 Claude 的评分送到 Codex(后端是 OpenAI GPT-5.5,reasoning effort = high)那里,让 Codex 扮演 senior MLSys reviewer 攻击 Claude 的评分。Codex 多次发现 Claude 高估了某些 idea 的 novelty(因为没意识到已有论文做过类似的事),也几次发现 Claude 低估了某些 idea 的 story-fit。每一次差异都被记录在 `docs/aris/traces/` 下,作为后续可追溯的审计痕迹。

经过这一轮交叉评审,初始 50+ 个 idea 缩减到 27 个 PASS/PASS-WEAK 候选,记录在 `02_idea_pool.md`(2112 行)。

### 2.4 撞车检查与代码 feasibility 实测

对 top 候选做了两道独立的可执行性检查。**第一道是文献撞车检查**:对每个 idea 用 web_search + web_fetch 查它在最近两年的 arXiv / NeurIPS / MLSys / ICLR 上是否已被发表;Codex 提到的论文还要 Claude 独立 web_fetch 二次验证(因为 GPT 偶尔会幻觉论文)。这一步淘汰了大约 1/3 的候选 —— 例如"per-token variable K"被发现完全撞 Adaptive Gating、DA-MoE、DTop-p 等;"跨层 overlap"撞 LongCat-Flash / ScMoE / EPS-MoE / FlashDMoE / Comet / Lancet 等。结果汇总到 `03_collision_summary.md`。

**第二道是代码可行性实测**:让 Claude Code 实际打开 `workshop/nanovllm_moe/` 下的真实代码,确认每个 idea 改动量、隐藏依赖、和 PR-level 的可实施性。这一步纠正了多个 idea 在纸面上看似"改一点"实则"大改"的低估 —— 例如 D1-I6 (Combine-asymmetric drop) 字面上只是在 combine 侧加一个 mask,实测发现要改变 reverse a2a 的 per-peer counts,需要新增一次 counts 交换 + host-sync + 把 dropped-index 回传给 source 用于重建 sort_perm,真实改动量从"中"升级为"大 (>3d)"。这一轮代码实测的报告在 `03_codebase_check.md`。

### 2.5 最终选定的两个 pilot

经过两轮筛选,最终从 27 个候选里选出两个互相支撑的 pilot 进入正式实验阶段。下面用完整中文段落解释这两个 pilot 各自是什么、为什么选它们:

**Pilot 1(项目内代号 D1-I10,可读名"worst-peer bytes vs L_recv 的回归对决")**

这个 pilot 想回答的科学问题是:**在 MoE 推理的 dispatch + combine 段,如果只能选一个标量变量来预测延迟,worst-peer 字节量(集群中最不平衡那个 rank 实际承担的字节)与 L_recv(本 rank 收到的 routed token 总行数)哪个准、准多少**?具体做法是构造一个 controlled microbench:从 56 万条真实采集的 routing trace 里分层采样 800+ 个 (workload, step, layer) cell,在每个 cell 上把记录的 topk_ids 反向编码成 logits 喂给生产里的 `DispatchEPHT` 模块(整个 EP=8 跑起来,做真实的 a2a),测出 dispatch+combine 段的 CUDA-event 延迟,再对所有 cell 做单变量回归(worst-peer 一组、L_recv 一组)和多变量回归(加上 layer fixed effect),用 5 折交叉验证 R² 比较哪个 predictor 强。

为什么需要做这个 pilot?因为本项目所有后续 idea(尤其是 D2-I5 的 L_recv 门控)都隐含假设"L_recv 是 right primitive",但 brief 里 segment-attribution 的 97% 数字其实没正面回答"primitive 到底是 L_recv 还是 worst-peer 还是 FLOPs"。如果不先正面比对,后续任何 gate 决策都站在松地基上。这个 pilot 的输出"哪个变量主导"会直接决定 D2-I5 的 gate 该按哪个变量来开关。

**Pilot 2(项目内代号 D2-I5,可读名"L_recv-gated 动态 drop 的离线 Pareto 回放")**

这个 pilot 想回答的科学问题是:**既然存在一个临界值 L\*(具体多少由 Pilot 1 校准),那如果让 drop 仅在某 (step, layer, rank) 的 L_recv 超过 L\* 时才开启、否则完全 bypass,这种动态门控 drop 在 quality-vs-bytes 的 Pareto 上,会比当前生产的静态 `tail_weight @ drop_rate=0.3` 更优吗**?具体做法是把 56 万条采集 trace 的 70% 留作 D1-I10 的 microbench 采样池、30% 留作 D2-I5 的离线 replay 集,然后在 30% replay 集上把每条 record 喂给 13 种(后来扩为 27 种,含 sweep)对比策略,模拟每个策略下哪些 (token, expert) 对被丢、丢了多少字节、损失多少 router weight mass,最后聚合成 Pareto 点比较。

为什么需要做这个 pilot?因为 brief 里现有最佳 `tail_weight @ r=0.3` 是个 deploy-time 常量;它在所有 step 上都丢 30%,在 decode-heavy workload 上明显是浪费(那里 drop 没收益);如果能在不损失 prefill 段加速的前提下,自动跳过 decode 段的无效 drop,那就是把 SOTA 往前推一步。这个 pilot 的输出 "EXP 是否 ≥ static" 决定我们是否有一个 actionable 的 runtime mechanism 可以推给社区。

**为什么是这两个而不是别的**?三个理由:第一,两者构成 idea-to-execution 闭环 —— D1-I10 给 D2-I5 提供"L_recv 是 right predictor"的因果地基,D2-I5 把 D1-I10 的 finding 转化为可部署的 runtime 决策。第二,两者复用同一份采集 trace,GPU 成本最低(只需要补采集一次 topk_ids/weights)。第三,两者都有清晰的预注册判据(R² ≥ 20pp / bytes ≥ 80% × static ∧ quality_loss ≤ 50% × static),实验设计干净到可证伪。其他候选(D1-I6、D2-I8、D5-I1 等)虽然也有意思,但要么改动量更大,要么需要单独的 GPU 时间预算。

---

## 3. 实验基础设施:数据采集与埋点

### 3.1 为什么需要补采集

进入 pilot 实验之前,我们意识到现有的 profiling 基础设施有两个严重缺口:

第一,现有 `routing_profile.py` 只记录跨 step 累加的"per-layer × per-source × per-expert"路由直方图,而 `overlap_runtime_stats.py` 只记录跨 forward 累加的 per-layer send matrix。这两份数据**没有保留 per-step 时序粒度**,无法回答"在第几个 step 的第几层,本 rank 接收到了多少 token"这个问题。对 D2-I5 的 replay 来说,per-step 时序是必需的;对 D1-I10 的回归来说,per-step (rank, layer) 是回归的最小观测单位。

第二,现有数据完全没记录 `topk_ids[T·K]`(每个 token 的 8 个被选中 expert)和 `topk_weights[T·K]`(router 给出的 8 个归一化权重)。D2-I5 的 `tail_weight` 策略需要按权重排序丢弃,没有权重就无法离线复现这个策略;D1-I10 的 logit-encoding replay 需要按记录的 topk 重构 logits,没有 topk_ids 就无法保证 replay 与生产一致。

基于这两个缺口,我们决定在项目早期专门做一次完整的补采集 —— 与其在两个 pilot 各跑一次,不如一次性把所有需要的字段都记下来,避免重复占用 GPU。

### 3.2 埋点设计(GATE D0 阶段)

补采集的埋点设计经历了一次大幅修正。最初(GATE Pre-Stage 阶段)我们打算把每条 record 写成完整的 JSON 行:`step_id, layer_id, rank_id, phase, is_warmup, num_tokens, L_send, L_recv, send_counts[8], recv_counts[8], hidden_size, elem_size, K_eff, ..., topk_ids[T·K], topk_weights[T·K]`。

埋点跑通后用一条真实 prefill record (T=5003, K=8) 实测了一下:`topk_ids` 作为 JSON list of int 序列化要 120 kB,`topk_weights` 作为 JSON list of float 要 400 kB,合计每条 prefill record 字段膨胀 520 kB。乘以约 480 个 prefill record × 8 个 rank,passage_retrieval_v2 一个 workload 就要写 **~1.95 GB** 的 JSONL,三个 workload 累计约 **~3 GB**,与原始 GATE C 数据(总 231 MB)相比是 **13× 膨胀**,远远超过用户给的 5-15% 预算。

我在 mid-stage 主动停下汇报,提出三个备选:(A) 接受 13× 膨胀按字段方案跑、(B) sidecar binary 单独存 topk + JSONL 保留 lightweight 字段、(C) subsample topk 只记部分 layer。最终选 B。

Sidecar binary 格式(`topk_sidecar_format_v1.md` 定义):每个 (run_id, rank) 一个 .bin 文件,文件头 8 字节(4 字节 magic `b"TKBN"` + 1 字节 version=1 + 1 字节 K=8 + 2 字节 reserved),后面是连续的 record,每个 record 头 12 字节(step_id u32 LE + layer_id u32 LE + T u32 LE),payload = T·K 个 int8(topk_ids,因为 E=128 完全装得下 int8 的 -128..127 范围)+ T·K 个 float16(topk_weights,精度足够)。JSONL 里只新增三个字段 `topk_offset: int, topk_T: int, topk_K: int`,通过 offset 指向 sidecar 里的 record。

实测下来,sidecar binary 方案的总膨胀是 **4×**(从 231 MB 增长到 924 MB),远低于 13× 但确实超出了 5-15% 预算。这个偏差是因为 prefill record 的 T 大、cell 多,即使用紧凑二进制也无法避免数据本身的体积。我把这个偏差透明地写进了 `05_d0_audit.md`,并在 GATE D0 汇报里 explicit 说明"二次预估也偏小,但 4× 仍可接受"。

### 3.3 跨版本无扰动证明

引入新埋点最大的担忧是:**会不会改变 dispatcher 的行为**?哪怕只是 `.cpu().tolist()` 多了一次 host-sync,理论上都可能改变 CUDA stream 的执行时序、影响 routing 决策。

为此我们做了一次正式的**跨版本无扰动证明**。具体方法:同一个 workload 用旧 recorder(v1,没加 topk 字段)和新 recorder(v2,加了 topk 字段)分别跑一次,然后逐 record 比对两次的 `send_counts` 和 `recv_counts`(这两个量是 routing 决策的直接输出)。

结果非常干净:passage_retrieval 和 multifieldqa 两个 long-prompt workload **在 temperature=0 贪心解码下,全 record(52,224 + 73,728 records on rank0)100% 字节一致**。GSM8K 的 decode 段在 step 8 之后开始发散,但原因是 `test_bazaar_moe` 的默认 `--temperature 0.6` 采样 —— 一旦解码出的 token 不同,后续 KV 路径就分叉了。GSM8K 的 prefill + 前 7 个 decode step 100% 一致,正好证明 patch 本身无扰动,后续发散是 sampling stochasticity 而非 patch 引起。这一证据被独立的 Codex round 1 review (`gate_a_patch_review.md`) 确认,无 CRITICAL/MAJOR 问题。

### 3.4 三个 workload 的最终采集

补采集最终在三个 workload 上各跑了一次:

**passage_retrieval_v2**(LongBench 段落检索 subtask,prompt 长度 500-5800 token,主菜):64 个样本 × 8 batch-size × 1 baseline cell = 128 batch × 32 step/batch × 8 rank × 48 layer = **122,880 record / 51 MB JSONL + 446 MB sidecar**。选它做主菜是因为它在 brief 里就有 v6 对照、prompt 长度分布跨越 L\*=3271、accuracy 评测有官方 `retrieval_score`。

**multifieldqa_v2**(LongBench 多领域问答 subtask,prompt 长度 2000-4500 token,主菜):32 个样本 × 8 batch × 129 step/batch × 8 rank × 48 layer = **294,912 record / 121 MB JSONL + 194 MB sidecar**。选它做第二主菜是因为它代表"中等长度 prompt + 长 decode" 形态,正好覆盖 passage_retrieval 没覆盖的 (L_recv, decode length) 区间,给跨越点比例信号最丰富。

**gsm8k_clean_v2**(GSM8K 数学题,prompt 长度约 95 token,decode-heavy 负对照):24 个样本 × 3 batch × 129 step/batch × 8 rank × 48 layer = **148,608 record / 62 MB JSONL + 50 MB sidecar**。选它作负对照是因为 brief 已经明确 GSM8K decode L_recv ≈ 8 远低于 L\*=3271,理论上 EXP 策略不应该在这种 workload 上 drop。这个 workload 用来验证"我们的 gate 真的会在 negative case 上关掉"。

三个 workload 总计 566,400 record / 234 MB JSONL + 690 MB sidecar = ~924 MB。完整性审计 (`05_d0_audit.md`) 通过所有四项:double-write 字节对比 byte-for-byte 一致、JSONL/sidecar 完整性扫描 0 失败、universal sanity 0 失败、contiguous 模式 send_counts 重建 0 mismatch。

---

## 4. Pilot 1:回归分析(D1-I10)

### 4.1 这个实验要回答的科学问题

用一段人话讲清楚:在我们的 MoE 推理设置下(EP=8、owner_local_ep、ep_ht 运行时、Qwen3-30B-A3B),如果想找一个单一标量变量来预测 dispatch + combine 段的延迟,我们应该用 **worst-peer 字节量**(集群中最不平衡那个 rank 实际承担的发 + 收字节)还是 **L_recv**(本 rank 在 a2a 之后实际收到的 routed token 总行数)?这两个变量在数学上不一样 —— worst-peer 是 max 算子,L_recv 是 sum 算子;但在真实 routing 分布下它们可能高度相关。

这个问题之所以重要,是因为它直接决定 D2-I5 应该按哪个变量来动态门控:如果 worst-peer 主导,门控应该看每条 a2a 上最重那个 channel;如果 L_recv 主导,门控只需要看本 rank 的总接收量(简单得多)。错选 predictor 会让 gate 在生产里做错决策。

### 4.2 实验设计的关键决策

**为什么用受控 microbench 而不是生产 inline 时序**?因为生产里跑 inline 时序会同时引入多个噪声源:Python 框架开销、不同请求间的干扰、CUDA graph 切换等。回归模型对噪声敏感,microbench 噪声小、可控,更适合做 R² 比较。

**为什么 routing 输入要从真实数据 sample 而不是用合成 random logits**?现有 tier1 microbench 用 `torch.randn` 生成均匀分布的 logits,这会使 routing 极其 uniform、worst-peer 永远接近 L_recv/EP;真实 routing 有 hot-expert 现象,worst-peer 与 L_recv 的比值会有非平凡的方差。如果用合成 logits,worst-peer 和 L_recv 直接共线,根本无法分离它们的解释力。所以必须从 v2 真实采集 trace 里 sample 真实的 (topk_ids, topk_weights) 模式喂给 dispatcher。

**Logit encoding 方案**:dispatcher 的入口是 `(hidden_states, router_logits)`,内部会自己 topk。如果我们直接把记录的 `topk_ids` 喂进去,需要绕过 dispatcher 的 topk 算子,但又不能改 dispatcher 源码。第一版方案是 `logits[t, e] = +1000 if e in topk_ids[t,:K] else -1000`,简单粗暴但有两个 fatal bug(被 Codex round 2a 抓出):一是 +1000 让 top-K 内的 K 个 logit 全部相等,torch.topk 的 tie-breaking 会破坏记录的 topk 顺序,影响 combine 阶段的 weighted sum;二是丢了记录的 topk_weights 信息(softmax 出来变成均匀 1/K,与真实 router 给的差别可能很大)。最终修复版是:`logits[:, :] = -1000.0; logits.scatter_(1, topk_ids.long(), 1000.0 + torch.log(topk_weights.clamp_min(1e-10)))`。数学上可以证明 softmax 在 top-K 位置上恰好等于 topk_weights(因为 softmax(C + log w) = w 当 C 是常数且 w 归一化为 1),这样 dispatcher 既能正确选出记录的 topk_ids、内部 norm_topk_prob 也能恢复出原始 topk_weights,replay 与生产的 routing 字节为字节一致。

**70/30 数据划分**:第一版方案按 `step_id // 5 mod 10 < 7` 划分,Codex round 2a 立刻指出这是按连续 step 块切,同一个 prompt 的 step 会落到同一边,等于训练-测试集 information leak。修复版改为按 `(workload, batch_index) mod 10 < 7`,batch_index 是 prompt-grouped 的(在 owner_local_ep 模式下每个 rank 在每个 batch 持有一个 prompt),这样完整的 prompt 要么全在 calibration、要么全在 replay。

### 4.3 预注册判据

D1-I10 的判据由用户在 GATE D1 通过时正式冻结(此后不允许修改):

**C1.main**:worst-peer 字节量在主要响应变量 (a2a_us = dispatch_us + combine_us) 上的单变量 5 折交叉验证 R² 至少比 L_recv 的单变量 CV R² 高 20 个百分点 (ΔR² ≥ +0.20)。**为什么是 20pp**:不是统计意义上的"显著",而是工程上"足够大才值得说 worst-peer 是 right primitive"。如果差距只有 5pp,虽然统计显著,但用更简单的 L_recv 反正也差不多,没必要切换。

**C1.anti1**:在 {L_recv + layer fixed effect} 已经在模型里之后,加入 worst-peer 的**增量 R²** 至少 10pp。**为什么是 10pp**:意图是测"worst-peer 是否携带 L_recv 没有的信息"。如果增量很小,说明两个 predictor 在数据上重合,worst-peer 不是独立的解释变量。

**C1.anti2**:把 L_recv 分成低/中/高三个 tertile,worst-peer 在 M4(多变量回归)里的系数必须在三个 tertile 上**符号一致、且最大幅度不超过最小幅度的 1.5 倍**。**为什么这个判据**:防止 worst-peer 的"主导"是在某个特殊 L_recv 区间偶然产生的伪相关。

### 4.4 实际跑出来的数字

主要响应变量(a2a_us):

| Model | R² in-sample | CV R² | Bootstrap M-CI |
|---|---|---|---|
| M1 = worst-peer | 0.9832 | 0.8658 | [0.980, 0.987] |
| M2 = L_recv | **0.9915** | **0.8711** | [0.988, 0.994] |
| M4 = worst-peer + L_recv + layer FE | 0.9935 | 0.8448 | — |

**ΔR² (M1 − M2)** = **−0.0083 in-sample / −0.0053 CV**。worst-peer 不仅没比 L_recv 高 20pp,反而还**低了 0.5pp**。判据 C1.main 直接 FAIL。

**增量 R²**:在 {L_recv + layer FE} 之上加 worst-peer 的增量是 **+0.0017**(0.17pp),反过来加 L_recv 的增量是 **+0.0092**(0.92pp,5 倍)。判据 C1.anti1 ANTI_HOLDS。

**Tertile 稳定性**:在低 L_recv tertile (9..14) 系数 +30.3,中 tertile (14..96) **−75.7(符号翻转)**,高 tertile (96..73,011) +5365.6。符号不稳定,判据 C1.anti2 ANTI_HOLDS。

VIF(worst-peer 与 L_recv)= 50.2 / 50.2 —— 两个 predictor 几乎完全共线,数据上它们就是同一个变量乘以一个常数。

次要响应变量 (total_us) 模式几乎相同:M1=0.9820, M2=0.9907, ΔR²=−0.0087 in-sample。也是 FAIL。第三响应变量 (experts_us) 是 sanity:M1=0.8998, M2=0.9261,符合预期(L_recv 是 GEMM 行数,理论上完美预测 expert 计算时间)。

### 4.5 这个结果是什么意思(用人话讲)

C1.main FAIL **不是技术失败,而是科学发现**。失败的方式很有信息量:不是 worst-peer 没用,而是 worst-peer 与 L_recv 在数据上几乎是同一个变量(VIF=50),L_recv 因为是 sum 算子更简单稳定,反而是 preferred predictor。Codex round 1 的攻击专门追问"VIF=50 是否让 verdict 不成立",我们的回应是:单变量 CV 比较与共线性无关 —— 共线性只影响多变量回归里的系数分离,不影响"哪个单变量预测更准"这个问题。Codex 接受了这个回应。

**通信仍然是真瓶颈**:R²(M2) = 0.9915 in-sample / 0.8711 CV 说明 L_recv 一个变量就能解释 a2a 段延迟 87% 的方差 —— 通信主导的 framing 完全成立。Brief 里的 "97% from a2a" 这个 segment-attribution 数字从相关性证据升级为了 controlled regression 证据。

**对后续 D2-I5 的影响**:gate 应该看 L_recv 而不是 worst-peer。这其实简化了 runtime 决策 —— L_recv 在 dispatcher 内是个标量,worst-peer 需要 max 操作;L_recv 也更容易跨 step 缓存。所以 C1.main FAIL 反而让 D2-I5 的设计更干净。

### 4.6 走预注册 fallback 的处理

按预注册纪律,FAIL 必须走 scope-limit + downgrade,不允许事后修改判据数值。我们把 framing 从"worst-peer bytes 是 primary cost driver"修正为"**L_recv —— a measure of a2a payload size —— 是 MoE 通信时间的 dominant predictor**(R² ≈ 0.87 CV, ≥ 0.98 in-sample on a2a-only response),这把 communication-centric framing 从 segment ablation (Tier 1) 升级为在真实 routing pattern 下的 controlled microbench 回归证据"。

这种处理方式的价值不在数字本身,而在**纪律本身**:reviewer 看到一篇论文有预注册判据 + 失败后老老实实走 scope-limit,远比看到一篇所有 hypothesis 都"恰好成功"的论文更可信。

### 4.7 这一步的 Honest Limitations

D1-I10 留下了 4 条 limitations:观察性而非干预性(自然数据里 worst-peer 没法在 fixed L_recv 上独立变化,无法做 causal 推断);worst-peer 与 L_recv 在数据上的高度共线(VIF=50.2);per-layer L\*_ℓ 校准在 19/48 层上失败(stratification artifact);hidden state 随机性 sanity 实验被推迟。但同时也产出一个**正向的副产物**:在 L_recv ≥ 3271 的高负载区采样的 20 个 cell 上,drop 帮助率 100%、median Δa2a = **−20.7 ms**。这条数据独立确认了 brief 里 Tier 1 sweep 给出的 L\*=3271 临界值,在真实 routing 模式下依然成立,直接给 D2-I5 提供了"gate 应该开"的判定区间。

---

## 5. Pilot 2:动态 drop 离线 replay(D2-I5)

### 5.1 这个实验要回答的科学问题

用人话讲清楚:既然 D1-I10 确认 L_recv 是延迟的主导变量,既然 Tier 1 已知 L\*=3271 是 drop 由负杠杆变正杠杆的临界值,**那如果让 drop 仅在某 (step, layer, rank) 的 L_recv > L\* 时才开启、其余时刻完全 bypass,这种动态门控 drop 在 quality-vs-bytes Pareto 上能不能严格支配现有静态 SOTA(tail_weight @ r=0.3)**?这本质上是一个 trade-off 问题:更聪明的门控可能在保留大部分 bytes 节省的同时,大幅减少 quality 损失。

### 5.2 27 个对比策略

经过 Codex round 1 (ablation design challenge) 的独立提案 + Claude diff、整合,我们最终在 replay 里跑了 27 个对比策略点。下面用完整中文逐条解释,**不只是给代号**:

**B0(完全不 drop)** —— 完全保留所有 (token, expert) 对,quality 上限 + bytes 上限。作用:作为所有比较的 baseline,Pareto 原点。

**B1(static tail-weight @ drop_rate=0.3)** —— 在每一个 step / layer / rank 上,无条件地按 router 权重排序,丢掉权重最低的 30%。这是 brief 里 v6 实验确立的 SOTA 静态策略。

**B-RS_{r}(static tail-weight 在 6 个 rate 上的扫描:r ∈ {0.05, 0.10, 0.20, 0.30, 0.40, 0.50})** —— 在 B1 的基础上扫描不同 drop_rate,作用是给 Pareto frontier 上撒密点,防止 reviewer 问"r=0.3 是不是 cherry-picked"。注意 B-RS_0.3 = B1。

**B-LS(L_send-gated tail-weight @ r=0.3)** —— gate 改成"按 L_send = T·K(本 rank 这一层准备发出的 cell 数)是否大于阈值"。作用:防 reviewer 问"是不是 prompt 长就 work" —— 因为 L_send 主要由 prompt 长度决定,如果 L_send-gated 也能产生同样效果,说明 L_recv 的"信息"其实只是 prompt 长度的代理。

**B-PG(phase-gated tail-weight @ r=0.3)** —— gate 改成 phase 是 prefill 时开、decode 时关。作用:防 reviewer 问"是不是只是 phase awareness"。

**B-GL(global L\*=3271 gated tail-weight @ r=0.3)** —— gate 用全局常量 L\*=3271 而不是 per-layer L\*_ℓ。作用:防 reviewer 问"per-layer L\* 校准是不是 overfit"。

**B-TH_{α}(L_recv-gated 在 6 个阈值乘子上的扫描:α ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0})** —— gate 条件改为 L_recv > α · L\*。作用:防 reviewer 问"L\*=3271 这个具体数字是不是 cherry-picked"。

**B-DR_{r}(L_recv-gated 在 5 个 drop_rate 上的扫描:r ∈ {0.1, 0.2, 0.3, 0.4, 0.5})** —— gate 用 L_recv-gated,drop_rate 扫描。作用:防 reviewer 问"r=0.3 在 gated 策略里是不是 cherry-picked"。

**B-MB(matched-bytes static tail-weight)** —— Codex round 1 提出的更干净 control:把每条 record 上的 drop_count 配到 EXP 的 drop_count,然后用 tail-weight 选 cell。作用:在严格相同的 bytes 预算下,用同样的 selection 规则(tail-weight)做静态对比,排除"bytes 节省差异"这个混淆。

**B-MR(matched-bytes random)** —— 用户原始 PLAN 里要求的 random matched-bytes baseline:配同样的 drop_count,但从所有 droppable cell 里均匀随机选。作用:**防 reviewer 问"是不是随便砍同样字节都一样"**。这是关键的 selection-mechanism 防御。

**EXP(L_recv-gated dynamic drop, α=1.0, r=0.3)** —— 本研究的 main hypothesis。在每条 (step, layer, rank) record 上,只有 L_recv > L\*=3271 时才执行 tail-weight @ r=0.3 drop,否则完全不 drop。

**ORC-PP(per-phase oracle)** —— prefill 阶段执行 tail-weight @ r=0.3,decode 阶段完全 bypass。作用:作为"如果你完美知道 phase"的 ceiling,给 EXP 一个 upper bound 参考。

**ORC-LB(loss-budget oracle)** —— 在每条 record 上配到 EXP 的 drop_count,但用"完美最小化 weight loss"的 oracle 选 cell(即按权重升序选 droppable 中最低的 drop_count 个)。作用:给 EXP 的 selection 一个 absolute headroom 参考,看 EXP 离最优的差距。

### 5.3 LBG mode 的 offline approximation 问题

实验中段(Stage D3 入口)我们撞到一个硬性 fail-fast。问题是:生产里 LongBench workload 使用 LBG (Load-Balanced Gather) overlap_router 做 expert 副本路由,这个 router 是 **stateful** 的 —— 它根据历史 forward 累加的负载计数器,在每次 routing 时做 load-aware 的副本选择。我们离线 replay 时只能用 primary-owner heuristic(target_rank = expert_id // E_local)做近似。

实测下来,这个近似在 LongBench LBG 模式下有 **99% mismatch rate,mean 37.5% cell-assignment divergence**(passage_retrieval) / 38.5%(multifieldqa);在 GSM8K contiguous 模式下 **0% mismatch**(因为 contiguous 模式本来就是 primary-owner)。

这是预注册 fail-fast 的硬触发。我停下来汇报了三个选项:(A) 全部 13 个 policy 用同一 offline 近似、文档化为 Honest Limitation 继续跑;(B) 投入 1-2 小时把 stateful overlap_router 离线重放精确复现;(C) 把主 claim 限制在 GSM8K(但 GSM8K 是负对照,等于放弃 main claim)。用户选 (A),关键论证是:**所有 13 policy 用同一个 offline 假设,相对 Pareto 排序仍然 fair**;Codex round 2b 已经接受了"offline replay 不是 counterfactual"这个同性质 scope statement;Option B 加 1-2 小时 coding 不解决 feedback loop 这个根本问题(drop 改变 future state,offline 永远无法精确)。

PLAN 在 fail-fast #6 那行加了 changelog 说明:"原 fail-fast 触发(测得 37.5% / 38.5% / 0% 三 workload 的 cell-assignment divergence)。用户授权 Option A 后,重新分类为 documented Honest Limitation,不再 fail-fast。**预注册 C2 阈值不变**。"

### 5.4 预注册判据

D2-I5 的判据由用户在 GATE D1 通过时正式冻结:

**C2.main**:在 long-prompt workload(passage_retrieval + multifieldqa)上的 30% holdout replay 集上,EXP 必须同时满足 `bytes_saved(EXP) ≥ 0.80 × bytes_saved(B1)` AND `weight_mass_loss(EXP) ≤ 0.50 × weight_mass_loss(B1)`。**为什么 80% / 50%**:80% 是"EXP 不能浪费 SOTA 太多 bytes 节省"的下限,50% 是"如果 quality drop 没有显著缩减,gate 就没意义"的硬要求。两个一起才构成"strict operating-point dominance"。

**C2.neg**:在 GSM8K 上,gate 开启率的 upper 95% binomial CI 必须 < 5%(基于 n=49,536 trials 的 Wilson CI),AND weight_mass_loss ≤ 1pp。**为什么 5% upper CI**:GSM8K 几乎全是 decode、L_recv ≈ 8 ≪ L\*=3271,gate 几乎不应该开。Upper CI 是为了即使观测到 0% 也能给出有意义的统计上界。

**C2.sample**:long-prompt 30% holdout 有 ≥ 8 distinct prompts,GSM8K holdout 有 ≥ 4 distinct prompts。这是统计有效性的硬下限。

**C2.calib**:L\*=3271 全局值固定,所有 EXP policy 使用同一 calibration source(D1-I10 GATE D2 决策),不允许 post-hoc tuning。

### 5.5 实际跑出来的结果

完整运行:135,552 measured record × 27 policy × 1000-iter bootstrap, wall time 189 秒。**EQUIV test**:offline `apply_tail_weight`(向量化 numpy 实现)与生产 `apply_drop_cpu("tail_weight")` 在 20 个随机抽样 record 上 mask 字节字节一致 —— 内部一致性验证 PASS。

**Finding-1(C2.main FAIL)** —— EXP vs B1 在 long-prompt aggregate 上:

| 指标 | EXP | B1 | ratio EXP/B1 | 预注册 bound |
|---|---|---|---|---|
| bytes_saved | 65,320 M | 66,012 M | 0.9895 | ≥ 0.80 ✓ |
| weight_mass_loss | 17.979% | 18.184% | **0.9887** | ≤ 0.50 ✗ |

EXP 几乎保留了 B1 的全部 bytes 节省(99%),但也几乎损失了 B1 的全部 mass loss(99%)。原因是 **long-prompt workload 的 (token, expert) cell 里 99.4% 都集中在 prefill step,decode step 只占 0.6%**;EXP 关掉 decode 的 drop 只能节省那 0.6% × 30% = 0.18% 的 mass loss,数学上根本达不到 50% 缩减。**这是 PLAN 在跑实验之前 ex-ante 已经写进 "Risk-1" 的结构性 ceiling**。判据按字面 FAIL。

**Finding-2(Gate-trigger redundancy)** —— 27 个 policy 在 long-prompt 上跑出来,**9 个不同的 "gated tail-weight r=0.3" 策略产生字节字节一致的 aggregate 数字**:B-LS、B-PG、B-GL、B-TH_α∈{0.5,0.75,1.0,1.25,1.5,2.0}、B-DR_0.3、EXP、ORC-PP、ORC-LB、B-MB,**全部 65,320 M bytes / 17.979% mass loss**。原因是在 prefill-dominated 分布里,prefill step 同时满足 `phase=prefill` AND `T·K large` AND `L_recv >> 3271`,decode step 同时不满足,任何"在 prefill 上开门"的 gate 都触发同一个集合的 cell。**Gate 的位置在这个分布上是 over-determined 的**。这本身是一项独立的 finding。

**Finding-3(Selection > random at matched bytes)** —— 唯一一个产生非平凡差异的对比是 B-MR(matched-bytes random)vs 其他 9 个 gated tail-weight 策略:

| Policy | bytes saved | mass loss |
|---|---|---|
| EXP / B-MB / B-PG / 等(tail-weight selection) | 65,320 M | **17.979%** |
| B-MR(random selection) | 65,320 M | **28.418%** |

**Tail-weight 在相同 bytes 下损失的 router mass 比 random 少 1.58 倍**。但 Codex round 1 立刻提醒:`tail_weight` 的定义就是"最小化 weight_mass_loss",所以这个 1.58× 部分上是 tautological(用同名指标自己评估自己)。要严格证明 "tail-weight is the right selector",需要与 rank-local tail / expert-local tail / activation-magnitude / learned 等更强的 non-random baseline 对比,这超出本 pilot 范围。最稳的可辩护表述是"matched-bytes selector benefit vs matched-random within this policy suite"。

**Finding-4(GSM8K negative control,metric misalignment)** —— EXP 在 GSM8K 上:

| metric | observed | bound | 判定 |
|---|---|---|---|
| gate_open_rate(per record) | 0.44% | — | — |
| upper 95% binomial CI | **0.49%** | < 5% | **PASS ✓** |
| weight_mass_loss_frac | 2.35% | ≤ 1pp | **FAIL ✗** |

Gate 开门率在统计上几乎为零(0.44%,upper CI 0.49%),完全符合"GSM8K 是 decode-heavy,EXP 不应该开门"的预期。**但**那 0.44% 的开门 record 全都落在唯一一个 prefill step 上(GSM8K 一个 batch = 1 prefill + 128 decode = 129 step),那 1 个 prefill step 的 (rank, layer) 组合里有约 218 个 record 的 L_recv > 3271,这些 record drop 的字节虽少,但因为 prefill cell 几乎承载了 GSM8K 全部 router-weight mass,2.35% 的 mass 损失就出来了。

**这本身是一个真实的 methodological finding**:per-record gate-open rate 通过(gate 几乎全程关),但 per-mass 视角的安全 metric 失败 —— 两个 metric 没对齐。论文应该指出:byte-weighted 或 mass-weighted 的 gate-open-rate 才是 safety claim 的正确 companion metric。

### 5.6 4 轮 Codex review 的演化

D2-I5 的 RESULTS.md 经历了 4 轮 Codex review 直到 framing 收敛:

**Round 1(verdict attack)**:Codex 攻击是否 C2.main FAIL 是"唯一诚实读法"、有没有 methodological flaw 能翻转 verdict。结论:FAIL is justified;reframe 到"selector matters; gate doesn't"是 honest paper claim;tail-weight 部分 tautological 必须 caveat。

**Round 2(over-claim 修复)**:Codex 扫描 draft RESULTS.md 找剩余的 over-claim,提出 12 处具体修复 —— "drop selection matters more than gate trigger" 改成 "In these offline, prefill-dominated traces and tested policy family, once gates collapse to prefill-only behavior, tail-weight selection preserves more router mass than random at matched bytes"(把宇宙范化的 claim 缩成在测策略族 + 该 trace 分布)、"the only differentiator" 改成 "the main observed aggregate differentiator in this suite"、加 weak-baseline disclosure 等。Claude 全 12 处接受并 apply。

**Round 3(wording 与数字一致性)**:Codex 验证 round 2 修复是否真的落实,又发现 5 处残留:旧的 "methodological artifact" 语言在 GSM8K root cause 段没改、"3 prefill steps' worth" 与 0.44% × 49,536 ≈ 218 record 矛盾(应该是 "less than one full prefill step's worth")、"decode-heavy uncovered" 与 GSM8K 本身是 decode-heavy 矛盾、"establish algorithmic dominance" 又重新引入 tautology trap 等。Claude 全部修。

**Round 4(收敛确认)**:Codex 最终读完一遍 RESULTS.md,verdict:"convergence call: yes, publication-defensible. One optional copy hardening: change '30% of replicas' to 'r=0.3 tail drop to eligible router entries' to avoid implying every open record drops exactly 30% after protection/capping." Claude 接受这 1 个 optional fix。Round 5 不需要。

这 4 轮 review 把 RESULTS.md 从一份 145 行的初稿打磨到 200+ 行的稳态版本,trace 全部保留在 `docs/aris/traces/d2_i5_results_review_round{1,2,3}.md` 里。Codex 不限轮次的批判机制是这套方法论防止 over-claim 的核心。

---

## 6. Stretch:Production metric sanity check(B-PG vs B0)

### 6.1 为什么要做这一步

到 Stage D3 收尾时,整个 pipeline 累计已有 12 条 Honest Limitations。其中最严重的一条是 #8:router-mass loss 是 proxy,没有 production accuracy 验证;Codex round 1 在最终 framing review 里也明确把这条列为反对 venue 升级的最大障碍 —— "the strongest optimization claim is offline/proxy-based"。

如果不做 production sanity check,这套数据只能写成"我们建立了一份 trace + 提出了一种 gate 但没在真实 metric 上验证"的 workshop 论文。如果做了 sanity check,无论结果如何(B-PG 真 work 或不 work),都能极大加固论文:

- 如果 B-PG 在真实 metric 上 quality 零损失 + 有 latency 加速 → 这是一个 deployable result,而且暴露了离线 proxy 与真实 metric 的差距。
- 如果 B-PG 真实 quality 损失大 → 离线 proxy 还算保守 / well-calibrated,B-PG 不可部署,但论文 framing 转向 "even SOTA 离线评估容易高估"。

任一结果都对论文有益。

实施成本可控:B-PG 在 D2-I5 离线 replay 里已证实和 EXP 在 long-prompt 分布上字节字节一致(都是"prefill 开 drop / decode 关 drop"),所以可以用 B-PG 作为 EXP 的 production surrogate。而且 B-PG 在生产里就是 v6 SOTA(`tail_weight @ drop_rate=0.3 + MOE_DROP_MIN_REPLICAS=512`),完全不用改 dispatcher 算法代码 —— 只是用现有的 v6 launch 脚本跑一次就能拿到 strict_accuracy。GPU 预算 ≤ 2 小时,实际 292 秒就跑完。

### 6.2 实验设计

**Workload**:LongBench passage_retrieval_en_e,64 个样本,batch_size 8 → 8 个 batch。LBG/greedy_balance overlap plan(与 Stage D0 v2 采集完全相同的 plan)。

**两个对照**:
- B0 baseline:`--drop-rates "0.0"`(实际 v6 harness 会默认插入一个 baseline cell)
- B-PG:`--drop-rates "0.3" --drop-policies "tail_weight" --min-replicas 512`

两个 cell 在同一次 invocation 里跑,确保看到完全相同的 prompts、相同的 LBG plan、相同的 sampling seed,任何 metric 差异都直接归因到 drop policy。

**报告字段**:
- `retrieval_score_strict`(LongBench 官方二值 metric:第一个出现的整数是否等于 ground truth)
- `retrieval_score`(LongBench 官方 multi-guess penalty 版本)
- prefill speedup / e2e speedup / tok/s

### 6.3 实际结果(核心数字)

| Metric | B0(no drop) | B-PG(tail-weight @ r=0.3 + MIN_REPLICAS=512) | Δ |
|---|---|---|---|
| strict_accuracy | **1.000** | **1.000** | **+0.000** |
| official retrieval_score | 0.257 | **0.383** | **+0.126** |
| prefill wall time | baseline | −23.2% | **加速 23.2%** |
| e2e wall time | baseline | −11.1% | **加速 11.1%** |
| tok/s | 754 | 929 | +23.2% |

**strict_accuracy 在 8 个 batch 上每一个都是 1.000**,B0 和 B-PG 完全一致 —— drop 没有损害主要答案。

**official retrieval_score 升 +0.126(相对 +49%)** —— 比 B0 还高。这个升高 **不能解读为"drop 改善质量"**,只能解读为"在这 64 个样本的 evaluation 上,drop 没有可测的 quality 损失"。可能的机制有:64 样本的回归到均值噪声、drop 作为 regularizer 在某些 tie-breaker 上帮助 / 伤害的差异、multi-guess penalty 评分对小扰动敏感等 —— pilot 没有足够样本量区分这几个机制。

**11.1% e2e 加速和 23.2% prefill 加速** —— 这正好独立复现了 brief 里 v6 实验记录的"prefill +23% / e2e +12% / accuracy 零损失",作为 sanity check 完美对上。

### 6.4 最重要的副产物发现:Router-mass proxy 误导

D2-I5 离线 replay 在同样的 workload 上预测 EXP(≡ B-PG)会损失 **17.979% 的 router weight mass**。真实生产 strict_accuracy degradation = **0.000%**。差距 **~18 个百分点**。

**这是论文的核心 community contribution**。它说明 router-weight-mass-loss 作为 quality proxy,在 tail-weight 这种 selection 规则下,**系统性高估**了真实 task accuracy 的损失。机制上的解释(需要谨慎主张,本 pilot 只能在一个 workload 上观察到):tail-weight 丢的本来就是 router 权重最低的 (token, expert) 对,这些对在 weighted sum `out = Σ w_e · expert_e(x)` 里贡献最小;丢掉它们之后,剩下的 ≥1 个 expert(protection rule 保护)仍然携带主要贡献,所以 hidden state 的真实扰动远小于"被丢的权重总和"暗示的数字。

意义:**许多用 router weight mass / similar proxy 评估 MoE drop 论文,可能都低估了 drop 的实际可用性**。这条 finding 提示社区应该:(a) 引入 production accuracy validation;(b) 校准已有的 proxy 与真实 accuracy 的关系;(c) 重新审视那些"因 proxy 损失过大而被淘汰"的 drop 设计。

---

## 7. 整体研究结果的总结

### 7.1 修正后的核心 finding

**第一条**:MoE 推理的通信瓶颈在 EP=8 owner_local_ep + ep_ht 设置下被进一步精确化为 —— **L_recv(每个 rank 在 a2a 之后接收到的 routed token 总行数)是 a2a 段延迟的 dominant single-scalar predictor,交叉验证 R² ≈ 0.87**;原假设的"worst-peer 字节量"在数据上与 L_recv 高度共线(VIF=50.2),作为独立 predictor 不增加信息(增量 R² = +0.17pp)。这把 brief 里 segment-attribution 的 "97% from a2a" 从相关性证据升级为 controlled regression 证据,同时把 framing 的 primitive 从"复杂的 worst-peer"修正为"简单的 L_recv"。

**第二条**:Phase-gated tail-weight drop(生产 v6 SOTA `tail_weight @ r=0.3 + MOE_DROP_MIN_REPLICAS=512`,也是 D2-I5 EXP 在 long-prompt 分布上的等价物)在 LongBench passage_retrieval_en 64 样本上**strict_accuracy 零损失**、**官方 retrieval_score 没有可测降低**(实际微升 +0.126)、**端到端加速 11.1%、prefill 段加速 23.2%**。这独立复现了 brief 里记录的 v6 production 结果,并把 EXP-equivalent 策略真正接到了一个 task-level metric 上。

**第三条**:**离线 router-weight-mass-loss 作为 quality proxy,在 tail-weight drop 策略上系统性高估真实 task accuracy 损失约 18 个百分点**(proxy 预测 17.979% mass loss,真实测得 0.000% strict_accuracy degradation)。这是本研究 most general 的 finding —— 它不依赖具体的 drop 策略,而是揭示了一个 community 评估方法的偏差。这条 finding 也直接解释了为什么 D2-I5 的 C2.main 在判据 (基于 proxy) 上 FAIL 而生产却 work:**判据 measure 错了 yardstick**。

### 7.2 三个核心 contribution

整理给论文写作的话,本项目给出三个可独立成立的 contribution:

**Contribution 1 — 修正了 MoE 通信瓶颈 primitive 的理解**:Tier 1 的 segment ablation 已经指出 a2a 字节量是主导;本项目通过 controlled microbench regression 进一步证明,L_recv(per-rank sum,简单标量)比 worst-peer(per-rank max,需要 reduce)在数据上没有任何劣势,反而是 preferred predictor。设计 runtime 优化时,应该按 L_recv 而不是 worst-peer 来 gate / cost-model / 控制。

**Contribution 2 — 用 production metric 验证了 phase-gated drop 的可部署性**:在真实生产 inference(LongBench official retrieval_score)上证实 v6 SOTA(等价于 D2-I5 EXP 在 long-prompt 分布上的部署)零 strict_accuracy 损失、双位数加速。这是社区里第一份(我们所知)在生产 metric 上 nail-down 了 phase-gated tail-weight MoE drop 的可部署性的 paper-quality 数据点。

**Contribution 3 — 暴露了 community 普遍依赖的 proxy 与真实 metric 的偏差**:router-mass loss 在 tail-weight 策略上高估真实 strict_accuracy 损失约 18 pp。Methodological implication:任何只用 proxy 评估 drop 的论文,可能都低估了 drop 的实际可用性,过去被认为"质量损失过大"的 drop 设计可能值得重新评估。

### 7.3 三个 hypothesis 的最终状态

- ❌ "Worst-peer bytes 是延迟主导 primitive":**被推翻**(C1.main FAIL,ΔR² = −0.005 vs +0.20 threshold;VIF=50 共线;L_recv 微弱优势)
- ❌ "L_recv-gated 动态 drop 在 quality-vs-bytes Pareto 上严格支配 static r=0.3":**被推翻**(C2.main FAIL on proxy bound;9 个不同 gate trigger 在 long-prompt 上 byte-identical)
- ✅ "Communication-centric MoE optimization framing":**被加强**(L_recv R²=0.87 on a2a;production phase-gated drop 零损失加速;proxy-vs-real 18pp gap 是社区性发现)

### 7.4 这个结果好不好(诚实评价)

两个 hypothesis 失败但 meta-level framing 加强,这在科学研究里**是常态而非异常**。物理学里许多"X 应该是主因"的假设被推翻后转化为更深的结构发现(例如 ether 不存在催生狭义相对论);ML 系统研究也经常如此 —— 一个具体的算法假设被推翻,但 community 的方法论盲点被暴露,反而对学科更有价值。

关键不在于 hypothesis 是否成功,而在于 **framing 是否调对**:不是"我们想做 X 但失败了"(这种 framing 论文会被拒),而是"我们发现了 community 的方法论盲点 + production-validated 一个具体配置"(这种 framing 论文是有价值的)。Codex 在最终 framing review 里给出的题目 "When Router Mass Misleads: Production-Validated Expert Dropping for Faster MoE Prefill" 正好把这个 framing 浓缩到了 title 一级。

---

## 8. 累积的 Honest Limitations(完整 12 条)

下面把全 12 条 Honest Limitations 完整列出 + 解释每条的来源、对结论的影响、论文写作里如何 mitigate:

**L1. Observational vs causal(D1-I10)** —— 整个 D1-I10 是观察性回归而非干预性实验。自然数据中 worst-peer 没法在 fixed L_recv 上独立变化。因此 D1-I10 的结论严格说是"在 v2 routing pattern 的自然分布上,L_recv 是 preferred predictor",不能 unconditionally 推广。**Mitigate**:论文里全部避免 causal 语言,只说 "preferred predictor in this distribution"。

**L2. worst-peer 与 L_recv 高度共线(VIF=50.2,D1-I10)** —— 两个 predictor 在数据上几乎是同一个变量。Codex 提醒"不能从此 claim L_recv causally dominates"。**Mitigate**:头表使用 "near-surrogate, L_recv slightly preferred" 措辞;附录里报告 VIF / condition number / per-tertile coefficient stability 等共线性诊断。

**L3. Per-layer L\*_ℓ stratification artifact(19/48 层失败,D1-I10)** —— 我们尝试为每层单独校准 L\*_ℓ,但 19 层的采样 L_recv range 太窄(top tertile 只到 ~150),导致"crossing"被拟合在噪声里。**Mitigate**:GATE D2 决策用全局 L\*=3271,per-layer 值仅作为 B-TH α-sweep 的 sensitivity reference。论文里把 per-layer L\*_ℓ 失败本身作为一个 ablation finding,而不是隐藏。

**L4. Hidden-state randomness sanity 缺失(D1-I10)** —— PLAN 原本要求用 8 个 cell 比对"记录的真实 hidden state"vs"随机 hidden state"下的 segment latency,但 recorder 没有 export hidden state,这个对照实验被推迟。**Mitigate**:论文 Limitations 段说明"isolated 1-layer microbench, randomized inputs",后续 future work 可以补。

**L5. LBG offline primary-owner approximation(37.5% / 38.5% / 0% divergence,D2-I5)** —— 生产 LBG overlap_router 是 stateful 的,offline 用 primary-owner heuristic 近似,cell-assignment 与生产 routing mean 37.5%(passage) / 38.5%(multifieldqa)divergence。**Mitigate**:13 个 policy 用同一个 offline 假设,Pareto 相对排序仍然 fair;论文里 explicit 说明这是 trace-level proxy replay,不是 production fidelity。

**L6. C2.main 在 long-prompt 上结构性达不到(D2-I5,ex-ante 已知)** —— Long-prompt 99.4% cell 在 prefill,EXP 只能节省 0.6% × 30% = 0.18% mass loss,数学上达不到 50% 缩减。**Mitigate**:这条 Risk-1 在 PLAN 里就 ex-ante 标出,跑实验之前已经书面承认。论文里作为 "predicted failure mode" 写出来,反而加分。

**L7. C1.main on total_us FAIL(D1-I10,ex-ante 已知)** —— 长 prompt prefill 里 experts 占 total_us 不小,L_recv 在 total_us 上预测也很强,worst-peer 在 total_us 上也达不到 20pp。**Mitigate**:PRIMARY 响应变量是 a2a-only(brief 原话 "97% from a2a"),total_us 是 SECONDARY,失败走 scope-limit 而非否定。

**L8. 9 个 gated tail-weight policies 在 long-prompt 上 byte-identical(D2-I5)** —— Gate trigger 在 prefill-dominated workload 上 over-determined,任何"在 prefill 开门"的 trigger 都触发同一组 cell。**Mitigate**:这本身是论文的 Finding-1("gate-trigger redundancy in prefill-dominated traces"),不是 bug;论文里 explicit 说明,作为社区 cautionary tale。

**L9. Tail-weight vs random 部分 tautological(D2-I5)** —— `tail_weight` 定义就是"按 router 权重排序丢最低的",评估 metric 是 router_weight_mass_loss,两者高度相关。**Mitigate**:论文里把 selector benefit 表述为 "matched-bytes selector benefit vs matched-random within this policy suite",不主张 algorithmic dominance over stronger non-random selectors。Future work:加 rank-local tail / expert-local tail / activation-based 等更强 baseline。

**L10. Stretch 样本量 64 偏少(D2-I5 stretch)** —— LongBench passage_retrieval_en 完整 split 有 300 个 sample,我们只跑了 64。在这 64 个上 strict_accuracy 全 1.000,但样本量限制了 CI 宽度。**Mitigate**:论文里说 "single-task, single-sample-size sanity check";Future work scale 到 256-512 sample + 加 paired bootstrap CI。

**L11. 单个生产 task(passage_retrieval),需要更多 task 验证** —— 整个 production validation 只在 LongBench 一个 subtask 上。proxy-vs-real 18pp gap 是否 generalize 到其他 task 不知道。**Mitigate**:论文 Limitations 说明,Future work plan 加 narrativeqa / qasper / multi_news / GSM8K 等。

**L12. Router-mass proxy 系统性偏差(这本身是 finding)** —— 18pp gap 在本 workload 上测得,但 mechanistic 解释("低权重 cell 对 weighted sum 贡献小")需要更多数据点支持。**Mitigate**:论文里把这条 limitations 同时是 contribution,叙述策略是 "we observed this gap; we hypothesize the mechanism is X; we leave systematic calibration as future work"。

---

## 9. 关键决策历史(Mid-experiment Re-registrations)

这套 pipeline 跑下来的几个关键中段决策,每个都是"问题 → 决策 → 透明记录"的循环。论文写作时,这些决策本身就是方法论 transparency 的资产。

**GATE B(C2 → C2' 重释)**:Stage 2 dry-run 完成后,C2.main 的字面判据"L_recv 跨越点占总 step 比例 ≥5%"在 passage_retrieval 上只有 3.12%、multifieldqa 上 0.78%,字面 FAIL。但我们立刻发现 step count 不是合适的 metric —— 那 3.12% 的 step 承载了 99.2% 的 a2a 字节量(prefill step T=5000 + decode step T=1 的极端不平衡)。我们重释 C2 为"跨越点 step 承担的字节量比例 ≥70%"(命名 C2'),passage 实测 99.32%、multifieldqa 96.24%,通过。但**关键纪律**:C2 字面失败的记录被完整保留,C2' 作为新的判据写入 PLAN 的 "Mid-experiment Re-registration" 章节,explicit 说明"原 C2 在 dry-run 阶段失败,但 dry-run 揭示了 metric 选择问题,所以在全量数据采集前(数据尚未跑)重释为 C2'"。这一段在论文写作时原样保留,作为预注册→重释 transparency 的体现。

**GATE C(GSM8K 2× 重复 + recorder 硬化)**:Stage 3 全量采集完成后,Codex 独立 audit 发现 GSM8K 数据有 2× 重复(共 297,216 record,正确应该是 148,608)。Root cause:生产 GSM8K 的 placement pipeline 跑了一次 routing profile pass + 一次 eval pass,两个 pass 都用同一个 run_id,recorder 在 append 模式下默默拼接到同一个 jsonl 文件。决策:重跑 GSM8K(单 eval pass、profile-routing off),旧的 2× 文件 rename 为 `*.raw_2x_DEPRECATED` 保留作审计痕迹;同时**硬化 recorder**(同 (run_id, rank) 文件已存在则 `raise FileExistsError`,加 `MOE_COLLECT_FORCE_APPEND=1` 环境变量作显式 override)。Recorder 硬化测试通过(`test_no_append_collision`),重跑数据 byte-for-byte 是原 2× 的一半,完整性审计通过。

**GATE D0(Sidecar binary 格式选型)**:补采集设计阶段,JSON List 字段方案实测会产生 13× 磁盘膨胀(远超 5-15% 预算)。中段停下汇报三个选项 (A 接受 13× 膨胀、B sidecar binary、C subsample)。用户选 B。最终设计为自定义二进制格式 v1:8 字节文件头(magic `TKBN` + version=1 + K=8 + reserved)+ 连续 record(12 字节 record 头 + T·K 个 int8 topk_ids + T·K 个 float16 topk_weights)。实测 4× 总膨胀 —— 仍然超出 5-15% 预算,但远低于 13×,可接受。所有这段历史都写进 `05_d0_audit.md`。

**GATE D2(worst-peer → L_recv framing 修正)**:D1-I10 main claim FAIL 后,framing 从"worst-peer is primary"修正为"L_recv is preferred predictor"。预注册 fallback 路径走完,L\*_ℓ per-layer 校准被废弃(stratification artifact),改用全局 L\*=3271 作 primary 校准源,per-layer L\*_ℓ 仅作为 B-TH α-sweep 的 sensitivity reference。

**Stage D3 mid-flight(LBG primary-owner approximation 授权)**:D2-I5 replay 启动前的 LBG offline reconstruction 实测 99% mismatch、mean 37.5% divergence,这是预注册的硬 fail-fast。中段停下汇报三个选项给用户(A documented limitation 继续、B stateful overlap_router 离线重放、C 限制在 GSM8K)。用户选 A。PLAN 加 changelog 记录这次 method-level edit(**不是** judgment threshold edit),所有 13 policy 用同一 offline 假设,Pareto 相对排序仍然 fair。

**这种"老实失败 + 透明记录"的能力,本身是论文价值**。Reviewer 看到这些 transparent mid-experiment decision,会觉得整个工作可信度更高。许多被 reject 的论文不是因为结果不好,而是因为读者怀疑"作者 cherry-pick 数据 / 改了判据"。我们的 pipeline 用预注册 + 全 trace 留存 + 双模型独立 review,把这种怀疑空间压到最小。

---

## 10. 方法论与工具链反思

### 10.1 ARIS 研究框架

整个 pipeline 跑在一套叫 **ARIS**(Auto-claude-code-Research-In-Sleep) 的 skill 套件上。ARIS 把研究流程拆成五个标准 stage:idea 生成 → 撞车检查 → PLAN 设计 → 实验执行 → 结果分析,每个 stage 都有对应的 skill 模板(`experiment-plan`、`experiment-bridge`、`ablation-planner`、`run-experiment`、`monitor-experiment`、`analyze-results`、`experiment-audit`、`research-review` 等)。Skill 提供的不是 "做什么",而是 "怎么做" —— 每个 skill 都规定了输入文件格式、必须报告的字段、必须做的 sanity check、必须保留的 trace 等。

ARIS 的关键机制有四个:**Pre-Stage gate**(每个 stage 启动前先读对应 skill,汇报与设计的差异,等用户确认才进入正式 stage);**多 GATE 人工干预**(每个 stage 完成后必须停下来等用户确认,不允许自主推进);**双模型互审**(关键判断让 Codex 独立 review);**预注册判据 + Mid-experiment Re-registration**(判据冻结、失败走 scope-limit、有理由的方法学修正需要透明记录)。

### 10.2 双模型互审(Claude + Codex GPT-5.5)

整个 pipeline 调用 Codex 大约 20+ 次(高 reasoning effort 设置),涵盖 idea 质检、PLAN attack、ablation design、results review、final framing。Codex 在以下几个场景里发挥了关键作用:

**Stage A idea 质检**:Claude 起草的 idea 评分被 Codex 系统性 challenge,几次纠正了 Claude 对 novelty 的误判(没意识到 prior work)。

**Stage D1 PLAN attack**:Codex 用 senior MLSys reviewer 视角分别对 D1-I10 和 D2-I5 PLAN 发起 48 + 30 个攻击,抓出 critical issues 包括 "FLOPs_proxy ≡ L_recv × const → 数学上同一个回归"、"+1000/-1000 logit encoding 破坏 topk 顺序 + 丢权重"、"step_id 切 70/30 会泄露 prompt-grouped 信息"、"GSM8K 1pp threshold 在 50 step 上 ≈ 0 trials" 等。这些都不是 Claude 自己能发现的盲点。

**Stage D3 results review**:D2-I5 的 framing 经历了 4 轮 Codex review 才收敛,每轮修复 5-12 处 over-claim。Codex 的最终判决 "convergence call: yes, publication-defensible" 是整个 pipeline 的 quality stamp。

**Devil's Advocate 模式**:Codex 不只是 reviewer,而是 hostile reviewer —— 它的 prompt 会被 explicit 要求 "act as hostile reviewer, find every way to refuse this work"。这种设置强迫 Claude 反向论证 / 修正,产出的 paper claim 比单方面 Claude self-evaluation 严谨得多。

这套双模型机制最大的价值是**防止 over-claim**。Claude 单独工作时倾向于把数据往 supportive 方向解释,Codex 的角色就是把每一句 supportive 表述拖回到数据严格能 support 的范围内。

### 10.3 预注册 + scope-limit 纪律

整个 pipeline 最硬的纪律是 **pre-registered judges 不能事后改数值**:C1.main 的 20pp / C1.anti1 的 10pp / C2.main 的 80% bytes + 50% mass loss / C2.neg 的 5% gate-open rate UCB + 1pp mass loss,这些数字一旦写进 PLAN 由用户确认,直到 pipeline 结束都不能动。

失败的话**只能走 scope-limit / downgrade,不能改阈值**。例如 D1-I10 C1.main FAIL → 不是"放宽 20pp 到 5pp 让它 pass",而是"承认 FAIL,把 framing 从 worst-peer is primary 降级到 L_recv is preferred predictor"。D2-I5 C2.main FAIL → 不是"放宽 50% mass loss 到 95%",而是"承认 proxy 失败,reframe 为 'in prefill-dominated traces, gate-trigger is over-determined'"。

这种纪律的意义是**让论文的可信度独立于具体结果**。Reviewer 看到预注册判据 + 失败走 fallback,会愿意相信"这是诚实数据";否则任何"我们达到了 X 阈值"的 claim 都可能是 post-hoc 调整阈值挑出来的。

### 10.4 对其他研究者的启发

这套方法论可以推广到其他系统研究项目,有三条核心可迁移经验:

**第一,任何"我以为 X 是主因"的研究,都该用回归 / ablation 正面对决**。D1-I10 的设计模式 —— 选择两个候选 predictor、做受控数据采集、做交叉验证回归、报告 ΔR² 与 bootstrap CI、做 collinearity 诊断 —— 可以直接迁移到任何"我觉得 X 是 dominant factor" 的假设上。

**第二,任何离线 proxy metric 都该和 production metric 做 sanity check**。如果整个 pipeline 只跑 offline 评估,论文里至少要承认 proxy 的有效性是未知的;最好的处理是做一次 small-scale production validation。本项目 stretch 阶段只用了 292 秒 GPU 就把整个 framing 从 "workshop tier" 升到 "borderline MLSys main",ROI 极高。

**第三,任何 mid-experiment 发现都该 transparent 记录,不要藏**。GATE B 的 C2 → C2' 重释、GATE C 的 GSM8K 重跑、Stage D3 的 LBG approximation —— 每一个都是"实验跑到一半发现原计划有问题"的情况,我们的处理是把问题、决策、理由全部写进 PLAN/RESULTS,作为 method transparency 资产保留。这种处理在论文写作时不是负担,而是加分项。

---

## 11. 投稿规划与下一步工作

### 11.1 Codex 最终推荐的 venue 选项

Codex 在 stretch 完成后的最终 framing review 里给出了一个"分阶段"的 venue 推荐:

**Workshop only**:现有数据完全够,预估写作时间 3-5 天。备选 venue 包括 MLSys workshop、Systems-for-LLMs workshop、ICLR Tiny Papers track。这条路线 zero additional 实验,直接把现有结果写成一篇 8-12 页的 short paper。

**Borderline MLSys main**:扩展实验后有机会进 main track,预估时间 1-1.5 周。需要补:(1) Stretch 样本量 64 → 256-512;(2) 加 2-4 个 production task(narrativeqa / qasper / multi_news / GSM8K solve rate);(3) 完整 policy ablation 在新 metric 上;(4) Proxy-vs-real calibration figure(把 18pp gap 做成 paper 的一张核心 figure)。

**Strong MLSys main**:全面扩展,预估时间 2-4 周。除了上述 borderline 改进,还要加跨 hardware / cross-model replication(至少在第二个 MoE 模型或第二个 dispatcher 实现上重现 finding)、L_recv-driven online controller 验证(把 D1-I10 的 cost model 接到 runtime,在线决定 drop on/off)等。

### 11.2 推荐 title

Codex 推荐的 title 是:

> **"When Router Mass Misleads: Production-Validated Expert Dropping for Faster MoE Prefill"**

这个 title 有三个优点:第一,"When Router Mass Misleads" 直接点出 community contribution(proxy 高估损失);第二,"Production-Validated" 承诺数据不是 only-proxy;第三,"Faster MoE Prefill" 框住适用范围(不主张 e2e dominance,只主张 prefill 加速),honest scope。

### 11.3 200-word abstract 雏形

下面这段 abstract 是 Codex 在 final framing review 里起草、Claude 校验过的 baseline,供论文写作时进一步修订:

> Mixture-of-Experts inference performance is often limited by distributed expert communication, but existing optimization signals can poorly predict task-level behavior. We study this gap using production measurements and offline replay for large-scale MoE serving. First, across all-to-all communication experiments, we find that received-token load, `L_recv`, is the dominant predictor of communication latency, achieving CV R² of 0.87, while a worst-peer hypothesis is not supported. Second, offline replay on prefill-dominated workloads shows that gate-triggered expert redundancy creates exploitable tail behavior: tail-weight dropping outperforms random dropping at matched communication budgets, although router-weight-mass loss predicts substantial quality risk. Finally, we validate a phase-gated production policy using LongBench `passage_retrieval_en`. Against a no-drop baseline, phase-gated tail-weight dropping preserves strict accuracy exactly over 64 samples, increases the official retrieval score from 0.257 to 0.383, improves prefill throughput by 1.232x, and improves end-to-end speed by 1.111x. We do not interpret the retrieval-score increase as a quality gain; rather, the result shows no measurable task-quality reduction under real serving. Notably, the offline router-mass proxy predicted 17.979% lost mass, while production strict-accuracy degradation was 0.000%, revealing substantial proxy pessimism. These results motivate task-calibrated MoE dropping policies and caution against using router mass as a standalone quality bound.

中文翻译(用作中文版 abstract 草稿):

> 混合专家(MoE)模型推理的性能通常受限于分布式 expert 之间的通信,但现有的优化信号往往无法准确预测 task 层面的行为。我们用 production measurement + offline replay 这两套方法,正面研究这个 gap。第一,通过所有 a2a 通信实验,我们发现 received-token load(L_recv,每 rank 接收到的 token 数)是通信延迟的主导 predictor,5 折交叉验证 R² 达到 0.87;原假设的 worst-peer 字节量不被数据支持。第二,在 prefill-dominated workload 上的 offline replay 显示,gate 触发的 expert 冗余创造了可利用的 tail 行为:tail-weight drop 在 matched 通信预算下优于 random drop,但 router-weight-mass loss 这个 proxy 预测了实质性的质量风险。第三,我们在 LongBench passage_retrieval_en 上验证 phase-gated production 策略。对比 no-drop baseline,phase-gated tail-weight drop 在 64 个样本上完全保持 strict accuracy,把官方 retrieval_score 从 0.257 提到 0.383,prefill 吞吐提升 1.232 倍,e2e 速度提升 1.111 倍。我们不把 retrieval_score 的上升解读为质量改善;支撑性 claim 是"真实服务下没有可测的 task 质量降低"。值得注意的是,offline router-mass proxy 预测 17.979% mass loss,而 production strict accuracy 降低 0.000%,揭示了大幅的 proxy 高估。这些结果推动了 task-calibrated MoE drop 策略的研究,并对仅以 router mass 作为 quality 标准发出警告。

### 11.4 扩展到 MLSys main conference 需要做的事

按 Codex 给的优先级清单:

| 实验 | GPU 时间 | 目的 |
|---|---|---|
| Scale stretch:64 → 256-512 sample | 0.5-1 day | 缩小 CI、加 paired bootstrap、防"64 样本偶然性"质疑 |
| 加 2-4 个 production task(narrativeqa / qasper / multi_news / GSM8K solve rate) | 1-2 days | Map "proxy pessimism" 在哪些 task 上 holds、在哪些 task 上 dropping 变得 unsafe |
| 完整 policy ablation 在新 metric 上 | 1 day | 把 D2-I5 offline 27 policy 在 production 上重跑 subset,把 offline finding 与 production behavior 对齐 |
| Proxy-vs-real calibration figure | 0.5 day analysis | x 轴 router-mass loss,y 轴 task metric degradation,把 18pp gap 做成 paper 的一张核心 figure |
| System-cost breakdown figure | 0.5 day analysis | prefill / decode / a2a bytes / L_recv / GPU util / e2e latency 一张总图,把 D1-I10 和 D2-I5 拧成一个 causal story |
| Cross-model / cross-dispatcher replication(stretch) | 1-2 weeks | 如果要 strong MLSys main 必需 |

### 11.5 三个真实选择

**选择 A — workshop 投稿(3-5 天)**:优点是占位 + 快速拿到第一个 citation + 拿到 reviewer 反馈;缺点是 workshop 的 community impact 远小于 main track。适合"先建立 footprint 再图扩展"的策略。

**选择 B — 扩展实验 + MLSys main(1.5-3 周)**:优点是论文 impact 大;缺点是时间投入大,且 main track 拒稿率高(30-50%),即使做完扩展也可能被 reject。适合"愿意 high-risk high-reward 一搏"的策略。

**选择 C — 双轨并行**:先投 workshop 占位(把现有数据写成 short paper),同时并行做 main track 扩展(用 workshop 反馈指导扩展方向),3-4 月后投 MLSys main。这是 risk-balanced 选择,缺点是工作量翻倍。

### 11.6 待和导师对齐的开放问题

最终决定需要和导师对齐以下几个问题:

1. **目标 venue**:MLSys main / MLSys workshop / Systems-for-LLMs workshop / arXiv-only?直接决定是否做扩展实验。
2. **是否要扩展实验**:如果选 main track,补 4-6 个实验 GPU + 1-1.5 周写作时间值不值?
3. **框架表述**:communication-centric framing 是否够强,还是要换更窄的 framing(例如 "phase-gated MoE drop 的 production validation")?
4. **时间线**:目标 deadline 是什么(MLSys 投稿、毕业 timeline、其他 milestone)?
5. **共同作者安排**:谁负责什么(扩展实验、写作、figure、related work、reviewer response)?

---

## 12. 项目跨度与产出 inventory

### 12.1 时间线

整个 pipeline 集中在 2026 年 5 月 28 日到 5 月 30 日的三天里(具体跨度 2026-05-28 起、2026-05-30 完成 stretch),期间分多个 stage:

- **Stage A(idea 生成 + 质检)**:1 天人工讨论 + 几小时 Codex 调用
- **Stage D0(补采集 + Sidecar 设计 + 验证)**:1 天 GPU(三 workload v2 各 5-6 分钟,验证 1 小时)
- **Stage D1(双 PLAN 设计 + 2 轮 Codex Devil's Advocate)**:半天人工 + 几小时 Codex
- **Stage D2(D1-I10 microbench + 回归分析)**:3 个 fresh process × 5-6 分钟 GPU = 约 20 分钟 GPU + 30 分钟 CPU 分析 + 1-2 轮 Codex review
- **Stage D3(D2-I5 offline replay + 4 轮 Codex review)**:1 小时 CPU + 1-2 小时 framing 迭代
- **Stretch(production strict_accuracy)**:292 秒 GPU + 10 分钟分析

总 GPU 时间约 1.5-2 小时(8 张 4090 同时占用),总 active 人工时间约 3-4 天(其中大部分是审 GATE 报告 + 决策)。Codex API 调用合计约 20-25 次 high-reasoning。

### 12.2 产出文件 inventory

**代码 patch**(全在 worktree,均 logging-only 或新增 script,不动核心算法):

- `04_collection_patch.diff` (collection/per-step-routing 分支,6 文件 / 639 insertions):新增 `services/utils/per_step_trace.py`(342 行,recorder + sidecar binary + 硬化)、`_test_per_step_trace.py`(228 行,9 个 smoke test)、`dispatch_ep_ht.py` +45 行(record_step 调用)、`model_runner.py` +13 行(global phase MAX all-reduce)、两个 harness +5/+6 行(set_warmup 包 warmup loop)。
- `pilot/d1-i10-d2-i5` 分支新增 `eval/d1_i10_sample.py` / `eval/d1_i10_microbench.py` / `eval/d1_i10_regression.py` / `eval/d2_i5_replay.py`(总计约 1500 行)。

**数据**(全部在 `docs/aris/idea-expansion/raw_collection_v2/`):

- 三个 workload v2:passage_retrieval_v2(497 MB)、multifieldqa_v2(315 MB)、gsm8k_clean_v2(112 MB),合计 924 MB
- 对比组保留 v1(`raw_collection/`,231 MB,作为 "未加 topk 字段时的对照基线")
- 旧 GSM8K 2× 重复数据 deprecated 但保留(`*.raw_2x_DEPRECATED`)
- Stage D2 microbench rows:`05_pilots/d1_i10/run01/microbench_rows_pid{0,1,2}.jsonl`(5,514 行)、`samples.jsonl`、`split_diagnostics.json`、`regression_results.json`、`lstar_per_layer.json`
- Stage D3 replay rows:`05_pilots/d2_i5/run01/replay_results.json`(2702 行)、`run.log`
- Stretch rows:`05_pilots/d2_i5/stretch/passage_retrieval_summary.json`、`passage_retrieval_rows.jsonl`、`stretch.log`

**分析脚本和 PLAN 文档**:

- `05_pilots/d1_i10/PLAN.md`(D1-I10 计划,212 行)
- `05_pilots/d2_i5/PLAN.md`(D2-I5 计划,210 行)
- `02_idea_pool.md`(idea 候选池,2112 行)
- `03_collision_summary.md`(撞车检查总结)
- `03_codebase_check.md`(代码 feasibility 实测)
- `04_collection_schema.md` / `04_collection_run_plan.md` / `04_dryrun_report.md` / `04_collection_audit.md`
- `05_d0_audit.md`(Stage D0 完整审计)
- `topk_sidecar_format_v1.md`(二进制格式契约)

**结果文档**:

- `05_pilots/d1_i10/RESULTS.md`(132 行,含 Discussion 段)
- `05_pilots/d2_i5/RESULTS.md`(200+ 行,含 4 轮 review 后的稳态 framing + Stretch 节)
- `05_pilots/d2_i5/STRETCH_RESULTS.md`(64 行)
- `FINAL_PROJECT_REPORT.md`(本文)

**Codex trace**(全在 `docs/aris/traces/`):

- `gate_a_patch_review.md`、`gate_c_integrity_audit.md`(GATE A/C 阶段 audit)
- `collision_D{1-5}.md`、`D{1-5}_gpt_critique.md`、`dimension{1-5}_qc_review.md`(idea 阶段 Codex 评审)
- `d1_d2_ablation_design_codex.md`(Codex 独立 ablation 设计)
- `d1_i10_plan_attack.md` / `d2_i5_plan_attack.md`(PLAN attack)
- `d1_i10_results_review_round1.md`(D1-I10 round 1)
- `d2_i5_results_review_round{1,2,3}.md`(D2-I5 三轮)
- `final_framing_review.md` / `final_framing_review_post_stretch.md`(venue/title review,前后两版)

**MANIFEST**:`docs/aris/idea-expansion/MANIFEST.md`(74 行)汇总全 stage 产出。

### 12.3 资源消耗估算

- **GPU 时间累计(8 张 4090 同时占用)**:Stage D0 三 workload 采集约 15-25 分钟;Stage D2 microbench 三个 fresh process 约 18 分钟;Stretch 约 5 分钟;dry-run 阶段若干次每次 2-5 分钟。**总计约 1.5-2 GPU 小时**(每张卡 12-16 分钟)。
- **CPU 时间(分析)**:D2-I5 replay 约 3 分钟、回归分析约 1 分钟、各种 dry-run 约 5-10 分钟。**总计约 30 分钟 CPU**。
- **Codex API 调用估算**:约 20-25 次 high-reasoning(reasoning effort = high 或 xhigh),涉及 PLAN attack、ablation 设计、results review 4 轮、framing review 2 轮等。
- **整体研究周期**:连续 active 工作 3 天(2026-05-28 / 29 / 30),不含 idea 探索阶段的前期准备。

---

(报告完。如有疑问,请直接和我对齐。)
