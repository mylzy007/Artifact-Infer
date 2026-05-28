# Prefill L Sweep — 苏格拉底式追问

> 目的：在你动手跑 Tier 1 之前，先把"为什么这么测"逐条逼问到底。
> 规则：本文不给方案、不下结论。每一节先复述你的当前选择 / 假设，
> 然后用一组追问把它压到墙角；如果某条假设禁不起追问，就回到设计阶段。
>
> 参考底稿：`docs/moe/prefill_drop_l_sweep_experiment_plan.md`
> 当前状态：GPU drop + bypass = 零开销 op，但 e2e 无收益；诊断指向
> decode `L_recv≈8` 在 4090 上 latency-bound。

---

## 关于实验本身的元问题（先答这两个，再答 1-6）

**M1.** 你这个 L sweep 实验，最终是要回答"drop 在更大的 prefill 上有没有用"，
还是要回答"drop 这条路在 4090/24GB 上死没死"？这两个问题听起来一样，但
推论方向相反：

- 前者要找正向 evidence（哪怕只在 L=32k 才正向），写论文时一句话就过去了。
- 后者需要排他性 evidence——证明在 24GB 可达 L 范围内**没有任何 (policy, rate, routing)**
  组合能反转。

你当前的两层方案（tail_weight 单策略、drop_rate 固定 0.3、LBG overlap 单 plan）
是不是只够回答第一个？如果是，第二个问题怎么办？

**M2.** 假设 Tier 1 跑完发现 `L* = 16384`。你接下来的两条路：
(a) Tier 2 跑 `prompt_len=2048, batch=8`，看 attention 把收益稀释多少；
(b) 直接停在 Tier 1，写"MoE-block 内部 L*=16k，full-prefill 不可达"。
**这两条路下论文 story 一样吗？** 如果不一样，那么 Tier 2 的"通过门槛"
（§5 表格里写的 `L*<8192` 才跑、`L*<32768` 限制点位）就不只是 VRAM 问题，
而是**叙事问题**——你是要给一个 hopeful upper bound，还是要给一个 closed
negative result？现在的门槛是按哪个写的？

---

## Q1. Tier 1 应该测什么——只 expert GEMM，还是包含 dispatch？

**你现在的选择**：测 `Dispatch + Experts + Combine` 整个 MoE block，不含 attention。

**追问**：

1. 为什么不只测 Experts？你在 P1 K_eff 实验中已经验证"省 expert 计算量"
   不是杠杆——既然 K_eff=6 e2e +1.9%，那 expert GEMM 时间显然不是
   bottleneck。**那么你期待 Tier 1 看到的"drop 加速"实际上来自 dispatch
   / combine 哪一段？你能不能事先写下这个预测？** 如果不写，sweep 跑完
   你会被自己事后合理化。

2. 反过来想：把 dispatch 计入 Tier 1，等于承认 dispatch 时间会**随 L 线性
   增长**。但 a2a count-exchange 的 latency 项基本是常数。**你是否假设
   payload a2a 的 bytes 项已经超过 latency 项？** 这个 crossover 本身是 L
   的函数。如果在你 sweep 的下界 `T_local=16` payload 还小于 latency，
   那 `dispatch_us` 这段曲线会是先平后涨——你能从这种形状反推出"drop 的
   收益来源"吗？还是只会得到一团噪声？

3. Phase 4 v1/v2 已经测过 `apply_drop` 自身 host↔device sync 是元凶。
   现在 GPU drop 把它做零了。**Tier 1 里你还测不测 `apply_drop_gpu_simple`
   这一段？** 如果不测，怎么回答审稿人 "how do we know drop kernel is not
   the bottleneck at small L"？如果测，它要不要单独列一段 segment？

4. 你的 §4.8 里 `dispatch_us = d0.elapsed_time(d1)`——这一段包含
   `ExpertOverlapRouter.route()` 的 CPU path（你自己在 §7.2 提到了）。
   **CPU path 在 L=16 和 L=4096 下成本一样吗？** 如果不一样，Tier 1 曲线
   斜率会被它污染。你打算把 overlap route 也算到"MoE block 时间"里，还
   是想办法剔出来？这关系到 §5 自动门控的 `L*` 是真的算力拐点，还是
   一个 CPU overhead artefact。

5. 一个更尖锐的追问：**Tier 1 的"break-even" 本身有意义吗？**
   假设 Tier 1 在 L=2048 处交叉，drop 比 baseline 快 50µs。你下游 48 层 ×
   prefill 1 步 = 2400µs。但 prefill 占 e2e 5%。
   **2400µs 的 prefill 加速，能不能让你在论文里讲一个 e2e 大于 1%的
   故事？** 还是说 Tier 1 的"正交点"本质上是个充分非必要条件——
   过得了 Tier 1 不代表 Tier 2 能赢，过不了 Tier 1 才能宣告 dead。
   如果是后者，Tier 1 真正的设计目标应该是**最大化 false-positive 率，最小化
   false-negative 率**——这跟"找精确拐点"是两件事。你的 §5 门控偏向哪边？

---

## Q2. L_recv 取值范围与刻度

**你现在的选择**：`T_local ∈ {16, 64, 128, 256, 512, 1024, 2048, 4096}`，
近似指数 + 部分线性段。

**追问**：

1. 用户问题里的 `[8, 32, 128, 512, 2048, 8192]` 是严格 4× 指数。你写的
   `16,64,128,256,512,...` 在 64→128→256→512 是 2×，其他段是 4×。
   **为什么在低 L 段加密、高 L 段稀疏？** 是因为你预判拐点更可能落在
   低 L 段、还是因为低 L 段噪声大需要更多点拟合？这两个直觉指向不同
   策略：前者要在拐点附近加密、后者要在所有 L 都加密（增加 repeat 而不是
   加点）。

2. `T_local=8` 这个点——decode 阶段每 GPU 每层就是 `L_recv≈8`。
   **你为什么不把它当 sweep 的第一个点？** 这是当前 workload 的真实
   运行状态，是 sweep 的"零点 anchor"。少了它，曲线左端没办法对照
   "decode 实测 tk/s"。如果把它加进来会破坏什么？

3. 上界 `T_local=4096` → `L_send_nominal=32k`。你在 §6.5 说 24GB 卡的
   batch×prompt_len 上限大约 32k。**这意味着 Tier 1 的上界已经卡在
   Tier 2 的物理上限了。** 如果 Tier 1 拐点在 `L_recv_max=32k` 这一点
   还没出现，Tier 2 就完全没必要跑——但你的 §5 表格里 `L* < 32768` 还
   允许跑 `{1x}`。这个边界条件是不是自相矛盾？你想用 Tier 1 的 L 上界
   反推 Tier 2 可行性，还是反过来？

4. **线性 vs 指数的本质**：如果 `total_us(L)` 是分段线性（小 L latency-bound
   常数 + 大 L compute-bound 线性），那线性刻度能看到拐点形状但浪费点；
   指数刻度对数轴上看是两条直线 + 一个折点。
   **你画图打算用 log-x 还是 lin-x？** 这决定了你应该按几何级数还是
   算术级数布点。§4.10 主图没说轴线性度。

5. 4090 上 EP-HT 的 ragged a2a 在小 L 下 NCCL 可能走 short-message 路径，
   大 L 下走 ring/tree。**你的 sweep 是否跨越了这个 NCCL 内部拐点？**
   如果跨越了，曲线上会出现一个**和 drop 完全无关的**台阶。你要怎么
   把它跟 drop 拐点区分开？需要不需要先单独跑一遍 `nccl-tests` 找出
   NCCL 自己的转折？

---

## Q3. Drop rate 怎么选——固定还是扫多个？

**你现在的选择**：固定 `drop_rate ∈ {0.0, 0.3}`，单点对照。

**追问**：

1. 你为什么挑 0.3？是从 Phase 4 v1/v2 的策略对比里继承下来的，还是因为
   你估算"超过 30% 会让 score 掉太多"？前者是 path dependence，后者是
   accuracy 约束。**如果是 path dependence，那 0.3 在 Tier 1 的相变图上
   只是一个切片——你怎么知道 0.5 不会让 break-even 提前 2× 出现？**

2. §7.4 你自己写了 effective drop fraction 可能只有 0.1（因为 all-remote token
   要保留最大 branch、local replica 不 drop）。**如果 `effective_drop_send_frac
   ≈ 0.1` 在所有 L 上都成立，那"rate=0.3"和"rate=0.5"在 Tier 1 曲线上可能根本
   不可区分。** 你打算先单独验证一下 effective fraction 对 nominal rate 的
   敏感度吗？还是直接跑主 sweep 时一并测？

3. 一个反向追问：drop rate 是不是错误的扫描维度？真正决定 a2a payload 减
   小量的是 **`effective_drop_recv_frac × L_recv`**——即"被砍掉的 a2a bytes"。
   **你能不能把 x 轴改成 `removed_recv_rows = effective_drop_recv_frac × L_recv_mean`？**
   这样不同 rate 的点都落在同一根曲线上，你可能根本不需要扫 rate。这是不是
   一个更经济、也更容易写成论文图的设计？

4. accuracy 维度：你 §8 明确说 Tier 1/Tier 2 都不跑 GSM。但 rate=0.3 + tail_weight
   在 Phase 4 v3 的真实 score 是多少？你的论文最后要不要把"L* + 对应 rate 下的
   score loss"画到同一张图上？如果要，**你现在的实验设计是不是过早地把
   accuracy 维度砍掉了**？

---

## Q4. 混淆变量怎么排除

### Q4.1 GPU clock 抖动

**你现在的做法**：`warmup_iters=5`、`iters=10`，每 cell 取 median + p10/p90。

**追问**：

1. RTX 4090 在持续负载下会因为温度降频。你 sweep 一次按 `(T_local, rate)` 笛卡尔
   积是 16 个 cell × 15 iters = 240 个 step，按每 step ~10ms 估算大约 2-3 秒。
   **这够短到不掉档吗？** 一个 sanity check：你能不能把 sweep 顺序随机化，
   然后验证"按 L 升序跑"和"打乱跑"的曲线一致？如果不一致，那就是热降频
   而不是 L 的效应。

2. `nvidia-smi --lock-gpu-clocks=2520,2520` 是更硬的隔离手段，但 4090 上锁频
   往往锁不住 boost。**你要不要先用 `nvidia-smi -q -d CLOCK` 在 sweep 进行中
   每秒采样，把实测频率作为 CSV 的一列？** 这样事后能直接看到曲线异常段
   是不是降频。

3. p10/p90 能反映抖动，但**不能反映系统性偏置**。你怎么排除"前几个 cell
   GPU 还冷、后几个 cell 已经热"？分批 sweep（每跑完一个 cell sleep N 秒）
   会不会比连跑更稳？或者反过来：连跑反而稳态更好？你打算先做哪个实验
   来确定这点？

### Q4.2 冷启动 / 第一次 kernel launch

**你现在的做法**：`warmup_iters=5`。

**追问**：

1. 5 次 warmup 对单个 cell 够吗？**但跨 cell 怎么办**——`T_local=16` warmup 完
   切到 `T_local=4096`，buffer shape 变了，Triton autotune cache 命中吗？
   你在 §4.3 说 "初始化一次最大容量 `T_cap=max_T_local`"——这只解决了
   workspace 不重分配，但 Triton kernel 的 BLOCK_M/N 调度可能仍然按
   `T_local` 选不同 config。如果是这样，每个 cell 应该独立 warmup，
   而不是依赖跨 cell 残留。你脚本里怎么实现？

2. NCCL 的 a2a 也有 first-call lazy init（buffer 分配、CUDA graph capture 等）。
   warmup_iters=5 是否覆盖到？一个 cheap check：**warmup 期间记录每次的
   total_us，看第 1 次 vs 第 5 次差多少**。如果第 5 次还在下降，warmup 不够。

3. CUDA events 的 `record()` 自身有 launch overhead，对 us 级测量是显著的。
   你测的 `dispatch_us = d0.elapsed_time(d1)` 是否减掉了 event 本身的
   `cudaEventRecord` cost？4 个 event ≈ 几 μs 的偏置，在 small L 下不容
   忽略。这个误差你打算建模、忽略，还是用 CUPTI 重新测？

### Q4.3 不同 L_recv 下 cache 行为

**追问**：

1. `T_local=16` 时 hidden = `16 × 2048 × 2B = 64KB`——L2 完全装下。
   `T_local=4096` 时 = `16MB`，远超 4090 的 96MB L2 但和 HBM 带宽相关。
   **拐点是不是其实就是 L2-residency 拐点，跟 drop 半毛钱关系都没有？**
   你打算怎么排除这个？一个对照实验：跑同样曲线但 drop=0，看 baseline 自己
   有没有相同位置的 kink。如果有，drop 那条曲线的"拐点"其实是 cache 拐点
   叠加 drop 效应——不能直接解读为"drop 在 L* 开始有效"。

2. 同样的 cache 论证适用于 EP a2a：4090 之间 PCIe Gen4 x16 ≈ 64GB/s，
   小消息下 latency dominate（~10μs/call），大消息下 bandwidth dominate。
   **NCCL 自身的 message-size scaling 已经会在某个 L 出现折点，这个折点
   位置和 drop 关系是什么？** 你要先剥离 NCCL 折点再看 drop 折点吗？

3. weight 是 `E_per_rank × N × H` ≈ `(128/8) × 768 × 2048 × 2B = 48MB`/rank。
   L2 装不下，每次 expert GEMM 都从 HBM 读 weight。**这意味着 expert
   计算时间在小 L 下是 weight-bound（不随 L 变化），大 L 下是 compute-bound
   （随 L 线性）**。drop 砍 30% 的 rows，对 weight-bound 段没用、对 compute-bound
   段有用——这本身就预测了一个拐点位置，**和你想测的拐点是不是一回事？**
   或者更激进：你测出的 L* 是不是其实在测 weight-residency 拐点？

---

## Q5. Tier 2 触发条件

**你现在的选择**：
- `L* < 8192` → 跑完整 4 点
- `8192 ≤ L* < 32768` → 跑 1-2 个 VRAM 允许点
- `L* ≥ 32768` → 最多跑 1 个 sanity
- 全段无交叉 → 不跑，写负结论

**追问**：

1. 这个表格的隐含逻辑是"L* 越大、drop 越没用、越不值得花卡时跑 Tier 2"。
   但你**实验目的的另一支**（M1 提到的 closed-negative）正好相反：
   **L* 越大的负结果越需要 Tier 2 验证**——因为审稿人会质疑"你 Tier 1
   只测了 MoE-block，attention 占大头时收益可能完全不一样"。
   你这个表格其实是为正向 story 写的。如果你想要 closed-negative，
   表格应该长什么样？

2. **不论 L\* 多少都跑一次 GSM8K 长 prompt** 这个想法你怎么看？
   GSM8K 长 prompt 可能是 1k tokens × batch 4 = 4k total。这个点你
   **不管 Tier 1 结论如何都已经知道实际是不是 e2e 加速**。它的成本是
   10 分钟卡时，但它锁定一个真实可写的数据点。
   反对意见：GSM8K decode 长度不可控，prefill speedup 会被 decode 时间稀释，
   单一数据点说明不了什么。你愿意付这 10 分钟吗？为什么？

3. **触发条件应该看 L\*，还是看 Tier 1 的 e2e-extrapolation？**
   Tier 1 给你 `dispatch_us(L)`、`experts_us(L)`、`combine_us(L)`。
   你能不能直接外推到 Tier 2 的 prefill 时间？
   `predicted_prefill_us = 48 × MoE_block_us(L) + attn_us(L) × 48`
   如果 attn_us 你已经知道（baseline 实测），那 Tier 2 跑不跑就是
   "外推预测的 prefill_speedup 大于多少 pp 才值得花卡时"——这是个
   threshold，不是 L*。**这个 threshold 你打算设多少？1%？3%？**

4. **门控本身有没有可能误判？** 假设 Tier 1 的 LBG sweep 拐点在 8k，
   但 controlled_balanced sweep 拐点在 4k。哪个触发 Tier 2？两个都触发
   还是 conservative max？你的 §5 没区分。

---

## Q6. 论文 story 怎么写

### Q6.1 如果 L\* 在可达区间之外

**追问**：

1. "无收益"和"在我能测的范围内无收益"在论文里**字面意义不同**。
   你能写得多强？最强能写到 "在 24GB 单卡可达 prefill 体量内 (`L_recv ≤ 32k`)，
   Token-replica drop 在 EP-HT MoE 推理上没有 e2e 收益" 还是更弱？
   你能不能现在就把这句话的精确措辞写出来？写出来之后回看实验设计，
   你的 sweep 上界 32k 够支撑这个 claim 吗？

2. 论文的 contribution 是什么？三选一（也可以是其他）：
   (a) **方法论**：提出 GPU drop 零开销实现 + bypass，作为开源贡献；
   (b) **诊断**：定位 EP MoE decode latency-bound 的精确机制；
   (c) **boundary**：给出 drop 适用边界 L*（即使是负的）。
   你现在的实验设计**强支撑哪一个**？如果是 (b)，那 Tier 1 应该有更多
   roofline 分析；如果是 (c)，那 sweep 必须覆盖到一个明确的折点（
   即使在外推区间）；如果是 (a)，那 Tier 1 数据图反而不是核心。
   你**打算押哪边**？

3. **负结果的可信度问题**：审稿人最常见的反驳是"你 drop policy 不够好"
   或"你没试 receive-side drop"。你 §8 明确不做 Triton fused drop / 不
   GPU 化 overlap route。**那么写论文时，你怎么 frame 这些"未做"——
   是 future work、还是 ablation gap？** 如果是 future work，你需要先
   argue"这些方向不会改变结论"；这个 argue 你能现在就给一段话吗？

### Q6.2 如果 L\* 落在可达区间

**追问**：

1. 假设 Tier 1 `L*=4k`，Tier 2 prefill speedup = 8%。**但 prefill 只占
   GSM8K e2e 5%**——8% × 5% = 0.4% e2e。这是个能讲的故事吗？还是
   你要把 workload 从 GSM8K 换到一个 prefill 占比更大的 task？
   你之前在 §0.4 写"提高 batch / 砍 context 不可行"——这个约束在
   "L* 可达"情景下是否要重新评估？比如换成 LongBench / 长文档 QA，
   prefill 占比可以到 30%+。这算 scope creep 还是合理 pivot？

2. 如果 L* 可达，你下一步还会做什么？论文 story 的逻辑应该是：
   "L* 存在 → 在合适 workload 上 X% 收益 → 进一步 ablation"。
   **"进一步 ablation"是什么？** policy 扫描？rate 扫描？receive-side
   drop？你需要先想清楚这是 1 周还是 4 周的工作量，再决定 Tier 1 当下
   是不是要扩 scope。现在写一个"if positive then ..." 的 contingency 计划。

3. **可达情景下的 risk**：L* 可达意味着 drop **某些时候有用**——这反而
   让 narrative 更复杂。你得回答 "什么时候有用、什么时候没用、怎么自动
   切换"。这是一个工程系统问题，比"drop 不行"难写得多。你确定你
   preferred outcome 是哪个？这个 preference 会不会偷偷影响你 sweep
   设计的选点和 effort 投入？

---

## 收尾元追问

**Z1.** 在跑 Tier 1 之前，你能不能写一份**预注册（pre-registration）**：

- 预测 Tier 1 曲线在每段 L 的形状（先平后涨 / 单调 / 双拐点 / ...）
- 预测 L* 大致位置（数量级即可）
- 预测 effective_drop_recv_frac 在小 L vs 大 L 的差异
- 预测哪段 segment（dispatch/experts/combine）主导拐点

写下来后跑实验，事后对照"哪些预测错了"。这个动作的成本是 20 分钟，
但能极大提升结果可解释性，也能保护你不被 post-hoc rationalization
带偏。**你愿意做吗？** 如果不愿意，原因是什么？

**Z2.** 这个实验如果最终是负结果，**你的下一步是 K_eff、receive-side
drop、还是直接换课题？** 在写实验代码之前先想清楚这个，能帮你判断
"Tier 1 该做多严"——如果下一步是换课题，Tier 1 越快越好（不要 LBG
+ controlled 都跑、不要扫 8 个 T_local）；如果下一步是 receive-side
drop，那这次 Tier 1 的 infra 要留好接口。**哪个？**

---

## 你需要先回答的最小问题集

如果你只想先答 3 个问题再决定怎么改方案，建议是：

- **M1**：实验是为正向 story 还是 closed-negative？
- **Q3.3**：x 轴是不是应该改成 `removed_recv_rows` 而不是 `L_recv`？
- **Z1**：愿不愿意做 pre-registration？

这三个问题答完后，§4-§6 的细节会自动收紧或放松。

---

# 我的回答（2026-05-26）

> 每条按：推荐方案 / 理由 / Trade-off / 额外考虑。

## M1. 正向 story 还是 closed-negative？

- **推荐**：定位为 closed-negative，框架是"在 24GB 可达 prefill 体量内 drop 没有 e2e 收益"，Tier 1 如果意外正向再 pivot。
- **理由**：你已经做了 CPU drop、GPU drop、K_eff 三轮，都是负或持平；负结果 + 诊断已经是当前 evidence 最强的方向。再去赌"扫到大 L 就翻盘"风险高，reviewer 也会觉得在凑正向。
- **Trade-off**：如果 L* 意外可达，你会因为没扩 policy/rate 而少一个 ablation 维度。
- **额外考虑**：先想好投稿场景——closed-negative 在 workshop / systems track 比顶会容易卖。

## M2. L*=16384 时两条路一样吗？

- **推荐**：不一样，必须走 (a)——跑至少 1 个 Tier 2 点。
- **理由**：Tier 1 只覆盖 MoE-block；不跑 Tier 2 就 claim "在 prefill 上没用"，审稿人一定问 attention 稀释。一个 Tier 2 点 ≈ 30 分钟卡时，是负结果可信度的最低必需投入。
- **Trade-off**：若 L*≥32k 已经 OOM，那只能跑最大可达点作为 sanity，并把 claim 缩到该上限。
- **额外考虑**：哪怕选 closed-negative 框架，这 1 个 Tier 2 点也不能砍。

---

## Q1.1 Tier 1 收益来源能否预先写下？

- **推荐**：能，且必须写。预测：drop 的加速主要来自 `experts_us`（GEMM FLOPs），dispatch/combine 是二阶。
- **理由**：K_eff=6 e2e 几乎不动说明小 L 下 expert GEMM 是 weight-bound；只有当 L 大到 GEMM 转 compute-bound 时，砍 30% rows 才会线性砍时间。dispatch/combine 的字节减少更多体现在带宽利用率而非 wall time。
- **Trade-off**：若实测打脸（dispatch 占主导），那是更有价值的诊断——值得专门一章。
- **额外考虑**：把这条预测写进 pre-reg；事后对照偏差 ≥30% 必须解释。

## Q1.2 dispatch 里 payload bytes vs latency？

- **推荐**：假设 L_recv ≤ 1024 是 latency-dominated，≥4096 是 bandwidth-dominated，曲线会有独立 kink。
- **理由**：8×4090 a2a 单调用 latency ~10μs，bandwidth ~50GB/s；H=2048, bf16 时 1024 行 ≈ 4MB → 80μs，刚好同量级。
- **Trade-off**：dispatch 自己的 kink 会污染 drop 的 kink 判定。
- **额外考虑**：必须先单独跑一遍 baseline-only sweep 确认 dispatch_us(L) 的形状，再叠 drop 曲线。

## Q1.3 要不要单独测 apply_drop kernel？

- **推荐**：在主 sweep 之外单做 1 个 cell 的 5-段计时（drop 单列），结论数值放附录。
- **理由**：审稿人一定问"你怎么证 GPU drop 是零开销"；有这个图能一句话顶回去。
- **Trade-off**：额外 CUDA event 在 small L 会引入 ~2μs 偏置——所以**不要**把它加进主 sweep 的 timing region。
- **额外考虑**：可以借此把 v1/v2 的 CPU drop 在同一脚本下当对照点，三条线一起画。

## Q1.4 ExpertOverlapRouter CPU path 算不算进 MoE block？

- **推荐**：剥出去——overlap routing 在 timed region 之前 precompute 一次，feed 结果进 dispatch。
- **理由**：CPU path 与 drop rate 正交、与 L 弱相关，留在里面会给曲线加个非线性噪声底，让 L* 估计偏。Tier 2 自然会把它算进去；Tier 1 的目的是隔离算法效应。
- **Trade-off**："不够真实"——但 Tier 1 本来就是 microbench，真实性交给 Tier 2。
- **额外考虑**：在 writeup 里明确说明这个剥离，否则 reviewer 看 Tier 1→Tier 2 数字不接会问。

## Q1.5 Tier 1 是 sufficient 还是 necessary？

- **推荐**：当 screening filter——优化 low false-negative，接受 high false-positive。
- **理由**：过 Tier 1 → 值得跑 Tier 2；不过 Tier 1 → dead end。两侧成本极度不对称（误杀 = 整条路放弃；误过 = 多花 1 小时卡时）。
- **Trade-off**：意味着 Tier 1 的"通过门槛"应该宽松——`delta_us ≤ +5%` 而不是要求严格 crossing。
- **额外考虑**：把这条放进 §5 门控表，把"全段无交叉"细化为"全段 delta>+5% 才算 STOP_NEGATIVE"。

---

## Q2.1 低 L 加密 vs 全程几何？

- **推荐**：改成纯 4× 几何级数 `{8, 32, 128, 512, 2048, 8192, 32768}`，拐点附近事后用 bisection 加点。
- **理由**：log-x 图上等几何间距 = 等信息密度；当前不规则间距是直觉偏置，没充分理由。
- **Trade-off**：丢掉了"低 L 段先密"的潜在好处，但低 L 噪声本来就大，加 iters 比加点更划算。
- **额外考虑**：跑完先看曲线再决定补点位置，别提前撒太多。

## Q2.2 为什么不放 L=8？

- **推荐**：加 `T_local=1`（L_recv≈8），仅做左端 anchor，不参与 L* 拟合。
- **理由**：decode 的真实运行点，能直接对照 Phase 4 实测 28.83 tk/s；曲线少这个点，论文左端没有锚定。
- **Trade-off**：L=8 噪声极大，median 可能不稳。
- **额外考虑**：在 anchor 点把 iters 提高到 30+ 来压噪声。

## Q2.3 32k 上界自相矛盾？

- **推荐**：Tier 1 上界提到 65k（T_local=8192），Tier 2 ceiling 留在 32k。
- **理由**：上界必须严格大于触发阈值，否则边缘 L* 不可验证。workspace `8*8192*8` 行 × 2048 × 2B ≈ 2GB，单层够用。
- **Trade-off**：65k 可能让 Triton autotune 走不同 path，需要 warmup 多 1-2 iters。
- **额外考虑**：把 §5 表里 `L*<32768 跑 1x` 这行删掉，统一用 `L*<24576` 才跑 Tier 2。

## Q2.4 log-x 还是 lin-x？

- **推荐**：主图 log-x + linear-y；inset 用 lin-x 放大 L* 附近。
- **理由**：log-x 揭示 scaling regime（latency-flat → bandwidth-linear）；lin-x inset 让 crossing 看着直观。
- **Trade-off**：两张图占版面，不过 inset 不占额外 figure 槽。
- **额外考虑**：千万别 log-log——会把 crossing 拍扁看不见。

## Q2.5 NCCL 内部 transition？

- **推荐**：先单独跑一次 `nccl-tests alltoall` 在 8×4090 上扫消息大小，把 transition 点（一般 64KB-256KB）overlay 到 dispatch_us 图上。
- **理由**：没这条参考线，dispatch_us 的任何 kink 都可能被审稿人质疑为 NCCL artefact。
- **Trade-off**：30 分钟额外工作。
- **额外考虑**：H=2048, bf16 下 1024 行 = 4MB > 256KB，主 sweep 大概率全在 ring 阶段——这条 overlay 反而能 reassure 自己。

---

## Q3.1 为什么 0.3？

- **推荐**：保留 0.3 作为主 rate，但在最大 L 上加 1 个 rate=0.5 的点作 monotonicity check。
- **理由**：0.3 是 v1/v2 已验证 score 不掉的最大 rate；换 rate 等于把 accuracy 维度强行引入，违背 §8 边界。
- **Trade-off**：单点 sensitivity 不算严格，但比完全不测好。
- **额外考虑**：若 rate=0.5 在大 L 让 L* 显著左移，说明值得把 rate 也扫——但这是 Phase 5 的事。

## Q3.2 Effective fraction ≈ 0.1 怎么办？

- **推荐**：主 sweep 前先单跑一次 effective_drop_send_frac(L) profile，5 分钟出结果。
- **理由**：如果 effective 在所有 L 都只 0.1，"30% drop"这个 label 就是误导，需要换说法或换 policy。
- **Trade-off**：可能发现 tail_weight 的 effective rate 远低于 nominal，要回去改策略——但这是好事，提前发现。
- **额外考虑**：effective fraction 跟 L 的关系本身就是一张诊断图（remote token 比例随 L 变化）。

## Q3.3 x 轴改成 removed_recv_rows？

- **推荐**：主图 x 轴仍用 L_recv（实验旋钮），但加一张副图 x=removed_recv_rows 把多 rate / 多 policy 折叠成单曲线。
- **理由**：L_recv 是 reviewer 期待看到的"控制变量"；removed_rows 是 derived 量，做副图既不抢戏又能强证"效应正比于砍掉的 bytes"。
- **Trade-off**：副图成本极低（同一份数据换算）。
- **额外考虑**：若副图真的折叠成一条线，那是论文里最有冲击力的一张图——优先级提高。

## Q3.4 accuracy 维度过早砍？

- **推荐**：Tier 1/2 不动；最终 writeup 时加一张 composite 图：x=L*, y=Δ(prefill_us), 用 v1/v2 已有 score 给点上色。
- **理由**：accuracy 数据已经有，跑性能时不要重测；composite 让一张图同时表达 perf × accuracy。
- **Trade-off**：composite 是 post-hoc；reviewer 可能问 score 是不是同 config 测的。
- **额外考虑**：记录清楚 v1/v2 测 score 时的 (policy, rate, placement) 是否完全等于当前 sweep。

---

## Q4.1 GPU clock 抖动

- **推荐**：(1) sweep 顺序随机化；(2) `nvidia-smi --lock-gpu-clocks=2200,2200` 尝试（4090 boost 锁不住就放弃）；(3) 每秒采 SM clock 写 CSV；(4) 单次 sweep ≤3 min 然后强制 sleep 30s 冷却。
- **理由**：随机化是检测降频污染的唯一可靠手段；其他都是软辅助。
- **Trade-off**：sleep 让 sweep 总时间翻倍。
- **额外考虑**：分析时画 `total_us vs cell_order`，无相关性才能交。

## Q4.2 冷启动

- **推荐**：每个 cell 独立 warmup 5 次；记录 warmup 第 1 次和第 5 次，若差 >10% 增到 10 次。
- **理由**：Triton autotune 按 shape 缓存，跨 cell 不命中；NCCL a2a 也有 lazy init。
- **Trade-off**：增加 wall time，但抑制冷启偏置。
- **额外考虑**：CUDA event 自身 ~1μs cost，small L 别忽视——写进 measurement uncertainty。

## Q4.3 Cache 行为

- **推荐**：先跑 drop=0 only 全程 baseline sweep，再跑 drop 全程；主曲线画 `delta_us(L) = drop_us - baseline_us`，把所有共有 cache/NCCL artefact 减掉。
- **理由**：subtract baseline = 消除所有 L 相关但 drop 无关的伪信号；这是这次实验最关键的方法论选择。
- **Trade-off**：delta 图对非专家不直观；要配 total 图。
- **额外考虑**：补一张 weight-residency roofline 注释 L 在什么位置 weight 从 L2 溢出到 HBM。

---

## Q5.1 closed-negative 友好的门控表？

- **推荐**：增加一行 "`L* ≥ 24576` 或全段无交叉 → 跑 1 个 Tier 2 max-reachable 点"。
- **理由**：closed-negative claim 必须有"最大可达 L 处实测仍无收益"的 e2e 数据点支撑。
- **Trade-off**：30-45 分钟额外卡时。
- **额外考虑**：把这点的 batch×prompt 选定为最贴近 LongBench 一段的 shape，顺便给 future-work 铺路。

## Q5.2 不论 L* 都跑 GSM8K sanity？

- **推荐**：跑，1 个点，batch=4 prompt≈1k max_new_tokens=128，匹配 Phase 4 历史 config。
- **理由**：和 phases v1/v2/GPU drop 历史数据 apples-to-apples，回归验证 + 论文末尾连续表必备。
- **Trade-off**：10-30 分钟。
- **额外考虑**：也是 GPU drop 实现没有回归的最后一道关——别省。

## Q5.3 用 L* 还是用外推 e2e 阈值触发？

- **推荐**：用外推阈值——`predicted_prefill_speedup ≥ 5%` 才跑 Tier 2。
- **理由**：L* 本身没说服力；5% prefill speedup 是"图上看得见"的最低门槛。
- **Trade-off**：5% 是 arbitrary，要在 pre-reg 里明确写。
- **额外考虑**：外推公式写在 pre-reg 里：`predicted = (MoE_block_speedup × MoE_fraction_of_prefill)`，MoE_fraction 用 Phase 3 实测的 ~70%。

## Q5.4 LBG vs controlled 不一致？

- **推荐**：以 LBG L* 为触发依据，controlled L* 仅作 lower bound 报告。
- **理由**：LBG 是生产路径，controlled 是理想化对照。
- **Trade-off**：若两者差 ≥2×，gap 本身就是 finding——值得单独写一段。
- **额外考虑**：结果表里两列并排展示，让审稿人看到 "idealized routing 帮 drop 多少"。

---

## Q6.1.1 最强 claim 措辞？

- **推荐**：「在 8×RTX 4090 24GB、Qwen3-30B-A3B、EP-HT owner-local 配置下，`tail_weight` policy 在 drop rate ≤ 30% 时，对 prefill token 总量 ≤ 32k 的 workload 无 e2e 收益。」
- **理由**：每个限定词都有数据支撑，没有外推。
- **Trade-off**：scope 窄；这就是诚实负结果的代价。
- **额外考虑**：补一句 "consistent with our CPU drop and K_eff hard-cap ablations"，三个独立负实验同向，credibility 翻倍。

## Q6.1.2 contribution 押哪边？

- **推荐**：主推 (b) 诊断，附 (c) boundary，(a) GPU drop kernel 当工具放附录。
- **理由**：(b) 是独有数据：从 CPU drop sync → GPU drop bypass → L* boundary 的完整诊断链；其他人复刻不了这条路径。
- **Trade-off**：(b) 主推的论文常被嫌"小"，投顶会风险高。
- **额外考虑**：配一张 MoE inference roofline 图（HBM bw / NCCL bw / GEMM compute 三条线 + 当前 workload 点），这是 (b) 的封面图。

## Q6.1.3 未做方向的 framing？

- **推荐**：用 roofline 论证"即便把这些方向做到极致，理论上界也低于观察到的 gap"。例如 dispatch+combine 占 MoE block ≤30%，全砍也只省 30%，仍不够补上 attention dilution。
- **理由**：roofline 上界 argument 是最难被 reviewer 反驳的；省下他们要求做 ablation。
- **Trade-off**：roofline 数算错就翻车，要严谨。
- **额外考虑**：现在就把这段话写出来——如果写不出来，说明这些方向其实还能翻盘，要回头补实验。

## Q6.2.1 L* 可达但 GSM8K e2e 只 0.4%？

- **推荐**：pivot 到 prefill-heavy workload（LongBench summarization、长 doc QA），prefill 占比 ≥30% 时再报。
- **理由**：0.4% e2e 在 noise 之内，不可发表；同 kernel 在 30% prefill share workload 上 ≈ 2.4% e2e，可发表的下限。
- **Trade-off**：算 scope creep，但 positive 已经触发，应当 follow through。
- **额外考虑**：换 workload 前确认 model 上下文长度够（Qwen3-30B-A3B 32k 上下文应该够 LongBench）。

## Q6.2.2 positive 情境下进一步 ablation？

- **推荐**：预设 2 周 budget = receive-side drop + rate sweep + 1 替代 policy + 1 替代 workload。
- **理由**：positive 单点不够，paper 至少要 4 个数据维度；2 周是这 4 维度的最低投入。
- **Trade-off**：若 L* 卡边缘，可能白花 2 周。
- **额外考虑**：Tier 1 脚本把 policy/rate 做成 CLI 参数，positive 时一行命令切换。

## Q6.2.3 preference 偷偷影响设计？

- **推荐**：承认 prefer closed-negative，明确在 pre-reg 里写下 L* 阈值和决策树并 commit 到 git。
- **理由**：pre-reg 是抵御 confirmation bias 的唯一机制。
- **Trade-off**：限制 post-hoc 灵活性。
- **额外考虑**：把 pre-reg 给一位同事看，让他签字（git PR review 也行）——social commitment 更硬。

---

## Z1 Pre-registration

- **推荐**：做，1 页 markdown，包含 (i) 4 段预测曲线形状、(ii) L* 数量级猜测、(iii) effective_drop_recv_frac 在小/大 L 的差异、(iv) 决策树。
- **理由**：20 分钟成本，事后对照偏差就是论文 discussion 章节的素材。
- **Trade-off**：略增 friction。
- **额外考虑**：写一行 "what would falsify my closed-negative hypothesis"——例如 "L_recv=8 上 drop 比 baseline 慢 < 2%"，这是真正的科学态度。

## Z2 负结果后下一步？

- **推荐**：pivot 到 receive-side drop / 重新 frame 成 "drop placement 的代价分析"，**不要**整个换课题。
- **理由**：Tier 1 infra、诊断链、GPU drop kernel 都可复用；全换课题等于扔掉这些沉没成本。给 receive-side 2 周硬时间盒，无正向信号就 stop。
- **Trade-off**：可能再花 2 周仍负。
- **额外考虑**：Tier 1 脚本现在就要留 receive-side hook（一个 callback 在 combine 之前砍 rows），不然 2 周后还要重写。

---

## 最值得马上做的 3 件事

1. **写 pre-registration**（Z1）——20 分钟。
2. **跑 effective_drop_send_frac(L) profile**（Q3.2）——5 分钟，可能直接改变 sweep 配置。
3. **跑 baseline-only sweep**（Q4.3 + Q2.5）——1 小时，是 delta 图的前提。

跑完这三步再开主 sweep，能让 Tier 1 数据可解释性翻一倍。
