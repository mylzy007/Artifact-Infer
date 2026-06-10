# MoE 推理 EP all-to-all 通信压缩：调研报告与方向规划

> 范围：多卡 Expert Parallelism（EP）**推理**场景下，**纯降低 all-to-all 通信量**的方法。
> 约束（本研究的取舍基准）：(1) 降通信量；(2) 允许用额外计算换通信；(3) 允许有损/近似算法。
> 不聚焦：负载均衡 / 专家放置（已在前期工作中触及天花板，本报告仅作背景）。

---

## 0. 一句话结论

无损系统优化（overlap、拓扑感知 all-to-all、FP8/FP4 量化）已被 DeepEP、COMET、MoRI-EP 等做到接近硬件极限；**真正的空白是"在冻结的预训练模型上、推理期、用离线标定的有损压缩器（low-rank / 学习式投影 / 稀疏化 + 误差补偿）压缩 all-to-all 的 payload"**。这条线最契合本研究的三个约束，且与现有 overlap / 量化 / 放置优化全部正交可叠加。

---

## 1. 背景：EP 推理里 all-to-all 到底有多贵

在 EP 下，attention 是 DP 形态（每卡冗余权重、各自处理本地 token），只有 routed FFN 被 EP 切分，靠两次 all-to-all（dispatch + combine）在卡间搬运 token。多份工作量化了这个开销：

- COMET（MLSys 2025, arXiv 2502.19811）：在 8×H800 + Megatron-LM 上，一个 MoE 层的跨卡通信可占整模型执行时间的约 **47%**。
- LSH-MoE（NeurIPS 2024, arXiv 2411.08446）：训练期 all-to-all 占比平均约 **45%**。
- 华为 CloudMatrix384（arXiv 2508.02520）：dispatch+combine ≥ **25%**。
- LatentMoE（NVIDIA）：在 GB200 NVL72 上通信:计算比可达 **9:1**。

**对本研究硬件的含义**：8×RTX 4090 无 NVLink、走 PCIe/受限互联，通信瓶颈比 H100 NVLink 更严重，因此"用计算换通信"在本硬件上的净收益**更大**——这是一个对我们有利的实验前提，也必须在论文里标注（换到高带宽互联收益会缩小）。

**通信量的基本分解**（每个 token-expert 对的字节数）：

```
bytes ≈ 维度 d × 每元素位宽 b
```

- 量化压第二项 b（16bit→8/4bit），维度不变；
- 低秩压第一项 d（d→ℓ），位宽不变；
- 二者相乘 → 可乘性叠加。这是后面方向 A 的核心立足点。

---

## 2. 已有方法分类调研

### 2.1 payload 低秩 / 降维压缩（与本研究最相关）

| 工作 | 阶段 | 手段 | 有损? | 结果 |
|---|---|---|---|---|
| SCoMoE (ICLR'23) | 训练 | 序列维 / 特征维线性投影降维 + token clustering | 是 | vs GShard 1.44×；OPUS-100 +2.8 BLEU/1.25× |
| **LSH-MoE (NeurIPS'24)** | **训练** | LSH 聚类只传质心 + 残差误差补偿 | 是 | 约 20% 码率；RoBERTa-MoE 1.6×、T5-MoE 2.2× 收敛加速，精度近无损 |
| LatentMoE / MoLAE (NVIDIA Nemotron 3) | 架构/预训练 | 投影到潜维 ℓ 做 dispatch/combine/专家计算 | 是 | 流量与权重读取降 d/ℓ；省下预算加大 N、K |
| Breaking MoE Trilemma (arXiv 2510.02345) | 训练 | 动态专家聚类 + 共享 base + INT4 低秩残差 + 层次路由 | 是 | 参数减约 80%、吞吐 +10–20% |

**关键判断**：低秩 / 学习式 payload 压缩**全部在训练期或架构期**，靠权重与压缩器协同适应。**没有人在冻结的预训练模型上、推理期做 payload 低秩压缩**——确认空白。

### 2.2 通信量化（FP8/FP4/MXFP4）

| 工作 | 阶段 | 手段 | 结果 |
|---|---|---|---|
| DeepEP (DeepSeek) | 推理 | FP8 dispatch + BF16 combine | 产线级 EP 通信库 |
| **MoRI-EP (AMD MI355X)** | 推理 | MXFP4 dispatch + FP8 blockwise combine | 往返带宽 **2.56×** 缩减（28,672→11,200 B/token） |
| MegaScale-MoE (EuroSys'26) | 训练 | FP8 (E4M3) per-token 量化 + FP32 reduction | MFU 1.88× |
| MxMoE (ICML'25) | — | 敏感度驱动混合精度（侧重权重/计算，非通信 payload） | 2.25-bit 比 GPTQ 低 2.4 PPL |

**关键判断**：量化只压"每元素位宽"，不利用 token 间冗余或低秩结构。**combine 端目前产线只敢 FP8（不敢 FP4）→ combine 端仍有精度余量**。低秩 / 稀疏化可与量化**乘性叠加**：先低秩投影到 ℓ 维，再把 ℓ 维 FP8/FP4 量化。

### 2.3 combine 端（专家输出）传输前压缩

**空白确认**：没有任何论文专门在第二次 all-to-all（combine）前对专家输出做 top-magnitude 稀疏化 + 残差修正。combine 与 dispatch 张量同尺寸、同样昂贵，但产线只对它做 FP8。误差反馈（error feedback，Seide 2014 / Stich 2018 / EF21）在分布式训练里成熟，**没被搬到 MoE EP 推理的 combine 端**。这是几乎零专门论文的干净切入点。

### 2.4 自适应 top-K / 动态路由

| 工作 | 手段 |
|---|---|
| Ada-K (ICLR'25, arXiv 2410.10456) | RL allocator 按 token 动态定专家数；FLOPs −25%、推理 +20% |
| Top-P (arXiv 2403.07652) | 按路由置信度累积到阈值 p 定专家数 |
| DA-MoE (arXiv 2409.06669) | 按 token 重要性（attention 权重）分配专家数 |

**关键判断**：全部在改激活专家的**数量**。**没有人保持数量不变、改"每个 token 传多少比特"——即码率自适应而非数量自适应**。这是方向 C 的差异化空白。

### 2.5 计算-通信重叠 / kernel 融合（无损、正交、可叠加背景）

Tutel（2DH all-to-all + pipelining）、FasterMoE（dynamic shadowing：传专家权重而非 token，1.37×）、ScMoE（跨层 shortcut，overlap 70–100%，推理 1.82×）、COMET（细粒度依赖分解，端到端 1.71×）、Flux（通信融进 GEMM epilogue）、EPS-MoE（推理期 pipeline，prefill +52.4%）、DeepEP hook-based overlap。
**这些不改变所传信息内容 → 与有损 payload 压缩正交可叠加，是基线背景，不是竞争对手。**

### 2.6 拓扑感知 / 去重 / 放置调度（无损降量背景）

DeepSpeed-MoE 层次 all-to-all、HierMoE（token 去重）、Speculative/Semantic MoE（推测式预调度提高本地激活率，专家层延迟 −41.8%）、**ExFlow（arXiv 2401.08383）**：利用层间专家亲和性把每层两次 all-to-all 降为一次，**推理期、冻结模型、无需微调**，报告 all-to-all 通信减少 56%–67%、吞吐 +60%–120%。
**ExFlow 是"推理期 + 冻结模型"的最强无损先例**——它证明在冻结模型上做推理期通信优化可行，但它走"减少 all-to-all 次数 / 放置"，**没碰 payload 压缩**，与我们正交。

---

## 3. 重点论文精读（可直接参考的方法）

### 3.1 LSH-MoE（NeurIPS 2024）——有损 payload 压缩 + 误差补偿的算法原型

**这是与本研究算法思路最接近的工作，值得详细拆解。**

**核心观察**：MoE 中送入 all-to-all 的 token 表示之间存在高相似性（很多 token 语义接近）。与其逐 token 传 d 维向量，不如把相似 token 聚成簇，只传簇的**质心（centroid）**，到接收端再"展开"回每个 token。

**方法分三步**：

1. **LSH 聚类（用计算换通信的核心）**：用 cross-polytope LSH（一种把相近向量大概率映射到同一桶的哈希）对每层 dispatch 前的 token 做哈希分桶。用 6 个 hash function 时压缩率约 20%（即只传约 1/5 的向量）。LSH 的好处是聚类几乎零训练、在线可算，开销远小于 K-means。
2. **只传质心过 all-to-all**：每个桶只把质心发到目标专家，专家对质心做 FFN。通信量从"token 数"降到"桶数"。
3. **残差误差补偿（精度的关键）**：直接用质心输出近似每个 token 的输出会掉精度。LSH-MoE 记录每个 token 相对其质心的残差 e = x − centroid，在 combine 端用这个残差做一阶修正，把输出还原回 token 级。论文消融显示：**去掉误差补偿，困惑度上升约 0.3 点**——这一步是有损方法能"近无损"的命门。

**结果**：RoBERTa-MoE 收敛加速 1.6×、T5-MoE 2.2×，精度几乎无损。

**我们能直接借鉴的三点**：
- "**先压缩 payload、专家算压缩后的表示、combine 端再还原**"这个三段式骨架可以整体搬到推理期。
- "**残差误差补偿**"是把"有损"变"近无损"的关键机制，我们的 combine 端稀疏化也应配一个残差/校正项。
- 它**把"相似性冗余"作为压缩来源**——这与低秩压缩"维度间相关性冗余"是互补的两种冗余，提示我们可以聚类 + 低秩组合。

**它和我们的根本区别（差异化定位）**：LSH-MoE 是**训练期**做的——它在训练中让权重适应"质心近似"，所以专家本就学会了在质心上工作。我们要做的是**冻结模型 + 推理期**，权重不动，所以不能照搬，必须靠"离线标定的压缩器 + 更强的误差补偿"来弥补权重没适应这件事。这正是难点，也正是新意所在。

### 3.2 Communication Compression for Tensor Parallel LLM Inference（arXiv 2411.09510）——证明"推理期 + 冻结模型 + 学习式压缩"可行

**这篇不是 MoE，而是张量并行（TP），但它是我们方法路线可行性的最强间接证据。**

**它做了什么**：在 TP 推理中（多卡协同算一个大矩阵乘，卡间要传 activation），对**选定的 activation** 做细粒度量化压缩，**全程冻结模型、纯推理期**。报告：选定激活压缩 3.5–4.5×，TTFT 最多降 2×，精度退化可忽略。

**最关键的一句**：它引用 Bian et al. 2024（"Does compressing activations help model-parallel training?"）发现——**学习式 autoencoder 压缩优于朴素量化 / TopK**。也就是说，在"压缩 activation"这件事上，学一个小的编码器/解码器（投影器）比简单截断或量化更划算。

**我们能直接借鉴的两点**：
- 它**证明了"推理期、冻结模型、压缩卡间 activation"这条路在 TP 上有效**——我们要做的是把它从 TP 搬到 MoE 的 EP all-to-all，域不同（EP 有动态路由、combine 端误差更敏感），但可行性有先例背书。
- "**学习式投影 > 朴素量化/TopK**"这个结论直接支持我们用"离线标定的低秩/autoencoder 投影器"而不是只做量化。

**它和我们的区别**：它是 TP（传的是层间 activation，静态切分），我们是 EP（传的是按 token 动态路由的 token、且有 combine 回传），MoE 的路由动态性和 combine 端误差敏感性是它没有的新问题。**把这条 TP 上验证过的思路迁移到 MoE EP，本身就是贡献点。**

---

## 4. 空白总结（机会矩阵）

| 维度 | 训练期 | 推理期（冻结模型） |
|---|---|---|
| 低比特量化 payload | MegaScale-MoE, FP8-Flow | **DeepEP FP8, MoRI MXFP4/FP8（成熟）** |
| 低秩 / 学习式压缩 payload | SCoMoE, LSH-MoE | **空白 ← 机会 A** |
| combine 端稀疏化 + 残差 | error-feedback 训练文献 | **空白 ← 机会 B** |
| 码率自适应（非数量自适应） | 无 | **空白 ← 机会 C** |
| 架构期潜空间投影 | LatentMoE（需预训练） | 不适用冻结模型 |
| 减少 all-to-all 次数 / 放置 | — | ExFlow（无损，正交，可叠加） |

---

## 5. 可尝试的方向（详略分明）

### 【略】不太靠谱 / 不优先的方向

- **方向 D：码率自适应路由单独成文**——把"动态 top-K"改成"动态码率"虽是空白，但单独做创新点偏薄、容易被审稿人视为 trick；**更适合作为方向 A 内部的一个模块**，不单独立项。
- **方向 E：重做通信感知负载均衡 / 放置**——前期已触天花板，且 Occult/GRACE-MoE/MoETuner/ExFlow 已较拥挤，略过。
- **方向 F：纯无损通信压缩（如 ZipCCL 思路）**——无损上限仅约 1.33×（Qwen3-A22B 实测），收益太小，不符合"可有损换大压缩比"的约束，略过。
- **方向 G：架构期潜空间投影（仿 LatentMoE）**——需从头预训练，与"冻结模型"前提冲突，且 NVIDIA 已做，略过。

### 【详】优先方向：推理期 EP payload 有损压缩（A + B + C 融合）

这是本研究接下来主攻的方向，把它定义为一个统一框架：**"重要性感知的 EP payload 压缩"——在冻结的预训练 MoE 上，推理期对 dispatch 和 combine 的 payload 做离线标定的低秩压缩 + 稀疏化 + 误差补偿，码率按 token 重要性分档，并与 FP8/FP4 量化乘性叠加。**

#### (1) 压缩器的离线标定（一次性、用计算换通信）

- 用一小批标定集（512–2048 条 prompt，覆盖目标域）跑一遍冻结模型，对每个 MoE 层、dispatch 端收集 token 激活矩阵 X ∈ R^{N×d}。
- 对每层做截断 SVD / PCA，得投影器 P_down ∈ R^{d×ℓ} 与重建器 P_up ∈ R^{ℓ×d}。ℓ 由谱能量保留阈值（如 90–95%）决定。可借鉴 MoLAE 的两步 Frobenius 对齐把已有专家权重对齐到共享投影。
- combine 端同理标定专家输出 Y 的投影器，或用 top-magnitude 稀疏 mask 的离线幅值阈值。
- 推理期代价：每次多两次小 GEMM（d×ℓ），换 all-to-all 流量降 d/ℓ 倍——这就是"计算换通信"的兑现点，在 4090 这种通信瓶颈硬件上尤其划算。

#### (2) 码率按 token 重要性 / 路由置信度分档（方向 C 内化为模块）

- 重要性信号：路由 gate 的 top-1 概率（高置信 = 可压更狠）、token 范数、attention 权重。
- 分档示例：高置信 → ℓ_low + FP4；中等 → ℓ_mid + FP8；低置信 / 异常 → 全维 FP8（保护）。
- 效果：dispatch 流量从固定 d×16bit 变成 token 依赖的 ℓ_i×b_i，平均码率可调。

#### (3) combine 端误差补偿（方向 B，空白最干净）

- 专家输出回传前做 top-magnitude 稀疏化（保留每个输出向量幅值最大的 k 个分量 + 索引），残差用 error-feedback 思路修正。
- **设计要点（非对称）**：combine 端误差直接进最终 hidden state，比 dispatch 端敏感，因此 combine 端应**更保守**（更高码率 / 更小稀疏率）——这与 MoRI"dispatch FP4 / combine FP8"的非对称经验一致。
- 这一块即使单独做（方向 B 独立成文），也是一个干净的"first paper"。

#### (4) 与量化乘性叠加

- 流水线：activation → P_down 投影到 ℓ 维 → FP8/FP4 量化 → all-to-all → 反量化 → P_up 重建 → 专家计算。
- 理论压缩比 ≈ (d/ℓ) × (16/b)。例：d=2048、ℓ=512（4×）、FP8（2×）→ 理论 8× dispatch 流量缩减，远超单独 MXFP4 的约 4×（精度需实测）。

#### (5) 实验设计

- **模型/硬件**：Qwen3-30B-A3B、8×RTX 4090、EP=8（现成配置）；可加 DeepSeek-V2-Lite / Mixtral 做泛化。
- **基线**：(1) BF16 all-to-all（精度上界）；(2) DeepEP FP8 dispatch + BF16 combine；(3) MoRI 式 MXFP4 dispatch + FP8 combine；(4) ExFlow（放置，正交对照）；(5) LSH-MoE 思路移植到推理（聚类质心）作为有损对照。
- **指标**：通信量（bytes/token，dispatch 与 combine 分开报）、端到端 latency/throughput（prefill 与 decode 分开）、精度（PPL + MMLU/GSM8K/HumanEval）、压缩比 vs 精度 Pareto 曲线、与量化叠加的增益曲线。
- **消融**：ℓ 扫描；分档策略 on/off；combine 误差补偿 on/off；低秩 vs 稀疏 vs 二者结合；per-layer vs per-expert 投影器。
- **计算换通信验证**：报告额外 GEMM 的 FLOPs/latency，证明在 4090 上净收益为正。

#### (6) 阶段化与 Go/No-Go 阈值

- **阶段 0（决定性最小验证，最先做）**：hook 出 Qwen3-30B-A3B 某层 dispatch 激活，跑 SVD，画"保留前 ℓ 维解释多少方差"的奇异值谱曲线。**这一张图直接拍板低秩这条路成不成立**：谱掉得快 → 可行；谱很平 → 放弃低秩、转纯 combine 稀疏化（方向 B）。半天可出结果。
- **阶段 1（dispatch-only 低秩，2–4 周）**：固定 ℓ=d/4，测精度掉点与通信降幅。Go：MMLU/GSM8K 掉点 <1%、dispatch 流量降 ≥3×。
- **阶段 2（加 combine + 误差补偿）**：Go：往返流量降 ≥4×、decode 吞吐 +≥20%、精度掉点 <1.5%。
- **阶段 3（码率自适应 + 量化叠加）**：目标：等精度下往返流量降 ≥6×，或等流量下精度优于纯 MXFP4。
- **回退条件**：若投影 GEMM 在 4090 上吃满 SM 反而拖慢端到端（计算换通信变负），退回"仅 combine 稀疏化 + 量化"的轻量配置。

---

## 6. Caveats（必须如实记录的风险与不确定性）

- **来源质量**：MoRI-EP "MXFP4 dispatch 精度无损"来自 AMD/LMSYS 产线博客（非同行评审），"无损"需自测。LatentMoE/Nemotron 3、部分 2026 arXiv 预印本非常新、多数未评审。
- **方法可行性是间接证据**：推理期 + 冻结模型 + 学习式激活压缩在 TP（arXiv 2411.09510）验证过，但搬到 MoE EP 有域差异（路由动态性、combine 端敏感性）需实验确认。
- **精度风险集中在 combine 端**：专家输出误差直接进 hidden state，combine 端有损压缩可能比 dispatch 端更易掉点，必须非对称设计。
- **硬件依赖**：8×4090 无 NVLink，通信瓶颈重 → 计算换通信收益大；换高带宽互联收益缩小，结论需标注硬件前提。
- **无损上限有限**：ZipCCL 显示 all-to-all 无损压缩仅约 1.33×，要大压缩比必须有损 → 精度-压缩比 Pareto 曲线是核心交付物，而非单点数字。
- **标定漂移**：离线标定的投影器/阈值可能随输入域漂移失配，需评估跨域鲁棒性或轻量在线更新。

---

## 7. 参考文献（按方向）

**payload 低秩 / 降维压缩**
- SCoMoE: Efficient Mixtures of Experts with Structured Communication. ICLR 2023. OpenReview s-c96mSU0u5.
- LSH-MoE: Communication-efficient MoE Training via Locality-Sensitive Hashing. NeurIPS 2024. arXiv 2411.08446.
- LatentMoE / Mixture-of-Latent-Experts (NVIDIA Nemotron 3). arXiv 2512.20856 / 2601.18089.
- Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression. arXiv 2510.02345.

**通信量化**
- DeepEP: an efficient expert-parallel communication library. DeepSeek, GitHub.
- MoRI-EP (AMD MI355X + SGLang). AMD/LMSYS technical blog, 2026.
- MegaScale-MoE. EuroSys 2026. arXiv 2505.11432.
- MxMoE: Mixed-precision Quantization for MoE. ICML 2025. arXiv 2505.05799.
- FP8-Flow-MoE. arXiv 2511.02302.

**自适应路由**
- Ada-K Routing. ICLR 2025. arXiv 2410.10456.
- Harder Tasks Need More Experts (Top-P). arXiv 2403.07652.
- DA-MoE. arXiv 2409.06669.

**overlap / 融合 / 拓扑 / 放置（背景）**
- Tutel: Adaptive Mixture-of-Experts at Scale. MLSys 2023.
- FasterMoE. PPoPP 2022.
- ScMoE / Shortcut-connected Expert Parallelism. arXiv 2404.05019.
- COMET: Fine-grained Computation-communication Overlapping for MoE. MLSys 2025. arXiv 2502.19811.
- Flux: Fast Software-based Communication Overlap. arXiv 2406.06858.
- EPS-MoE. arXiv 2410.12247.
- HierMoE. arXiv 2508.09591.
- Speculative MoE / Semantic Parallelism. arXiv 2503.04398.
- ExFlow: Exploiting Inter-Layer Expert Affinity. arXiv 2401.08383.

**方法可行性先例（TP）**
- Communication Compression for Tensor Parallel LLM Inference. arXiv 2411.09510.
- Bian et al. Does compressing activations help model-parallel training? 2024.

**误差反馈（combine 端补偿可借鉴）**
- Seide et al. 1-bit SGD. 2014. / Stich et al. 2018. / EF21 (Richtárik et al.).

**无损通信压缩（上限参照）**
- ZipCCL: Lossless Data Compression of Communication Collectives. arXiv 2604.27844.
