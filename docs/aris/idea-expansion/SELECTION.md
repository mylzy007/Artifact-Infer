# ARIS Idea Selection

Date: 2026-05-29

本轮按用户给定优先级裁决：先看撞车安全，再看是否推进 communication-centric 主故事，再看 pilot 可行性，最后才看 novelty。结论偏保守：凡是完全撞已有论文、或明显撞 Phase 3 / Phase 4 / decode 旧结论的，不因原始分高而进入强推。

## 1. 候选总览表

| idea_id | 赌注 | 撞车 | 一致性 | story_fit | 改动量 | 裁决 |
|---|---|---|---|---|---|---|
| D1-I1 | router 加 per-peer byte cost 降 dispatch/combine bytes | 接近 LocMoE/ETR/Semantic Parallelism，需调整 | 与 Phase 3 不同，是 runtime routing | 中高 | 中 | 暂缓；accuracy 风险和 router 改动不如 D2-I5 干净 |
| D1-I3 | FP8/INT8 只压 a2a payload | 接近 FP8-Flow/DeepEP FP8，需 composition framing | 与 drop 正交 | 高 | 大 | 暂缓；工程量大，像 applied FP8 |
| D1-I4 | byte-budget runtime controller | 接近 BrownoutServe/Capacity-Aware，需 deep read | 与静态 drop 不同 | 高 | 中 | 候补；可作为 D2-I5 后续扩展，不先做 |
| D1-I5 | min-max worst-peer-byte placement | 文献可做，但 placement 空间拥挤 | **危险：Phase 3 placement 旧结论复现风险** | 高 | 中 | 不强推；只允许先做 Jaccard 生死测试 |
| D1-I6 | combine-only drop + post-expert `||g·o||` | 可做，专家剪枝文献只是邻近 | 与 Phase 4 symmetric/router-score drop 不同 | 高 | 中 | **强烈推荐** |
| D1-I10 | worst-peer bytes 性能模型 | 接近 characterization work | 不撞旧结论，但 standalone 空 | 高 | 小 | 做支撑证据，不作主 pilot |
| D2-I2 | phase-specific replica budget / shadow replica | 接近 CRAFT/MoE-Infinity，弱 | memory/cache residency 味道重 | 中 | 中 | 降级；不选 |
| D2-I4 | prefill/decode 双 placement plan | 可做但 placement 拥挤 | **危险：Phase 3 placement 类** | 中高 | 中-大 | 不强推；同 D1-I5 需 Jaccard 先验 |
| D2-I5 | per-layer `L_recv > L*_ell` 微切换 | 可做，未被 DuoServe/SMIDT 覆盖 | 独立，直接使用 L* finding | 很高 | 中 | **强烈推荐，立刻 pilot 首选** |
| D2-I8 | decode expert re-co-location | 接近 cache/offload，但有差异 | 不撞 decode drop 结论，机制不同 | 中 | 大 | 暂缓；session routing 改动过大 |
| D3-I1 | request-class policy lookup | 接近/弱 | Phase 4 per-class sweep 变体 | 中低 | 中 | 只作 baseline |
| D3-I2 | early-layer `L_recv` feedback mid-request switch | 可做，接近 Capacity-Aware/Semantic Parallelism | 与静态 drop 不同 | 高 | 中 | 候补；可并入 D2-I5 |
| D3-I3 | microbatch-level MoE policy | 高风险，接近 Orca/Sarathi/Aurora/EPS-MoE | 与旧结论不冲突 | 高 | 中-大 | 暂缓；scheduler API 风险较高 |
| D3-I4 | elastic replica pool with routing tag | 接近 ElasticMoE/CRAFT/MemServe | 资源管理味道重 | 中 | 中-大 | 降级 |
| D3-I5 | MoE-byte admission feature | 接近 BrownoutServe/SCORPIO/JITServe | 不撞旧结论 | 中 | 中-大 | 降级为 feature，不作 standalone |
| D3-I8 | chunk size target MoE L* | 接近 Sarathi/Dynamic SplitFuse | cross-service 集成，且像调参 | 中 | 大 | 不选 |
| D4-I1 | per-layer `L*_ell` artifact | near-dead novelty | 只是支撑表 | 中 | 小 | 不作 standalone |
| D4-I2 | pre-dispatch `L_recv` estimator | 可做/需绑定 actuator | 独立 | 中高 | 中 | 支撑 D2-I5/D4-I4，不单独选 |
| D4-I3 | regime classifier | 接近 cost-model bundle | 独立但抽象薄 | 中 | 小-中 | 暂缓 |
| D4-I4 | one-way don't-drop safety guard | 可做 | 独立；避免 Phase 4 负收益区 | 中高 | 小-中 | 推荐作 D2-I5 的 safety companion |
| D4-I7 | BOCPD online L* recalibration | 可做但算法 prior-heavy | 独立 | 中 | 中 | 暂缓；生产漂移故事不急 |
| D4-I8 | conformal drop-decision precision | 可做 | 独立 | 中 | 中 | 暂缓；像统计 wrapper |
| D5-I1 | pad-only causal microbench: bytes => latency | 可做 | 独立，把 97% finding 从相关变因果 | 高 | 小 | **强烈推荐作支撑 pilot** |
| D5-I3 | a2a bytes information lower bound | 可做 | 独立 | 中 | 小 | 候补；偏理论支撑 |
| D5-I5 | replica-minimality above L* | 可做但可能已在数据中 | **危险：Phase 4 MIN_REPLICAS 旧数据重述** | 中 | 小 | 不选 standalone |
| D5-I7 | K_eff failure model | 可做但 retrospective | Phase 4 P1 失败解释 | 中 | 小 | appendix/分析支撑，不选 |
| D5-I8 | byte-equivalence classes | 接近 iso-traffic ablation，已降级 | 独立但薄 | 中 | 小-中 | 暂缓 |
| D5-I9 | decode-hopelessness theorem | SD/MTP 部分撞 MoESD/Utility-Driven SD | **前半撞 decode 旧结论** | 高但偏解释 | 小 | 不选 standalone |
| D5-I10 | budgeted semantic collective | primitive 空间接近 NCCL EP/DeepEP/UCCL-EP | 独立但 API 大改 | 中高 | 大 | 不选 pilot |

## 2. 强烈推荐做 pilot 的 2-3 个

### 1. D2-I5: L_recv-gated continuous phase detection

**为什么选它**：

- 撞车安全：不是 DistServe/SplitWise，也不是 DuoServe-MoE 的 prefetch；它的 lever 是 layer-local drop gate，判据是 `L_recv_ell > L*_ell`。
- story_fit：最直接把 communication-centric finding 落成机制。Brief 里的关键不是 "prefill/decode" 标签，而是 `L_recv` 是否越过 `L*`。
- pilot 可行性：复用现有 drop infra，`total_recv/L_recv` 已经在 `dispatch_ep_ht.py` 路径里存在；先离线 replay，不需要先改 runtime。
- novelty：不是最高，但排序里 novelty 最后；它胜在干净、可证伪、和旧结果正交。

一句话 pilot 思路：重放 LongBench + GSM8K trace，按每层 `L_recv_ell` 与校准 `L*_ell` 仿真 drop on/off，比较 binary phase switch、static `tail_weight@r=0.3`、oracle per-phase static。

### 2. D1-I6: Combine-asymmetric drop

**为什么选它**：

- 撞车安全：邻近的 Not All Experts Equal / Finding Fantastic Experts 是 expert-level skipping，不是 per-(expert, token) combine payload drop。
- story_fit：正中 97% finding 里的 combine 半边；它问的是 dispatch 与 combine 两段信息不对称是否能换来更好的 quality-vs-byte Pareto。
- pilot 可行性：需要 combine-side hook 和 `||g·o||` 统计，改动中等；最小 pilot 可以先做离线 replay / cosine proxy。
- novelty：比 D2-I5 更像一个新 operator；但它的 latency 天花板只有 combine 那一半，所以排第二。

一句话 pilot 思路：在记录/可重放的 expert output 上按 `||g_e,t · o_e,t||_2` 做 combine-only percentile drop，对比等 combine-byte 下的 symmetric `tail_weight` drop 的 layer output cosine 和任务 proxy。

### 3. D5-I1: causal-isolation bytes microbench

**为什么选它**：

- 撞车安全：方法论邻近 roofline，但不是已有 MoE idea 冲突；也不撞 Phase 3/4 旧结论。
- story_fit：它不提供新机制，但能把 "97% 来自 a2a payload" 从 segment ablation 的相关证据升级为更硬的因果斜率。
- pilot 可行性：最小改动，半天级别；复用已有 dispatch/bench scaffold。
- novelty：低，但这是支撑 pilot，不是主 paper claim。它能保护 D2-I5/D1-I6 的故事不被审稿人打成经验巧合。

一句话 pilot 思路：固定 `L_recv`、routing 分布和 expert compute，只通过 padding 改 wire bytes，拟合 per-byte latency slope，检查 slope 在 drop rates 上是否稳定。

## 3. 排除清单

**完全撞已有论文 / 机制塌缩**：

- D1-I2：跨层 permutation 复用机制不成立，最强解释塌进 ExFlow / Path-Constrained MoE。
- D1-I7：per-token variable K 完全撞 Adaptive Gating、DA-MoE、DTop-p、Dynamic MoE、DirMoE 等。
- D1-I8：跨层 overlap 撞 LongCat-Flash/ScMoE、EPS-MoE、FlashDMoE、Comet、Lancet。
- D2-I3：phase-structural K 是 dynamic K 家族的退化特例，撞 SMIDT / Adaptive Gating / SERE 等。
- D2-I6：decode fast-path CUDA graph 撞 KTransformers、Foundry、Blink、FlashDMoE。
- D2-I7 / D3-I9：prefill->decode expert prefetch 撞 DuoServe-MoE。
- D2-I10：phase overlap 撞 LongCat-Flash/ScMoE、EPS-MoE、FlashDMoE。
- D3-I7：decode-length predictor 撞 S3、SSJF、LTR、TRAIL、EGTP 等 length-prediction 系。
- D3-I10：heterogeneous fleet 基本是 DistServe/SplitWise/SemiPD/DuoServe-MoE 变体。
- D5-I2：drop = router perturbation 是显然 reparameterization，撞 noisy top-k / Capacity-Aware / Turn Waste into Worth。
- D5-I10：作为 primitive 撞 NCCL EP、DeepEP、UCCL-EP、Occult；只能作为 semantic layer，改动量又太大。

**撞自己旧结论 / 高概率重述已有数据**：

- D1-I5：placement 类，正面撞 Phase 3 "单换 placement 不再有显著加速"。除非先证明 min-max bottleneck-cut 与 LBG-overlap=0.25 的布局 Jaccard 距离足够大，否则直接复现旧结论。
- D2-I4：双 placement plan 同样撞 Phase 3 placement 收敛结论；没有 Jaccard 差异就不是新方向。
- D3-I1：per-class Phase 4 sweep，本质还是用旧 drop actions 做 lookup。
- D5-I5：`MOE_DROP_MIN_REPLICAS=128 vs 512` 可能已经在 Phase 4/v6 数据里，只是换 framing。
- D5-I7：K_eff failure model 是 Phase 4 P1 负结论解释，适合作 appendix，不适合作主 pilot。
- D5-I9：decode-hopelessness 前半就是 brief 里的 decode `L_recv≈8` 结构性无收益；SD/MTP 反转部分又撞 MoESD / Utility-Driven SD。

**工程量/集成风险过高，当前不适合立刻 pilot**：

- D1-I3：FP8 a2a 要集成 DeepEP/FP8 path，不是简单 quant/dequant。
- D2-I8：decode expert re-co-location 需要 session state、per-session routing override、迁移/副本生命周期。
- D3-I8：chunked-prefill 与当前 MoE workshop 是 cross-service integration。
- D5-I10：budgeted collective API 会波及 dispatch/combine/backend/caller，pilot 改动面过大。

## 4. 总体判断

### 只能选 1 个立刻 pilot

选 **D2-I5**。

理由很直接：它最符合四条排序。撞车上没有 fatal prior；一致性上不撞 Phase 3 placement、Phase 4 K_eff、decode no-benefit 旧结论；story 上把 `L*` / `L_recv` 从 finding 变成 runtime decision；pilot 上可以先离线 replay，改动量比 controller、placement、FP8、budgeted collective 都小。它不是最炫的 idea，但最像能快速给出 yes/no 信号的主线机制。

D1-I6 排第二，因为它更像新 operator，但 latency 主张天然只覆盖 combine 半边，pilot 还要拿到 post-expert contribution。D5-I1 排第三，因为它更像证据底座，不是主机制。

### 这批 idea 够 MLSys 还是只够 workshop?

诚实说：**单个 idea 现在都不够稳的 MLSys full paper**。D2-I5 一个 pilot 若只证明 "按 L_recv 阈值开关 drop" 有效，更像 workshop / systems note。要够 MLSys，需要拼成一个闭环但不能堆组件：

1. D5-I1 证明 bytes=>latency 的因果斜率；
2. D1-I10 给出 worst-peer bytes / `L_recv` 的可解释模型；
3. D2-I5 用这个模型做 runtime gate；
4. D1-I6 或 D4-I4 提供一个非平凡 operator / safety boundary。

如果 D2-I5 在 mixed/chunked workload 上能稳定超过 static/binary phase baseline，且 D1-I6 给出等 byte 下质量优势，这批有机会往 MLSys 投。否则目前更像 workshop 强稿：finding 很好，机制还需要一个真正的新系统动作站住。

### 被低估的 idea

我想为 **D5-I1** 说一句。它不是论文主菜，但它是这批 idea 里最便宜、最能提升整条 story 可信度的支撑实验。没有它，"97% 来自 bytes" 仍容易被打成 segment ablation 相关性；有它，D2-I5/D1-I6 的动机硬很多。

另一个高上限但我不强推的是 **D1-I5**。如果 Jaccard 生死测试显示 min-max bottleneck placement 与 Phase 3 LBG 真选出明显不同布局，它会重新变强；但在当前信息下，按你的第一条标准，它不能越过 Phase 3 旧结论风险直接进强推。
