0. 背景
0.1 研究脉络
本工作隶属 MoE inference 在 owner_local_ep + ep_ht runtime 下的优化研究。模型 Qwen3-30B-A3B（48 层，E=128，K=8），硬件 8×RTX 4090 24GB，主 workload GSM8K（T=8）。

研究目标随阶段演化：

阶段	主题	核心动作
Phase 3	静态 expert placement（含 overlap）	设计/比较 LBG vs round_robin、numa_local_first、overlap=0/0.25/0.5 等组合，找最优静态副本表
Phase 4 v1/v2	Token-replica drop（CPU 实现）	设计 ≥4 类 drop 策略（tail_weight、random、cross_numa_first、per_expert_uniform、hot_expert_relief 等），动态决定每步丢哪些 (token, expert) 副本
Phase 4 GPU drop（当前）	drop 算子 GPU 化 + 小批量 bypass	消除 drop 自身开销，判断 drop 是否还有 e2e 收益空间
Phase 4 P1	Router K_eff 硬截	作为 drop 的对照，验证"只保留 top-K_eff expert"是否能省 e2e
0.2 各阶段关键结论
Phase 3（静态 placement）

LBG（load-balanced grouped）与 round_robin 在 overlap=0.25 下端到端时间基本持平（~115s），各 placement 方差比策略差异更大
overlap=0.25 是 communication vs computation 的 sweet spot；overlap=0 通信过多，overlap=0.5 redundant 计算过多
结论：静态 placement 调优在当前 workload 下已接近瓶颈，单纯换 placement 无法继续显著加速
Phase 4 v1/v2（CPU drop）

所有 drop 策略相比 baseline e2e +10~25%（变慢），与"drop 应当通过减少 expert 副本来加速"的直觉相反
诊断定位主因：apply_drop CPU 实现每层每 decode step 都做 host↔device 同步（.cpu()、.item() 多次），单次几百 μs，48 层 × 63 decode steps 累积起来远超 drop 节省的 GEMM 时间
score 方面：保守策略（tail_weight、cross_numa_first）保持 baseline 水平（±1pp），激进策略（random@0.2）掉点显著（−10pp+）
Phase 4 GPU drop + bypass（本工作）

GPU 化 apply_drop_gpu_simple（zero-sync 路径），加上 MOE_DROP_MIN_REPLICAS=128 小批量 bypass
实测三种实现对比（baseline → tail_weight@0.1）：
实现	decode tk/s	Δe2e vs base
cpu_b0（CPU drop）	25.42	+11%（慢）
gpu_b0（GPU drop 无 bypass）	24.14	+18%（慢）
gpu_b128（GPU drop + bypass）	28.83	−1%（持平）
核心发现：仅 GPU 化不够（每 decode step 仍累积 launch 开销），必须配合 bypass 才能彻底消除 drop 在 decode 路径上的开销
现状：drop 算子已成"零开销" op，但也没换来 e2e 收益
Phase 4 P1（Router K_eff 硬截）

K_eff	score	Δe2e
8 (base)	0.814	0%
7	0.777	+3.0%
6	0.746	+1.9%
5	0.711	−2.9%
4	0.639	+0.2%
精度随 K_eff 单调下降（K=7 已掉 3.65pp），e2e 几乎不动
结论：硬截 K 是负杠杆，验证了瓶颈不在"该不该算这些 expert"
0.3 当前研究问题
GPU drop + bypass 把 drop 做到零开销之后，仍未带来 e2e 收益。诊断指向根本性的 workload 适配问题：

decode 步每层每 GPU 接收 L_recv = T×K/EP ≈ 8 在 4090 上是 latency-bound，省 30% 副本 ≈ 省 0
bypass 在 decode 上保护了正确性（不引入开销），但也意味着 decode 永远拿不到 drop 的收益
prefill 占 e2e ~5%，杠杆太短
进一步观察：即使在 prefill 上，drop 也让 prefill_tok_s 从 235 → 223（−5%），说明当前 GSM8K 的 prefill 体量（L_recv ≈ 1200/GPU/layer）仍未进入 compute-bound 区间
0.4 已被排除的方向
方向	排除原因
继续优化静态 placement	Phase 3 已收敛
改 drop 策略（更精细的 grouped/hot_expert 等）	Phase 4 v3 验证策略间差异 < e2e 噪声
Router K_eff 硬截	P1 验证为负杠杆
Speculative decoding	repo 内零实现，从零接入数周
Layer skip / early exit	需训 router，离题
提高 batch / 砍 context	24GB VRAM 限制 + GSM8K 答案长度不可控
0.5 引出 Prefill L Sweep 的必要性
目前未定的关键问题：

drop 在多大的 L_recv 上才真正有 e2e 收益？这个拐点 L* 是否在 4090 24GB 可达的 prefill 体量内？

回答这个问题前，无法判断：

是该继续投入找"更大 prefill workload"
还是直接收尾把当前结果写为完整负结果
本规划用一个隔离微基准（Tier 1）+ 条件触发的端到端验证（Tier 2）来回答此问题，决定 Phase 4 的最终走向。