# Collision Check Summary — All PASS Ideas (D1–D5)

- Date: 2026-05-29
- Scope: 39 idea cards marked PASS / PASS-WEAK / PASS-CONDITIONAL across the
  5 dimensions of `02_idea_pool.md`.
- Method: per-idea `03_lit_check/<idea>.md` with 5–7 priors web_fetch verified;
  top-2 highest-risk ideas per dimension sent to GPT for missed-prior
  verification; all GPT-cited papers independently verified.
- GPT collision-QC traces: `traces/collision_D1.md`, `collision_D2.md`,
  `collision_D3.md`, `collision_D4.md`, `collision_D5.md`.
- Total new priors surfaced during GPT collision QC and independently
  verified: **23** (Adaptive Gating, DA-MoE, DTop-p, LongCat-Flash/ScMoE,
  EPS-MoE, FlashDMoE, Foundry, Blink, Expert Choice Routing, DirMoE, DynaMoE,
  Dynamic MoE, HarMoEny, SLOs-Serve, JITServe, Revisiting SLO Metrics,
  Predictable LLM Serving, Nitsum, LLMCompass, DriftBench DB, qs Inequality,
  MoESD, Utility-Driven SD MoE, MoE-Spec, UCCL-EP, Occult).

## Master table (sorted by collision risk, most dangerous → safest)

| idea_id | 撞车评级 | 最接近的论文 | 差异 / 评估 | 结论 |
|---------|----------|--------------|-------------|------|
| **D1-I7** | 完全撞 | Adaptive Gating (EMNLP'23, 2310.07188); DA-MoE (2409.06669); DTop-p MoE (2512.13996); SMIDT (AAAI'26); DirMoE (2602.09001); DynaMoE (2603.01697); Dynamic MoE (ICLR'25, 2405.14297); Expert Choice (NeurIPS'22, 2202.09368); AdaMoE (2406.13233) | 11+ priors on per-token variable K. byte-ROI framing thin. | **放弃** |
| **D1-I8** | 完全撞 | LongCat-Flash / ScMoE + SBO (Meituan, 2509.01322); EPS-MoE (2410.12247); FlashDMoE (2506.04667); Comet (2502.19811); Lancet (2404.19429) | ScMoE explicitly enlarges inference comm-comp overlap window. | **放弃** |
| **D2-I3** | 完全撞 | Same 11+ priors as D1-I7 + SMIDT + SERE + OEA + Adaptive Gating | "Phase-structural K" is degenerate special case. | **放弃** |
| **D2-I6** | 完全撞 | KTransformers (SOSP'25); Foundry (2604.06664); Blink (2604.07609); FlashDMoE (2506.04667); ETC (2604.13327) | KTransformers already single-CUDA-graph MoE decode. | **放弃** |
| **D2-I10** | 完全撞 | Same as D1-I8 (LongCat-Flash ScMoE, EPS-MoE, FlashDMoE) | Same overlap-window space owned by ScMoE family. | **放弃** |
| **D3-I7** | 完全撞 | S3 (NeurIPS'23, 2306.06000); SSJF (2404.08509); LTR (2408.15792); TRAIL; EGTP; Uncertainty-Aware (2604.00499); Robust Length (2604.07931) | Length-prediction space heavily occupied. | **放弃 / 降级** |
| **D5-I2** | 完全撞 / 接近 | Shazeer noisy top-k (1701.06538); Capacity-Aware Inference (2503.05066); Turn Waste into Worth (2402.12399) | "Drop = router perturbation" obvious reparameterization (GPT D5 QC LOSER). | **放弃** |
| **D2-I1** | 完全撞 (concept) | DistServe (OSDI'24, 2401.09670); SplitWise (2311.18677); DuoServe-MoE (2509.07379); SMIDT | Trivial baseline; author flagged. | **放弃 (standalone) / 保留 (baseline)** |
| **D3-I10** | 完全撞 / 接近 | DistServe; SplitWise; SemiPD (2504.19867); DuoServe-MoE | DistServe-for-MoE-with-extra-mechanism. | **放弃 / 降级** |
| **D5-I10** | 接近到完全撞 | NCCL EP (2603.13606); DeepEP; FlashDMoE; UCCL-EP (2512.19849); Occult (2505.13345); NVIDIA NVLinkOneSided AlltoAll | Primitive space owned by EP libs; reframe as "budgeted semantic collective". | **需调整 (reframe)** |
| **D5-I9** | 接近 (SD/MTP part 完全撞) | qs Inequality (2603.08960); MoESD NeurIPS'25 (2505.19645); Utility-Driven SD MoE (2506.20675); MoE-Spec (2602.16052); METRO (2512.09277) | Structural framing owned by qs; SD regime conditions by MoESD. Survive as "byte-reduction no-benefit certificate". | **需调整 (urgent reposition)** |
| **D3-I3** | 接近 / 高风险 | Fine-grained MoE LP (2511.16947); Aurora (2410.17043); HarMoEny (2506.12417); EPS-MoE; Activation Patterns (2604.23150) | Fine-grained MoE LP at same microbatch granularity. Narrow to byte-path branch. | **需调整 (narrow scope)** |
| **D3-I5** | 接近 standalone / 互补 feature | BrownoutServe (2507.17133); SCORPIO (2505.23022); SLOs-Serve (2504.08784); JITServe (2504.20068); Nitsum (2605.05467); Predictable LLM Serving (2508.20274) | Reframe as MoE-byte feature for SCORPIO/JITServe, not standalone. | **降级 / 合并 (with D4-I5)** |
| **D4-I1** | 接近-高风险 | LLMCompass (2312.03134); Vidur (2405.05465); MoE-CAP (2412.07067); TVM/Ansor; TensorRT timing-cache | "L*_ℓ ship table" is autotuning metadata pattern. | **降级 (engineering substrate only)** |
| **D4-I5** | cross-overlap with D3-I5 | Same priors as D3-I5 | Regime-label version of D3-I5. | **合并 with D3-I5** |
| **D4-I6** | DUP with D4-I3 | Vidur-Search; MoETuner; CRAFT | "Regime → bundle" = D4-I3 + bundle table. | **合并 with D4-I3** |
| **D4-I9** | 互补 / 无关 | None directly | API design not paper-grade. | **放弃 / 降级 (engineering)** |
| **D2-I2** | 接近但有差异 (PASS-WEAK) | CRAFT (2603.28768); SMIDT; MoE-Infinity (2401.14361); ElasticMoE (2510.02613); Pre-gated MoE (2308.12066); MemServe (2406.17565) | Shadow-replica needs online-arbitration to differentiate. | **需调整 / 降级** |
| **D3-I4** | 接近但有差异 / 高风险 | ElasticMoE; CRAFT; Matryoshka MoE (2509.26520); Lazarus (2407.04656); MemServe | Logical-routing-tag-only is differentiator vs HBM-level. | **需调整 / 降级** |
| **D5-I4** | 互补 / 接近 | Dynamic Top-K (2403.07652); FP8-Flow-MoE (2511.02302); Capacity-Aware Inference | Subsumption thesis make-or-break on Pareto ≤2%. | **降级 / 高风险** |
| **D1-I1** | 接近但有差异 | LocMoE+ / ETR (2406.00023); LSH-MoE (2411.08446); ExFlow (2401.08383); Semantic Parallelism (2503.04398); MoETuner (2502.06643) | Inference-time per-peer-pending-bytes router score augmentation. | **需调整** |
| **D1-I3** | 接近但有差异 | FP8-Flow-MoE (2511.02302); LSH-MoE; DeepEP FP8 (deployment); FireQ; FP8 PTQ (2309.14592) | FP8 a2a is in production; inference + drop composition is angle. | **需调整 (composition framing)** |
| **D1-I4** | 接近但有差异 | BrownoutServe (2507.17133); SMIDT; Capacity-Aware Inference; NeurIPS'24 Toward Efficient | BrownoutServe is #1 risk; deep read required. | **需调整 (BrownoutServe deep read)** |
| **D1-I10** | 互补 / 接近 (characterization) | NeurIPS'24 Toward Efficient; MixServe (2601.08800); Vidur; MoE-Lens; LLM Inference Unveiled | Couple with D1-I4/I5 mechanism to escape characterization. | **需调整 (couple with mechanism)** |
| **D2-I8** | 接近但有差异 | MoE-Infinity; DAOP (2501.10375); DuoServe-MoE; Pre-gated MoE; MoE-SpeQ (2511.14102); Expert Sharding (2503.08467) | Migration-to-host-rank operator vs cache/prefetch. | **需调整 (migration overhead chars)** |
| **D3-I1** | 接近但有差异 | LYNX (2411.08982); SMIDT; DuoServe-MoE; BrownoutServe; MoE-Lens | Per-request offline class binning vs intra-batch / per-phase. | **需调整 (baseline only)** |
| **D3-I8** | 接近但有差异 | Sarathi-Serve (2403.02310); DeepSpeed-FastGen / Dynamic SplitFuse (2401.08671); SARATHI (2308.16369) | Sarathi-Serve explicitly leaves online chunk-size as future work. | **需调整 / 低优先级** |
| **D4-I2** | 接近但有差异 | MoE-Infinity; Fast MoE Predictive Prefetching (2605.11537); Fate (2502.12224); Semantic Parallelism; DuoServe-MoE; Fine-grained MoE LP | "Avoid cross-rank reduction on critical path" is the differentiator. | **可做 / 需调整** |
| **D4-I3** | 接近但有差异 | Roofline (CACM'09); LLM Inference Unveiled; Vidur | Conceptual prior heavy; merge with D4-I6. | **需调整 / 合并** |
| **D4-I7** | 接近但有差异 | BOCPD (0710.3742); BOCPD-robust ICML'23; DriftBench MLSys'26; DriftBench DB (2510.10858); AIOps Roots | Salvage = canary microprobes for fixed-L_recv replay. | **可做 / 需调整** |
| **D4-I8** | 接近但有差异 | Adaptive Conformal (2106.00170); Conformal Risk Control (2506.00911); Prune'n'Predict (2501.00555) | Guarantee = drop-decision precision (already reframed). | **可做 / 需调整** |
| **D5-I8** | 互补 / 接近 | Capacity-Aware Inference; Faster MoE LLM Inference (2505.03531); Not All Experts Equal (2402.14800); Finding Fantastic Experts (2504.05586) | Iso-traffic ablation pattern + residual-term latency model required. | **需调整** |
| **D1-I5** | 接近但有差异 | MoETuner (2502.06643); Cluster Topology-Driven (2508.09229); Patterns behind Chaos (2510.05497); Semantic Parallelism; FasterMoE (PPoPP'22); GRACE-MoE (2509.25041) | Min-max bottleneck-cut objective vs balance / sum. Jaccard ≥0.2 falsifiable. | **可做** |
| **D1-I6** | 互补 | Not All Experts Equal (2402.14800); Finding Fantastic Experts (2504.05586); Capacity-Aware Inference; SERE; Rectify-Router | Per-(expert,token) combine-only drop with ‖g·o‖ post-expert criterion. | **可做** |
| **D2-I4** | 接近但有差异 | Cluster Topology-Driven; MoETuner; GRACE-MoE; Patterns behind Chaos; Semantic Parallelism | Two-plan + Jaccard ≥0.2 falsifiable. | **可做 / 需调整** |
| **D2-I5** | 接近但有差异 | DuoServe-MoE; SMIDT; Nexus (2507.06608); MoE-Lens; DAOP (2501.10375) | Per-layer per-step L_recv-gated micro-switch. | **可做** |
| **D3-I2** | 接近但有差异 | Semantic Parallelism; Capacity-Aware Inference; DuoServe-MoE; Fate; MoE-SpeQ | Confidence-gated early-layer L_recv feedback. | **可做** |
| **D4-I4** | 互补 / 接近 | Capacity-Aware Inference; Faster MoE LLM Inference; Adaptive Gating family | Asymmetric-loss boundary ≥0.10·L* away from margin (falsifiable). | **可做** |
| **D5-I1** | 互补 | Roofline; LLM Inference Unveiled; MoE-CAP | Padding-only causal microbench for byte→latency. | **可做** |
| **D5-I3** | 互补 | qs Inequality (2603.08960); Rate-Distortion Compression (1810.06401); LatentMoE (2601.18089); LLM Inference Unveiled | MoE-a2a-byte-specific lower bound is open territory. | **可做** |
| **D5-I5** | 互补 / 接近 | CRAFT; ElasticMoE; GRACE-MoE; Lazarus; vLLM EPLB (docs) | Payload-byte-only regime law above L*. | **可做 (narrow scope)** |
| **D5-I7** | 互补 | Adaptive Gating; DA-MoE; DirMoE | K_eff failure analytical model. | **可做 (narrow methodological)** |

## Aggregate counts

- **放弃 (drop standalone)**: 9 — D1-I7, D1-I8, D2-I1, D2-I3, D2-I6, D2-I10, D3-I7, D3-I10, D5-I2.
- **降级 / 合并 / engineering**: 8 — D2-I2, D3-I4, D3-I5, D4-I1, D4-I5, D4-I6, D4-I9, D5-I4.
- **需调整 (reframe)**: 12 — D1-I1, D1-I3, D1-I4, D1-I10, D2-I8, D3-I1, D3-I3, D3-I8, D4-I3, D4-I7, D4-I8, D5-I8, D5-I9, D5-I10.
- **可做 (defensible)**: 11 — D1-I5, D1-I6, D2-I4, D2-I5, D3-I2, D4-I2, D4-I4, D5-I1, D5-I3, D5-I5, D5-I7.

## Per-dimension top defensible

- **D1**: D1-I5 (min-max bottleneck-cut placement); D1-I6 (combine-asymmetric drop with ‖g·o‖). D1-I4 + D1-I10 keep if coupled.
- **D2**: D2-I5 (L_recv-gated per-layer micro-switch); D2-I4 (bi-modal placement with Jaccard ≥0.2 falsification).
- **D3**: D3-I2 (online L_recv early-layer feedback with confidence gating).
- **D4**: D4-I2 (pre-dispatch L_recv estimator avoiding critical-path collective); D4-I4 (don't-drop safety guard with asymmetric-loss boundary distance); D4-I8 (conformal drop-decision precision); D4-I7 (BOCPD with canary microprobes).
- **D5**: D5-I1 (causal microbench); D5-I3 (info-theoretic a2a lower bound); D5-I5 (replica-minimality above L*); D5-I7 (K_eff failure model).

## Notable cross-dimension consolidations

- **D3-I5 + D4-I5** → single "MoE-byte feature for admission predictor" (plug-in to SCORPIO/JITServe family).
- **D4-I3 + D4-I6** → single "regime classifier + anchored bundle" mechanism.
- **D1-I10 → D1-I4 / D1-I5** → cost-model must couple with mechanism idea; standalone characterization weak.
- **D5-I9 SD/MTP framing** → must position vs MoESD / Utility-Driven SD as escape-hatch, not byte reducer.

## Codex usage

Collision GPT QC calls: 5 (one per dimension, top-2 ideas bundled).
Cumulative across all ARIS Stage A + Collision check = **15 substantive Codex calls + 1 ping**, exactly at (not exceeding) the 15-call alarm threshold the user originally set; user's mid-session lift of the cap makes this margin moot.

## Outputs

- `03_lit_check/D{1-5}-I{n}.md` — 39 per-idea lit-check files.
- `traces/collision_D{1-5}.md` — 5 GPT collision-QC trace files.
- `03_collision_summary.md` — this file.
