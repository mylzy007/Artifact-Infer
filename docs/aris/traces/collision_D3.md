# Trace — Dimension 3 Collision GPT QC

- Date: 2026-05-29
- Reviewer backend: codex, reasoning_effort=high
- Thread ID: 019e6fc8-4625-76c0-8043-17630873c901
- Sent ideas: D3-I3 (strongest D3, high-risk vs Fine-grained MoE LP), D3-I5 (admission, BrownoutServe risk)

## GPT raw response summary

```
D3-I3 missed priors:
- Aurora (arXiv:2410.17043) — MoE inference + a2a comm scheduling
- EPS-MoE (arXiv:2410.12247) — expert pipeline scheduler
- HarMoEny (arXiv:2506.12417) — multi-GPU MoE dynamic token redistribution
- Scaling Multi-Node MoE Activation Patterns (arXiv:2604.23150) — workload-aware
  micro-batch grouping on Qwen3-class
Salvage: narrow to single-node 8×4090 EP=8 byte-path branch thresholding;
output-invariant; runtime systems feature inside continuous batching.
Verdict: 接近 / high risk (not 完全撞 if narrowed).

D3-I5 missed priors:
- SLOs-Serve (arXiv:2504.08784) — multi-SLO dynamic token allocation
- JITServe (arXiv:2504.20068) — SLO-aware imprecise-info serving
- Revisiting SLO Metrics (arXiv:2410.14257) — SLO gaming + smooth-goodput
- Predictable LLM Serving (arXiv:2508.20274) — topology/fabric signals p99
- Nitsum (arXiv:2605.05467) — tiered-SLO TP-aware runtime
Salvage: do NOT sell as admission system; sell as MoE-specific risk feature.
Verdict: 接近 standalone / 互补 as feature.
```

## Claude independent verification of GPT-cited papers

All 6 papers independently verified=YES via web_fetch:
- Aurora (2410.17043) — verified via D1 trace.
- EPS-MoE (2410.12247) — verified via D1 collision trace.
- HarMoEny (2506.12417) — verified=YES.
- Activation Patterns (2604.23150) — verified via D1 trace.
- SLOs-Serve (2504.08784) — verified=YES.
- JITServe (2504.20068) — verified=YES.
- Revisiting SLO Metrics (2410.14257) — verified=YES.
- Predictable LLM Serving (2508.20274) — verified=YES.
- Nitsum (2605.05467) — verified=YES.

Zero verification failures.

## Verdict update

- D3-I3: remains "接近但有差异 / 高风险"; salvage = narrow single-node byte-path
  branch thresholding (not a new MoE scheduler). Update md committed.
- D3-I5: downgraded from "接近" to "互补 as feature / 接近 standalone"; reposition
  as MoE-byte-pressure predictor INPUT to SCORPIO/JITServe family. Update md
  committed.
