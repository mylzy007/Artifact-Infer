# Trace — Dimension 2 Collision GPT QC

- Date: 2026-05-29
- Reviewer backend: codex, reasoning_effort=high
- Thread ID: 019e6fc2-8fa0-7bd1-9c47-284ec0c88e55
- Sent ideas: D2-I6, D2-I3 (top-2 完全撞 in Claude's pre-QC rating)

## GPT raw response summary

```
D2-I6 missed priors:
- Foundry (arXiv:2604.06664, April 2026): template-based CUDA graph context
  materialization for LLM serving, covers MoE up to 235B
- Blink (arXiv:2604.07609, April 2026): persistent GPU scheduler + device-side
  CUDA graph launch, removes CPU from inference path
- SGLang piecewise CUDA graph (engineering, not academic)
- GPT explicitly cautioned: "Cap-and-Spill / FARS" is NOT a reliable academic
  prior — skipped.
Claude's 完全撞 verdict: confirmed correct.
Only narrow salvage: consumer-GPU EP=8 PCIe Qwen3 static graph (engineering note).

D2-I3 missed priors:
- Expert Choice Routing (Zhou et al., NeurIPS 2022, arXiv:2202.09368): variable
  experts per token via top-k-tokens (selecting reverse direction).
- DirMoE (Vahidi et al., ICLR 2026, arXiv:2602.09001): Bernoulli + sparsity
  penalty controlling expected active experts.
- DynaMoE (Gülmez, 2026, arXiv:2603.01697): dynamic token-level expert
  activation + layer-wise adaptive capacity.
- Dynamic MoE (Guo et al., ICLR 2025, arXiv:2405.14297): auto-tuning per-token
  K during training.
Claude's 完全撞 verdict: confirmed correct.
Only narrow salvage: "post-hoc serving policy" tied to L* diagnosis (not routing).
```

## Claude independent verification of GPT-cited papers

- Foundry (arXiv:2604.06664) — verified=YES.
- Blink (arXiv:2604.07609) — verified=YES.
- Expert Choice Routing (arXiv:2202.09368, NeurIPS 2022) — verified=YES.
- DirMoE (arXiv:2602.09001) — verified=YES.
- DynaMoE (arXiv:2603.01697) — verified=YES.
- Dynamic MoE / Auto-Tuning (arXiv:2405.14297) — verified=YES.
Zero verification failures.

## Verdict update

Both ideas remain 完全撞 with even more priors than Claude initially listed.
D2-I6 and D2-I3 marked "放弃" in conclusions.
