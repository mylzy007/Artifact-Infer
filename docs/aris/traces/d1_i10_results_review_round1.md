# Trace — D1-I10 RESULTS round 1 (Codex attack)

- Date: 2026-05-30
- Reviewer: Codex (`mcp__codex__codex`), `model_reasoning_effort=xhigh`, sandbox=read-only
- Thread ID: 019e74d5-f76d-76b2-86c7-252d61629dd6
- cwd: `/home/lzy/Artifact-Infer-pilot`

## What Codex was given
Full D1-I10 pre-registered C1 thresholds + observed regression results (n=919, R² values + CIs + VIFs + tertile breakouts + L*_ℓ summary). Asked: is the FAIL verdict justified, can it flip, what's the honest reframing.

## Codex bottom line (verbatim)
> "C1.main FAIL is justified under the registered contract. There is no honest path from these numbers to 'worst-peer bytes beats L_recv as the dominant latency predictor.' But 'L_recv dominates' needs careful wording. ... 'worst-peer does not add meaningful information beyond L_recv, and L_recv is the better primary scalar in this experiment.' They do not support a strong universal claim that worst-peer is useless, or that L_recv is causally dominant in all regimes."

## Material attacks (responses in RESULTS.md)

1. **Could C1.main flip given VIF=50?** Answer: No. CV comparison of single-predictor models is valid regardless of collinearity. Bootstrap CIs are clean-separated, M2 > M1. Verdict stands.

2. **"L_recv dominates" overclaim warning**. Better wording: "L_recv is the preferred primary predictor; worst_peer is statistically non-superior". Adopted in RESULTS.md.

3. **Compute reverse incremental R²** (L_recv added to wp+FE). Done: +0.92pp vs forward +0.17pp. L_recv adds ~5× more, consistent with "preferred" not "dominates".

4. **Smarter stratification** to break collinearity (matched L_recv with varied skew). Acknowledged as out-of-scope for this pilot; flagged for post-pilot interventions.

5. **L*_ℓ audit (30× gap vs tier1)**. Done: per-layer stratification artifact (decode-dominated layers have narrow L_recv ranges). At L_recv ≥ 3271, drop helps 100% of 20 sampled cells, confirming tier1. **Recommendation: use L*_global = 3271 for D2-I5**, not per-layer L*_ℓ.

6. **Scope all claims to tertile 2** (load-dominated regime, L_recv ≳ 100). Accepted; scope statement added to RESULTS.md.

7. **High-load-only CV R²** request. Done: L_recv beats wp by 9pp CV in high-load tertile (0.767 vs 0.680).

8. **"L_recv causally dominates" overclaim**. Avoided. Claim is observational on natural distribution of v2 routing.

## What Codex left for D3 framing review
- Whether the L_recv-as-predictor framing is sufficient to carry a paper, or if the worst-peer FAIL substantively weakens the "communication-centric" thesis.
- Whether per-layer L*_ℓ noise (29/48 with crossing) is a publishable finding ("per-layer L* varies but we cannot pin it down with this sampling") or a methodological hole.

Verdict accepted. No further round 1 iterations needed (Codex agreed with verdict; only attacked framing language and demanded specific checks, all addressed).
