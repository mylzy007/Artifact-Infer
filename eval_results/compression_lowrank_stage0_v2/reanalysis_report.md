# Stage 0 reanalysis: well-conditioned per-expert spectrum

Per-expert SVD restricted to experts with n_e >= 4096 tokens (2 * d), so the spectrum is well-conditioned. Smaller experts give rank-trivial cumulative variance at d/4 and were dropped from the earlier per-expert table.

## Global spectrum (re-confirm, all routed tokens, top-1-agnostic)
| layer | EV@d/16 | EV@d/8 | EV@d/4 | EV@d/2 |
|---:|---:|---:|---:|---:|
| 2 | 0.629 | 0.718 | 0.821 | 0.925 |
| 6 | 0.396 | 0.530 | 0.692 | 0.867 |
| 12 | 0.367 | 0.498 | 0.661 | 0.846 |
| 24 | 0.385 | 0.511 | 0.669 | 0.848 |
| 36 | 0.435 | 0.557 | 0.702 | 0.863 |
| 46 | 0.550 | 0.660 | 0.781 | 0.903 |

## Per-expert token count distribution
| layer | min | p25 | median | p75 | max | # experts with n_e >= 4096 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 0 | 41 | 232 | 6101 | 1 |
| 6 | 0 | 0 | 3 | 75 | 13295 | 1 |
| 12 | 0 | 1 | 33 | 299 | 4965 | 1 |
| 24 | 0 | 1 | 35 | 287 | 4520 | 1 |
| 36 | 0 | 0 | 32 | 295 | 3090 | 0 |
| 46 | 0 | 0 | 2 | 223 | 4592 | 1 |

## Well-conditioned per-expert spectrum (n_e >= 2d=4096)
Statistics taken across the experts with enough samples in each layer (only a handful of 'popular' experts qualify under this calibration set).

| layer | # experts | EV@d/8 min/med/max | EV@d/4 min/med/max | rank@0.90 min/med/max | rank@0.95 min/med/max | rank@0.99 min/med/max |
|---:|---:|---|---|---|---|---|
| 2 | 1 | 0.660/0.660/0.660 | 0.816/0.816/0.816 | 783/783/783 | 1075/1075/1075 | 1604/1604/1604 |
| 6 | 1 | 0.633/0.633/0.633 | 0.780/0.780/0.780 | 933/933/933 | 1263/1263/1263 | 1759/1759/1759 |
| 12 | 1 | 0.631/0.631/0.631 | 0.787/0.787/0.787 | 880/880/880 | 1184/1184/1184 | 1689/1689/1689 |
| 24 | 1 | 0.611/0.611/0.611 | 0.773/0.773/0.773 | 912/912/912 | 1213/1213/1213 | 1702/1702/1702 |
| 36 | 0 | - | - | - | - | - |
| 46 | 1 | 0.775/0.775/0.775 | 0.879/0.879/0.879 | 598/598/598 | 910/910/910 | 1515/1515/1515 |