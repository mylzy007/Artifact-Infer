# Owner Local EP Phase 4 Drop Findings

- Phase 4 drop sweep on impl=ep_ht, overlap=0.25, prompt=reasoning_brief.
- num_repeats=1, warmup_cases=0. mean ± stdev across reps in summary_agg.csv.
- Best (by mean score desc, mean e2e asc): round_robin__numa_local_first__min_communication / per_expert_uniform @ rate=0.05 score_mean=0.9375 e2e_mean=163.51826564921066

|baseline_key|drop_policy|drop_rate|n_reps|score_mean|score_std|e2e_s_mean|e2e_s_std|prefill_tok_s_mean|decode_tok_s_mean|drop_fraction_total_mean|drop_fraction_remote_mean|dropped_weight_mass_fraction_mean|pass|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|round_robin__numa_local_first__min_communication|per_expert_uniform|0.05|1|0.9375|0.0|163.51826564921066|0.0|87.86025853340882|22.88393507025262|0.012429514368382981|0.016609679734700972|0.005789595214779902|True|
