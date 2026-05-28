# Owner Local EP Phase 4 Drop Findings

- Phase 4 drop sweep on impl=ep_ht, overlap=0.25, prompt=reasoning_brief.
- num_repeats=1, warmup_cases=0. mean ± stdev across reps in summary_agg.csv.
- Best (by mean score desc, mean e2e asc): round_robin__numa_local_first__min_communication / per_expert_uniform @ rate=0.05 score_mean=0.94921875 e2e_mean=153.79838913632557

|baseline_key|drop_policy|drop_rate|n_reps|score_mean|score_std|e2e_s_mean|e2e_s_std|prefill_tok_s_mean|decode_tok_s_mean|drop_fraction_total_mean|drop_fraction_remote_mean|dropped_weight_mass_fraction_mean|pass|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|round_robin__numa_local_first__min_communication|per_expert_uniform|0.05|1|0.94921875|0.0|153.79838913632557|0.0|90.75441810401497|23.932029576004634|0.012515186828990107|0.016725998639821743|0.005828553023095476|True|
