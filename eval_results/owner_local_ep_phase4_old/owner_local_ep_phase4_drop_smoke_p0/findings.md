# Owner Local EP Phase 4 Drop Findings

- Phase 4 drop sweep on impl=ep_ht, overlap=0.25, prompt=reasoning_brief.
- num_repeats=1, warmup_cases=0. mean ± stdev across reps in summary_agg.csv.
- Best (by mean score desc, mean e2e asc): round_robin__numa_local_first__min_communication / tail_weight @ rate=0.1 score_mean=0.75 e2e_mean=27.50711128860712

|baseline_key|drop_policy|drop_rate|n_reps|score_mean|score_std|e2e_s_mean|e2e_s_std|prefill_tok_s_mean|decode_tok_s_mean|drop_fraction_total_mean|drop_fraction_remote_mean|dropped_weight_mass_fraction_mean|pass|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|round_robin__numa_local_first__min_communication|none|0.0|1|0.75|0.0|58.2646220237948|0.0|3.626551271511679|4.855561624919316|None|None|None|True|
|round_robin__numa_local_first__min_communication|tail_weight|0.1|1|0.75|0.0|27.50711128860712|0.0|14.870950888839621|4.808831595113936|None|None|None|True|
