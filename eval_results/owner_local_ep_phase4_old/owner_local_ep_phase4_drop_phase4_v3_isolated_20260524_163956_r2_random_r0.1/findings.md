# Owner Local EP Phase 4 Drop Findings

- Phase 4 drop sweep on impl=ep_ht, overlap=0.25, prompt=reasoning_brief.
- num_repeats=1, warmup_cases=0. mean ± stdev across reps in summary_agg.csv.
- Best (by mean score desc, mean e2e asc): round_robin__numa_local_first__min_communication / random @ rate=0.1 score_mean=0.859375 e2e_mean=176.98268081666902

|baseline_key|drop_policy|drop_rate|n_reps|score_mean|score_std|e2e_s_mean|e2e_s_std|prefill_tok_s_mean|decode_tok_s_mean|drop_fraction_total_mean|drop_fraction_remote_mean|dropped_weight_mass_fraction_mean|pass|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|round_robin__numa_local_first__min_communication|random|0.1|1|0.859375|0.0|176.98268081666902|0.0|78.22830551376964|20.79236101945328|0.09740999402449955|0.13013512691785234|0.09617544584871025|True|
