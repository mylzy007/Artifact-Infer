# Owner Local EP Phase 4 Drop Findings

- Phase 4 drop sweep on impl=ep_ht, overlap=0.25, prompt=reasoning_brief.
- num_repeats=1, warmup_cases=0. mean ± stdev across reps in summary_agg.csv.
- Best (by mean score desc, mean e2e asc): round_robin__numa_local_first__min_communication / random @ rate=0.1 score_mean=0.8515625 e2e_mean=173.01359807839617

|baseline_key|drop_policy|drop_rate|n_reps|score_mean|score_std|e2e_s_mean|e2e_s_std|prefill_tok_s_mean|decode_tok_s_mean|drop_fraction_total_mean|drop_fraction_remote_mean|dropped_weight_mass_fraction_mean|pass|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|round_robin__numa_local_first__min_communication|random|0.1|1|0.8515625|0.0|173.01359807839617|0.0|80.17461578498387|20.259938279204302|0.09725843654415083|0.12999662710680376|0.09604139601420081|True|
