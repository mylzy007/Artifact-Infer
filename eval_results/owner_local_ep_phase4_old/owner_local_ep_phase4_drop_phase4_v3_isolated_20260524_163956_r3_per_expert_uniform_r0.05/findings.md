# Owner Local EP Phase 4 Drop Findings

- Phase 4 drop sweep on impl=ep_ht, overlap=0.25, prompt=reasoning_brief.
- num_repeats=1, warmup_cases=0. mean ± stdev across reps in summary_agg.csv.
- Best (by mean score desc, mean e2e asc): round_robin__numa_local_first__min_communication / per_expert_uniform @ rate=0.05 score_mean=0.94140625 e2e_mean=159.5185499456711

|baseline_key|drop_policy|drop_rate|n_reps|score_mean|score_std|e2e_s_mean|e2e_s_std|prefill_tok_s_mean|decode_tok_s_mean|drop_fraction_total_mean|drop_fraction_remote_mean|dropped_weight_mass_fraction_mean|pass|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|round_robin__numa_local_first__min_communication|per_expert_uniform|0.05|1|0.94140625|0.0|159.5185499456711|0.0|83.84666947251804|23.77414938814379|0.012356307752640997|0.016510397676754107|0.0057532737354544575|True|
