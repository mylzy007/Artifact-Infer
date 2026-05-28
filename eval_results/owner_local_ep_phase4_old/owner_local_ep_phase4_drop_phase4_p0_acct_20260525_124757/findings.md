# Owner Local EP Phase 4 Drop Findings

- Phase 4 drop sweep on impl=ep_ht, overlap=0.25, prompt=reasoning_brief.
- num_repeats=1, warmup_cases=1. mean ± stdev across reps in summary_agg.csv.
- Best (by mean score desc, mean e2e asc): round_robin__numa_local_first__min_communication / cross_numa_first @ rate=0.05 score_mean=0.84375 e2e_mean=40.481952402275056

|baseline_key|drop_policy|drop_rate|n_reps|score_mean|score_std|e2e_s_mean|e2e_s_std|prefill_tok_s_mean|decode_tok_s_mean|drop_fraction_total_mean|drop_fraction_remote_mean|dropped_weight_mass_fraction_mean|pass|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|round_robin__numa_local_first__min_communication|none|0.0|1|0.78125|0.0|38.51232624985278|0.0|45.52258397644324|13.950130017185217|0.0|0.0|0.0|True|
|round_robin__numa_local_first__min_communication|tail_weight|0.05|1|0.6875|0.0|42.781125214882195|0.0|41.378553050565415|12.892708985704866|0.04999131341209173|0.06699090224551933|0.019327014326273564|True|
|round_robin__numa_local_first__min_communication|tail_weight|0.1|1|0.75|0.0|36.939634919166565|0.0|43.88730519382064|13.67337948431401|0.09993919388464212|0.13394893960727813|0.04502639510004882|True|
|round_robin__numa_local_first__min_communication|tail_weight|0.2|1|0.75|0.0|42.35117453476414|0.0|42.637082152634875|12.475634205586333|0.20000868658790827|0.26802701126358536|0.10633595381173129|True|
|round_robin__numa_local_first__min_communication|random|0.05|1|0.78125|0.0|38.794012574944645|0.0|43.99194525950903|13.607337997117416|0.04999131341209173|0.06697855759940159|0.04889781219992584|True|
|round_robin__numa_local_first__min_communication|random|0.1|1|0.78125|0.0|40.167046824935824|0.0|44.06384259997341|13.481797593834598|0.09993919388464212|0.13389957289566556|0.09863834435930531|True|
|round_robin__numa_local_first__min_communication|random|0.2|1|0.75|0.0|44.256249621044844|0.0|43.21325133815441|11.898531996555555|0.20000868658790827|0.26777147841255994|0.19731798201204348|True|
|round_robin__numa_local_first__min_communication|cross_numa_first|0.05|1|0.84375|0.0|40.481952402275056|0.0|42.272156893690706|13.399157383885314|0.04999131341209173|0.0669909834753634|0.019327067148907827|True|
|round_robin__numa_local_first__min_communication|cross_numa_first|0.1|1|0.71875|0.0|39.224111639894545|0.0|45.30545528813661|12.500733802268211|0.09993919388464212|0.13394893960727813|0.04502639510004882|True|
|round_robin__numa_local_first__min_communication|cross_numa_first|0.2|1|0.75|0.0|38.903182508889586|0.0|43.72216389173433|12.207623644638568|0.20000868658790827|0.26802343628744374|0.10632689067078605|True|
