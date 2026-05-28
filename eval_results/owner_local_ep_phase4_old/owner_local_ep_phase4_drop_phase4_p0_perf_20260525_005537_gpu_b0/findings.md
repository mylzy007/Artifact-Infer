# Owner Local EP Phase 4 Drop Findings

- Phase 4 drop sweep on impl=ep_ht, overlap=0.25, prompt=reasoning_brief.
- num_repeats=3, warmup_cases=1. mean ± stdev across reps in summary_agg.csv.
- Best (by mean score desc, mean e2e asc): round_robin__numa_local_first__min_communication / none @ rate=0.0 score_mean=0.8072916666666666 e2e_mean=101.98563547323768

|baseline_key|drop_policy|drop_rate|n_reps|score_mean|score_std|e2e_s_mean|e2e_s_std|prefill_tok_s_mean|decode_tok_s_mean|drop_fraction_total_mean|drop_fraction_remote_mean|dropped_weight_mass_fraction_mean|pass|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|round_robin__numa_local_first__min_communication|none|0.0|3|0.8072916666666666|0.00813150781041458|101.98563547323768|1.72087975217718|239.21271056694354|28.745421706488713|None|None|None|True|
|round_robin__numa_local_first__min_communication|tail_weight|0.05|3|0.796875|0.0067658234670659265|113.9835214621077|4.2814205679686035|226.49187639914325|25.71267883188811|None|None|None|True|
|round_robin__numa_local_first__min_communication|tail_weight|0.1|3|0.7916666666666666|0.005966895436140417|120.46155951994781|9.43856954162501|218.00596074123396|24.144570264936885|None|None|None|True|
|round_robin__numa_local_first__min_communication|tail_weight|0.2|3|0.7981770833333334|0.01578692142315383|118.85936953825876|5.209370425910303|220.64314343300825|24.752939171831898|None|None|None|True|
|round_robin__numa_local_first__min_communication|random|0.05|3|0.7942708333333334|0.009830513587592122|114.34470590452354|2.7433770760042036|233.93509759440323|24.694952077768487|None|None|None|True|
|round_robin__numa_local_first__min_communication|random|0.1|3|0.7526041666666666|0.014788823817188212|111.05832350455846|1.2206141258397194|227.39426223238016|25.68849052173103|None|None|None|True|
|round_robin__numa_local_first__min_communication|random|0.2|3|0.5950520833333334|0.1635190652508778|114.94832755237196|4.110769130738762|227.4406011741745|26.6979121063322|None|None|None|True|
|round_robin__numa_local_first__min_communication|cross_numa_first|0.05|3|0.8046875|0.0067658234670659265|116.2893424066715|3.7975595901126313|224.31701367374603|25.138126050304965|None|None|None|True|
|round_robin__numa_local_first__min_communication|cross_numa_first|0.1|3|0.796875|0.017026948998205758|116.92860516843696|4.172096982466285|222.47957722520187|24.563644998417146|None|None|None|True|
|round_robin__numa_local_first__min_communication|cross_numa_first|0.2|3|0.7890625|0.014084184669781208|116.47137278597802|4.76875593533796|225.8563167308051|25.78560316825639|None|None|None|True|
