# Owner Local EP Phase 4 Drop Findings

- Phase 4 drop sweep on impl=ep_ht, overlap=0.25, prompt=reasoning_brief.
- num_repeats=3, warmup_cases=1. mean ± stdev across reps in summary_agg.csv.
- Best (by mean score desc, mean e2e asc): round_robin__numa_local_first__min_communication / cross_numa_first @ rate=0.2 score_mean=0.80078125 e2e_mean=115.76099822608133

|baseline_key|drop_policy|drop_rate|n_reps|score_mean|score_std|e2e_s_mean|e2e_s_std|prefill_tok_s_mean|decode_tok_s_mean|drop_fraction_total_mean|drop_fraction_remote_mean|dropped_weight_mass_fraction_mean|pass|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|round_robin__numa_local_first__min_communication|none|0.0|3|0.7877604166666666|0.01578692142315383|105.44082526971276|11.080031893291695|238.07183784380712|27.93436704463581|None|None|None|True|
|round_robin__numa_local_first__min_communication|tail_weight|0.05|3|0.7903645833333334|0.005966895436140417|128.27266747985655|3.8702164697127786|212.94255077389028|22.46357730916049|None|None|None|True|
|round_robin__numa_local_first__min_communication|tail_weight|0.1|3|0.79296875|0.00390625|116.86271568729232|2.678928811331625|212.2381178200338|25.42021907919099|None|None|None|True|
|round_robin__numa_local_first__min_communication|tail_weight|0.2|3|0.7903645833333334|0.021513947450336336|112.9320446968389|0.20217043700163964|209.0156324381617|26.486590057119656|None|None|None|True|
|round_robin__numa_local_first__min_communication|random|0.05|3|0.79296875|0.010334966058846057|114.70678796060383|1.4092378663671616|215.9032059545245|25.40913693796871|None|None|None|True|
|round_robin__numa_local_first__min_communication|random|0.1|3|0.7369791666666666|0.05389115126712207|114.93680230636771|2.718822402949805|224.77906361631722|25.133061012602642|None|None|None|True|
|round_robin__numa_local_first__min_communication|random|0.2|3|0.63671875|0.0542673593337883|112.85475122199084|1.541395531351488|229.71030132410283|24.90961480303002|None|None|None|True|
|round_robin__numa_local_first__min_communication|cross_numa_first|0.05|3|0.7955729166666666|0.012556836928376244|117.46908227074891|5.931418332565507|214.53041145225407|25.03246561292923|None|None|None|True|
|round_robin__numa_local_first__min_communication|cross_numa_first|0.1|3|0.79296875|0.01953125|123.19440893735737|7.86814486708339|225.61553280488786|23.785670730079215|None|None|None|True|
|round_robin__numa_local_first__min_communication|cross_numa_first|0.2|3|0.80078125|0.0067658234670659265|115.76099822608133|2.8953882366714008|219.09036979892593|24.982406032028035|None|None|None|True|
