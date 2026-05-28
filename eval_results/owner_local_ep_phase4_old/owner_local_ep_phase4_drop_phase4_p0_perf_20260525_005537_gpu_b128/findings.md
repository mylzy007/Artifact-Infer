# Owner Local EP Phase 4 Drop Findings

- Phase 4 drop sweep on impl=ep_ht, overlap=0.25, prompt=reasoning_brief.
- num_repeats=3, warmup_cases=1. mean ± stdev across reps in summary_agg.csv.
- Best (by mean score desc, mean e2e asc): round_robin__numa_local_first__min_communication / cross_numa_first @ rate=0.2 score_mean=0.8203125 e2e_mean=106.16094850453858

|baseline_key|drop_policy|drop_rate|n_reps|score_mean|score_std|e2e_s_mean|e2e_s_std|prefill_tok_s_mean|decode_tok_s_mean|drop_fraction_total_mean|drop_fraction_remote_mean|dropped_weight_mass_fraction_mean|pass|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|round_robin__numa_local_first__min_communication|none|0.0|3|0.7994791666666666|0.0022552744890219755|104.16870566038415|3.2234803984096088|235.50440125005605|28.088370283614328|None|None|None|True|
|round_robin__numa_local_first__min_communication|tail_weight|0.05|3|0.7903645833333334|0.014788823817188212|104.22183263100062|1.3915840307668443|227.32006923934796|29.227614580661456|None|None|None|True|
|round_robin__numa_local_first__min_communication|tail_weight|0.1|3|0.78125|0.0078125|103.44544681084032|0.44338123548844527|223.4979724011987|28.8340992547375|None|None|None|True|
|round_robin__numa_local_first__min_communication|tail_weight|0.2|3|0.80078125|0.010334966058846057|114.76787443862607|9.046625297752447|217.7156259635524|26.04271934727978|None|None|None|True|
|round_robin__numa_local_first__min_communication|random|0.05|3|0.8151041666666666|0.04627436365603349|107.58647893198456|3.7544164089349303|223.30411131901306|27.30370850186189|None|None|None|True|
|round_robin__numa_local_first__min_communication|random|0.1|3|0.8033854166666666|0.029577647634376425|105.42347838341568|2.676878219536376|221.42305294457063|27.43711890169392|None|None|None|True|
|round_robin__numa_local_first__min_communication|random|0.2|3|0.6966145833333334|0.17730818292388772|105.57401803865407|7.358037713508165|244.01045601429993|28.811691415884386|None|None|None|True|
|round_robin__numa_local_first__min_communication|cross_numa_first|0.05|3|0.80078125|0.010334966058846057|104.03441131596144|0.6436552077140066|225.02511553614852|28.699063371298276|None|None|None|True|
|round_robin__numa_local_first__min_communication|cross_numa_first|0.1|3|0.80859375|0.010334966058846057|107.89875613696252|4.441027361138779|225.37339695501996|27.325902873559073|None|None|None|True|
|round_robin__numa_local_first__min_communication|cross_numa_first|0.2|3|0.8203125|0.00390625|106.16094850453858|2.834058150275072|222.4379131759495|27.780014076393446|None|None|None|True|
