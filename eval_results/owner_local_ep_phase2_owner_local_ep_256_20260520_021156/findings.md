# Owner Local EP Phase 2 Findings

- Placement sweep impl: ep_ht.
- Static placements evaluated: contiguous, round_robin, fixed_random_shuffle, load_balanced_greedy_with_locality_tiebreak, communication_aware_greedy.
- Placement protocol and owner_local_ep runtime semantics stayed unchanged; only script-layer sweep/reporting was added.
- Best current static placement by end-to-end time: round_robin.
- Replica work should inherit the best passing placement from this summary once quality/runtime tradeoffs look stable.

|placement|pass|score|e2e_s|prefill_tok_s|decode_tok_s|cross_traffic_ratio|gpu_cv|profile_pairwise_l1_mean|
|---|---|---|---|---|---|---|---|---|
|contiguous|True|0.015625|100.31945610512048|61.01604182253482|42.21663686264647|0.8749836334135218|0.07198852473111322|4081.111607142857|
|round_robin|True|0.01171875|100.00216745492071|61.946496952166406|42.2048865876366|0.8747850417230492|0.052601391541546125|4099.818452380952|
|fixed_random_shuffle|True|0.015625|107.86134806508198|54.376475846606155|41.21014177902006|0.8746555185936952|0.10800049322510892|4047.360119047619|
|load_balanced_greedy_with_locality_tiebreak|True|0.0234375|104.68378363922238|59.43109647832253|40.59931105163512|0.8749844948128102|0.0008572983211021364|4143.800595238095|
|communication_aware_greedy|True|0.01171875|103.40944771375507|57.346233413329614|42.62011007647955|0.8706616799478775|0.14354279574515166|4086.9002976190477|
