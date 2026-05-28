# Owner Local EP Phase 1 Findings

- Baseline impl candidates: ep_ll_triton, ep_ht.
- Recommended baseline impl for later placement/replica work: ep_ht.
- Routing profile source semantics should now be `owner_rank`, avoiding the old repeated-source dp_leader interpretation.
- Dominant time on the current best run: prefill.
- Phase 2 is ready if at least one impl passed and emitted routing_profile_quality.json.

|impl|pass|average_score|e2e_total_time_s|prefill_throughput_tok_s|decode_throughput_tok_s|profile_identical_fraction_mean|profile_pairwise_l1_mean|profile_pairwise_cosine_mean|
|---|---|---|---|---|---|---|---|---|
|ep_ll_triton|False|None|None|None|None||||
|ep_ht|True|0.015625|102.38088160101324|59.219364524605304|41.41068937274603|0.125|4014.964285714286|0.9924599421882361|
