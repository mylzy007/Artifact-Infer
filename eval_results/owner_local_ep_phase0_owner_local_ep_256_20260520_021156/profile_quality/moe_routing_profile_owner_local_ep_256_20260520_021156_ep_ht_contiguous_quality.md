# Routing Profile Quality Report

- Profile: `/home/lzy/Artifact-Infer/eval_results/owner_local_ep_phase0_owner_local_ep_256_20260520_021156/profiles/moe_routing_profile_owner_local_ep_256_20260520_021156_ep_ht_contiguous.json`
- TP=1 EP=8 world_size=8
- Layers=2 experts=128

## Summary

- identical_fraction_mean: `0.125`
- pairwise_l1_mean: `560.5357142857142`
- pairwise_cosine_mean: `0.4063763775160413`
- all_layers_identical: `False`
- fully_degenerate_layer_ids: `[]`

## Per Layer

| layer | identical_fraction | pairwise_l1_mean | pairwise_cosine_mean | entropy_mean | all_rows_identical |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.125 | 556.0714285714286 | 0.41043984608788364 | 2.2656754610171266 | False |
| 1 | 0.125 | 565.0 | 0.40231290894419897 | 2.2337385122762656 | False |
