# Owner Local EP Phase 0 Findings

- Bring-up impls requested: ep_ll_torch, ep_ll_triton, ep_ht.
- Owner-local geometry fixed to tp=1, dp=world=8, ep=world=8.
- Impls that passed end-to-end bring-up: ep_ht.
- Current source semantics: owner_local_ep profiles should report `owner_rank`; vllm_dp_ep remains `dp_leader`.
- Phase 1 is ready if at least one fast impl has pass=true and a real routing profile.

|impl|pass|exit_code|output_order_ok|profile_src_dim|profile_source_definition|profile_path|
|---|---|---|---|---|---|---|
|ep_ll_torch|False|1|False|None|None|None|
|ep_ll_triton|False|1|False|None|None|None|
|ep_ht|True|0|True|8|owner_rank|/home/lzy/Artifact-Infer/eval_results/owner_local_ep_phase0_owner_local_ep_256_20260520_021156/profiles/moe_routing_profile_owner_local_ep_256_20260520_021156_ep_ht_contiguous.json|
