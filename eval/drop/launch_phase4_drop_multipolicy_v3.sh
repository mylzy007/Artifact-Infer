#!/usr/bin/env bash
# v3: run v2 sweep with 5 GPU policies, 3 rates, across 2 Phase 3 overlap plans.
# Each invocation reuses the v2 sweep script with a different overlap plan.
# Total: ~75 min (2 plans × ~35 min each on 8×4090).

set -u
PLAN_DIR=/home/lzy/Artifact-Infer/eval_results/owner_local_ep_phase3_overlap_phase3_joint_align_phase2_ov025_merged_numa_fixed_20260522/overlap_plans
PLAN_RR_MC="${PLAN_DIR}/moe_overlap_plan_round_robin__numa_local_first__ov0p25_min_communication_20260522_170527.json"
PLAN_LBG_GB="${PLAN_DIR}/moe_overlap_plan_load_balanced_greedy_with_locality_tiebreak__numa_local_first__ov0p25_greedy_balance_20260522_171150.json"
OUT_ROOT=/home/lzy/Artifact-Infer/eval_results/owner_local_ep_phase4_drop_longbench_v3

POLICIES="tail_weight,random,cross_numa_first,weighted_tail,cross_numa_uniform"

run_one() {
  local TAG="$1"; local PLAN="$2"
  CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:/home/lzy/miniconda3/envs/vllm/bin:$PATH \
    FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_tier2 \
    /home/lzy/miniconda3/envs/vllm/bin/torchrun --nproc_per_node=8 --master_port="$3" \
    -m eval.drop.run_owner_local_ep_phase4_drop_multipolicy_v2 \
    --output-dir "${OUT_ROOT}/${TAG}" \
    --dataset /home/lzy/datasets/moe_benchmarks/prepared/leval.Generation_multidoc_qa.custom.jsonl \
    --num-samples 80 --batch-size 8 --warmup-batches 3 --max-new-tokens 64 \
    --drop-rates "0.1,0.3,0.5" \
    --drop-policies "${POLICIES}" \
    --expert-overlap-path "${PLAN}" \
    --gpu-memory-utilization 0.95 --max-num-batched-tokens 6144
}

mkdir -p "${OUT_ROOT}"
echo "=== run 1/2: RR / numa_local_first / min_communication ==="
run_one "rr_mincomm" "${PLAN_RR_MC}" 29521
echo
echo "=== run 2/2: LBG / numa_local_first / greedy_balance ==="
run_one "lbg_greedybal" "${PLAN_LBG_GB}" 29522
echo
echo "DONE — outputs in ${OUT_ROOT}/{rr_mincomm,lbg_greedybal}"
