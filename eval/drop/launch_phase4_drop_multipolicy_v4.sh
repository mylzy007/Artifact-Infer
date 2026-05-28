#!/usr/bin/env bash
# v4: same axes as v3 but
#   - max_new_tokens=256 (lets model emit CoT preamble + actual answer)
#   - num-samples=64 (reduce to keep budget; 2 plans × 64 samples × 256 decode)
#   - bench stores full gen_text + ref_text (not [:200]) for offline re-scoring
#
# ETA: ~120 min on 8×4090
#   per batch ≈ 5s prefill + ~20s decode = ~25s
#   per cell = 8 batches × 25s = ~200s
#   per plan = 16 cells × 200s = ~54 min
#   total = 2 plans × 54 min ≈ 110 min

set -u
PLAN_DIR=/home/lzy/Artifact-Infer/eval_results/owner_local_ep_phase3_overlap_phase3_joint_align_phase2_ov025_merged_numa_fixed_20260522/overlap_plans
PLAN_RR_MC="${PLAN_DIR}/moe_overlap_plan_round_robin__numa_local_first__ov0p25_min_communication_20260522_170527.json"
PLAN_LBG_GB="${PLAN_DIR}/moe_overlap_plan_load_balanced_greedy_with_locality_tiebreak__numa_local_first__ov0p25_greedy_balance_20260522_171150.json"
OUT_ROOT=/home/lzy/Artifact-Infer/eval_results/owner_local_ep_phase4_drop_longbench_v4
POLICIES="tail_weight,random,cross_numa_first,weighted_tail,cross_numa_uniform"

run_one() {
  local TAG="$1"; local PLAN="$2"; local PORT="$3"
  CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:/home/lzy/miniconda3/envs/vllm/bin:$PATH \
    FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_tier2 \
    /home/lzy/miniconda3/envs/vllm/bin/torchrun --nproc_per_node=8 --master_port="$PORT" \
    -m eval.drop.run_owner_local_ep_phase4_drop_multipolicy_v2 \
    --output-dir "${OUT_ROOT}/${TAG}" \
    --dataset /home/lzy/datasets/moe_benchmarks/prepared/leval.Generation_multidoc_qa.custom.jsonl \
    --num-samples 64 --batch-size 8 --warmup-batches 3 \
    --max-new-tokens 256 \
    --drop-rates "0.1,0.3,0.5" \
    --drop-policies "${POLICIES}" \
    --expert-overlap-path "${PLAN}" \
    --gpu-memory-utilization 0.95 --max-num-batched-tokens 6144
}

mkdir -p "${OUT_ROOT}"
echo "=== v4 run 1/2: RR / numa_local_first / min_communication ==="
run_one "rr_mincomm" "${PLAN_RR_MC}" 29531
echo
echo "=== v4 run 2/2: LBG / numa_local_first / greedy_balance ==="
run_one "lbg_greedybal" "${PLAN_LBG_GB}" 29532
echo
echo "DONE — outputs in ${OUT_ROOT}/{rr_mincomm,lbg_greedybal}"
