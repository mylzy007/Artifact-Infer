#!/usr/bin/env bash
# v5: cross-dataset validation on LongBench multifieldqa_en
#
# Why multifieldqa_en vs original target 2wikimqa:
#   - 2wikimqa median prompt = 6.6k tokens, can't fit in 24GB cards (KV insufficient at gpu_util=0.90,
#     combine OOM at gpu_util=0.95)
#   - multifieldqa_en has 33 prompts in [2000, 4500] tokens, similar to LEval (v4 dataset) shape
#   - Reference style is similar (short factoid answers, p50=15 tokens, max 83)
#   - Different task content (single-doc QA vs multi-doc QA in LEval) -> still cross-validates
#
# Setup matches v4 as closely as possible for direct comparison:
#   - 5 GPU policies × 3 rates × 2 plans
#   - max_new_tokens=128 (refs short; LEval was 256 because LEval refs were 106 chars vs ~50 here)
#   - gpu_memory_utilization=0.95, max_num_batched_tokens=6144 (same as v4)
#   - num_samples=32 (4 batches × 8), limited by dataset size in our length range
#
# ETA: ~40 min

set -u
PLAN_DIR=/home/lzy/Artifact-Infer/eval_results/owner_local_ep_phase3_overlap_phase3_joint_align_phase2_ov025_merged_numa_fixed_20260522/overlap_plans
PLAN_RR_MC="${PLAN_DIR}/moe_overlap_plan_round_robin__numa_local_first__ov0p25_min_communication_20260522_170527.json"
PLAN_LBG_GB="${PLAN_DIR}/moe_overlap_plan_load_balanced_greedy_with_locality_tiebreak__numa_local_first__ov0p25_greedy_balance_20260522_171150.json"
OUT_ROOT=/home/lzy/Artifact-Infer/eval_results/owner_local_ep_phase4_drop_longbench_v5
POLICIES="tail_weight,random,cross_numa_first,weighted_tail,cross_numa_uniform"
DATASET=/home/lzy/datasets/moe_benchmarks/prepared/longbench.multifieldqa_en.custom.jsonl

run_one() {
  local TAG="$1"; local PLAN="$2"; local PORT="$3"
  CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:/home/lzy/miniconda3/envs/vllm/bin:$PATH \
    FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_tier2 \
    /home/lzy/miniconda3/envs/vllm/bin/torchrun --nproc_per_node=8 --master_port="$PORT" \
    -m eval.drop.run_owner_local_ep_phase4_drop_multipolicy_v2 \
    --output-dir "${OUT_ROOT}/${TAG}" \
    --dataset "${DATASET}" \
    --num-samples 32 --batch-size 8 --warmup-batches 2 \
    --max-new-tokens 128 \
    --min-prompt-tokens 2000 --max-prompt-tokens 4500 \
    --drop-rates "0.1,0.3,0.5" \
    --drop-policies "${POLICIES}" \
    --expert-overlap-path "${PLAN}" \
    --gpu-memory-utilization 0.95 --max-num-batched-tokens 6144
}

mkdir -p "${OUT_ROOT}"
rm -rf "${OUT_ROOT}/rr_mincomm" "${OUT_ROOT}/lbg_greedybal"
echo "=== v5 run 1/2: RR / min_comm on multifieldqa_en ==="
run_one "rr_mincomm" "${PLAN_RR_MC}" 29561
echo
echo "=== v5 run 2/2: LBG / greedy_balance on multifieldqa_en ==="
run_one "lbg_greedybal" "${PLAN_LBG_GB}" 29562
echo
echo "DONE — outputs in ${OUT_ROOT}/{rr_mincomm,lbg_greedybal}"
