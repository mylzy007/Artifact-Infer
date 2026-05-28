#!/usr/bin/env bash
# Full drop sweep on LongBench passage_retrieval_en_e with OFFICIAL pipeline.
#
# Dataset: passage_retrieval_en_e (extended subset, 300 prompts)
#   - 112 prompts fit our 24GB memory budget ([500, 5800] context tokens)
#   - Use 64 per cell (8 batches × 8) for stable accuracy estimate
#
# Pipeline matches THUDM/LongBench eval verbatim:
#   - Official template (no anti-CoT modification needed: model just outputs "Paragraph N")
#   - Official max_new_tokens=32
#   - Official `retrieval_score` metric (binary 1/0 modulo multi-guess penalty)
#   - Also report `retrieval_score_strict` (1.0 iff first integer in pred == gt)
#
# Sweep: 2 plans × (1 baseline + 5 policies × 3 rates) = 32 cells
# ETA: ~70 min on 8×4090

set -u
PLAN_DIR=/home/lzy/Artifact-Infer/eval_results/owner_local_ep_phase3_overlap_phase3_joint_align_phase2_ov025_merged_numa_fixed_20260522/overlap_plans
PLAN_RR_MC="${PLAN_DIR}/moe_overlap_plan_round_robin__numa_local_first__ov0p25_min_communication_20260522_170527.json"
PLAN_LBG_GB="${PLAN_DIR}/moe_overlap_plan_load_balanced_greedy_with_locality_tiebreak__numa_local_first__ov0p25_greedy_balance_20260522_171150.json"
OUT_ROOT=/home/lzy/Artifact-Infer/eval_results/passage_retrieval_drop_sweep

POLICIES="tail_weight,random,cross_numa_first,weighted_tail,cross_numa_uniform"

run_one() {
  local TAG="$1"; local PLAN="$2"; local PORT="$3"
  CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:/home/lzy/miniconda3/envs/vllm/bin:$PATH \
    FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_tier2 \
    /home/lzy/miniconda3/envs/vllm/bin/torchrun --nproc_per_node=8 --master_port="$PORT" \
    -m eval.drop.run_owner_local_ep_phase4_drop_passage_retrieval_v6 \
    --output-dir "${OUT_ROOT}/${TAG}" \
    --num-samples 64 --batch-size 8 --warmup-batches 2 \
    --max-new-tokens 32 \
    --min-prompt-tokens 500 --max-prompt-tokens 5800 \
    --drop-rates "0.1,0.3,0.5" \
    --drop-policies "${POLICIES}" \
    --expert-overlap-path "${PLAN}" \
    --gpu-memory-utilization 0.92 --max-num-batched-tokens 6144
}

mkdir -p "${OUT_ROOT}"
rm -rf "${OUT_ROOT}/rr_mincomm" "${OUT_ROOT}/lbg_greedybal"
echo "=== run 1/2: RR/min_comm on passage_retrieval_en_e ==="
run_one "rr_mincomm" "${PLAN_RR_MC}" 29581
echo
echo "=== run 2/2: LBG/greedy_balance on passage_retrieval_en_e ==="
run_one "lbg_greedybal" "${PLAN_LBG_GB}" 29582
echo
echo "DONE — outputs in ${OUT_ROOT}/{rr_mincomm,lbg_greedybal}"
