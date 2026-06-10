#!/usr/bin/env bash
set -euo pipefail

cd /home/lzy/Artifact-Infer

DATA=/home/lzy/Artifact-Infer/eval_results/ep_ll_mmax_probe_4rows.jsonl
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:/home/lzy/miniconda3/envs/vllm/bin:$PATH

run_one() {
  local label="$1"
  local m="$2"
  local port="$3"
  local outdir="$4"
  echo "=== RUN ${label} ===" | tee -a /home/lzy/Artifact-Infer/eval_results/ep_ll_mmax4_batch_20260603.log
  FLASHINFER_WORKSPACE_BASE="/tmp/flashinfer_ep_ll_mmax4_${label}" \
    /home/lzy/miniconda3/envs/vllm/bin/torchrun \
      --nproc_per_node=8 \
      --master_port="${port}" \
      -m eval.drop.run_owner_local_ep_phase4_ep_ll_mmax_passage_retrieval \
      --output-dir "${outdir}" \
      --dataset "${DATA}" \
      --num-samples 4 \
      --batch-size 4 \
      --warmup-batches 0 \
      --m-max-values "${m}" \
      --max-new-tokens 8 \
      --max-num-batched-tokens 4352 \
      --min-prompt-tokens 3800 \
      --max-prompt-tokens 4200 \
      > "${outdir}/run.log" 2>&1
}

run_one auto -1 29641 /home/lzy/Artifact-Infer/eval_results/ep_ll_mmax4_auto_20260603
run_one m768 768 29642 /home/lzy/Artifact-Infer/eval_results/ep_ll_mmax4_m768_20260603
run_one m512 512 29643 /home/lzy/Artifact-Infer/eval_results/ep_ll_mmax4_m512_20260603
