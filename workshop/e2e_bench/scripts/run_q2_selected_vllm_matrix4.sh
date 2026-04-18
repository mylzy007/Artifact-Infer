#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "$ACTIVATE"
fi

source /home/lzy/miniconda3/etc/profile.d/conda.sh
conda activate vllm

RUNNER="${ROOT_DIR}/scripts/run_deepseek_v2_lite_vllm_longbench_multi.sh"
LOG_DIR="${ROOT_DIR}/refine-logs/q2-run-logs"
mkdir -p "$LOG_DIR"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
BENCHMARK_NUM_PROMPTS="${BENCHMARK_NUM_PROMPTS:-128}"
BENCHMARK_MAX_CONCURRENCY="${BENCHMARK_MAX_CONCURRENCY:-8}"
BENCHMARK_OUTPUT_LEN="${BENCHMARK_OUTPUT_LEN:-32}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
CHUNK_MAX_NUM_BATCHED_TOKENS="${CHUNK_MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
ENABLE_MIXED_PREFILL_DECODE="${ENABLE_MIXED_PREFILL_DECODE:-0}"

should_skip_case() {
  local dataset_id="$1"
  local case_name="$2"
  local bench_json="/home/lzy/eval/vllm_deepseek_v2_lite_matrix4/${dataset_id}/${case_name}/bench.json"
  if [[ ! -f "$bench_json" ]]; then
    return 1
  fi

  /home/lzy/miniconda3/envs/vllm/bin/python - "$bench_json" "$BENCHMARK_NUM_PROMPTS" <<'PY'
import json
import sys
from pathlib import Path

bench_json = Path(sys.argv[1])
expected = int(sys.argv[2])
obj = json.loads(bench_json.read_text(encoding="utf-8"))
completed = int(obj.get("completed", 0) or 0)
failed = int(obj.get("failed", 0) or 0)
num_prompts = int(obj.get("num_prompts", 0) or 0)
raise SystemExit(0 if completed == expected and failed == 0 and num_prompts == expected else 1)
PY
}

run_dataset_cases() {
  local dataset_id="$1"
  local benchmark_dataset="$2"
  local benchmark_subset="$3"
  local custom_dataset_path="${4:-}"
  local case_filter="${5:-}"

  local log_file="${LOG_DIR}/${dataset_id}.$(date +%Y%m%d_%H%M%S).log"
  echo "[run] ${dataset_id} cases=${case_filter:-ALL}" | tee -a "$log_file"

  CUDA_DEVICES="$CUDA_DEVICES" \
  BENCHMARK_DATASET="$benchmark_dataset" \
  BENCHMARK_SUBSET="$benchmark_subset" \
  BENCHMARK_NUM_PROMPTS="$BENCHMARK_NUM_PROMPTS" \
  BENCHMARK_MAX_CONCURRENCY="$BENCHMARK_MAX_CONCURRENCY" \
  BENCHMARK_OUTPUT_LEN="$BENCHMARK_OUTPUT_LEN" \
  CHUNKED_PREFILL_SIZE="$CHUNKED_PREFILL_SIZE" \
  CHUNK_MAX_NUM_BATCHED_TOKENS="$CHUNK_MAX_NUM_BATCHED_TOKENS" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  ENABLE_MIXED_PREFILL_DECODE="$ENABLE_MIXED_PREFILL_DECODE" \
  CUSTOM_DATASET_PATH="$custom_dataset_path" \
  CASE_FILTER="$case_filter" \
  bash "$RUNNER" 2>&1 | tee -a "$log_file"
}

run_remaining_cases_2wikimqa() {
  local dataset_id="longbench_2wikimqa"
  local missing_cases=()
  local case_name
  for case_name in nochunk_noeplb nochunk_eplb chunk_noeplb chunk_eplb; do
    if should_skip_case "$dataset_id" "$case_name"; then
      echo "[skip] ${dataset_id}/${case_name} already complete"
      continue
    fi
    missing_cases+=("$case_name")
  done
  if ((${#missing_cases[@]} == 0)); then
    echo "[skip] ${dataset_id} all four cases already complete"
    return 0
  fi
  local joined
  joined="$(IFS=,; echo "${missing_cases[*]}")"
  run_dataset_cases "$dataset_id" "longbench" "2wikimqa" "" "$joined"
}

run_legal_or_backup() {
  local smoke_log="${LOG_DIR}/legal_contract_qa_smoke.$(date +%Y%m%d_%H%M%S).log"
  set +e
  CUDA_DEVICES="$CUDA_DEVICES" \
  BENCHMARK_DATASET="leval" \
  BENCHMARK_SUBSET="Generation/legal_contract_qa" \
  CUSTOM_DATASET_PATH="/home/lzy/datasets/moe_benchmarks/prepared/leval.Generation_legal_contract_qa.custom.jsonl" \
  BENCHMARK_NUM_PROMPTS="8" \
  BENCHMARK_MAX_CONCURRENCY="8" \
  BENCHMARK_OUTPUT_LEN="$BENCHMARK_OUTPUT_LEN" \
  CHUNKED_PREFILL_SIZE="$CHUNKED_PREFILL_SIZE" \
  CHUNK_MAX_NUM_BATCHED_TOKENS="$CHUNK_MAX_NUM_BATCHED_TOKENS" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  ENABLE_MIXED_PREFILL_DECODE="$ENABLE_MIXED_PREFILL_DECODE" \
  CASE_FILTER="chunk_noeplb" \
  bash "$RUNNER" >"$smoke_log" 2>&1
  local smoke_status=$?
  set -e

  if [[ $smoke_status -eq 0 ]]; then
    echo "[smoke-pass] legal_contract_qa"
    run_dataset_cases \
      "leval_Generation_legal_contract_qa" \
      "leval" \
      "Generation/legal_contract_qa" \
      "/home/lzy/datasets/moe_benchmarks/prepared/leval.Generation_legal_contract_qa.custom.jsonl"
    return 0
  fi

  echo "[smoke-fail] legal_contract_qa -> fallback paper_assistant"
  run_dataset_cases \
    "leval_Generation_paper_assistant" \
    "leval" \
    "Generation/paper_assistant" \
    ""
}

run_remaining_cases_2wikimqa
run_dataset_cases "longbench_triviaqa" "longbench" "triviaqa"
run_legal_or_backup

echo "[done] q2 selected datasets full-case batch finished"
