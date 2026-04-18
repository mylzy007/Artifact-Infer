#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "$ACTIVATE"
fi

PYTHON_BIN="${PYTHON_BIN:-/home/lzy/miniconda3/envs/vllm/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi
PYTHON_BIN_DIR="$(cd "$(dirname "$PYTHON_BIN")" && pwd)"
export PATH="$PYTHON_BIN_DIR:$PATH"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

BENCHMARK_DATA_ROOT="${BENCHMARK_DATA_ROOT:-$HOME/datasets/moe_benchmarks}"
PREPARED_BENCHMARK_DIR="${PREPARED_BENCHMARK_DIR:-$BENCHMARK_DATA_ROOT/prepared}"
REBUILD_PREPARED_BENCHMARKS="${REBUILD_PREPARED_BENCHMARKS:-0}"

MODEL_DIR="${MODEL_DIR:-$HOME/models/DeepSeek-V2-Lite-Chat}"
MODEL_CONFIG="${MODEL_CONFIG:-$MODEL_DIR/config.json}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseekv2lite_vllm}"

HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-32080}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.68}"
GPU_MEMORY_UTILIZATION_RETRY_VALUES="${GPU_MEMORY_UTILIZATION_RETRY_VALUES:-$GPU_MEMORY_UTILIZATION,0.64,0.60,0.56,0.52,0.48,0.44,0.40,0.36}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"

CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
NOCHUNK_MAX_NUM_BATCHED_TOKENS="${NOCHUNK_MAX_NUM_BATCHED_TOKENS:-32768}"
CHUNK_MAX_NUM_BATCHED_TOKENS="${CHUNK_MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_PARTIAL_PREFILLS="${MAX_NUM_PARTIAL_PREFILLS:-1}"
MAX_LONG_PARTIAL_PREFILLS="${MAX_LONG_PARTIAL_PREFILLS:-1}"
ENABLE_MIXED_PREFILL_DECODE="${ENABLE_MIXED_PREFILL_DECODE:-}"
DISABLE_MIXED_PREFILL_DECODE="${DISABLE_MIXED_PREFILL_DECODE:-1}"

if [[ -n "$ENABLE_MIXED_PREFILL_DECODE" ]]; then
  if [[ "$ENABLE_MIXED_PREFILL_DECODE" == "1" ]]; then
    DISABLE_MIXED_PREFILL_DECODE="0"
  else
    DISABLE_MIXED_PREFILL_DECODE="1"
  fi
fi

EXPERT_RECORDER_MODE="${EXPERT_RECORDER_MODE:-stat}"
EXPERT_RECORDER_ENFORCE_EAGER="${EXPERT_RECORDER_ENFORCE_EAGER:-1}"
GENERATE_HEATMAPS="${GENERATE_HEATMAPS:-1}"
HEATMAP_LAYER="${HEATMAP_LAYER:--1}"
HEATMAP_SKIP_INITIAL_STEPS="${HEATMAP_SKIP_INITIAL_STEPS:-0}"
HEATMAP_VMAX_PERCENTILE="${HEATMAP_VMAX_PERCENTILE:-99}"

EPLB_STEP_INTERVAL="${EPLB_STEP_INTERVAL:-100}"
EPLB_WINDOW_SIZE="${EPLB_WINDOW_SIZE:-200}"
EPLB_LOG_BALANCEDNESS="${EPLB_LOG_BALANCEDNESS:-1}"
EPLB_LOG_BALANCEDNESS_INTERVAL="${EPLB_LOG_BALANCEDNESS_INTERVAL:-1}"
EPLB_USE_ASYNC="${EPLB_USE_ASYNC:-0}"

LONGBENCH_SOURCE_PATH="${LONGBENCH_SOURCE_PATH:-$BENCHMARK_DATA_ROOT/longbench}"
LONGBENCH_SUBSET="${LONGBENCH_SUBSET:-multifieldqa_en}"
LONGBENCH_NUM_PROMPTS="${LONGBENCH_NUM_PROMPTS:-128}"
LONGBENCH_MAX_CONCURRENCY="${LONGBENCH_MAX_CONCURRENCY:-8}"
LONGBENCH_OUTPUT_LEN="${LONGBENCH_OUTPUT_LEN:-32}"

BENCHMARK_DATASET="${BENCHMARK_DATASET:-}"
BENCHMARK_SUBSET="${BENCHMARK_SUBSET:-}"
BENCHMARK_SOURCE_PATH="${BENCHMARK_SOURCE_PATH:-}"
CUSTOM_DATASET_PATH="${CUSTOM_DATASET_PATH:-}"
BENCHMARK_NUM_PROMPTS="${BENCHMARK_NUM_PROMPTS:-}"
BENCHMARK_MAX_CONCURRENCY="${BENCHMARK_MAX_CONCURRENCY:-}"
BENCHMARK_OUTPUT_LEN="${BENCHMARK_OUTPUT_LEN:-}"

REQUEST_RATE="${REQUEST_RATE:-inf}"
NUM_WARMUPS="${NUM_WARMUPS:-0}"
RESULT_ROOT="${RESULT_ROOT:-$HOME/eval/vllm_deepseek_v2_lite_matrix4}"
CASE_FILTER="${CASE_FILTER:-}"

mkdir -p "$RESULT_ROOT" "$PREPARED_BENCHMARK_DIR"
LOCAL_NO_PROXY="127.0.0.1,localhost"
if [[ -n "${NO_PROXY:-}" ]]; then
  LOCAL_NO_PROXY="${LOCAL_NO_PROXY},${NO_PROXY}"
fi
export NO_PROXY="$LOCAL_NO_PROXY"
export no_proxy="$LOCAL_NO_PROXY"
SUMMARY_CSV="$RESULT_ROOT/summary.csv"
if [[ ! -f "$SUMMARY_CSV" || ! -s "$SUMMARY_CSV" || "${RESET_SUMMARY_CSV:-0}" == "1" ]]; then
cat >"$SUMMARY_CSV" <<'CSV'
profile_name,case_name,chunked_prefill,eplb,dataset_name,completed,failed,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,p99_ttft_ms,mean_tpot_ms,p99_tpot_ms,result_dir,bench_json,expert_record_dir
CSV
fi

if [[ ! -f "$MODEL_CONFIG" ]]; then
  echo "ERROR: model config not found: $MODEL_CONFIG"
  exit 1
fi

count_csv_items() {
  local value="$1"
  awk -F',' '{print NF}' <<<"$value"
}

NUM_VISIBLE_GPUS="$(count_csv_items "$CUDA_DEVICES")"
if [[ -z "$TENSOR_PARALLEL_SIZE" ]]; then
  TENSOR_PARALLEL_SIZE="$NUM_VISIBLE_GPUS"
fi

if (( TENSOR_PARALLEL_SIZE <= 0 )); then
  echo "ERROR: TENSOR_PARALLEL_SIZE must be > 0"
  exit 1
fi

sanitize_name() {
  local value="$1"
  value="${value//\//_}"
  value="${value// /_}"
  value="${value//./_}"
  while [[ "$value" == *"__"* ]]; do
    value="${value//__/_}"
  done
  echo "$value"
}

dedupe_csv_preserve_order() {
  local raw="$1"
  awk -v raw="$raw" 'BEGIN {
    n = split(raw, parts, ",");
    out = "";
    for (i = 1; i <= n; ++i) {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", parts[i]);
      if (parts[i] == "" || seen[parts[i]]++) {
        continue;
      }
      out = (out == "" ? parts[i] : out "," parts[i]);
    }
    print out;
  }'
}

GPU_MEMORY_UTILIZATION_RETRY_VALUES="$(dedupe_csv_preserve_order "$GPU_MEMORY_UTILIZATION_RETRY_VALUES")"

if [[ -z "$BENCHMARK_DATASET" ]]; then
  BENCHMARK_DATASET="longbench"
  BENCHMARK_SUBSET="${LONGBENCH_SUBSET}"
  BENCHMARK_SOURCE_PATH="${LONGBENCH_SOURCE_PATH}"
  BENCHMARK_NUM_PROMPTS="${LONGBENCH_NUM_PROMPTS}"
  BENCHMARK_MAX_CONCURRENCY="${LONGBENCH_MAX_CONCURRENCY}"
  BENCHMARK_OUTPUT_LEN="${LONGBENCH_OUTPUT_LEN}"
fi

if [[ -z "$BENCHMARK_NUM_PROMPTS" ]]; then
  BENCHMARK_NUM_PROMPTS="128"
fi
if [[ -z "$BENCHMARK_MAX_CONCURRENCY" ]]; then
  BENCHMARK_MAX_CONCURRENCY="8"
fi
if [[ -z "$BENCHMARK_OUTPUT_LEN" ]]; then
  BENCHMARK_OUTPUT_LEN="32"
fi

DATASET_NAME="custom"
PROFILE_NAME=""
DATASET_PATH=""

case "$BENCHMARK_DATASET" in
  longbench)
    if [[ -z "$BENCHMARK_SUBSET" ]]; then
      BENCHMARK_SUBSET="${LONGBENCH_SUBSET}"
    fi
    if [[ -z "$BENCHMARK_SOURCE_PATH" ]]; then
      BENCHMARK_SOURCE_PATH="${LONGBENCH_SOURCE_PATH}"
    fi
    PROFILE_NAME="longbench_$(sanitize_name "$BENCHMARK_SUBSET")"
    if [[ -n "$CUSTOM_DATASET_PATH" ]]; then
      DATASET_PATH="$CUSTOM_DATASET_PATH"
    else
      DATASET_PATH="$PREPARED_BENCHMARK_DIR/longbench.${BENCHMARK_SUBSET}.custom.jsonl"
    fi
    ;;
  leval)
    if [[ -z "$BENCHMARK_SUBSET" ]]; then
      BENCHMARK_SUBSET="Generation/multidoc_qa"
    fi
    if [[ -z "$BENCHMARK_SOURCE_PATH" ]]; then
      BENCHMARK_SOURCE_PATH="$BENCHMARK_DATA_ROOT/leval"
    fi
    PROFILE_NAME="leval_$(sanitize_name "$BENCHMARK_SUBSET")"
    if [[ -n "$CUSTOM_DATASET_PATH" ]]; then
      DATASET_PATH="$CUSTOM_DATASET_PATH"
    else
      DATASET_PATH="$PREPARED_BENCHMARK_DIR/leval.$(sanitize_name "$BENCHMARK_SUBSET").custom.jsonl"
    fi
    ;;
  longbench_v2)
    BENCHMARK_SUBSET="${BENCHMARK_SUBSET:-}"
    if [[ -z "$BENCHMARK_SOURCE_PATH" ]]; then
      BENCHMARK_SOURCE_PATH="$BENCHMARK_DATA_ROOT/longbench_v2"
    fi
    PROFILE_NAME="longbench_v2"
    if [[ -n "$CUSTOM_DATASET_PATH" ]]; then
      DATASET_PATH="$CUSTOM_DATASET_PATH"
    else
      DATASET_PATH="$PREPARED_BENCHMARK_DIR/longbench_v2.custom.jsonl"
    fi
    ;;
  *)
    echo "ERROR: unsupported BENCHMARK_DATASET='$BENCHMARK_DATASET'. Use longbench | leval | longbench_v2."
    exit 1
    ;;
esac

dataset_has_prompt_column() {
  local path="$1"
  "$PYTHON_BIN" - "$path" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        for _ in range(8):
            line = f.readline()
            if not line:
                break
            obj = json.loads(line)
            if "prompt" in obj and isinstance(obj["prompt"], str):
                raise SystemExit(0)
            if obj:
                raise SystemExit(2)
except FileNotFoundError:
    raise SystemExit(3)
except Exception:
    raise SystemExit(4)

raise SystemExit(5)
PY
}

prepare_custom_dataset() {
  if [[ -n "$CUSTOM_DATASET_PATH" ]]; then
    if [[ ! -f "$DATASET_PATH" ]]; then
      echo "ERROR: CUSTOM_DATASET_PATH does not exist: $DATASET_PATH"
      exit 1
    fi
    return 0
  fi

  local -a args
  args=(
    "$ROOT_DIR/scripts/prepare_bench_custom_jsonl.py"
    --dataset "$BENCHMARK_DATASET"
    --source "$BENCHMARK_SOURCE_PATH"
    --out "$DATASET_PATH"
    --output-tokens "$BENCHMARK_OUTPUT_LEN"
  )
  if [[ "$BENCHMARK_DATASET" != "longbench_v2" && -n "$BENCHMARK_SUBSET" ]]; then
    args+=(--subset "$BENCHMARK_SUBSET")
  fi
  if [[ -f "$DATASET_PATH" ]] && [[ "$REBUILD_PREPARED_BENCHMARKS" != "1" ]]; then
    if ! dataset_has_prompt_column "$DATASET_PATH"; then
      echo "[warn] existing dataset lacks 'prompt' column, rebuilding: $DATASET_PATH"
      args+=(--force)
    fi
  fi
  if [[ "$REBUILD_PREPARED_BENCHMARKS" == "1" ]]; then
    args+=(--force)
  fi
  "$PYTHON_BIN" "${args[@]}"
}

append_summary() {
  local summary_csv="$SUMMARY_CSV"
  local bench_json="$1"
  local profile_name="$2"
  local case_name="$3"
  local enable_chunk="$4"
  local enable_eplb="$5"
  local dataset_name="$6"
  local result_dir="$7"
  local expert_record_dir="$8"

  "$PYTHON_BIN" - "$summary_csv" "$bench_json" "$profile_name" "$case_name" "$enable_chunk" \
    "$enable_eplb" "$dataset_name" "$result_dir" "$expert_record_dir" <<'PY'
import csv
import json
import sys
from pathlib import Path

(
    summary_csv,
    bench_json,
    profile_name,
    case_name,
    enable_chunk,
    enable_eplb,
    dataset_name,
    result_dir,
    expert_record_dir,
) = sys.argv[1:]

fieldnames = [
    "profile_name",
    "case_name",
    "chunked_prefill",
    "eplb",
    "dataset_name",
    "completed",
    "failed",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "result_dir",
    "bench_json",
    "expert_record_dir",
]

path = Path(bench_json)
if not path.exists():
    result = {}
else:
    result = json.loads(path.read_text(encoding="utf-8"))

def g(key, default=""):
    value = result.get(key, default)
    return default if value is None else value

def f(value):
    try:
        return f"{float(value):.6f}"
    except Exception:
        return ""

row = {
    "profile_name": profile_name,
    "case_name": case_name,
    "chunked_prefill": enable_chunk,
    "eplb": enable_eplb,
    "dataset_name": dataset_name,
    "completed": str(g("completed")),
    "failed": str(g("failed")),
    "request_throughput": f(g("request_throughput")),
    "output_throughput": f(g("output_throughput")),
    "total_token_throughput": f(g("total_token_throughput")),
    "mean_ttft_ms": f(g("mean_ttft_ms")),
    "p99_ttft_ms": f(g("p99_ttft_ms")),
    "mean_tpot_ms": f(g("mean_tpot_ms")),
    "p99_tpot_ms": f(g("p99_tpot_ms")),
    "result_dir": result_dir,
    "bench_json": bench_json,
    "expert_record_dir": expert_record_dir,
}

summary_path = Path(summary_csv)
rows = []
if summary_path.exists() and summary_path.stat().st_size > 0:
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for existing in reader:
            if (
                existing.get("profile_name") == profile_name
                and existing.get("case_name") == case_name
            ):
                continue
            rows.append(existing)

rows.append(row)

with summary_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
}

emit_bench_jsonl_compat() {
  local bench_json="$1"
  local bench_jsonl="$2"
  "$PYTHON_BIN" - "$bench_json" "$bench_jsonl" <<'PY'
import json
import sys
from pathlib import Path

bench_json = Path(sys.argv[1])
bench_jsonl = Path(sys.argv[2])

if not bench_json.exists():
    raise SystemExit(0)

obj = json.loads(bench_json.read_text(encoding="utf-8"))
bench_jsonl.write_text(json.dumps(obj, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

CURRENT_SERVER_PID=""
CURRENT_SERVER_PGID=""
SERVER_STOP_TIMEOUT_SECS="${SERVER_STOP_TIMEOUT_SECS:-120}"

cleanup_current_server() {
  local target_pid="$CURRENT_SERVER_PID"
  local target_pgid="$CURRENT_SERVER_PGID"
  if [[ -z "$target_pgid" ]] && [[ -n "$target_pid" ]] && ps -p "$target_pid" >/dev/null 2>&1; then
    target_pgid="$(ps -o pgid= "$target_pid" 2>/dev/null | tr -d '[:space:]')"
  fi

  local waited=0
  if [[ -n "$target_pid" ]] && ps -p "$target_pid" >/dev/null 2>&1; then
    kill "$target_pid" >/dev/null 2>&1 || true
    while ps -p "$target_pid" >/dev/null 2>&1; do
      if (( waited >= SERVER_STOP_TIMEOUT_SECS )); then
        break
      fi
      sleep 1
      waited=$((waited + 1))
    done
  fi

  if [[ -n "$target_pgid" ]] && ps -o pid= -g "$target_pgid" >/dev/null 2>&1; then
    kill -"${target_pgid}" >/dev/null 2>&1 || true
    waited=0
    while ps -o pid= -g "$target_pgid" >/dev/null 2>&1; do
      if (( waited >= SERVER_STOP_TIMEOUT_SECS )); then
        echo "WARN: server pgid $target_pgid did not exit within ${SERVER_STOP_TIMEOUT_SECS}s; sending SIGKILL."
        kill -9 -"${target_pgid}" >/dev/null 2>&1 || true
        break
      fi
      sleep 1
      waited=$((waited + 1))
    done
  fi

  if [[ -n "$target_pid" ]]; then
    wait "$target_pid" 2>/dev/null || true
  fi

  CURRENT_SERVER_PID=""
  CURRENT_SERVER_PGID=""
}
trap cleanup_current_server EXIT

run_case() {
  local case_name="$1"
  local enable_chunk="$2"
  local enable_eplb="$3"
  local port="$4"

  local result_dir="$RESULT_ROOT/$PROFILE_NAME/$case_name"
  local expert_record_dir="$result_dir/expert_records"
  local mapping_jsonl="$result_dir/expert_mapping.jsonl"
  local forward_steps_jsonl="$result_dir/forward_steps.jsonl"
  local record_meta_json="$result_dir/record_meta.json"
  local heatmap_dir="$result_dir/heatmap"
  local run_log="$result_dir/run.log"
  local server_log="$result_dir/server.log"
  local bench_log="$result_dir/bench.log"
  local bench_json="$result_dir/bench.json"
  local bench_jsonl="$result_dir/bench.jsonl"
  local record_control_log="$result_dir/record_control.log"
  local served_name="${SERVED_MODEL_NAME}_${case_name}"
  local max_num_batched_tokens="$NOCHUNK_MAX_NUM_BATCHED_TOKENS"

  if [[ "$enable_chunk" == "1" ]]; then
    max_num_batched_tokens="$CHUNK_MAX_NUM_BATCHED_TOKENS"
  fi

  mkdir -p "$result_dir" "$expert_record_dir"
  rm -f "$run_log" "$server_log" "$bench_log" "$bench_json" "$bench_jsonl" \
    "$record_control_log" \
    "$mapping_jsonl" "$forward_steps_jsonl" "$record_meta_json" \
    "$result_dir/expert_record_files.txt"
  rm -rf "$heatmap_dir"
  rm -f "$expert_record_dir"/*.pt 2>/dev/null || true

  {
    echo "=== [$PROFILE_NAME/$case_name] $(date -u '+%F %T UTC') ==="
    echo "model_dir=$MODEL_DIR"
    echo "cuda_devices=$CUDA_DEVICES num_visible_gpus=$NUM_VISIBLE_GPUS"
    echo "tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
    echo "chunked_prefill=$enable_chunk chunk_size=$CHUNKED_PREFILL_SIZE eplb=$enable_eplb"
    echo "max_num_partial_prefills=$MAX_NUM_PARTIAL_PREFILLS max_long_partial_prefills=$MAX_LONG_PARTIAL_PREFILLS enable_mixed_prefill_decode=$([[ \"$DISABLE_MIXED_PREFILL_DECODE\" == \"1\" ]] && echo 0 || echo 1)"
    echo "expert_recorder_mode=$EXPERT_RECORDER_MODE enforce_eager=$EXPERT_RECORDER_ENFORCE_EAGER"
    echo "gpu_memory_utilization_initial=$GPU_MEMORY_UTILIZATION"
    echo "gpu_memory_utilization_retry_values=$GPU_MEMORY_UTILIZATION_RETRY_VALUES"
    echo "max_model_len=$MAX_MODEL_LEN max_num_batched_tokens=$max_num_batched_tokens max_num_seqs=$MAX_NUM_SEQS"
    echo "benchmark_dataset=$BENCHMARK_DATASET dataset=$DATASET_NAME subset=${BENCHMARK_SUBSET:-<none>} num_prompts=$BENCHMARK_NUM_PROMPTS max_concurrency=$BENCHMARK_MAX_CONCURRENCY"
    echo "benchmark_source_path=$BENCHMARK_SOURCE_PATH"
    echo "dataset_path=$DATASET_PATH"
    echo "result_dir=$result_dir"
  } | tee -a "$run_log"

  local ready=0
  local startup_failure=0
  local active_gpu_memory_utilization=""
  local util_attempt
  local attempt_idx=0
  local -a serve_args

  IFS=',' read -r -a _gpu_util_candidates <<<"$GPU_MEMORY_UTILIZATION_RETRY_VALUES"
  for util_attempt in "${_gpu_util_candidates[@]}"; do
    util_attempt="$(awk -v v="$util_attempt" 'BEGIN { gsub(/^[[:space:]]+|[[:space:]]+$/, "", v); print v }')"
    if [[ -z "$util_attempt" ]]; then
      continue
    fi
    attempt_idx=$((attempt_idx + 1))
    active_gpu_memory_utilization="$util_attempt"

    serve_args=(
      -m vllm.entrypoints.cli.main
      serve
      "$MODEL_DIR"
      --served-model-name "$served_name"
      --host "$HOST"
      --port "$port"
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
      --enable-expert-parallel
      --gpu-memory-utilization "$active_gpu_memory_utilization"
      --max-model-len "$MAX_MODEL_LEN"
      --max-num-batched-tokens "$max_num_batched_tokens"
      --max-num-seqs "$MAX_NUM_SEQS"
      --trust-remote-code
      --expert-distribution-recorder-mode "$EXPERT_RECORDER_MODE"
      --expert-distribution-recorder-output "$expert_record_dir"
    )

    if [[ "$enable_chunk" == "1" ]]; then
      serve_args+=(
        --enable-chunked-prefill
        --max-num-partial-prefills "$MAX_NUM_PARTIAL_PREFILLS"
        --max-long-partial-prefills "$MAX_LONG_PARTIAL_PREFILLS"
        --long-prefill-token-threshold "$CHUNKED_PREFILL_SIZE"
      )
    else
      serve_args+=(--no-enable-chunked-prefill)
    fi

    if [[ "$DISABLE_MIXED_PREFILL_DECODE" == "1" ]]; then
      serve_args+=(--disable-mixed-prefill-decode)
    fi

    if [[ "$EXPERT_RECORDER_ENFORCE_EAGER" == "1" ]]; then
      serve_args+=(--enforce-eager)
    fi

    if [[ "$enable_eplb" == "1" ]]; then
      serve_args+=(
        --enable-eplb
        --eplb-config
        "{\"window_size\":${EPLB_WINDOW_SIZE},\"step_interval\":${EPLB_STEP_INTERVAL},\"log_balancedness\":$([[ "$EPLB_LOG_BALANCEDNESS" == "1" ]] && echo true || echo false),\"log_balancedness_interval\":${EPLB_LOG_BALANCEDNESS_INTERVAL},\"use_async\":$([[ "$EPLB_USE_ASYNC" == "1" ]] && echo true || echo false)}"
      )
    fi

    echo "[1/4] start server... attempt=$attempt_idx gpu_memory_utilization=$active_gpu_memory_utilization" | tee -a "$run_log"
    : >"$server_log"
    (
      export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
      exec setsid "$PYTHON_BIN" "${serve_args[@]}"
    ) >"$server_log" 2>&1 &
    CURRENT_SERVER_PID=$!
    CURRENT_SERVER_PGID="$(ps -o pgid= "$CURRENT_SERVER_PID" 2>/dev/null | tr -d '[:space:]')"
    {
      echo "[$(date -u '+%F %T UTC')] server_started pid=$CURRENT_SERVER_PID pgid=$CURRENT_SERVER_PGID gpu_memory_utilization=$active_gpu_memory_utilization attempt=$attempt_idx"
    } >>"$record_control_log"

    ready=0
    startup_failure=0
    for _ in $(seq 1 240); do
      if curl -fsS --noproxy '*' --max-time 2 "http://$HOST:$port/health" >/dev/null 2>&1; then
        ready=1
        break
      fi
      if ! ps -p "$CURRENT_SERVER_PID" >/dev/null 2>&1; then
        startup_failure=1
        break
      fi
      sleep 2
    done

    if [[ "$ready" == "1" ]]; then
      break
    fi

    echo "WARN: [$case_name] server startup failed for gpu_memory_utilization=$active_gpu_memory_utilization" | tee -a "$run_log"
    tail -n 120 "$server_log" | tee -a "$run_log" || true
    cleanup_current_server

    if grep -q "Free memory on device .* is less than desired GPU memory utilization" "$server_log"; then
      echo "WARN: [$case_name] retrying with a lower gpu_memory_utilization..." | tee -a "$run_log"
      continue
    fi

    if [[ "$startup_failure" == "1" ]]; then
      echo "ERROR: [$case_name] server exited early." | tee -a "$run_log"
    else
      echo "ERROR: [$case_name] server not ready in time." | tee -a "$run_log"
    fi
    return 1
  done

  if [[ "$ready" != "1" ]]; then
    echo "ERROR: [$case_name] exhausted gpu_memory_utilization retries: $GPU_MEMORY_UTILIZATION_RETRY_VALUES" | tee -a "$run_log"
    tail -n 120 "$server_log" | tee -a "$run_log" || true
    cleanup_current_server
    return 1
  fi

  echo "[2/4] benchmark..." | tee -a "$run_log"
  {
    echo "[$(date -u '+%F %T UTC')] benchmark_begin"
  } >>"$record_control_log"
  local -a bench_args
  bench_args=(
    -m vllm.entrypoints.cli.main
    bench
    serve
    --backend openai
    --base-url "http://$HOST:$port"
    --model "$served_name"
    --tokenizer "$MODEL_DIR"
    --dataset-name "$DATASET_NAME"
    --dataset-path "$DATASET_PATH"
    --num-prompts "$BENCHMARK_NUM_PROMPTS"
    --request-rate "$REQUEST_RATE"
    --max-concurrency "$BENCHMARK_MAX_CONCURRENCY"
    --num-warmups "$NUM_WARMUPS"
    --custom-output-len "$BENCHMARK_OUTPUT_LEN"
    --result-dir "$result_dir"
    --result-filename "$(basename "$bench_json")"
    --save-result
    --save-detailed
    --disable-shuffle
    --skip-chat-template
    --request-id-prefix "${PROFILE_NAME}_${case_name}-"
    --temperature 0
    --trust-remote-code
  )

  local bench_status=0
  set +e
  env \
    -u HTTP_PROXY \
    -u HTTPS_PROXY \
    -u http_proxy \
    -u https_proxy \
    -u ALL_PROXY \
    -u all_proxy \
    NO_PROXY="$NO_PROXY" \
    no_proxy="$no_proxy" \
    "$PYTHON_BIN" "${bench_args[@]}" 2>&1 | tee "$bench_log"
  bench_status=${PIPESTATUS[0]}
  set -e
  {
    echo "[$(date -u '+%F %T UTC')] benchmark_end status=$bench_status"
  } >>"$record_control_log"
  if [[ -f "$bench_json" ]]; then
    emit_bench_jsonl_compat "$bench_json" "$bench_jsonl"
  fi

  echo "[3/4] shutdown server..." | tee -a "$run_log"
  {
    echo "[$(date -u '+%F %T UTC')] shutdown_begin"
  } >>"$record_control_log"
  cleanup_current_server
  {
    echo "[$(date -u '+%F %T UTC')] shutdown_end"
  } >>"$record_control_log"
  sleep 1

  echo "[3.1/4] collect dump manifest..." | tee -a "$run_log"
  find "$expert_record_dir" -maxdepth 1 -type f -name '*.pt' | sort \
    >"$result_dir/expert_record_files.txt" || true
  local record_file_count
  record_file_count="$(wc -l < "$result_dir/expert_record_files.txt" | tr -d '[:space:]')"
  if [[ "$record_file_count" == "0" ]]; then
    find "$result_dir" -maxdepth 2 -type f -name 'expert_distribution_recorder*.pt' | sort \
      >"$result_dir/expert_record_files.txt" || true
    record_file_count="$(wc -l < "$result_dir/expert_record_files.txt" | tr -d '[:space:]')"
  fi
  {
    echo "[$(date -u '+%F %T UTC')] dump_manifest_count=$record_file_count"
  } >>"$record_control_log"

  echo "[4/4] generate heatmaps..." | tee -a "$run_log"
  if [[ "$GENERATE_HEATMAPS" == "1" && ( "$EXPERT_RECORDER_MODE" == "stat" || "$EXPERT_RECORDER_MODE" == "stat_approx" ) ]]; then
    if [[ -s "$result_dir/expert_record_files.txt" ]]; then
      "$PYTHON_BIN" "$ROOT_DIR/scripts/plot_expert_distribution_heatmap.py" \
        --manifest "$result_dir/expert_record_files.txt" \
        --out-dir "$heatmap_dir" \
        --mapping-jsonl "$mapping_jsonl" \
        --forward-steps-jsonl "$forward_steps_jsonl" \
        --record-meta-json "$record_meta_json" \
        --layer "$HEATMAP_LAYER" \
        --skip-initial-steps "$HEATMAP_SKIP_INITIAL_STEPS" \
        --vmax-percentile "$HEATMAP_VMAX_PERCENTILE" \
        --case-name "$case_name" \
        --model-name "$MODEL_DIR" \
        --enable-eplb "$enable_eplb" \
        2>&1 | tee -a "$run_log"
      {
        echo "[$(date -u '+%F %T UTC')] heatmap_generated=1"
      } >>"$record_control_log"
    else
      echo "WARN: [$case_name] No expert record dump found; skip heatmap generation." | tee -a "$run_log"
      {
        echo "[$(date -u '+%F %T UTC')] heatmap_generated=0 reason=no_record_dump"
      } >>"$record_control_log"
    fi
  else
    echo "WARN: [$case_name] Heatmap generation skipped for EXPERT_RECORDER_MODE=$EXPERT_RECORDER_MODE." | tee -a "$run_log"
    {
      echo "[$(date -u '+%F %T UTC')] heatmap_generated=0 reason=unsupported_mode mode=$EXPERT_RECORDER_MODE"
    } >>"$record_control_log"
  fi

  append_summary \
    "$bench_json" \
    "$PROFILE_NAME" \
    "$case_name" \
    "$enable_chunk" \
    "$enable_eplb" \
    "$DATASET_NAME" \
    "$result_dir" \
    "$expert_record_dir"

  {
    echo "bench_json=$bench_json"
    echo "bench_jsonl=$bench_jsonl"
    echo "server_log=$server_log"
    echo "bench_log=$bench_log"
    echo "record_control_log=$record_control_log"
    echo "expert_record_dir=$expert_record_dir"
    echo "expert_record_manifest=$result_dir/expert_record_files.txt"
    echo "heatmap_dir=$result_dir/heatmap"
    echo "mapping_jsonl=$mapping_jsonl"
    echo "forward_steps_jsonl=$forward_steps_jsonl"
    echo "record_meta_json=$record_meta_json"
    echo "=== [$PROFILE_NAME/$case_name] done ==="
    echo
  } | tee -a "$run_log"

  if (( bench_status != 0 )); then
    return "$bench_status"
  fi
}

if [[ -z "$CUSTOM_DATASET_PATH" && ! -e "$BENCHMARK_SOURCE_PATH" ]]; then
  echo "ERROR: BENCHMARK_SOURCE_PATH does not exist: $BENCHMARK_SOURCE_PATH"
  exit 1
fi

prepare_custom_dataset

declare -a CASE_NAMES=(
  "nochunk_noeplb"
  "nochunk_eplb"
  "chunk_noeplb"
  "chunk_eplb"
)
declare -a CASE_CHUNK=("0" "0" "1" "1")
declare -a CASE_EPLB=("0" "1" "0" "1")

for idx in "${!CASE_NAMES[@]}"; do
  if [[ -n "$CASE_FILTER" ]]; then
    if [[ ",$CASE_FILTER," != *",${CASE_NAMES[$idx]},"* ]]; then
      continue
    fi
  fi
  run_case \
    "${CASE_NAMES[$idx]}" \
    "${CASE_CHUNK[$idx]}" \
    "${CASE_EPLB[$idx]}" \
    "$((BASE_PORT + idx))"
done

echo
echo "All results are under: $RESULT_ROOT/$PROFILE_NAME"
echo "Summary CSV: $SUMMARY_CSV"
