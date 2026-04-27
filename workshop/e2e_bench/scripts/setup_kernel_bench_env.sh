#!/usr/bin/env bash
# Set up a local environment to run the kernel-level MoE balanceness test
# (`tests/kernels/moe/modular_kernel_tools/benchmark_eplb_multigpu.py`).
#
# Strategy:
#   - Use the user's local vLLM source checkout in --vllm-src.
#   - Create a dedicated uv venv at --venv-dir.
#   - Install vLLM as editable, but reuse a precompiled wheel (so we don't
#     need to rebuild ~1k C++/CUDA kernels). The Python source layer comes
#     from --vllm-src so any test additions there are picked up.
#   - Drop the new benchmark file at
#     <vllm-src>/tests/kernels/moe/modular_kernel_tools/benchmark_eplb_multigpu.py
#     (extracted from workshop/e2e_bench/e2e_artifact.diff).
#
# Usage:
#   bash scripts/setup_kernel_bench_env.sh \
#     [--venv-dir /home/yyx/personal/inference/vllm-bench/.venv] \
#     [--vllm-src /home/yyx/personal/inference/vllm] \
#     [--diff workshop/e2e_bench/e2e_artifact.diff]

set -euo pipefail

VENV_DIR="${VENV_DIR:-/home/yyx/personal/inference/vllm-bench/.venv}"
VLLM_SRC="${VLLM_SRC:-/home/yyx/personal/inference/vllm}"
ARTIFACT_DIFF="${ARTIFACT_DIFF:-workshop/e2e_bench/e2e_artifact.diff}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir) VENV_DIR="$2"; shift 2 ;;
    --vllm-src) VLLM_SRC="$2"; shift 2 ;;
    --diff)     ARTIFACT_DIFF="$2"; shift 2 ;;
    --python)   PYTHON_VERSION="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

echo "[setup] VENV_DIR=$VENV_DIR"
echo "[setup] VLLM_SRC=$VLLM_SRC"
echo "[setup] ARTIFACT_DIFF=$ARTIFACT_DIFF"

if [[ ! -d "$VLLM_SRC" ]]; then
  echo "[setup] ERROR: vLLM source not found at $VLLM_SRC" >&2
  exit 1
fi
if [[ ! -f "$ARTIFACT_DIFF" ]]; then
  echo "[setup] ERROR: artifact diff not found at $ARTIFACT_DIFF" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] ERROR: uv is required (https://docs.astral.sh/uv/)." >&2
  exit 1
fi

mkdir -p "$(dirname "$VENV_DIR")"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[setup] creating venv at $VENV_DIR (python $PYTHON_VERSION)"
  uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
else
  echo "[setup] reusing existing venv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[setup] installing pip + setuptools + wheel"
uv pip install --upgrade pip setuptools wheel

echo "[setup] installing vLLM editable from $VLLM_SRC (precompiled wheels)"
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT:-nightly}"
uv pip install --editable "$VLLM_SRC" --torch-backend=auto --index-strategy unsafe-best-match

# accelerate is required by AutoModelForCausalLM(device_map="auto"), which
# capture_routing_trace.py uses in --mode hf. The vLLM editable install
# above does not pull it in by default.
echo "[setup] installing accelerate (needed by capture_routing_trace.py --mode hf)"
uv pip install "accelerate>=1.0"

echo "[setup] sanity-check torch and vllm"
python - <<'PY'
import torch, vllm
print(f"torch       = {torch.__version__} (cuda={torch.version.cuda}, devices={torch.cuda.device_count()})")
print(f"vllm        = {vllm.__version__}")
print(f"vllm path   = {vllm.__file__}")
import vllm._C  # ensure C++ kernels load
print(f"vllm._C    = {vllm._C.__file__}")
PY

DEST="$VLLM_SRC/tests/kernels/moe/modular_kernel_tools/benchmark_eplb_multigpu.py"
if [[ -f "$DEST" ]]; then
  echo "[setup] benchmark file already present at $DEST (skipping extraction)"
else
  echo "[setup] extracting benchmark_eplb_multigpu.py from $ARTIFACT_DIFF -> $DEST"
  python - <<PY
import os
diff_path = os.path.abspath("$ARTIFACT_DIFF")
dest = os.path.abspath("$DEST")
header = "diff --git a/tests/kernels/moe/modular_kernel_tools/benchmark_eplb_multigpu.py"
with open(diff_path, encoding="utf-8") as f:
    lines = f.readlines()
start = next(i for i, l in enumerate(lines) if l.startswith(header))
end = next(
    (j for j in range(start + 1, len(lines)) if lines[j].startswith("diff --git ")),
    len(lines),
)
out = []
for k in range(start, end):
    line = lines[k]
    if line.startswith("+") and not line.startswith("+++"):
        out.append(line[1:])
os.makedirs(os.path.dirname(dest), exist_ok=True)
with open(dest, "w", encoding="utf-8") as f:
    f.writelines(out)
print(f"wrote {len(out)} lines to {dest}")
PY
fi

echo "[setup] verifying benchmark module imports"
(
  cd "$VLLM_SRC"
  python - <<'PY'
import sys, importlib
sys.path.insert(0, ".")
m = importlib.import_module(
    "tests.kernels.moe.modular_kernel_tools.benchmark_eplb_multigpu"
)
print("benchmark module ok:", m.__file__)
PY
)

echo
echo "[setup] DONE"
echo "Activate the venv with:"
echo "    source $VENV_DIR/bin/activate"
echo
echo "Run a small smoke test from inside $VLLM_SRC:"
cat <<EOSMOKE
    cd $VLLM_SRC
    python -m tests.kernels.moe.modular_kernel_tools.benchmark_eplb_multigpu \\
        --world-size 8 \\
        --routing-space physical_gpu \\
        --backend allgather_reducescatter \\
        --num-experts 64 --topk 6 \\
        -k 2048 -n 1408 \\
        --total-tokens 8192 \\
        --target-dest-cv 0.10 \\
        --num-random-trials 1 --sample-attempts 200 \\
        --warmup-iters 3 --iters 5 \\
        --mode off
EOSMOKE
