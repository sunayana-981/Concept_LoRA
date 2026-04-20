#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: ./run_dams_all.sh [extra run_sae_inference.py args]"
  echo
  echo "Runs: python run_sae_inference.py --all --dams_only --subset_seed 0 --dams_css_saturation 0.5"
  echo "Logs: out/logs/dams_all_<timestamp>.log"
  echo
  echo "Examples:"
  echo "  ./run_dams_all.sh"
  echo "  ./run_dams_all.sh --max_images 2000"
  echo "  ./run_dams_all.sh --device cpu --subset_seed 42"
  exit 0
fi

PATCHSAE_PY="/home/sunayana/miniconda3/envs/patchsae/bin/python"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if ! "$PYTHON_BIN" -c "import numpy" >/dev/null 2>&1; then
  if [[ -x "$PATCHSAE_PY" ]] && "$PATCHSAE_PY" -c "import numpy" >/dev/null 2>&1; then
    PYTHON_BIN="$PATCHSAE_PY"
  fi
fi

if ! "$PYTHON_BIN" -c "import numpy" >/dev/null 2>&1; then
  echo "[ERROR] Could not find a Python interpreter with numpy installed."
  echo "        Activate the patchsae env or set PYTHON_BIN explicitly."
  echo "        Example: PYTHON_BIN=/home/sunayana/miniconda3/envs/patchsae/bin/python ./run_dams_all.sh"
  exit 1
fi

LOG_DIR="$SCRIPT_DIR/out/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%F_%H-%M-%S)"
LOG_FILE="$LOG_DIR/dams_all_${TIMESTAMP}.log"

EXTRA_ARGS=("$@")
if [[ " ${EXTRA_ARGS[*]} " != *" --subset_seed "* ]]; then
  EXTRA_ARGS+=(--subset_seed 0)
fi
if [[ " ${EXTRA_ARGS[*]} " != *" --dams_css_saturation "* ]]; then
  EXTRA_ARGS+=(--dams_css_saturation 0.5)
fi

echo "============================================================"
echo "DAMS all-datasets run"
echo "Started: $(date)"
echo "Python: $PYTHON_BIN"
echo "Log: $LOG_FILE"
echo "============================================================"

set +e
"$PYTHON_BIN" run_sae_inference.py --all --dams_only "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

echo
echo "Finished: $(date)"
echo "Exit code: $STATUS"
echo "Log file: $LOG_FILE"

exit "$STATUS"
