#!/usr/bin/env bash
# RCH pre-check P1/P2 runner. Place this folder at ii_graph_v2_mod/experiments/rch_probe/
# Usage:
#   bash experiments/rch_probe/run_rch_probe.sh self-test
#   bash experiments/rch_probe/run_rch_probe.sh Industrial --check|--smoke|--run [extra args]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then PY="$CONDA_PREFIX/bin/python"
else PY="$(command -v python3 || true)"; fi
[[ -z "$PY" ]] && { echo 'Python not found. Run: conda activate graphrec' >&2; exit 2; }
"$PY" -c 'import numpy,scipy,sklearn,joblib' || { echo 'Missing packages. Run: conda activate graphrec' >&2; exit 2; }

# One BLAS thread per worker; parallelism comes from joblib workers (RCH_JOBS).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
JOBS="${RCH_JOBS:-8}"

if [[ "${1:-}" == "self-test" ]]; then
  exec "$PY" -u "$SCRIPT_DIR/run_rch_probe.py" --mode self-test
fi

DOMAIN="${1:-}"; MODE="${2:-}"
if [[ "$DOMAIN" != Industrial && "$DOMAIN" != Musical ]]; then
  echo 'Usage: run_rch_probe.sh self-test | Industrial|Musical --check|--smoke|--run [extra args]' >&2; exit 2; fi
shift 2 || true
mkdir -p outputs/rch_precheck_v1/logs
LOG="outputs/rch_precheck_v1/logs/${DOMAIN}_${MODE#--}_$(date +%Y%m%d_%H%M%S).log"
case "$MODE" in
  --check) "$PY" -u "$SCRIPT_DIR/run_rch_probe.py" --domain "$DOMAIN" --mode check "$@" ;;
  --smoke) "$PY" -u "$SCRIPT_DIR/run_rch_probe.py" --domain "$DOMAIN" --mode smoke --jobs 2 "$@" 2>&1 | tee "$LOG" ;;
  --run)   "$PY" -u "$SCRIPT_DIR/run_rch_probe.py" --domain "$DOMAIN" --mode run --jobs "$JOBS" "$@" 2>&1 | tee "$LOG" ;;
  *) echo 'Expected --check, --smoke or --run' >&2; exit 2 ;;
esac
