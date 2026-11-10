#!/usr/bin/env bash
set -euo pipefail
DOMAIN="${1:-}"; MODE="${2:-}"
if [[ "$DOMAIN" != Industrial && "$DOMAIN" != Musical ]]; then echo 'Usage: bash /path/to/run_portable.sh Industrial|Musical --check|--smoke|--run' >&2; exit 2; fi
if [[ "$MODE" != --check && "$MODE" != --smoke && "$MODE" != --run ]]; then echo 'Expected --check, --smoke, --run' >&2; exit 2; fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-1}"
DEVICE="${EXPERIMENT_DEVICE:-cuda}"
# Prefer explicitly activated Conda env; never rely on .pyenv shims.
if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then PY="$CONDA_PREFIX/bin/python"
else PY="$(command -v python3 || true)"; fi
if [[ -z "$PY" ]]; then echo 'Python not found. Activate the appropriate environment.' >&2; exit 2; fi
"$PY" -c 'import torch,sklearn,scipy,numpy; print("ENV Python/Torch:",__import__("sys").executable,torch.__version__)' || { echo 'Python dependencies missing. Activate the environment containing torch, sklearn, scipy, numpy.' >&2; exit 2; }
BASE=("$PY" -u "$SCRIPT_DIR/run_level1_v2.py" --domain "$DOMAIN" --seed 42 --k 256 --dim 64 --soft-temperature 0.10 --balance-weight 0.01 --device "$DEVICE")
# Optional explicit overrides without editing any source. e.g. EXP2_SEQUENCE_PATH=/path/to/file
case "$MODE" in
 --check) "${BASE[@]}" --dry-run ;;
 --smoke) "${BASE[@]}" --steps 5 --batch 64 --eval-every 5 --lambda-grid '0.1' --nulls 2 --bootstrap 20 --eval-positives 200 --swaps-per-edge 0.01 --out-root 'outputs/discrete_mi_level1_v2_portable_smoke' ;;
 --run) "${BASE[@]}" --steps 1200 --batch 512 --eval-every 200 --lambda-grid '0.01,0.1,1.0' --nulls 100 --bootstrap 500 --eval-positives 30000 --swaps-per-edge 2 --out-root 'outputs/discrete_mi_level1_v2_portable' ;;
esac
