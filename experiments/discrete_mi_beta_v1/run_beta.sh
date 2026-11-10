#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then PY="$CONDA_PREFIX/bin/python"; else PY="$(command -v python3)"; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-1}"
DOMAIN="${1:-}"
MODE="${2:---check}"
if [[ "$DOMAIN" != Industrial && "$DOMAIN" != Musical ]]; then echo 'Usage: bash experiments/discrete_mi_beta_v1/run_beta.sh Industrial|Musical --check|--smoke|--run [--modality text|image|both]' >&2;exit 2;fi
case "$MODE" in --check) M=check;; --smoke) M=smoke;; --run) M=run;; *) echo 'Expected --check, --smoke or --run' >&2;exit 2;; esac
shift 2 || true
DEVICE="${EXPERIMENT_DEVICE:-cuda}"
if [[ "$M" == check ]]; then DEVICE=cpu;fi
exec "$PY" -u "$HERE/run_beta.py" --domain "$DOMAIN" --mode "$M" --device "$DEVICE" "$@"
