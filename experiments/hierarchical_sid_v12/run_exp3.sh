#!/usr/bin/env bash
# Portable Experiment 3 runner. Run from any directory. No hardcoded usernames / GPUs.
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$DIR/../.." && pwd)"
cd "$ROOT"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PY="$CONDA_PREFIX/bin/python"
else
  PY="${PYTHON_BIN:-python}"
fi
exec "$PY" -u "$DIR/run_exp3_v12.py" "$@"
