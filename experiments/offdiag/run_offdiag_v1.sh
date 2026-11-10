#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/ii_graph_v2_mod"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONHASHSEED=0
if [[ "${1:-}" == '--check' ]]; then
  python experiments/offdiag/run_offdiag_v1.py --domain Industrial --dry-run
  python experiments/offdiag/run_offdiag_v1.py --domain Musical --dry-run
  exit 0
fi
if [[ "${1:-}" == '--smoke' ]]; then
 for domain in Industrial Musical; do
  python experiments/offdiag/run_offdiag_v1.py --domain "$domain" --seed 42 --k 256 --nulls 10 --bootstraps 30 --max-positive 500 --swaps-per-edge 0.1 --out-root outputs/offdiag_analysis_v1_smoke
 done
 exit 0
fi
if [[ "${1:-}" != '--run' ]]; then
 echo 'Usage: bash experiments/offdiag/run_offdiag_v1.sh --check|--smoke|--run' >&2
 exit 2
fi
for domain in Industrial Musical; do
 python experiments/offdiag/run_offdiag_v1.py --domain "$domain" --seed 42 --k 256 --nulls 100 --bootstraps 500 --max-positive 30000 --swaps-per-edge 2 --out-root outputs/offdiag_analysis_v1
done
python experiments/offdiag/aggregate_offdiag_v1.py
