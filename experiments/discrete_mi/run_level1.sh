#!/usr/bin/env bash
set -euo pipefail
# Usage: bash experiments/discrete_mi/run_level1.sh Industrial --check|--smoke|--run
DOMAIN="${1:-}"
MODE="${2:-}"
if [[ "$DOMAIN" != Industrial && "$DOMAIN" != Musical ]]; then
 echo 'First argument must be Industrial or Musical' >&2; exit 2
fi
if [[ "$MODE" != --check && "$MODE" != --smoke && "$MODE" != --run ]]; then
 echo 'Usage: bash experiments/discrete_mi/run_level1.sh Industrial|Musical --check|--smoke|--run' >&2;exit 2
fi
cd "$HOME/ii_graph_v2_mod"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
DEVICE="${EXPERIMENT_DEVICE:-cuda}"
# If running CPU-only, use: CUDA_VISIBLE_DEVICES='' EXPERIMENT_DEVICE=cpu bash ...
BASE=(python -u experiments/discrete_mi/run_level1.py --domain "$DOMAIN" --seed 42 --k 256 --dim 64 --device "$DEVICE")
case "$MODE" in
 --check) "${BASE[@]}" --dry-run ;;
 --smoke) "${BASE[@]}" --steps 5 --batch 64 --eval-every 5 --lambda-grid '0.1' --nulls 2 --bootstrap 20 --eval-positives 200 --swaps-per-edge 0.01 --out-root 'outputs/discrete_mi_level1_v1_smoke' ;;
 --run) "${BASE[@]}" --steps 1200 --batch 512 --eval-every 200 --lambda-grid '0.01,0.1,1.0' --nulls 100 --bootstrap 500 --eval-positives 30000 --swaps-per-edge 2 --out-root 'outputs/discrete_mi_level1_v1' ;;
esac
