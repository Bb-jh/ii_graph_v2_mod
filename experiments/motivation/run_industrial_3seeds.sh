#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/ii_graph_v2_mod"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=experiments/motivation
for SEED in 42 43 44; do
  IN="outputs/motivation_routing_e/industrial_split${SEED}"
  UNI="outputs/motivation_shared_unique/industrial_split${SEED}"
  OUT="outputs/motivation_ut_decision/industrial_split${SEED}"
  test -f "$IN/routing_candidates.npz" || { echo "Missing $IN; run earlier routing prepare+fit first" >&2; exit 1; }
  test -f "$IN/expressibility_scores.npz" || { echo "Missing $IN/expressibility_scores.npz" >&2; exit 1; }
  if [ ! -f "$UNI/shared_unique_scores.npz" ]; then
    echo "[$SEED] Build original signed U_T/U_V using U2 only"
    python -u "$PY/run_shared_unique_validity.py" --input "$IN" --out "$UNI" \
      --dataset Industrial_and_Scientific --seed "$SEED" --q 0.1 0.2 0.3 \
      --C 1.0 --min-positive 5 --behavior-bins 4 --random-repeats 3
  fi
  if [ ! -f "$OUT/routing_deltas_item_boot.csv" ]; then
    echo "[$SEED] Compare U vs exact U-strata quota-matched raw cosine; item bootstrap"
    python -u "$PY/run_ut_decision.py" --input "$IN" --u-scores "$UNI/shared_unique_scores.npz" \
      --out "$OUT" --dataset Industrial_and_Scientific --seed "$SEED" \
      --bins 4 --q 0.1 0.2 0.3 --bootstrap 500
  else
    echo "[$SEED] Existing result found; skipping"
  fi
done
python "$PY/summarize_ut_3seeds.py" --root outputs/motivation_ut_decision --seeds 42 43 44
