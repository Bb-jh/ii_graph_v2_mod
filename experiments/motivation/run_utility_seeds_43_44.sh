#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/ii_graph_v2_mod"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
PY=experiments/motivation
OUTROOT=outputs/modality_utility_v2

if [[ "${1:-}" == "--check" ]]; then
  echo '[CHECK] Code and datasets'
  for f in "$PY/build_utility_candidates.py" "$PY/evaluate_modality_utility.py" "$PY/freeze_utility_predictions.py" \
    shared_data/Industrial_and_Scientific/5core_v2/item_mapping.tsv \
    shared_data/Musical_Instruments/5core_v2/item_mapping.tsv \
    "$HOME/datasets/baseline_sequences/Industrial_5core_v2.txt" \
    "$HOME/datasets/baseline_sequences/Musical_5core_v2.txt" \
    artifacts/embeddings/Industrial_and_Scientific/full/clip_vit_b32.npz \
    artifacts/embeddings/Musical_Instruments/full/clip_vit_b32.npz; do
    test -f "$f" || { echo "MISSING: $f"; exit 1; }
    echo "OK $f"
  done
  for name in Industrial Musical; do
    d="$OUTROOT/${name}_CLIP_42"
    for f in utility_metrics.csv permutation_utility.csv predicted_allocation.json; do
      test -s "$d/$f" || { echo "MISSING SEED 42: $d/$f"; exit 1; }
    done
  done
  python "$PY/freeze_utility_predictions.py" --check-only
  echo '[CHECK PASS]'
  exit 0
fi
if [[ "${1:-}" != "--run" && "${1:-}" != "--summarize" ]]; then
  echo "Usage: bash $0 --check | --run | --summarize" >&2
  exit 2
fi
if [[ "${1:-}" == "--summarize" ]]; then
  python "$PY/freeze_utility_predictions.py"
  exit 0
fi
bash "$0" --check
for dataset in Industrial Musical; do
  if [[ "$dataset" == Industrial ]]; then
    full=Industrial_and_Scientific
    seq="$HOME/datasets/baseline_sequences/Industrial_5core_v2.txt"
  else
    full=Musical_Instruments
    seq="$HOME/datasets/baseline_sequences/Musical_5core_v2.txt"
  fi
  mapping="shared_data/$full/5core_v2/item_mapping.tsv"
  embedding="artifacts/embeddings/$full/full/clip_vit_b32.npz"
  for seed in 43 44; do
    out="$OUTROOT/${dataset}_CLIP_${seed}"
    if [[ -s "$out/utility_metrics.csv" && -s "$out/permutation_utility.csv" && -s "$out/predicted_allocation.json" && -s "$out/manifest.json" ]]; then
      echo "[SKIP completed] $dataset seed=$seed"
      continue
    fi
    mkdir -p "$out"
    if [[ ! -s "$out/utility_candidates.npz" ]]; then
      echo "[BUILD] $dataset seed=$seed"
      python -u "$PY/build_utility_candidates.py" --mapping "$mapping" --sequences "$seq" \
        --embeddings "$embedding" --out "$out" --dataset "$full" --seed "$seed" --k 20
    else
      echo "[REUSE candidate] $dataset seed=$seed"
    fi
    echo "[EVALUATE] $dataset seed=$seed (100 permutations, 500 bootstraps)"
    python -u "$PY/evaluate_modality_utility.py" \
      --input "$out/utility_candidates.npz" --out "$out" --dataset "$full" \
      --encoder CLIP --seed "$seed" --folds 5 --fold-mode anchor \
      --C 1.0 --permutations 100 --bootstrap 500 --bins 4
    echo "[COMPLETE] $dataset seed=$seed"
  done
done
python "$PY/freeze_utility_predictions.py"
