#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/ii_graph_v2_mod"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate graphrec
fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
for SEED in 43 44 45 46; do
  INPUT="outputs/motivation_routing_e/industrial_split${SEED}"
  OUTPUT="outputs/motivation_routing_reliability/industrial_split${SEED}"
  echo "===== SEED ${SEED} ====="
  if [ -f "$OUTPUT/routing_reliability_comparison.csv" ]; then
    echo "SKIP: existing comparison $OUTPUT"
    continue
  fi
  if [ ! -f "$INPUT/expressibility_scores.npz" ] || [ ! -f "$INPUT/routing_candidates.npz" ]; then
    python -u experiments/motivation/run_routing_e_ablation.py \
      --stage all \
      --dataset Industrial_and_Scientific \
      --mapping shared_data/Industrial_and_Scientific/5core_v2/item_mapping.tsv \
      --sequences "$HOME/datasets/baseline_sequences/Industrial_5core_v2.txt" \
      --embeddings artifacts/embeddings/Industrial_and_Scientific/full/clip_vit_b32.npz \
      --out "$INPUT" \
      --seed "$SEED" --u1-fraction 0.5 --u2-fraction 0.25 \
      --parent-k 400 --behavior-k 50 --min-parent-edges 50 \
      --folds 3 --q 0.1 0.2 0.3
  else
    echo "Reuse cached prepare/fit: $INPUT"
    if [ ! -f "$INPUT/threshold_tuning.csv" ]; then
      python -u experiments/motivation/run_routing_e_ablation.py \
        --stage evaluate --out "$INPUT" --seed "$SEED" --q 0.1 0.2 0.3
    fi
  fi
  python -u experiments/motivation/run_routing_reliability_compare.py \
    --input "$INPUT" --out "$OUTPUT" --q 0.1 0.2 0.3 \
    --behavior-bins 4 --include-prior-threshold
done
python experiments/motivation/summarize_routing_multiseed.py
