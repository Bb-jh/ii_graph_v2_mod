# Graph-guided Hierarchical Multimodal Tokenization

Turns each item into Text / Graph / Image hierarchical SIDs (RQ-VAE) guided by I-I graph
pseudo-labels with adaptive weighting, then trains a T5 generative recommender on those SIDs.

## Layout

```
run_pipeline.py          entry point (Stages 1-6)
configs/                 beauty.json (main), smoke.json (tiny synthetic run)
gsid/                    pipeline stages
  config.py              all hyperparameters and ablation flags
  data.py                MACRec-format loader + synthetic dataset
  stage1_graph.py        I-I relation graph W (UltraGCN B x multimodal reliability, Top-K)
  stage2_pseudo.py       LightGCN encoder + recursive k-means -> Global/Local/Fine labels
  stage3_gates.py        Confidence C and Compatibility M (precomputed)
  stage4_tokenizer.py    T / V / G RQ-VAE + adaptive hierarchy guidance
  stage5_sid.py          SIDs [T3 | G3 | V3 | dedup], vocabulary
  stage6_recommender.py  T5, trie-constrained beam search, HR/NDCG (all/warm/cold)
hierarchy/               shared guidance modules used by the pipeline
  adaptive_distillation.py   level-conditional contrastive, ST-softmax, EMA, C*M*R*N weights
  confidence.py              C = cbrt(A*T*S)
  sampling.py                cluster-aware batches, EdgeDrop
tests/                   unit tests + end-to-end smoke test
scripts/data_prep/       Amazon 2023 preprocessing, CLIP embedding generation/alignment, split audit
scripts/diagnostics/     static C*M gate diagnostics (no training)
legacy/                  earlier graph-exploration code (kept runnable, not used by the pipeline)
outputs/legacy/          results of the earlier experiments
```

## Run

```bash
# main run; data/Beauty must hold Beauty.{train,valid,test}.inter, Beauty.item.json and embedding .npy files
python run_pipeline.py --config configs/beauty.json --output outputs/beauty_main

# ablation: reuse fixed Stages 1-3 and rerun 4-6
python run_pipeline.py --config configs/beauty.json --output outputs/beauty_noM \
  --upstream outputs/beauty_main --stages 4-6 --set 'guidance.components=["C","R","N"]'

# synthetic smoke run
python run_pipeline.py --config configs/smoke.json --output /tmp/smoke --synthetic /tmp/sdata
```

Ablation flags (`--set`): `guidance.mode="none"|"fixed"`, `guidance.components=[...]`,
`guidance.fixed_levels=[2,3]`, `sid.use_graph=false`, `tokenizer.edge_drop=false`.

## Tests

```bash
python -m unittest discover -s tests -t .                 # pipeline
cd legacy && python -m unittest discover -s tests -t .    # legacy code
```

Diagnostics and data-prep scripts that import repository packages are run as modules from the root,
e.g. `python -m scripts.diagnostics.run_adaptive_diagnostics --demo --output DIR`.
Legacy scripts are run as `python legacy/<script>.py` from the root; their defaults read/write `outputs/legacy/`.
