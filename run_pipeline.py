"""End-to-end Graph-guided Hierarchical Multimodal Tokenization pipeline.

  S1 relation graph -> S2 pseudo-labels -> S3 C/M gates -> S4 tokenizer
  -> S5 SIDs -> S6 T5 generative recommender

Examples:
  python run_pipeline.py --config configs/beauty.json --output outputs/beauty_main
  # ablation reusing fixed stages 1-3 of the main run:
  python run_pipeline.py --config configs/beauty.json --output outputs/beauty_noM \\
      --upstream outputs/beauty_main --stages 4-6 --set 'guidance.components=["C","R","N"]'
  # smoke run on a synthetic dataset:
  python run_pipeline.py --config configs/smoke.json --output /tmp/gsid_smoke --synthetic /tmp/gsid_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch

from gsid import stage1_graph, stage2_pseudo, stage3_gates, stage4_tokenizer, stage5_sid, stage6_recommender
from gsid.config import load_config, to_dict
from gsid.data import load_dataset, make_synthetic_dataset

STAGES = {1: "s1_graph", 2: "s2_pseudo", 3: "s3_gates", 4: "s4_tokenizer", 5: "s5_sid", 6: "s6_rec"}


def parse_stages(text):
    low, _, high = text.partition("-")
    stages = list(range(int(low), int(high or low)+1))
    if not stages or stages[0] < 1 or stages[-1] > 6:
        raise argparse.ArgumentTypeError("stages must be within 1-6, e.g. 4-6")
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", type=parse_stages, default=list(range(1, 7)))
    parser.add_argument("--upstream", type=Path, help="Read artifacts of stages not in this output from here")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="dotted.key=json_value")
    parser.add_argument("--force", action="store_true", help="Rerun stages that already finished")
    parser.add_argument("--synthetic", type=Path, help="Generate a synthetic dataset here and use it")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.synthetic:
        for key, value in make_synthetic_dataset(args.synthetic).items():
            setattr(cfg.data, key, value)
    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(to_dict(cfg), indent=2))
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    def stage_dir(stage, for_reading=False):
        own = args.output / STAGES[stage]
        if for_reading and not (own / "done.json").exists() and args.upstream:
            other = args.upstream / STAGES[stage]
            if (other / "done.json").exists():
                return other
        if for_reading and not (own / "done.json").exists():
            raise FileNotFoundError(f"Stage {stage} output missing: run it or pass --upstream")
        return own

    print(f"[data] loading {cfg.data.dataset} from {cfg.data.data_dir}", flush=True)
    data = load_dataset(cfg.data)
    print(f"[data] items={data.n_items} train={len(data.train)} valid={len(data.valid)} test={len(data.test)} "
          f"cold_items={int((~data.train_items).sum())}", flush=True)
    for stage in args.stages:
        out = stage_dir(stage)
        if (out / "done.json").exists() and not args.force:
            print(f"[stage{stage}] already done; skipping (use --force to rerun)", flush=True)
            continue
        started = time.time()
        if stage == 1:
            stats = stage1_graph.run(data, cfg, out, device)
        else:
            graph = sparse.load_npz(stage_dir(1, True) / "graph.npz").tocsr()
        if stage == 2:
            stats = stage2_pseudo.run(graph, cfg.pseudo, cfg.seed, out, device)
        if stage in (3, 4):
            pseudo = dict(np.load(stage_dir(2, True) / "pseudo_labels.npz"))
        if stage == 3:
            stats = stage3_gates.run(data, graph, pseudo, cfg.gates, out, device)
        if stage == 4:
            gates = dict(np.load(stage_dir(3, True) / "gates.npz"))
            stats = stage4_tokenizer.run(data, graph, pseudo, gates, cfg, out, device)
        if stage == 5:
            stats = stage5_sid.run(data, graph, cfg, stage_dir(4, True), out, device)
        if stage == 6:
            stats = stage6_recommender.run(data, cfg, stage_dir(5, True), out, device)
        (out / "done.json").write_text(json.dumps(dict(seconds=time.time()-started, stats=stats), indent=2))
    print(f"[done] {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
