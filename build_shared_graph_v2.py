#!/usr/bin/env python3
"""Dataset-independent, train-only I-I graph construction for Amazon Reviews 2023.

Version 2 defaults to a connectivity-oriented behavioral backbone:
min co-occurrence=1, union Top-K=50, retaining the same DA-MRS-inspired
multimodal reliability weight reweighting. Compare against stricter settings
before selecting the final graph; denser does not automatically mean better.

Reuses the repository's existing collaborative, semantic and multimodal-reweight
modules. Accepts any supported category with processed_v2/5core outputs and
precomputed CLIP embeddings aligned to its item_mapping.tsv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path

import networkx as nx
import numpy as np

from analysis.statistics import save_analysis
from config import CollaborativeGraphConfig
from data_loader import Interaction
from evidence import EdgeEvidenceStore
from graph.collaborative import build_collaborative_graph, compute_behavior_evidence
from graph.multimodal_reweight import build_multimodal_behavior_graph
from graph.semantic import score_candidate_edges

DATASETS = ("Industrial_and_Scientific", "Musical_Instruments", "Office_Products")


def read_mapping(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty item mapping: {path}")
    for idx, row in enumerate(rows):
        if int(row["item_id"]) != idx:
            raise ValueError(f"Mapping row {idx} is not a contiguous zero-based item ID")
    ids = [row["parent_asin"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ASIN in item mapping")
    return ids


def read_train(path: Path, dataset: str, universe: set[str]) -> list[Interaction]:
    records: list[Interaction] = []
    seen_pairs: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            uid, iid = str(row["user_id"]), str(row["parent_asin"])
            if iid not in universe:
                raise ValueError(f"Train item not in mapping: {iid}, line {line_number}")
            pair = (uid, iid)
            if pair in seen_pairs:
                raise ValueError(f"Duplicate train user-item: {pair}")
            seen_pairs.add(pair)
            records.append(Interaction(uid, iid, int(row["timestamp"]), dataset))
    if not records:
        raise ValueError("Train split is empty")
    return records


def verify_dataset_lock(processed: Path, manifest: dict, mapping: Path, train: Path) -> bool:
    lock_path = processed / "dataset_lock.json"
    if not lock_path.exists():
        print("WARNING: dataset_lock.json missing; skipping checksum verification", flush=True)
        return False
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = lock.get("file_sha256", {})
    for source in (mapping, train, processed / "manifest.json"):
        checksum = expected.get(source.name)
        if not checksum:
            raise ValueError(f"Dataset lock missing SHA256: {source.name}")
        h = hashlib.sha256()
        with source.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != checksum:
            raise ValueError(f"Dataset lock SHA256 mismatch: {source}")
    if lock.get("dataset_statistics") != manifest.get("statistics"):
        raise ValueError("Dataset lock / manifest statistics differ")
    return True


def read_aligned_embeddings(path: Path, expected_ids: list[str]):
    with np.load(path, allow_pickle=False) as d:
        ids = [str(x) for x in d["item_ids"].tolist()]
        if ids != expected_ids:
            raise ValueError("CLIP item_ids order differs from item_mapping.tsv")
        text = np.asarray(d["text_embeddings"], dtype=np.float32)
        image = np.asarray(d["image_embeddings"], dtype=np.float32)
        tm = np.asarray(d["text_valid_mask"], dtype=bool)
        im = np.asarray(d["image_valid_mask"], dtype=bool)
    n = len(expected_ids)
    if text.ndim != 2 or image.ndim != 2 or text.shape != image.shape or text.shape[0] != n:
        raise ValueError(f"Embedding shape mismatch: text={text.shape}, image={image.shape}, items={n}")
    if tm.shape != (n,) or im.shape != (n,):
        raise ValueError("Embedding valid masks have wrong shape")
    if not np.isfinite(text).all() or not np.isfinite(image).all():
        raise ValueError("NaN or Inf in CLIP embeddings")
    return text, image, tm, im


def check_graphs(behavior: nx.Graph, shared: nx.Graph, ids: list[str]):
    if set(behavior.nodes) != set(ids) or set(shared.nodes) != set(ids):
        raise ValueError("Graph item universe differs from mapping")
    if {frozenset(e) for e in behavior.edges()} != {frozenset(e) for e in shared.edges()}:
        raise ValueError("Multimodal reweighting unexpectedly changed graph topology")
    for _, _, data in shared.edges(data=True):
        weight = float(data["weight"])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid multimodal edge weight: {weight}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--processed-root", type=Path, default=Path.home() / "datasets/amazon2023/processed_v2/5core")
    parser.add_argument("--embedding-root", type=Path, default=Path("artifacts/embeddings"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--weighting", choices=("count", "degree_aware", "pmi"), default="count")
    parser.add_argument("--min-cooccurrence", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--symmetrize", choices=("mutual", "union"), default="union")
    parser.add_argument("--lambda-multimodal", type=float, default=0.3)
    parser.add_argument("--run-tag", type=str, default=None, help="Optional suffix for a separate output directory")
    args = parser.parse_args()

    if args.min_cooccurrence < 1 or args.top_k < 1 or not 0 <= args.lambda_multimodal <= 1:
        parser.error("min-cooccurrence and top-k must be positive; lambda must be in [0,1]")

    processed = args.processed_root / args.dataset
    embedding_path = args.embedding_root / args.dataset / "5core_v2" / "clip_vit_b32_aligned.npz"
    tag = args.run_tag or f"{args.weighting}_co{args.min_cooccurrence}_k{args.top_k}_{args.symmetrize}_lam{args.lambda_multimodal:g}"
    if tag in (".", "..") or not all(c.isalnum() or c in "-_." for c in tag):
        parser.error("run-tag must contain only letters, digits, underscores, hyphens or dots")
    output = args.output_root / args.dataset / f"5core_v2_trainonly_{tag}"
    for path in (processed / "item_mapping.tsv", processed / "train.jsonl", processed / "manifest.json", embedding_path):
        if not path.is_file():
            parser.error(f"Missing required file: {path}")

    mapping = processed / "item_mapping.tsv"
    train_path = processed / "train.jsonl"
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("variant") != "5core":
        raise ValueError("Expected the final 5core dataset variant")
    locked = verify_dataset_lock(processed, manifest, mapping, train_path)
    ids = read_mapping(mapping)
    train = read_train(train_path, args.dataset, set(ids))
    if len(ids) != manifest["statistics"]["items"] or len(train) != manifest["output_counts"]["train"]:
        raise ValueError("Item or train counts differ from manifest")
    text, image, text_mask, image_mask = read_aligned_embeddings(embedding_path, ids)
    print(f"[{args.dataset}] items={len(ids):,}, train={len(train):,}, text valid={int(text_mask.sum()):,}, image valid={int(image_mask.sum()):,}; lock_verified={locked}", flush=True)

    config = CollaborativeGraphConfig(weighting=args.weighting, min_cooccurrence=args.min_cooccurrence,
                                      top_k=args.top_k, symmetrize=args.symmetrize)
    evidence = EdgeEvidenceStore()
    print("Computing train-only co-occurrence...", flush=True)
    scores = compute_behavior_evidence(train, config, evidence)
    behavior = build_collaborative_graph(ids, scores, config)
    print(f"Behavior edges: {behavior.number_of_edges():,}", flush=True)

    edges = list(behavior.edges())
    score_candidate_edges(edges, ids, text, text_mask, "text", evidence)
    score_candidate_edges(edges, ids, image, image_mask, "image", evidence)
    shared = build_multimodal_behavior_graph(
        ids, behavior, evidence, lambda_multimodal=args.lambda_multimodal,
        reliability_mode="strength_x_consistency", cosine_transform="clip01",
        behavior_normalization="log_max", missing_modality="behavior_only",
    )
    check_graphs(behavior, shared, ids)
    # Reliability weight changes must be auditable; never infer graph quality from edge count alone.
    changed_weights = sum(
        abs(float(behavior[u][v]["weight"]) - float(shared[u][v]["weight"])) > 1e-8
        for u, v in behavior.edges()
    )
    print(f"Topology preserved; reweighted edges={changed_weights:,}/{behavior.number_of_edges():,}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    save_analysis({"collaborative": behavior, "multimodal_behavior": shared}, output / "graph_analysis.json")
    for name, graph in (("collaborative", behavior), ("multimodal_behavior", shared)):
        with (output / f"{name}.gpickle").open("wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "dataset": args.dataset, "dataset_variant": "5core_v2", "train_only": True,
        "processed_dir": str(processed), "embedding_path": str(embedding_path),
        "dataset_lock_verified": locked, "items": len(ids), "train_interactions": len(train),
        "text_valid": int(text_mask.sum()), "image_valid": int(image_mask.sum()),
        "behavior_edges": behavior.number_of_edges(), "shared_edges": shared.number_of_edges(),
        "reweighted_edges_vs_raw_behavior": changed_weights,
        "weighting": args.weighting, "min_cooccurrence": args.min_cooccurrence,
        "top_k": args.top_k, "symmetrize": args.symmetrize,
        "lambda_multimodal": args.lambda_multimodal, "reliability_mode": "strength_x_consistency",
        "cosine_transform": "clip01", "behavior_normalization": "log_max",
        "missing_modality": "behavior_only", "topology_equal": True,
    }
    (output / "run_config.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"GRAPH SUCCESS: {output}", flush=True)


if __name__ == "__main__":
    main()
