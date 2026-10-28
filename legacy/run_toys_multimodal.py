"""Toys/RecBole-style runner for a multimodal-aware Behavior I-I graph.

This script is the practical continuation of ``run_toys_behavior.py``.
It first builds the same Behavior I-I graph from ``*.train.inter`` and then:

  Behavior edge E_B (fixed)
      -> Text cosine on E_B
      -> Image cosine on E_B
      -> multimodal reliability M_ij
      -> W_ij = (1-lambda) B_ij + lambda M_ij

No Text/Image KNN graph is constructed in this runner. Therefore multimodal
information reweights the behavior graph without changing its topology.

Expected embedding NPZ keys:
    text_embeddings      [N, D_t]
    image_embeddings     [N, D_v]
Optional:
    item_ids             [N]       (if omitted, row index == item id)
    text_valid_mask      [N]
    image_valid_mask     [N]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse

from analysis.statistics import save_analysis
from embeddings.base import EmbeddingOutput
from evidence import EdgeEvidenceStore
from graph.multimodal_reweight import build_multimodal_behavior_graph
from graph.semantic import score_candidate_edges
from run_toys_behavior import build_behavior_graph, load_item_meta, load_reconstructed_train


def load_embedding_npz(path: Path, n_items: int) -> tuple[EmbeddingOutput, EmbeddingOutput]:
    data = np.load(path, allow_pickle=True)
    required = {"text_embeddings", "image_embeddings"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Embedding NPZ is missing keys: {sorted(missing)}")

    text_all = np.asarray(data["text_embeddings"], dtype=np.float32)
    image_all = np.asarray(data["image_embeddings"], dtype=np.float32)
    if text_all.ndim != 2 or image_all.ndim != 2:
        raise ValueError("text_embeddings and image_embeddings must be 2-D arrays")

    if "item_ids" in data.files:
        source_ids = [str(x) for x in data["item_ids"].tolist()]
        source_index = {item: i for i, item in enumerate(source_ids)}
        target_ids = [str(i) for i in range(n_items)]
        missing_ids = [item for item in target_ids if item not in source_index]
        if missing_ids:
            raise ValueError(
                f"Embedding NPZ misses {len(missing_ids)} item ids; "
                f"examples={missing_ids[:5]}"
            )
        rows = np.asarray([source_index[item] for item in target_ids], dtype=np.int64)
    else:
        if len(text_all) < n_items or len(image_all) < n_items:
            raise ValueError(
                f"Embedding rows are smaller than item universe: "
                f"text={len(text_all)}, image={len(image_all)}, items={n_items}"
            )
        rows = np.arange(n_items, dtype=np.int64)

    text = text_all[rows]
    image = image_all[rows]

    text_mask = np.ones(n_items, dtype=bool)
    image_mask = np.ones(n_items, dtype=bool)
    if "text_valid_mask" in data.files:
        text_mask = np.asarray(data["text_valid_mask"], dtype=bool)[rows]
    if "image_valid_mask" in data.files:
        image_mask = np.asarray(data["image_valid_mask"], dtype=bool)[rows]

    return EmbeddingOutput(text, text_mask), EmbeddingOutput(image, image_mask)


def sparse_behavior_to_networkx(G: sparse.csr_matrix, evidence: EdgeEvidenceStore) -> nx.Graph:
    g = nx.Graph()
    item_ids = [str(i) for i in range(G.shape[0])]
    g.add_nodes_from(item_ids)
    upper = sparse.triu(G, k=1).tocoo()
    for i, j, score in zip(upper.row, upper.col, upper.data):
        u, v = str(int(i)), str(int(j))
        score = float(score)
        g.add_edge(u, v, weight=score, behavior_score=score)
        evidence.update(u, v, cooccurrence_count=int(score), behavior_score=score)
    return g


def networkx_to_sparse(g: nx.Graph, n_items: int) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for u, v, attrs in g.edges(data=True):
        i, j = int(u), int(v)
        w = float(attrs.get("weight", 0.0))
        rows.extend((i, j))
        cols.extend((j, i))
        vals.extend((w, w))
    return sparse.csr_matrix(
        (np.asarray(vals, dtype=np.float32), (rows, cols)),
        shape=(n_items, n_items),
        dtype=np.float32,
    )


def _top_neighbors(g: nx.Graph, item: str, titles: dict[int, str], top_n: int) -> list[dict]:
    rows = []
    for neighbor, attrs in g[item].items():
        row = {
            "item_id": int(neighbor),
            "title": titles.get(int(neighbor), ""),
            "weight": float(attrs.get("weight", 0.0)),
        }
        for key in (
            "behavior_score",
            "normalized_behavior_score",
            "text_similarity",
            "image_similarity",
            "semantic_strength",
            "cross_modal_consistency",
            "multimodal_reliability",
            "final_score",
        ):
            value = attrs.get(key)
            if value is not None:
                row[key] = float(value)
        rows.append(row)
    rows.sort(key=lambda x: x["weight"], reverse=True)
    return rows[:top_n]


def make_case_study(
    behavior: nx.Graph,
    multimodal: nx.Graph,
    titles: dict[int, str],
    top_n: int = 10,
) -> dict:
    chosen = [
        node for node, degree in sorted(behavior.degree, key=lambda x: x[1], reverse=True)
        if degree > 0
    ][:3]
    return {
        item: {
            "title": titles.get(int(item), ""),
            "degree": int(behavior.degree[item]),
            "behavior_neighbors": _top_neighbors(behavior, item, titles, top_n),
            "multimodal_neighbors": _top_neighbors(multimodal, item, titles, top_n),
        }
        for item in chosen
    }


def export_top_reweighted_edges(
    g: nx.Graph,
    titles: dict[int, str],
    path: Path,
    limit: int = 500,
) -> None:
    edges = sorted(
        g.edges(data=True),
        key=lambda x: float(x[2].get("weight", 0.0)),
        reverse=True,
    )[:limit]
    fields = [
        "item_i", "title_i", "item_j", "title_j",
        "behavior_score", "normalized_behavior_score",
        "text_similarity", "image_similarity", "semantic_strength",
        "cross_modal_consistency", "multimodal_reliability", "final_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for u, v, a in edges:
            writer.writerow({
                "item_i": int(u),
                "title_i": titles.get(int(u), ""),
                "item_j": int(v),
                "title_j": titles.get(int(v), ""),
                **{k: a.get(k) for k in fields[4:]},
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--embedding-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/legacy/Toys_multimodal_behavior"))
    parser.add_argument("--min-cooc", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--symmetrize", choices=["union", "mutual"], default="mutual")
    parser.add_argument("--lambda-mm", type=float, default=0.3)
    parser.add_argument(
        "--reliability",
        choices=["strength_x_consistency", "mean", "min"],
        default="strength_x_consistency",
    )
    parser.add_argument("--cosine-transform", choices=["clip01", "shift01"], default="clip01")
    parser.add_argument(
        "--behavior-normalization",
        choices=["log_max", "max", "minmax", "none"],
        default="log_max",
    )
    parser.add_argument(
        "--missing-modality",
        choices=["behavior_only", "single", "zero"],
        default="behavior_only",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    n_items, titles = load_item_meta(args.items)
    print(f"[meta] items={n_items:,}")

    A, n_users = load_reconstructed_train(args.train, n_items)
    print(f"[train] users={n_users:,}")
    print(f"[train] reconstructed user-item interactions={A.nnz:,}")

    behavior_sparse = build_behavior_graph(
        A,
        min_cooc=args.min_cooc,
        top_k=args.top_k,
        symmetrize=args.symmetrize,
    )
    sparse.save_npz(args.output / "behavior_graph.npz", behavior_sparse)

    evidence = EdgeEvidenceStore()
    behavior = sparse_behavior_to_networkx(behavior_sparse, evidence)
    item_ids = [str(i) for i in range(n_items)]

    text_emb, image_emb = load_embedding_npz(args.embedding_npz, n_items)
    candidates = list(behavior.edges())
    print(f"[multimodal] scoring {len(candidates):,} fixed behavior edges only")
    score_candidate_edges(
        candidates, item_ids, text_emb.values, text_emb.valid_mask, "text", evidence
    )
    score_candidate_edges(
        candidates, item_ids, image_emb.values, image_emb.valid_mask, "image", evidence
    )

    multimodal = build_multimodal_behavior_graph(
        item_ids,
        behavior,
        evidence,
        lambda_multimodal=args.lambda_mm,
        reliability_mode=args.reliability,
        cosine_transform=args.cosine_transform,
        behavior_normalization=args.behavior_normalization,
        missing_modality=args.missing_modality,
    )

    if set(behavior.edges()) != set(multimodal.edges()):
        raise RuntimeError("Topology changed unexpectedly; multimodal graph must preserve E_B")

    mm_sparse = networkx_to_sparse(multimodal, n_items)
    sparse.save_npz(args.output / "multimodal_behavior_graph.npz", mm_sparse)

    evidence.export_csv(args.output / "edge_evidence.csv")
    save_analysis(
        {"collaborative": behavior, "multimodal_behavior": multimodal},
        args.output / "graph_analysis.json",
    )
    cases = make_case_study(behavior, multimodal, titles, top_n=10)
    (args.output / "case_study.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    export_top_reweighted_edges(multimodal, titles, args.output / "top_edges_preview.csv")

    config = {
        "min_cooc": args.min_cooc,
        "top_k": args.top_k,
        "symmetrize": args.symmetrize,
        "lambda_multimodal": args.lambda_mm,
        "reliability_mode": args.reliability,
        "cosine_transform": args.cosine_transform,
        "behavior_normalization": args.behavior_normalization,
        "missing_modality": args.missing_modality,
        "behavior_edges": behavior.number_of_edges(),
        "multimodal_edges": multimodal.number_of_edges(),
        "topology_preserved": True,
    }
    (args.output / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    available = sum(
        bool(rec.get("reliability_available")) for rec in evidence.records.values()
        if "final_score" in rec
    )
    print("\n===== MULTIMODAL REWEIGHT SUMMARY =====")
    print(f"behavior edges        : {behavior.number_of_edges():,}")
    print(f"multimodal edges      : {multimodal.number_of_edges():,}")
    print(f"reliability available : {available:,}/{behavior.number_of_edges():,}")
    print("topology preserved    : True")
    print(f"lambda                : {args.lambda_mm}")
    print(f"saved to              : {args.output}")


if __name__ == "__main__":
    main()
