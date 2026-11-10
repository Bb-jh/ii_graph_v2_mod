#!/usr/bin/env python3
"""Motivation follow-up: B/T/V shared-edge coverage and train-degree-stratified overlap.

Reuse *exact* graph creation functions from run_motivation_cpu_v2.py. This is a
single-seed exploratory diagnostic, not a statistical significance test or a
reproduction of the model's similarity-gated Global edge definition.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from run_motivation_cpu_v2 import (
    behavior_graph,
    load_embeddings,
    load_mapping,
    load_train_only,
    matched_semantic_graph,
)

PAIRS = (("B", "T"), ("B", "V"), ("T", "V"))
DEGREE_BINS = ((1, 1, "1"), (2, 4, "2-4"), (5, 9, "5-9"),
               (10, 19, "10-19"), (20, 49, "20-49"),
               (50, np.inf, "50+"))


def edge_codes(graph: sparse.csr_matrix) -> np.ndarray:
    """Encode each undirected graph edge once as row*n+col (row<col)."""
    upper = sparse.triu(graph, k=1, format="coo")
    n = graph.shape[0]
    return np.unique(upper.row.astype(np.int64) * n + upper.col.astype(np.int64))


def incident_item_count(codes: np.ndarray, n: int) -> int:
    if not len(codes):
        return 0
    return int(np.unique(np.concatenate((codes // n, codes % n))).size)


def covered_item_mask(codes: np.ndarray, n: int) -> np.ndarray:
    covered = np.zeros(n, dtype=bool)
    covered[codes // n] = True
    covered[codes % n] = True
    return covered


def edge_report(graphs: dict[str, sparse.csr_matrix], train_degrees: np.ndarray):
    n = next(iter(graphs.values())).shape[0]
    edges = {v: edge_codes(g) for v, g in graphs.items()}
    eB, eT, eV = (edges[v] for v in ("B", "T", "V"))
    bt = np.intersect1d(eB, eT, assume_unique=True)
    bv = np.intersect1d(eB, eV, assume_unique=True)
    tv = np.intersect1d(eT, eV, assume_unique=True)
    triple = np.intersect1d(bt, eV, assume_unique=True)
    b_any = np.union1d(bt, bv)
    b_only = np.setdiff1d(eB, b_any, assume_unique=True)
    b_t_only = np.setdiff1d(bt, triple, assume_unique=True)
    b_v_only = np.setdiff1d(bv, triple, assume_unique=True)
    categories = {
        "B_only": b_only,
        "B_and_T_not_V": b_t_only,
        "B_and_V_not_T": b_v_only,
        "B_and_T_and_V": triple,
        "B_and_either_T_or_V": b_any,
        "T_and_V": tv,
    }
    all_b = covered_item_mask(eB, n)
    shared = covered_item_mask(triple, n)
    output = {
        "common_items": n,
        "edge_counts": {v: int(len(c)) for v, c in edges.items()},
        "pairwise_intersection_edges": {
            "B-T": int(len(bt)), "B-V": int(len(bv)), "T-V": int(len(tv))
        },
        "triple_intersection_edges": int(len(triple)),
        "triple_fraction_of_B_edges": float(len(triple) / len(eB)) if len(eB) else None,
        "triple_fraction_of_T_edges": float(len(triple) / len(eT)) if len(eT) else None,
        "triple_fraction_of_V_edges": float(len(triple) / len(eV)) if len(eV) else None,
        "triple_incident_items": int(shared.sum()),
        "triple_item_coverage_all_common": float(shared.mean()),
        "triple_item_coverage_B_connected": float(shared.sum() / all_b.sum()) if all_b.sum() else None,
        "B_incident_items": int(all_b.sum()),
        "B_edge_category_partition_check": bool(
            len(b_only) + len(b_t_only) + len(b_v_only) + len(triple) == len(eB)
        ),
    }
    category_rows = [
        {
            "category": name,
            "edges": int(len(codes)),
            "fraction_of_B_edges": float(len(codes) / len(eB)) if len(eB) else None,
            "incident_items": incident_item_count(codes, n),
            "item_coverage": float(incident_item_count(codes, n) / n),
        }
        for name, codes in categories.items()
    ]
    degree_rows = []
    for lo, hi, label in DEGREE_BINS:
        mask = (train_degrees >= lo) & (train_degrees <= hi)
        count = int(mask.sum())
        degree_rows.append({
            "degree_bin": label, "items": count,
            "share_of_common_items": count / n,
            "B_connected_items": int((mask & all_b).sum()),
            "triple_covered_items": int((mask & shared).sum()),
            "triple_item_coverage": float((mask & shared).sum() / count) if count else None,
        })
    return output, category_rows, degree_rows


def overlap_rows(neighbors: dict[str, np.ndarray], train_degrees: np.ndarray, ks: list[int]):
    """Directed Top-K overlap; eligible if both views have at least k neighbors."""
    n = len(train_degrees)
    out = []
    for k in ks:
        for a, b in PAIRS:
            na, nb = neighbors[a], neighbors[b]
            if na.shape[1] < k or nb.shape[1] < k:
                raise ValueError(f"k={k} exceeds available kNN prefix for {a}-{b}")
            eligible = (np.sum(na[:, :k] >= 0, axis=1) == k) & (
                np.sum(nb[:, :k] >= 0, axis=1) == k
            )
            # Both are generated as ordered, unique nearest-neighbor prefixes.
            # Count exact item-level overlap without expensive per-row Python sets.
            # k <= 50 by default; broadcast 25692*k*k booleans in 1K batches.
            matches = np.zeros(n, dtype=np.int16)
            for start in range(0, n, 1024):
                end = min(n, start + 1024)
                aa = na[start:end, :k]
                bb = nb[start:end, :k]
                matches[start:end] = (
                    (aa[:, :, None] == bb[:, None, :]) &
                    (aa[:, :, None] >= 0) & (bb[:, None, :] >= 0)
                ).sum(axis=(1, 2))
            for lo, hi, label in ((1, np.inf, "ALL"), *DEGREE_BINS):
                mask = eligible & (train_degrees >= lo) & (train_degrees <= hi)
                count = int(mask.sum())
                observed = float(matches[mask].mean() / k) if count else None
                # Independent uniformly random size-k sets of neighbors from N-1
                # candidates: E[ |A∩B| / k ] = k/(N-1), not a learned baseline.
                random_expected = float(k / (n - 1)) if n > 1 else None
                out.append({
                    "k": k, "pair": f"{a}-{b}", "degree_bin": label,
                    "eligible_items": count,
                    "eligible_share_in_bin": count / max(1, int(((train_degrees >= lo) & (train_degrees <= hi)).sum())),
                    "observed_overlap_at_k": observed,
                    "uniform_random_expected_at_k": random_expected,
                    "observed_over_random_ratio": (observed / random_expected)
                    if observed is not None and random_expected else None,
                })
    return out


def write_csv(path: Path, rows: list[dict]):
    assert rows, path
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--sequences", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--behavior-k", type=int, default=50)
    ap.add_argument("--symmetrize", choices=("union", "mutual"), default="union")
    ap.add_argument("--overlap-ks", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--density-tolerance", type=float, default=0.03)
    ap.add_argument("--min-train-degree", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    for p in (args.mapping, args.sequences, args.embeddings):
        if not p.is_file():
            ap.error(f"missing input: {p}")
    if args.behavior_k < max(args.overlap_ks) or min(args.overlap_ks) < 1:
        ap.error("overlap ks must be positive and <= behavior-k")
    if args.threads < 1:
        ap.error("threads must be >=1")
    # Do not overwrite earlier results.
    args.out.mkdir(parents=True, exist_ok=False)
    ids = load_mapping(args.mapping)
    x, degrees, n_events = load_train_only(args.sequences, len(ids))
    emb = load_embeddings(args.embeddings, ids)
    valid = (degrees >= args.min_train_degree) & emb["text"][1] & emb["image"][1]
    selected = np.flatnonzero(valid)
    n = len(selected)
    print(f"Common items: {n:,} / {len(ids):,}; train events: {n_events:,}", flush=True)
    if n <= 1:
        raise ValueError("not enough common items")
    bg, bn = behavior_graph(x, degrees, selected, args.behavior_k, args.symmetrize)
    base_edges = bg.nnz // 2
    print(f"B graph: {base_edges:,} edges", flush=True)
    tg, tn, tmeta = matched_semantic_graph(*emb["text"], selected, base_edges,
        max(args.overlap_ks), args.symmetrize, args.threads, args.density_tolerance)
    print(f"T graph: {tmeta}", flush=True)
    vg, vn, vmeta = matched_semantic_graph(*emb["image"], selected, base_edges,
        max(args.overlap_ks), args.symmetrize, args.threads, args.density_tolerance)
    print(f"V graph: {vmeta}", flush=True)
    graphs = {"B": bg, "T": tg, "V": vg}
    neighbors = {"B": bn, "T": tn, "V": vn}
    summary, cats, coverage = edge_report(graphs, degrees[selected])
    overlaps = overlap_rows(neighbors, degrees[selected], args.overlap_ks)
    summary.update({
        "dataset": args.dataset, "seed": args.seed,
        "total_items": len(ids), "train_events": n_events, "users": x.shape[0],
        "B_topk": args.behavior_k, "semantic_k": {"T": tmeta["directed_k"], "V": vmeta["directed_k"]},
        "symmetrize": args.symmetrize, "min_train_degree": args.min_train_degree,
        "notes": [
            "Exact intersections of independently constructed, density-matched undirected B/T/V kNN graphs; not cosine-threshold or weighted gating.",
            "B graph derives from train-only sequences (leave-last-two-out); degree bins count distinct training users per item.",
            "The uniform random neighbor baseline assumes independent uniform size-k sets and ignores popularity, graph topology, and modality biases.",
            "Overlaps use the directed top-k neighbor prefixes; triple edge intersections use symmetrized full graphs.",
            "Edge intersection fractions are descriptive, and a high/low coverage alone does not establish downstream improvement.",
        ],
    })
    (args.out / "shared_relation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(args.out / "shared_edge_categories.csv", cats)
    write_csv(args.out / "shared_coverage_by_degree.csv", coverage)
    write_csv(args.out / "neighbor_overlap_by_degree.csv", overlaps)
    print("SHARED:", {k: summary[k] for k in (
        "triple_intersection_edges", "triple_fraction_of_B_edges",
        "triple_incident_items", "triple_item_coverage_all_common")}, flush=True)
    print("SUCCESS:", args.out, flush=True)


if __name__ == "__main__":
    main()
