"""Adapters between saved SciPy I-I graphs and the supplied pseudo-label code.

The supplied LightGCN / recursive K-means implementations remain unchanged.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def dump_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_graph(path: Path, item_ids_path: Path | None = None):
    graph = sparse.load_npz(path).tocsr().astype(np.float32)
    graph.sum_duplicates()
    if graph.shape[0] != graph.shape[1] or graph.shape[0] == 0:
        raise ValueError("Expected a non-empty square sparse adjacency matrix")
    if not np.all(np.isfinite(graph.data)) or np.any(graph.data < 0):
        raise ValueError("Graph weights must be finite and non-negative")
    if np.any(graph.diagonal() != 0):
        raise ValueError("Self edges are unsupported; provide a loop-free I-I graph")
    graph.eliminate_zeros()
    difference = graph - graph.T
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-6:
        raise ValueError("Graph must be symmetric (undirected)")
    if graph.nnz == 0:
        raise ValueError("Graph has no positive edges")
    ids = np.arange(graph.shape[0], dtype=np.int64)
    if item_ids_path:
        ids = np.load(item_ids_path, allow_pickle=False)
        if ids.dtype.kind not in "iuUS":
            raise ValueError("item_ids must contain integers or strings, not object/pickle values")
        if ids.dtype.kind == "S":
            ids = ids.astype(str)
    if ids.ndim != 1 or len(ids) != graph.shape[0] or len(np.unique(ids)) != len(ids):
        raise ValueError("item_ids must uniquely identify every adjacency row")
    return graph, ids


def prepare_graph(graph, ids, output: Path):
    """Export each undirected edge once; compute LCC in original row coordinates."""
    upper = sparse.triu(graph, k=1, format="coo")
    count, components = connected_components(graph, directed=False)
    sizes = np.bincount(components)
    lcc_nodes = np.flatnonzero(components == int(np.argmax(sizes))).astype(np.int64)
    lcc = graph[lcc_nodes][:, lcc_nodes]
    degrees = np.diff(lcc.indptr)
    if np.any(degrees >= len(lcc_nodes)-1):
        raise ValueError("An LCC node has no non-neighbor for BPR negative sampling; supplied trainer would loop forever")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "edges.npz", src=upper.row.astype(np.int64),
                        dst=upper.col.astype(np.int64), weight=upper.data.astype(np.float32))
    np.save(output / "item_ids.npy", ids, allow_pickle=False)
    np.save(output / "lcc_nodes.npy", lcc_nodes, allow_pickle=False)
    isolated = int(np.sum(np.diff(graph.indptr) == 0))
    stats = dict(nodes=len(ids), edges=len(upper.data), components=int(count), largest_component=len(lcc_nodes),
                 lcc_edges=int(lcc.nnz//2), isolated_nodes=isolated,
                 outside_lcc_nodes=len(ids)-len(lcc_nodes), labeled_fraction=len(lcc_nodes)/len(ids),
                 item_id_convention="item_ids[row] identifies original sparse adjacency row; no ID shift")
    dump_json(output / "graph_stats.json", stats)
    return stats


def export_training_labels(output: Path, graph, ids, metadata: Path | None = None):
    """Restore labels to all item rows and disambiguate parent-relative child IDs."""
    nodes = np.load(output / "lcc_nodes.npy", allow_pickle=False)
    with (output / "hierarchical_labels.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    actual_nodes = np.array([int(r["graph_node"]) for r in rows], dtype=np.int64)
    if not np.array_equal(actual_nodes, nodes):
        raise ValueError("Hierarchy row order does not match LCC nodes")
    if [r["item_id"] for r in rows] != list(map(str, ids[nodes])):
        raise ValueError("Hierarchy item IDs do not match original graph rows")
    path_labels = np.array([[int(r[c]) for c in ("global_label", "local_label", "fine_label")] for r in rows])
    if path_labels.shape != (len(nodes), 3) or np.any(path_labels < 0):
        raise ValueError("Invalid LCC hierarchy labels")
    structural = np.load(output / "structural_embeddings.npy", allow_pickle=False, mmap_mode="r")
    if structural.ndim != 2 or structural.shape[0] != len(nodes) or not np.all(np.isfinite(structural)):
        raise ValueError("Structural embedding shape/values do not match LCC")
    labels = np.full((len(ids), 3), -1, dtype=np.int64)
    all_paths = labels.copy()
    all_paths[nodes] = path_labels
    unique_paths = []
    for depth in range(1, 4):
        paths, inverse = np.unique(path_labels[:, :depth], axis=0, return_inverse=True)
        labels[nodes, depth-1] = inverse
        unique_paths.append(paths)
    valid = np.zeros(len(ids), dtype=bool)
    valid[nodes] = True
    g_index = {tuple(p): i for i, p in enumerate(unique_paths[0])}
    l_index = {tuple(p): i for i, p in enumerate(unique_paths[1])}
    local_parent = np.array([g_index[tuple(p[:1])] for p in unique_paths[1]], dtype=np.int64)
    fine_parent = np.array([l_index[tuple(p[:2])] for p in unique_paths[2]], dtype=np.int64)
    if not np.array_equal(local_parent[labels[nodes, 1]], labels[nodes, 0]):
        raise AssertionError("Local-to-global parent mapping is inconsistent")
    if not np.array_equal(fine_parent[labels[nodes, 2]], labels[nodes, 1]):
        raise AssertionError("Fine-to-local parent mapping is inconsistent")
    np.savez_compressed(output / "pseudo_labels.npz", item_ids=ids, labels=labels, path_labels=all_paths,
                        valid_mask=valid, lcc_nodes=nodes, local_parent=local_parent, fine_parent=fine_parent)
    with (output / "all_item_labels.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "graph_node", "has_label", "status", "global_id", "local_id", "fine_id",
                         "global_path", "local_path", "fine_path"])
        degrees = np.diff(graph.indptr)
        for row, item in enumerate(ids):
            status = "labeled" if valid[row] else ("isolated" if degrees[row] == 0 else "outside_lcc")
            writer.writerow([item, row, bool(valid[row]), status, *labels[row], *all_paths[row]])
    dump_json(output / "label_schema.json", {
        "labels_shape": list(labels.shape), "columns": ["global_id", "local_id", "fine_id"],
        "label_counts": [len(p) for p in unique_paths], "valid_items": int(valid.sum()),
        "unlabeled_items": int((~valid).sum()), "unlabeled_value": -1,
        "row_mapping": "item_ids[row] is the item identity; MACRec default IDs equal row index",
        "labels": "Each column has globally unique contiguous cluster IDs within this run",
        "path_labels": "Original recursive K-means (global, local-within-global, fine-within-local) IDs",
        "parents": "local_parent[local_id] = global_id; fine_parent[fine_id] = local_id",
        "training": "Filter with valid_mask before indexing embeddings or computing guidance losses",
        "scope": "Largest connected component only; not semantic ground truth; IDs are run-specific",
    })
    if metadata:
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError("Expected an item-ID keyed JSON metadata dictionary")
        lcc = graph[nodes][:, nodes].tocoo()
        same = labels[nodes[lcc.row], 2] == labels[nodes[lcc.col], 2]
        strength = np.bincount(lcc.row[same], weights=lcc.data[same], minlength=len(nodes))
        sizes = np.bincount(labels[nodes, 2])
        chosen = np.argsort(-sizes, kind="stable")[:10]
        case_rows = []
        for cid in chosen:
            positions = np.flatnonzero(labels[nodes, 2] == cid)
            ranked = positions[np.lexsort((nodes[positions], -strength[positions]))[:8]]
            for pos in ranked:
                node = int(nodes[pos])
                record = meta.get(str(ids[node]), {})
                title = str(record.get("title", "")) if isinstance(record, dict) else ""
                case_rows.append(dict(item_id=str(ids[node]), title=title, global_id=int(labels[node, 0]),
                                      local_id=int(labels[node, 1]), fine_id=int(cid), fine_size=int(sizes[cid]),
                                      internal_weighted_degree=float(strength[pos])))
        with (output / "hierarchy_cases.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(case_rows[0]))
            writer.writeheader()
            writer.writerows(case_rows)
    return dict(label_counts=[len(p) for p in unique_paths], valid_items=int(valid.sum()),
                unlabeled_items=int((~valid).sum()))
