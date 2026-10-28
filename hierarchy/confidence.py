"""Offline graph-teacher confidence in original adjacency-row coordinates."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist


def graph_teacher_confidence(graph, embeddings, labels, *, tau_s=5.0, eps=1e-12, chunk_size=1024):
    """Return A/T/S/C and evidence counts, each [items, levels].

    embeddings must use the SAME coordinates/preprocessing as recursive K-means.
    labels contain globally unique IDs per level; -1 marks unlabeled rows.
    Centroids are reconstructed as means of the final assigned embeddings.
    Assignment compares only siblings; a parent with one child gets A=1 (the
    assignment is forced). Self edges are ignored. T and n use only neighbors in
    the same parent E_l(i); S = 1-exp(-n/tau_s[l]). A, T, S are absolute [0,1]
    scales, so C = cbrt(A*T*S) needs no kappa or rank/percentile normalization.
    tau_s is a scalar or one value per level. Unlabeled rows get C=0.
    """
    graph = sparse.csr_matrix(graph, dtype=np.float64, copy=True)
    graph.sum_duplicates()
    graph.setdiag(0)
    graph.eliminate_zeros()
    z = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels)
    if y.ndim != 2 or not y.shape[1] or y.dtype.kind not in "iu":
        raise ValueError("labels must be an integer [items, levels] array")
    n, levels = y.shape
    if graph.shape != (n, n) or z.ndim != 2 or z.shape[0] != n or not z.shape[1]:
        raise ValueError("graph, embeddings and labels must share item rows")
    tau_s = np.broadcast_to(np.asarray(tau_s, dtype=np.float64), (levels,))
    if not np.all(np.isfinite(tau_s)) or np.any(tau_s <= 0) or chunk_size < 1:
        raise ValueError("tau_s and chunk_size must be positive")
    valid = np.all(y >= 0, axis=1)
    if np.any(np.any(y >= 0, axis=1) != valid):
        raise ValueError("Each row must be fully labeled or fully unlabeled")
    if not np.all(np.isfinite(z[valid])) or not np.all(np.isfinite(graph.data)) or np.any(graph.data < 0):
        raise ValueError("Expected finite embeddings and finite nonnegative graph weights")
    for level in range(1, levels):
        pairs = np.unique(y[valid, level-1:level+1], axis=0)
        if len(np.unique(pairs[:, 1])) != len(pairs):
            raise ValueError("Child cluster IDs must be globally unique within a level")
    assignment = np.zeros((n, levels))
    topology = np.zeros_like(assignment)
    counts = np.zeros((n, levels), dtype=np.int64)
    edges = graph.tocoo()
    for level in range(levels):
        parent = np.zeros(n, dtype=np.int64) if level == 0 else y[:, level-1]
        for pid in np.unique(parent[valid]):
            rows = np.flatnonzero(valid & (parent == pid))
            clusters, inverse = np.unique(y[rows, level], return_inverse=True)
            if len(clusters) < 2:
                assignment[rows, level] = 1
                continue
            centers = np.zeros((len(clusters), z.shape[1]))
            np.add.at(centers, inverse, z[rows])
            centers /= np.bincount(inverse)[:, None]
            for start in range(0, len(rows), chunk_size):
                batch = rows[start:start+chunk_size]
                own = inverse[start:start+chunk_size]
                distances = cdist(z[batch], centers, metric="euclidean")
                positive = distances[np.arange(len(batch)), own].copy()
                distances[np.arange(len(batch)), own] = np.inf
                negative = distances.min(axis=1)
                assignment[batch, level] = np.clip((negative-positive)/(negative+eps), 0, 1)
        usable = valid[edges.row] & valid[edges.col] & (parent[edges.row] == parent[edges.col])
        src, dst, weights = edges.row[usable], edges.col[usable], edges.data[usable]
        total = np.bincount(src, weights=weights, minlength=n)
        same = y[src, level] == y[dst, level]
        within = np.bincount(src[same], weights=weights[same], minlength=n)
        topology[:, level] = np.divide(within, total, out=np.zeros(n), where=total > 0)
        counts[:, level] = np.bincount(src, minlength=n)
    support = -np.expm1(-counts/tau_s)
    confidence = np.cbrt(assignment * topology * support)
    return dict(assignment=assignment.astype(np.float32), topology=topology.astype(np.float32),
                support=support.astype(np.float32), confidence=confidence.astype(np.float32),
                neighbor_count=counts)
