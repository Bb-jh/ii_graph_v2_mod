"""Semantic graph/evidence construction without allocating full N x N matrix."""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np

from config import SemanticGraphConfig
from evidence import EdgeEvidenceStore, canonical_edge


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def build_semantic_graph(
    item_ids: list[str], embeddings: np.ndarray, valid_mask: np.ndarray,
    cfg: SemanticGraphConfig, modality: str, evidence: EdgeEvidenceStore
) -> nx.Graph:
    """KNN graph for a modality. Invalid embeddings never create edges."""
    z = _normalize(embeddings)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    selected_by: dict[tuple[str, str], int] = defaultdict(int)
    score_by_edge: dict[tuple[str, str], float] = {}
    n = len(item_ids)

    for start in range(0, n, cfg.chunk_size):
        end = min(start + cfg.chunk_size, n)
        sims = z[start:end] @ z.T
        for local_i, i in enumerate(range(start, end)):
            if not valid_mask[i]:
                continue
            sims[local_i, ~valid_mask] = -np.inf
            sims[local_i, i] = -np.inf
            k = min(cfg.top_k, max(int(valid_mask.sum()) - 1, 0))
            if k <= 0:
                continue
            idx = np.argpartition(-sims[local_i], kth=k - 1)[:k]
            idx = idx[np.argsort(-sims[local_i, idx])]
            for j in idx:
                score = float(sims[local_i, j])
                if not np.isfinite(score) or score < cfg.sim_threshold:
                    continue
                edge = canonical_edge(item_ids[i], item_ids[int(j)])
                selected_by[edge] += 1
                score_by_edge[edge] = max(score_by_edge.get(edge, -np.inf), score)

    if cfg.symmetrize == "mutual":
        edges = {e for e, c in selected_by.items() if c >= 2}
    elif cfg.symmetrize == "union":
        edges = set(selected_by)
    else:
        raise ValueError(f"Unknown symmetrize: {cfg.symmetrize}")

    g = nx.Graph()
    g.add_nodes_from(item_ids)
    field = f"{modality}_similarity"
    for u, v in edges:
        score = score_by_edge[(u, v)]
        g.add_edge(u, v, weight=score, **{field: score})
        evidence.update(u, v, **{field: score})
    return g


def score_candidate_edges(
    candidate_edges: list[tuple[str, str]], item_ids: list[str], embeddings: np.ndarray,
    valid_mask: np.ndarray, modality: str, evidence: EdgeEvidenceStore,
    chunk_size: int = 8192,
) -> None:
    """Score only supplied edges (useful for behavior-backbone experiments).

    Candidate edges are processed in chunks so a large Behavior graph can be
    scored without building a full N x N similarity matrix or allocating all
    endpoint embeddings at once.
    """
    z = _normalize(embeddings)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    index = {item: i for i, item in enumerate(item_ids)}
    field = f"{modality}_similarity"

    valid_edges: list[tuple[str, str, int, int]] = []
    for u, v in candidate_edges:
        i, j = index.get(u), index.get(v)
        if i is None or j is None or not valid_mask[i] or not valid_mask[j]:
            continue
        valid_edges.append((u, v, i, j))

    for start in range(0, len(valid_edges), chunk_size):
        batch = valid_edges[start:start + chunk_size]
        left = np.fromiter((x[2] for x in batch), dtype=np.int64, count=len(batch))
        right = np.fromiter((x[3] for x in batch), dtype=np.int64, count=len(batch))
        scores = np.einsum("ij,ij->i", z[left], z[right], optimize=True)
        for (u, v, _, _), score in zip(batch, scores):
            evidence.update(u, v, **{field: float(score)})
