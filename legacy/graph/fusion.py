"""Candidate/fusion strategies kept swappable on purpose."""
from __future__ import annotations

import networkx as nx
import numpy as np
from typing import Optional, Union

from config import FusionConfig
from evidence import EdgeEvidenceStore, canonical_edge


def fuse_graphs(
    item_ids: list[str], collaborative: nx.Graph, semantic: dict[str, nx.Graph],
    evidence: EdgeEvidenceStore, cfg: FusionConfig
) -> Union[dict[str, nx.Graph], nx.Graph]:
    if cfg.strategy in {"none", "separate"}:
        return {"collaborative": collaborative, **semantic}

    behavior_edges = {canonical_edge(u, v) for u, v in collaborative.edges()}
    semantic_edges = {
        canonical_edge(u, v)
        for g in semantic.values()
        for u, v in g.edges()
    }
    if cfg.strategy == "behavior_backbone":
        candidates = behavior_edges
    elif cfg.strategy == "semantic_backbone":
        candidates = semantic_edges
    elif cfg.strategy == "union":
        candidates = behavior_edges | semantic_edges
    else:
        raise ValueError(f"Unknown fusion strategy: {cfg.strategy}")

    # Normalize behavior evidence only over candidate edges to make scales comparable.
    candidates = sorted(candidates)
    behavior_raw = [float(evidence.records.get(e, {}).get("behavior_score", 0.0)) for e in candidates]
    behavior_norm = _minmax(behavior_raw)

    g = nx.Graph()
    g.add_nodes_from(item_ids)
    for edge, b_norm in zip(candidates, behavior_norm):
        u, v = edge
        rec = evidence.records.get(edge, {})
        sims = [
            float(rec[k]) for k in ("text_similarity", "image_similarity")
            if k in rec
        ]
        semantic_score = _reduce(sims, cfg.modality_reduce) if sims else 0.0
        modality_agreement = _agreement(rec)
        evidence.update(u, v, semantic_score=semantic_score, modality_agreement=modality_agreement)

        # Placeholder comparison rule, not a claimed final method.
        final = cfg.behavior_weight * float(b_norm) + cfg.semantic_weight * semantic_score
        evidence.update(u, v, normalized_behavior_score=float(b_norm), final_score=float(final))
        g.add_edge(
            u, v,
            weight=float(final),
            behavior_score=float(rec.get("behavior_score", 0.0)),
            semantic_score=float(semantic_score),
            modality_agreement=modality_agreement,
        )
    return g


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    x = np.asarray(values, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return [1.0 if hi > 0 else 0.0] * len(values)
    return ((x - lo) / (hi - lo)).astype(float).tolist()


def _reduce(values: list[float], mode: str) -> float:
    if mode == "mean":
        return float(np.mean(values))
    if mode == "min":
        return float(np.min(values))
    if mode == "max":
        return float(np.max(values))
    raise ValueError(f"Unknown modality_reduce: {mode}")


def _agreement(rec: dict) -> Optional[float]:
    if "text_similarity" not in rec or "image_similarity" not in rec:
        return None
    # Simple interpretable evidence for case study; not yet a final research definition.
    return float(1.0 - abs(float(rec["text_similarity"]) - float(rec["image_similarity"])))
