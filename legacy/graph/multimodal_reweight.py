"""Multimodal reliability reweighting on a fixed behavior I-I backbone.

This module implements the experiment discussed in the project notes:

1) Build/prune the behavior I-I graph first.
2) Freeze its edge set E_B.
3) Score only E_B with text/image cosine similarity.
4) Convert the two semantic signals into a multimodal reliability M_ij.
5) Soft-reweight the behavior strength without creating/removing edges.

The default reliability is intentionally simple and interpretable:

    semantic_strength = mean(text_sim_01, image_sim_01)
    consistency       = 1 - |text_sim_01 - image_sim_01|
    M_ij              = semantic_strength * consistency
    W_ij              = (1-lambda) * B_ij + lambda * M_ij

where cosine similarities are clipped to [0, 1] before reliability is computed.
This is a DA-MRS-inspired experiment, not a claim that the exact paper formula is
reproduced verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np

from evidence import EdgeEvidenceStore


@dataclass(frozen=True)
class MultimodalReliabilityResult:
    text_similarity_01: Optional[float]
    image_similarity_01: Optional[float]
    semantic_strength: Optional[float]
    cross_modal_consistency: Optional[float]
    multimodal_reliability: Optional[float]
    reliability_available: bool


def build_multimodal_behavior_graph(
    item_ids: list[str],
    collaborative: nx.Graph,
    evidence: EdgeEvidenceStore,
    *,
    lambda_multimodal: float = 0.3,
    reliability_mode: str = "strength_x_consistency",
    cosine_transform: str = "clip01",
    behavior_normalization: str = "log_max",
    missing_modality: str = "behavior_only",
) -> nx.Graph:
    """Soft-reweight a fixed behavior graph using text/image reliability.

    The returned graph has exactly the same nodes and edges as ``collaborative``.
    Only edge weights and diagnostic attributes change.

    Args:
        lambda_multimodal: semantic contribution in [0, 1].
        reliability_mode:
            - strength_x_consistency (default)
            - mean
            - min
        cosine_transform:
            - clip01: clip raw cosine to [0,1]
            - shift01: map [-1,1] -> [0,1]
        behavior_normalization:
            - log_max: log1p(score) / max(log1p(score))
            - max: score / max(score)
            - minmax: standard min-max
            - none: raw behavior score
        missing_modality:
            - behavior_only: if text or image is unavailable, keep B_ij only
            - single: use the available modality as M_ij
            - zero: use M_ij=0
    """
    if not 0.0 <= lambda_multimodal <= 1.0:
        raise ValueError("lambda_multimodal must be in [0, 1]")

    edges = list(collaborative.edges())
    behavior_raw = [
        float(collaborative[u][v].get(
            "behavior_score",
            collaborative[u][v].get("weight", 0.0),
        ))
        for u, v in edges
    ]
    behavior_norm = _normalize_behavior(behavior_raw, behavior_normalization)

    g = nx.Graph()
    g.add_nodes_from(item_ids)

    for (u, v), b_raw, b_norm in zip(edges, behavior_raw, behavior_norm):
        rec = evidence.get(u, v)
        reliability = compute_multimodal_reliability(
            rec.get("text_similarity"),
            rec.get("image_similarity"),
            mode=reliability_mode,
            cosine_transform=cosine_transform,
            missing_modality=missing_modality,
        )

        if reliability.multimodal_reliability is None:
            # Missing multimodal evidence should not erase a valid behavior edge.
            final = float(b_norm)
        else:
            final = (
                (1.0 - lambda_multimodal) * float(b_norm)
                + lambda_multimodal * float(reliability.multimodal_reliability)
            )

        attrs = {
            "behavior_score": float(b_raw),
            "normalized_behavior_score": float(b_norm),
            "text_similarity": rec.get("text_similarity"),
            "image_similarity": rec.get("image_similarity"),
            "text_similarity_01": reliability.text_similarity_01,
            "image_similarity_01": reliability.image_similarity_01,
            "semantic_strength": reliability.semantic_strength,
            "cross_modal_consistency": reliability.cross_modal_consistency,
            "multimodal_reliability": reliability.multimodal_reliability,
            "reliability_available": bool(reliability.reliability_available),
            "final_score": float(final),
            "weight": float(final),
        }
        # NetworkX accepts None-valued attributes; keeping them is useful for case study.
        g.add_edge(u, v, **attrs)
        evidence.update(u, v, **attrs)

    return g


def compute_multimodal_reliability(
    text_similarity: Optional[float],
    image_similarity: Optional[float],
    *,
    mode: str = "strength_x_consistency",
    cosine_transform: str = "clip01",
    missing_modality: str = "behavior_only",
) -> MultimodalReliabilityResult:
    """Convert text/image cosine similarities into an interpretable reliability."""
    t = _transform_optional(text_similarity, cosine_transform)
    i = _transform_optional(image_similarity, cosine_transform)

    if t is not None and i is not None:
        strength = 0.5 * (t + i)
        consistency = 1.0 - abs(t - i)
        if mode == "strength_x_consistency":
            mm = strength * consistency
        elif mode == "mean":
            mm = strength
        elif mode == "min":
            mm = min(t, i)
        else:
            raise ValueError(f"Unknown reliability_mode: {mode}")
        return MultimodalReliabilityResult(t, i, strength, consistency, float(mm), True)

    available = t if t is not None else i
    if available is None:
        return MultimodalReliabilityResult(t, i, None, None, None, False)

    if missing_modality == "behavior_only":
        return MultimodalReliabilityResult(t, i, available, None, None, False)
    if missing_modality == "single":
        return MultimodalReliabilityResult(t, i, available, None, float(available), True)
    if missing_modality == "zero":
        return MultimodalReliabilityResult(t, i, available, None, 0.0, True)
    raise ValueError(f"Unknown missing_modality: {missing_modality}")


def _transform_optional(value: Optional[float], mode: str) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    if mode == "clip01":
        return float(np.clip(value, 0.0, 1.0))
    if mode == "shift01":
        return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))
    raise ValueError(f"Unknown cosine_transform: {mode}")


def _normalize_behavior(values: list[float], mode: str) -> list[float]:
    if not values:
        return []
    x = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(x)):
        raise ValueError("behavior scores contain NaN/Inf")
    if np.any(x < 0):
        raise ValueError("behavior scores must be non-negative")

    if mode == "none":
        return x.astype(float).tolist()
    if mode == "log_max":
        y = np.log1p(x)
        hi = float(y.max())
        return (y / hi).astype(float).tolist() if hi > 0 else [0.0] * len(values)
    if mode == "max":
        hi = float(x.max())
        return (x / hi).astype(float).tolist() if hi > 0 else [0.0] * len(values)
    if mode == "minmax":
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            return [1.0 if hi > 0 else 0.0] * len(values)
        return ((x - lo) / (hi - lo)).astype(float).tolist()
    raise ValueError(f"Unknown behavior_normalization: {mode}")
