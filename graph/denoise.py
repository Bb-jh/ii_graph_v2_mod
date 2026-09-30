"""Experimental cross-modal denoising.

This is intentionally isolated because the exact DA-MRS-inspired criterion is
not fixed yet. Do not treat the default threshold strategy as the final method.
"""
from __future__ import annotations

import networkx as nx

from config import DenoiseConfig


def apply_cross_modal_consistency(
    graphs: dict[str, nx.Graph], cfg: DenoiseConfig
) -> dict[str, nx.Graph]:
    if not cfg.enabled:
        return graphs
    if cfg.strategy != "threshold":
        raise ValueError(f"Unknown denoise strategy: {cfg.strategy}")

    denoised = {name: g.copy() for name, g in graphs.items()}
    removed = set()
    for g in graphs.values():
        for u, v, data in g.edges(data=True):
            if float(data.get("weight", 0.0)) < cfg.consistency_threshold:
                removed.add((u, v))
    for g in denoised.values():
        g.remove_edges_from([(u, v) for u, v in removed if g.has_edge(u, v)])
    return denoised
