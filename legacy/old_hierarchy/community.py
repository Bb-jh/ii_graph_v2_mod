"""Experimental community discovery only.

This module intentionally does not claim that communities are already valid
Global/Local/Fine pseudo-labels. First inspect whether the constructed graph
has stable/useful community structure.
"""
from __future__ import annotations

import networkx as nx

from config import HierarchyConfig


def detect_communities(g: nx.Graph, cfg: HierarchyConfig) -> dict[str, int]:
    if cfg.algorithm != "louvain":
        raise NotImplementedError("Only Louvain is wired for the skeleton. Add Leiden after graph case-study validation.")
    communities = nx.community.louvain_communities(
        g, weight="weight", resolution=cfg.resolution, seed=0
    )
    return {node: cid for cid, nodes in enumerate(communities) for node in nodes}
