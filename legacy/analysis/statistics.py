from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np


def graph_statistics(g: nx.Graph) -> dict:
    n = g.number_of_nodes()
    m = g.number_of_edges()
    degrees = np.asarray([d for _, d in g.degree()], dtype=float)
    weights = np.asarray([float(d.get("weight", 0.0)) for _, _, d in g.edges(data=True)], dtype=float)
    return {
        "nodes": n,
        "edges": m,
        "density": float(nx.density(g)) if n > 1 else 0.0,
        "isolated_nodes": int(len(list(nx.isolates(g)))),
        "connected_components": int(nx.number_connected_components(g)) if n else 0,
        "largest_component": int(max((len(c) for c in nx.connected_components(g)), default=0)),
        "avg_degree": float(degrees.mean()) if degrees.size else 0.0,
        "max_degree": int(degrees.max()) if degrees.size else 0,
        "weight_mean": float(weights.mean()) if weights.size else 0.0,
        "weight_std": float(weights.std()) if weights.size else 0.0,
    }


def edge_overlap(a: nx.Graph, b: nx.Graph) -> dict:
    ea = {frozenset(e) for e in a.edges()}
    eb = {frozenset(e) for e in b.edges()}
    inter = ea & eb
    union = ea | eb
    return {
        "intersection": len(inter),
        "union": len(union),
        "jaccard": len(inter) / len(union) if union else 0.0,
        "a_covered": len(inter) / len(ea) if ea else 0.0,
        "b_covered": len(inter) / len(eb) if eb else 0.0,
    }


def save_analysis(graphs: dict[str, nx.Graph], path: Path) -> dict:
    report = {"graphs": {name: graph_statistics(g) for name, g in graphs.items()}, "overlap": {}}
    names = list(graphs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f"{names[i]}__vs__{names[j]}"
            report["overlap"][key] = edge_overlap(graphs[names[i]], graphs[names[j]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
