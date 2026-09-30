from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


def top_neighbors(g: nx.Graph, item_id: str, top_n: int = 10) -> list[dict]:
    if item_id not in g:
        return []
    rows = []
    for neighbor, attrs in g[item_id].items():
        row = {"item_id": neighbor, **attrs}
        rows.append(row)
    rows.sort(key=lambda x: float(x.get("weight", 0.0)), reverse=True)
    return rows[:top_n]


def run_case_studies(
    graphs: dict[str, nx.Graph], item_ids: list[str], requested: tuple[str, ...],
    top_n: int, output_path: Path
) -> dict:
    # If user did not specify items, choose a few high-degree behavior nodes when available.
    chosen = [x for x in requested if x in item_ids]
    if not chosen:
        source = graphs.get("collaborative") or next(iter(graphs.values()))
        chosen = [n for n, _ in sorted(source.degree, key=lambda x: x[1], reverse=True)[:3]]
    report = {
        item: {name: top_neighbors(g, item, top_n) for name, g in graphs.items()}
        for item in chosen
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
