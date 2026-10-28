"""Edge-level intermediate evidence.

Instead of collapsing all information immediately into one graph weight, keep
behavior/text/image signals so later experiments can reuse them for fusion,
case studies, hierarchy guidance, or distillation.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def canonical_edge(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u <= v else (v, u)


@dataclass
class EdgeEvidenceStore:
    records: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    def update(self, u: str, v: str, **values: Any) -> None:
        if u == v:
            return
        key = canonical_edge(u, v)
        rec = self.records.setdefault(key, {"item_i": key[0], "item_j": key[1]})
        rec.update(values)

    def get(self, u: str, v: str) -> dict[str, Any]:
        return self.records.get(canonical_edge(u, v), {})

    def export_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.records:
            path.write_text("item_i,item_j\n", encoding="utf-8")
            return
        fieldnames = ["item_i", "item_j"]
        extra = sorted({k for r in self.records.values() for k in r if k not in fieldnames})
        fieldnames += extra
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for key in sorted(self.records):
                writer.writerow(self.records[key])
