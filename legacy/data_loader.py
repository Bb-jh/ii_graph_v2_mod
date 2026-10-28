"""Amazon interaction/metadata loading utilities.

The loader deliberately does not hard-code one exact raw filename. Different
Amazon-2018 dumps/repositories use slightly different names, so common patterns
are searched unless templates are supplied in config.py.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import DataConfig


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    timestamp: int
    category: str


def _iter_json_lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _resolve_file(raw_dir: Path, category: str, template: Optional[str], kind: str) -> Path:
    if template:
        path = raw_dir / template.format(category=category)
        if path.exists():
            return path
        raise FileNotFoundError(path)

    if kind == "interaction":
        candidates = [
            f"{category}_5.json.gz",
            f"reviews_{category}_5.json.gz",
            f"{category}.json.gz",
        ]
    else:
        candidates = [
            f"meta_{category}.json.gz",
            f"meta_{category}.json",
        ]

    for name in candidates:
        path = raw_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {kind} file for {category} under {raw_dir}. "
        f"Tried: {candidates}. Set a template in DataConfig if your naming differs."
    )


def load_category(category: str, cfg: DataConfig) -> list[Interaction]:
    path = _resolve_file(cfg.raw_data_dir, category, cfg.interaction_file_template, "interaction")
    interactions: list[Interaction] = []
    for row in _iter_json_lines(path):
        user = row.get("user_id") or row.get("reviewerID")
        item = row.get("parent_asin") or row.get("asin") or row.get("item_id")

        if user is None or item is None:
            continue

        ts = int(row.get("timestamp", row.get("unixReviewTime", 0)))

        # Amazon 2023 timestamp: milliseconds -> seconds
        if ts >= 10**11:
            ts //= 1000

        interactions.append(
            Interaction(str(user), str(item), ts, category)
        )
    return interactions


def load_item_metadata(category: str, cfg: DataConfig) -> dict[str, dict]:
    path = _resolve_file(
        cfg.raw_data_dir,
        category,
        cfg.metadata_file_template,
        "metadata"
    )

    meta: dict[str, dict] = {}

    for row in _iter_json_lines(path):
        # Amazon 2023: parent_asin / Amazon 2018: asin
        item = row.get("parent_asin") or row.get("asin") or row.get("item_id")

        if item is None:
            continue

        # Description
        description = row.get("description", "")
        if isinstance(description, list):
            description = " ".join(map(str, description))

        # Image URL
        image_url = row.get("image_url") or row.get("imUrl")

        # Amazon 2023: images 필드
        if not image_url:
            images = row.get("images") or []

            if isinstance(images, list):
                for image in images:
                    if isinstance(image, dict):
                        image_url = (
                            image.get("large")
                            or image.get("hi_res")
                            or image.get("thumb")
                        )
                        if image_url:
                            break

        # Amazon 2018: imageURLHighRes / imageURL
        if not image_url:
            urls = row.get("imageURLHighRes") or row.get("imageURL")
            if isinstance(urls, list) and urls:
                image_url = urls[0]

        meta[str(item)] = {
            "title": str(row.get("title", "") or ""),
            "description": str(description or ""),
            "image_url": image_url,
            "raw": row,
        }

    return meta


def build_item_universe(
    interactions: list[Interaction], metadata: dict[str, dict]
) -> list[str]:
    """Use one explicit item universe across all graph modules."""
    interaction_items = {x.item_id for x in interactions}
    return sorted(interaction_items.intersection(metadata.keys()))


def restrict_interactions(
    interactions: list[Interaction], allowed_items: set[str]
) -> list[Interaction]:
    return [x for x in interactions if x.item_id in allowed_items]
