from __future__ import annotations

from pathlib import Path

import numpy as np

from embeddings.base import EmbeddingOutput


def load_precomputed(path: Path, target_item_ids: list[str]) -> tuple[EmbeddingOutput, EmbeddingOutput]:
    data = np.load(path, allow_pickle=False)
    source_ids = [str(x) for x in data["item_ids"].tolist()]
    index = {item: i for i, item in enumerate(source_ids)}
    missing = [item for item in target_item_ids if item not in index]
    if missing:
        raise ValueError(f"Precomputed embeddings miss {len(missing)} target items; examples={missing[:5]}")
    rows = np.asarray([index[item] for item in target_item_ids])
    text = np.asarray(data["text_embeddings"])[rows]
    image = np.asarray(data["image_embeddings"])[rows]
    text_mask = np.ones(len(rows), dtype=bool)
    if "text_valid_mask" in data:
        text_mask = np.asarray(data["text_valid_mask"], dtype=bool)[rows]
    image_mask = np.ones(len(rows), dtype=bool)
    if "image_valid_mask" in data:
        image_mask = np.asarray(data["image_valid_mask"], dtype=bool)[rows]
    return (
        EmbeddingOutput(text, text_mask),
        EmbeddingOutput(image, image_mask),
    )
