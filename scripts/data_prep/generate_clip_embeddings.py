
import argparse
import gzip
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"
DATA_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "amazon2023" / "raw"
OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "embeddings"
DIM = 512

DATASETS = (
    "Industrial_and_Scientific",
    "Musical_Instruments",
    "Office_Products",
)


def prepare_manifest(metadata_path, manifest_path):
    """Create a stable, resumable item manifest."""
    if manifest_path.exists():
        return

    temp_path = manifest_path.with_suffix(".building.sqlite")
    temp_path.unlink(missing_ok=True)

    connection = sqlite3.connect(temp_path)
    connection.execute(
        """
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            item_id TEXT UNIQUE NOT NULL,
            item_text TEXT NOT NULL,
            image_url TEXT
        )
        """
    )

    count = 0

    with gzip.open(metadata_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            item_id = row.get("parent_asin")
            if not item_id:
                continue

            description = row.get("description") or ""
            if isinstance(description, list):
                description = " ".join(map(str, description))

            title = str(row.get("title") or "")
            item_text = f"{title} {description}".strip()

            image_url = None
            for image in row.get("images") or []:
                if isinstance(image, dict):
                    image_url = (
                        image.get("large")
                        or image.get("hi_res")
                        or image.get("thumb")
                    )
                    if image_url:
                        break

            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO items
                (item_id, item_text, image_url)
                VALUES (?, ?, ?)
                """,
                (str(item_id), item_text, image_url),
            )

            count += cursor.rowcount

            if count and count % 10000 == 0:
                connection.commit()

    connection.commit()
    connection.close()
    os.replace(temp_path, manifest_path)

    print(f"Manifest created: {count:,} unique items", flush=True)


def load_image(url):
    if not url:
        return None

    for attempt in range(3):
        try:
            response = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()

            with Image.open(BytesIO(response.content)) as img:
                return img.convert("RGB")

        except Exception:
            if attempt < 2:
                time.sleep(attempt + 1)

    return None


def extract_features(output):
    features = getattr(output, "pooler_output", output)
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"Unexpected CLIP output: {type(features)}")
    return F.normalize(features.float(), dim=-1)


def encode_chunk(rows, model, processor, device, batch_size, workers):
    text_parts = []
    image_parts = []
    text_masks = []
    image_masks = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            texts = [row[2] for row in batch]
            urls = [row[3] for row in batch]

            images = list(pool.map(load_image, urls))
            positions = [
                i for i, image in enumerate(images)
                if image is not None
            ]

            text_inputs = processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            with torch.inference_mode():
                text_output = model.get_text_features(**text_inputs)
                text_values = extract_features(text_output)

            image_values = np.zeros(
                (len(batch), DIM), dtype=np.float32
            )

            if positions:
                valid_images = [images[i] for i in positions]

                image_inputs = processor(
                    images=valid_images,
                    return_tensors="pt",
                ).to(device)

                with torch.inference_mode():
                    image_output = model.get_image_features(**image_inputs)
                    features = extract_features(image_output)

                image_values[positions] = features.cpu().numpy()

            for img in images:
                if img is not None:
                    img.close()

            text_parts.append(text_values.cpu().numpy())
            image_parts.append(image_values)

            text_masks.extend(bool(t.strip()) for t in texts)
            image_masks.extend(img is not None for img in images)

    return {
        "item_ids": np.asarray([row[1] for row in rows]),
        "text_embeddings": np.concatenate(text_parts).astype(np.float32),
        "image_embeddings": np.concatenate(image_parts).astype(np.float32),
        "text_valid_mask": np.asarray(text_masks, dtype=bool),
        "image_valid_mask": np.asarray(image_masks, dtype=bool),
    }


def save_chunk(path, payload):
    temp = Path(str(path) + ".tmp")
    with open(temp, "wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(temp, path)


def finalize(chunks, output_path, total):
    """Merge checkpoints using disk-backed arrays."""
    print("Merging checkpoints...", flush=True)

    work_dir = output_path.parent / "merge_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "text_embeddings": np.lib.format.open_memmap(
            work_dir / "text.npy", mode="w+",
            dtype=np.float32, shape=(total, DIM)
        ),
        "image_embeddings": np.lib.format.open_memmap(
            work_dir / "image.npy", mode="w+",
            dtype=np.float32, shape=(total, DIM)
        ),
        "text_valid_mask": np.lib.format.open_memmap(
            work_dir / "text_mask.npy", mode="w+",
            dtype=bool, shape=(total,)
        ),
        "image_valid_mask": np.lib.format.open_memmap(
            work_dir / "image_mask.npy", mode="w+",
            dtype=bool, shape=(total,)
        ),
    }

    ids = []
    offset = 0

    for chunk in chunks:
        with np.load(chunk, allow_pickle=False) as data:
            count = len(data["item_ids"])
            ids.extend(data["item_ids"].tolist())

            for key, array in arrays.items():
                array[offset:offset + count] = data[key]

            offset += count

    if offset != total or len(set(ids)) != total:
        raise RuntimeError("Invalid item count or duplicate item IDs")

    for array in arrays.values():
        array.flush()

    temp_output = Path(str(output_path) + ".tmp")

    with open(temp_output, "wb") as f:
        np.savez_compressed(
            f,
            item_ids=np.asarray(ids),
            model_name=np.asarray(MODEL_NAME),
            **arrays,
        )

    os.replace(temp_output, output_path)
    print(f"Final NPZ saved: {output_path}", flush=True)

    # Checkpoints are intentionally preserved for recovery.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.batch_size < 1 or args.chunk_size < 1 or args.workers < 1:
        parser.error("Batch size, chunk size and workers must be positive")
    if args.limit < 0:
        parser.error("Limit cannot be negative")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device = "cuda:0"

    data_root = (args.data_dir or Path(os.environ.get("AMAZON2023_RAW_DIR", DATA_ROOT))).expanduser().resolve()
    output_root = (args.output_root or OUTPUT_ROOT).expanduser().resolve()
    metadata_path = data_root / f"meta_{args.dataset}.jsonl.gz"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    run_name = "full" if args.limit == 0 else f"pilot{args.limit}"
    output_dir = output_root / args.dataset / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)

    manifest_path = output_dir / "items.sqlite"
    prepare_manifest(metadata_path, manifest_path)

    db = sqlite3.connect(manifest_path)
    total = db.execute("SELECT COUNT(*) FROM items").fetchone()[0]

    if args.limit:
        total = min(total, args.limit)

    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Items: {total:,}", flush=True)

    output_path = output_dir / "clip_vit_b32.npz"
    if output_path.exists():
        print(f"Already completed: {output_path}", flush=True)
        db.close()
        return

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    if int(model.config.projection_dim) != DIM:
        raise RuntimeError("Unexpected CLIP embedding dimension")

    chunk_files = []
    completed = 0

    for start in range(0, total, args.chunk_size):
        count = min(args.chunk_size, total - start)
        chunk_path = chunk_dir / f"chunk_{start:09d}.npz"

        rows = db.execute(
            """
            SELECT id, item_id, item_text, image_url
            FROM items ORDER BY id
            LIMIT ? OFFSET ?
            """,
            (count, start),
        ).fetchall()

        if len(rows) != count:
            raise RuntimeError("Manifest size mismatch")

        if chunk_path.exists():
            with np.load(chunk_path, allow_pickle=False) as existing:
                saved_ids = existing["item_ids"].tolist()
                expected_ids = [row[1] for row in rows]
                if saved_ids != expected_ids:
                    raise RuntimeError(f"Checkpoint mismatch: {chunk_path}")
                if (
                    existing["text_embeddings"].shape != (count, DIM)
                    or existing["image_embeddings"].shape != (count, DIM)
                ):
                    raise RuntimeError(f"Invalid checkpoint: {chunk_path}")

            print(f"Skipping completed chunk: {start}", flush=True)

        else:
            payload = encode_chunk(
                rows, model, processor, device,
                args.batch_size, args.workers,
            )
            save_chunk(chunk_path, payload)
            print(
                f"Saved chunk: {start + count:,}/{total:,} "
                f"Image valid: {payload['image_valid_mask'].sum()}/{count}",
                flush=True,
            )

        chunk_files.append(chunk_path)
        completed += count

    db.close()

    finalize(chunk_files, output_path, total)
    print("SUCCESS", flush=True)


if __name__ == "__main__":
    main()
