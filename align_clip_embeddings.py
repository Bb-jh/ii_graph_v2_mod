#!/usr/bin/env python3
"""Align full-domain CLIP embeddings with the frozen Amazon 2023 5-core item mapping.

Run from repository root. Requires numpy and embeddings/precomputed.py.
Does not re-encode text/images or alter the original CLIP NPZ or dataset.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from embeddings.precomputed import load_precomputed


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Industrial_and_Scientific')
    parser.add_argument('--processed-root', type=Path, default=Path.home() / 'datasets/amazon2023/processed_v2/5core')
    parser.add_argument('--embedding-root', type=Path, default=Path('artifacts/embeddings'))
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()

    dataset_dir = args.processed_root / args.dataset
    mapping_path = dataset_dir / 'item_mapping.tsv'
    source_path = args.embedding_root / args.dataset / 'full' / 'clip_vit_b32.npz'
    output_dir = args.output_dir or (args.embedding_root / args.dataset / '5core_v2')
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mapping_path.is_file():
        parser.error(f'No item mapping: {mapping_path}')
    if not source_path.is_file():
        parser.error(f'No CLIP embedding file: {source_path}')

    ids = []
    with mapping_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if not {'item_id', 'parent_asin'}.issubset(reader.fieldnames or []):
            parser.error('item_mapping.tsv needs columns item_id and parent_asin')
        for position, row in enumerate(reader):
            if int(row['item_id']) != position:
                raise ValueError(f'Mapping IDs are not contiguous at row {position}: {row}')
            ids.append(row['parent_asin'])

    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate parent_asin in item_mapping.tsv')

    text, image = load_precomputed(source_path, ids)
    text_values = np.asarray(text.values, dtype=np.float32)
    image_values = np.asarray(image.values, dtype=np.float32)
    text_mask = np.asarray(text.valid_mask, dtype=bool)
    image_mask = np.asarray(image.valid_mask, dtype=bool)

    expected_shape = (len(ids), 512)
    if text_values.shape != expected_shape or image_values.shape != expected_shape:
        raise ValueError(f'Expected {expected_shape}: got text={text_values.shape}, image={image_values.shape}')
    if not np.isfinite(text_values).all() or not np.isfinite(image_values).all():
        raise ValueError('NaN/Inf detected in embeddings')
    if np.any(~image_mask & np.any(image_values != 0, axis=1)):
        raise ValueError('Invalid image masks do not correspond to zero vectors')

    target = output_dir / 'clip_vit_b32_aligned.npz'
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix='.npz', delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        np.savez_compressed(temp_path,
            item_ids=np.asarray(ids, dtype=str),
            text_embeddings=text_values,
            image_embeddings=image_values,
            text_valid_mask=text_mask,
            image_valid_mask=image_mask,
            model_name=np.asarray('openai/clip-vit-base-patch32'),
        )
        with np.load(temp_path, allow_pickle=False) as saved:
            if not np.array_equal(saved['item_ids'], np.asarray(ids)):
                raise ValueError('Saved item order did not match mapping')
            if not np.array_equal(saved['text_valid_mask'], text_mask) or not np.array_equal(saved['image_valid_mask'], image_mask):
                raise ValueError('Saved masks differ')
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)

    report = {
        'dataset': args.dataset,
        'source_embedding': str(source_path.resolve()),
        'source_item_mapping': str(mapping_path.resolve()),
        'mapping_sha256': sha256(mapping_path),
        'aligned_embedding': str(target.resolve()),
        'items': len(ids),
        'text_shape': list(text_values.shape),
        'image_shape': list(image_values.shape),
        'text_valid': int(text_mask.sum()),
        'image_valid': int(image_mask.sum()),
        'text_coverage_pct': round(100 * float(text_mask.mean()), 4) if len(ids) else 0,
        'image_coverage_pct': round(100 * float(image_mask.mean()), 4) if len(ids) else 0,
    }
    (output_dir / 'alignment_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print('ALIGNMENT SUCCESS', flush=True)


if __name__ == '__main__':
    main()
