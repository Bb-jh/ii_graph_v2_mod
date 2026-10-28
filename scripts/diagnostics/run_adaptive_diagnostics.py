"""Measure the static gate C*M (Confidence x Compatibility) on a synthetic demo or a snapshot.

R and N̂ need a training run (EMA of RQ losses) and are not measured here.
This measures weight attenuation, not downstream model quality.
Run from the repository root: python -m scripts.diagnostics.run_adaptive_diagnostics --demo --output DIR
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from hierarchy.adaptive_distillation import compatibility, hierarchy_masks


def summarize(compat, confidence, valid):
    """Per-level/compatibility-band stats over V only."""
    c = confidence.float()
    gate = c * compat
    bands = [("all", 0, 1.01), ("M<0.25", 0, .25), ("0.25<=M<0.5", .25, .5),
             ("0.5<=M<0.75", .5, .75), ("M>=0.75", .75, 1.01)]
    rows = []
    for level in range(c.shape[1]):
        for band, low, high in bands:
            selected = valid[:, level] & (compat[:, level] >= low) & (compat[:, level] < high)
            count = int(selected.sum())
            row = dict(level=level, band=band, count=count, valid_fraction=float(valid[:, level].float().mean()))
            for name, value in dict(C=c, M=compat, CM=gate).items():
                row[name+"_mean"] = float(value[selected, level].mean()) if count else None
            base = c[selected, level].sum()
            row["CM_retained_vs_C"] = float(gate[selected, level].sum()/base) if float(base) > 0 else None
            for q in (.1, .5, .9):
                row[f"CM_p{int(q*100)}"] = float(torch.quantile(gate[selected, level], q)) if count else None
            row["C_zero_fraction"] = float((c[selected, level] == 0).float().mean()) if count else None
            rows.append(row)
    return rows


def synthetic_snapshots():
    """Controlled hierarchy features, NOT trained Text/Image representations."""
    generator = torch.Generator().manual_seed(42)
    labels = torch.stack(torch.meshgrid(torch.arange(2), torch.arange(2), torch.arange(2), torch.arange(12),
                                        indexing="ij"), -1).reshape(-1, 4)[:, :3]
    labels = torch.stack([labels[:, 0], labels[:, 0]*2+labels[:, 1], labels[:, 0]*4+labels[:, 1]*2+labels[:, 2]], 1)
    centers = [F.normalize(torch.randn(n, 16, generator=generator), dim=-1) for n in (2, 4, 8)]
    structured = F.normalize(sum(center[labels[:, level]] for level, center in enumerate(centers)), dim=-1)
    noise = F.normalize(torch.randn(len(labels), 16, generator=generator), dim=-1)
    for mixture in (0.0, 0.5, 1.0):
        original = F.normalize((1-mixture)*structured + mixture*noise, dim=-1)
        yield f"synthetic_noise{mixture}", dict(original_z=original, confidence=torch.ones(len(labels), 3), labels=labels)


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as data:
        required = ('original_z', 'confidence', 'labels')
        missing = set(required)-set(data.files)
        if missing:
            raise ValueError(f"Missing snapshot arrays: {sorted(missing)}")
        return {key: torch.from_numpy(data[key].copy()) for key in required}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--demo', action='store_true', help='Synthetic only; no actual training evaluation')
    mode.add_argument('--snapshot', type=Path, help='Aligned NPZ (original_z, confidence, labels) for ONE modality batch')
    parser.add_argument('--output', type=Path, required=True, help='New or empty result directory')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.05, 0.1, 0.2])
    args = parser.parse_args()
    if any(not np.isfinite(t) or t <= 0 for t in args.temperatures):
        parser.error('temperatures must be finite and positive')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('Use a new or empty output directory')
    snapshots = synthetic_snapshots() if args.demo else [(args.snapshot.stem, load_snapshot(args.snapshot))]
    summaries, items = [], []
    for name, data in snapshots:
        candidate, positive, valid = hierarchy_masks(data['labels'])
        for temperature in args.temperatures:
            compat = compatibility(data['original_z'], candidate, positive, valid, temperature=temperature)
            prefix = dict(snapshot=name, temperature=temperature)
            summaries.extend(dict(**prefix, **row) for row in summarize(compat, data['confidence'], valid))
            for i in range(compat.shape[0]):
                for level in range(compat.shape[1]):
                    c = float(data['confidence'][i, level])
                    items.append(dict(**prefix, batch_row=i, level=level, valid=bool(valid[i, level]),
                                      C=c, M=float(compat[i, level]), CM=c*float(compat[i, level])))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'summary.csv', summaries)
    write_csv(args.output/'items.csv', items)
    metadata = dict(mode='synthetic_demo' if args.demo else 'snapshot',
                    source=str(args.snapshot) if args.snapshot else None,
                    temperatures=args.temperatures, synthetic_seed=42 if args.demo else None,
                    caveat='Static gate C*M only; R and N need training-time loss EMAs. No downstream quality claim.',
                    retained_definition='sum(C*M)/sum(C) over V; undefined when sum(C)=0')
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print('SYNTHETIC DIAGNOSTIC' if args.demo else 'SNAPSHOT DIAGNOSTIC')
    print(f"{'snapshot':30s} {'tau':>5s} {'lvl':>3s} {'C':>6s} {'M':>6s} {'CM':>6s} {'retained':>9s}")
    for row in summaries:
        if row['band'] != 'all':
            continue
        def fmt(key):
            value = row[key]
            return f'{value:.4f}' if value is not None else 'NA'
        print(f"{row['snapshot']:30s} {row['temperature']:5.2f} {row['level']:3d} "
              f"{fmt('C_mean'):>6s} {fmt('M_mean'):>6s} {fmt('CM_mean'):>6s} {fmt('CM_retained_vs_C'):>9s}")
    print(f'Results: {args.output.resolve()}')


if __name__ == '__main__':
    main()
