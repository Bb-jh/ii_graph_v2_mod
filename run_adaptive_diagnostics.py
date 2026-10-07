"""Measure adaptive gate suppression on a synthetic demo or a training snapshot.

This measures weight attenuation, not downstream model quality.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from hierarchy.adaptive_distillation import adaptive_weights, hierarchy_candidate_mask


VARIANTS = ("CN", "CNM", "CN_sqrtM", "CN_floorM")


def weight_variants(confidence, need, compatibility):
    base = confidence * need
    return dict(CN=base, CNM=base*compatibility,
                CN_sqrtM=base*compatibility.sqrt(),
                CN_floorM=base*(0.2+0.8*compatibility))


def summarize(result, confidence):
    """Per-level/mismatch-band stats; exclude unusable candidate sets explicitly."""
    c = confidence.float()
    n, m, usable = (result[k] for k in ("need", "compatibility", "usable"))
    variants = weight_variants(c, n, m)
    bands = [("all", 0, 1.01), ("Dorig<0.25", 0, .25),
             ("0.25<=Dorig<0.5", .25, .5), ("0.5<=Dorig<0.75", .5, .75),
             ("Dorig>=0.75", .75, 1.01)]
    rows = []
    for level in range(c.shape[1]):
        for band, low, high in bands:
            selected = usable[:, level] & ((1-m[:, level]) >= low) & ((1-m[:, level]) < high)
            count = int(selected.sum())
            row = dict(level=level, band=band, count=count,
                       usable_fraction=float(usable[:, level].float().mean()))
            values = dict(C=c, N=n, M=m, Dorig=1-m, **variants)
            for name, value in values.items():
                row[name+"_mean"] = float(value[selected, level].mean()) if count else None
            base = variants["CN"][selected, level].sum()
            for name in VARIANTS:
                row[name+"_retained_vs_CN"] = (
                    float(variants[name][selected, level].sum()/base) if float(base) > 0 else None)
            alpha = variants['CNM'][selected, level]
            row['alpha_p10'] = float(torch.quantile(alpha, .1)) if count else None
            row['alpha_p50'] = float(torch.quantile(alpha, .5)) if count else None
            row['alpha_p90'] = float(torch.quantile(alpha, .9)) if count else None
            row['alpha_below_0.01_fraction'] = float((alpha < .01).float().mean()) if count else None
            row['C_zero_fraction'] = float((c[selected, level] == 0).float().mean()) if count else None
            rows.append(row)
    return rows


def synthetic_snapshots():
    """Controlled embedding examples, NOT trained Text/Image representations."""
    generator = torch.Generator().manual_seed(42)
    graph = F.normalize(torch.randn(96, 12, generator=generator), dim=-1)
    noise = F.normalize(torch.randn(96, 12, generator=generator), dim=-1)
    labels = torch.zeros(96, 1, dtype=torch.long)
    for mixture in (0.0, 0.5, 1.0):
        original = graph if mixture == 0 else F.normalize((1-mixture)*graph + mixture*noise, dim=-1)
        for state, student in (("student_equals_original", original), ("student_equals_teacher", graph)):
            yield f"synthetic_mix{mixture}_{state}", dict(
                graph_z=graph[:, None, :], student_z=student[:, None, :],
                original_z=original, confidence=torch.ones(96, 1), labels=labels)


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as data:
        required = ('graph_z', 'student_z', 'original_z', 'confidence', 'labels')
        missing = set(required)-set(data.files)
        if missing:
            raise ValueError(f"Missing snapshot arrays: {sorted(missing)}")
        snapshot = {key: torch.from_numpy(data[key].copy()) for key in required}
        if 'candidate_mask' in data:
            snapshot['candidate_mask'] = torch.from_numpy(data['candidate_mask'].copy())
    return snapshot


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--demo', action='store_true', help='Synthetic only; no actual training evaluation')
    mode.add_argument('--snapshot', type=Path, help='Aligned NPZ from ONE modality and training step')
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
        mask = data.get('candidate_mask')
        if mask is None:
            mask = hierarchy_candidate_mask(data['labels'])
        for temperature in args.temperatures:
            result = adaptive_weights(data['graph_z'], data['student_z'], data['original_z'],
                                      data['confidence'], mask, temperature=temperature,
                                      modality_temperature=temperature)
            prefix = dict(snapshot=name, temperature=temperature)
            for row in summarize(result, data['confidence']):
                summaries.append(dict(**prefix, **row))
            variants = weight_variants(data['confidence'], result['need'], result['compatibility'])
            for i in range(data['confidence'].shape[0]):
                for level in range(data['confidence'].shape[1]):
                    usable = bool(result['usable'][i, level])
                    items.append(dict(**prefix, batch_row=i, level=level, usable=usable,
                        C=float(data['confidence'][i, level]), N=float(result['need'][i, level]),
                        M=float(result['compatibility'][i, level]),
                        Dorig=float(1-result['compatibility'][i, level]) if usable else None,
                        **{k: float(v[i, level]) for k, v in variants.items()}))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'summary.csv', summaries)
    write_csv(args.output/'items.csv', items)
    # Analytic limiting case: student relation == original relation, C=1.
    controlled = []
    for d in (0, .1, .25, .5, .75, .9, 1):
        variants = weight_variants(torch.tensor(1.), torch.tensor(float(d)), torch.tensor(1.-d))
        controlled.append(dict(Dorig=d, N=d, M=1-d, **{k: float(v) for k, v in variants.items()}))
    write_csv(args.output/'controlled_curve.csv', controlled)
    metadata = dict(mode='synthetic_demo' if args.demo else 'training_snapshot',
                    source=str(args.snapshot) if args.snapshot else None,
                    temperatures=args.temperatures, floor=0.2, synthetic_seed=42 if args.demo else None,
                    caveat='Weight attenuation only; no evidence of downstream quality or over-suppression.',
                    retained_definition='sum(variant)/sum(CN) over usable rows; undefined when sum(CN)=0')
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print('SYNTHETIC DIAGNOSTIC' if args.demo else 'TRAINING SNAPSHOT DIAGNOSTIC')
    print('No downstream quality claim. Retained = sum(CNM)/sum(CN), not loss/gradient ratio.')
    print(f"{'snapshot':55s} {'tau':>5s} {'lvl':>3s} {'C':>6s} {'N':>6s} {'M':>6s} {'alpha':>7s} {'retained':>9s}")
    for row in summaries:
        if row['band'] != 'all':
            continue
        def fmt(key):
            value = row[key]
            return f'{value:.4f}' if value is not None else 'NA'
        print(f"{row['snapshot']:55s} {row['temperature']:5.2f} {row['level']:3d} "
              f"{fmt('C_mean'):>6s} {fmt('N_mean'):>6s} {fmt('M_mean'):>6s} "
              f"{fmt('CNM_mean'):>7s} {fmt('CNM_retained_vs_CN'):>9s}")
    print(f'Results: {args.output.resolve()}')


if __name__ == '__main__':
    main()
