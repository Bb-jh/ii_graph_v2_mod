#!/usr/bin/env python3
"""Aggregate Utility v2 seeds 42/43/44 without changing pre-grid predictions.
Frozen primary predictions are the pre-existing user-registered (6,2) for both datasets.
Computed 3-seed utility-based allocation is diagnostic only, NOT a retroactive prediction.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path('outputs/modality_utility_v2')
SUMMARY = ROOT / '_preregistered_summary'
DOMAINS = {'Industrial': 'Industrial_and_Scientific', 'Musical': 'Musical_Instruments'}
SEEDS = (42, 43, 44)


def readrows(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def is_true(s):
    return str(s).strip().lower() == 'true'


def check_seed42():
    for domain, full in DOMAINS.items():
        folder = ROOT / f'{domain}_CLIP_42'
        allocation = json.loads((folder / 'predicted_allocation.json').read_text())
        if (allocation.get('L_T'), allocation.get('L_V')) != (6, 2):
            raise RuntimeError(f'Seed42 primary prediction changed: {domain}: {allocation}')
        if allocation.get('dataset') != full or allocation.get('seed') != 42:
            raise RuntimeError(f'Seed42 identity mismatch: {folder}')
        perm = readrows(folder / 'permutation_utility.csv')
        if {r['modality'] for r in perm} != {'T', 'V'}:
            raise RuntimeError(f'Bad seed42 permutation rows: {folder}')
        metrics = readrows(folder / 'utility_metrics.csv')
        if {r['split'] for r in metrics} != {'U2_OOF', 'U3_TRANSFER', 'U3_TRANSFER_ITEM_FOLD'}:
            raise RuntimeError(f'Missing U3 Item Fold split in {folder}')
        print(f'[BASELINE VERIFIED] {domain} seed42: frozen 6/2; U3 item fold present')


def dumpcsv(path, rows):
    # New files only: no overwrite of any preexisting frozen summary.
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite: {path}; inspect/backup first')
    if not rows:
        return
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def number(r, key):
    v = float(r[key])
    if not math.isfinite(v):
        raise ValueError(f'non-finite {key} in {r}')
    return v


def aggregate():
    check_seed42()
    seedrows = []
    for short, full in DOMAINS.items():
        for seed in SEEDS:
            folder = ROOT / f'{short}_CLIP_{seed}'
            perm = readrows(folder / 'permutation_utility.csv')
            metr = readrows(folder / 'utility_metrics.csv')
            alloc = json.loads((folder / 'predicted_allocation.json').read_text())
            manifest = json.loads((folder / 'manifest.json').read_text()) if (folder/'manifest.json').exists() else {}
            if manifest and (manifest.get('fold_mode') != 'anchor' or manifest.get('permutations') != 100 or manifest.get('bootstrap') != 500 or manifest.get('folds') != 5):
                raise RuntimeError(f'Inconsistent evaluation settings: {folder}')
            if alloc.get('dataset') != full or int(alloc.get('seed', -1)) != seed:
                raise RuntimeError(f'Bad allocation identity in {folder}')
            pm = {r['modality']: r for r in perm}
            mm = {r['split']: r for r in metr}
            if set(pm) != {'T', 'V'} or not {'U2_OOF', 'U3_TRANSFER', 'U3_TRANSFER_ITEM_FOLD'}.issubset(mm):
                raise RuntimeError(f'Incomplete metrics {folder}')
            if any(int(pm[m]['permutations']) != 100 or int(pm[m]['bootstrap_valid']) != 500 for m in ('T','V')):
                raise RuntimeError(f'Incomplete significance tests {folder}')
            row = dict(domain=short, dataset=full, encoder='CLIP', seed=seed,
                       n_edges=int(mm['U2_OOF']['n_edges']), positive_rate=number(mm['U2_OOF'], 'positive_rate'))
            for mod in ('T', 'V'):
                r = pm[mod]
                row[f'u_{mod}_u2'] = number(mm['U2_OOF'], f'u_{mod}')
                row[f'u_{mod}_u3_item_fold'] = number(mm['U3_TRANSFER_ITEM_FOLD'], f'u_{mod}')
                if abs(row[f'u_{mod}_u2'] - number(r, 'observed_u_auc')) > 1e-10:
                    raise RuntimeError(f'U2 utility mismatch: {folder}, {mod}')
                row[f'{mod}_permutation_p'] = number(r, 'p_one_sided')
                row[f'{mod}_ci_low'] = number(r, 'bootstrap_ci_low')
                row[f'{mod}_ci_high'] = number(r, 'bootstrap_ci_high')
                row[f'{mod}_perm_pass'] = is_true(r['passes_permutation'])
                row[f'{mod}_ci_pass'] = is_true(r['ci_excludes_zero'])
                row[f'{mod}_u3_sign_pass'] = is_true(r['u3_sign_matches'])
                row[f'{mod}_significant'] = is_true(r['significant'])
                row[f'{mod}_effective'] = max(0., row[f'u_{mod}_u2']) if row[f'{mod}_significant'] else 0.
                expected_effective = float(alloc[f'u_{mod}_effective'])
                if abs(row[f'{mod}_effective'] - expected_effective) > 1e-10:
                    raise RuntimeError(f'Effective utility mismatch: {folder}, {mod}')
            row.update(per_seed_L_T=int(alloc['L_T']), per_seed_L_V=int(alloc['L_V']))
            seedrows.append(row)
    summaryrows = []
    for short, full in DOMAINS.items():
        r = [row for row in seedrows if row['domain'] == short]
        o = dict(domain=short, dataset=full, n_seeds=len(r), seeds='42,43,44',
                 frozen_L_T=6, frozen_L_V=2, frozen_prediction_source='pre_grid_seed42_registration')
        for mod in ('T', 'V'):
            for key in (f'u_{mod}_u2', f'u_{mod}_u3_item_fold', f'{mod}_effective'):
                vals = [z[key] for z in r]
                o[f'{key}_mean'] = statistics.mean(vals)
                o[f'{key}_sd'] = statistics.stdev(vals)
                o[f'{key}_min'] = min(vals)
                o[f'{key}_max'] = max(vals)
            o[f'{mod}_significant_seeds'] = sum(z[f'{mod}_significant'] for z in r)
            o[f'{mod}_positive_u2_seeds'] = sum(z[f'u_{mod}_u2'] > 0 for z in r)
            o[f'{mod}_positive_u3_seeds'] = sum(z[f'u_{mod}_u3_item_fold'] > 0 for z in r)
        t = o['T_effective_mean']; v = o['V_effective_mean']; total=t+v
        # Diagnostic (not frozen) allocation computed from mean of per-seed effective utility
        lv = 2 + math.floor(4*v/total + .5) if total > 0 else 4
        o['computed_from_3seed_L_T'] = 8-lv
        o['computed_from_3seed_L_V'] = lv
        o['computed_matches_frozen'] = (8-lv, lv) == (6, 2)
        o['allocation_note'] = '3seed computation is diagnostic only; do not alter pre-grid frozen prediction'
        summaryrows.append(o)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    per = SUMMARY / 'per_seed_utility.csv'
    summ = SUMMARY / 'mean_sd_and_allocations_FROZEN.csv'
    if per.exists() or summ.exists():
        raise FileExistsError(f'Summary already exists. Refusing overwrite: {per} / {summ}')
    dumpcsv(per, seedrows)
    dumpcsv(summ, summaryrows)
    print(f'[WROTE] {per}')
    print(f'[WROTE] {summ}')
    for r in summaryrows:
        print(f"{r['domain']}: frozen=(6,2), 3seed diagnostic=({r['computed_from_3seed_L_T']},{r['computed_from_3seed_L_V']}), match={r['computed_matches_frozen']}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    check_seed42() if args.check_only else aggregate()
