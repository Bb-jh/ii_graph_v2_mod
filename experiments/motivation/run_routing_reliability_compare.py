#!/usr/bin/env python3
"""Held-out U3, equal-edge-budget comparison of behavior-conditioned routing scores.

Uses immutable outputs from run_routing_e_ablation.py. Does not refit models.
For B-matched Raw, mimic the E route's *U1 behavior-feature stratum sizes*
while selecting the highest raw semantic similarity in each stratum.
B-matched Raw is a conditional diagnostic, NOT a deployment-ready score.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def load_npz(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Need prior prepare/fit outputs.")
    with np.load(path, allow_pickle=False) as src:
        return {k: src[k] for k in src.files}


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    # Identical deterministic tie-break to original script.
    return np.lexsort((np.arange(len(scores)), -scores))[:k].astype(np.int32)


def strata_from_features(base: np.ndarray, n_bins: int) -> tuple[np.ndarray, dict]:
    """Fixed, outcome-blind quantile bins for each of three U1 behavior features."""
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError(f"Expected three behavior features; got shape={base.shape}")
    columns = []
    descriptions = {}
    sizes = []
    for feature in range(3):
        cuts = np.unique(np.quantile(base[:, feature], np.linspace(0, 1, n_bins+1)[1:-1]))
        b = np.searchsorted(cuts, base[:, feature], side="left").astype(np.int32)
        columns.append(b)
        sizes.append(len(cuts) + 1)
        descriptions[f"u1_feature_{feature}"] = {
            "thresholds": cuts.tolist(), "n_bins_actual": len(cuts)+1}
    strata = columns[0] + sizes[0] * (columns[1] + sizes[1] * columns[2])
    values, counts = np.unique(strata, return_counts=True)
    descriptions['n_nonempty_strata'] = len(values)
    descriptions['min_stratum_size'] = int(counts.min())
    descriptions['max_stratum_size'] = int(counts.max())
    return strata, descriptions


def matched_raw_indices(raw_s: np.ndarray, strata: np.ndarray,
                        reference_idx: np.ndarray, k: int) -> np.ndarray:
    """Exactly match E top-k per-stratum quotas, rank by raw cosine within each."""
    if len(reference_idx) != k:
        raise ValueError("Reference selection has wrong budget")
    chosen = []
    for stratum in np.unique(strata[reference_idx]):
        quota = int(np.count_nonzero(strata[reference_idx] == stratum))
        pool = np.flatnonzero(strata == stratum)
        if len(pool) < quota:
            raise AssertionError('Quota impossible despite reference originating in the stratum')
        local = np.lexsort((pool, -raw_s[pool]))[:quota]
        chosen.append(pool[local])
    selected = np.concatenate(chosen).astype(np.int32)
    if len(selected) != k or len(np.unique(selected)) != k:
        raise AssertionError("B-matched Raw did not return exactly k unique edges")
    return selected


def evaluate(d: dict, idx: np.ndarray, score: np.ndarray, k: int) -> dict:
    vals = {
        'u2_recurrence': float(np.mean(d['y2'][idx])),
        'u3_recurrence': float(np.mean(d['y3'][idx])),
        'u3_recurrence_2users': float(np.mean(d['y3_2users'][idx])),
        'u3_norm_strength': float(np.mean(d['strength3'][idx])),
    }
    positive = score[idx] > 0
    vals['active_edges'] = int(positive.sum())
    vals['active_fraction'] = float(positive.mean())
    vals['selected_edges'] = int(len(idx))
    vals['selected_edge_fraction'] = float(len(idx) / len(d['a']))
    vals['u3_recurrence_active_only'] = (
        float(np.mean(d['y3'][idx[positive]])) if positive.any() else float('nan'))
    vals['mean_selected_u1_weight'] = float(np.mean(d['wb'][idx]))
    return vals


def read_prior_thresholds(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            out[(row['modality'], int(round(float(row['q'])*1000)))] = float(row['selected_tau'])
    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f'No results to write: {path}')
    keys = list(rows[0])
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True, help='Existing routing_e/industrial_split42 dir')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--q', nargs='+', type=float, default=[.1,.2,.3])
    ap.add_argument('--behavior-bins', type=int, default=4,
                    help='Quantile bins per U1 behavior feature; 4 gives up to 64 cells')
    ap.add_argument('--include-prior-threshold', action='store_true',
                    help='Reuse the previously tuned U2 threshold for comparability')
    args = ap.parse_args()
    if args.behavior_bins < 2 or not args.q or any(not (0 < q < 1) for q in args.q):
        ap.error('Invalid q/behavior-bins')
    if args.input.resolve() == args.out.resolve():
        ap.error('Use a separate output folder to protect original results')

    d = load_npz(args.input/'routing_candidates.npz')
    e = load_npz(args.input/'expressibility_scores.npz')
    required_d = {'a','wb','st','sv','bfeatures','y2','y3','y3_2users','strength3'}
    required_e = {f'{v}_{m}_full' for m in ('T','V') for v in ('E','pbase','pfull')}
    if required_d - d.keys() or required_e - e.keys():
        raise KeyError(f'Missing keys: candidates {required_d-d.keys()}, scores {required_e-e.keys()}')
    n = len(d['a'])
    if any(len(d[key]) != n for key in required_d) or any(len(e[key]) != n for key in required_e):
        raise ValueError('Cache length mismatch')
    if len(d['wb']) != 149825:
        print(f'NOTE: edge count is {n}; not the prior Industrial seed42 value 149825.')
    if (d['wb'] < 0).any():
        raise ValueError('Negative behavior weights detected')
    if not all(np.isfinite(d[x]).all() for x in ('wb','st','sv','bfeatures','strength3')):
        raise ValueError('Nonfinite values in candidates')
    if not all(np.isfinite(e[x]).all() for x in required_e):
        raise ValueError('Nonfinite predictions')
    for v in ('T','V'):
        p=e[f'pfull_{v}_full'].astype(float)
        p0=e[f'pbase_{v}_full'].astype(float)
        E=e[f'E_{v}_full'].astype(float)
        if np.min(p)<0 or np.max(p)>1 or np.min(p0)<0 or np.max(p0)>1:
            raise ValueError('Probability prediction outside [0,1]')
        if not np.allclose(E, np.maximum(p-p0,0),rtol=2e-4,atol=3e-7):
            raise AssertionError(f'{v}: E cache differs from positive incremental prediction')
    strata, strat_info = strata_from_features(d['bfeatures'].astype(float), args.behavior_bins)
    thresholds = read_prior_thresholds(args.input/'threshold_tuning.csv') if args.include_prior_threshold else {}
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[]; comparisons=[]; selections={}

    for m,s_key in [('T','st'),('V','sv')]:
        wb=d['wb'].astype(float)
        s=d[s_key].astype(float)
        E=e[f'E_{m}_full'].astype(float)
        p=e[f'pfull_{m}_full'].astype(float)
        pbase=e[f'pbase_{m}_full'].astype(float)
        raw=wb*np.maximum(s,0)
        escore=wb*E
        scores={
            'Raw_wB_times_s': raw,
            'Expressibility_wB_times_E': escore,
            'Total_reliability_P_aug': p,
            'Behavior_only_wB': wb,
        }
        # Extra control identifies how much stems from learned behavior-only features
        # rather than similarity. Useful because p_aug contains behavior features.
        scores['Behavior_only_P_base']=pbase
        for q in args.q:
            k=max(1, int(np.floor(n*q)))
            if k>=n: raise ValueError('Invalid budget')
            reference=top_indices(escore,k)
            methods={name:(score,top_indices(score,k)) for name,score in scores.items()}
            matched_idx=matched_raw_indices(s,strata,reference,k)
            # B-matched Raw is a conditional matched comparator; its selection
            # is by RAW COSINE, not raw*wB; distribution matches wB*E by construction.
            methods['B_matched_Raw_s_given_E_strata']=(np.maximum(s,0),matched_idx)
            if args.include_prior_threshold:
                tau=thresholds.get((m,int(round(q*1000))))
                if tau is None:
                    raise ValueError(f'Prior tuned threshold missing for {m}, q={q}')
                gated=raw*(s>tau)
                methods['Raw_plus_threshold_prior']=(gated,top_indices(gated,k))
            outcomes={}
            for method,(score,idx) in methods.items():
                stats=evaluate(d,idx,score,k)
                stats['mean_selected_raw_similarity']=float(np.mean(s[idx]))
                if len(idx)!=k or len(np.unique(idx))!=k:
                    raise AssertionError(f'Budget mismatch for {m}, {q}, {method}')
                key=f'{m}_{int(round(q*1000))}_{method}'
                selections[key]=idx
                row=dict(modality=m,q=q,k=k,method=method,**stats)
                rows.append(row)
                outcomes[method]=stats
            target=outcomes['Expressibility_wB_times_E']
            for method,stats in outcomes.items():
                if method=='Expressibility_wB_times_E':continue
                comparisons.append(dict(modality=m,q=q,k=k,comparison='E_minus_'+method,
                                        delta_u3_recurrence_pp=100*(target['u3_recurrence']-stats['u3_recurrence']),
                                        delta_u3_norm_strength=target['u3_norm_strength']-stats['u3_norm_strength']))
            short=' | '.join(f'{key}={100*outcomes[key]["u3_recurrence"]:.3f}%'
                             for key in ['Raw_wB_times_s','Expressibility_wB_times_E',
                                         'Total_reliability_P_aug','Behavior_only_wB',
                                         'B_matched_Raw_s_given_E_strata'])
            print(f'{m} q={q:g} k={k}: {short}',flush=True)
    write_csv(args.out/'routing_reliability_comparison.csv',rows)
    write_csv(args.out/'routing_reliability_deltas.csv',comparisons)
    np.savez_compressed(args.out/'selected_edge_indices.npz',**selections)
    manifest={
        'candidate_edges':n,'input_folder':str(args.input),'q':args.q,
        'edge_budget':'global exact floor(q*N), deterministic ties by input index',
        'split':'U1 behavior candidates; U2 full-fit predictions; U3 held out',
        'B_matched_raw':'Same U1 base-feature joint quantile-stratum selected counts as top-k(wB*E); choose highest original raw cosine within stratum. Uses E stratum allocation, but no U2/U3 outcomes directly for ranking within stratum. Diagnostic only.',
        'behavior_bin_scheme':strat_info,
        'P_aug':'pfull_{modality}_full = P_hat(Y_U2=1 | B,s)',
        'P_base':'pbase_{modality}_full = P_hat(Y_U2=1 | B)',
        'E':'max(P_aug - P_base,0); wB * E for ranking',
        'topk_zero_caveat':'Top-k always includes exactly k edge IDs. Active weight > 0 measured separately; if lower than k, do not interpret zeros as routed edges.',
        'data_reuse':'Original cached outputs unchanged. No U3 labels used for score training, binning, or score tuning.',
        'evaluation':'U3 recurrence, >=2 user recurrence, normalized strength (proxy, NOT HR/NDCG)',
        'limitations':['Only seed42 without repeats','Parent-local vs global model fallback inherited',
                       'Behavior-only P_base baseline can reveal behavior covariate effects',
                       'B-matched Raw uses E quota per stratum and is not a standalone learned routing rule',
                       'Scores across methods have different scales; top-k ranking comparison only',
                       'Multiple edges share items; no independent-edge statistical claim'],
        'prior_threshold_included':args.include_prior_threshold,
    }
    (args.out/'comparison_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('SUCCESS:',args.out.resolve(),flush=True)


if __name__=='__main__':
    main()
