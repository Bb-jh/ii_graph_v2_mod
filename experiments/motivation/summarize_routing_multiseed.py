#!/usr/bin/env python3
"""Summarize repeated user-disjoint routing evaluations without assuming independent seeds."""
import argparse
from pathlib import Path
import pandas as pd

METHODS = [
    'Raw_wB_times_s', 'Raw_plus_threshold_prior', 'Expressibility_wB_times_E',
    'Total_reliability_P_aug', 'Behavior_only_wB', 'Behavior_only_P_base',
    'B_matched_Raw_s_given_E_strata'
]
COMPARISONS = [
    ('E_minus_threshold','Expressibility_wB_times_E','Raw_plus_threshold_prior'),
    ('E_minus_Bmatched','Expressibility_wB_times_E','B_matched_Raw_s_given_E_strata'),
    ('E_minus_raw','Expressibility_wB_times_E','Raw_wB_times_s'),
    ('Paug_minus_Pbase','Total_reliability_P_aug','Behavior_only_P_base'),
    ('Paug_minus_E','Total_reliability_P_aug','Expressibility_wB_times_E'),
]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('outputs/motivation_routing_reliability'))
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    p.add_argument('--out',type=Path,default=Path('outputs/motivation_routing_reliability/multiseed_summary'))
    args=p.parse_args()
    all_rows=[]
    missing=[]
    for seed in args.seeds:
        path=args.root/f'industrial_split{seed}'/'routing_reliability_comparison.csv'
        if not path.is_file():
            missing.append(str(path));continue
        df=pd.read_csv(path)
        for col in ['modality','q','method','u3_recurrence','u3_norm_strength','selected_edges','active_edges']:
            if col not in df.columns:raise ValueError(f'{path}: missing {col}')
        if df.duplicated(['modality','q','method']).any():raise ValueError(f'{path}: duplicated method/q')
        expected={(m,q,method) for m in ('T','V') for q in (.1,.2,.3) for method in METHODS}
        observed={(str(r.modality),round(float(r.q),5),str(r.method)) for r in df.itertuples()}
        if expected != observed:
            raise ValueError(f'{path}: missing {expected-observed}, extra {observed-expected}')
        for (m,q),group in df.groupby(['modality','q']):
            if len(set(group['selected_edges']))!=1: raise ValueError(f'{path}: unequal edge budget {m} {q}')
            if (group['active_edges']!=group['selected_edges']).any():
                print(f'WARNING seed={seed} {m} q={q}: some selected edges inactive')
        df['seed']=seed
        all_rows.append(df)
    if missing:raise FileNotFoundError('Missing complete seed results:\n'+'\n'.join(missing))
    full=pd.concat(all_rows,ignore_index=True)
    args.out.mkdir(parents=True,exist_ok=True)
    full.to_csv(args.out/'all_seed_results.csv',index=False)
    methods=(full.groupby(['modality','q','method'],as_index=False)
             .agg(n_seeds=('seed','nunique'),mean_recurrence=('u3_recurrence','mean'),sd_recurrence=('u3_recurrence','std'),
                  mean_strength=('u3_norm_strength','mean'),sd_strength=('u3_norm_strength','std'),
                  mean_active=('active_edges','mean')))
    methods.to_csv(args.out/'method_mean_sd.csv',index=False)
    delta_rows=[]
    for (seed,m,q),g in full.groupby(['seed','modality','q']):
        data=g.set_index('method')
        for name,target,control in COMPARISONS:
            delta_rows.append(dict(seed=int(seed),modality=m,q=float(q),comparison=name,
                delta_recurrence_pp=100*(data.loc[target,'u3_recurrence']-data.loc[control,'u3_recurrence']),
                delta_strength=data.loc[target,'u3_norm_strength']-data.loc[control,'u3_norm_strength']))
    deltas=pd.DataFrame(delta_rows)
    deltas.to_csv(args.out/'per_seed_deltas.csv',index=False)
    summary=(deltas.groupby(['modality','q','comparison'],as_index=False)
             .agg(n_seeds=('seed','nunique'),mean_delta_pp=('delta_recurrence_pp','mean'),sd_delta_pp=('delta_recurrence_pp','std'),
                  positive_seeds=('delta_recurrence_pp',lambda s:int((s>0).sum())),
                  mean_delta_strength=('delta_strength','mean'),sd_delta_strength=('delta_strength','std')))
    summary.to_csv(args.out/'delta_mean_sd.csv',index=False)
    print('SUCCESS:',args.out)
    print('\n=== q=20% DELTAS (mean +/- sample SD, percentage points) ===')
    x=summary[summary.q.round(5)==.2]
    for r in x.itertuples():
        print(f'{r.modality:<2} {r.comparison:<24} {r.mean_delta_pp:+.3f} +/- {r.sd_delta_pp:.3f} pp; positive {r.positive_seeds}/{r.n_seeds}')
    print('\nNOTE: five splits overlap on the same dataset; do not treat as independent experiments.')

if __name__=='__main__':main()
