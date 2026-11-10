#!/usr/bin/env python3
"""Aggregate multiple seeds per dataset+encoder, pre-register allocation before grid."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
args=p.parse_args();root=Path(args.root);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
metrics=[];perms=[]
for fn in sorted(root.glob('**/utility_metrics.csv')):
    ppath=fn.parent/'permutation_utility.csv'
    if not ppath.exists():continue
    metrics.append(pd.read_csv(fn));perms.append(pd.read_csv(ppath))
if not metrics:raise SystemExit('No completed runs under '+str(root))
m=pd.concat(metrics,ignore_index=True);pval=pd.concat(perms,ignore_index=True)
main=m[m.split=='U2_OOF'].copy();cols=['delta_T','delta_V','u_T','u_V','dll_T','dll_V','dpr_T','dpr_V']
summary=main.groupby(['dataset','encoder'],dropna=False)[cols].agg(['mean','std','count'])
summary.columns=['_'.join(x) for x in summary.columns];summary=summary.reset_index()
# Strict familywise optional user choice not automatically applied; p<.05 per modality per seed
# Multi-seed decision: all 3 seeds independently pass the pre-registered three-part rule.
alloc=[]
for (ds,enc),grp in pval.groupby(['dataset','encoder']):
    original=m[(m.dataset==ds)&(m.encoder==enc)&(m.split=='U2_OOF')]
    decisions={}
    for comp in ('T','V'):
        g=grp[grp.modality==comp]
        passes=(len(g)>=3 and bool((g.significant.astype(str).str.lower()=='true').all()) and
                bool((g.p_one_sided<.05).all()) and bool(g.ci_excludes_zero.astype(str).str.lower().eq('true').all()) and
                bool(g.u3_sign_matches.astype(str).str.lower().eq('true').all()))
        effect=max(0,float(original['u_'+comp].mean())) if passes else 0.
        decisions[comp]=effect
    total=sum(decisions.values());r=decisions['V']/total if total>0 else None
    nV=int(np.floor(4*r+.5)) if r is not None else 2
    alloc.append(dict(dataset=ds,encoder=enc,seeds=int(original.seed.nunique()),
                      n_permutations_min=int(grp.permutations.min()),
                      u_T_accepted=decisions['T'],u_V_accepted=decisions['V'],r_V=r,
                      predicted_L_T=6-nV,predicted_L_V=2+nV,
                      both_zero=(total==0),rule='all 3 seeds: p<.05, CI excludes zero, U3 sign matches U2; nonpositive utility zero; tie 4/4'))
m.to_csv(out/'all_utility_metrics.csv',index=False)
pval.to_csv(out/'all_permutation_metrics.csv',index=False)
summary.to_csv(out/'utility_multiseed_mean_sd.csv',index=False)
allocation_df=pd.DataFrame(alloc)
if len(allocation_df) and bool((allocation_df.seeds>=3).all()):
    allocation_df.to_csv(out/'predicted_allocations_FROZEN.csv',index=False)
else:
    allocation_df.to_csv(out/'predicted_allocations_PROVISIONAL.csv',index=False)
    print('WARNING: fewer than 3 seeds for at least one dataset/encoder; NOT frozen')
print(summary.to_string(index=False));print(pd.DataFrame(alloc).to_string(index=False))
