#!/usr/bin/env python3
"""Held-out U3: U vs EXACT U-stratum quota-matched cosine, and wB vs P(Y|B).
Read-only cached edge inputs. Uses identical top-q budgets, no U3 in selection.
Outputs degree/coverage, paired item Poisson bootstrap and optional U2-label-free
within-stratum raw ranking randomization (NOT a conditional-U model refit).
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

def load(p):
    if not p.is_file(): raise FileNotFoundError(p)
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def csvout(p,rows):
    if not rows:return
    with p.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def top(s,k):
    s=np.asarray(s,float)
    if np.any(~np.isfinite(s)):raise ValueError('Nonfinite selection scores')
    return np.lexsort((np.arange(len(s)),-s))[:k]
def bins_from_bfeatures(b,n_bins):
    # Must match run_conditional_u_permutation.py (np.searchsorted(..., side='right'))
    b=np.asarray(b,float)
    if b.ndim!=2 or b.shape[1]!=3:raise ValueError('Expected 3 U1 log-degree/coocc features')
    code=[];shapes=[];cuts_all=[]
    for j in range(3):
        cuts=np.unique(np.quantile(b[:,j],np.linspace(0,1,n_bins+1)[1:-1]))
        code.append(np.searchsorted(cuts,b[:,j],side='right'))
        shapes.append(len(cuts)+1);cuts_all.append(cuts.tolist())
    return np.ravel_multi_index(code,shapes).astype(np.int32),cuts_all

def matched_raw_cosine(s,groups,reference_idx):
    selected=[];quota=[]
    values,counts=np.unique(groups[reference_idx],return_counts=True)
    for h,c in zip(values,counts):
        pool=np.flatnonzero(groups==h)
        chosen=pool[np.lexsort((pool,-np.asarray(s[pool],float)))[:int(c)]]
        selected.append(chosen);quota.append((int(h),int(c),len(pool)))
    idx=np.concatenate(selected).astype(int)
    if len(idx)!=len(reference_idx) or len(np.unique(idx))!=len(idx):raise AssertionError('strata mismatch')
    return idx,quota

def metric(d,idx,eligible_items):
    a,b=d['a'][idx].astype(int),d['b'][idx].astype(int)
    touched=np.unique(np.r_[a,b])
    return dict(u3_recurrence=float(np.mean(d['y3'][idx])),
       u3_recurrence_2users=float(np.mean(d['y3_2users'][idx])),
       u3_norm_strength=float(np.mean(d['strength3'][idx])),
       item_coverage=float(len(touched)/eligible_items),n_unique_items=len(touched),
       mean_log_min_degree=float(np.mean(d['bfeatures'][idx,0])),
       mean_log_max_degree=float(np.mean(d['bfeatures'][idx,1])),
       mean_log_u1_coocc=float(np.mean(d['bfeatures'][idx,2])),
       mean_wB=float(np.mean(d['wb'][idx])))

def item_bootstrap(d,ix1,ix2,reps,seed):
    # Poisson(1) item resampling, induced edge multiplicity w_i*w_j. A pair is
    # represented only when both endpoints are sampled. Induced-subgraph
    # bootstrap is approximate, and does not capture shared users.
    if reps<=0:return {}
    a=np.asarray(d['a'],int);b=np.asarray(d['b'],int)
    n=int(max(a.max(),b.max())+1);rng=np.random.default_rng(seed)
    yi=np.asarray(d['y3'],float);si=np.asarray(d['strength3'],float)
    dr=[];ds=[];skipped=0
    for _ in range(reps):
        iw=rng.poisson(1,n).astype(float)
        wa=iw[a[ix1]]*iw[b[ix1]];wb=iw[a[ix2]]*iw[b[ix2]]
        if not wa.sum() or not wb.sum():skipped+=1;continue
        dr.append(100*(np.dot(wa,yi[ix1])/wa.sum()-np.dot(wb,yi[ix2])/wb.sum()))
        ds.append(np.dot(wa,si[ix1])/wa.sum()-np.dot(wb,si[ix2])/wb.sum())
    if len(dr)<max(20,reps//2):raise RuntimeError('Too few bootstrap samples')
    return dict(item_boot_ci_rec_low_pp=float(np.quantile(dr,.025)),
                item_boot_ci_rec_high_pp=float(np.quantile(dr,.975)),
                item_boot_ci_strength_low=float(np.quantile(ds,.025)),
                item_boot_ci_strength_high=float(np.quantile(ds,.975)),
                boot_valid=len(dr),boot_skipped=skipped)

def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    d=load(args.input/'routing_candidates.npz')
    u=load(args.u_scores)
    e=load(args.input/'expressibility_scores.npz')
    n=len(d['wb']); needed=('a','b','wb','st','sv','bfeatures','y3','strength3','y3_2users')
    if any(key not in d for key in needed):raise KeyError('Missing candidate fields: '+str([x for x in needed if x not in d]))
    for key in needed:
        if len(d[key])!=n:raise ValueError('Edge count mismatch: '+key)
    for key in ('UT_signed','UV_signed'):
        if key not in u or len(u[key])!=n:raise ValueError('Missing/misaligned '+key)
    if not np.array_equal(d.get('a',[]),u.get('a',d['a'])) or not np.array_equal(d.get('b',[]),u.get('b',d['b'])):
        raise ValueError('Saved U score candidate edge alignment mismatch')
    manifest_path=args.input/'prepare_manifest.json'
    pm=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    universe=int(pm.get('eligible_items',max(np.max(d['a']),np.max(d['b']))+1))
    groups,cuts=bins_from_bfeatures(d['bfeatures'],args.bins)
    # bfeatures are log1p(min U1 endpoint degree, max U1 endpoint degree, U1 co-occ)
    # Stratified degree group for descriptive evaluation: low/mid/high by min endpoint degree.
    degcuts=np.unique(np.quantile(d['bfeatures'][:,0],[1/3,2/3]))
    edge_degree_group=np.searchsorted(degcuts,d['bfeatures'][:,0],side='right')
    metrics=[];deltas=[];quotas=[];degree=[];selections={}
    scores={'Behavior_wB':np.asarray(d['wb'],float),
            'Behavior_Pbase':np.asarray(e['pbase_T_full'],float),
            'Rank_average_wB_Pbase':None}
    if not np.allclose(e['pbase_T_full'],e['pbase_V_full'],atol=1e-5):
        print('WARNING: T/V Pbase predictions differ. Using T reference for behavior-only.')
    # Rank average is a candidate ranking fusion (not calibrated weighted value)
    def ranks(x):
        r=np.empty(n,float);r[top(x,n)]=np.arange(n,dtype=float);return 1.0-r/max(1,n-1)
    scores['Rank_average_wB_Pbase']=.5*(ranks(scores['Behavior_wB'])+ranks(scores['Behavior_Pbase']))
    for q in args.q:
        k=int(np.floor(n*q))
        if not 1<=k<n:raise ValueError('Invalid q '+str(q))
        for m in ('T','V'):
            signed=np.asarray(u['U'+m+'_signed'],float)
            xscore=np.asarray(d['wb'],float)*np.maximum(signed,0)
            original=top(xscore,k)
            if int(np.sum(xscore[original]>0))<k:
                print('WARNING: U_'+m+' q='+str(q)+' contains zero-score routed edges')
            raw_s=np.asarray(d['st' if m=='T' else 'sv'],float)
            raw_w=np.asarray(d['wb'],float)*np.maximum(raw_s,0)
            matched,quota=matched_raw_cosine(raw_s,groups,original)
            raw=top(raw_w,k)
            comps={'U_'+m:(original,xscore),'Binmatched_RawCosine_'+m:(matched,raw_s),
                   'Raw_wB_gs_'+m:(raw,raw_w)}
            for name,score in scores.items():comps[name]=(top(score,k),score)
            for name,(ix,score) in comps.items():
                selections[f'{m}_q{int(q*100)}_{name}']=ix.astype(np.int32)
                metrics.append(dict(dataset=args.dataset,seed=args.seed,modality=m,q=q,k=k,method=name,
                     selected_edges=len(ix),active_edges=int(np.sum(np.asarray(score)[ix]>0)),
                     **metric(d,ix,universe)))
                for h in np.unique(edge_degree_group):
                    ixh=ix[edge_degree_group[ix]==h]
                    if len(ixh):
                        degree.append(dict(dataset=args.dataset,seed=args.seed,modality=m,q=q,method=name,
                            degree_group=['Low','Mid','High'][int(h)],selected_edges=len(ixh),
                            selection_fraction=float(len(ixh)/k),**metric(d,ixh,universe)))
            for h,c,poolsz in quota:quotas.append(dict(dataset=args.dataset,seed=args.seed,modality=m,q=q,stratum=h,
                                               quota=c,pool_size=poolsz))
            for other in ('Binmatched_RawCosine_'+m,'Raw_wB_gs_'+m):
                target=comps[other][0]
                oa=metric(d,original,universe);ob=metric(d,target,universe)
                boot=item_bootstrap(d,original,target,args.bootstrap,args.seed*10000+int(q*100)*13+ord(m)+len(other))
                deltas.append(dict(dataset=args.dataset,seed=args.seed,modality=m,q=q,k=k,
                   comparison='U_'+m+'_minus_'+other,delta_rec_pp=100*(oa['u3_recurrence']-ob['u3_recurrence']),
                   delta_strength=oa['u3_norm_strength']-ob['u3_norm_strength'],
                   delta_coverage=oa['item_coverage']-ob['item_coverage'],**boot))
            print(f'{m} q={q:.1f} U={100*metric(d,original,universe)["u3_recurrence"]:.3f}% '
                  f'bin-raw={100*metric(d,matched,universe)["u3_recurrence"]:.3f}% '
                  f'coverage wB={metric(d,comps["Behavior_wB"][0],universe)["item_coverage"]:.3f} '
                  f'Pbase={metric(d,comps["Behavior_Pbase"][0],universe)["item_coverage"]:.3f}',flush=True)
    csvout(args.out/'routing_comparison.csv',metrics)
    csvout(args.out/'routing_deltas_item_boot.csv',deltas)
    csvout(args.out/'degree_strata.csv',degree)
    csvout(args.out/'stratum_quotas.csv',quotas)
    np.savez_compressed(args.out/'selection_indices.npz',**selections)
    (args.out/'manifest.json').write_text(json.dumps(dict(dataset=args.dataset,seed=args.seed,n_edges=n,
      q=args.q,bins=args.bins,behavior_quantile_cuts_log_features=cuts,
      bin_definition="Each U1 log1p(min degree), log1p(max degree), log1p(U1 cooccurrence) quantile; searchsorted side=right; independent 3D joint strata",
      matched_definition="For each U_m top-k, match exact quota per stratum. Within each stratum choose HIGHEST RAW COSINE s^m, not wB*s^m. This comparator depends on U_m stratum counts.",
      prior_baselines="Earlier routing_reliability_compare: E top-k quota and raw cosine, bins side=left. Earlier conditional_u_permutation: U top-k quota but wB*max(cosine,0) within bin and side=right. Current: U quota, raw cosine, side=right.",
      item_coverage_denominator=universe,degree_grouping='min U1 degree edge quantile; selected edges grouped post selection',
      item_bootstrap='Poisson item multiplicity w[a]*w[b] over fixed U3; approximate induced-subgraph paired CI; no user-resampling or model-training uncertainty',
      no_leakage='No U3 label used to rank/select/tune; U3 used for evaluation only',
      folds_caveat='Historical cache used edge 3-fold OOF, NOT item-disjoint 5-fold; stored U full-fit on U2 is evaluated on disjoint U3.'),indent=2,ensure_ascii=False)+'\n')
    print('SUCCESS',args.out,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--u-scores',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dataset',default='Industrial_and_Scientific')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--q',type=float,nargs='+',default=[.1,.2,.3])
    p.add_argument('--bins',type=int,default=4)
    p.add_argument('--bootstrap',type=int,default=500)
    a=p.parse_args()
    if a.bootstrap<0 or a.bins<2:p.error('bad bootstrap/bins')
    run(a)
if __name__=='__main__':main()
