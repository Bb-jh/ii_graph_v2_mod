#!/usr/bin/env python3
"""Conditional permutation diagnostic for U_T and U_V routing.

Reuses U1 candidate edges and U2/U3 from routing experiment; U3 NEVER used in fit.
The previous artifacts are EDGE-fold OOF (3), not item-disjoint 5-fold. The
present heldout-U3 permutation test does not depend on those OOF predictions.
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from run_routing_e_ablation import read_cache, features, fit_pair, top_indices
from run_shared_unique_validity import joint_features

def log(*a):print(time.strftime('%Y-%m-%d %H:%M:%S'), *a, flush=True)
def save_csv(p,rows):
    if not rows:return
    with open(p,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def strata_codes(b,bins):
    b=np.asarray(b,dtype=float)
    codes=[]; shape=[]
    for j in range(b.shape[1]):
        cuts=np.unique(np.quantile(b[:,j],np.linspace(0,1,bins+1)[1:-1]))
        codes.append(np.searchsorted(cuts,b[:,j],side='right'))
        shape.append(len(cuts)+1)
    return np.ravel_multi_index(codes,shape)

def shuffled_within(s,strata,rng):
    z=np.asarray(s).copy()
    for c in np.unique(strata):
        ix=np.flatnonzero(strata==c)
        if len(ix)>1:z[ix]=s[rng.permutation(ix)]
    return z

def predict(base,t,v,y,group,C,min_positive,global_only,change):
    all_idx=np.arange(len(y))
    pB,pTV,local_count=fit_pair(base,joint_features(base,t,v),y,all_idx,all_idx,C,min_positive,group,not global_only)
    if change=='T':
        _,pSingle,_=fit_pair(base,features(base,v)[1],y,all_idx,all_idx,C,min_positive,group,not global_only)
        # Permute T, estimate conditional T contribution = P(B,T_perm,V)-P(B,V)
    else:
        _,pSingle,_=fit_pair(base,features(base,t)[1],y,all_idx,all_idx,C,min_positive,group,not global_only)
        # Permute V, estimate conditional V contribution = P(B,T,V_perm)-P(B,T)
    return pTV-pSingle, pTV, local_count

def scores_all(d, comp, u, q):
    wb=np.asarray(d['wb'],float); n=len(wb);k=max(1,int(np.floor(n*q)))
    primary=wb*np.maximum(u,0)
    idx=top_indices(primary,k)
    return idx,primary,int(np.count_nonzero(primary>0))

def metric(d,idx):
    return dict(u3_recurrence=float(np.mean(d['y3'][idx])),
                u3_norm_strength=float(np.mean(d['strength3'][idx])),
                u3_recurrence_2users=float(np.mean(d['y3_2users'][idx])))

def boot_delta(d,idx_a,idx_b,rng,reps):
    # Paired fixed-selection edge-level bootstrap. Network edges share items/users;
    # it is a descriptive CI, NOT valid item/user cluster-robust inference.
    ia=np.asarray(idx_a);ib=np.asarray(idx_b)
    union=np.union1d(ia,ib)
    pos_a=np.searchsorted(union,np.sort(ia));pos_b=np.searchsorted(union,np.sort(ib))
    y=np.asarray(d['y3'][union],float);z=np.asarray(d['strength3'][union],float)
    diffr=[];diffs=[]
    for _ in range(reps):
        w=rng.poisson(1,len(union)).astype(float)
        wa=w[pos_a];wb=w[pos_b]
        if wa.sum()==0 or wb.sum()==0:continue
        diffr.append((wa @ y[pos_a]/wa.sum()-wb @ y[pos_b]/wb.sum())*100)
        diffs.append(wa @ z[pos_a]/wa.sum()-wb @ z[pos_b]/wb.sum())
    return dict(ci_rec_low_pp=float(np.percentile(diffr,2.5)),
                ci_rec_high_pp=float(np.percentile(diffr,97.5)),
                ci_strength_low=float(np.percentile(diffs,2.5)),
                ci_strength_high=float(np.percentile(diffs,97.5)))

def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    d=read_cache(args.input/'routing_candidates.npz')
    n=len(d['wb']);b=np.asarray(d['bfeatures'],float);y=np.asarray(d['y2'],int)
    par=np.asarray(d['parent'])
    if n!=len(y) or any(len(d[k])!=n for k in ('st','sv','y3','strength3','parent')):
        raise RuntimeError('Candidate shape mismatch')
    strata=strata_codes(b,args.bins)
    scount=np.unique(strata,return_counts=True)[1]
    log('N edges',n,'strata',len(scount),'singleton strata',int((scount==1).sum()),'min/median',scount.min(),np.median(scount))
    # Baseline re-fit using EXACT same models as the permutation refits.
    # This prevents comparing different model configurations across null/real.
    observed={};selected={};rows=[];baselines=[];rankings={};otherpred={}
    original_ptv=None
    for comp in ('T','V'):
        u,pjoint,loc=predict(b,d['st'],d['sv'],y,par,args.C,args.min_positive,args.global_only,comp)
        observed[comp]=u
        reference=args.reference
        if reference is None:
            guess=args.input.parent.parent.parent/'motivation_shared_unique'/args.input.name/'shared_unique_scores.npz'
            if guess.exists():reference=guess
        if reference is not None and reference.exists():
            with np.load(reference,allow_pickle=False) as z:
                key='UT_signed' if comp=='T' else 'UV_signed'
                if key not in z:raise RuntimeError('Reference missing '+key)
                err=float(np.max(np.abs(u-z[key])))
                log(comp,'reference max absolute difference',err)
                if err>1e-4:raise RuntimeError('Refit differs from prior U scores; verify model settings / cache alignment')
        otherpred[comp]=pjoint
        if original_ptv is None:original_ptv=pjoint
        else:
            if not np.allclose(original_ptv,pjoint,atol=1e-10):raise RuntimeError("Inconsistent joint predictions")
        log(comp,'original refit complete; local joint predictions',loc,'positive U fraction',float(np.mean(u>0)))
        for q in args.q:
            idx,score,npos=scores_all(d,comp,u,q);selected[comp,q]=idx
            rankings[comp,q]=score
            r=dict(dataset=args.dataset,seed=args.seed,component='U_'+comp,modality=comp,q=q,
                   method='observed',permutation=-1,k=len(idx),active_edges=int(np.sum(score[idx]>0)),
                   positive_pool=npos,**metric(d,idx))
            rows.append(r)
            _,pbase,_=fit_pair(b,b,y,np.arange(n),np.arange(n),args.C,args.min_positive,par,not args.global_only)
            stv=d['st'] if comp=='T' else d['sv']
            raw_score=d['wb']*np.maximum(stv,0)
            for meth,raw in [('Raw_wB_s',raw_score),('Behavior_wB',d['wb']),
                             ('Total_reliability_P_aug',pjoint),('Behavior_only_P_base',pbase)]:
                jx=top_indices(raw,len(idx))
                baselines.append(dict(dataset=args.dataset,seed=args.seed,component='U_'+comp,q=q,
                                      method=meth,k=len(jx),**metric(d,jx)))
            # B-matched Raw: reproduce U's selected behavior-stratum counts;
            # choose highest raw semantic score within each stratum.
            matched=[]
            for h in np.unique(strata[idx]):
                count=int(np.sum(strata[idx]==h))
                pool=np.flatnonzero(strata==h)
                matched.extend(pool[np.argsort(-raw_score[pool],kind='stable')[:count]])
            matched=np.asarray(matched,dtype=int)
            if len(matched)!=len(idx):raise RuntimeError('B-matched budget mismatch')
            baselines.append(dict(dataset=args.dataset,seed=args.seed,component='U_'+comp,q=q,
                      method='B_matched_Raw_score',k=len(matched),**metric(d,matched)))
    rng=np.random.default_rng(args.seed+args.permutation_seed_offset)
    indices={}
    for comp in ('T','V'):
        for it in range(args.repeats):
            if comp=='T':
                t=shuffled_within(d['st'],strata,rng);v=d['sv'];changed=float(np.mean(t!=d['st']))
            else:
                v=shuffled_within(d['sv'],strata,rng);t=d['st'];changed=float(np.mean(v!=d['sv']))
            u,_,loc=predict(b,t,v,y,par,args.C,args.min_positive,args.global_only,comp)
            for q in args.q:
                idx,score,npos=scores_all(d,comp,u,q)
                indices[comp,q,it]=idx
                rows.append(dict(dataset=args.dataset,seed=args.seed,component='U_'+comp,modality=comp,
                     q=q,method='permuted',permutation=it,k=len(idx),
                     active_edges=int(np.sum(score[idx]>0)),positive_pool=npos,**metric(d,idx)))
            log(comp,'permutation',it+1,'/',args.repeats,'changed fraction',f'{changed:.2%}',
                'parent-local',loc, 'U positive',f'{np.mean(u>0):.2%}')
            save_csv(args.out/'permutation_all.csv',rows)
    save_csv(args.out/'routing_baselines.csv',baselines)
    # Top-q Jaccard diagnostics (selection overlap, NOT feature attribution proof).
    overlaps=[]
    wb=np.asarray(d['wb'],float)
    pbase_ref=fit_pair(b,b,y,np.arange(n),np.arange(n),args.C,args.min_positive,par,not args.global_only)[0]
    def overlap(a,bb):
        c=len(np.intersect1d(a,bb,assume_unique=True));k=len(a)
        return c,c/(2*k-c),c/k
    for q in args.q:
        k=max(1,int(np.floor(n*q)))
        uvidx=selected['V',q]
        for label,score in [('Raw_V_cosine',d['sv']),('Raw_V_wB_cosine',wb*np.maximum(d['sv'],0)),
                  ('UT_wB',rankings['T',q]),('Pbase',pbase_ref)]:
            ix=top_indices(score,k);cnt,j,frac=overlap(uvidx,ix)
            overlaps.append(dict(dataset=args.dataset,seed=args.seed,q=q,comparison='UV_vs_'+label,
                                 overlap_edges=cnt,jaccard=j,overlap_over_top=frac,k=k))
    save_csv(args.out/'overlap_diagnostics.csv',overlaps)
    # Degree-bin performance is descriptive among globally selected edges.
    # It is NOT a per-bin equal-budget test.
    degbin=strata_codes(b[:,:2],3)
    degree_rows=[]
    for comp in ('T','V'):
        for q in args.q:
            for meth,ix in [('U_'+comp,selected[comp,q]),('Raw_wB_s',top_indices(wb*np.maximum(d['st'] if comp=='T' else d['sv'],0),len(selected[comp,q])))]:
                for h in np.unique(degbin):
                    ii=ix[degbin[ix]==h]
                    if len(ii):degree_rows.append(dict(dataset=args.dataset,seed=args.seed,modality=comp,q=q,method=meth,
                                  degree_stratum=int(h),n_selected=len(ii),**metric(d,ii)))
    save_csv(args.out/'degree_strata_descriptive.csv',degree_rows)
    summaries=[]
    for comp in ('T','V'):
        for q in args.q:
            rec=[r for r in rows if r['component']=='U_'+comp and r['q']==q]
            obs=next(r for r in rec if r['method']=='observed')
            null=[r for r in rec if r['method']=='permuted']
            dif=np.array([obs['u3_recurrence']-r['u3_recurrence'] for r in null])*100
            strength=np.array([obs['u3_norm_strength']-r['u3_norm_strength'] for r in null])
            # Exact Monte Carlo one-sided p is >= 1/(n_perm + 1), NOT p=0.
            p=(1+sum(r['u3_recurrence']>=obs['u3_recurrence'] for r in null))/(len(null)+1)
            # Bootstrap of observed minus mean-permutation outcomes using paired edge draws;
            # sampled null selection is held fixed for each draw then averaged.
            br=[];bs=[];boot_rng=np.random.default_rng(args.seed*1000+int(q*100)+ord(comp))
            # Need selected null indices; recompute deterministic permutation sequence to avoid saving large arrays?
            # Instead store a second lightweight index archive in loop below, persisted as npz.
            null_idx=[indices[comp,q,it] for it in range(args.repeats)]
            for _ in range(args.bootstrap):
                # independent Poisson edge weights on the shared candidate universe
                # rather than independent resampling of selected sets.
                w=boot_rng.poisson(1,n)
                oi=selected[comp,q];wo=w[oi].astype(float)
                if wo.sum()==0:continue
                r_obs=np.dot(wo,d['y3'][oi])/wo.sum()
                s_obs=np.dot(wo,d['strength3'][oi])/wo.sum()
                nr=[];ns=[]
                for ix in null_idx:
                    ww=w[ix].astype(float)
                    if ww.sum():
                        nr.append(np.dot(ww,d['y3'][ix])/ww.sum());ns.append(np.dot(ww,d['strength3'][ix])/ww.sum())
                if nr:
                    br.append(100*(r_obs-np.mean(nr)));bs.append(s_obs-np.mean(ns))
            summaries.append(dict(dataset=args.dataset,seed=args.seed,component='U_'+comp,q=q,k=obs['k'],
                observed_u3_rec=obs['u3_recurrence'],null_u3_rec_mean=float(np.mean([r['u3_recurrence'] for r in null])),
                delta_rec_pp=float(dif.mean()),null_rec_sd_pp=float(np.std([r['u3_recurrence']*100 for r in null],ddof=1)),
                observed_u3_strength=obs['u3_norm_strength'],
                null_u3_strength_mean=float(np.mean([r['u3_norm_strength'] for r in null])),
                delta_strength=float(strength.mean()),
                bootstrap_ci_rec_low_pp=float(np.percentile(br,2.5)),
                bootstrap_ci_rec_high_pp=float(np.percentile(br,97.5)),
                bootstrap_ci_strength_low=float(np.percentile(bs,2.5)),
                bootstrap_ci_strength_high=float(np.percentile(bs,97.5)),
                permutation_p_one_sided=float(p),permutations=len(null),
                observed_active=obs['active_edges'],
                null_active_min=min(r['active_edges'] for r in null)))
    save_csv(args.out/'permutation_summary.csv',summaries)
    (args.out/'manifest.json').write_text(json.dumps(dict(dataset=args.dataset,seed=args.seed,methods='full U2 fit U3 heldout',
        N=n,repeats=args.repeats,bootstrap=args.bootstrap,bins=args.bins,
        permutation='shuffle modality cosine within quantile strata of 3 U1 behavior features; retrain joint and conditional models',
        clipping='signed conditional contribution retained until routing, then max(x,0)',
        caveat='Existing OOF was EDGE 3-fold, not item-disjoint 5-fold; this test uses heldout U3, no U3 labels for training. Edge bootstrap ignores graph dependence. 10 permutations min attainable empirical p=1/11.'),indent=2,ensure_ascii=False)+'\n')
    log('SUCCESS',args.out)
    for r in summaries:log(r['component'],'q',r['q'],'observed',f"{r['observed_u3_rec']:.4f}",
         'null mean',f"{r['null_u3_rec_mean']:.4f}",'delta pp',f"{r['delta_rec_pp']:+.3f}",
         'perm p',r['permutation_p_one_sided'],'bootstrap CI',
         (round(r['bootstrap_ci_rec_low_pp'],3),round(r['bootstrap_ci_rec_high_pp'],3)))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=None,help='Optional prior shared_unique_scores.npz')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dataset',default='Industrial_and_Scientific')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--q',nargs='+',type=float,default=[.1,.2,.3])
    p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--bins',type=int,default=4)
    p.add_argument('--bootstrap',type=int,default=300)
    p.add_argument('--C',type=float,default=1.)
    p.add_argument('--min-positive',type=int,default=5)
    p.add_argument('--global-only',action='store_true')
    p.add_argument('--permutation-seed-offset',type=int,default=90210)
    a=p.parse_args()
    if a.repeats<1 or a.bootstrap<20 or a.bins<2: p.error('invalid repeats/bootstrap/bins')
    run(a)
if __name__=='__main__':main()
