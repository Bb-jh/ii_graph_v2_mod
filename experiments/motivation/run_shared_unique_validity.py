#!/usr/bin/env python3
"""S/UT/UV U3-validity diagnostic using frozen U1 candidate edges and U2 fits.

Never trains on U3. Distinguishes signed decomposition from routing [x]+.
NOT a downstream RQ-VAE / recommendation evaluation.
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import brier_score_loss
from run_routing_e_ablation import read_cache, features, fit_pair, top_indices


def log(*x): print(time.strftime('%Y-%m-%d %H:%M:%S'), *x, flush=True)

def dump_csv(path, rows):
    if not rows:return
    with path.open('w', newline='', encoding='utf-8') as f:
        wr=csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

def joint_features(base,st,sv):
    b=np.asarray(base, dtype=np.float64)
    t=np.asarray(st,dtype=np.float64).reshape(-1,1)
    v=np.asarray(sv,dtype=np.float64).reshape(-1,1)
    # Nested feature set: behavior + both modalities + same interactions as single-view fits
    # plus direct T*V interaction; avoids claiming an unrestricted conditional estimator.
    return np.column_stack((b,t,v,b*t,b*v,t*v))

def strata_features(base,bins):
    """Quantile bins for each of the 3 existing U1 behavior predictors."""
    b=np.asarray(base,dtype=np.float64)
    z=[]
    for j in range(b.shape[1]):
        cuts=np.unique(np.quantile(b[:,j],np.linspace(0,1,bins+1)[1:-1]))
        z.append(np.searchsorted(cuts,b[:,j],side='right'))
    return np.ravel_multi_index(z, tuple([bins]*b.shape[1]))

def matched_random(ref_idx, strata, seed):
    """Keep selected counts by behavior-feature strata, randomize identities in strata.
    No replacement; overlaps with reference possible and are reported.
    """
    rng=np.random.default_rng(seed)
    chosen=[]
    for h in np.unique(strata[ref_idx]):
        n=int(np.count_nonzero(strata[ref_idx]==h))
        pool=np.flatnonzero(strata==h)
        chosen.append(rng.choice(pool,size=n,replace=False))
    out=np.concatenate(chosen)
    assert len(out)==len(ref_idx) and len(np.unique(out))==len(out)
    return out

def measure(d,idx,score,component,method,q,k,rep=None):
    a=np.asarray(idx,dtype=int)
    positives=(score[a]>0)
    result=dict(dataset='',seed='',component=component,modality=component if component in ('UT','UV') else 'shared',
       q=q,k=k,method=method,random_repeat='' if rep is None else rep,
       selected_edges=len(a),active_edges=int(positives.sum()),active_fraction=float(positives.mean()),
       u2_recurrence=float(d['y2'][a].mean()),u3_recurrence=float(d['y3'][a].mean()),
       u3_recurrence_2users=float(d['y3_2users'][a].mean()),
       u3_norm_strength=float(d['strength3'][a].mean()),
       u3_recurrence_active_only=float(d['y3'][a[positives]].mean()) if positives.any() else float('nan'),
       mean_selected_u1_weight=float(d['wb'][a].mean()),
       mean_selected_text_similarity=float(d['st'][a].mean()),
       mean_selected_image_similarity=float(d['sv'][a].mean()))
    return result

def stats_signed(x,comp):
    a=np.asarray(x,float)
    return dict(component=comp,n_edges=len(a),mean=float(a.mean()),median=float(np.median(a)),
       std=float(a.std()),positive_fraction=float(np.mean(a>0)),negative_fraction=float(np.mean(a<0)),
       zero_fraction=float(np.mean(a==0)),positive_mean=float(a[a>0].mean()) if np.any(a>0) else 0.,
       max=float(a.max()),min=float(a.min()))

def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    d=read_cache(args.input/'routing_candidates.npz')
    z=read_cache(args.input/'expressibility_scores.npz')
    n=len(d['wb']); y2=np.asarray(d['y2'],dtype=np.int32)
    for key in ('wb','st','sv','bfeatures','parent','y3','strength3'):
        if len(d[key]) != n:raise ValueError(f'candidate array length mismatch {key}')
    for key in ('pbase_T_full','pbase_V_full','pfull_T_full','pfull_V_full'):
        if key not in z:raise ValueError(f'Missing {key}. Run old routing E --stage fit first.')
    if not np.array_equal(z['pbase_T_full'],z['pbase_V_full']):
        raise RuntimeError('Text and Image behavior-only baseline predictions disagree; cannot consistently decompose.')
    if np.any(~np.isfinite(d['wb'])):raise ValueError('Nonfinite behavior weights')
    pB=np.asarray(z['pbase_T_full'],float)
    pT=np.asarray(z['pfull_T_full'],float)
    pV=np.asarray(z['pfull_V_full'],float)
    b=np.asarray(d['bfeatures'],float)
    joint=joint_features(b,d['st'],d['sv'])
    allidx=np.arange(n)
    # Existing fit_pair learns a shared global logistic model and, where possible,
    # parent-specific models on U2. Same C, min-positive and local override.
    pB_joint,pTV,updated=fit_pair(b,joint,y2,allidx,allidx,args.C,args.min_positive,
                                    np.asarray(d['parent']),not args.global_only)
    max_base_diff=float(np.max(np.abs(pB_joint-pB)))
    if max_base_diff>1e-4:
        raise RuntimeError(f'Inconsistent shared P(Y|B): max delta {max_base_diff:.7g}. Check C/local settings.')
    for key,p in [('pB',pB),('pT',pT),('pV',pV),('pTV',pTV)]:
        if np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):raise ValueError('Bad probability '+key)
    ET=pT-pB; EV=pV-pB; ETV=pTV-pB
    UT=pTV-pV; UV=pTV-pT; S=ET+EV-ETV
    decomposition_error=float(max(
        np.max(np.abs(S+UT-ET)),np.max(np.abs(S+UV-EV)),
        np.max(np.abs(S+UT+UV-ETV))))
    # Verify inclusion-exclusion identities, including S+UT+UV = ETV.
    if decomposition_error>1e-10:raise RuntimeError('Decomposition algebra mismatch')
    wb=np.asarray(d['wb'],float)
    scores={'S':wb*np.maximum(S,0), 'UT':wb*np.maximum(UT,0), 'UV':wb*np.maximum(UV,0),
            'Raw_T':wb*np.maximum(d['st'],0), 'Raw_V':wb*np.maximum(d['sv'],0),
            'Behavior':wb}
    signed={'ET':ET,'EV':EV,'ETV':ETV,'S':S,'UT':UT,'UV':UV}
    stat_rows=[stats_signed(arr,c) for c,arr in signed.items()]
    for r in stat_rows:r.update(dataset=args.dataset,seed=args.seed)
    # Save separate signed and clipped values, allowing later level-conditional analysis.
    np.savez_compressed(args.out/'shared_unique_scores.npz',
        a=d['a'],b=d['b'],parent=d['parent'],wb=d['wb'],st=d['st'],sv=d['sv'],
        pB=pB.astype('f4'),pT=pT.astype('f4'),pV=pV.astype('f4'),pTV=pTV.astype('f4'),
        ET_signed=ET.astype('f4'),EV_signed=EV.astype('f4'),ETV_signed=ETV.astype('f4'),
        S_signed=S.astype('f4'),UT_signed=UT.astype('f4'),UV_signed=UV.astype('f4'),
        S_positive=np.maximum(S,0).astype('f4'), UT_positive=np.maximum(UT,0).astype('f4'),
        UV_positive=np.maximum(UV,0).astype('f4'))
    dump_csv(args.out/'contribution_diagnostics.csv',stat_rows)
    strata=strata_features(d['bfeatures'],args.behavior_bins)
    results=[]; overlaps=[]
    for q in args.q:
        k=max(1,int(np.floor(n*q)))
        chosen={name:top_indices(score,k) for name,score in scores.items()}
        for name,score in scores.items():
            rec=measure(d,chosen[name],score,name,name,q,k)
            rec.update(dataset=args.dataset,seed=args.seed)
            results.append(rec)
        for comp in ('S','UT','UV'):
            baseidx=chosen[comp]
            for i in range(args.random_repeats):
                ix=np.random.default_rng(args.seed*10000+int(q*1000)*100+i).choice(n,k,replace=False)
                rec=measure(d,ix,np.ones(n),comp,'Random',q,k,i)
                rec.update(dataset=args.dataset,seed=args.seed);results.append(rec)
                mi=matched_random(baseidx,strata,args.seed*10000+int(q*1000)*100+i+999)
                rec=measure(d,mi,np.ones(n),comp,'Behavior_strata_matched_random',q,k,i)
                rec.update(dataset=args.dataset,seed=args.seed);results.append(rec)
                overlaps.append(dict(dataset=args.dataset,seed=args.seed,component=comp,q=q,repeat=i,
                     matched_overlap=int(np.intersect1d(mi,baseidx).size),
                     matched_overlap_fraction=float(np.intersect1d(mi,baseidx).size/k)))
        log('q',q,'k',k,'all component and matched/random controls evaluated')
    dump_csv(args.out/'routing_validity.csv',results)
    dump_csv(args.out/'matched_random_overlap.csv',overlaps)
    comparisons=[]
    for q in args.q:
        subset={r['method']:r for r in results if r['q']==q and r['method'] in scores}
        for x,y in [('S','UT'),('S','UV'),('S','Raw_T'),('S','Raw_V'),('S','Behavior'),
                    ('UT','UV'),('UT','Raw_T'),('UV','Raw_V')]:
            a,c=subset[x],subset[y]
            comparisons.append(dict(dataset=args.dataset,seed=args.seed,q=q,comparison=f'{x}_minus_{y}',
                delta_u3_recurrence_pp=100*(a['u3_recurrence']-c['u3_recurrence']),
                delta_u3_norm_strength=a['u3_norm_strength']-c['u3_norm_strength']))
    dump_csv(args.out/'component_deltas.csv',comparisons)
    metadata=dict(dataset=args.dataset,seed=args.seed,input=str(args.input),n_edges=n,
          folds_note='Reuses existing full-U2 pB,pT,pV; fits pTV on U2 only.',
          joint_features='b, text cosine, image cosine, b*text, b*image, text*image',
          local_models=not args.global_only,C=args.C,min_positive=args.min_positive,
          parent_local_joint_predictions=int(updated),parent_local_joint_fraction=float(updated/n),
          pB_consistency_max_abs_error=max_base_diff, algebra_max_abs_error=decomposition_error,
          u2_brier_joint=float(brier_score_loss(y2,pTV)),
          u2_brier_behavior=float(brier_score_loss(y2,pB)),
          caveats=['All probabilities are full-U2 fitted; training metrics on U2 are in-sample.',
           'S is inclusion-exclusion interaction: can be negative, including synergy; not guaranteed shared causal evidence.',
           'Joint model pTV may have extra flexibility and be miscalibrated; heldout U3 is used only for evaluation.',
           'Zero-valued scores may be top-k fillers; active_fraction records effective budget.',
           'All edges are within one behavior-only coarse parent of cached Experiment C; no level analysis here.',
           'Behavior-strata-matched random is diagnostic only, not a propensity-corrected causal effect.',
           'Results measure U3 edge recurrence, not downstream RQ-VAE HR/NDCG.'])
    (args.out/'run_manifest.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    log('SUCCESS',args.out,'N',n, 'local joint',updated,'/',n)
    for q in args.q:
        sub={r['method']:r for r in results if r['q']==q and r['method'] in scores}
        print('q',q,' | '.join(f'{method}={sub[method]["u3_recurrence"]:.5f}(active {sub[method]["active_fraction"]:.1%})' for method in scores),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True,help='Previous motivation_routing_e/industrial_splitNN output directory')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--dataset',default='Industrial_and_Scientific')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--q',nargs='+',type=float,default=[0.1,0.2,0.3])
    ap.add_argument('--C',type=float,default=1.)
    ap.add_argument('--min-positive',type=int,default=5)
    ap.add_argument('--global-only',action='store_true')
    ap.add_argument('--behavior-bins',type=int,default=4)
    ap.add_argument('--random-repeats',type=int,default=3)
    x=ap.parse_args()
    if x.behavior_bins<2 or x.random_repeats<1 or any(q<=0 or q>=1 for q in x.q):ap.error('Invalid bins/repeats/q')
    run(x)
if __name__=='__main__':main()
