#!/usr/bin/env python3
"""Paired fixed-candidate AUC audit for entropy-matched Exp2.

Does not train, select hyperparameters, write into the source results, or resample U3 to select models.
The unit of uncertainty is item-cluster (multiplier/Poisson) on fixed U3 edge/nonedge pairs.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'/'discrete_mi_portable_v2'))
import run_level1_v2 as v2
import offdiag_core as core

PAIRS = [('code_mi','infonce'),('code_mi','rq'),('infonce','rq'),('code_mi','kmeans'),('infonce','kmeans')]

def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def auc_fast(y,weights,cache):
    # Same weighted ROC AUC as sklearn, but sorted scores and ties are cached.
    order,starts=cache
    y=y[order]
    w=weights[order]
    wp=np.where(y==1,w,0.0); wn=np.where(y==0,w,0.0)
    p=np.add.reduceat(wp,starts); n=np.add.reduceat(wn,starts)
    totalp=p.sum();totaln=n.sum()
    if totalp<=0 or totaln<=0:return np.nan
    n_before=np.cumsum(n)-n
    return float(np.dot(p,n_before+0.5*n)/(totalp*totaln))

def build_cache(scores):
    # stable ascending ordering and tie blocks
    order=np.argsort(scores,kind='mergesort')
    s=scores[order]
    starts=np.r_[0,1+np.flatnonzero(s[1:]!=s[:-1])]
    return order,starts

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--domain',choices=['Industrial','Musical'],default='Industrial')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--audit-seed',type=int,default=20261111)
    p.add_argument('--source-root',default='outputs/discrete_mi_beta_match_v1/full')
    p.add_argument('--out-root',default='outputs/discrete_mi_auc_paired_audit_v1')
    p.add_argument('--mapping-path',default=None)
    p.add_argument('--embedding-path',default=None)
    p.add_argument('--sequence-path',default=None)
    args=p.parse_args()
    if args.bootstrap<30:raise ValueError('Bootstrap must be >=30')
    dom=f'{args.domain}_K256_seed{args.seed}'
    src=(ROOT/args.source_root/dom).resolve()
    out=(ROOT/args.out_root/dom).resolve()
    if not (src/'DONE.json').is_file():raise FileNotFoundError(f'Missing completed entropy-matched result: {src}')
    done=json.loads((src/'DONE.json').read_text())
    if done.get('status')!='success' or not all(done.get('matched',{}).values()):
        raise RuntimeError('Source not a complete entropy-matched run')
    summary=list(csv.DictReader((src/'summary.csv').open()))
    if len(summary)!=8:raise RuntimeError(f'Expected 8 rows, got {len(summary)}')
    selected_by_mod={mod:json.loads((src/mod/'selection.json').read_text())['selected'] for mod in ('text','image')}
    a=SimpleNamespace(domain=args.domain,mapping_path=args.mapping_path,embedding_path=args.embedding_path,
        sequence_path=args.sequence_path,seed=args.seed,k=256,eval_positives=30000,dry_run=False)
    dat=v2.make_data(a)
    y=dat['u3'][2].astype(np.uint8)
    ea,eb=dat['u3'][:2]
    n=dat['n']
    print('U3 candidates:',len(y),'positive:',int(y.sum()),'negative:',int((y==0).sum()),flush=True)
    if int(y.sum())!=30000 or int((y==0).sum())!=120000:
        raise RuntimeError('U3 candidate counts changed; do not silently compare different evaluations')
    print('U3 PAIR SHA256:',sha(np.column_stack((ea,eb,y))),flush=True)
    if out.exists():raise FileExistsError('Refusing to overwrite results: '+str(out))
    out.mkdir(parents=True)
    outputs=[];score_by_mod={}
    for mod in ('text','image'):
        scores={}
        for method in ('rq','infonce','code_mi','kmeans'):
            r=next(x for x in summary if x['modality']==mod and x['method']==method)
            path=src/mod/f'{method}_codes.npy'
            if not path.is_file():raise FileNotFoundError(f'Missing selected codes: {path}')
            codes=np.load(path,allow_pickle=False)
            if len(codes)!=n:raise ValueError(f'Code vector item mismatch {path}: {len(codes)} vs {n}')
            if method=='kmeans':alpha=float(r['alpha'])
            else:
                sel=selected_by_mod[mod][method]
                alpha=float(sel['alpha'])
                if not np.isclose(float(sel['beta']),float(r['beta'])) or not np.isclose(float(sel['lam']),float(r['lam'])):
                    raise RuntimeError('Selection metadata disagrees with summary')
            e1=dat['edges'][0]
            p1=core.code_joint(codes,*e1[:3],256)*float(e1[2].sum())
            joint=(p1+alpha);joint/=joint.sum()
            s=core.score_pmi(codes,ea,eb,joint)[2]
            auc=float(roc_auc_score(y,s))
            expected=float(r['transition_auc'])
            if not np.isclose(auc,expected,rtol=0,atol=1e-8):
                raise RuntimeError(f'IDENTITY FAIL {mod}/{method}: re-evaluated {auc:.12f} != saved {expected:.12f}. Check data versions/IDs/seed.')
            print(f'IDENTITY PASS {mod}/{method}: AUC {auc:.9f} alpha={alpha}',flush=True)
            scores[method]=s
            outputs.append(dict(modality=mod,method=method,auc=auc,alpha=alpha,codes_sha256=sha(codes)))
        score_by_mod[mod]=scores
    # Draw weights shared across modalities and methods, preserving correlations.
    # Every item has one Poisson multiplier; every U3 candidate has product endpoint weight.
    rng=np.random.default_rng(args.audit_seed)
    caches={(mod,m):build_cache(s) for mod,sc in score_by_mod.items() for m,s in sc.items()}
    obs={(mod,m):float(roc_auc_score(y,s)) for mod,sc in score_by_mod.items() for m,s in sc.items()}
    draws={(mod,l,r):np.zeros(args.bootstrap,dtype=np.float64) for mod in ('text','image') for l,r in PAIRS}
    for t in range(args.bootstrap):
        item_mult=rng.poisson(1,size=n)
        w=(item_mult[ea].astype(np.float64)*item_mult[eb])
        if w[y==1].sum()==0 or w[y==0].sum()==0:raise RuntimeError('Empty positive/negative bootstrap replicate')
        au={key:auc_fast(y,w,cache) for key,cache in caches.items()}
        if t==0:
            # Independent identity check of cached weighted AUC implementation, with all-ones weights.
            ones=np.ones(len(y),dtype=float)
            for key,cache in caches.items():
                aa=auc_fast(y,ones,cache)
                if not np.isclose(aa,obs[key],atol=1e-10,rtol=0):raise RuntimeError(f'Cached AUC identity failed {key}: {aa} != {obs[key]}')
            print('WEIGHTED AUC IDENTITY PASS',flush=True)
        for mod,l,r in draws:draws[(mod,l,r)][t]=au[(mod,l)]-au[(mod,r)]
        if (t+1)%50==0:print('PAIRED ITEM-CLUSTER BOOTSTRAP',t+1,'/',args.bootstrap,flush=True)
    rows=[]
    for (mod,l,r),d in draws.items():
        point=obs[(mod,l)]-obs[(mod,r)]
        avg=float(d.mean());lo,hi=np.quantile(d,[.025,.975]);bias=avg-point
        # Basic confidence interval: reflective correction. Centered interval is mean-shifted percentile (diagnostic).
        bl,bh=2*point-hi,2*point-lo
        cl,ch=point+lo-avg,point+hi-avg
        rows.append(dict(domain=args.domain,seed=args.seed,modality=mod,lhs=l,rhs=r,
            lhs_auc=obs[(mod,l)],rhs_auc=obs[(mod,r)],observed_auc_diff=point,
            bootstrap_mean=avg,bootstrap_bias=bias,
            percentile_lo=lo,percentile_hi=hi,basic_lo=bl,basic_hi=bh,
            centered_lo=cl,centered_hi=ch,
            centered_excludes_zero=bool(cl>0 or ch<0),
            n_replicates=len(d)))
    write_csv(out/'auc_identity.csv',outputs)
    write_csv(out/'paired_auc_ci.csv',rows)
    np.savez_compressed(out/'paired_auc_draws.npz',**{f'{m}_{l}_minus_{r}':d for (m,l,r),d in draws.items()})
    (out/'audit_metadata.json').write_text(json.dumps(dict(method='Fixed U3 candidate item-cluster Poisson multiplier paired AUC bootstrap',
        source=str(src),seed=args.seed,audit_seed=args.audit_seed,reps=args.bootstrap,
        candidate_sha256=sha(np.column_stack((ea,eb,y))),
        n_candidates=len(y),n_positive=int(y.sum()),n_negative=int((y==0).sum()),
        caveats=['This is item-cluster uncertainty for a fixed held-out candidate set, not user-level resampled U3 graphs.',
        'Trained tokenizer and U1 PMI tables are held fixed; training variance is not included.',
        'Product of independent item multipliers is a sensitivity bootstrap and can distort code-pair distribution.',
        'Percentile, basic and centered intervals are all reported; choice of interval can affect inference.',
        'Code-MI is pair-code objective (F2), not hierarchical pseudo-label MI (F1).']),indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps(dict(status='success',rows=len(rows),identity='PASS'))+'\n')
    print('AUDIT SUCCESS',out,'rows',len(rows),flush=True)

if __name__=='__main__':main()
