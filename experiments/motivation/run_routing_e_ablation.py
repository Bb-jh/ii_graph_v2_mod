#!/usr/bin/env python3
"""Content-expressibility routing diagnostic (Industrial 5-core default).

User disjoint U1 (behavior graph), U2 (expressibility fit / threshold tuning),
U3 (untouched evaluation). All edges are U1 within-parent B edges.
Stages: prepare, fit, evaluate, all. Dependency modules from existing C1/A scripts.

IMPORTANT: This is an edge-routing heldout proxy, NOT a trained RQ-VAE ranking ablation.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from run_c1_pilot import (read_users_train, csr_users, partition_by_count,
                          collect_candidate_edges, csv_write)
from run_motivation_cpu_v2 import load_mapping, load_embeddings, behavior_graph, to_igraph


def progress(*a): print(time.strftime('%Y-%m-%d %H:%M:%S'),*a,flush=True)

def jsonsave(p,ob): p.write_text(json.dumps(ob,ensure_ascii=False,indent=2)+'\n')

def read_cache(p):
    if not p.exists(): raise FileNotFoundError(f'Missing cache: {p}. Run --stage prepare first.')
    with np.load(p, allow_pickle=False) as f:return {k:f[k] for k in f.files}

def user_eval(co, deg, a, b):
    vals = np.asarray(co[a,b]).reshape(-1).astype(np.float64)
    denom = np.sqrt(deg[a].astype(float)*deg[b].astype(float))
    norm = np.divide(vals,denom,out=np.zeros_like(vals),where=denom>0)
    return (vals>0).astype(np.uint8), (vals>=2).astype(np.uint8), norm.astype(np.float32)

def prepare(args):
    ids=load_mapping(args.mapping)
    _,seqs=read_users_train(args.sequences,len(ids))
    rng=np.random.default_rng(args.seed)
    perm=rng.permutation(len(seqs))
    n1=int(round(len(seqs)*args.u1_fraction)); n2=int(round(len(seqs)*args.u2_fraction))
    if n1<=0 or n2<=0 or n1+n2>=len(seqs):raise ValueError('U1/U2/U3 must be nonempty')
    groups=[perm[:n1],perm[n1:n1+n2],perm[n1+n2:]]
    x=[csr_users([seqs[i] for i in g],len(ids)) for g in groups]
    deg1=np.asarray(x[0].sum(axis=0)).reshape(-1).astype(np.int64)
    embs=load_embeddings(args.embeddings,ids)
    sel=np.flatnonzero((deg1>=1)&embs['text'][1]&embs['image'][1])
    progress('Users U1/U2/U3:',*[len(g) for g in groups],'Items:',len(sel))
    bg,_=behavior_graph(x[0],deg1,sel,args.behavior_k,'union')
    labels,calib=partition_by_count(to_igraph(bg),args.parent_k,args.seed)
    progress('U1 behavior graph edges:',bg.nnz//2,'Parent:',calib)
    if calib['count_error']>args.max_parent_count_error:raise RuntimeError('Parent calibration failed')
    xsel=[xx[:,sel] for xx in x]
    co1=(xsel[0].T@xsel[0]).tocsr();co1.setdiag(0);co1.eliminate_zeros()
    norms={}
    for m in ('text','image'):
        z=np.ascontiguousarray(embs[m][0][sel],dtype=np.float32)
        z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
        norms[m]=z
    ed=collect_candidate_edges(bg,labels,co1,deg1[sel],norms['text'],norms['image'])
    # Same C2 eligibility rule for an apples-to-apples comparison.
    parents,count=np.unique(ed['parent'],return_counts=True)
    eligible_parents=parents[count>=args.min_parent_edges]
    mask=np.isin(ed['parent'],eligible_parents)
    if not mask.any():raise RuntimeError('No eligible within-parent behavior edges')
    ed={k:v[mask] for k,v in ed.items()}
    a,b=ed['a'],ed['b']
    metrics={}
    for idx,name in [(1,'u2'),(2,'u3')]:
        co=(xsel[idx].T@xsel[idx]).tocsr();co.setdiag(0);co.eliminate_zeros()
        deg=np.asarray(xsel[idx].sum(axis=0)).reshape(-1)
        metrics[name]=user_eval(co,deg,a,b)
    common=dict(a=a,b=b,parent=ed['parent'],wb=ed['weight'].astype(np.float32),
        st=ed['st'].astype(np.float32),sv=ed['sv'].astype(np.float32),
        bfeatures=ed['features'].astype(np.float32),
        y2=metrics['u2'][0],y2_2users=metrics['u2'][1],strength2=metrics['u2'][2],
        y3=metrics['u3'][0],y3_2users=metrics['u3'][1],strength3=metrics['u3'][2])
    np.savez_compressed(args.out/'routing_candidates.npz',**common)
    jsonsave(args.out/'prepare_manifest.json',dict(
        seed=args.seed,users_u1=len(groups[0]),users_u2=len(groups[1]),users_u3=len(groups[2]),
        eligible_items=int(len(sel)),candidate_edges=int(len(a)),parents=int(len(eligible_parents)),
        parent_calibration=calib,behavior_edges=int(bg.nnz//2),
        positive_rate_u2=float(np.mean(common['y2'])),positive_rate_u3=float(np.mean(common['y3'])),
        note='U1 builds graph; U2 trains E and tunes threshold; U3 is never used for fitting.'))
    progress('PREPARE SUCCESS:',len(a),'eligible edges')


def features(bf,s):
    """B base: log1p(min/max U1 degrees, U1 coocc), full adds s and B*s interactions."""
    b=np.asarray(bf,dtype=np.float64)
    s=np.asarray(s,dtype=np.float64).reshape(-1,1)
    return b, np.column_stack((b,s,b*s))


def model_predict(xtrain,ytrain,xtest,C):
    scaler=StandardScaler()
    train=scaler.fit_transform(xtrain)
    model=LogisticRegression(C=C,max_iter=400,solver='lbfgs',tol=1e-5)
    model.fit(train,ytrain)
    return model.predict_proba(scaler.transform(xtest))[:,1]


def fit_pair(b,f,y,train_idx,test_idx,C,min_positive,groups,local):
    """Global fallback plus parent-wise models; prevent single-class small groups."""
    tr=train_idx;te=test_idx
    ytr=y[tr]
    if np.unique(ytr).size<2:
        p=np.full(len(te),ytr.mean(),float)
        return p.copy(),p.copy(),0
    p0=model_predict(b[tr],ytr,b[te],C)
    p1=model_predict(f[tr],ytr,f[te],C)
    updated=0
    if local:
        # Strict parent-specific estimation, falling back to global for sparse parents.
        for grp in np.unique(groups[te]):
            ti=tr[groups[tr]==grp]; qi=np.flatnonzero(groups[te]==grp)
            if len(ti)<40 or min(int(y[ti].sum()),int(len(ti)-y[ti].sum()))<min_positive:
                continue
            try:
                p0[qi]=model_predict(b[ti],y[ti],b[te[qi]],C)
                p1[qi]=model_predict(f[ti],y[ti],f[te[qi]],C)
                updated+=len(qi)
            except ValueError:pass
    return p0,p1,updated


def top_indices(scores,k):
    # Deterministic ties. Ranks are descending, stable by fixed edge index.
    return np.lexsort((np.arange(len(scores)), -scores))[:k]


def jaccard_top(x,z,k):
    a=top_indices(x,k);b=top_indices(z,k)
    joint=np.intersect1d(a,b,assume_unique=True).size
    return joint, joint/(2*k-joint),joint/k


def fit(args):
    d=read_cache(args.out/'routing_candidates.npz')
    y=d['y2'].astype(np.int32); groups=d['parent']; n=len(y)
    if min(y.sum(),n-y.sum())<10:raise RuntimeError('U2 outcomes have insufficient class diversity')
    cross=KFold(n_splits=args.folds,shuffle=True,random_state=args.seed+1000)
    predictions={}
    for label,s in [('T',d['st']),('V',d['sv'])]:
        b,f=features(d['bfeatures'],s)
        p0o=np.empty(n,float);p1o=np.empty(n,float)
        for fold,(tr,te) in enumerate(cross.split(np.arange(n)),1):
            p0,p1,upd=fit_pair(b,f,y,tr,te,args.C,args.min_positive,groups,not args.global_only)
            p0o[te]=p0;p1o[te]=p1
            progress(label,'OOF fold',fold,'parent-local predictions',upd,'/',len(te))
        full=np.arange(n)
        p0f,p1f,upd=fit_pair(b,f,y,full,full,args.C,args.min_positive,groups,not args.global_only)
        progress(label,'full U2 fit parent-local predictions',upd,'/',n)
        eo=np.maximum(p1o-p0o,0.0);ef=np.maximum(p1f-p0f,0.0)
        predictions[f'E_{label}_oof']=eo.astype(np.float32)
        predictions[f'E_{label}_full']=ef.astype(np.float32)
        predictions[f'pbase_{label}_oof']=p0o.astype(np.float32)
        predictions[f'pfull_{label}_oof']=p1o.astype(np.float32)
        predictions[f'pbase_{label}_full']=p0f.astype(np.float32)
        predictions[f'pfull_{label}_full']=p1f.astype(np.float32)
    np.savez_compressed(args.out/'expressibility_scores.npz',**predictions)
    results=[];rank=[];calibration=[]
    for label,s in [('T',d['st']),('V',d['sv'])]:
        e=predictions[f'E_{label}_oof'];ef=predictions[f'E_{label}_full']
        corr=float(spearmanr(e,s).statistic) if np.std(e)>0 else float('nan')
        corr_full=float(spearmanr(ef,s).statistic) if np.std(ef)>0 else float('nan')
        for model_name,pred in [('base',predictions[f'pbase_{label}_oof']),('augmented',predictions[f'pfull_{label}_oof'])]:
            for bin_id in range(10):
                lo,hi=bin_id/10,(bin_id+1)/10
                mask=(pred>=lo)&((pred<=hi) if bin_id==9 else (pred<hi))
                calibration.append(dict(modality=label,model=model_name,bin=bin_id,n=int(mask.sum()),mean_pred=float(pred[mask].mean()) if mask.any() else '',observed_recurrence=float(y[mask].mean()) if mask.any() else ''))
        rank.append(dict(modality=label,n_edges=n,global_spearman_oof=corr,
            global_spearman_full=corr_full,E_oof_zero_fraction=float((e==0).mean()),
            E_full_zero_fraction=float((ef==0).mean()),
            oof_brier_base=float(np.mean((predictions[f'pbase_{label}_oof']-y)**2)),
            oof_brier_augmented=float(np.mean((predictions[f'pfull_{label}_oof']-y)**2))))
        for q in args.q:
            k=max(1,int(np.floor(n*q)))
            ix,j,rate=jaccard_top(e,s,k)
            results.append(dict(modality=label,score='E_oof_vs_raw',q=q,k=k,
                overlap_edges=ix,jaccard=j,overlap_over_top=rate,
                zero_E_frac=float((e==0).mean())))
            ix,j,rate=jaccard_top(ef,s,k)
            results.append(dict(modality=label,score='E_full_vs_raw',q=q,k=k,
                overlap_edges=ix,jaccard=j,overlap_over_top=rate,
                zero_E_frac=float((ef==0).mean())))
    csv_write(args.out/'topk_overlap.csv',results)
    csv_write(args.out/'global_spearman.csv',rank)
    csv_write(args.out/'probability_calibration.csv',calibration)
    jsonsave(args.out/'fit_manifest.json',dict(
        target='U2 recurrence: whether >=1 independent U2 user co-interacts with pair',
        base_features=['log1p_degree_min','log1p_degree_max','log1p_U1_coocc'],
        augmented_features='base + raw modality cosine + cosine*each_base_feature',
        E_definition='max(P_augmented - P_base, 0) using logistic probability',
        folds=args.folds,C=args.C,local_models=not args.global_only,
        min_positive_per_parent=args.min_positive,
        note='OOF E for diagnostic ranking; full-U2-fitted E for U3 evaluation. U3 untouched.'))
    progress('FIT SUCCESS')


def metrics(d,idx,score):
    y2=d['y2'][idx].astype(float);y3=d['y3'][idx].astype(float)
    y32=d['y3_2users'][idx].astype(float)
    strength=d['strength3'][idx].astype(float)
    active=(score[idx]>0)
    return dict(u2_recurrence=float(y2.mean()),u3_recurrence=float(y3.mean()),
        u3_recurrence_2users=float(y32.mean()),u3_norm_strength=float(strength.mean()),
        effective_nonzero=int(active.sum()),active_fraction=float(active.mean()))


def eval_route(args):
    d=read_cache(args.out/'routing_candidates.npz')
    e=read_cache(args.out/'expressibility_scores.npz')
    n=len(d['a']); y2=d['y2'];results=[];tuning=[];selected={}
    for label,s in [('T',d['st']),('V',d['sv'])]:
        wb=d['wb'].astype(float)
        raw=wb*np.maximum(s,0)
        E=e[f'E_{label}_full'].astype(float)
        calibrated=wb*E
        if not np.isfinite(calibrated).all():raise RuntimeError('Nonfinite E scores')
        nonzero_E=int(np.count_nonzero(E))
        progress(label,'E_full positive:',nonzero_E,'/',n)
        for q in args.q:
            k=max(1,int(np.floor(n*q)))
            # Training-only threshold choice, with EXACT same nominal top-k budget
            # and exclusion of invalid thresholds with fewer than k active edges.
            taus=np.unique(np.quantile(s,np.linspace(0,0.9,19)))
            taus=np.unique(np.r_[-1.0001,0.0,taus])
            candidates=[]
            for tau in taus:
                gated=raw*(s>tau)
                active=int(np.count_nonzero(gated>0))
                if active<k:continue
                ids=top_indices(gated,k)
                candidates.append((float(y2[ids].mean()),float(tau),ids,gated,active))
            if not candidates:
                raise RuntimeError(f'No eligible threshold for {label} q={q}; raw positive count < k. Reduce q.')
            # maximize U2 recurrence, then prefer higher threshold on ties;
            # U3 outcomes are never consulted.
            quality,tau,idx_threshold,gated,numactive=max(candidates,key=lambda x:(x[0],x[1]))
            tuning.append(dict(modality=label,q=q,k=k,selected_tau=tau,
                               u2_selected_recurrence=quality,n_eligible_thresholds=len(candidates),
                               threshold_positive_edges=numactive))
            for method,score,idx in (
                ('Raw',raw,top_indices(raw,k)),
                ('Raw+threshold',gated,idx_threshold),
                ('Calibrated',calibrated,top_indices(calibrated,k))):
                result=dict(modality=label,q=q,k=k,method=method,threshold=tau if method=='Raw+threshold' else '',
                    **metrics(d,idx,score))
                results.append(result)
                selected[f'{label}_{method.replace("+","_")}_{int(q*1000)}']=np.asarray(idx,np.int32)
                progress(label,q,method,'U3 recurrence',f'{result["u3_recurrence"]:.5f}',
                         'active',result['effective_nonzero'],'/',k)
            # all methods use same k; zero weights are not secretly counted as supervised edges
            # candidate-selection metrics include 0-weight fillers and report them explicitly.
    csv_write(args.out/'routing_ablation.csv',results)
    csv_write(args.out/'threshold_tuning.csv',tuning)
    np.savez_compressed(args.out/'selected_edge_indices.npz',**selected)
    jsonsave(args.out/'evaluation_manifest.json',dict(
        stage='Held-out U3 routing proxy; NOT RQ-VAE/T5 downstream HR/NDCG',
        edge_budget='identical global top floor(q*eligible_edges) for each method within same modality',
        baseline_raw='wB * max(cosine,0)',
        baseline_threshold='wB * max(cosine,0) * I[cosine>tau]; tau selected on U2 only',
        calibrated='wB * E_full (trained on U2 only)',
        zero_score_warning='Top-k quota filled with zero-score edges if insufficient positive scores. See effective_nonzero and active_fraction; zeros do not receive guidance.',
        caveats=['E OOF scores used for diagnostic rankings; E full U2 for U3 routing',
            'Selected-edge recurrence is only a proxy, not downstream ranking improvement.',
            'U2 optimization of threshold may be optimistic on U2; U3 is independent.',
            'Edge outcomes share users/items; naive independent-edge uncertainty is invalid.',
            'No claim of statistical significance from one user split.']))
    progress('EVALUATE SUCCESS',args.out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','fit','evaluate','all'],default='all')
    p.add_argument('--dataset',default='Industrial_and_Scientific')
    p.add_argument('--mapping',type=Path,default=Path('shared_data/Industrial_and_Scientific/5core_v2/item_mapping.tsv'))
    p.add_argument('--sequences',type=Path,default=Path.home()/'datasets/baseline_sequences/Industrial_5core_v2.txt')
    p.add_argument('--embeddings',type=Path,default=Path('artifacts/embeddings/Industrial_and_Scientific/full/clip_vit_b32.npz'))
    p.add_argument('--out',type=Path,default=Path('outputs/motivation_routing_e/industrial_split42'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--u1-fraction',type=float,default=0.5)
    p.add_argument('--u2-fraction',type=float,default=0.25)
    p.add_argument('--parent-k',type=int,default=400)
    p.add_argument('--behavior-k',type=int,default=50)
    p.add_argument('--min-parent-edges',type=int,default=50)
    p.add_argument('--max-parent-count-error',type=float,default=0.1)
    p.add_argument('--folds',type=int,default=3)
    p.add_argument('--C',type=float,default=1.0)
    p.add_argument('--min-positive',type=int,default=5)
    p.add_argument('--global-only',action='store_true',help='Disable parent-wise models, use global models')
    p.add_argument('--q',type=float,nargs='+',default=[0.1,0.2,0.3])
    args=p.parse_args()
    if any(not 0<v<1 for v in (args.u1_fraction,args.u2_fraction)) or args.u1_fraction+args.u2_fraction>=1:
        p.error('Invalid user disjoint fractions')
    if args.folds<2 or any(not 0<q<1 for q in args.q):p.error('Invalid folds/q')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.stage in ('prepare','all'):prepare(args)
    if args.stage in ('fit','all'):fit(args)
    if args.stage in ('evaluate','all'):eval_route(args)

if __name__=='__main__':main()
