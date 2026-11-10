#!/usr/bin/env python3
"""Item-grouped 5-fold utility, U3 held-out-by-item transfer and full-model transfer.
Pre-registered acceptance: p<0.05, 95% item bootstrap CI excludes 0, and U3 utility sign matches U2.
All model fits train exclusively on U2; U3 labels are for diagnostic validation and predeclared acceptance only.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

MODELS={'B':['B0','B1','B2'],'BT':['B0','B1','B2','T'],'BV':['B0','B1','B2','V'],
        'BTV':['B0','B1','B2','T','V'],'T':['T'],'TV':['T','V']}

def writecsv(path,rows):
    if not rows:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def getdata(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def design(d):
    b=d['B'];return {'B0':b[:,0].astype(float),'B1':b[:,1].astype(float),'B2':b[:,2].astype(float),
                      'T':d['st'].astype(float),'V':d['sv'].astype(float)}

def matrix(feat,names):return np.column_stack([feat[k] for k in names])

def fit_predict(x,y,tr,te,C):
    ytr=y[tr]
    if len(np.unique(ytr))<2:return np.full(len(te),float(np.mean(ytr)))
    sc=StandardScaler().fit(x[tr]);model=LogisticRegression(C=C,max_iter=300,solver='lbfgs')
    model.fit(sc.transform(x[tr]),ytr)
    return model.predict_proba(sc.transform(x[te]))[:,1]

def scores(y,p):
    if len(np.unique(y))<2:return dict(auc=np.nan,prauc=np.nan,logloss=np.nan)
    p=np.clip(p,1e-8,1-1e-8)
    return dict(auc=float(roc_auc_score(y,p)),prauc=float(average_precision_score(y,p)),
                logloss=float(log_loss(y,p,labels=[0,1])))

def fold_sets(d,folds,seed,mode):
    # Item-group assignment is fixed before fitting. Both endpoints in held-out
    # folds is strictly item-disjoint, but evaluates only intra-fold edges (~1/K).
    n=int(max(d['a'].max(),d['b'].max()))+1
    rng=np.random.default_rng(seed+311);assign=rng.integers(0,folds,size=n)
    a,b=d['a'],d['b']
    # In anchor mode every edge tested once; train edges incident to heldout
    # anchor items are removed. Neighbors may occur in other train edges.
    chosen_anchor=np.where(rng.random(len(a))<.5,a,b)
    for f in range(folds):
        unseen=assign==f
        tr=~(unseen[a]|unseen[b])
        te=(unseen[a]&unseen[b]) if mode=='both' else unseen[chosen_anchor]
        if not tr.any() or not te.any():continue
        yield tr,te

def grouped_bins(d,bins):
    b=d['B']; code=np.zeros(len(b),dtype=np.int64)
    for j in range(3):
        q=np.unique(np.quantile(b[:,j],np.linspace(0,1,bins+1)))
        c=np.searchsorted(q[1:-1],b[:,j],side='right')
        code=code*bins+c
    return code

def shuffle_in_bins(arr,codes,rng):
    out=arr.copy()
    for group in np.unique(codes):
        ix=np.flatnonzero(codes==group)
        if len(ix)>1:out[ix]=arr[ix[rng.permutation(len(ix))]]
    return out

def pred_oof(d,feat,folds,seed,mode,C,only=('B','BT','BV','BTV','T','TV'), target='y2'):
    y=d['y2'];n=len(y);pred={key:np.full(n,np.nan) for key in only}
    if target not in ('y2','y3'):raise ValueError('Unknown prediction target')
    for f,(tr,te) in enumerate(fold_sets(d,folds,seed,mode)):
        it=np.flatnonzero(tr);iv=np.flatnonzero(te)
        for name in only:
            pred[name][iv]=fit_predict(matrix(feat,MODELS[name]),y,it,iv,C)
    return pred

def pred_full(d,feat,C,only=('B','BT','BV','BTV','T','TV')):
    idx=np.arange(len(d['y2']));return {m:fit_predict(matrix(feat,MODELS[m]),d['y2'],idx,idx,C) for m in only}

def paired_deltas(s):
    return {'delta_T':s['BT']['auc']-s['B']['auc'], 'delta_V':s['BV']['auc']-s['B']['auc'],
       'u_T':s['BTV']['auc']-s['BV']['auc'],'u_V':s['BTV']['auc']-s['BT']['auc'],
       'dll_T':s['BV']['logloss']-s['BTV']['logloss'],
       'dll_V':s['BT']['logloss']-s['BTV']['logloss'],
       'dpr_T':s['BTV']['prauc']-s['BV']['prauc'],
       'dpr_V':s['BTV']['prauc']-s['BT']['prauc']}

def evaluate(y,pred,mask):
    return {m:scores(y[mask],p[mask]) for m,p in pred.items()}

def item_boot(d,y,pred,mask,mode,reps,seed):
    # Anchor item cluster bootstrap, keeping all its (masked) edges together.
    rng=np.random.default_rng(seed+789)
    a,b=d['a'],d['b'];n=int(max(a.max(),b.max()))+1
    # Each edge assigned consistently to one anchor item; avoids double counts
    anchor=np.minimum(a,b)
    tested=np.flatnonzero(mask)
    aa=anchor[tested]
    order=np.argsort(aa);tested=tested[order];aa=aa[order]
    ids,first,count=np.unique(aa,return_index=True,return_counts=True)
    diffs=[]
    for rep in range(reps):
        selected=rng.integers(0,len(ids),size=len(ids));ix=np.concatenate([tested[first[j]:first[j]+count[j]] for j in selected])
        if len(np.unique(y[ix]))<2:continue
        # repeated sampled items must retain multiplicities
        met={m:scores(y[ix],pred[m][ix]) for m in ('B','BT','BV','BTV')}
        diffs.append(paired_deltas(met))
    out=[]
    for key in ('u_T','u_V'):
        v=np.array([z[key] for z in diffs]);out.append((key,float(np.quantile(v,.025)),float(np.quantile(v,.975)),len(v)))
    return out

def evaluate_group(y,pred,mask):
    if mask.sum()<25 or len(np.unique(y[mask]))<2:return None
    return paired_deltas(evaluate(y,pred,mask))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--dataset',required=True);ap.add_argument('--encoder',default='CLIP');ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--folds',type=int,default=5);ap.add_argument('--fold-mode',choices=('anchor','both'),default='anchor')
    ap.add_argument('--C',type=float,default=1.);ap.add_argument('--permutations',type=int,default=100)
    ap.add_argument('--bootstrap',type=int,default=500);ap.add_argument('--bins',type=int,default=4)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    d=getdata(args.input); f=design(d);y2=d['y2'];y3=d['y3'];N=len(y2)
    preds=pred_oof(d,f,args.folds,args.seed,args.fold_mode,args.C)
    valid=np.isfinite(preds['BTV'])
    for name,p in preds.items():
        if not np.array_equal(np.isfinite(p),valid):raise RuntimeError('Models have different evaluation masks')
    report=[];summary=[]
    def record(split,mask,pr):
        metrics=evaluate(y2 if split=='U2_OOF' else y3,pr,mask)
        delta=paired_deltas(metrics)
        row=dict(dataset=args.dataset,encoder=args.encoder,seed=args.seed,split=split,n_edges=int(mask.sum()),
                 positive_rate=float((y2 if split=='U2_OOF' else y3)[mask].mean()),**delta)
        for mod,sc in metrics.items():
            for key,value in sc.items():row[f'{mod}_{key}']=value
        report.append(row);return delta
    observed=record('U2_OOF',valid,preds)
    # U3 is never used for model fitting; full U2 model predicts all candidate edges.
    allidx=np.arange(N)
    full=pred_full(d,f,args.C)
    u3_full=record('U3_TRANSFER',np.ones(N,dtype=bool),full)
    # Distinct cross-item-fold U3 transfer: each fold's classifier is fitted on U2
    # edges not touching its held-out items. Predict the matching U3 edge labels.
    # A full-U2 fit restricted to `valid` would be identical to U3_TRANSFER when
    # anchor mode covers all edges; it was the old bug and must not be repeated.
    u3_fold_preds=preds  # U2-fold models evaluated against U3 labels, no U3 fitting
    u3_valid=np.isfinite(u3_fold_preds['BTV'])
    for name,pred in u3_fold_preds.items():
        if not np.array_equal(np.isfinite(pred),u3_valid):
            raise RuntimeError('U3 fold models have inconsistent evaluation masks')
    u3_fold=record('U3_TRANSFER_ITEM_FOLD',u3_valid,u3_fold_preds)
    if not np.array_equal(valid,u3_valid):
        raise RuntimeError('U2/U3 heldout-fold edge mask mismatch')
    for component in ('T','V'):
        # Original item-degree: degree of less active endpoint and more active endpoint
        deg=d['deg1'];item_score=np.minimum(deg[d['a']],deg[d['b']])
        quant=np.unique(np.quantile(item_score,[0,1/3,2/3,1]))
        code=np.searchsorted(quant[1:-1],item_score,side='right')
        for g,name in enumerate(('Low','Mid','High')):
            mask=valid&(code==g)
            dlt=evaluate_group(y2,preds,mask)
            if dlt is not None:
                summary.append(dict(dataset=args.dataset,encoder=args.encoder,seed=args.seed,
                                    split='U2_OOF',degree_group=name,n_edges=int(mask.sum()),
                                    u_T=dlt['u_T'],u_V=dlt['u_V']))
        break
    writecsv(out/'utility_metrics.csv',report);writecsv(out/'degree_utility.csv',summary)
    ci=item_boot(d,y2,preds,valid,args.fold_mode,args.bootstrap,args.seed)
    codes=grouped_bins(d,args.bins)
    rows=[]
    for label,base_name in (('T','BV'),('V','BT')):
        baseline=scores(y2[valid],preds[base_name][valid])['auc']
        obs=observed['u_'+label]
        checkpoint=out/f'permutation_null_{label}.csv'
        previous={}
        if checkpoint.exists():
            with checkpoint.open(newline='') as stream:
                for row in csv.DictReader(stream):previous[int(row['repeat'])]=float(row['null_utility'])
        if previous and max(previous)>=args.permutations:
            raise RuntimeError('Permutation checkpoint has more repetitions than requested; use a new output directory')
        null=[];null_rows=[]
        print('Permutation',label,'observed utility',obs,'resuming',len(previous),flush=True)
        for j in range(args.permutations):
            if j in previous:val=previous[j]
            else:
                # Independent repeat-specific RNG enables exact resumption.
                rng=np.random.default_rng(args.seed+100000*j+(110 if label=='T' else 220))
                shuffled=shuffle_in_bins(f[label],codes,rng)
                ff=dict(f);ff[label]=shuffled
                pm=pred_oof(d,ff,args.folds,args.seed,args.fold_mode,args.C,only=('BTV',))['BTV']
                val=scores(y2[valid],pm[valid])['auc']-baseline
            null.append(val)
            null_rows.append(dict(repeat=j,null_utility=val))
            if j not in previous:
                writecsv(checkpoint,null_rows)
            if (j+1)%10==0:print(label,'permutation',j+1,'/',args.permutations,flush=True)
        p=(1+sum(v>=obs for v in null))/(1+len(null)) if null else np.nan
        cil,cir,nvalid=next((lo,hi,nn) for key,lo,hi,nn in ci if key=='u_'+label)
        u3_value=u3_fold['u_'+label]
        ci_excludes_zero=bool(np.isfinite(cil) and np.isfinite(cir) and (cil>0 or cir<0))
        same_sign=bool(np.isfinite(u3_value) and np.isfinite(obs) and obs!=0 and u3_value!=0 and np.sign(obs)==np.sign(u3_value))
        passed=bool(p<.05 and ci_excludes_zero and same_sign)
        rows.append(dict(dataset=args.dataset,encoder=args.encoder,seed=args.seed,modality=label,
                         observed_u_auc=obs,u3_item_fold_u_auc=u3_value,bootstrap_ci_low=cil,bootstrap_ci_high=cir,
                         bootstrap_valid=nvalid,permutations=len(null),null_mean=float(np.mean(null)) if null else np.nan,
                         null_sd=float(np.std(null,ddof=1)) if len(null)>1 else np.nan,
                         p_one_sided=p,passes_permutation=bool(p<.05),
                         ci_excludes_zero=ci_excludes_zero,u3_sign_matches=same_sign,
                         significant=passed))
    writecsv(out/'permutation_utility.csv',rows)
    # Pre-registered allocation: non-significant or non-positive utility => zero.
    positive={r['modality']:max(0,r['observed_u_auc']) if r['significant'] else 0. for r in rows}
    tot=sum(positive.values());rv=positive['V']/tot if tot>0 else None
    # If both zero, allocate evenly but mark no evidence; do not invent unique allocation
    Lv=int(np.floor(4*rv+.5)) if tot>0 else 2
    alloc=dict(dataset=args.dataset,encoder=args.encoder,seed=args.seed,u_T_effective=positive['T'],
               u_V_effective=positive['V'],r_V=rv,L_T=2+(4-Lv),L_V=2+Lv,
               both_zero=(tot==0),tie_policy='half-up round for V; both-zero 4/4 fallback',
               caution='pilot allocation; freeze per dataset only after multiseed aggregation')
    (out/'predicted_allocation.json').write_text(json.dumps(alloc,indent=2,allow_nan=False)+'\n')
    manifest=dict(dataset=args.dataset,encoder=args.encoder,seed=args.seed,folds=args.folds,fold_mode=args.fold_mode,
        n_candidates=N,n_oof=int(valid.sum()),oof_coverage=float(valid.mean()),u3_fold_mask_n=int(u3_valid.sum()),permutations=args.permutations,
        bootstrap=args.bootstrap,bins=args.bins,C=args.C,seed42_split_rebuilt=True,
        significance_rule='p<0.05 AND bootstrap 95% CI excludes zero AND sign(U3 item-fold utility)==sign(U2 OOF utility)',
        allocation_rule='non-significant or nonpositive U2 utility -> 0; 8 total, 2 minimum each, half-up V rounding, 4+4 if both zero',
        u3_transfer_modes='U3_TRANSFER=full U2 fit; U3_TRANSFER_ITEM_FOLD=U2 item-heldout-fold models predictions against U3 labels',
        caveat='anchor mode holds out assigned anchor items, not both endpoints. both mode covers within-fold pairs only. U3 labels used only for validation and pre-registered accept/reject, not fitted.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Utility:',report[0],flush=True);print('Permutation:',rows,flush=True);print('Allocation:',alloc,flush=True)
if __name__=='__main__':main()
