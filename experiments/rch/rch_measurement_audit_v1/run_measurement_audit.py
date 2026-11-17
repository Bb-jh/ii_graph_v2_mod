#!/usr/bin/env python3
"""Pre-R1 measurement validity audit. U1 spectral tree / content only; NO U2/U3.

Three complementary tests:
 (1) fixed-test learning curves for root and selected inner nodes;
 (2) matched-n node-wise probes, recording infeasibility explicitly;
 (3) shared pooled conditional probe with per-parent intercepts and local child labels.
Every hyperparameter is tuned INSIDE training split. Heldout is never used for selection.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'experiments'/'rch_probe'))
import run_rch_probe as core


def save_csv(path,rows):
    if not rows: return
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def split_train_test(y,seed,test_fraction=.2):
    n=len(y); ids=np.arange(n)
    unique, counts=np.unique(y,return_counts=True)
    strat=y if counts.min()>=2 and len(unique)>1 else None
    return train_test_split(ids,test_size=test_fraction,random_state=seed,stratify=strat)

def ce_prior(ytr,yte,k):
    # Laplace-smoothed TRAIN-ONLY prior, cannot peek into heldout frequencies.
    p=(np.bincount(ytr,minlength=k)+1.)/(len(ytr)+k)
    return -np.log(p[yte])

def fit_logistic_prob(Xtr,ytr,Xte,k,C,seed,max_iter=160):
    if len(np.unique(ytr))<2:
        probs=np.tile((np.bincount(ytr,minlength=k)+1.)/(len(ytr)+k),(Xte.shape[0],1))
        return probs
    sc=StandardScaler().fit(Xtr)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',ConvergenceWarning)
        model=LogisticRegression(C=C,max_iter=max_iter,random_state=seed)
        model.fit(sc.transform(Xtr),ytr)
    pr=np.zeros((Xte.shape[0],k),dtype=float)
    pr[:,model.classes_.astype(int)]=model.predict_proba(sc.transform(Xte))
    pr=np.maximum(pr,1e-8);pr/=pr.sum(axis=1,keepdims=True)
    return pr

def choose_c(X,y,tr,k,grid,seed,max_iter):
    if len(np.unique(y[tr]))<2:return grid[0]
    a,b=split_train_test(y[tr],core.stable_seed(seed,'inner'))
    a,b=tr[a],tr[b]
    res=[]
    for C in grid:
        pr=fit_logistic_prob(X[a],y[a],X[b],k,C,seed,max_iter)
        loss=-np.log(pr[np.arange(len(b)),y[b]]).mean()
        res.append((loss,C))
    return float(min(res)[1])

def bootstrap_ratio(gain,prior,reps,seed,groups=None):
    rng=np.random.default_rng(seed);n=len(gain);out=[]
    if groups is None:
        buckets=[np.arange(n)]
    else:
        buckets=[np.flatnonzero(groups==g) for g in np.unique(groups)]
    for _ in range(reps):
        ix=np.concatenate([rng.choice(v,size=len(v),replace=True) for v in buckets])
        out.append(gain[ix].mean()/max(prior[ix].mean(),1e-12))
    return [float(v) for v in np.quantile(out,[.025,.975])]

def eval_fixed_split(X,y,tr,te,train_sizes,c_grid,seed,bootstrap=200,perms=2,max_iter=160,tag=None):
    k=int(y.max()+1)
    # Same fixed heldout set for every n, nested train subsets.
    rng=np.random.default_rng(core.stable_seed(seed,'train-order'))
    perm=rng.permutation(tr)
    rows=[]
    for n in train_sizes:
        if n>len(tr):
            rows.append(dict(**(tag or {}),requested_train_n=n,available_train_n=len(tr),status='insufficient_train_items'))
            continue
        use=perm[:n]
        if len(np.unique(y[use]))<k:
            rows.append(dict(**(tag or {}),requested_train_n=n,available_train_n=len(tr),status='missing_training_class'))
            continue
        C=choose_c(X,y,use,k,c_grid,core.stable_seed(seed,'cv',n),max_iter)
        probs=fit_logistic_prob(X[use],y[use],X[te],k,C,core.stable_seed(seed,'fit',n),max_iter)
        model_ce=-np.log(probs[np.arange(len(te)),y[te]])
        prior_ce=ce_prior(y[use],y[te],k)
        gain=prior_ce-model_ce
        null=[]
        for r in range(perms):
            ys=np.array(y[use],copy=True)
            np.random.default_rng(core.stable_seed(seed,'shuffle',n,r)).shuffle(ys)
            pr=fit_logistic_prob(X[use],ys,X[te],k,C,core.stable_seed(seed,'nullfit',n,r),max_iter)
            null.append(-np.log(pr[np.arange(len(te)),y[te]]))
        null_ce=np.mean(null,axis=0)
        ci=bootstrap_ratio(gain,prior_ce,bootstrap,core.stable_seed(seed,'boot',n))
        nullctrl=null_ce-model_ce
        nci=bootstrap_ratio(nullctrl,prior_ce,bootstrap,core.stable_seed(seed,'nullboot',n))
        rows.append(dict(**(tag or {}),requested_train_n=n,available_train_n=len(tr),test_n=len(te),
            C=C,R=float(gain.mean()/prior_ce.mean()),R_lo=ci[0],R_hi=ci[1],
            prior_ce=float(prior_ce.mean()),model_ce=float(model_ce.mean()),null_ce=float(null_ce.mean()),
            null_control_gain=float(nullctrl.mean()/prior_ce.mean()),null_control_lo=nci[0],null_control_hi=nci[1],
            status='ok'))
    return rows

def get_nodes(labels,items,depth):
    if depth==1:
        return [(0,items,labels[items,0].astype(int))]
    result=[]
    for p in np.unique(labels[items,depth-2]):
        ids=items[labels[items,depth-2]==p]
        yy=labels[ids,depth-1]
        if len(np.unique(yy))<2:continue
        _,yc=np.unique(yy,return_inverse=True)
        result.append((int(p),ids,yc.astype(int)))
    return result

def prep_node_splits(labels,items,depth,seed,test_frac=.2):
    """Every parent has one fixed heldout 20% split. Parent-ID already determined from frozen U1 graph."""
    objects=[];excluded=[]
    for p,ids,yy in get_nodes(labels,items,depth):
        if len(ids)<25 or np.bincount(yy).min()<3:
            excluded.append(dict(depth=depth,parent=p,n_items=len(ids),status='insufficient_parent_or_child_samples'))
            continue
        a,b=split_train_test(yy,core.stable_seed(seed,'heldout',depth,p),test_frac)
        if len(np.unique(yy[a]))<len(np.unique(yy)) or len(np.unique(yy[b]))<len(np.unique(yy)):
            excluded.append(dict(depth=depth,parent=p,n_items=len(ids),status='missing_class_in_train_or_test'))
            continue
        objects.append((p,ids,yy,a,b))
    return objects,excluded

def pooled_probe(objs,feats,depth,weight_decays,seed,bootstrap,perms,epochs,smoke=False):
    """Shared nonlinear Text encoder + independent parent-local child heads.

    Truly shares representation across nodes without assuming child index '0' in
    unrelated parents represents the same semantic category. Parent-only prior
    and within-parent shuffled-label null are fitted using TRAIN ONLY.
    """
    import torch
    from torch import nn
    torch.set_num_threads(min(4,torch.get_num_threads()))
    ps=[];ys=[];ix=[];is_tr=[];psize={}
    for j,(p,ids,y,a,b) in enumerate(objs):
        ii=np.concatenate([a,b]);ps.extend([j]*len(ii));ys.extend(y[ii]);ix.extend(ids[ii]);is_tr.extend([True]*len(a)+[False]*len(b))
        psize[j]=int(max(y)+1)
    ps=np.asarray(ps,dtype=int);ys=np.asarray(ys,dtype=int);ix=np.asarray(ix,dtype=int);is_tr=np.asarray(is_tr,dtype=bool)
    P=len(objs);K=int(ys.max()+1);D=feats.shape[1]
    tr=np.flatnonzero(is_tr);te=np.flatnonzero(~is_tr)
    scaler=StandardScaler().fit(feats[ix[tr]])
    X=np.clip(scaler.transform(feats[ix]),-8,8).astype('float32')
    XT=torch.from_numpy(X);YT=torch.from_numpy(ys.astype('int64'));PT=torch.from_numpy(ps.astype('int64'))
    class SharedProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared=nn.Sequential(nn.Linear(D,64),nn.ReLU())
            self.w=nn.Parameter(torch.zeros(P,64,K))
            self.b=nn.Parameter(torch.zeros(P,K))
            nn.init.normal_(self.w,std=.02)
            valid=torch.zeros((P,K),dtype=torch.bool)
            for j in range(P):valid[j,:psize[j]]=True
            self.register_buffer('valid',valid)
        def forward(self,x,p):
            h=self.shared(x)
            logits=torch.bmm(h.unsqueeze(1),self.w[p]).squeeze(1)+self.b[p]
            return logits.masked_fill(~self.valid[p],-1e8)
    def train_predict(a,ytrain,predict_indices,wd,fitseed,nepochs):
        torch.manual_seed(int(fitseed)%(2**31-1))
        model=SharedProbe()
        opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=wd)
        yp=YT.clone()
        yp[a]=torch.as_tensor(ytrain,dtype=torch.int64)
        batch=512
        for epoch in range(nepochs):
            gen=torch.Generator().manual_seed(int(core.stable_seed(fitseed,'epoch',epoch))%(2**31-1))
            order=torch.as_tensor(a,dtype=torch.int64)[torch.randperm(len(a),generator=gen)]
            model.train()
            for bb in order.split(batch):
                opt.zero_grad()
                loss=nn.functional.cross_entropy(model(XT[bb],PT[bb]),yp[bb])
                loss.backward();opt.step()
        model.eval()
        pp=[]
        with torch.no_grad():
            for bb in torch.as_tensor(predict_indices,dtype=torch.int64).split(1024):
                pp.append(torch.softmax(model(XT[bb],PT[bb]),dim=-1).numpy())
        pr=np.concatenate(pp,axis=0)
        return np.maximum(pr,1e-12)
    # CV is INTERNAL to train items, preserving parent-child strata where feasible.
    joint=ps[tr].astype(str)+'_'+ys[tr].astype(str)
    _,cnt=np.unique(joint,return_counts=True)
    a0,b0=train_test_split(np.arange(len(tr)),test_size=.2,random_state=core.stable_seed(seed,'cv'),stratify=joint if cnt.min()>=2 else None)
    a,b=tr[a0],tr[b0]
    best=[]
    for wd in weight_decays:
        pr=train_predict(a,ys[a],b,wd,core.stable_seed(seed,'cvtrain',str(wd)),max(6,epochs//2))
        loss=float(-np.log(pr[np.arange(len(b)),ys[b]]).mean())
        best.append((loss,float(wd)))
    wd=min(best)[1]
    def prior_for(ytrain):
        out=np.empty((len(te),K),dtype=float)
        for parent in range(P):
            vals=ytrain[ps[tr]==parent]
            k=psize[parent]
            freq=np.bincount(vals,minlength=k)+1.
            out[ps[te]==parent]=0
            out[np.ix_(np.flatnonzero(ps[te]==parent),np.arange(k))]=freq/freq.sum()
        return out
    pr=train_predict(tr,ys[tr],te,wd,core.stable_seed(seed,'actual'),epochs)
    pc=prior_for(ys[tr]);prior=-np.log(np.maximum(pc[np.arange(len(te)),ys[te]],1e-12))
    ce=-np.log(pr[np.arange(len(te)),ys[te]])
    null_ce=[]
    for r in range(perms):
        yperm=ys[tr].copy();rng=np.random.default_rng(core.stable_seed(seed,'null',r))
        for p in range(P):
            loc=np.flatnonzero(ps[tr]==p);yperm[loc]=rng.permutation(yperm[loc])
        pn=train_predict(tr,yperm,te,wd,core.stable_seed(seed,'nullfit',r),epochs)
        null_ce.append(-np.log(pn[np.arange(len(te)),ys[te]]))
    nc=np.mean(null_ce,axis=0)
    gain=prior-ce;ctrl=nc-ce
    ci=bootstrap_ratio(gain,prior,bootstrap,core.stable_seed(seed,'boot'),groups=ps[te])
    nullci=bootstrap_ratio(ctrl,prior,bootstrap,core.stable_seed(seed,'nullboot'),groups=ps[te])
    summary=[dict(depth=depth,probe='shared_MLP_parent_local_heads',parents=P,n_train=len(tr),n_test=len(te),weight_decay=wd,
        R=float(gain.mean()/prior.mean()),R_lo=ci[0],R_hi=ci[1],prior_ce=float(prior.mean()),model_ce=float(ce.mean()),null_ce=float(nc.mean()),
        null_control_R=float(ctrl.mean()/prior.mean()),null_control_lo=nullci[0],null_control_hi=nullci[1],status='ok',
        interpretation='transductive U1 labels, independent item-heldout, not strict cold-item')]
    per=[]
    for j,(parent,ids,y,a,b) in enumerate(objs):
        mask=ps[te]==j
        ci=bootstrap_ratio(gain[mask],prior[mask],bootstrap,core.stable_seed(seed,'parentboot',parent))
        nullci=bootstrap_ratio(ctrl[mask],prior[mask],bootstrap,core.stable_seed(seed,'parentnullboot',parent))
        per.append(dict(depth=depth,parent=parent,n_train=int(np.sum(ps[tr]==j)),n_test=int(mask.sum()),R=float(gain[mask].mean()/prior[mask].mean()),
            R_lo=ci[0],R_hi=ci[1],null_control_R=float(ctrl[mask].mean()/prior[mask].mean()),
            null_control_lo=nullci[0],null_control_hi=nullci[1],pass_T=bool(ci[0]>.02 and gain[mask].mean()>.005 and nullci[0]>0),status='ok'))
    return summary,per

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain',choices=['Industrial','Musical'],required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--mode',choices=['smoke','run'],default='run')
    p.add_argument('--part',choices=['all','root','matched','pooled'],default='all')
    p.add_argument('--depths',default='1,2,3')
    p.add_argument('--train-sizes',default='500,1000,1500,2000,5000,12000')
    p.add_argument('--c-grid',default='0.01,0.1,1')
    p.add_argument('--max-iter',type=int,default=180)
    p.add_argument('--pooled-epochs',type=int,default=22)
    p.add_argument('--pooled-weight-decays',default='0.0001,0.001')
    p.add_argument('--permutations',type=int,default=3)
    p.add_argument('--bootstrap',type=int,default=200)
    p.add_argument('--max-nodes-per-depth',type=int,default=0,help='0 means all')
    p.add_argument('--out-root',default='outputs/rch_measurement_audit_v1')
    args=p.parse_args();os.chdir(REPO)
    ns=core.parse_args(['--domain',args.domain,'--seed',str(args.seed),'--mode','run'])
    data=core.load_server_data(ns)
    items=np.asarray(data['train_items']);n=int(data['n'])
    feats=core.l2norm(np.asarray(data['emb']['text'][0]))
    treepath=REPO/'outputs'/'rch_next_audit_v1'/'run'/f'{args.domain}_seed{args.seed}'/'spectral_recursive_tree_labels.npy'
    if not treepath.exists():raise SystemExit(f'Missing frozen U1 tree: {treepath}. Run prior recursive audit. This tool MUST NOT refit teacher.')
    labels=np.load(treepath)
    if labels.shape!=(n,3):raise ValueError(f'Expected labels shape {(n,3)}, got {labels.shape}')
    if np.any(labels[items]<0):raise ValueError('Missing U1 item labels')
    if feats.shape[0]!=n:raise ValueError('Embedding and item-index mismatch')
    dest=REPO/args.out_root/args.mode/f'{args.domain}_seed{args.seed}'
    dest.mkdir(parents=True,exist_ok=True)
    sizes=[int(x) for x in args.train_sizes.split(',')]
    depths=[int(x) for x in args.depths.split(',')]
    if not set(depths).issubset({1,2,3}):raise ValueError('Frozen tree available only for depth 1..3. D4/D5 must be exported from prior construction first.')
    grid=[float(x) for x in args.c_grid.split(',')]
    if args.mode=='smoke':
        sizes=[100,200];grid=[.1];args.permutations=1;args.bootstrap=30;args.max_iter=60;args.max_nodes_per_depth=2;args.pooled_epochs=6
    manifest=dict(domain=args.domain,seed=args.seed,tree=str(treepath),tree_sha256=sha(treepath),
        train_item_count=len(items),args=vars(args),
        constraints=['U1 fixed transductive labels, NOT true cold-item evaluation','No U2/U3 behavior used',
        'Train/test are item-disjoint within each parent and fixed across matched n',
        'Pooled probe: shared nonlinear encoder + parent-local output heads, NOT a recommender/tokenizer',
        'R=(prior-model)/prior. Null shuffled within parent only for significance, not effect size',
        'Parent-specific pass is exploratory. Bootstrap conditions on fitted probe/selection; no model refits',
        'D1..3 only because frozen D4+ global assignments are not provided.'])
    manifest_path=dest/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old!=manifest:
            raise RuntimeError('Existing results use different parameters/tree; use a distinct --out-root or move old outputs. Do not mix cached runs.')
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2))
    print('START R MEASUREMENT AUDIT',args.domain,args.mode,'tree',treepath,flush=True)
    for depth in depths:
        objs,excluded=prep_node_splits(labels,items,depth,args.seed)
        if args.max_nodes_per_depth:
            for p,ids,*_ in objs[args.max_nodes_per_depth:]:
                excluded.append(dict(depth=depth,parent=p,n_items=len(ids),status='smoke_or_user_node_cap'))
            objs=objs[:args.max_nodes_per_depth]
        if excluded:save_csv(dest/f'excluded_depth{depth}.csv',excluded)
        print(f'Depth {depth}: {len(objs)} eligible parents',flush=True)
        if not objs:continue
        rootrows=[];matchedrows=[]
        if args.part in ('all','root','matched'):
            for p,ids,yy,a,b in objs:
                seed=core.stable_seed(args.seed,'measurement',depth,p)
                X=feats[ids]
                cache=dest/f'node_depth{depth}_parent{p}.json'
                if cache.exists():
                    val=json.loads(cache.read_text())
                    print(f'  node {p} cached',flush=True)
                else:
                    root_one=[];matched_one=[]
                    if args.part in ('all','root') and depth==1:
                        root_one=eval_fixed_split(X,yy,a,b,sizes,grid,seed,args.bootstrap,args.permutations,args.max_iter,
                            {'depth':depth,'parent':p,'audit':'root_learning_curve'})
                    if args.part in ('all','matched'):
                        matched=[v for v in (500,1000,1500) if v in sizes]
                        if args.mode=='smoke':matched=sizes
                        matched_one=eval_fixed_split(X,yy,a,b,matched,grid,seed,args.bootstrap,args.permutations,args.max_iter,
                            {'depth':depth,'parent':p,'parent_n':len(ids),'audit':'matched_node'})
                    val={'root':root_one,'matched':matched_one}
                    cache.write_text(json.dumps(val,indent=2))
                    print(f'  node {p} n={len(ids)} train={len(a)} test={len(b)} saved',flush=True)
                rootrows.extend(val['root']);matchedrows.extend(val['matched'])
            if rootrows:save_csv(dest/'root_learning_curve.csv',rootrows)
            if matchedrows:save_csv(dest/f'node_matched_depth{depth}.csv',matchedrows)
        if args.part in ('all','pooled'):
            target=dest/f'pooled_depth{depth}_summary.csv'
            if target.exists() and (dest/f'pooled_depth{depth}_parents.csv').exists():
                print(f'  pooled depth{depth} cached',flush=True)
            else:
                print('  pooled training...',flush=True)
                summ,per=pooled_probe(objs,feats,depth,[float(z) for z in args.pooled_weight_decays.split(',')],
                    core.stable_seed(args.seed,'pooled',depth),args.bootstrap,args.permutations,args.pooled_epochs,args.mode=='smoke')
                if summ:save_csv(target,summ)
                if per:save_csv(dest/f'pooled_depth{depth}_parents.csv',per)
                print('  pooled R=',summ[0]['R'] if summ else 'NA',flush=True)
    print('MEASUREMENT AUDIT SUCCESS',dest,flush=True)

if __name__=='__main__':main()
