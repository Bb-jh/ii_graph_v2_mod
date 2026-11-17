#!/usr/bin/env python3
"""Follow-up to R measurement audit: shared probe underfit/overfit sanity check.
Frozen U1 tree; same parent-wise item-heldout split; train-only validation.
NO U2/U3 access, NO strict cold-item claim. Requires prior rch_measurement_audit_v1.
"""
import argparse, csv, hashlib, json, os, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'experiments'/'rch_measurement_audit_v1'))
import run_measurement_audit as old
core=old.core

def dump(p,rs):
    if not rs:return
    p.parent.mkdir(parents=True,exist_ok=True)
    ks=list(dict.fromkeys(k for row in rs for k in row))
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=ks);w.writeheader();w.writerows(rs)

def joint_validation(ps,ys,train,seed):
    joint=ps[train].astype(str)+'_'+ys[train].astype(str)
    _,cnt=np.unique(joint,return_counts=True)
    strat=joint if cnt.min()>=2 else None
    a,b=train_test_split(np.arange(len(train)),test_size=.2,random_state=int(seed)%(2**32-1),stratify=strat)
    return train[a],train[b]

def assemble(objs,feats,seed):
    ids=[];ps=[];ys=[];trainflag=[];sizes=[];parents=[]
    for j,(p,loc,lab,a,b) in enumerate(objs):
        ii=np.concatenate((a,b))
        ids.extend(loc[ii]); ps.extend([j]*len(ii)); ys.extend(lab[ii]);
        trainflag.extend([True]*len(a)+[False]*len(b));sizes.append(int(lab.max()+1));parents.append(int(p))
    ids=np.array(ids);ps=np.array(ps);ys=np.array(ys);flag=np.array(trainflag)
    train=np.flatnonzero(flag);test=np.flatnonzero(~flag)
    trainfit,val=joint_validation(ps,ys,train,core.stable_seed(seed,'cv'))
    scale=StandardScaler().fit(feats[ids[trainfit]])
    X=np.clip(scale.transform(feats[ids]),-8,8).astype('float32')
    return X,ps,ys,trainfit,val,test,sizes,parents

def prior_ce(ps,y,fit,eval_ix,sizes):
    # Smoothed, parent-only, training-only prior.
    result=np.zeros(len(eval_ix),dtype=float)
    for j,k in enumerate(sizes):
        tr=fit[ps[fit]==j];e=np.flatnonzero(ps[eval_ix]==j)
        q=(np.bincount(y[tr],minlength=k)+1.)/(len(tr)+k)
        result[e]=-np.log(q[y[eval_ix[e]]])
    return result

def logistic_reference(X,ps,y,trainfit,val,test,sizes,parents,seed,max_iter=300):
    # Fit each parent independently. Tune C on heldout *training validation*.
    rs=[];per=[];all_ce=[];all_prior=[]
    for j,k in enumerate(sizes):
        a=trainfit[ps[trainfit]==j];b=val[ps[val]==j];t=test[ps[test]==j]
        if not len(a) or not len(b) or not len(t) or len(np.unique(y[a]))<k:
            per.append(dict(parent=parents[j],status='insufficient_parent_validation',n_train=len(a),n_val=len(b),n_test=len(t)));continue
        best=(1e20,None)
        for C in (.001,.01,.1,1.,10.):
            pp=old.fit_logistic_prob(X[a],y[a],X[b],k,C,core.stable_seed(seed,'logit',j,C),max_iter)
            ce=float(-np.log(np.maximum(pp[np.arange(len(b)),y[b]],1e-12)).mean())
            if ce<best[0]:best=(ce,C)
        C=best[1]
        # Fair comparison: both Logistic and neural train on the SAME trainfit split.
        tr=a;pp=old.fit_logistic_prob(X[tr],y[tr],X[t],k,C,core.stable_seed(seed,'logit-final',j),max_iter)
        ce=-np.log(np.maximum(pp[np.arange(len(t)),y[t]],1e-12))
        pri=prior_ce(ps,y,tr,t,sizes)
        all_ce.append(ce);all_prior.append(pri)
        per.append(dict(parent=parents[j],status='ok',n_train=len(tr),n_test=len(t),C=C,model_ce=ce.mean(),prior_ce=pri.mean(),R=1-ce.mean()/pri.mean(),val_ce=best[0]))
    if not all_ce:return [],per
    ce=np.concatenate(all_ce);prior=np.concatenate(all_prior)
    rs=[dict(model='independent_logistic',status='ok',n_test=len(ce),R=1-ce.mean()/prior.mean(),prior_ce=prior.mean(),model_ce=ce.mean())]
    return rs,per

def train_neural(X,ps,ys,trainfit,val,test,sizes,kind,hidden,wd,seed,epochs,patience,log_every,shuffle_within=False):
    import torch
    from torch import nn
    torch.set_num_threads(min(4,torch.get_num_threads()))
    P=len(sizes);K=max(sizes);D=X.shape[1]
    torch.manual_seed(int(seed)%(2**31-1))
    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            if kind=='linear':self.enc=nn.Identity();h=D
            elif kind=='shared':self.enc=nn.Sequential(nn.Linear(D,hidden),nn.ReLU());h=hidden
            elif kind=='shared_residual':
                self.enc=nn.Sequential(nn.Linear(D,hidden),nn.ReLU());h=hidden+D
            else:raise ValueError(kind)
            self.w=nn.Parameter(torch.empty(P,h,K));nn.init.normal_(self.w,std=.01)
            self.b=nn.Parameter(torch.zeros(P,K))
            mask=torch.zeros((P,K),dtype=torch.bool)
            for j,k in enumerate(sizes):mask[j,:k]=True
            self.register_buffer('mask',mask)
        def forward(self,x,p):
            h=self.enc(x)
            if kind=='shared_residual':h=torch.cat((x,h),dim=1)
            z=torch.bmm(h[:,None,:],self.w[p]).squeeze(1)+self.b[p]
            return z.masked_fill(~self.mask[p],-1e6)
    x=torch.from_numpy(X)
    pp=torch.as_tensor(ps,dtype=torch.long)
    target=np.array(ys,copy=True)
    if shuffle_within:
        rng=np.random.default_rng(core.stable_seed(seed,'shuffle'))
        for j in range(P):
            loc=trainfit[ps[trainfit]==j];target[loc]=rng.permutation(target[loc])
    y=torch.as_tensor(target,dtype=torch.long)
    model=Probe(); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=wd)
    def ce_for(indices,eval_y=None):
        model.eval();loss=[]
        with torch.no_grad():
            for loc in np.array_split(indices,max(1,int(np.ceil(len(indices)/2048)))):
                if len(loc):
                    b=torch.as_tensor(loc,dtype=torch.long)
                    z=model(x[b],pp[b]);labels=torch.as_tensor(ys[loc] if eval_y is None else eval_y[loc],dtype=torch.long)
                    loss.append(nn.functional.cross_entropy(z,labels,reduction='none').numpy())
        return np.concatenate(loss)
    curves=[];best=float('inf');best_epoch=0;best_state=None;nonimprove=0
    # Every epoch has deterministic shuffled batches; checkpoints picked ONLY by validation CE.
    for epoch in range(1,epochs+1):
        model.train()
        g=torch.Generator().manual_seed(int(core.stable_seed(seed,'epoch',epoch))%(2**31-1))
        perm=torch.as_tensor(trainfit,dtype=torch.long)[torch.randperm(len(trainfit),generator=g)]
        for batch in perm.split(512):
            opt.zero_grad();loss=nn.functional.cross_entropy(model(x[batch],pp[batch]),y[batch]);loss.backward();opt.step()
        tr=float(ce_for(trainfit,target).mean());va=float(ce_for(val).mean())
        curves.append(dict(epoch=epoch,train_ce=tr,val_ce=va))
        if va<best-1e-5:
            best=va;best_epoch=epoch;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};nonimprove=0
        else:nonimprove+=1
        if epoch==1 or epoch%log_every==0:
            print(f'      {kind} wd={wd} epoch={epoch} trainCE={tr:.4f} valCE={va:.4f} best={best:.4f}',flush=True)
        if nonimprove>=patience:break
    model.load_state_dict(best_state)
    # Evaluate test just ONCE at validation-selected checkpoint (no test-based selection).
    test_ce=ce_for(test)
    return test_ce,curves,best,best_epoch

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--domain',required=True,choices=['Industrial','Musical']);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--mode',choices=['smoke','run'],default='run')
    p.add_argument('--depths',default='1,2,3');p.add_argument('--architectures',default='linear,shared,shared_residual')
    p.add_argument('--hidden',type=int,default=256);p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=18);p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--weight-decays',default='0.0001,0.001')
    p.add_argument('--skip-logistic',action='store_true');p.add_argument('--logistic-max-iter',type=int,default=300)
    p.add_argument('--out-root',default='outputs/rch_pooled_sanity_v2')
    args=p.parse_args();os.chdir(REPO)
    ns=core.parse_args(['--domain',args.domain,'--seed',str(args.seed),'--mode','run'])
    data=core.load_server_data(ns);items=np.asarray(data['train_items']);n=int(data['n'])
    feats=core.l2norm(np.asarray(data['emb']['text'][0]))
    tree=REPO/'outputs'/'rch_next_audit_v1'/'run'/f'{args.domain}_seed{args.seed}'/'spectral_recursive_tree_labels.npy'
    if not tree.exists():raise FileNotFoundError(tree)
    lab=np.load(tree)
    if lab.shape!=(n,3):raise ValueError(f'Expected {(n,3)} labels, got {lab.shape}')
    dest=REPO/args.out_root/args.mode/f'{args.domain}_seed{args.seed}';dest.mkdir(parents=True,exist_ok=True)
    manifest={'domain':args.domain,'seed':args.seed,'tree_sha256':old.sha(tree),'args':vars(args)}
    mf=dest/'manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=manifest:raise RuntimeError('Changed settings. Use different --out-root, or move previous output.')
    mf.write_text(json.dumps(manifest,indent=2))
    summary=[];parent_rows=[]
    for depth in map(int,args.depths.split(',')):
        objs,excluded=old.prep_node_splits(lab,items,depth,args.seed)
        if excluded:dump(dest/f'excluded_depth{depth}.csv',excluded)
        if args.mode=='smoke':objs=objs[:2]
        if not objs:continue
        print(f'\nDEPTH {depth} parents={len(objs)}',flush=True)
        X,ps,y,tr,val,te,sizes,parents=assemble(objs,feats,core.stable_seed(args.seed,'pooled',depth))
        # Important: original audit pooled use all tr; split now for internal validation.
        prior=prior_ce(ps,y,tr,te,sizes)
        if not args.skip_logistic:
            rr,pr=logistic_reference(X,ps,y,tr,val,te,sizes,parents,core.stable_seed(args.seed,'logit',depth),args.logistic_max_iter if args.mode=='run' else 60)
            for row in rr:row['depth']=depth;summary.append(row)
            for row in pr:row.update(depth=depth,model='independent_logistic');parent_rows.append(row)
            dump(dest/f'logistic_depth{depth}_parents.csv',pr)
            print('  logistic R:',rr[0]['R'] if rr else None,flush=True)
        models=[]
        archs=args.architectures.split(',');wd_vals=[float(v) for v in args.weight_decays.split(',')]
        if args.mode=='smoke':archs=archs[:2];wd_vals=wd_vals[:1]
        for kind in archs:
            for wd in wd_vals:
                print(f'  training {kind} wd={wd}',flush=True)
                key=f'{kind}_wd{wd:g}'
                ce,curve,vc,be=train_neural(X,ps,y,tr,val,te,sizes,kind,args.hidden,wd,core.stable_seed(args.seed,'neural',depth,kind,wd),6 if args.mode=='smoke' else args.epochs,args.patience,args.log_every)
                for r in curve:r.update(depth=depth,model=kind,weight_decay=wd)
                dump(dest/f'depth{depth}_{key}_curve.csv',curve)
                R=1-ce.mean()/prior.mean()
                modelrow=dict(depth=depth,model=kind,weight_decay=wd,hidden=args.hidden,n_train=len(tr),n_val=len(val),n_test=len(te),best_epoch=be,val_ce=vc,model_ce=float(ce.mean()),prior_ce=float(prior.mean()),R=float(R),status='ok')
                models.append((vc,modelrow,ce))
                summary.append(modelrow)
                print(f'      BEST epoch={be} valCE={vc:.4f} testR={R:.4f} (diagnostic only, NOT used for selection)',flush=True)
        winner=min(models,key=lambda q:q[0])
        modelrow=winner[1];ce=winner[2];selected=dict(modelrow,model='VALIDATION_SELECTED_'+modelrow['model'])
        summary.append(selected)
        # test CIs and per-parent ratio for the validation-chosen neural candidate only
        gains=prior-ce
        lo,hi=old.bootstrap_ratio(gains,prior,300,core.stable_seed(args.seed,'boot',depth),groups=ps[te])
        selected['R_lo']=lo;selected['R_hi']=hi
        for j,parent in enumerate(parents):
            loc=np.flatnonzero(ps[te]==j)
            a,b=old.bootstrap_ratio(gains[loc],prior[loc],300,core.stable_seed(args.seed,'boot',depth,parent))
            parent_rows.append(dict(depth=depth,parent=parent,model=selected['model'],status='ok',n_test=len(loc),R=float(gains[loc].mean()/prior[loc].mean()),R_lo=a,R_hi=b,prior_ce=float(prior[loc].mean()),model_ce=float(ce[loc].mean())))
        dump(dest/f'depth{depth}_neural_selected_parents.csv',[r for r in parent_rows if r.get('depth')==depth and r.get('model')==selected['model']])
        dump(dest/f'depth{depth}_model_comparison.csv',[r for r in summary if r.get('depth')==depth])
        print(f'  SELECTED ON VALIDATION: {selected["model"]} R={selected["R"]:.4f} CI=[{lo:.4f},{hi:.4f}]',flush=True)
    dump(dest/'model_comparison_all.csv',summary)
    print('\nPOOLED SANITY V2 SUCCESS:',dest,flush=True)
    print('NOTE: Does not establish cold-item performance; no U2/U3 was accessed.',flush=True)

if __name__=='__main__':main()
