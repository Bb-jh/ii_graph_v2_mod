#!/usr/bin/env python3
"""Pre-R1 frozen-teacher pooled probe: seed stability & within-parent permutation null.
Single pre-registered shared 256-ReLU + parent-local heads, fixed AdamW/wd.
Validation early stopping repeated for every permutation, no U2/U3 access.
Permutation within (parent, trainfit/val/test) independently, holding item split
and child class margins constant. p-values are Monte Carlo conditional-label
randomization diagnostics; not claims of strict cold-start generalization.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, hashlib
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'experiments'/'rch_pooled_sanity_v2'))
import run_pooled_sanity as v2
old=v2.old
core=old.core

def write(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)

def read(path):
    import pandas as pd
    return pd.read_csv(path).to_dict('records') if path.exists() else []

def train_eval(X,ps,ys,tr,val,te,sizes,seed,hidden=256,wd=.0001,epochs=70,patience=12,batch_size=512):
    import torch
    from torch import nn
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(int(seed)%(2**31-1))
    D=X.shape[1];P=len(sizes);K=max(sizes)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc=nn.Sequential(nn.Linear(D,hidden),nn.ReLU())
            self.w=nn.Parameter(torch.empty(P,hidden,K));nn.init.normal_(self.w,std=.01)
            self.b=nn.Parameter(torch.zeros(P,K))
            mask=torch.zeros((P,K),dtype=torch.bool)
            for j,k in enumerate(sizes):mask[j,:k]=True
            self.register_buffer('mask',mask)
        def forward(self,x,p):
            h=self.enc(x)
            logits=torch.bmm(h[:,None,:],self.w[p]).squeeze(1)+self.b[p]
            return logits.masked_fill(~self.mask[p],-1e6)
    model=Net();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=wd)
    XT=torch.from_numpy(X);PT=torch.as_tensor(ps,dtype=torch.long);YT=torch.as_tensor(ys,dtype=torch.long)
    def losses(indices):
        model.eval();out=[]
        with torch.no_grad():
            for ix in np.array_split(indices,max(1,int(np.ceil(len(indices)/2048)))):
                if len(ix):
                    b=torch.as_tensor(ix,dtype=torch.long)
                    logits=model(XT[b],PT[b])
                    out.append(nn.functional.cross_entropy(logits,YT[b],reduction='none').numpy())
        return np.concatenate(out)
    best=np.inf;best_state=None;best_epoch=0;bad=0
    for epoch in range(1,epochs+1):
        model.train()
        gen=torch.Generator().manual_seed(int(core.stable_seed(seed,'epoch',epoch))%(2**31-1))
        order=torch.as_tensor(tr,dtype=torch.long)[torch.randperm(len(tr),generator=gen)]
        for bb in order.split(batch_size):
            opt.zero_grad();L=nn.functional.cross_entropy(model(XT[bb],PT[bb]),YT[bb]);L.backward();opt.step()
        vl=float(losses(val).mean())
        if vl<best-1e-5:
            best=vl;best_epoch=epoch;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};bad=0
        else: bad+=1
        if bad>=patience:break
    model.load_state_dict(best_state)
    return losses(te),dict(best_epoch=best_epoch,val_ce=best,train_ce=float(losses(tr).mean()))

def bootstrap(gain,prior,group,seed,n=400):
    return old.bootstrap_ratio(gain,prior,n,seed,groups=group)

def permuted_labels(y,ps,tr,val,te,seed):
    z=y.copy();rng=np.random.default_rng(seed)
    # Separate permutations retain observed class counts in each split of each parent.
    for ix in (tr,val,te):
        for p in np.unique(ps[ix]):
            loc=ix[ps[ix]==p]
            z[loc]=rng.permutation(z[loc])
    return z

def score(y,ps,tr,te,sizes,ce):
    prior=v2.prior_ce(ps,y,tr,te,sizes)
    gain=prior-ce
    return float(gain.sum()/prior.sum()),prior,gain

def benjamini_hochberg(ps):
    vals=np.asarray(ps,float);m=len(vals)
    order=np.argsort(vals);q=np.empty(m,float)
    sq=np.minimum.accumulate((vals[order]*m/np.arange(1,m+1))[::-1])[::-1]
    q[order]=np.minimum(1.,sq)
    return q

def evaluate(args,depth,probe_seed,data,tree,root):
    # Avoid overwriting original v2 data; cache both observed & shuffled runs.
    out=root/f'probe_seed{probe_seed}'/f'depth{depth}'
    out.mkdir(parents=True,exist_ok=True)
    existing=out/'observed.json'
    if existing.exists() and not args.force:
        obs=json.loads(existing.read_text())
    else:
        obs=None
    items=np.asarray(data['train_items']);feats=core.l2norm(np.asarray(data['emb']['text'][0]))
    objs,excluded=old.prep_node_splits(tree,items,depth,probe_seed)
    if excluded:write(out/'excluded.csv',excluded)
    if not objs:print(f'No eligible nodes at d{depth}',flush=True);return
    X,ps,y,tr,val,te,sizes,parents=v2.assemble(objs,feats,core.stable_seed(probe_seed,'pooled',depth))
    manifest=dict(teacher_seed=args.teacher_seed,depth=depth,probe_seed=probe_seed,tree_sha256=args.tree_sha,model='shared256_relu_parent_heads',wd=args.wd,epochs=args.epochs,patience=args.patience,
                  bootstrap=args.bootstrap,seed_split='v2_assemble',n_train=len(tr),n_val=len(val),n_test=len(te),parents=parents)
    mf=out/'manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=manifest and not args.force:
        raise RuntimeError(f'Changed configuration in {out}; change --out-root or use --force (recomputes).')
    mf.write_text(json.dumps(manifest,indent=2))
    if obs is None or args.force:
        ce,diag=train_eval(X,ps,y,tr,val,te,sizes,core.stable_seed(probe_seed,'actual',depth),args.hidden,args.wd,args.epochs,args.patience)
        R,prior,gain=score(y,ps,tr,te,sizes,ce)
        ci=bootstrap(gain,prior,ps[te],core.stable_seed(probe_seed,'boot',depth),args.bootstrap)
        parent_scores=[]
        for j,parent in enumerate(parents):
            ix=np.flatnonzero(ps[te]==j)
            rp=float(gain[ix].sum()/prior[ix].sum())
            lo,hi=old.bootstrap_ratio(gain[ix],prior[ix],args.bootstrap,core.stable_seed(probe_seed,'boot',depth,parent))
            parent_scores.append(dict(parent=int(parent),n_test=len(ix),R=rp,R_lo=lo,R_hi=hi,prior_ce=float(prior[ix].mean()),model_ce=float(ce[ix].mean())))
        obs=dict(depth=depth,probe_seed=probe_seed,whole=dict(R=R,R_lo=ci[0],R_hi=ci[1],n_test=len(te),**diag),parents=parent_scores)
        existing.write_text(json.dumps(obs,indent=2))
    else:
        print(f'  cached observed depth={depth} probe_seed={probe_seed}',flush=True)
    print(f'  observed d={depth} seed={probe_seed}: R={obs["whole"]["R"]:.5f}; CI [{obs["whole"]["R_lo"]:.5f},{obs["whole"]["R_hi"]:.5f}]',flush=True)
    null_rows=[]
    for r in range(args.permutations):
        cache=out/f'null_{r:03d}.json'
        if cache.exists() and not args.force:
            null=json.loads(cache.read_text())
        else:
            yp=permuted_labels(y,ps,tr,val,te,core.stable_seed(probe_seed,'perm',depth,r))
            ce,diag=train_eval(X,ps,yp,tr,val,te,sizes,core.stable_seed(probe_seed,'nullfit',depth,r),args.hidden,args.wd,args.epochs,args.patience)
            scoreall,prior,gain=score(yp,ps,tr,te,sizes,ce)
            each={str(parent):float(gain[ps[te]==j].sum()/prior[ps[te]==j].sum()) for j,parent in enumerate(parents)}
            null=dict(permutation=r,pooled_R=scoreall,parents=each,**diag)
            cache.write_text(json.dumps(null,indent=2))
        null_rows.append(null)
        if (r+1)%5==0 or r==0 or r+1==args.permutations:
            print(f'    null {r+1}/{args.permutations} complete',flush=True)
    pooled_null=np.array([n['pooled_R'] for n in null_rows])
    whole=dict(depth=depth,probe_seed=probe_seed,observed_R=obs['whole']['R'],R_lo=obs['whole']['R_lo'],R_hi=obs['whole']['R_hi'],
               null_mean=float(pooled_null.mean()),null_q95=float(np.quantile(pooled_null,.95)),null_max=float(pooled_null.max()),
               p_mc=float((1+(pooled_null>=obs['whole']['R']).sum())/(len(pooled_null)+1)),n_permutations=len(pooled_null),n_test=len(te))
    per=[]
    for parent_obs in obs['parents']:
        parent=int(parent_obs['parent']);nullv=np.array([n['parents'][str(parent)] for n in null_rows])
        pv=float((1+(nullv>=parent_obs['R']).sum())/(len(nullv)+1))
        per.append(dict(depth=depth,probe_seed=probe_seed,**parent_obs,null_mean=float(nullv.mean()),null_q95=float(np.quantile(nullv,.95)),p_mc=pv))
    q=benjamini_hochberg([r['p_mc'] for r in per])
    for row,v in zip(per,q):
        row['q_BH']=float(v)
        row['pass_preliminary']=bool(row['R_lo']>args.delta and row['p_mc']<=args.alpha and row['q_BH']<=args.alpha and row['n_test']>=args.min_test)
    write(out/'pooled_summary.csv',[whole]);write(out/'parent_decisions.csv',per)
    write(out/'null_distribution.csv',[dict(permutation=r['permutation'],pooled_R=r['pooled_R'],best_epoch=r['best_epoch'],val_ce=r['val_ce']) for r in null_rows])
    print(f'  permutation pooled p={whole["p_mc"]:.4f}, R_null_max={whole["null_max"]:.4f}',flush=True)

def aggregate(root,seeds,depths,delta,alpha,min_test):
    import pandas as pd
    whole=[];parents=[]
    for seed in seeds:
        for depth in depths:
            path=root/f'probe_seed{seed}'/f'depth{depth}'
            whole.extend(read(path/'pooled_summary.csv'))
            parents.extend(read(path/'parent_decisions.csv'))
    if whole:write(root/'pooled_summary_all.csv',whole)
    if parents:write(root/'parent_decisions_all.csv',parents)
    stable=[]
    for depth in depths:
        rd=[r for r in parents if int(r['depth'])==depth]
        for p in sorted({int(r['parent']) for r in rd}):
            sub=[r for r in rd if int(r['parent'])==p]
            rs=np.array([float(v['R']) for v in sub])
            prelim=sum(str(v['pass_preliminary']).lower()=='true' for v in sub)
            stable.append(dict(depth=depth,parent=p,seed_count=len(sub),mean_R=float(rs.mean()),min_R=float(rs.min()),max_R=float(rs.max()),positive_seeds=int((rs>0).sum()),prelim_pass_seeds=prelim,
                               stable_preliminary_pass=bool(len(sub)==len(seeds) and prelim==len(seeds))))
    if stable:write(root/'seed_stability.csv',stable)
    print('\n===== POOLED NULL / STABILITY SUMMARY =====',flush=True)
    for x in whole:print(f'depth={x["depth"]} seed={x["probe_seed"]} R={float(x["observed_R"]):+.5f} perm_p={float(x["p_mc"]):.4f}',flush=True)
    print(f'Saved: {root}',flush=True)
    print('IMPORTANT: Minimum possible p = 1/(permutations+1); with 19 perms p_min=0.05, BH q values cannot reliably pass 0.05 across 16 parents.',flush=True)
    print('Do not treat stable_preliminary_pass as final RCH acceptance: teacher selection on U2 and strict cold-item evaluation remain separate.',flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--domain',choices=['Industrial','Musical'],required=True)
    p.add_argument('--teacher-seed',type=int,default=42)
    p.add_argument('--probe-seeds',default='42,43,44')
    p.add_argument('--depths',default='2,3')
    p.add_argument('--permutations',type=int,default=19)
    p.add_argument('--bootstrap',type=int,default=400)
    p.add_argument('--delta',type=float,default=.01)
    p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--min-test',type=int,default=60)
    p.add_argument('--hidden',type=int,default=256)
    p.add_argument('--wd',type=float,default=.0001)
    p.add_argument('--epochs',type=int,default=70)
    p.add_argument('--patience',type=int,default=12)
    p.add_argument('--out-root',default='outputs/rch_pooled_null_stability_v1')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--force',action='store_true')
    args=p.parse_args()
    if args.permutations<1:raise ValueError('Need at least 1 null permutation.')
    if args.bootstrap<20:raise ValueError('Use >=20 bootstrap replicates.')
    os.chdir(REPO)
    seeds=[int(v) for v in args.probe_seeds.split(',')]
    depths=[int(v) for v in args.depths.split(',')]
    if args.smoke:seeds=seeds[:1];depths=depths[:1];args.permutations=2;args.epochs=5;args.patience=4;args.bootstrap=30
    ns=core.parse_args(['--domain',args.domain,'--seed',str(args.teacher_seed),'--mode','run'])
    data=core.load_server_data(ns)
    n=int(data['n']);tree_path=REPO/'outputs'/'rch_next_audit_v1'/'run'/f'{args.domain}_seed{args.teacher_seed}'/'spectral_recursive_tree_labels.npy'
    if not tree_path.exists():raise FileNotFoundError(tree_path)
    tree=np.load(tree_path)
    if tree.shape!=(n,3):raise ValueError(f'Expected {(n,3)} labels but got {tree.shape}')
    args.tree_sha=old.sha(tree_path)
    root=REPO/args.out_root/('smoke' if args.smoke else 'run')/f'{args.domain}_teacher{args.teacher_seed}'
    for seed in seeds:
        for depth in depths:
            print(f'\n===== depth={depth} probe_seed={seed} =====',flush=True)
            evaluate(args,depth,seed,data,tree,root)
    aggregate(root,seeds,depths,args.delta,args.alpha,args.min_test)
    print('\nRCH POOLED NULL STABILITY SUCCESS',flush=True)

if __name__=='__main__':main()
