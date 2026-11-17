#!/usr/bin/env python3
"""R1-lite: frozen U1 spectral tree, Text held-out R vs depth.

Requires experiments/rch_probe/run_rch_probe.py and the earlier
rch_next_audit_v1 spectral_recursive_tree_labels.npy for k4 reproducibility.
No U3 consumed. No RCH selection. Not an inductive cold-item experiment.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time, hashlib
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO/'experiments'/'rch_probe'))
import run_rch_probe as core


def write_csv(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def evaluate_text(nodes, y, feats, edges, n, cfg, seed):
    """Corrected held-out R and item-bootstrap CI, with separate shuffled-label control."""
    y0=np.asarray(y)
    _, yy=np.unique(y0,return_inverse=True)
    k=int(yy.max())+1
    cnt=np.bincount(yy,minlength=k)
    row=dict(n_items=len(nodes),n_children=k,min_child=int(cnt.min()),max_child=int(cnt.max()))
    if k<2 or cnt.min()<cfg['min_child']:
        return dict(row,status='insufficient_child')
    a,b,w=core.local_edges(edges[1],nodes,n)
    q=core.modularity(yy,a,b,w,k)
    row.update(Q_B_U2=q['Q'],within_frac_U2=q['within_frac'],n_edges_U2=q['n_edges'])
    # Stratified sample preserves rare child classes.
    keep=np.arange(len(nodes))
    if len(keep)>cfg['max_items']:
        rng=np.random.default_rng(core.stable_seed(seed,'sample'))
        alloc=np.full(k,max(cfg['folds'],cfg['min_child']),dtype=int)
        if alloc.sum()>cfg['max_items']:
            return dict(row,status='sample_budget_exceeded')
        capacity=cnt-alloc
        if np.any(capacity<0): return dict(row,status='class_too_small')
        remain=cfg['max_items']-alloc.sum()
        extras=remain*capacity/max(capacity.sum(),1)
        alloc+=np.minimum(capacity,np.floor(extras).astype(int))
        remain=cfg['max_items']-alloc.sum()
        for j in np.argsort(-(extras-np.floor(extras))):
            if remain==0:break
            take=min(remain, max(cnt[j]-alloc[j],0))
            alloc[j]+=take;remain-=take
        keep=np.sort(np.concatenate([rng.choice(np.flatnonzero(yy==j),int(alloc[j]),replace=False) for j in range(k)]))
    yy=yy[keep]; xx=feats[nodes[keep]]
    folds=core.make_folds(yy,cfg['folds'],seed)
    prior=np.empty(len(yy)); ce=np.empty(len(yy)); null=np.empty(len(yy))
    # As in previous code, choose C from the first fold's training data.
    C=core.select_c(xx,yy,folds[0][0],k,cfg['c_grid'],cfg['max_iter'],seed)
    for foldno,(tr,te) in enumerate(folds):
        prior[te]=core.prior_ce(yy[tr],yy[te],k)
        ce[te]=core.fit_ce(xx[tr],yy[tr],xx[te],yy[te],k,C,cfg['max_iter'],seed)
        nc=np.zeros(len(te))
        for perm in range(cfg['permutations']):
            rng=np.random.default_rng(core.stable_seed(seed,'perm',foldno,perm))
            nc+=core.fit_ce(xx[tr],rng.permutation(yy[tr]),xx[te],yy[te],k,C,cfg['max_iter'],seed)
        null[te]=nc/cfg['permutations']
    den=float(prior.mean()); gain=prior-ce; ctrl=null-ce
    if den<0.05: return dict(row,status='prior_entropy_low',den_T=den)
    lo,hi=core.boot_ratio(gain,prior,cfg['bootstrap'],core.stable_seed(seed,'bootR'))
    clo,chi=core.boot_ratio(ctrl,prior,cfg['bootstrap'],core.stable_seed(seed,'bootNull'))
    R=float(gain.mean()/den)
    row.update(status='ok',n_items_R=len(yy),C_T=C,den_T=den,ce_model_T=float(ce.mean()),
               ce_null_T=float(null.mean()),R_T=R,R_T_lo=lo,R_T_hi=hi,
               null_control_gap_T=float(ctrl.mean()),null_control_ci_lo_T=clo,
               pass_T=bool(lo>cfg['delta_rel'] and gain.mean()>cfg['delta_abs'] and clo>0))
    return row


def plot_curve(rows,path):
    try:import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    except ImportError:
        print('Plot skipped (optional matplotlib); all CSV saved',flush=True);return
    fig,ax=plt.subplots(figsize=(8,4.5))
    for family in ('k4','k16'):
        r=sorted((x for x in rows if x['family']==family),key=lambda x:x['depth'])
        if not r:continue
        ax.plot([x['depth'] for x in r],[x['frac_evaluated_pass_T'] for x in r],marker='o',label=f'{family} evaluated pass')
        ax.plot([x['depth'] for x in r],[x['frac_all_nodes_pass_T'] for x in r],marker='x',linestyle='--',label=f'{family} total pass')
    ax.set(xlabel='Tree depth (root=1)',ylabel='Fraction passing corrected R',ylim=(-0.05,1.05));ax.legend()
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain',choices=['Industrial','Musical'],required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--family',choices=['both','k4','k16'],default='both')
    p.add_argument('--mode',choices=['smoke','run'],default='run')
    p.add_argument('--jobs',type=int,default=1,help='reserved, run sequentially for resumable per-node caching')
    p.add_argument('--max-depth-k4',type=int,default=5)
    p.add_argument('--max-depth-k16',type=int,default=3)
    p.add_argument('--min-leaf',type=int,default=60)
    p.add_argument('--bootstrap',type=int,default=200)
    p.add_argument('--permutations',type=int,default=2)
    p.add_argument('--out-root',default='outputs/rch_depth_curve_v1')
    args=p.parse_args()
    os.chdir(REPO)
    ns=core.parse_args(['--domain',args.domain,'--seed',str(args.seed),'--mode','run'])
    data=core.load_server_data(ns)
    items=np.asarray(data['train_items']);n=data['n']
    feats=core.l2norm(data['emb']['text'][0])
    cfg=dict(folds=3,
             c_grid=[.1] if args.mode=='smoke' else [.01,.1,1.],
             max_iter=120 if args.mode=='smoke' else 200,
             max_items=800 if args.mode=='smoke' else 6000,
             bootstrap=40 if args.mode=='smoke' else args.bootstrap,
             permutations=1 if args.mode=='smoke' else args.permutations,
             min_child=5, delta_rel=.02,delta_abs=.005)
    dest=REPO/args.out_root/args.mode/f'{args.domain}_seed{args.seed}'
    dest.mkdir(parents=True,exist_ok=True)
    cfg_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    oldpath=REPO/'outputs'/'rch_next_audit_v1'/'run'/f'{args.domain}_seed{args.seed}'/'spectral_recursive_tree_labels.npy'
    oldroot=REPO/'outputs'/'rch_next_audit_v1'/'run'/f'{args.domain}_seed{args.seed}'/'root_codes.npz'
    if 'k4' in args.family or args.family=='both':
        if not oldpath.exists():
            raise SystemExit(f'Need original frozen k4 tree: {oldpath}\nRun prior RCH next audit --part recursive first; do not silently refit root.')
        old=np.load(oldpath,mmap_mode='r')
        assert old.shape==(n,3),f'Shape mismatch {old.shape} vs {(n,3)}'
        if oldroot.exists():
            rr=np.load(oldroot)['s0']
            assert np.array_equal(old[items,0],rr),'k4 root labels mismatch'
    else:old=None
    summaries=[]
    families=['k4','k16'] if args.family=='both' else [args.family]
    for family in families:
        k=int(family[1:]);maxdepth=args.max_depth_k4 if k==4 else args.max_depth_k16
        # With >=60 per leaf, full k16 depth3 (4096 leaves) is infeasible here.
        if family=='k4':
            parent_nodes=[(0,items,None)]
        else:
            print('Generating k16 root using same U1 spectral generator, seed=0',flush=True)
            zz=core.behavior_spectral(data['edges'][0],items,n,40)
            yy=MiniBatchKMeans(n_clusters=k,random_state=core.stable_seed(args.seed,'spectral',k,0),
                n_init=3,batch_size=min(2048,len(zz)),max_iter=200).fit_predict(zz)
            parent_nodes=[(0,items,yy)]
        nextid=1
        for depth in range(1,maxdepth+1):
            print(f'[{family}] depth {depth}: parents={len(parent_nodes)}',flush=True)
            children=[];rows=[]
            for nodeid,sub,root_y in parent_nodes:
                tag=dict(family=family,depth=depth,parent=int(nodeid),n_items=int(len(sub)))
                filepath=dest/f'{family}_depth{depth}_parent{nodeid}.json'
                nextnodes=[]
                # Child IDs recomputed consistently from the frozen old tree for d<=3.
                if depth<=3 and family=='k4':
                    y=np.asarray(old[sub,depth-1],dtype=np.int64)
                    vals=np.unique(y)
                    if np.any(vals<0): raise ValueError('missing saved label')
                    child_sizes=np.array([(y==j).sum() for j in vals])
                    construction='frozen_prior_k4'
                elif depth==1 and family=='k16':
                    y=root_y;vals=np.unique(y);child_sizes=np.bincount(y);construction='spectral_root'
                else:
                    if len(sub)<k*args.min_leaf:
                        rows.append(dict(tag,status='stopped_parent_too_small',min_leaf=args.min_leaf))
                        continue
                    zz=core.behavior_spectral(data['edges'][0],sub,n,min(12,len(sub)-2))
                    y=MiniBatchKMeans(n_clusters=k,random_state=core.stable_seed(args.seed,'depth_curve',family,depth,nodeid),
                        n_init=3,batch_size=min(2048,len(zz)),max_iter=200).fit_predict(zz)
                    vals=np.unique(y);child_sizes=np.array([(y==j).sum() for j in vals]);construction='new_spectral'
                if len(vals)!=k or child_sizes.min()<args.min_leaf:
                    rows.append(dict(tag,status='stopped_child_too_small',min_child=int(child_sizes.min()),
                                     min_leaf=args.min_leaf,construction=construction))
                    continue
                if filepath.exists():
                    row=json.loads(filepath.read_text())
                    if row.get('n_items')!=len(sub) or row.get('min_leaf')!=args.min_leaf or row.get('eval_config_hash')!=cfg_hash:
                        raise RuntimeError(f'Stale cache: {filepath}')
                else:
                    ss=core.stable_seed(args.seed,'depth_curve',family,depth,nodeid)
                    row={**tag,**evaluate_text(sub,y,feats,data['edges'],n,cfg,ss),
                         'construction':construction,'min_leaf':args.min_leaf,'eval_config_hash':cfg_hash}
                    filepath.write_text(json.dumps(row,indent=2,default=float))
                rows.append(row)
                if depth<maxdepth:
                    for v in vals:
                        children.append((nextid,sub[y==v],None));nextid+=1
                print(f'  node {nodeid} n={len(sub)} min_child={child_sizes.min()} R={row.get("R_T",float("nan")):.4f} pass={row.get("pass_T")}',flush=True)
                if args.mode=='smoke':break
            write_csv(dest/f'{family}_depth{depth}_splits.csv',rows)
            evaluated=[r for r in rows if r.get('status')=='ok']
            passed=[r for r in evaluated if r.get('pass_T')]
            total_n=sum(r['n_items'] for r in rows)
            entry=dict(family=family,depth=depth,total_parent_nodes=len(rows),evaluated=len(evaluated),
               stopped=len(rows)-len(evaluated),frac_evaluated_pass_T=len(passed)/len(evaluated) if evaluated else float('nan'),
               frac_all_nodes_pass_T=len(passed)/len(rows) if rows else float('nan'),
               item_weighted_pass_T=sum(r['n_items'] for r in passed)/max(total_n,1),
               median_R_T=float(np.median([r['R_T'] for r in evaluated])) if evaluated else float('nan'),
               n_items_parent_total=int(total_n),min_leaf=args.min_leaf)
            summaries.append(entry);write_csv(dest/'depth_curve_summary.csv',summaries)
            if args.mode=='smoke':break
            if not children:break
            parent_nodes=children
    plot_curve(summaries,dest/'depth_curve.png')
    manifest=dict(domain=args.domain,seed=args.seed,config=vars(args),
       input_tree=str(oldpath) if old is not None else None,
       input_tree_sha256=hashlib.sha256(Path(oldpath).read_bytes()).hexdigest() if old is not None else None,
       caution=['Exploratory; U2 is reused after P1 selection',
                'U1 transductively contains probe-heldout items; not inductive cold items',
                'R is prior-vs-model CE reduction; null only as separate check',
                'Depth with no evaluable splits is capacity limited, not a realizability failure',
                'k4 previously generated d<=3 uses original unscreened spectral tree'])
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('DEPTH CURVE SUCCESS',dest)

if __name__=='__main__':main()
