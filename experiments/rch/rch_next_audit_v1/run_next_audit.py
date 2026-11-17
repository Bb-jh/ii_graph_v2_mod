#!/usr/bin/env python3
"""RCH next audit: degree/component and seed stability, spectral recursive P2,
locked-root U3. Uses existing reviewed RCH code and Exp2/3 make_data loader.

Important: item-heldout content probes use transductively learned U1 labels;
not a strict new-item cold-start experiment. Root fixed k4 seed0 before U3.
"""
import argparse, csv, hashlib, json, os, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments' / 'rch_probe'))
import run_rch_probe as core


def normalized_realizability(feats, y, k, cfg, seed):
    """Corrected R: prior minus actual probe CE, divided by prior CE.
    Shuffled-label training is a *separate* negative control.
    Both R and actual-vs-null CE gap have item-bootstrap 95% CIs.
    """
    folds = core.make_folds(y, cfg['folds'], seed)
    n = len(y)
    prior = np.empty(n)
    for tr, te in folds:
        prior[te] = core.prior_ce(y[tr], y[te], k)
    result = {}
    fitted = {}
    for mod in ('T', 'V'):
        X = feats[mod]
        C = core.select_c(X, y, folds[0][0], k, cfg['c_grid'], cfg['max_iter'], seed)
        ce = np.empty(n)
        null = np.zeros(n)
        for foldno, (tr, te) in enumerate(folds):
            ce[te] = core.fit_ce(X[tr], y[tr], X[te], y[te], k, C, cfg['max_iter'], seed)
            for perm in range(cfg['permutations']):
                rng = np.random.default_rng(core.stable_seed(seed, mod, foldno, perm))
                null[te] += core.fit_ce(X[tr], rng.permutation(y[tr]), X[te], y[te], k, C, cfg['max_iter'], seed)
            null[te] /= cfg['permutations']
        fitted[mod] = ce
        denom = float(prior.mean())
        effect = prior - ce
        control = null - ce
        meaningful = denom >= cfg['eps']
        ci = core.boot_ratio(effect, prior, cfg['bootstrap'], core.stable_seed(seed, mod, 'effect')) if meaningful else [float('nan')]*2
        cci = core.boot_ratio(control, prior, cfg['bootstrap'], core.stable_seed(seed, mod, 'control')) if meaningful else [float('nan')]*2
        result[mod] = dict(C=C, ce_prior=denom, ce_model=float(ce.mean()), ce_null=float(null.mean()),
                           dCE=float(effect.mean()), denominator=denom,
                           R=float(effect.mean()/denom) if meaningful else float('nan'), R_ci=ci,
                           control_gap=float(control.mean()), control_ci=cci)
    # Direct held-out CE difference from adding Image to Text; NOT conditional MI.
    X = np.concatenate([feats['T'], feats['V']], axis=1)
    dt = feats['T'].shape[1]
    C = core.select_c(X, y, folds[0][0], k, cfg['c_grid'], cfg['max_iter'], seed)
    ce = np.empty(n)
    null = np.zeros(n)
    for foldno, (tr, te) in enumerate(folds):
        ce[te] = core.fit_ce(X[tr], y[tr], X[te], y[te], k, C, cfg['max_iter'], seed)
        for perm in range(cfg['permutations']):
            rng = np.random.default_rng(core.stable_seed(seed, 'VT', foldno, perm))
            xn = X[tr].copy()
            xn[:, dt:] = xn[rng.permutation(len(tr)), dt:]
            xt = X[te].copy()
            xt[:, dt:] = xt[rng.permutation(len(te)), dt:]
            null[te] += core.fit_ce(xn, y[tr], xt, y[te], k, C, cfg['max_iter'], seed)
        null[te] /= cfg['permutations']
    den = fitted['T']
    ok = den.mean() >= cfg['eps']
    eff = den - ce
    ctrl = null - ce
    result['V|T'] = dict(C=C, ce_prior=float(prior.mean()), ce_model=float(ce.mean()),
                          ce_null=float(null.mean()), dCE=float(eff.mean()), denominator=float(den.mean()),
                          R=float(eff.mean()/den.mean()) if ok else float('nan'),
                          R_ci=core.boot_ratio(eff, den, cfg['bootstrap'], core.stable_seed(seed,'VT','effect')) if ok else [float('nan')]*2,
                          control_gap=float(ctrl.mean()),
                          control_ci=core.boot_ratio(ctrl, den, cfg['bootstrap'], core.stable_seed(seed,'VT','control')) if ok else [float('nan')]*2)
    return result


def corrected_pass(res, delta_rel, delta_abs):
    """Exploratory bootstrap gate: corrected effect lower CI > threshold,
    positive absolute effect, and actual probe better than shuffled control.
    This is NOT a calibrated permutation p-value.
    """
    return bool(np.isfinite(res['R_ci'][0]) and res['R_ci'][0] > delta_rel
                and res['dCE'] > delta_abs and res['control_ci'][0] > 0)

core.realizability = normalized_realizability
core.passes = corrected_pass
# Reused evaluate_split needs one extra output for the null control CI.
_old_evaluate = core.evaluate_split

def corrected_evaluate(*args, **kwargs):
    return _old_evaluate(*args, **kwargs)


def jsonwrite(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))


def adjacency_components(edges, items, n):
    a, b, w = core.local_edges(edges, items, n)
    m = len(items)
    A = sparse.coo_matrix((np.ones(len(a)*2, np.uint8),
                            (np.r_[a,b], np.r_[b,a])), shape=(m,m)).tocsr()
    A.setdiag(0); A.eliminate_zeros()
    ncomp, comp = connected_components(A, directed=False)
    degree_count = np.bincount(np.r_[a,b], minlength=m)
    degree_weight = np.bincount(np.r_[a,b], weights=np.r_[w,w], minlength=m)
    return ncomp, comp, degree_count, degree_weight


def quantile_labels(x, q):
    # Preserve ties: do not split many identical zero-degree items arbitrarily.
    thresholds = np.unique(np.quantile(x, np.linspace(0, 1, q+1)[1:-1]))
    return np.searchsorted(thresholds, x, side='left').astype(int)


def spectral_root(dat, seed, dim):
    items = np.asarray(dat['train_items'])
    z = core.behavior_spectral(dat['edges'][0], items, dat['n'], dim)
    assignments = {}
    for s in range(3):
        kmseed = core.stable_seed(seed, 'spectral', 4, s)
        y = MiniBatchKMeans(n_clusters=4, random_state=kmseed, n_init=3,
                             batch_size=min(2048,len(z)), max_iter=200).fit_predict(z)
        assignments[s] = y.astype(np.int32)
    return z, assignments


def degree_audit(dat, root_codes, feats, cfg, seed, out, jobs, smoke):
    items = np.asarray(dat['train_items']); n=dat['n']
    ncomp, comp, deg, wdeg = adjacency_components(dat['edges'][0], items, n)
    per_cluster=[]
    for s,y in root_codes.items():
        for c in range(4):
            idx = y==c; comps, counts = np.unique(comp[idx], return_counts=True)
            per_cluster.append(dict(seed=s, child=c, items=int(idx.sum()),
               mean_degree=float(deg[idx].mean()), median_degree=float(np.median(deg[idx])),
               p90_degree=float(np.quantile(deg[idx],.9)), mean_weighted_degree=float(wdeg[idx].mean()),
               frac_zero_degree=float((deg[idx]==0).mean()), components_touched=len(comps),
               dominant_component_fraction=float(counts.max()/idx.sum())))
    core.save_csv(out/'degree_clusters.csv', per_cluster)
    stability=[]
    for s in range(3):
        for t in range(s+1,3):
            stability.append(dict(seed_a=s,seed_b=t,ARI=float(adjusted_rand_score(root_codes[s],root_codes[t])),
                                  NMI=float(normalized_mutual_info_score(root_codes[s],root_codes[t]))))
    core.save_csv(out/'seed_stability.csv', stability)
    # Degree-only control (quartiles of actual U1 degree; ties preserved)
    degy = quantile_labels(deg, 4)
    children = len(np.unique(degy))
    if children >= 2:
        row = core.evaluate_split(dict(family='degree_only',name='degree_only_quartiles', k_requested=4),items,degy,
                                  dat['edges'],n,feats,cfg,core.stable_seed(seed,'degree_only'))
        core.save_csv(out/'degree_only_baseline.csv',[row])
    else:
        core.save_csv(out/'degree_only_baseline.csv',[dict(status='single_degree_group',n_groups=children)])
    # Within U1-degree quintile R: stratified heldout-item fit; fixed spectral k4 seed0.
    # Note small strata / absent classes produce NA rather than artificial zeros.
    groups=quantile_labels(deg,5)
    jobs_list=[]
    for g in np.unique(groups):
        indices=np.flatnonzero(groups==g)
        y=root_codes[0][indices]
        counts=np.bincount(y,minlength=4)
        if len(indices) < cfg['min_items'] or (counts>0).sum()<2 or counts[counts>0].min()<max(cfg['folds'],5):
            jobs_list.append(dict(stratum=int(g),n_items=len(indices),status='insufficient_classes',
                                  n_active_classes=int((counts>0).sum())))
            continue
        # Re-map present labels to contiguous classes; compare predictive advantage within same degree bin.
        _, ys=np.unique(y,return_inverse=True)
        subcfg=cfg.copy()
        subcfg['max_items']=min(cfg['max_items'],len(indices))
        jobs_list.append((g,indices,ys,subcfg))
    def run_stratum(job):
        g,indices,y,cf=job
        f={'T':feats['T'][items[indices]],'V':feats['V'][items[indices]]}
        res=normalized_realizability(f,y,int(y.max())+1,cf,core.stable_seed(seed,'stratum',g))
        r=res['T']; return dict(stratum=int(g),n_items=len(indices),n_active_classes=int(y.max())+1,
             R_T=r['R'],R_T_lo=r['R_ci'][0],R_T_hi=r['R_ci'][1],dCE_T=r['dCE'],
             null_control_gap_T=r['control_gap'],null_control_ci_lo_T=r['control_ci'][0],
             pass_T=corrected_pass(r,cf['delta_rel'],cf['delta_abs']),status='ok')
    computed=Parallel(n_jobs=jobs,verbose=5)(delayed(run_stratum)(j) for j in jobs_list if isinstance(j,tuple))
    core.save_csv(out/'degree_strata_R.csv',[j for j in jobs_list if isinstance(j,dict)]+computed)
    jsonwrite(out/'degree_info.json',dict(n_components=ncomp,largest_component_fraction=float(np.bincount(comp).max()/len(items)),
                zero_degree_fraction=float((deg==0).mean()),degree_quantile_group_sizes=np.bincount(groups).tolist(),
                degree_only_group_sizes=np.bincount(degy).tolist(),
                caution='Within-degree R>0 is evidence against pure degree-bin explanation, not proof of semantic collaborative groups.'))


def recursive_tree(dat, root_codes, cfg, seed, dim, out, jobs, smoke):
    items=np.asarray(dat['train_items']); n=dat['n']
    depth=3
    codes=np.full((n,depth),-1,dtype=np.int32)
    # Completely freeze root from P1: k4 seed0 (not refit on U2 or U3)
    codes[items,0]=root_codes[0]
    logs=[]; next_id=4
    parents={int(x):items[root_codes[0]==x] for x in np.unique(root_codes[0])}
    for level in (1,2):
        newer={}
        for parent, subitems in sorted(parents.items()):
            # At each node, rebuild normalized U1 graph; no U2/U3 enters construction.
            if len(subitems)<max(4*cfg['min_child'],cfg['min_items']):
                child=np.zeros(len(subitems),dtype=int); how='unsplit_small'
            else:
                try:
                    z=core.behavior_spectral(dat['edges'][0],subitems,n,min(dim,len(subitems)-2))
                    child=MiniBatchKMeans(n_clusters=4,random_state=core.stable_seed(seed,'rec',level,parent),
                        n_init=3,batch_size=min(2048,len(z)),max_iter=200).fit_predict(z)
                    how='spectral'
                except Exception as e:
                    # Explicitly record failure, never pretend this is a successful split.
                    raise RuntimeError(f'Spectral recursive node level={level+1}, parent={parent}, n={len(subitems)} failed') from e
            sizes=np.bincount(child)
            for j in np.unique(child):
                group=subitems[child==j]
                gid=next_id;next_id+=1
                codes[group,level]=gid
                newer[gid]=group
            logs.append(dict(depth=level+1,parent=parent,n_items=len(subitems),method=how,
                             n_children=len(np.unique(child)), min_child=int(sizes.min()),max_child=int(sizes.max())))
        parents=newer
    core.validate_hierarchy(codes,items,n)
    np.save(out/'spectral_recursive_tree_labels.npy',codes)
    core.save_csv(out/'spectral_tree_build.csv',logs)
    rows=core.run_p2(codes,items,dat['edges'],n,
                     {'T':core.l2norm(dat['emb']['text'][0]), 'V':core.l2norm(dat['emb']['image'][0])},
                     cfg,seed,jobs,smoke)
    summary,paths,decision=core.summarize_p2(rows,codes,items,cfg)
    core.save_csv(out/'spectral_p2_splits.csv',rows)
    core.save_csv(out/'spectral_p2_depth_summary.csv',summary)
    jsonwrite(out/'spectral_p2_paths.json',paths)
    jsonwrite(out/'spectral_p2_decision.json',decision)
    return decision


def u3_locked_root(dat, root_codes, out):
    if len(dat['edges'])<3 or dat['edges'][2] is None:
        jsonwrite(out/'u3_locked_root.json',{'status':'unavailable','reason':'make_data edges[2] missing'})
        return
    items=np.asarray(dat['train_items']);n=dat['n']
    a,b,w=core.local_edges(dat['edges'][2],items,n)
    m=core.modularity(root_codes[0],a,b,w,4)
    jsonwrite(out/'u3_locked_root.json', dict(status='ok',root='behavior_spectral_k4_s0',
                lock='P1 U2 selected BEFORE this U3 evaluation; do not retune using U3',
                Q_B_U3=m['Q'],within_frac_U3=m['within_frac'],n_edges_U3=m['n_edges']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain',choices=['Industrial','Musical'],required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--part',choices=['all','degree','recursive','u3'],default='all')
    p.add_argument('--mode',choices=['smoke','run'],default='run')
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--out-root',default='outputs/rch_next_audit_v1')
    p.add_argument('--spectral-dim',type=int,default=40)
    p.add_argument('--recursive-dim',type=int,default=12)
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--permutations',type=int,default=5)
    p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--max-items',type=int,default=12000)
    p.add_argument('--force',action='store_true',help='Allow replacing a completed stage only')
    a=p.parse_args()
    os.chdir(ROOT)
    ns=core.parse_args(['--domain',a.domain,'--seed',str(a.seed),'--mode','run'])
    dat=core.load_server_data(ns)
    dest=ROOT/a.out_root/a.mode/f'{a.domain}_seed{a.seed}'
    dest.mkdir(parents=True,exist_ok=True)
    cfg=dict(folds=3 if a.mode=='smoke' else a.folds,
             permutations=1 if a.mode=='smoke' else a.permutations,
             bootstrap=50 if a.mode=='smoke' else a.bootstrap,
             c_grid=[0.1] if a.mode=='smoke' else [.01,.1,1.],max_iter=200 if a.mode=='smoke' else 300,
             max_items=1500 if a.mode=='smoke' else a.max_items,
             min_items=60,min_child=5,eps=.05,delta_grid=[.01,.02,.05],delta_rel=.02,delta_abs=.005)
    items=np.asarray(dat['train_items'])
    zfile=dest/'root_codes.npz'
    if zfile.exists():
        z=np.load(zfile)
        roots={s:z[f's{s}'] for s in range(3)}
        assert all(len(v)==len(items) for v in roots.values()), 'stale root cache'
    else:
        print('Computing U1 spectral root, 3 k=4 seeds...',flush=True)
        _,roots=spectral_root(dat,a.seed,a.spectral_dim)
        np.savez_compressed(zfile,**{f's{s}':y for s,y in roots.items()})
    jsonwrite(dest/'manifest.json',dict(protocol='RCH_NEXT_AUDIT_V1', domain=a.domain, seed=a.seed,
          git_commit=os.popen('git rev-parse HEAD 2>/dev/null').read().strip(),config=vars(a),
          cfg=cfg, root='behavior_spectral_k4_s0',
          known_limitations=[
              'The P1 spectral embedding uses 40 eigenvectors even when k=4: full U1 normalized adjacency.',
              'Graph U1 transductively includes probe-heldout items; not true cold-item evaluation.',
              'Bootstrap CI is item-bootstrap of heldout CE, not a formal conditional permutation p-value.',
              'Different local modularities across node sizes are not directly comparable.',
              'U3 locked root is final one-time check; must not be used for root hyperparameter selection.']))
    feats={'T':core.l2norm(dat['emb']['text'][0]),'V':core.l2norm(dat['emb']['image'][0])}
    stages=['degree','recursive'] if a.part=='all' else [a.part]
    filenames={'degree':'DEGREE_DONE.json','recursive':'RECURSIVE_DONE.json','u3':'U3_DONE.json'}
    for stage in stages:
        if (dest/filenames[stage]).exists() and not a.force:
            print('SKIP completed stage',stage,flush=True);continue
        print('START',stage,flush=True);st=time.time()
        if stage=='degree':degree_audit(dat,roots,feats,cfg,a.seed,dest,a.jobs,a.mode=='smoke')
        if stage=='recursive':recursive_tree(dat,roots,cfg,a.seed,a.recursive_dim,dest,a.jobs,a.mode=='smoke')
        if stage=='u3':u3_locked_root(dat,roots,dest)
        jsonwrite(dest/filenames[stage], {'status':'success','stage':stage,'seconds':round(time.time()-st,1)})
        print('DONE',stage,'seconds',round(time.time()-st,1),flush=True)
    print('AUDIT SUCCESS',dest,flush=True)

if __name__=='__main__':main()
