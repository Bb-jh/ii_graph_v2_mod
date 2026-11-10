#!/usr/bin/env python3
"""Read-only audit of Experiment 2 v2 paired U3 code-MI bootstrap.
No retraining and no modifications to existing experimental results.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments'/'discrete_mi_portable_v2'
sys.path.insert(0,str(EXP))
import offdiag_core as core

PAIRS=[('code_mi','infonce'),('code_mi','rq'),('infonce','rq')]

def mi(c,a,b,w,k):
    return core.measures(core.code_joint(c,a,b,w,k))['mi_real']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--domain',choices=['Industrial','Musical'],required=True)
    p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--seed',type=int,default=20261111)
    p.add_argument('--out-root',default='outputs/discrete_mi_bootstrap_audit_v1')
    p.add_argument('--source-root',default='outputs/discrete_mi_level1_v2_portable')
    a=p.parse_args()
    src=ROOT/a.source_root/f'{a.domain}_K256_seed42'
    if not (src/'DONE.json').is_file():raise FileNotFoundError(f'Completed V2 output missing: {src}')
    config=json.loads((src/'config.json').read_text())
    k=int(config['k']);split_seed=int(config['seed'])
    if a.bootstrap<100:raise ValueError('At least 100 bootstrap draws required')
    # Identical data resolution, eligible-item intersection and stratified user split as V2.
    dataset,seqname=core.DOMAINS[a.domain]
    from run_level1_v2 import resolve_input
    from argparse import Namespace
    ns=Namespace(mapping_path=config.get('mapping_path'),embedding_path=config.get('embedding_path'),sequence_path=config.get('sequence_path'))
    mapping=resolve_input('mapping',dataset,seqname,ns.mapping_path)
    embfile=resolve_input('embedding',dataset,seqname,ns.embedding_path)
    seqfile=resolve_input('sequence',dataset,seqname,ns.sequence_path)
    ids=core.load_mapping(mapping);n=len(ids)
    _,seqs=core.read_users_train(seqfile,n)
    groups,_=core.split_stratified(seqs,split_seed)
    embeddings=core.load_embeddings(embfile,ids)
    eligible=np.flatnonzero(embeddings['text'][1]&embeddings['image'][1]).astype(np.int32)
    x=core.csr_users([seqs[int(i)] for i in groups[2]],n)
    aa,bb,ww,_=core.edges_from_users(x,eligible)
    print(f'U3 reconstructed: {len(aa)} edges, n_items={n}, k={k}',flush=True)
    codes={}
    for mod in ('text','image'):
        for method in ('rq','infonce','code_mi'):
            f=src/mod/f'{method}_codes.npy'
            if not f.is_file():raise FileNotFoundError(f)
            c=np.load(f,allow_pickle=False)
            if len(c)!=n:raise ValueError(f'Item count mismatch: {f}, {len(c)} != {n}')
            codes[(mod,method)]=c.astype(np.int32,copy=False)
    out=ROOT/a.out_root/f'{a.domain}_K{k}_seed{split_seed}'
    if out.exists():raise FileExistsError(f'Audit output already exists, refusing overwrite: {out}')
    out.mkdir(parents=True)
    old={}
    old_file=src/'paired_comparisons.csv'
    if old_file.exists():
        with old_file.open(newline='') as f:
            old={(r['modality'],r['lhs'],r['rhs']):r for r in csv.DictReader(f)}
    results=[]; arrays={}
    for mod in ('text','image'):
        cset={method:codes[(mod,method)] for method in ('rq','infonce','code_mi')}
        point={method:mi(c,aa,bb,ww,k) for method,c in cset.items()}
        rng=np.random.default_rng(a.seed+(0 if mod=='text' else 1))
        draws={pair:np.empty(a.bootstrap,dtype=np.float64) for pair in PAIRS}
        # All comparisons within modality receive the exact same item resampling.
        for t in range(a.bootstrap):
            counts=rng.poisson(1,size=n)
            w=ww*counts[aa]*counts[bb]
            if not np.any(w>0):raise RuntimeError('A bootstrap draw has no positive-weight edges')
            jm={method:mi(c,aa,bb,w,k) for method,c in cset.items()}
            for pair in PAIRS:draws[pair][t]=jm[pair[0]]-jm[pair[1]]
            if (t+1)%100==0:print(mod,'bootstrap',t+1,'/',a.bootstrap,flush=True)
        for lhs,rhs in PAIRS:
            obs=point[lhs]-point[rhs]; d=draws[(lhs,rhs)]
            qlo,qhi=np.quantile(d,[.025,.975]);mean=float(d.mean())
            # Basic bootstrap uses reflected quantiles, a bias sensitivity diagnostic.
            blo,bhi=2*obs-qhi,2*obs-qlo
            centered_lo,centered_hi=obs+np.quantile(d-mean,[.025,.975])
            oldrow=old.get((mod,lhs,rhs),{})
            row=dict(domain=a.domain,modality=mod,lhs=lhs,rhs=rhs,
                J_lhs=point[lhs],J_rhs=point[rhs],observed_J_diff=obs,
                bootstrap_mean=mean,bootstrap_sd=float(d.std(ddof=1)),
                bootstrap_bias=mean-obs,percentile_lo=float(qlo),percentile_hi=float(qhi),
                basic_lo=float(blo),basic_hi=float(bhi),
                mean_centered_lo=float(centered_lo),mean_centered_hi=float(centered_hi),
                percentile_excludes_zero=bool(qlo>0 or qhi<0),
                basic_excludes_zero=bool(blo>0 or bhi<0),
                old_J_diff=oldrow.get('J_diff',''),old_CI_low=oldrow.get('J_diff_ci_low',''),old_CI_high=oldrow.get('J_diff_ci_high',''))
            results.append(row)
            arrays[f'{mod}_{lhs}_minus_{rhs}']=d
            print(f'{mod} {lhs}-{rhs}: point {obs:+.6f}, boot mean {mean:+.6f}, percentile [{qlo:+.6f},{qhi:+.6f}], basic [{blo:+.6f},{bhi:+.6f}]',flush=True)
    with (out/'paired_J_bootstrap_audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    np.savez_compressed(out/'paired_bootstrap_draws.npz',**arrays)
    (out/'audit_manifest.json').write_text(json.dumps(dict(source=str(src),bootstrap=a.bootstrap,seed=a.seed,split_seed=split_seed,k=k,n_items=n,n_U3_edges=len(aa),null_recomputed=False,method='paired Poisson(1) item multiplier with shared samples, raw full-U3 code MI; percentile vs basic vs mean-centered CI',note='These CI methods are sensitivity analyses, not independent confirmation of valid coverage; finite-sample nonlinear MI bias can be substantial.'),indent=2)+'\n')
    print('AUDIT SUCCESS:',out,flush=True)

if __name__=='__main__':main()
