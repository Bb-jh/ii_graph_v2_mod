#!/usr/bin/env python3
"""Industrial per-anchor neighbor popularity control, evaluation-only.

U1 defines graph, candidate edges, anchor and neighbor degree.
U2-trained P_base is loaded unchanged; U3 alone defines held-out outcomes/lift.
Directed neighbor slots are the primary analysis unit (edges may repeat both ways).
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from run_neighbor_reliability import (read_npz, reconstruct_degree, make_arcs,
                                      select_neighbor_slots)
from run_c1_pilot import read_users_train, csr_users


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def reconstruct_u3(sequences, n_items, selected_original, seed, fraction_u1, fraction_u2, expected_u3, a, b, cache):
    _, seqs = read_users_train(sequences, n_items)
    perm = np.random.default_rng(seed).permutation(len(seqs))
    n1 = int(round(len(seqs)*fraction_u1))
    n2 = int(round(len(seqs)*fraction_u2))
    indices = perm[n1+n2:]
    if len(indices) != expected_u3:
        raise ValueError(f'U3 size mismatch {len(indices)} != {expected_u3}')
    x3 = csr_users([seqs[int(i)] for i in indices], n_items)[:,selected_original].tocsr()
    d3 = np.asarray(x3.sum(axis=0)).ravel().astype(np.int64)
    co = (x3.T @ x3).tocsr()
    co.setdiag(0)
    co.eliminate_zeros()
    n_co = np.asarray(co[a,b]).ravel().astype(np.float64)
    expected_y3 = (n_co>0).astype(np.uint8)
    if not np.array_equal(expected_y3,np.asarray(cache['y3'],dtype=np.uint8)):
        bad = np.count_nonzero(expected_y3 != cache['y3'])
        raise ValueError(f'U3 recurrence fails cached alignment audit ({bad} edges)')
    if 'y3_2users' in cache and not np.array_equal((n_co>=2).astype(np.uint8),cache['y3_2users']):
        raise ValueError('U3 >=2-users mismatch')
    norm = np.divide(n_co,np.sqrt(d3[a].astype(float)*d3[b].astype(float)),
                     out=np.zeros_like(n_co),where=(d3[a]>0)&(d3[b]>0))
    if not np.allclose(norm,np.asarray(cache['strength3'],float),atol=2e-6,rtol=1e-4):
        raise ValueError('U3 normalized cooccurrence strength mismatch')
    expected = d3[a].astype(float) * d3[b].astype(float) / len(indices)
    valid = expected>0
    lift = np.full(len(a),np.nan,dtype=np.float64)
    lift[valid]=n_co[valid]/expected[valid]
    loglift = np.log1p(lift)
    return lift,loglift,valid,dict(u3_users=len(indices),zero_expected_edges=int((~valid).sum()),
                                    u3_item_active=int((d3>0).sum()))


def popularity_slots(k,eid,anchor,neighbor,wb,degree):
    # anchor asc, degree(neighbor) desc, wb desc, neighbor id asc
    ordering=np.lexsort((neighbor,-wb[eid],-degree[neighbor],anchor))
    ar=anchor[ordering]
    first=np.r_[True,ar[1:]!=ar[:-1]]
    start=np.maximum.accumulate(np.where(first,np.arange(len(ar)),0))
    return ordering[(np.arange(len(ar))-start)<k]


def mean_and_valid(values):
    vals=np.asarray(values,float)
    valid=np.isfinite(vals)
    return (float(vals[valid].mean()) if valid.any() else float('nan'),float(valid.mean()))


def metrics(idx,eid,anchor,neighbor,degree,cache,lift,loglift,universe):
    e=eid[idx]
    ne=degree[neighbor[idx]]
    ml,valid=mean_and_valid(lift[e]); mlog,_=mean_and_valid(loglift[e])
    # Zero-fill sensitivity: undefined lift is assigned zero as a diagnostic only.
    return dict(n_directed_slots=len(idx),n_unique_undirected_edges=len(np.unique(e)),
                anchor_coverage=len(np.unique(anchor[idx]))/universe,
                n_covered_anchors=len(np.unique(anchor[idx])),
                recurrence=float(np.mean(cache['y3'][e])),
                norm_strength=float(np.mean(cache['strength3'][e])),
                mean_lift=ml,mean_log1p_lift=mlog,lift_defined_fraction=valid,
                mean_lift_zero_filled=float(np.mean(np.nan_to_num(lift[e],nan=0))),
                mean_log1p_lift_zero_filled=float(np.mean(np.nan_to_num(loglift[e],nan=0))),
                mean_log_neighbor_degree=float(np.log1p(ne).mean()),
                neighbor_degree_median=float(np.median(ne)),
                neighbor_degree_p90=float(np.percentile(ne,90)))


def summarize_anchor(idx,anchor,metric,n):
    """Anchor-aggregated sums and valid counts (omit NaN for lift)."""
    v=metric[idx]; anch=anchor[idx]; valid=np.isfinite(v)
    return (np.bincount(anch[valid],weights=v[valid],minlength=n),
            np.bincount(anch[valid],minlength=n).astype(float))


def boot_delta(ia,ib,anchor,metric,n,reps,rng,scale=1):
    sa,ca=summarize_anchor(ia,anchor,metric,n)
    sb,cb=summarize_anchor(ib,anchor,metric,n)
    if not ca.sum() or not cb.sum():
        return float('nan'),float('nan'),float('nan'),0
    obs=(sa.sum()/ca.sum()-sb.sum()/cb.sum())*scale
    results=[]
    for _ in range(reps):
        # Same resampled anchors for both methods, using exact sampling with replacement.
        w=np.bincount(rng.integers(0,n,size=n),minlength=n).astype(float)
        ca_r=w@ca; cb_r=w@cb
        if ca_r>0 and cb_r>0:results.append(scale*((w@sa)/ca_r-(w@sb)/cb_r))
    return float(obs),float(np.quantile(results,.025)),float(np.quantile(results,.975)),len(results)


def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    d=read_npz(args.input/'routing_candidates.npz')
    e=read_npz(args.input/'expressibility_scores.npz')
    pm=json.loads((args.input/'prepare_manifest.json').read_text())
    if int(pm['seed']) != args.seed:raise ValueError('Seed mismatch')
    a=np.asarray(d['a'],int);b=np.asarray(d['b'],int)
    nedge=len(a)
    if args.expected_edges is not None and nedge!=args.expected_edges:raise ValueError('Unexpected candidate count')
    n=int(pm['eligible_items'])
    degree,sel,info=reconstruct_degree(args.mapping,args.sequences,args.embeddings,args.seed,args.u1_fraction,n)
    if 'bfeatures' not in d:raise KeyError('bfeatures missing')
    ef=d['bfeatures']
    if not (np.allclose(ef[:,0],np.log1p(np.minimum(degree[a],degree[b])),atol=1e-4,rtol=1e-5) and
            np.allclose(ef[:,1],np.log1p(np.maximum(degree[a],degree[b])),atol=1e-4,rtol=1e-5)):
        raise ValueError('Cached behavior degree alignment mismatch')
    pbase=np.asarray(e['pbase_T_full'],float)
    if 'pbase_V_full' in e and not np.allclose(pbase,e['pbase_V_full'],atol=1e-5):
        raise ValueError('Pbase T/V mismatch')
    if len(pbase)!=nedge:raise ValueError('Pbase length mismatch')
    wb=np.asarray(d['wb'],float)
    if not np.isfinite(pbase).all() or not np.isfinite(wb).all():raise ValueError('Nonfinite score')
    lift, loglift, liftvalid, aud = reconstruct_u3(args.sequences,info['n_mapping_items'],sel,args.seed,
                              args.u1_fraction,args.u2_fraction,int(pm['users_u3']),a,b,d)
    eid,anchor,neighbor=make_arcs(a,b,degree,n)
    counts=np.bincount(anchor,minlength=n)
    covered=np.flatnonzero(counts>0)
    if not len(covered):raise ValueError('No anchors')
    # U1 neighbor degree tertiles, calculated on all candidate item endpoints, U3 not used.
    threshold=np.quantile(degree[covered].astype(float),[1/3,2/3])
    neighbor_group=np.searchsorted(threshold,degree,side='right')
    names=('Low','Mid','High')
    rng=np.random.default_rng(args.seed+19471)
    rows=[]; strata=[]; deltas=[]; audits=[]
    values={'recurrence':np.asarray(d['y3'],float)[eid],
            'norm_strength':np.asarray(d['strength3'],float)[eid],
            'lift':lift[eid], 'log1p_lift':loglift[eid]}
    for k in args.k:
        indices={
            'W_B':select_neighbor_slots(wb,k,eid,anchor,neighbor,n),
            'P_base':select_neighbor_slots(pbase,k,eid,anchor,neighbor,n),
            'Popularity':popularity_slots(k,eid,anchor,neighbor,wb,degree)}
        for name,idx in indices.items():
            stat=metrics(idx,eid,anchor,neighbor,degree,d,lift,loglift,n)
            rows.append(dict(dataset=args.dataset,seed=args.seed,k=k,method=name,**stat))
            for g,label in enumerate(names):
                j=idx[neighbor_group[neighbor[idx]]==g]
                if not len(j):continue
                strata.append(dict(dataset=args.dataset,seed=args.seed,k=k,method=name,
                                   neighbor_degree_group=label,**metrics(j,eid,anchor,neighbor,degree,d,lift,loglift,n)))
        anchor_budget=np.bincount(anchor[indices['W_B']],minlength=n)
        for name,idx in indices.items():
            if not np.array_equal(anchor_budget,np.bincount(anchor[idx],minlength=n)):
                raise ValueError('Per-anchor neighbor budget not matched')
        audits.append(dict(k=k,candidate_anchors=len(covered),directed_slots=len(indices['W_B']),
                     expected_slots=int(np.minimum(counts,k).sum()),
                     popularity_tiebreak='neighbor U1 degree desc, wB desc, neighbor ID asc',
                     mean_lift_defined_candidate=float(liftvalid.mean())))
        for a_name,b_name in [('P_base','W_B'),('P_base','Popularity')]:
            for group,label in [(None,'All')]+list(enumerate(names)):
                ia=indices[a_name];ib=indices[b_name]
                if group is not None:
                    ia=ia[neighbor_group[neighbor[ia]]==group]
                    ib=ib[neighbor_group[neighbor[ib]]==group]
                for metric_name,v in values.items():
                    scale=100 if metric_name=='recurrence' else 1
                    if len(ia)==0 or len(ib)==0:continue
                    delta,lo,hi,valid=boot_delta(ia,ib,anchor,v,n,args.bootstrap,rng,scale)
                    deltas.append(dict(dataset=args.dataset,seed=args.seed,k=k,
                                       comparison=a_name+'_minus_'+b_name,
                                       neighbor_degree_group=label,metric=metric_name,
                                       unit='percentage_points' if metric_name=='recurrence' else 'raw',
                                       delta=delta,ci95_low=lo,ci95_high=hi,
                                       bootstrap_valid=valid,bootstrap_reps=args.bootstrap,
                                       selected_slots_a=len(ia),selected_slots_b=len(ib)))
        print(f"k={k}: W_B={100*next(r['recurrence'] for r in rows if r['k']==k and r['method']=='W_B'):.3f}% "
              f"P_base={100*next(r['recurrence'] for r in rows if r['k']==k and r['method']=='P_base'):.3f}% "
              f"Popularity={100*next(r['recurrence'] for r in rows if r['k']==k and r['method']=='Popularity'):.3f}%",flush=True)
    write_csv(args.out/'selection_summary.csv',rows)
    write_csv(args.out/'neighbor_degree_strata.csv',strata)
    write_csv(args.out/'paired_anchor_bootstrap_ci.csv',deltas)
    write_csv(args.out/'audit.csv',audits)
    manifest=dict(dataset=args.dataset,seed=args.seed,neighbor_k=args.k,bootstrap=args.bootstrap,
          item_crossfitting='NOT PERFORMED: cached P_base is fitted on U2, historic routing E uses edge-based 3-fold OOF; U3 held out',
          ranking_data='U1 behavior / degrees; U2 P_base fit; no U3 selection',
          evaluation='directed anchor-neighbor slots, paired anchored bootstrap',
          neighbor_degree_thresholds=threshold.tolist(),neighbor_degree_basis='U1 train unique-user degree',
          lift='observed U3 pair cooccurring users / (U3 individual item user counts product / number U3 users)',
          lift_undefined='if either U3 item count is zero, lift is NaN, excluded from primary mean and bootstrap; fraction reported; zero-fill sensitivity separately',
          lift_warning='rare expected counts yield large lift; log1p-lift and finite fraction are essential diagnostics',
          bootstrap_notes='resample source anchors, fixed U3 and fitted Pbase; does not fully account for shared neighbor dependence or split/model uncertainty',
          degree_strata_notes='condition on selected neighbor degree groups, not strict degree-matched selection; contrasts can differ in anchor composition',
          candidates=nedge,items=n,source_anchors=len(covered),degree_info=info,**aud)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print('SUCCESS',args.out,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--mapping',type=Path,required=True)
    ap.add_argument('--sequences',type=Path,required=True)
    ap.add_argument('--embeddings',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--dataset',default='Industrial_and_Scientific')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--u1-fraction',type=float,default=.5)
    ap.add_argument('--u2-fraction',type=float,default=.25)
    ap.add_argument('--expected-edges',type=int,default=149825)
    ap.add_argument('--k',type=int,nargs='+',default=[5,10,20])
    ap.add_argument('--bootstrap',type=int,default=500)
    args=ap.parse_args()
    if args.bootstrap<50:ap.error('bootstrap >= 50')
    run(args)

if __name__=='__main__':main()
