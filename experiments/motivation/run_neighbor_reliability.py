#!/usr/bin/env python3
"""Per-anchor top-k behavior-edge neighbor selection, held-out U3 validation.

No parameter fitting. Uses cached U2-fitted P(Y|B) and U1 train item degrees.
Each undirected candidate is treated as two directed anchor->neighbor arcs.
The primary estimand is directed neighbor-slot recurrence, not union-edge recurrence.
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np


def read_npz(p):
    if not p.is_file(): raise FileNotFoundError(p)
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}

def csvwrite(p,rows):
    if not rows: return
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def reconstruct_degree(mapping, sequences, embeddings, seed, u1_fraction, expected_items):
    with mapping.open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f,delimiter='\t'))
    ids=[]
    for i,r in enumerate(rows):
        if int(r['item_id'])!=i: raise ValueError('item_mapping.tsv not zero-based sequential')
        ids.append(r['parent_asin'])
    users={}
    with sequences.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                u,it=map(int,line.split())
                if it<1 or it>len(ids):raise ValueError('Sequence item index out of range')
                users.setdefault(u,[]).append(it-1)
    users_sorted=[users[u] for u in sorted(users)]
    rng=np.random.default_rng(seed)
    perm=rng.permutation(len(users_sorted))
    n1=int(round(len(users_sorted)*u1_fraction))
    if n1<=0 or n1>=len(users_sorted):raise ValueError('Invalid split')
    degree=np.zeros(len(ids),dtype=np.int32)
    for ix in perm[:n1]:
        s=users_sorted[int(ix)]
        if len(s)<3:raise ValueError('Expected leave-two-out sequence')
        degree[np.unique(np.asarray(s[:-2],dtype=np.int32))]+=1
    with np.load(embeddings,allow_pickle=False) as z:
        embids=[str(a) for a in z['item_ids']]
        lookup={asin:i for i,asin in enumerate(embids)}
        ix=np.array([lookup.get(x,-1) for x in ids],dtype=np.int64)
        if (ix<0).any():raise ValueError('Embedding item ids missing mapped items')
        valid=np.asarray(z['text_valid_mask'][ix],bool)&np.asarray(z['image_valid_mask'][ix],bool)
    sel=np.flatnonzero((degree>=1)&valid)
    if len(sel)!=expected_items:
        raise ValueError(f'U1 eligible item mismatch {len(sel)} != {expected_items}; check masks, seed, encoder and split. Original pipeline also checks embedding finiteness/norm.')
    return degree[sel].astype(np.int32),sel,dict(n_users=len(users_sorted),u1_users=n1,eligible_items=len(sel),n_mapping_items=len(ids))

def make_arcs(a,b,degree,n):
    eid=np.tile(np.arange(len(a),dtype=np.int32),2)
    anchor=np.r_[a,b].astype(np.int32)
    neighbor=np.r_[b,a].astype(np.int32)
    if anchor.min()<0 or neighbor.min()<0 or anchor.max()>=n or neighbor.max()>=n:raise ValueError('Bad edge indices')
    return eid,anchor,neighbor

def select_neighbor_slots(score,k,eid,anchor,neighbor,n):
    """Sort by anchor, descending edge score, neighbor id. First min(k, degree) each."""
    ordering=np.lexsort((neighbor, -np.asarray(score[eid],float), anchor))
    sorted_anchor=anchor[ordering]
    # position relative to the beginning of each anchor run
    first=np.r_[True,sorted_anchor[1:]!=sorted_anchor[:-1]]
    start=np.maximum.accumulate(np.where(first,np.arange(len(ordering)),0))
    return ordering[(np.arange(len(ordering))-start)<k]

def item_bootstrap(anchor,chosen_a,chosen_b,metric,reps,rng):
    # Paired cluster bootstrap: resample ANCHOR ITEMS, retaining their selected arcs.
    # Anchor inclusion weights shared across both methods. Conditioning on U3 fixed.
    n=int(anchor.max())+1
    def collapse(idx):
        sums=np.bincount(anchor[idx],weights=metric[idx],minlength=n)
        counts=np.bincount(anchor[idx],minlength=n)
        return sums,counts
    sa,ca=collapse(chosen_a);sb,cb=collapse(chosen_b)
    diffs=[]
    for _ in range(reps):
        w=rng.poisson(1,size=n).astype(float)
        da=np.dot(w,ca);db=np.dot(w,cb)
        if da and db:
            diffs.append(np.dot(w,sa)/da-np.dot(w,sb)/db)
    if len(diffs)<max(20,reps//2):raise RuntimeError('Insufficient bootstrap samples')
    return float(np.quantile(diffs,0.025)),float(np.quantile(diffs,0.975)),len(diffs)

def summarize_selection(indices,eid,anchor,neighbor,arr,degree,universe):
    chosen_edge=eid[indices]
    deg=np.asarray(degree,dtype=float)
    covered=np.unique(anchor[indices])
    unique_edges=np.unique(chosen_edge)
    return dict(n_directed_slots=int(len(indices)),n_unique_undirected_edges=int(len(unique_edges)),
        n_covered_anchors=int(len(covered)),anchor_coverage=float(len(covered)/universe),
        n_unique_incident_items=int(len(np.unique(np.r_[anchor[indices],neighbor[indices]]))),
        u3_recurrence=float(np.mean(arr['y3'][chosen_edge])),
        u3_recurrence_2users=float(np.mean(arr['y3_2users'][chosen_edge])),
        u3_norm_strength=float(np.mean(arr['strength3'][chosen_edge])),
        mean_log_anchor_degree=float(np.mean(np.log1p(deg[anchor[indices]]))),
        mean_log_neighbor_degree=float(np.mean(np.log1p(deg[neighbor[indices]]))),
        median_neighbor_degree=float(np.median(deg[neighbor[indices]])),
        neighbor_degree_p90=float(np.percentile(deg[neighbor[indices]],90)),
        mean_wB=float(np.mean(arr['wb'][chosen_edge])))

def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    d=read_npz(args.input/'routing_candidates.npz')
    e=read_npz(args.input/'expressibility_scores.npz')
    for key in ('a','b','wb','y3','y3_2users','strength3'):
        if key not in d:raise KeyError('Missing candidate key '+key)
    a,b=np.asarray(d['a'],int),np.asarray(d['b'],int)
    n_edges=len(a)
    if not all(len(d[k])==n_edges for k in ('b','wb','y3','y3_2users','strength3')):raise ValueError('Candidate shape mismatch')
    if np.any(a==b):raise ValueError('Self-edge present')
    if 'pbase_T_full' not in e:raise KeyError('Missing pbase_T_full; fit routing E first')
    prob=np.asarray(e['pbase_T_full'],float)
    if len(prob)!=n_edges:raise ValueError('Pbase size mismatch')
    if 'pbase_V_full' in e and not np.allclose(prob,e['pbase_V_full'],atol=1e-5):
        raise ValueError('Text/Image Pbase mismatch: cannot claim shared behavior-only estimator')
    if np.any(~np.isfinite(prob)) or np.any(~np.isfinite(d['wb'])):raise ValueError('Nonfinite scores')
    pm=args.input/'prepare_manifest.json'
    if not pm.exists():raise FileNotFoundError(pm)
    manifest=json.loads(pm.read_text())
    if int(manifest['seed'])!=args.seed:raise ValueError('Seed does not match input manifest')
    n=int(manifest['eligible_items'])
    degree,sel,info=reconstruct_degree(args.mapping,args.sequences,args.embeddings,args.seed,args.u1_fraction,n)
    if 'bfeatures' not in d: raise KeyError('Need cached bfeatures for alignment audit')
    expected_min=np.log1p(np.minimum(degree[a],degree[b]))
    expected_max=np.log1p(np.maximum(degree[a],degree[b]))
    if not np.allclose(d['bfeatures'][:,0],expected_min,atol=1e-4,rtol=1e-5) or not np.allclose(d['bfeatures'][:,1],expected_max,atol=1e-4,rtol=1e-5):
        raise ValueError('U1 degree mismatch against cached behavior features: check embedding validity, sequence split, item mapping, seed')
    eid,anchor,neighbor=make_arcs(a,b,degree,n)
    anchor_counts=np.bincount(anchor,minlength=n)
    covered=np.flatnonzero(anchor_counts>0)
    if not len(covered):raise ValueError('No anchor candidates')
    deg_cuts=np.quantile(degree[covered].astype(float),[1/3,2/3])
    deg_group=np.searchsorted(deg_cuts,degree,side='right').astype(np.int8)
    labels=('Low','Mid','High')
    rng=np.random.default_rng(args.seed+8523)
    rows=[];deltas=[];strata=[];audit=[]
    for k in args.k:
        selections={}
        for name,score in [('W_B',np.asarray(d['wb'],float)),('P_base',prob)]:
            indices=select_neighbor_slots(score,k,eid,anchor,neighbor,n)
            selections[name]=indices
            stats=summarize_selection(indices,eid,anchor,neighbor,d,degree,n)
            rows.append(dict(dataset=args.dataset,seed=args.seed,k=k,method=name,**stats))
            for g,label in enumerate(labels):
                ii=indices[deg_group[anchor[indices]]==g]
                if not len(ii): continue
                metrics=summarize_selection(ii,eid,anchor,neighbor,d,degree,n)
                # coverage within group: every candidate anchor in group should be represented.
                eligible_in_group=np.count_nonzero(deg_group[covered]==g)
                metrics['anchor_coverage_in_group']=metrics['n_covered_anchors']/eligible_in_group
                strata.append(dict(dataset=args.dataset,seed=args.seed,k=k,method=name,degree_group=label,
                    eligible_anchors_in_group=int(eligible_in_group),**metrics))
        wa=selections['W_B']; pa=selections['P_base']
        if len(wa)!=len(pa):raise AssertionError('Different directed-slot budgets')
        if not np.array_equal(np.unique(anchor[wa]),np.unique(anchor[pa])):raise AssertionError('Different anchor coverage')
        # Both methods select the same number of neighbors per anchor.
        if not np.array_equal(np.bincount(anchor[wa],minlength=n),np.bincount(anchor[pa],minlength=n)):
            raise AssertionError('Different per-anchor slot budgets')
        audit.append(dict(k=k,selected_slots=len(wa),candidate_anchors=len(covered),
            expected_slots=int(np.minimum(anchor_counts,k).sum()),
            n_shared_unique_undirected_edges=len(np.intersect1d(np.unique(eid[wa]),np.unique(eid[pa])))))
        for g,group in [(None,'All')]+list(enumerate(labels)):
            iwa=wa if g is None else wa[deg_group[anchor[wa]]==g]
            ipa=pa if g is None else pa[deg_group[anchor[pa]]==g]
            if not len(iwa) or not len(ipa):
                continue
            for name,key,scale in [('recurrence','y3',100.),('norm_strength','strength3',1.)]:
                metric=np.asarray(d[key],float)[eid]
                obs=float(scale*(np.mean(metric[ipa])-np.mean(metric[iwa])))
                lo,hi,valid=item_bootstrap(anchor,ipa,iwa,metric,args.bootstrap,rng)
                deltas.append(dict(dataset=args.dataset,seed=args.seed,k=k,degree_group=group,
                    metric=name,unit='percentage_points' if name=='recurrence' else 'normalized_strength',
                    pbase_minus_wb=obs,ci95_low=lo*scale,ci95_high=hi*scale,
                    bootstrap_valid=valid,bootstrap_reps=args.bootstrap,
                    n_anchor_slots=int(len(iwa))))
        print(f"k={k:2d}  W_B recurrence={100*rows[-2]['u3_recurrence']:.3f}%  "
              f"Pbase={100*rows[-1]['u3_recurrence']:.3f}%  "
              f"coverage={rows[-1]['anchor_coverage']:.3f}  slots={len(wa)}",flush=True)
    csvwrite(args.out/'neighbor_selection.csv',rows)
    csvwrite(args.out/'anchor_degree_strata.csv',strata)
    csvwrite(args.out/'paired_anchor_bootstrap_ci.csv',deltas)
    csvwrite(args.out/'selection_audit.csv',audit)
    metadata=dict(dataset=args.dataset,seed=args.seed,k=args.k,bootstrap=args.bootstrap,
        bootstrap_method='paired Poisson anchor-item cluster bootstrap; source-anchor clusters; not model training/user uncertainty',
        input=str(args.input),train_degree='Reconstructed exact U1 train unique-user counts using same split and original item embedding valid masks',
        item_degree_tertile_thresholds=deg_cuts.tolist(),n_candidate_edges=n_edges,n_eligible_items=n,
        n_candidate_anchors=int(len(covered)),eligibility='Same-parent U1 behavior edges, original routing E candidate set',
        primary_estimand='directed anchor-neighbor slots; an undirected edge is evaluated twice if chosen by both endpoints',
        coverage='candidate anchor coverage exactly matched; this does not imply unique union-edge counts match',
        probability='Cached U2-fitted P(Y_U2 recurrence|U1 behavior features) full-fit predictions; not item-disjoint 5-fold',
        no_U3_leakage='Ranking depends only on U1 and U2, U3 used for held-out evaluation',
        degree_info=info)
    (args.out/'manifest.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    print('SUCCESS:',args.out,flush=True)

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
    ap.add_argument('--k',type=int,nargs='+',default=[5,10,20])
    ap.add_argument('--bootstrap',type=int,default=500)
    args=ap.parse_args()
    if args.bootstrap<50:ap.error('Bootstrap should be >=50 for confidence intervals')
    run(args)
if __name__=='__main__':main()
