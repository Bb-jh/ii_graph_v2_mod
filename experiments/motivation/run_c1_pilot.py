#!/usr/bin/env python3
"""C1 pilot: Content-expressible Behavior Relation on disjoint training users.

Place beside run_motivation_cpu_v2.py. Uses only train prefix of SASRec
leave-two-out sequence file; U2 is accessed only after U1-based selection.
Single-split sanity diagnostic, NOT statistical significance or causal proof.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from run_motivation_cpu_v2 import load_mapping, load_embeddings, behavior_graph, to_igraph, leiden


def read_users_train(path: Path, n_items: int):
    from collections import defaultdict
    users = defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): continue
            tok = line.split()
            if len(tok) != 2: raise ValueError(f'Bad sequence line {line_no}')
            u, i = map(int, tok)
            if not 1 <= i <= n_items: raise ValueError(f'Bad item {i} line {line_no}')
            users[u].append(i - 1)
    if not users: raise ValueError('Empty sequence file')
    ids = sorted(users)
    out = []
    for u in ids:
        seq = users[u]
        if len(seq) < 3: raise ValueError(f'User {u} has <3 events; unexpected split format')
        out.append(np.unique(np.asarray(seq[:-2], dtype=np.int32)))
    return ids, out


def csr_users(seqs, n_items):
    rr = np.concatenate([np.full(len(s), i, np.int32) for i,s in enumerate(seqs)])
    cc = np.concatenate(seqs)
    x = sparse.coo_matrix((np.ones(len(rr), np.float32), (rr,cc)),
        shape=(len(seqs), n_items)).tocsr()
    x.sum_duplicates(); x.data[:]=1
    return x


def split_users(seqs, frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(seqs))
    cut = int(round(frac * len(seqs)))
    if not (0 < cut < len(seqs)): raise ValueError('Need nonempty U1 and U2')
    return perm[:cut], perm[cut:]


def partition_by_count(graph, target, seed, evaluations=17, low=1e-4, high=1500.0):
    """Standalone count calibration, independent of Experiment A code dependencies."""
    lo, hi = float(low), float(high)
    cache = {}
    def eva(r):
        r = float(r)
        if r not in cache:
            lab, st = leiden(graph, r, seed)
            cache[r] = (lab, st)
        return cache[r][1]['n_clusters']
    eva(lo); eva(hi)
    for _ in range(evaluations):
        mid = float(np.sqrt(lo * hi))
        if mid == lo or mid == hi: break
        k = eva(mid)
        if k < target: lo = mid
        elif k > target: hi = mid
        else: break
    best = min(cache, key=lambda r: abs(cache[r][1]['n_clusters'] - target))
    return cache[best][0], dict(cache[best][1], resolution=best,
        target_clusters=target, count_error=abs(cache[best][1]['n_clusters']-target)/target)


def upper_edges(matrix):
    t = sparse.triu(matrix, k=1, format='coo')
    return t.row.astype(np.int32), t.col.astype(np.int32), t.data.astype(np.float64)


def collect_candidate_edges(graph, labels, co, deg, normalized_t, normalized_v):
    a, b, w = upper_edges(graph)
    within = labels[a] == labels[b]
    a, b, w = a[within], b[within], w[within]
    # Counts from unpruned train-only U1 co-occurrence, not final top-k weights.
    count = np.asarray(co[a,b]).ravel().astype(float)
    # Ensure both view embeddings valid; caller selected common-valid items.
    st = np.einsum('ij,ij->i', normalized_t[a], normalized_t[b], optimize=True)
    sv = np.einsum('ij,ij->i', normalized_v[a], normalized_v[b], optimize=True)
    p = labels[a]
    # endpoint feature sorting ensures symmetric undirected edge features
    dmin = np.minimum(deg[a], deg[b]); dmax = np.maximum(deg[a], deg[b])
    features = np.column_stack((np.log1p(dmin), np.log1p(dmax), np.log1p(count))).astype(np.float32)
    return dict(a=a,b=b,weight=w,count=count,st=st,sv=sv,parent=p,features=features)


def top_q_by_parent(parent, scores, q, min_parent_edges):
    selected = np.zeros(len(parent), dtype=bool)
    eligible = np.zeros(len(parent), dtype=bool)
    for group in np.unique(parent):
        idx = np.flatnonzero(parent==group)
        if len(idx) < min_parent_edges: continue
        eligible[idx] = True
        count = max(1, int(np.floor(q * len(idx))))
        order = np.lexsort((idx,-scores[idx]))
        selected[idx[order[:count]]] = True
    return selected, eligible


def matched_controls(parent, features, selected, eligible, seed, caliper=0.80, neighbors=5):
    """Within-parent U1-feature nearest control; no U2 feature is used.

    Controls sampled with replacement. Failure to meet caliper remains unmatched.
    Returns selected indices and matching control indices; report matched-only
    estimand, coverage, and covariate imbalance.
    """
    rng = np.random.default_rng(seed)
    selected_ids = np.flatnonzero(selected)
    controls = np.full(len(selected_ids), -1, np.int64)
    groups = defaultdict(list)
    for loc, eid in enumerate(selected_ids): groups[int(parent[eid])].append(loc)
    for grp, locs in groups.items():
        locs = np.asarray(locs, np.int64)
        pool = np.flatnonzero((parent == grp)&eligible&(~selected))
        if len(pool)==0: continue
        nn = NearestNeighbors(n_neighbors=min(neighbors,len(pool)), algorithm='auto')
        nn.fit(features[pool])
        distances, candidates = nn.kneighbors(features[selected_ids[locs]])
        for h, loc in enumerate(locs):
            good = np.flatnonzero(distances[h] <= caliper)
            if len(good):
                choice = rng.choice(good)
                controls[loc] = pool[candidates[h, choice]]
    keep = controls >= 0
    return selected_ids[keep], controls[keep], len(selected_ids)


def eval_pairs(a, b, co2, deg2):
    counts = np.asarray(co2[a,b]).ravel().astype(np.float64)
    denom = np.sqrt(deg2[a].astype(float) * deg2[b].astype(float))
    weighted = np.divide(counts, denom, out=np.zeros_like(counts), where=denom>0)
    endpoint_active = (deg2[a]>0)&(deg2[b]>0)
    return dict(recurrence=float(np.mean(counts>0)) if len(counts) else None,
        recurrence_2users=float(np.mean(counts>=2)) if len(counts) else None,
        norm_strength=float(np.mean(weighted)) if len(counts) else None,
        endpoint_active=float(np.mean(endpoint_active)) if len(counts) else None,
        n=int(len(counts)))


def csv_write(path, rows):
    if not rows: return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('dataset','mapping','sequences','embeddings','out'):
        p.add_argument('--'+k,required=True,type=Path if k!='dataset' else str)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--u1-fraction',type=float,default=0.5)
    p.add_argument('--parent-k',type=int,default=400)
    p.add_argument('--behavior-k',type=int,default=50)
    p.add_argument('--q',type=float,nargs='+',default=[0.1,0.2,0.3])
    p.add_argument('--min-parent-edges',type=int,default=50)
    p.add_argument('--match-caliper',type=float,default=0.8)
    p.add_argument('--match-neighbors',type=int,default=5)
    p.add_argument('--max-parent-count-error',type=float,default=0.1)
    args=p.parse_args()
    if not 0<args.u1_fraction<1 or not all(0<q<0.5 for q in args.q):p.error('Invalid split/q')
    args.out.mkdir(parents=True,exist_ok=True)
    print('Reading mapped data and train-only sequences...',flush=True)
    ids=load_mapping(args.mapping)
    user_ids, seqs=read_users_train(args.sequences,len(ids))
    u1_idx,u2_idx=split_users(seqs,args.u1_fraction,args.seed)
    x1=csr_users([seqs[i] for i in u1_idx],len(ids))
    x2=csr_users([seqs[i] for i in u2_idx],len(ids))
    deg1=np.asarray(x1.sum(axis=0)).ravel().astype(np.int64)
    deg2all=np.asarray(x2.sum(axis=0)).ravel().astype(np.int64)
    embs=load_embeddings(args.embeddings,ids)
    sel=np.flatnonzero((deg1>=1)&embs['text'][1]&embs['image'][1])
    print(f'U1 users={len(u1_idx):,} U2 users={len(u2_idx):,} eligible U1 items={len(sel):,}',flush=True)
    if len(sel)<args.parent_k:raise ValueError('Not enough U1 items for parent count')
    bg,_=behavior_graph(x1,deg1,sel,args.behavior_k,'union')
    print(f'U1 B edges={bg.nnz//2:,}; calibrating parent K={args.parent_k}',flush=True)
    labels,calib=partition_by_count(to_igraph(bg),args.parent_k,args.seed)
    print('Parent calibration:',calib,flush=True)
    if calib['count_error']>args.max_parent_count_error:raise RuntimeError('Parent calibration unsuccessful')
    x1_sel=x1[:,sel]; x2_sel=x2[:,sel]
    co1=(x1_sel.T@x1_sel).tocsr();co1.setdiag(0);co1.eliminate_zeros()
    co2=(x2_sel.T@x2_sel).tocsr();co2.setdiag(0);co2.eliminate_zeros()
    deg2=np.asarray(x2_sel.sum(axis=0)).ravel().astype(np.int64)
    norms={}
    for typ in ('text','image'):
        z=np.ascontiguousarray(embs[typ][0][sel],dtype=np.float32)
        z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
        norms[typ]=z
    ed=collect_candidate_edges(bg,labels,co1,deg1[sel],norms['text'],norms['image'])
    del co1
    n=len(ed['a']); print(f'Within-parent U1 B candidates={n:,}',flush=True)
    if n==0:raise RuntimeError('No within-parent candidate edges')
    results=[]
    for q in args.q:
        for modality,score in [('T',ed['st']),('V',ed['sv'])]:
            chosen, eligible=top_q_by_parent(ed['parent'],score,q,args.min_parent_edges)
            a,b,total=matched_controls(ed['parent'],ed['features'],chosen,eligible,
                seed=args.seed+int(q*10000)+(1 if modality=='T' else 2),
                caliper=args.match_caliper,neighbors=args.match_neighbors)
            print(f'{modality} q={q:.2f}: top edges={total:,} matched={len(a):,} ({len(a)/max(total,1):.1%})',flush=True)
            for label,idx in [('top',a),('matched',b)]:
                metric=eval_pairs(ed['a'][idx],ed['b'][idx],co2,deg2)
                results.append(dict(dataset=args.dataset,seed=args.seed,modality=modality,q=q,
                    group=label,**metric,matched_edge_pairs=len(a),total_top_edges=total,
                    matching_success=len(a)/max(total,1), eligible_parent_edges=int(eligible.sum()),
                    eligible_parents=int(len(np.unique(ed['parent'][eligible]))),
                    candidate_edges=n,
                    mean_log_degree_min=float(ed['features'][idx,0].mean()) if len(idx) else None,
                    mean_log_degree_max=float(ed['features'][idx,1].mean()) if len(idx) else None,
                    mean_log_u1_coocc=float(ed['features'][idx,2].mean()) if len(idx) else None))
    csv_write(args.out/'c1_metrics.csv',results)
    comparisons=[]
    for q in args.q:
        for modality in ('T','V'):
            z=[r for r in results if r['q']==q and r['modality']==modality]
            top,control=next(r for r in z if r['group']=='top'),next(r for r in z if r['group']=='matched')
            out=dict(modality=modality,q=q,matched_edge_pairs=top['n'],matching_success=top['matching_success'])
            for name in ('recurrence','recurrence_2users','norm_strength','endpoint_active'):
                out['top_'+name]=top[name];out['control_'+name]=control[name]
                out['delta_'+name]=top[name]-control[name] if top[name] is not None else None
            comparisons.append(out)
            print('  delta recurrence=',out['delta_recurrence'],'delta strength=',out['delta_norm_strength'],flush=True)
    csv_write(args.out/'c1_comparison.csv',comparisons)
    meta=dict(dataset=args.dataset,seed=args.seed,u1_fraction=args.u1_fraction,
        users=len(user_ids),users_u1=len(u1_idx),users_u2=len(u2_idx),
        total_items=len(ids),eligible_items=len(sel),u1_graph_edges=bg.nnz//2,
        parent_calibration=calib,candidate_edges=n,
        candidate_item_coverage=int(np.unique(np.r_[ed['a'],ed['b']]).size),
        u1_train_unique_interactions=x1.nnz,u2_train_unique_interactions=x2.nnz,
        q=args.q,behavior_k=args.behavior_k,min_parent_edges=args.min_parent_edges,
        matching={'caliper_euclidean_log_features':args.match_caliper,
                  'with_replacement':True,'pool':'same parent, non-selected U1 B edges',
                  'features':'log1p U1 endpoint degree min/max and U1 co-occurrence count'},
        caveats=['Pilot only: no CI or significance test.',
                 'U2 is not used for candidate ranking, parent construction or matching.',
                 'Recurrence 0 means unobserved, not a negative preference.',
                 'Matched-only estimand if matching is incomplete.',
                 'Matching does not guarantee exact degree/strength balance; inspect covariate means.',
                 'Behavior candidates restricted to same U1 parent and both CLIP views valid.'])
    (args.out/'c1_manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('SUCCESS:',args.out,flush=True)

if __name__=='__main__':main()
