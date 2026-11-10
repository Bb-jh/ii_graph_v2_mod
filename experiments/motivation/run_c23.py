#!/usr/bin/env python3
"""C2/C3 user-disjoint validation. Place next to run_c1_pilot.py and run_motivation_cpu_v2.py.

C2: independent per-parent Text/Image Top-q lists on identical U1 behavior edges.
C3: U2 heldout recurrence of T-only, V-only, Both and matched Neither edges.
All selection and matching are U1-only; U2 never defines labels, strata or controls.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from run_c1_pilot import (read_users_train, csr_users, split_users, partition_by_count,
    collect_candidate_edges, top_q_by_parent, matched_controls, eval_pairs, csv_write)
from run_motivation_cpu_v2 import load_mapping, load_embeddings, behavior_graph, to_igraph


def safe_mean(x):
    return float(np.mean(x)) if len(x) else None


def smd(a, b):
    if not len(a) or not len(b): return [None] * a.shape[1]
    x, y = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((x.var(axis=0) + y.var(axis=0)) / 2)
    diff = np.abs(x.mean(axis=0) - y.mean(axis=0))
    return [float(v) if np.isfinite(v) else None for v in np.divide(diff, pooled, out=np.zeros_like(diff), where=pooled>1e-12)]


def c2_stats(parent, st, sv, tmask, vmask, eligible, q):
    """Within each sufficiently sized parent, return per-parent overlap and rank association."""
    records = []
    for p in np.unique(parent[eligible]):
        ix = np.flatnonzero((parent == p) & eligible)
        if len(ix) < 2: continue
        t, v = tmask[ix], vmask[ix]
        n = len(ix)
        both = int(np.count_nonzero(t & v)); nt, nv = int(t.sum()), int(v.sum())
        union = int(np.count_nonzero(t | v))
        rho = float(spearmanr(st[ix], sv[ix]).statistic) if np.std(st[ix]) > 0 and np.std(sv[ix]) > 0 else None
        records.append(dict(parent=int(p), q=q, candidate_edges=n, top_t=nt, top_v=nv,
            both=both, t_only=int(np.count_nonzero(t & ~v)),
            v_only=int(np.count_nonzero(v & ~t)), neither=int(np.count_nonzero(~t & ~v)),
            jaccard=both / union if union else None,
            overlap_fraction=both / min(nt,nv) if min(nt,nv) else None,
            random_expected_both=nt*nv/n,
            spearman=rho))
    return records


def run(args):
    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    ids = load_mapping(args.mapping)
    user_ids, seqs = read_users_train(args.sequences,len(ids))
    u1_idx,u2_idx = split_users(seqs,args.u1_fraction,args.seed)
    x1=csr_users([seqs[i] for i in u1_idx],len(ids))
    x2=csr_users([seqs[i] for i in u2_idx],len(ids))
    deg1=np.asarray(x1.sum(axis=0)).ravel().astype(np.int64)
    embeddings=load_embeddings(args.embeddings,ids)
    sel=np.flatnonzero((deg1>=1)&embeddings['text'][1]&embeddings['image'][1])
    print(f'U1={len(u1_idx):,}, U2={len(u2_idx):,}, eligible items={len(sel):,}',flush=True)
    bg,_=behavior_graph(x1,deg1,sel,args.behavior_k,'union')
    print(f'U1 B edges={bg.nnz//2:,}',flush=True)
    labels,calib=partition_by_count(to_igraph(bg),args.parent_k,args.seed)
    print('Parent:',calib,flush=True)
    if calib['count_error'] > args.max_parent_count_error: raise RuntimeError('Parent calibration failed')
    x1_sel,x2_sel=x1[:,sel],x2[:,sel]
    co1=(x1_sel.T@x1_sel).tocsr();co1.setdiag(0);co1.eliminate_zeros()
    norms={}
    for typ in ('text','image'):
        z=np.ascontiguousarray(embeddings[typ][0][sel],dtype=np.float32)
        z /= np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
        norms[typ]=z
    ed=collect_candidate_edges(bg,labels,co1,deg1[sel],norms['text'],norms['image'])
    del co1,bg
    print(f'Within-parent candidates={len(ed["a"]):,}',flush=True)
    # C2: no U2 data enters any selection or association calculation.
    all_c2=[];all_c3=[];all_bal=[]
    co2=(x2_sel.T@x2_sel).tocsr();co2.setdiag(0);co2.eliminate_zeros()
    deg2=np.asarray(x2_sel.sum(axis=0)).ravel().astype(np.int64)
    for q in args.q:
        tm,eligible=top_q_by_parent(ed['parent'],ed['st'],q,args.min_parent_edges)
        vm,eligible_v=top_q_by_parent(ed['parent'],ed['sv'],q,args.min_parent_edges)
        assert np.array_equal(eligible,eligible_v)
        categories={'T_only':tm & ~vm, 'V_only':vm & ~tm,
                    'Both':tm & vm, 'Neither':eligible & ~tm & ~vm}
        assert sum(int(a.sum()) for a in categories.values()) == int(eligible.sum())
        rows=c2_stats(ed['parent'],ed['st'],ed['sv'],tm,vm,eligible,q)
        all_c2.extend(rows)
        both=int(categories['Both'].sum()); tcount=int(tm.sum());vcount=int(vm.sum())
        uni=int(np.count_nonzero(tm|vm));chance=float(np.sum([r['random_expected_both'] for r in rows]))
        print(f'q={q:.2f} T={tcount:,} V={vcount:,} Both={both:,} '
              f'Jaccard={both/uni:.4f} expected both={chance:.0f}',flush=True)
        # U2 evaluation occurs ONLY after candidate generation and matching.
        for k, mask in categories.items():
            ids_cat=np.flatnonzero(mask)
            if not len(ids_cat):
                all_c3.append(dict(seed=args.seed,q=q,category=k,candidate_edges=0,
                    matched=0,match_rate=None,top_recurrence=None,control_recurrence=None,
                    delta_recurrence=None,top_recurrence_2users=None,control_recurrence_2users=None,
                    delta_recurrence_2users=None,top_norm_strength=None,control_norm_strength=None,
                    delta_norm_strength=None,top_endpoint_active=None,control_endpoint_active=None,
                    control_unique=None,control_reuse_rate=None))
                continue
            # All categories' U2 outcomes are described. For C3 controls, each
            # selected category is compared against the SAME eligible Neither pool.
            if k == 'Neither':
                top=eval_pairs(ed['a'][ids_cat],ed['b'][ids_cat],co2,deg2)
                all_c3.append(dict(seed=args.seed,q=q,category=k,candidate_edges=len(ids_cat),
                    matched=0,match_rate=None,top_recurrence=top['recurrence'],control_recurrence=None,
                    delta_recurrence=None,top_recurrence_2users=top['recurrence_2users'],control_recurrence_2users=None,
                    delta_recurrence_2users=None,top_norm_strength=top['norm_strength'],control_norm_strength=None,
                    delta_norm_strength=None,top_endpoint_active=top['endpoint_active'],control_endpoint_active=None,
                    control_unique=None,control_reuse_rate=None))
                continue
            # matched_controls expects eligible True only for candidate category or Neither;
            # this forces the control pool to Neither, rather than other selected categories.
            pool_eligible = mask | categories['Neither']
            top_id, ctrl_id, total=matched_controls(ed['parent'],ed['features'],mask,pool_eligible,
                seed=args.seed*100000 + int(round(q*1000))*100 + {'T_only':1,'V_only':2,'Both':3}[k],
                caliper=args.match_caliper,neighbors=args.match_neighbors)
            top=eval_pairs(ed['a'][top_id],ed['b'][top_id],co2,deg2)
            ctrl=eval_pairs(ed['a'][ctrl_id],ed['b'][ctrl_id],co2,deg2)
            delta=lambda name: top[name]-ctrl[name] if top[name] is not None else None
            unique=int(np.unique(ctrl_id).size)
            all_c3.append(dict(seed=args.seed,q=q,category=k,candidate_edges=total,
                matched=len(top_id),match_rate=len(top_id)/total,
                top_recurrence=top['recurrence'],control_recurrence=ctrl['recurrence'],
                delta_recurrence=delta('recurrence'),
                top_recurrence_2users=top['recurrence_2users'],control_recurrence_2users=ctrl['recurrence_2users'],
                delta_recurrence_2users=delta('recurrence_2users'),
                top_norm_strength=top['norm_strength'],control_norm_strength=ctrl['norm_strength'],
                delta_norm_strength=delta('norm_strength'),top_endpoint_active=top['endpoint_active'],
                control_endpoint_active=ctrl['endpoint_active'],control_unique=unique,
                control_reuse_rate=1-unique/max(len(ctrl_id),1)))
            balanced=smd(ed['features'][top_id],ed['features'][ctrl_id])
            all_bal.append(dict(seed=args.seed,q=q,category=k,matched=len(top_id),
                smd_log_degree_min=balanced[0],smd_log_degree_max=balanced[1],
                smd_log_u1_coocc=balanced[2],
                mean_top_log_u1_coocc=safe_mean(ed['features'][top_id,2]),
                mean_control_log_u1_coocc=safe_mean(ed['features'][ctrl_id,2])))
            print(f'  {k}: matched {len(top_id):,}/{total:,}, '
                  f'delta R={delta("recurrence")}, SMD={balanced}',flush=True)
    csv_write(out/'c2_parent.csv',all_c2)
    # aggregate global metrics must use edge counts, NOT mean over parents.
    agg=[]
    for q in args.q:
        a=[r for r in all_c2 if r['q']==q]
        nt=sum(r['top_t'] for r in a);nv=sum(r['top_v'] for r in a)
        both=sum(r['both'] for r in a); universe=sum(r['candidate_edges'] for r in a)
        expected=sum(r['random_expected_both'] for r in a)
        rho=[r['spearman'] for r in a if r['spearman'] is not None]
        agg.append(dict(q=q,parents=len(a),eligible_edges=universe,text_top=nt,image_top=nv,
            both=both,t_only=nt-both,v_only=nv-both,neither=universe-nt-nv+both,
            jaccard=both/(nt+nv-both) if nt+nv>both else None,
            intersection_over_top=both/min(nt,nv) if min(nt,nv) else None,
            chance_expected_intersection=expected,
            observed_over_chance=both/expected if expected else None,
            parent_macro_spearman=safe_mean(rho)))
    csv_write(out/'c2_summary.csv',agg)
    csv_write(out/'c3_matched.csv',all_c3)
    csv_write(out/'c3_balance.csv',all_bal)
    manifest=dict(dataset=args.dataset,seed=args.seed,users_u1=len(u1_idx),users_u2=len(u2_idx),
        eligible_items=len(sel),candidate_edges=len(ed['a']),parent_calibration=calib,
        config=dict(u1_fraction=args.u1_fraction,parent_k=args.parent_k,behavior_k=args.behavior_k,
            q=args.q,min_parent_edges=args.min_parent_edges,match_caliper=args.match_caliper,
            match_neighbors=args.match_neighbors),
        notes=['C2 per-parent rank correlation is computed on identical U1 Behavior edges.',
            'Chance expected intersection assumes independent uniform Top-q choices within each parent.',
            'C3 primary contrasts each T-only/V-only/Both category against same-parent matched Neither edges.',
            'U2 outcomes are never used for ranking, parent construction or matching.',
            'Matched-controls sampled with replacement. No CI or significance test in this pilot.',
            'SMD refers to marginal covariate balance; high match rate does not imply full balance.',
            'C3 controls have low semantic ranking by definition; U2 recurrence difference is predictive association, not proof of causal modality-specific benefit.'])
    (out/'c23_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print('SUCCESS:',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True)
    for v in ('mapping','sequences','embeddings','out'):p.add_argument('--'+v,required=True,type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--u1-fraction',type=float,default=0.5)
    p.add_argument('--parent-k',type=int,default=400)
    p.add_argument('--behavior-k',type=int,default=50)
    p.add_argument('--q',type=float,nargs='+',default=[.1,.2,.3])
    p.add_argument('--min-parent-edges',type=int,default=50)
    p.add_argument('--match-caliper',type=float,default=.8)
    p.add_argument('--match-neighbors',type=int,default=5)
    p.add_argument('--max-parent-count-error',type=float,default=.1)
    a=p.parse_args()
    if not 0<a.u1_fraction<1 or not all(0<q<.5 for q in a.q):p.error('Invalid split fraction or q')
    run(a)

if __name__=='__main__':main()
