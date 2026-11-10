#!/usr/bin/env python3
"""Experiment A degree-preserving U-I null model.

Reuses *actual* Experiment A common item set, T/V Leiden partitions and
same graph construction and Leiden calibration implementation.
Does NOT change the user's baseline code or result files.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from run_motivation_cpu_v2 import load_mapping, load_train_only, load_embeddings, behavior_graph, to_igraph, save_json
from run_experiment_a_multiseed import calibrate_count, dump_csv


def swap_edges_preserving_both_degrees(x, seed, swaps_per_edge):
    """Simple bipartite double-edge swaps: (u,a),(v,b) -> (u,b),(v,a).

    Both user/item degrees preserved exactly. No multiedges, no self-edges.
    Coordinates are binary U-I edges, NOT individual repeated events.
    """
    coo = x.tocoo(copy=True)
    u = np.asarray(coo.row, dtype=np.int32).copy()
    it = np.asarray(coo.col, dtype=np.int32).copy()
    m = len(u)
    n_items = x.shape[1]
    if m < 2:
        raise ValueError('Not enough edges to perform swaps')
    rng = np.random.default_rng(seed)
    lookup = set((u.astype(np.int64)*n_items+it).tolist())
    if len(lookup) != m:
        raise ValueError('U-I graph contains duplicate incidence edges')
    attempts = int(np.ceil(m*swaps_per_edge))
    accepted = 0
    for _ in range(attempts):
        a, b = int(rng.integers(m)), int(rng.integers(m))
        if a == b:
            continue
        ua, ub = int(u[a]), int(u[b])
        ia, ib = int(it[a]), int(it[b])
        if ua == ub or ia == ib:
            continue
        old_a = ua*n_items+ia
        old_b = ub*n_items+ib
        new_a = ua*n_items+ib
        new_b = ub*n_items+ia
        if new_a in lookup or new_b in lookup:
            continue
        lookup.remove(old_a)
        lookup.remove(old_b)
        lookup.add(new_a)
        lookup.add(new_b)
        it[a], it[b] = ib, ia
        accepted += 1
    xr = sparse.csr_matrix((np.ones(m, dtype=np.float32), (u,it)), shape=x.shape)
    xr.sum_duplicates()
    xr.data[:] = 1
    if xr.nnz != x.nnz:
        raise RuntimeError('Randomization changed number of unique U-I edges')
    orig_u = np.diff(x.indptr)
    rand_u = np.diff(xr.indptr)
    orig_i = np.asarray(x.sum(axis=0)).ravel().astype(np.int64)
    rand_i = np.asarray(xr.sum(axis=0)).ravel().astype(np.int64)
    if not np.array_equal(orig_u, rand_u) or not np.array_equal(orig_i, rand_i):
        raise RuntimeError('CRITICAL: randomization did not preserve both degrees')
    return xr, {'swap_attempts': attempts, 'accepted_swaps': accepted,
                'acceptance_rate': accepted / attempts if attempts else 0,
                'accepted_per_edge': accepted / m, 'ui_edges': m,
                'user_degree_preserved': True, 'item_degree_preserved': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a-out', type=Path, required=True, help='Existing FINAL Experiment A output dir containing graph_cache/manifest.json and partitions/')
    p.add_argument('--mapping', type=Path, required=True)
    p.add_argument('--sequences', type=Path, required=True)
    p.add_argument('--embeddings', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--random-seeds', type=int, nargs='+', default=[202,203,204], help='Independent random graph seeds')
    p.add_argument('--swaps-per-edge', type=float, default=10., help='Attempted double-edge swaps per unique U-I edge')
    p.add_argument('--leiden-seeds', type=int, nargs='+', default=None, help='Defaults to seeds in Experiment A config')
    p.add_argument('--targets', type=int, nargs='+', default=None, help='Defaults to targets in Experiment A config')
    p.add_argument('--threads', type=int, default=4)
    args=p.parse_args()
    if len(set(args.random_seeds)) != len(args.random_seeds) or not args.random_seeds:
        p.error('Random seeds must be nonempty and unique')
    if args.swaps_per_edge <= 0: p.error('swaps-per-edge must be positive')
    a_dir=args.a_out.resolve()
    conf=json.loads((a_dir/'config.json').read_text())
    manifest=json.loads((a_dir/'graph_cache/manifest.json').read_text())
    targets=args.targets or conf['targets']
    seeds=args.leiden_seeds or conf['seeds']
    for k, val in [('mapping',args.mapping),('sequences',args.sequences),('embeddings',args.embeddings)]:
        if Path(conf[k]).resolve() != val.resolve():
            print(f'WARNING: A originally used {k}={conf[k]}, null uses {val.resolve()}. Verify same files.',file=sys.stderr)
    ids=load_mapping(args.mapping)
    x, degree, _=load_train_only(args.sequences,len(ids))
    embeddings=load_embeddings(args.embeddings,ids)
    selected=np.flatnonzero((degree>=1)&embeddings['text'][1]&embeddings['image'][1])
    digest=hashlib.sha256(np.asarray(selected,dtype=np.int64).tobytes()).hexdigest()
    if digest != manifest['selected_item_ids_sha256'] or len(selected)!=manifest['common_items']:
        raise RuntimeError('Item universe HASH MISMATCH vs original A. STOP: cannot compare index-aligned partitions.')
    # Strict test: original B must be reconstructable from the same data.
    original_b, _=behavior_graph(x,degree,selected,int(conf['behavior_k']),'union')
    cached_b=sparse.load_npz(a_dir/'graph_cache/B.npz').tocsr()
    if original_b.shape != cached_b.shape or (original_b != cached_b).nnz:
        diff=(original_b-cached_b)
        if diff.nnz and float(np.max(np.abs(diff.data))) > 1e-6:
            raise RuntimeError('Cached A Behavior graph DIFFERENT from reconstructed B. STOP.')
    del original_b, cached_b
    args.out.mkdir(parents=True,exist_ok=True)
    out_config=dict(a_out=str(a_dir),random_seeds=args.random_seeds,leiden_seeds=seeds,
        targets=targets,swaps_per_edge=args.swaps_per_edge,selected_item_ids_sha256=digest,
        graph_method='same baseline degree-normalized B + top-k union',
        null_method='bipartite double-edge swapping, fixed binary user and item degrees')
    cp=args.out/'null_config.json'
    if cp.exists() and json.loads(cp.read_text())!=out_config:
        raise RuntimeError('Existing output config mismatch. Choose a NEW --out directory.')
    save_json(cp,out_config)
    real_rows=list(csv.DictReader((a_dir/'cross_view.csv').open()))
    real_lookup={(int(r['target_clusters']),int(r['seed']),r['pair']):r for r in real_rows}
    rows=[]; diagnostics=[]
    for rand_seed in args.random_seeds:
        print(f'\n===== RANDOM GRAPH {rand_seed} =====',flush=True)
        xr, swap=swap_edges_preserving_both_degrees(x,rand_seed,args.swaps_per_edge)
        if swap['accepted_per_edge']<3:
            print('WARNING: <3 accepted swaps / U-I edge. Null may be under-mixed.',flush=True)
        bg,_=behavior_graph(xr,degree,selected,int(conf['behavior_k']),'union')
        n_edges=int(bg.nnz//2)
        diagnostics.append(dict(random_seed=rand_seed,**swap,behavior_edges=n_edges,
            real_behavior_edges=manifest['edge_counts']['B'],
            edge_count_ratio=n_edges/max(1,manifest['edge_counts']['B'])))
        print('  swaps:',swap,' B edges:',n_edges,' real:',manifest['edge_counts']['B'],flush=True)
        g=to_igraph(bg)
        del xr,bg
        for seed in seeds:
            cache={}
            for target in targets:
                b_labels, resolution, stats=calibrate_count(g,target,seed,cache,
                    float(conf['resolution_low']),float(conf['resolution_high']),int(conf['calibration_evals']))
                calibrated=abs(stats['clusters']-target)/target <= float(conf['count_tolerance'])
                for view in ('T','V'):
                    ref_file=a_dir/'partitions'/f'{view}_k{target}_seed{seed}.npy'
                    if not ref_file.exists(): raise FileNotFoundError(ref_file)
                    t_labels=np.load(ref_file)
                    real_row=real_lookup.get((target,seed,f'B-{view}'))
                    if real_row is None: raise RuntimeError(f'Missing original real B-{view} cross AMI {target} seed {seed}')
                    if not calibrated or real_row['calibrated'].lower()!='true':
                        ami=ari=None
                    else:
                        ami=float(adjusted_mutual_info_score(b_labels,t_labels))
                        ari=float(adjusted_rand_score(b_labels,t_labels))
                    rows.append(dict(random_seed=rand_seed,leiden_seed=seed,target_clusters=target,
                        pair=f'B-{view}',null_ami=ami,null_ari=ari,
                        real_ami=real_row['ami'],real_ari=real_row['ari'],
                        delta_ami=(float(real_row['ami'])-ami) if ami is not None else None,
                        delta_ari=(float(real_row['ari'])-ari) if ari is not None else None,
                        null_clusters=stats['clusters'],null_median_size=stats['median_size'],
                        resolution=resolution,calibrated=calibrated))
                print(f'  null={rand_seed} Leiden={seed} K={target} clusters={stats["clusters"]} calibrated={calibrated}',flush=True)
        dump_csv(args.out/'null_ami_per_seed.csv',rows,list(rows[0]))
        dump_csv(args.out/'swap_diagnostics.csv',diagnostics,list(diagnostics[0]))
    summary=[]
    for target in targets:
        for pair in ('B-T','B-V'):
            records=[r for r in rows if r['target_clusters']==target and r['pair']==pair and r['null_ami'] is not None]
            full=len(records)==len(args.random_seeds)*len(seeds)
            for rs in args.random_seeds:
                part=[r['null_ami'] for r in records if r['random_seed']==rs]
                summary.append(dict(target_clusters=target,pair=pair,random_seed=rs,
                    mean_null_ami=float(np.mean(part)) if len(part)==len(seeds) else '',
                    mean_real_ami=float(np.mean([float(real_lookup[(target,s,pair)]['ami']) for s in seeds])) if full else '',
                    delta_real_minus_null=(float(np.mean([float(real_lookup[(target,s,pair)]['ami']) for s in seeds]))-float(np.mean(part))) if full else '',
                    calibrated=full))
    dump_csv(args.out/'null_ami_summary.csv',summary,
        ['target_clusters','pair','random_seed','mean_null_ami','mean_real_ami','delta_real_minus_null','calibrated'])
    save_json(args.out/'null_manifest.json',dict(config=out_config,
        all_null_partitions_calibrated=all(r['calibrated'] for r in rows),
        notes=['Only U-I incidences randomize; actual T/V partitions reused.',
               'Null graph edge count can differ naturally because different repeated cooccurrences.',
               '3 random graphs do NOT support p<0.05 empirical randomization significance.',
               'U-I degree preservation does NOT preserve sequence position or user-specific ordering.',
               'Real-vs-null AMI gap is descriptive; does not establish mapping correctness.']))
    print('\nSUCCESS:',args.out,flush=True)
    for r in summary:
        print(f"{r['pair']} K={r['target_clusters']} random={r['random_seed']} "
              f"Real={r['mean_real_ami']} Null={r['mean_null_ami']} Delta={r['delta_real_minus_null']}",flush=True)

if __name__=='__main__':main()
