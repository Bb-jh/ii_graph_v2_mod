#!/usr/bin/env python3
"""Train-only independent B/T/V graph and granularity agreement pilot.

Run with --help for CLI. Intentionally does not import the project's fusion code.
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def load_mapping(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    assert rows and {'item_id', 'parent_asin'} <= set(rows[0]), 'Mapping columns missing'
    ids = []
    for i, row in enumerate(rows):
        assert int(row['item_id']) == i, f'Unexpected 0-based mapping at {i}'
        ids.append(row['parent_asin'])
    assert len(ids) == len(set(ids))
    return ids


def load_train_only(sequences, n):
    users = defaultdict(list)
    with open(sequences, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            assert len(parts) == 2, f'Bad sequence line {lineno}'
            u, it = map(int, parts)
            assert 1 <= it <= n, (lineno, it, n)
            users[u].append(it - 1)
    assert users, 'No sequences'
    rows, cols = [], []
    n_train_events = 0
    for row, (_, seq) in enumerate(sorted(users.items())):
        assert len(seq) >= 3, 'Missing leave-two-out history'
        train = seq[:-2]
        n_train_events += len(train)
        for item in set(train):
            rows.append(row)
            cols.append(item)
    x = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(users), n))
    x.sum_duplicates()
    x.data[:] = 1
    degree = np.asarray(x.sum(axis=0)).ravel().astype(np.int64)
    return x, degree, n_train_events


def load_embeddings(path, ids):
    with np.load(path, allow_pickle=False) as z:
        lookup = {str(asin): i for i, asin in enumerate(z['item_ids'])}
        assert len(lookup) == len(z['item_ids'])
        indices = np.array([lookup.get(v, -1) for v in ids], dtype=np.int64)
        if np.any(indices < 0):
            raise ValueError(f'NPZ missing {np.count_nonzero(indices<0)} mapped items')
        data = {}
        for name in ('text', 'image'):
            emb = np.asarray(z[f'{name}_embeddings'][indices], dtype=np.float32)
            valid = np.asarray(z[f'{name}_valid_mask'][indices], dtype=bool)
            valid &= np.isfinite(emb).all(axis=1)
            valid &= np.linalg.norm(emb, axis=1) > 1e-10
            data[name] = (emb, valid)
    return data


def sym_knn(indices, weights, n, mode='union'):
    k = indices.shape[1]
    r = np.repeat(np.arange(n, dtype=np.int64), k)
    c = indices.reshape(-1).astype(np.int64)
    w = weights.reshape(-1).astype(np.float32)
    keep = (c >= 0) & (c != r) & np.isfinite(w) & (w > 0)
    directed = sparse.coo_matrix((w[keep], (r[keep], c[keep])), shape=(n, n)).tocsr()
    directed.eliminate_zeros()
    if mode == 'mutual':
        graph = directed.minimum(directed.T)
    else:
        graph = directed.maximum(directed.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph.tocsr()


def behavior_graph(x, degree, selected, k, mode='union'):
    # Co-occurrence is computed from training histories only. Rows with no training
    # interaction remain in the universe but are not used in selected graph.
    co = (x[:, selected].T @ x[:, selected]).tocsr()
    co.setdiag(0)
    co.eliminate_zeros()
    inv = np.reciprocal(np.sqrt(np.maximum(degree[selected], 1)), dtype=np.float64)
    co = (sparse.diags(inv) @ co @ sparse.diags(inv)).tocsr()
    co.data = co.data.astype(np.float32)
    n = len(selected)
    idx = np.full((n, k), -1, dtype=np.int32)
    val = np.zeros((n, k), dtype=np.float32)
    for i in range(n):
        a, b = co.indptr[i], co.indptr[i+1]
        if a == b: continue
        ids, vv = co.indices[a:b], co.data[a:b]
        take = np.argsort(-vv, kind='stable')[:k]
        idx[i, :len(take)] = ids[take]
        val[i, :len(take)] = vv[take]
    return sym_knn(idx, val, n, mode), idx


def semantic_knn_candidates(emb, valid, selected, kmax, threads=4):
    """Retrieve the same directed top-k prefix once, for density calibration."""
    import faiss
    z = np.ascontiguousarray(emb[selected], dtype=np.float32)
    vm = np.asarray(valid[selected], dtype=bool)
    z[~vm] = 0
    faiss.normalize_L2(z)
    faiss.omp_set_num_threads(threads)
    idx = faiss.IndexFlatIP(z.shape[1])
    idx.add(z)
    # Query extra candidates to exclude self / invalid vectors. Rare invalids may
    # still result in <kmax neighbors; report the shortfall if this happens.
    query_k = min(len(z), kmax + max(64, int((~vm).sum()) + 2))
    out_idx = np.full((len(z), kmax), -1, dtype=np.int32)
    out_sim = np.zeros((len(z), kmax), dtype=np.float32)
    block = 1024
    for start in range(0, len(z), block):
        end = min(len(z), start + block)
        ds, js = idx.search(z[start:end], query_k)
        for local, i in enumerate(range(start, end)):
            if not vm[i]:
                continue
            good = (js[local] != i) & (js[local] >= 0) & vm[js[local]] & (ds[local] > 0)
            good_idx, good_sim = js[local][good][:kmax], ds[local][good][:kmax]
            out_idx[i, :len(good_idx)] = good_idx
            out_sim[i, :len(good_sim)] = good_sim
    return out_idx, out_sim


def matched_semantic_graph(emb, valid, selected, reference_edges, minimum_k,
                           mode='union', threads=4, tolerance=0.03):
    """Select directed k by measured undirected edge count, not mean-degree heuristic."""
    n = len(selected)
    approx = max(minimum_k, int(np.ceil(2 * reference_edges / n)))
    kmax = min(n - 1, max(approx * 3, minimum_k + 32))
    inds, sims = semantic_knn_candidates(emb, valid, selected, kmax, threads)
    lo, hi = minimum_k, kmax
    tested = {}

    def measure(k):
        if k not in tested:
            g = sym_knn(inds[:, :k], sims[:, :k], n, mode)
            tested[k] = (g.nnz // 2, g)
        return tested[k][0]

    while lo <= hi:
        mid = (lo + hi) // 2
        e = measure(mid)
        if e < reference_edges:
            lo = mid + 1
        elif e > reference_edges:
            hi = mid - 1
        else:
            break
    for k in (minimum_k, max(minimum_k, hi), min(kmax, lo), kmax):
        measure(k)
    best = min(tested, key=lambda k: (abs(tested[k][0]-reference_edges), k))
    edge_count, graph = tested[best]
    relative_error = abs(edge_count-reference_edges)/max(reference_edges, 1)
    if relative_error > tolerance:
        raise ValueError(f'Density mismatch: reference={reference_edges}, selected={edge_count}, '
                         f'k={best}, relative_error={relative_error:.2%} > {tolerance:.2%}. '
                         'Increase --density-tolerance only as a documented exploratory choice.')
    return graph, inds[:, :best], {'directed_k': best, 'edges': edge_count,
                                  'relative_edge_error': relative_error}

def to_igraph(matrix):
    import igraph as ig
    tri = sparse.triu(matrix, k=1).tocoo()
    edges = list(zip(tri.row.tolist(), tri.col.tolist()))
    g = ig.Graph(n=matrix.shape[0], edges=edges, directed=False)
    g.es['weight'] = tri.data.astype(float).tolist()
    return g


def leiden(g, resolution, seed, iterations=3):
    import leidenalg
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights='weight', resolution_parameter=resolution, seed=seed, n_iterations=iterations)
    labels = np.asarray(part.membership, dtype=np.int32)
    sizes = np.bincount(labels)
    return labels, {'n_clusters': int(len(sizes)), 'median_cluster_size': float(np.median(sizes)), 'singleton_fraction': float((sizes == 1).sum()/len(sizes))}


def search_resolution(g, targets, seed, rounds=18, low=1e-5, high=1000.0,
                      tolerance=0.25):
    """Calibrate multiple median targets using one cached log-resolution sweep.

    Leiden cannot guarantee an arbitrary target median. Failed targets are explicitly
    marked; never silently call the nearest partition an achieved calibration.
    """
    resolutions = np.geomspace(low, high, rounds)
    cache = {}
    for r in resolutions:
        labels, stats = leiden(g, float(r), seed)
        cache[float(r)] = (labels, dict(stats, resolution=float(r)))
    # Refine around the two adjacent sweep points closest to each requested
    # target. This adds resolution coverage without assuming strict monotonicity.
    for target in targets:
        ranked = sorted(range(len(resolutions)),
                        key=lambda i: abs(np.log(max(cache[float(resolutions[i])][1]['median_cluster_size'],1)/target)))
        add = set()
        for i in ranked[:2]:
            if i > 0:
                add.update(np.geomspace(resolutions[i-1], resolutions[i], 6))
            if i + 1 < len(resolutions):
                add.update(np.geomspace(resolutions[i], resolutions[i+1], 6))
        for r in sorted(add):
            r = float(r)
            if r not in cache:
                labels, stats = leiden(g, r, seed)
                cache[r] = (labels, dict(stats, resolution=r))
    results = {}
    for target in targets:
        best = min(cache.values(), key=lambda pair: abs(np.log(max(pair[1]['median_cluster_size'], 1)/target)))
        observed = best[1]['median_cluster_size']
        relerr = abs(observed-target)/target
        results[target] = (best[0], dict(best[1], target_median_size=target,
                                       relative_target_error=relerr,
                                       calibrated=bool(relerr <= tolerance)))
    return results

def overlap(a, b, k, n):
    totals = []
    for i in range(n):
        aa = set(int(v) for v in a[i, :k] if v >= 0)
        bb = set(int(v) for v in b[i, :k] if v >= 0)
        if len(aa) >= k and len(bb) >= k:
            totals.append(len(aa & bb) / k)
    return {'mean': float(np.mean(totals)) if totals else None, 'eligible_items': len(totals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--mapping', type=Path, required=True)
    ap.add_argument('--sequences', type=Path, required=True)
    ap.add_argument('--embeddings', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--behavior-k', type=int, default=50)
    ap.add_argument('--symmetrize', choices=['union','mutual'], default='union')
    ap.add_argument('--median-targets', type=int, nargs='+', default=[128, 64, 32, 16])
    ap.add_argument('--overlap-ks', type=int, nargs='+', default=[5,10,20])
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--resolution-rounds', type=int, default=18)
    ap.add_argument('--density-tolerance', type=float, default=0.03)
    ap.add_argument('--median-tolerance', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min-train-degree', type=int, default=1)
    args = ap.parse_args()
    for p in (args.mapping, args.sequences, args.embeddings):
        assert p.is_file(), f'Missing {p}'
    assert args.behavior_k >= max(args.overlap_ks)
    assert args.threads >= 1
    args.out.mkdir(parents=True, exist_ok=False)
    ids = load_mapping(args.mapping)
    x, degrees, n_events = load_train_only(args.sequences, len(ids))
    emb = load_embeddings(args.embeddings, ids)
    valid = (degrees >= args.min_train_degree) & emb['text'][1] & emb['image'][1]
    selected = np.flatnonzero(valid)
    print('Item universe:',len(ids),'selected B/T/V common set:',len(selected),'train events:',n_events,flush=True)
    assert len(selected) > 500
    bg, bn = behavior_graph(x, degrees, selected, args.behavior_k, args.symmetrize)
    reference_edges = bg.nnz // 2
    print('B edges:', reference_edges, 'average degree:', bg.nnz/bg.shape[0], flush=True)
    tg, tn, tmeta = matched_semantic_graph(*emb['text'], selected, reference_edges,
                                          max(args.overlap_ks), args.symmetrize,
                                          args.threads, args.density_tolerance)
    print('T edges:', tmeta, flush=True)
    vg, vn, vmeta = matched_semantic_graph(*emb['image'], selected, reference_edges,
                                          max(args.overlap_ks), args.symmetrize,
                                          args.threads, args.density_tolerance)
    print('V edges:', vmeta, flush=True)
    graphs = {'B': bg, 'T': tg, 'V': vg}
    manifest = dict(dataset=args.dataset, seed=args.seed, common_items=len(selected),
                    total_items=len(ids), users=x.shape[0], train_events=n_events,
                    train_only=True, embedding='CLIP',
                    weights='B cooccurrence/sqrt(item_degrees), T/V cosine',
                    symmetry=args.symmetrize, behavior_topk=args.behavior_k,
                    semantic_directed_k={'T': tmeta['directed_k'], 'V': vmeta['directed_k']},
                    graph_edge_counts={v: int(g.nnz//2) for v,g in graphs.items()},
                    density_relative_error={'T': tmeta['relative_edge_error'],
                                            'V': vmeta['relative_edge_error']},
                    caveats=['No diffusion applied', 'Only one seed used in pilot',
                             'KNN neighborhood estimates on density-calibrated semantic graphs'])
    save_json(args.out / 'manifest.json', manifest)
    igs = {key: to_igraph(g) for key,g in graphs.items()}
    calibrations = {key: search_resolution(g, args.median_targets, args.seed,
                    rounds=args.resolution_rounds, tolerance=args.median_tolerance)
                    for key,g in igs.items()}
    cal_rows=[]
    with (args.out/'calibration.csv').open('w', newline='') as f:
        fields=['view','target_median','observed_median','clusters','resolution',
                'relative_target_error','calibrated']
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for key in igs:
            for target in args.median_targets:
                _, stats=calibrations[key][target]
                row={'view':key,'target_median':target,
                     'observed_median':stats['median_cluster_size'],
                     'clusters':stats['n_clusters'],
                     'resolution':stats['resolution'],
                     'relative_target_error':stats['relative_target_error'],
                     'calibrated':stats['calibrated']}
                writer.writerow(row)
                cal_rows.append(row)
                print('Calibration:',row,flush=True)
    with (args.out/'agreement.csv').open('w',newline='') as f:
        fields=['target_median','pair','ami','ari','calibrated',
                'B_median','T_median','V_median','B_clusters','T_clusters','V_clusters']
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for target in args.median_targets:
            partitions={key: calibrations[key][target][0] for key in igs}
            stats={key: calibrations[key][target][1] for key in igs}
            valid=all(stats[key]['calibrated'] for key in igs)
            for a,b in [('B','T'),('B','V'),('T','V')]:
                writer.writerow(dict(target_median=target,pair=f'{a}-{b}',
                    ami=float(adjusted_mutual_info_score(partitions[a],partitions[b])) if valid else '',
                    ari=float(adjusted_rand_score(partitions[a],partitions[b])) if valid else '',
                    calibrated=valid,
                    **{f'{key}_median':stats[key]['median_cluster_size'] for key in stats},
                    **{f'{key}_clusters':stats[key]['n_clusters'] for key in stats}))
            print('target',target,'actual median',
                  {k:v['median_cluster_size'] for k,v in stats.items()},
                  'calibrated:',valid,flush=True)
    manifest['granularity_calibration'] = cal_rows
    save_json(args.out / 'manifest.json', manifest)
    with (args.out/'neighbor_overlap.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['k','pair','mean','eligible_items'])
        writer.writeheader()
        for k in args.overlap_ks:
            for a,b,na,nb in [('B','T',bn,tn),('B','V',bn,vn),('T','V',tn,vn)]:
                result=overlap(na,nb,k,len(selected))
                writer.writerow(dict(k=k,pair=f'{a}-{b}',**result))
    print('SUCCESS:',args.out,flush=True)

if __name__=='__main__': main()
