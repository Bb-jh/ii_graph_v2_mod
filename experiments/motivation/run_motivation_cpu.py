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


def semantic_graph(emb, valid, selected, k, mode='union', threads=4):
    import faiss
    z = np.ascontiguousarray(emb[selected], dtype=np.float32)
    vm = valid[selected]
    z[~vm] = 0
    faiss.normalize_L2(z)
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatIP(z.shape[1])
    index.add(z)
    # Oversample so self and invalid items can be rejected safely.
    query_k = min(len(z), max(k + 32, 2*k + 1))
    distances, neigh = index.search(z, query_k)
    top = np.full((len(z), k), -1, dtype=np.int32)
    scores = np.zeros((len(z), k), dtype=np.float32)
    for i in range(len(z)):
        if not vm[i]: continue
        js = neigh[i]
        ds = distances[i]
        ok = (js != i) & (js >= 0) & vm[js] & (ds > 0)
        js, ds = js[ok][:k], ds[ok][:k]
        top[i, :len(js)] = js
        scores[i, :len(js)] = ds
    return sym_knn(top, scores, len(z), mode), top


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


def search_resolution(g, target, seed, low=0.0001, high=10.0, rounds=12):
    # Calibrate to target median size independently on each graph.
    candidates = []
    for res in np.geomspace(low, high, rounds):
        labels, stats = leiden(g, float(res), seed)
        candidates.append((abs(np.log(max(stats['median_cluster_size'],1)/target)), labels, stats, res))
    best = min(candidates, key=lambda x: x[0])
    return best[1], dict(best[2], resolution=float(best[3]), target_median_size=target)


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
    ap.add_argument('--resolution-rounds', type=int, default=6)
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
    # Semantic k is chosen by density of B, based on B's mean degree.
    average_degree = bg.nnz / bg.shape[0]
    semantic_k = min(max(1, round(average_degree / (2 if args.symmetrize=='union' else 1))), args.behavior_k * 2)
    semantic_k = max(semantic_k, max(args.overlap_ks))
    print('B edges:',bg.nnz//2,'average degree:',average_degree,'semantic directed k:',semantic_k,flush=True)
    tg, tn = semantic_graph(*emb['text'], selected, semantic_k, args.symmetrize, args.threads)
    vg, vn = semantic_graph(*emb['image'], selected, semantic_k, args.symmetrize, args.threads)
    graphs = {'B': bg, 'T': tg, 'V': vg}
    manifest = dict(dataset=args.dataset, seed=args.seed, common_items=len(selected), total_items=len(ids), users=x.shape[0], train_events=n_events, train_only=True, embedding='CLIP', weights='B cooccurrence/sqrt(item_degrees), T/V cosine', symmetry=args.symmetrize, behavior_topk=args.behavior_k, semantic_directed_k=semantic_k, graph_edge_counts={v: int(g.nnz//2) for v,g in graphs.items()}, caveats=['T/V edge counts are approximately, not exactly, density matched to B', 'AMI/ARI computed on a common item set; disconnected nodes are retained', 'No diffusion applied', 'Only one seed used in pilot'])
    save_json(args.out / 'manifest.json', manifest)
    igs = {key: to_igraph(g) for key,g in graphs.items()}
    with (args.out/'agreement.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['target_median','pair','ami','ari','B_median','T_median','V_median','B_clusters','T_clusters','V_clusters'])
        writer.writeheader()
        for target in args.median_targets:
            partitions={}; stats={}
            for key,g in igs.items():
                partitions[key], stats[key]=search_resolution(g,target,args.seed,rounds=args.resolution_rounds)
            for a,b in [('B','T'),('B','V'),('T','V')]:
                writer.writerow(dict(target_median=target,pair=f'{a}-{b}',ami=float(adjusted_mutual_info_score(partitions[a],partitions[b])),ari=float(adjusted_rand_score(partitions[a],partitions[b])), **{f'{key}_median':stats[key]['median_cluster_size'] for key in stats}, **{f'{key}_clusters':stats[key]['n_clusters'] for key in stats}))
            print('target',target,'actual median',{k:v['median_cluster_size'] for k,v in stats.items()},flush=True)
    with (args.out/'neighbor_overlap.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['k','pair','mean','eligible_items'])
        writer.writeheader()
        for k in args.overlap_ks:
            for a,b,na,nb in [('B','T',bn,tn),('B','V',bn,vn),('T','V',tn,vn)]:
                result=overlap(na,nb,k,len(selected))
                writer.writerow(dict(k=k,pair=f'{a}-{b}',**result))
    print('SUCCESS:',args.out,flush=True)

if __name__=='__main__': main()
