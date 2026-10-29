#!/usr/bin/env python3
"""Leakage-controlled, exploratory edge holdout test for graph partitions.

Identical observed topology for every variant. Holdout edges are removed BEFORE
Louvain. No heldout edge attributes are used to build weights or communities.
This evaluates graph-topology generalization, NOT future recommendation accuracy.
"""
import argparse
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np

DATASETS = ('Industrial_and_Scientific', 'Musical_Instruments', 'Office_Products')


def read_graph(path):
    with path.open('rb') as f:
        return pickle.load(f)


def edge_weight(attr, variant):
    b = float(attr['normalized_behavior_score'])
    if variant == 'behavior_only':
        return b
    if variant.startswith('consistency'):
        m = attr.get('multimodal_reliability')
        lam = 0.1 if variant == 'consistency_lam01' else 0.3
    elif variant == 'mean_lam03':
        t, v = attr.get('text_similarity_01'), attr.get('image_similarity_01')
        m = None if t is None or v is None else 0.5 * (float(t) + float(v))
        lam = 0.3
    else:
        raise ValueError(variant)
    # Missing any modality: preserve behavior exactly (original policy).
    if m is None:
        return b
    return (1.0-lam)*b + lam*float(m)


def pick_negatives(positive_edges, all_neighbors, nodes, rng):
    """Source-matched random non-edges; reject ALL original graph edges."""
    negatives = []
    n = len(nodes)
    for u, _ in positive_edges:
        forbidden = all_neighbors[u]
        for _ in range(10000):
            v = nodes[int(rng.integers(n))]
            if v != u and v not in forbidden:
                negatives.append((u, v))
                break
        else:
            raise RuntimeError(f'Could not sample negative for {u}')
    return negatives


def evaluate_partition(groups, pos, neg):
    pos_rate = sum(groups[u] == groups[v] for u, v in pos) / len(pos)
    neg_rate = sum(groups[u] == groups[v] for u, v in neg) / len(neg)
    # Binary community-membership score; ROC AUC of this binary score.
    auc = 0.5 + 0.5*(pos_rate-neg_rate)
    return {
        'heldout_same_community_rate': pos_rate,
        'nonedge_same_community_rate': neg_rate,
        'community_lift': pos_rate / neg_rate if neg_rate else None,
        'binary_link_auc': auc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=DATASETS, required=True)
    ap.add_argument('--graph-root', type=Path, default=Path('outputs'))
    ap.add_argument('--graph-tag', default='5core_v2_trainonly_count_co1_k50_mutual_lam0.3')
    ap.add_argument('--holdout', type=float, default=0.1)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--resolution', type=float, default=1.0)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()
    if not (0 < args.holdout < 0.5):
        ap.error('--holdout must be between 0 and 0.5')
    graph_dir = args.graph_root / args.dataset / args.graph_tag
    path = graph_dir / 'multimodal_behavior.gpickle'
    if not path.is_file():
        ap.error(f'Graph not found: {path}')
    original = read_graph(path)
    if not isinstance(original, nx.Graph) or original.is_directed():
        raise TypeError('Expected an undirected NetworkX Graph')
    if not original.number_of_edges():
        raise ValueError('No edges')
    edges = list(original.edges())
    nodes = list(original.nodes())
    all_neighbors = {u: set(original.neighbors(u)) for u in nodes}
    variants = ['behavior_only','consistency_lam03','consistency_lam01','mean_lam03']
    print(f'Loaded {len(nodes):,} nodes and {len(edges):,} edges', flush=True)
    results = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(edges))
        n_hold = max(1, int(len(edges)*args.holdout))
        heldout = [edges[int(j)] for j in perm[:n_hold]]
        retained = [edges[int(j)] for j in perm[n_hold:]]
        negatives = pick_negatives(heldout, all_neighbors, nodes, rng)
        print(f'Seed {seed}: train_edges={len(retained):,}, heldout={len(heldout):,}', flush=True)
        row = {'seed':seed, 'retained_edges':len(retained), 'heldout_edges':len(heldout), 'variants':{}}
        for variant in variants:
            graph = nx.Graph()
            graph.add_nodes_from(nodes)
            for u,v in retained:
                w = edge_weight(original[u][v], variant)
                if not np.isfinite(w) or w < 0:
                    raise ValueError(f'Invalid weight {variant} {u},{v}: {w}')
                graph.add_edge(u,v,weight=w)
            comms = nx.community.louvain_communities(
                graph, weight='weight', seed=seed, resolution=args.resolution
            )
            groups = {u:j for j,group in enumerate(comms) for u in group}
            scores = evaluate_partition(groups,heldout,negatives)
            scores.update({'clusters':len(comms),'max_cluster_size':max(map(len,comms))})
            row['variants'][variant] = scores
            print(f'  {variant:22s} AUC={scores["binary_link_auc"]:.5f} '
                  f'positive={scores["heldout_same_community_rate"]:.4f} '
                  f'negative={scores["nonedge_same_community_rate"]:.4f}',flush=True)
            del graph
        results.append(row)
    summary = {}
    for name in variants:
        summary[name] = {}
        for metric in ['binary_link_auc','heldout_same_community_rate','nonedge_same_community_rate']:
            values = [r['variants'][name][metric] for r in results]
            summary[name][metric+'_mean'] = float(np.mean(values))
            summary[name][metric+'_std'] = float(np.std(values))
    output = args.output or graph_dir/'heldout_validation'/'comparison.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    payload = {
        'dataset':args.dataset,'source_graph':str(path), 'method':'edge holdout, identical observed topology, source-matched random nonedges',
        'holdout_fraction':args.holdout,'seeds':args.seeds,'resolution':args.resolution,
        'caveat':'Exploratory structural evaluation; nonedges are unlabeled, and withheld edges originate from a graph already pruned using full train cooccurrence. Not temporal recommendation evaluation.',
        'summary':summary,'runs':results,
    }
    output.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(f'VALIDATION SUCCESS: {output}',flush=True)

if __name__ == '__main__':
    main()
