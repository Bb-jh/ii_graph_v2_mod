#!/usr/bin/env python3
"""Compare behavior-only vs multimodal-weighted communities on a fixed I-I topology.

Uses identical community algorithm/seed/resolution; evaluates both partitions on
identical reference graphs to avoid misleading cross-weight modularity comparisons.
Read-only graph inputs. Output: community_comparison.json and partitions TSV.
"""
from __future__ import annotations
import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
import networkx as nx
import numpy as np

CATEGORIES = ('Industrial_and_Scientific', 'Musical_Instruments', 'Office_Products')

def read_graph(path):
    with path.open('rb') as f:
        return pickle.load(f)

def communities(g, seed, resolution):
    if hasattr(nx.community, 'louvain_communities'):
        return nx.community.louvain_communities(g, weight='weight', seed=seed, resolution=resolution)
    try:
        import community as community_louvain
    except ImportError as exc:
        raise RuntimeError('Install python-louvain or upgrade networkx to >=2.8') from exc
    part = community_louvain.best_partition(g, weight='weight', random_state=seed, resolution=resolution)
    groups = {}
    for node, cluster in part.items():
        groups.setdefault(cluster, set()).add(node)
    return list(groups.values())

def assignment(groups):
    labels = {}
    for i, group in enumerate(groups):
        for node in group:
            if node in labels:
                raise ValueError('Node assigned more than once')
            labels[node] = i
    return labels

def evaluate(ref, labels, groups):
    total_edges = ref.number_of_edges()
    same_edges = 0
    intra_weight = 0.0
    total_weight = 0.0
    for u, v, a in ref.edges(data=True):
        w = float(a['weight'])
        total_weight += w
        if labels[u] == labels[v]:
            same_edges += 1
            intra_weight += w
    sizes = np.asarray([len(x) for x in groups], dtype=int)
    return {
        'modularity': float(nx.community.modularity(ref, groups, weight='weight')),
        'intra_edge_ratio': same_edges / total_edges if total_edges else None,
        'intra_weight_ratio': intra_weight / total_weight if total_weight else None,
        'clusters': len(groups),
        'singletons': int(np.count_nonzero(sizes == 1)),
        'clusters_under_5': int(np.count_nonzero(sizes < 5)),
        'median_cluster_size': float(np.median(sizes)),
        'max_cluster_size': int(sizes.max()),
        'nodes_in_clusters_ge5_pct': float(sizes[sizes >= 5].sum() / sizes.sum() * 100),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=CATEGORIES, default='Industrial_and_Scientific')
    p.add_argument('--graph-dir', type=Path, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resolution', type=float, default=1.0)
    p.add_argument('--output-dir', type=Path, default=None)
    args = p.parse_args()
    gd = args.graph_dir or Path('outputs') / args.dataset / '5core_v2_trainonly_count_co1_k50_mutual_lam0.3'
    out = args.output_dir or gd / 'community_validation'
    a = read_graph(gd / 'collaborative.gpickle')
    b = read_graph(gd / 'multimodal_behavior.gpickle')
    if not isinstance(a, nx.Graph) or not isinstance(b, nx.Graph) or a.is_directed() or b.is_directed():
        raise TypeError('Expected undirected NetworkX graphs')
    if set(a.nodes) != set(b.nodes) or {frozenset(x) for x in a.edges} != {frozenset(x) for x in b.edges}:
        raise AssertionError('Graph topology differs: cannot make controlled weight-only comparison')
    for graph, label in ((a,'behavior'),(b,'multimodal')):
        weights = np.fromiter((data.get('weight',float('nan')) for _,_,data in graph.edges(data=True)), dtype=float)
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise AssertionError(f'{label} has non-finite or negative weights')
    # Graph weights have different ranges: normalize each by positive global mean.
    # This leaves modularity/partition unchanged for each graph (scale-invariant).
    print(f'Graph verified: nodes={a.number_of_nodes():,}, edges={a.number_of_edges():,}', flush=True)
    parts = {}
    labels = {}
    for label, graph in [('behavior',a),('multimodal',b)]:
        print(f'Running Louvain: {label}', flush=True)
        parts[label] = communities(graph,args.seed,args.resolution)
        labels[label] = assignment(parts[label])
        assert len(labels[label]) == len(graph)
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ordered = sorted(a.nodes())
    ari = float(adjusted_rand_score([labels['behavior'][n] for n in ordered],[labels['multimodal'][n] for n in ordered]))
    nmi = float(normalized_mutual_info_score([labels['behavior'][n] for n in ordered],[labels['multimodal'][n] for n in ordered]))
    comparison = {
        'dataset':args.dataset,'graph_dir':str(gd),'algorithm':'Louvain',
        'seed':args.seed,'resolution':args.resolution,
        'nodes':a.number_of_nodes(),'edges':a.number_of_edges(),
        'evaluations':{
            name:{'on_behavior_graph':evaluate(a,labels[name],parts[name]),
                  'on_multimodal_graph':evaluate(b,labels[name],parts[name])}
            for name in ['behavior','multimodal']
        },
        'partition_agreement':{'ARI':ari,'NMI':nmi},
        'interpretation_note':'Shared-reference modularity/intra-edge scores are descriptive, not independent proof of recommendation quality. Both partitions are derived from the same graph structure; evaluate on held-out TRAIN-only edges or downstream performance before claiming improvement.'
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/'community_comparison.json').write_text(json.dumps(comparison,indent=2,ensure_ascii=False),encoding='utf-8')
    with (out/'community_labels.tsv').open('w',encoding='utf-8') as f:
        f.write('parent_asin\tbehavior_cluster\tmultimodal_cluster\n')
        for n in ordered:
            f.write(f'{n}\t{labels["behavior"][n]}\t{labels["multimodal"][n]}\n')
    for name in ['behavior','multimodal']:
        print(name, 'on behavior:', {k:comparison['evaluations'][name]['on_behavior_graph'][k] for k in ('modularity','intra_edge_ratio','clusters','singletons','nodes_in_clusters_ge5_pct')}, flush=True)
        print(name, 'on multimodal:',{k:comparison['evaluations'][name]['on_multimodal_graph'][k] for k in ('modularity','intra_edge_ratio')},flush=True)
    print(f'Partition agreement: ARI={ari:.4f}, NMI={nmi:.4f}',flush=True)
    print('VALIDATION SUCCESS:',out,flush=True)
if __name__=='__main__': main()
