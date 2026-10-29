#!/usr/bin/env python3
"""Controlled reweighting experiment on an existing fixed-topology graph.

Experiments: behavior-only, strength_x_consistency lambda=.3 (existing),
strength_x_consistency lambda=.1, mean lambda=.3.
No co-occurrence regeneration; outputs only analysis JSON (original graphs untouched).
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import math
import networkx as nx
from compare_graph_communities import communities, assignment, evaluate
from graph.multimodal_reweight import compute_multimodal_reliability

DATASETS = ('Industrial_and_Scientific', 'Musical_Instruments', 'Office_Products')


def load(path):
    with path.open('rb') as f:
        return pickle.load(f)


def variant(base, name, lam, mode):
    g = nx.Graph()
    g.add_nodes_from(base.nodes(data=True))
    for u, v, data in base.edges(data=True):
        b = float(data['normalized_behavior_score'])
        r = compute_multimodal_reliability(data.get('text_similarity'), data.get('image_similarity'),
            mode=mode, cosine_transform='clip01', missing_modality='behavior_only')
        m = r.multimodal_reliability
        w = b if m is None else (1 - lam) * b + lam * m
        if not math.isfinite(w) or w < 0:
            raise ValueError(f'Invalid weight at {(u,v)}: {w}')
        g.add_edge(u,v,weight=w)
    return g


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=DATASETS)
    p.add_argument('--graph-dir', type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resolution', type=float, default=1.0)
    args=p.parse_args()
    gd=args.graph_dir or Path('outputs')/args.dataset/'5core_v2_trainonly_count_co1_k50_mutual_lam0.3'
    behavior=load(gd/'collaborative.gpickle')
    current=load(gd/'multimodal_behavior.gpickle')
    if not isinstance(behavior,nx.Graph) or not isinstance(current,nx.Graph):
        raise TypeError('Expected NetworkX graphs')
    if set(behavior.nodes)!=set(current.nodes) or {frozenset(x) for x in behavior.edges}!={frozenset(x) for x in current.edges}:
        raise AssertionError('Topology mismatch')
    graphs={
        'behavior_only': behavior,
        'consistency_lam03': current,
        'consistency_lam01': variant(current,'consistency_lam01',.1,'strength_x_consistency'),
        'mean_lam03': variant(current,'mean_lam03',.3,'mean'),
    }
    # Check the formula reproduces the existing λ=.3 graph, before comparing anything.
    check=variant(current,'reproduction',.3,'strength_x_consistency')
    max_diff=max((abs(check[u][v]['weight']-current[u][v]['weight']) for u,v in current.edges),default=0)
    if max_diff>1e-6:
        raise AssertionError(f'Current graph formula not reproduced (max abs diff={max_diff:.8g}); abort')
    print(f'Formula verified (max abs diff={max_diff:.3g}); nodes={len(behavior):,}, edges={behavior.number_of_edges():,}',flush=True)
    out={
        'dataset':args.dataset,'graph_dir':str(gd),'seed':args.seed,'resolution':args.resolution,
        'node_count':len(behavior),'edge_count':behavior.number_of_edges(),'max_formula_difference':max_diff,
        'notes':['All candidates use precisely the same fixed topology.','Same-reference metrics compare partitions on identical graph weights; not independent ground-truth quality.','No held-out edge prediction or downstream recommendation validation is performed.'],
        'variants':{},
    }
    # Reference graphs held fixed for cross evaluation.
    references={'behavior_reference':behavior,'existing_mm_reference':current}
    for key,g in graphs.items():
        print('Louvain:',key,flush=True)
        part=communities(g,args.seed,args.resolution)
        labels=assignment(part)
        if set(labels)!=set(behavior.nodes):
            raise AssertionError(f'Missing nodes for {key}')
        out['variants'][key]={
            'self':evaluate(g,labels,part),
            'cross_reference':{refname:evaluate(ref,labels,part) for refname,ref in references.items()},
        }
        v=out['variants'][key]
        print(f"  clusters={v['self']['clusters']}, max_size={v['self']['max_cluster_size']}, "
              f"B-ref mod={v['cross_reference']['behavior_reference']['modularity']:.4f}, "
              f"MM-ref mod={v['cross_reference']['existing_mm_reference']['modularity']:.4f}",flush=True)
    dest=gd/'reweight_ablation'
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/'comparison.json'
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    print('ABLATION SUCCESS:',path,flush=True)

if __name__=='__main__':
    main()
