#!/usr/bin/env python3
"""Read-only, cross-dataset diagnostics for the shared I-I graph.

Analyzes behavior graph and fixed-topology multimodal-reweighted graph.
Requires numpy, networkx. Uses stdlib pickle on locally trusted graph files only.
"""
import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
import networkx as nx
import numpy as np

DATASETS = ('Industrial_and_Scientific', 'Musical_Instruments', 'Office_Products')

def load_graph(p):
    with p.open('rb') as f:
        return pickle.load(f)  # Only open locally generated trusted files.

def stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'count': 0}
    return {'count': int(len(x)), 'mean': float(x.mean()), 'std': float(x.std()),
            'min': float(x.min()), 'p10': float(np.quantile(x,.1)),
            'p25': float(np.quantile(x,.25)), 'median': float(np.quantile(x,.5)),
            'p75': float(np.quantile(x,.75)), 'p90': float(np.quantile(x,.9)),
            'p99': float(np.quantile(x,.99)), 'max': float(x.max())}

def pearson(x, y):
    x = np.asarray(x,dtype=np.float64); y = np.asarray(y,dtype=np.float64)
    if len(x)<2 or x.std()==0 or y.std()==0: return None
    return float(np.corrcoef(x,y)[0,1])

def rank(x):
    # Average ranks for ties (no scipy dependency)
    x=np.asarray(x); idx=np.argsort(x,kind='mergesort'); sorted_x=x[idx]
    boundary=np.r_[0,np.flatnonzero(sorted_x[1:]!=sorted_x[:-1])+1,len(x)]
    out=np.empty(len(x),dtype=np.float64)
    for a,b in zip(boundary[:-1],boundary[1:]):out[idx[a:b]]=(a+b-1)/2
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=DATASETS,default='Industrial_and_Scientific')
    p.add_argument('--graph-dir',type=Path,default=None)
    p.add_argument('--output',type=Path,default=None)
    a=p.parse_args()
    root=a.graph_dir or Path('outputs')/a.dataset/'5core_v2_trainonly_count_co1_k50_mutual_lam0.3'
    base=load_graph(root/'collaborative.gpickle'); mm=load_graph(root/'multimodal_behavior.gpickle')
    if set(base.nodes)!=set(mm.nodes) or {frozenset(e) for e in base.edges}!={frozenset(e) for e in mm.edges()}:
        raise RuntimeError('Topology mismatch')
    edges=list(base.edges()); E=len(edges)
    if not E: raise RuntimeError('No edges')
    b_raw=np.fromiter((float(base[u][v].get('behavior_score',base[u][v]['weight'])) for u,v in edges),float,count=E)
    b_norm=np.fromiter((float(mm[u][v]['normalized_behavior_score']) for u,v in edges),float,count=E)
    w=np.fromiter((float(mm[u][v]['weight']) for u,v in edges),float,count=E)
    t=np.fromiter((float(mm[u][v].get('text_similarity_01')) if mm[u][v].get('text_similarity_01') is not None else np.nan for u,v in edges),float,count=E)
    im=np.fromiter((float(mm[u][v].get('image_similarity_01')) if mm[u][v].get('image_similarity_01') is not None else np.nan for u,v in edges),float,count=E)
    reli=np.fromiter((float(mm[u][v].get('multimodal_reliability')) if mm[u][v].get('multimodal_reliability') is not None else np.nan for u,v in edges),float,count=E)
    valid=np.isfinite(reli); same=np.isfinite(t)&np.isfinite(im)
    if not np.isfinite(b_raw).all() or not np.isfinite(w).all():raise RuntimeError('non-finite behavior/final weights')
    co1=np.isclose(b_raw,1.0); co2=~co1
    # Count-based graph with --weighting count only; otherwise co1 is not reliable.
    count_based= bool(np.all(np.isclose(b_raw,np.rint(b_raw))) and np.all(b_raw>=1))
    k=max(1,int(round(E*.1)))
    top_b=set(np.argpartition(b_norm,-k)[-k:]); top_w=set(np.argpartition(w,-k)[-k:])
    top_w_idx=np.fromiter(top_w,int)
    deg=np.fromiter((d for _,d in mm.degree()),int,count=mm.number_of_nodes())
    result={'dataset':a.dataset,'graph_dir':str(root),'nodes':mm.number_of_nodes(),'edges':E,
        'topology_preserved':True,
        'behavior_raw':stats(b_raw),'behavior_normalized':stats(b_norm),'final_weight':stats(w),
        'text_similarity':stats(t),'image_similarity':stats(im),'multimodal_reliability':stats(reli),
        'evidence_coverage':{'both_modalities':int(same.sum()),'both_pct':float(same.mean()*100),
                             'reliability_available':int(valid.sum()),'reliability_pct':float(valid.mean()*100)},
        'weight_change':{'delta_from_normalized_behavior':stats(w-b_norm),
                         'pearson_behavior_vs_final':pearson(b_norm,w),
                         'spearman_behavior_vs_final':pearson(rank(b_norm),rank(w)),
                         'top10pct_edge_overlap':float(len(top_b&top_w)/k)},
        'degree':stats(deg),
        'top10pct_final_edges':{'behavior_score':stats(b_raw[top_w_idx]),
                                'text_similarity':stats(t[top_w_idx]),
                                'image_similarity':stats(im[top_w_idx]),
                                'reliability':stats(reli[top_w_idx])}}
    if count_based:
        result['cooccurrence']={'co1_edges':int(co1.sum()),'co1_pct':float(co1.mean()*100),
            'co_ge2_edges':int(co2.sum()),'co_ge2_pct':float(co2.mean()*100),
            'co1':{'final_weight':stats(w[co1]),'reliability':stats(reli[co1]),'text_sim':stats(t[co1]),'image_sim':stats(im[co1])},
            'co_ge2':{'final_weight':stats(w[co2]),'reliability':stats(reli[co2]),'text_sim':stats(t[co2]),'image_sim':stats(im[co2])},
            'top10pct_co1_pct':float(co1[top_w_idx].mean()*100)}
    else:
        result['cooccurrence']={'note':'Unavailable: behavior weighting is not raw count'}
    path=a.output or root/'edge_quality.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Edges: {E:,} | Evidence both: {same.mean()*100:.2f}%')
    if count_based:print(f'Co-occurrence=1: {co1.sum():,} ({co1.mean()*100:.2f}%)')
    print(f'Behavior/final Spearman: {result["weight_change"]["spearman_behavior_vs_final"]}')
    print(f'Top 10% edge overlap: {result["weight_change"]["top10pct_edge_overlap"]:.3f}')
    print(f'EDGE QUALITY SUCCESS: {path}')
if __name__=='__main__':main()
