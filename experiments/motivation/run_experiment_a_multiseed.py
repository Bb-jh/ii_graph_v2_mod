#!/usr/bin/env python3
"""Experiment A: controlled multi-seed independent B/T/V partition agreement.

Place next to run_motivation_cpu_v2.py. No changes to existing V2/V3 files.
Train-only graph data, common item universe, density-matched B/T/V graphs.
Calibrate cluster *count* separately per view/seed, record median, IQR, singletons.
Compute paired-seed cross view and all seed-pair within view AMI/ARI.
Creates reusable graph cache + per-partition checkpoints, supports reruns.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from run_motivation_cpu_v2 import (load_mapping, load_train_only, load_embeddings,
    behavior_graph, matched_semantic_graph, to_igraph, leiden, save_json)

VIEWS = ('B','T','V')
PAIRS = (('B','T'),('B','V'),('T','V'))


def dump_csv(path, rows, columns):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def partition_stats(labels):
    sizes = np.bincount(labels)
    return dict(clusters=int(len(sizes)), median_size=float(np.median(sizes)),
        q25_size=float(np.percentile(sizes, 25)), q75_size=float(np.percentile(sizes,75)),
        singleton_fraction_clusters=float(np.mean(sizes==1)),
        singleton_fraction_items=float(np.sum(sizes==1)/len(labels)),
        largest_cluster=int(sizes.max()))


def calibrate_count(g, target, seed, cache, low, high, max_evals):
    """Log-space bracket + bisection on observed count; no assumption of perfect monotonicity.

    Share cache across targets for same view+seed. Select nearest absolute count.
    """
    def evaluate(res):
        key=float(res)
        if key not in cache:
            lab, st=leiden(g, key, seed)
            cache[key]=(lab, st)
        return cache[key][1]['n_clusters']
    lo,hi=float(low),float(high)
    nlo,nhi=evaluate(lo),evaluate(hi)
    if not (min(nlo,nhi) <= target <= max(nlo,nhi)):
        # Range may still yield closest available partition, marked uncalibrated.
        pass
    for _ in range(max_evals):
        mid=float(np.sqrt(lo*hi))
        if mid==lo or mid==hi: break
        count=evaluate(mid)
        if count < target: lo=mid
        elif count > target: hi=mid
        else: break
    # Include whole cache: sometimes Leiden is not monotonic in resolution.
    best_res=min(cache, key=lambda r:(abs(cache[r][1]['n_clusters']-target),
        abs(np.log(max(cache[r][1]['n_clusters'],1)/target))))
    labels=cache[best_res][0]
    return labels, best_res, partition_stats(labels)


def summarize_values(values):
    x=np.asarray(values,dtype=float)
    return {'mean':float(x.mean()), 'std':float(x.std(ddof=1)) if len(x)>1 else 0.0,
      'min':float(x.min()), 'max':float(x.max()), 'n':int(len(x))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True)
    p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--sequences',type=Path,required=True)
    p.add_argument('--embeddings',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--behavior-k',type=int,default=50)
    p.add_argument('--targets',type=int,nargs='+',default=[200,400,800,1600])
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--density-tolerance',type=float,default=0.03)
    p.add_argument('--count-tolerance',type=float,default=0.10)
    p.add_argument('--resolution-low',type=float,default=1e-5)
    p.add_argument('--resolution-high',type=float,default=1000.0)
    p.add_argument('--calibration-evals',type=int,default=16)
    p.add_argument('--rebuild-graphs',action='store_true')
    args=p.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or len(args.seeds)<2: p.error('Need >=2 distinct seeds')
    if len(set(args.targets))!=len(args.targets) or min(args.targets)<2: p.error('Targets must be distinct >=2')
    if not 0<args.count_tolerance<1: p.error('count tolerance must be between 0 and 1')
    if args.resolution_high<=args.resolution_low or args.calibration_evals<2: p.error('Invalid calibration bounds')
    args.out.mkdir(parents=True,exist_ok=True)
    cache_dir=args.out/'graph_cache'; cache_dir.mkdir(exist_ok=True)
    part_dir=args.out/'partitions'; part_dir.mkdir(exist_ok=True)
    config=dict(dataset=args.dataset,mapping=str(args.mapping.resolve()),
        sequences=str(args.sequences.resolve()),embeddings=str(args.embeddings.resolve()),
        behavior_k=args.behavior_k,density_tolerance=args.density_tolerance,threads=args.threads,
        targets=args.targets,seeds=args.seeds,count_tolerance=args.count_tolerance,
        resolution_low=args.resolution_low,resolution_high=args.resolution_high,
        calibration_evals=args.calibration_evals,
        note='Clusters are calibrated to similar COUNT, not necessarily similar size distributions. '
            'Seeds change Leiden initialization, not graph construction or data splits. '
            'No diffusion. Bootstrap CIs not calculated.')
    config_path=args.out/'config.json'
    if config_path.exists() and not args.rebuild_graphs:
        old=json.loads(config_path.read_text())
        for k in ('dataset','mapping','sequences','embeddings','behavior_k','density_tolerance',
                  'resolution_low','resolution_high','calibration_evals','targets','seeds','count_tolerance'):
            if old.get(k)!=config[k]:
                raise ValueError(f'Existing output configuration mismatch at {k}. Use a NEW --out directory.')
    else:
        save_json(config_path,config)
    manifest_path=cache_dir/'manifest.json'
    if args.rebuild_graphs or not (manifest_path.exists() and all((cache_dir/f'{v}.npz').exists() for v in VIEWS)):
        print('Building B/T/V graphs (train only). No GPU.',flush=True)
        ids=load_mapping(args.mapping)
        x,degree,events=load_train_only(args.sequences,len(ids))
        embs=load_embeddings(args.embeddings,ids)
        selected=np.flatnonzero((degree>=1)&embs['text'][1]&embs['image'][1])
        bg,bn=behavior_graph(x,degree,selected,args.behavior_k,'union')
        ref_edges=bg.nnz//2
        tg,tn,tm=matched_semantic_graph(*embs['text'],selected,ref_edges,20,'union',args.threads,args.density_tolerance)
        vg,vn,vm=matched_semantic_graph(*embs['image'],selected,ref_edges,20,'union',args.threads,args.density_tolerance)
        graphs={'B':bg,'T':tg,'V':vg}
        for view,matrix in graphs.items():
            sparse.save_npz(cache_dir/f'{view}.npz',matrix,compressed=False)
        manifest=dict(total_items=len(ids),common_items=len(selected),train_events=events,users=x.shape[0],
            selected_item_ids_sha256=__import__('hashlib').sha256(np.asarray(selected,dtype=np.int64).tobytes()).hexdigest(),
            edge_counts={v:int(g.nnz//2) for v,g in graphs.items()},semantic_k={'T':tm,'V':vm},
            note='All graphs use the same selected item indexing and are immutable across seeds.')
        save_json(manifest_path,manifest)
        print('Graph counts:',manifest['edge_counts'],flush=True)
    else:
        manifest=json.loads(manifest_path.read_text())
        print('Reusing graph cache:',manifest['edge_counts'],flush=True)
    igs={view:to_igraph(sparse.load_npz(cache_dir/f'{view}.npz')) for view in VIEWS}
    n=manifest['common_items']
    cal_rows=[]; labels={}
    for view in VIEWS:
        for seed in args.seeds:
            resolution_cache={}
            for target in args.targets:
                base=f'{view}_k{target}_seed{seed}'
                label_path=part_dir/f'{base}.npy'
                meta_path=part_dir/f'{base}.json'
                if label_path.exists() and meta_path.exists():
                    lab=np.load(label_path)
                    record=json.loads(meta_path.read_text())
                    if len(lab)!=n: raise ValueError(f'Label length mismatch for {base}')
                else:
                    print(f'Calibrating {base}...',flush=True)
                    lab,res,stats=calibrate_count(igs[view],target,seed,resolution_cache,
                        args.resolution_low,args.resolution_high,args.calibration_evals)
                    err=abs(stats['clusters']-target)/target
                    record=dict(view=view,seed=seed,target_clusters=target,
                        resolution=float(res),relative_count_error=float(err),
                        calibrated=bool(err<=args.count_tolerance),**stats)
                    np.save(label_path,lab)
                    save_json(meta_path,record)
                    print(f"  clusters={stats['clusters']} median={stats['median_size']} "
                          f"resolution={res:.5g} calibrated={record['calibrated']}",flush=True)
                labels[(view,target,seed)]=lab
                cal_rows.append(record)
            del resolution_cache
    cal_fields=['view','seed','target_clusters','resolution','clusters','relative_count_error',
        'calibrated','median_size','q25_size','q75_size','singleton_fraction_clusters',
        'singleton_fraction_items','largest_cluster']
    dump_csv(args.out/'calibration.csv',cal_rows,cal_fields)
    meta={(r['view'],r['target_clusters'],r['seed']):r for r in cal_rows}
    cross=[]; within=[]
    for target in args.targets:
        print('Scoring agreement for target clusters:',target,flush=True)
        for seed in args.seeds:
            for a,b in PAIRS:
                ra,rb=meta[(a,target,seed)],meta[(b,target,seed)]
                calibrated=bool(ra['calibrated'] and rb['calibrated'])
                cross.append(dict(target_clusters=target,seed=seed,pair=f'{a}-{b}',
                    calibrated=calibrated,ami=float(adjusted_mutual_info_score(labels[a,target,seed],labels[b,target,seed])) if calibrated else '',
                    ari=float(adjusted_rand_score(labels[a,target,seed],labels[b,target,seed])) if calibrated else '',
                    a_clusters=ra['clusters'],b_clusters=rb['clusters']))
        for view in VIEWS:
            for s1,s2 in itertools.combinations(args.seeds,2):
                ra,rb=meta[(view,target,s1)],meta[(view,target,s2)]
                calibrated=bool(ra['calibrated'] and rb['calibrated'])
                within.append(dict(target_clusters=target,view=view,seed_a=s1,seed_b=s2,
                    calibrated=calibrated,ami=float(adjusted_mutual_info_score(labels[view,target,s1],labels[view,target,s2])) if calibrated else '',
                    ari=float(adjusted_rand_score(labels[view,target,s1],labels[view,target,s2])) if calibrated else ''))
    dump_csv(args.out/'cross_view.csv',cross,['target_clusters','seed','pair','calibrated','ami','ari','a_clusters','b_clusters'])
    dump_csv(args.out/'within_view.csv',within,['target_clusters','view','seed_a','seed_b','calibrated','ami','ari'])
    summary=[]
    for target in args.targets:
        for metric in ('ami','ari'):
            within_means={}
            for v in VIEWS:
                vals=[float(r[metric]) for r in within if r['target_clusters']==target and r['view']==v and r['calibrated']]
                within_means[v]=float(np.mean(vals)) if len(vals)==len(args.seeds)*(len(args.seeds)-1)//2 else None
                summary.append(dict(target_clusters=target,metric=metric,comparison=f'within_{v}',
                    **(summarize_values(vals) if vals else dict(mean='',std='',min='',max='',n=0)),
                    fully_calibrated=within_means[v] is not None))
            for a,b in PAIRS:
                vals=[float(r[metric]) for r in cross if r['target_clusters']==target and r['pair']==f'{a}-{b}' and r['calibrated']]
                full=len(vals)==len(args.seeds)
                # Mean within stability of the two views is the reference at this granularity.
                baseline=(within_means[a]+within_means[b])/2 if within_means[a] is not None and within_means[b] is not None else None
                summary.append(dict(target_clusters=target,metric=metric,comparison=f'cross_{a}-{b}',
                    **(summarize_values(vals) if vals else dict(mean='',std='',min='',max='',n=0)),
                    fully_calibrated=full,within_baseline=baseline if baseline is not None else '',
                    cross_minus_within=(float(np.mean(vals))-baseline) if full and baseline is not None else ''))
    dump_csv(args.out/'summary.csv',summary,
        ['target_clusters','metric','comparison','mean','std','min','max','n','fully_calibrated',
         'within_baseline','cross_minus_within'])
    save_json(args.out/'run_manifest.json',dict(config=config,graph_manifest=manifest,
        fully_calibrated=bool(all(r['calibrated'] for r in cal_rows)),
        interpretation='Compare cross-view decline with decline in each view’s within-view seed agreement. '
        'Do not claim depth-dependent semantic divergence solely from falling cross-view AMI. '
        'Cross_minus_within is descriptive, not an inferential test.'))
    print('SUCCESS:',args.out,flush=True)
    print('Fully calibrated:',all(r['calibrated'] for r in cal_rows),flush=True)

if __name__=='__main__': main()
