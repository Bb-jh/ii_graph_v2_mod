#!/usr/bin/env python3
"""Build new modality-utility candidates. Uses train prefix only, activity-stratified U1/U2/U3."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy import sparse
from run_c1_pilot import read_users_train,csr_users
from run_motivation_cpu_v2 import load_mapping,load_embeddings,behavior_graph

def split_stratified(seqs,seed,frac=(.5,.25,.25)):
    rng=np.random.default_rng(seed)
    activity=np.array([len(s) for s in seqs]); quantiles=np.unique(np.quantile(activity,[0,.2,.4,.6,.8,1]))
    bands=np.searchsorted(quantiles[1:-1],activity,side='right')
    groups=[[],[],[]]
    for band in np.unique(bands):
        ix=np.flatnonzero(bands==band);rng.shuffle(ix)
        n1=round(frac[0]*len(ix));n2=round(frac[1]*len(ix))
        for out,part in zip(groups,(ix[:n1],ix[n1:n1+n2],ix[n1+n2:])):out.extend(part)
    return [np.array(g,dtype=np.int64) for g in groups], dict(activity_quantiles=quantiles.tolist(), activity_counts=np.bincount(bands).tolist())

def eval_edges(x,a,b):
    deg=np.asarray(x.sum(axis=0)).ravel().astype(np.int64)
    co=x.T@x
    cnt=np.asarray(co[a,b]).ravel().astype(np.float64)
    return (cnt>0).astype(np.uint8),cnt,deg

def main():
    p=argparse.ArgumentParser()
    for name in ('mapping','sequences','embeddings','out','dataset'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--k',type=int,default=20)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    ids=load_mapping(Path(args.mapping)); _,seqs=read_users_train(Path(args.sequences),len(ids))
    groups,strata=split_stratified(seqs,args.seed)
    xs=[csr_users([seqs[int(i)] for i in group],len(ids)) for group in groups]
    emb=load_embeddings(Path(args.embeddings),ids)
    deg_all=np.asarray(xs[0].sum(axis=0)).ravel().astype(np.int64)
    valid=emb['text'][1]&emb['image'][1]&(deg_all>0)
    selected=np.flatnonzero(valid)
    if len(selected)<100:raise ValueError('Too few eligible items')
    # symmetrized union of per-item top-k, exactly same rule across datasets
    graph,_=behavior_graph(xs[0],deg_all,selected,args.k,'union')
    edge=sparse.triu(graph,k=1,format='coo'); a=edge.row.astype(np.int32); b=edge.col.astype(np.int32)
    wb=edge.data.astype(np.float32); n=len(a)
    if not n:raise ValueError('No candidate edges')
    x1=xs[0][:,selected]
    co=x1.T@x1
    co1=np.asarray(co[a,b]).ravel().astype(float)
    degrees=deg_all[selected]
    feats=np.column_stack((np.log1p(wb),np.log1p(np.minimum(degrees[a],degrees[b])),np.log1p(np.maximum(degrees[a],degrees[b])))).astype(np.float32)
    sim={}
    for name,field in (('st','text'),('sv','image')):
        z=np.ascontiguousarray(emb[field][0][selected],dtype=np.float32)
        z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
        sim[name]=np.einsum('ij,ij->i',z[a],z[b],optimize=True).astype(np.float32)
    y2,c2,d2=eval_edges(xs[1][:,selected],a,b)
    y3,c3,d3=eval_edges(xs[2][:,selected],a,b)
    np.savez_compressed(out/'utility_candidates.npz',a=a,b=b,original_item_index=selected,
        wb=wb,B=feats,st=sim['st'],sv=sim['sv'],y2=y2,y3=y3,
        co2=c2.astype(np.float32),co3=c3.astype(np.float32),deg1=degrees,deg2=d2,deg3=d3)
    manifest=dict(dataset=args.dataset,seed=args.seed,behavior_topk=args.k,graph_mode='union',
        edges=n,n_selected_items=len(selected),users=[len(g) for g in groups],
        y2_prevalence=float(y2.mean()),y3_prevalence=float(y3.mean()),
        features=['log1p_wb','log1p_min_U1_degree','log1p_max_U1_degree'],
        user_activity_stratification=strata,split_mode='train_prefix_users',
        common_valid_embeddings_only=True, no_leiden_parent_filter=True)
    (out/'candidate_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='user_activity_stratification'},indent=2),flush=True)
if __name__=='__main__':main()
