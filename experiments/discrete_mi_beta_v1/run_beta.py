#!/usr/bin/env python3
"""Exploratory, U2-only entropy-matched beta sweep for Exp2 level-1.
Reuses V2 Portable training/validation unchanged except common balance-weight beta.
Never selects using U3. Keeps previous results untouched.
"""
import argparse, csv, json, sys, os, copy, hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from scipy import sparse

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'/'discrete_mi_portable_v2'))
import run_level1_v2 as v2
import offdiag_core as core

METHODS=('rq','infonce','code_mi')
PAIRS=(('code_mi','infonce'),('code_mi','rq'),('infonce','rq'),('code_mi','kmeans'),('infonce','kmeans'))

def entropy(c, items, k):
    counts=np.bincount(c[items],minlength=k).astype(float)
    p=counts/max(counts.sum(),1)
    return float(-np.sum(p[p>0]*np.log(p[p>0])))

def writecsv(path,rows):
    if not rows: return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def user_graph(x_el,eligible,mult):
    deg=np.asarray(x_el.T@mult).ravel()
    cooc=(x_el.T@x_el.multiply(mult[:,None])).tocsr()
    cooc.setdiag(0);cooc.eliminate_zeros()
    u=sparse.triu(cooc,k=1,format='coo')
    i,j=u.row,u.col
    w=u.data.astype(float)/np.maximum(np.sqrt(deg[i]*deg[j]),1e-12)
    return eligible[i].astype(np.int32),eligible[j].astype(np.int32),w

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--domain',choices=['Industrial','Musical'],required=True)
    p.add_argument('--modality',choices=['text','image','both'],default='both')
    p.add_argument('--mode',choices=['check','smoke','run'],default='check')
    p.add_argument('--beta-grid',default='0.01,0.03,0.1,0.3,1.0')
    p.add_argument('--lambda-grid',default='0.01,0.1,1.0')
    p.add_argument('--tolerance-pct',type=float,default=1.0)
    p.add_argument('--out-root',default='outputs/discrete_mi_beta_match_v1')
    p.add_argument('--device',default='cuda')
    p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--nulls',type=int,default=100)
    p.add_argument('--mapping-path',default=None)
    p.add_argument('--embedding-path',default=None)
    p.add_argument('--sequence-path',default=None)
    args=p.parse_args()
    if args.tolerance_pct<=0:raise ValueError('tolerance must be positive')
    beta=tuple(float(x) for x in args.beta_grid.split(','))
    lambdas=tuple(float(x) for x in args.lambda_grid.split(','))
    if not beta or any(v<0 for v in beta) or not lambdas or any(v<0 for v in lambdas):raise ValueError('Invalid grids')
    if args.mode=='smoke':
        beta=(0.01,0.1);lambdas=(0.1,)
    if args.mode!='check' and args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: export CUDA_VISIBLE_DEVICES=1 and use --device cuda, or --device cpu')
    a=argparse.Namespace(domain=args.domain, mapping_path=args.mapping_path, embedding_path=args.embedding_path,sequence_path=args.sequence_path, seed=42,k=256,dim=64,steps=5 if args.mode=='smoke' else 1200,batch=64 if args.mode=='smoke' else 512,lr=0.001,temperature=0.2,soft_temperature=0.10,balance_weight=0.01,lambda_grid=','.join(map(str,lambdas)),eval_every=5 if args.mode=='smoke' else 200,eval_positives=200 if args.mode=='smoke' else 30000,bootstrap=20 if args.mode=='smoke' else args.bootstrap,nulls=2 if args.mode=='smoke' else args.nulls,swaps_per_edge=.01 if args.mode=='smoke' else 2.0,modality=args.modality,out_root=args.out_root,device=args.device,dry_run=args.mode=='check')
    print('BETA-MATCH PROTOCOL: U2 item entropy target and U2 transition AUC only; no U3 model selection',flush=True)
    print('BETA GRID:',beta,'LAMBDA GRID:',lambdas,'steps:',a.steps,flush=True)
    v2.seed_all(a.seed)
    dat=v2.make_data(a)
    if args.mode=='check':
        print('CHECK PASS: all input paths and required modules found');return
    dest=(ROOT/args.out_root/('smoke' if args.mode=='smoke' else 'full')/f'{args.domain}_K256_seed42').resolve()
    if dest.exists():raise FileExistsError('Refusing to overwrite existing output: '+str(dest))
    dest.mkdir(parents=True)
    (dest/'config.json').write_text(json.dumps(dict(vars(args),beta_grid=beta,lambda_grid=lambdas,steps=a.steps,batch=a.batch,seed=42,k=256,dim=64),indent=2)+'\n')
    e1,e2,e3=dat['edges'];n=dat['n']; eligible=dat['eligible'];device=torch.device(args.device)
    # ONLY U2 incident items determine U2 entropy; identical item population for all methods.
    u2_items=np.unique(np.r_[e2[0],e2[1]]).astype(np.int32)
    if len(u2_items)<256:raise RuntimeError('U2 item pool too small to match K=256')
    mods=('text','image') if args.modality=='both' else (args.modality,)
    summary=[];candidate_rows=[];picked={};matches={}
    for mod in mods:
        print('\n===',mod.upper(),'===',flush=True)
        z=core.norm(dat['emb'][mod][0]);train=dat['train_items']
        # Kmeans baseline from Exp1 if available, otherwise reproduce U1-only fitting as portable v2.
        path=ROOT/'outputs'/'offdiag_analysis_v1'/f'{args.domain}_CLIP_K256_seed42'/f'{mod}_codes.npy'
        if path.is_file():
            km_codes=np.load(path,allow_pickle=False).astype(np.int32)
            source='exp1_cached'
        else:
            print('Generating U1-only reference Kmeans (Exp1 equivalent)',flush=True)
            km=MiniBatchKMeans(n_clusters=256,random_state=42,batch_size=2048,n_init=5,max_iter=200)
            km.fit(z[train]);km_codes=km.predict(z).astype(np.int32)
            source='generated_u1'
        target_h=entropy(km_codes,u2_items,256)
        print('TARGET U2 KMeans H=',round(target_h,6),'U2 item count=',len(u2_items),'source=',source,flush=True)
        init=v2.init_codebook_kmeans(a,z,train,device)
        selected={}
        for method in METHODS:
            eligible_candidates=[]
            for b in beta:
                a.balance_weight=b
                for lam in ((0.,) if method=='rq' else lambdas):
                    model,logs=v2.train_candidate(a,mod,z,train,e1,method,lam,init,dest,device)
                    codes,h=v2.encode(model,z,device,a.soft_temperature)
                    h_u2=entropy(codes,u2_items,256)
                    devpct=100*abs(h_u2-target_h)/max(abs(target_h),1e-12)
                    alpha,auc,alpha_aucs=v2.measure_validation(codes,dat['u2'],e1,256)
                    row=dict(domain=args.domain,modality=mod,method=method,beta=b,lam=lam,alpha=alpha,u2_auc=auc,u2_h=h_u2,target_u2_kmeans_h=target_h,u2_entropy_gap_pct=devpct,matched_1pct=devpct<=args.tolerance_pct,u2_used_codes=int(np.unique(codes[u2_items]).size),last_task=logs[-1]['task'],last_batch_codes=logs[-1]['batch_hard_used'])
                    candidate_rows.append(row)
                    print('CANDIDATE',mod,method,'beta',b,'lam',lam,'U2 H',round(h_u2,4),'gap%',round(devpct,3),'U2 AUC',round(auc,5),'MATCH',row['matched_1pct'],flush=True)
                    # Keep winner among *matching* candidates; U2 AUC breaks ties, deterministic smaller beta/lambda.
                    if row['matched_1pct']:
                        rank=(auc,-devpct,-b,-lam)
                        if method not in selected or rank>selected[method][0]:
                            selected[method]=(rank,row,copy.deepcopy({key:value.detach().cpu() for key,value in model.state_dict().items()}),codes.copy(),h.copy())
                    del model
            if method not in selected:
                print('MATCH_FAILED',mod,method,': no candidate within prescribed U2 entropy tolerance; do NOT use U3 to tune',flush=True)
                matches[(mod,method)]=False
            else:
                matches[(mod,method)]=True
                print('U2 SELECTED',mod,method,selected[method][1],flush=True)
        moddir=dest/mod;moddir.mkdir()
        np.save(moddir/'kmeans_codes.npy',km_codes)
        # U3 evaluation is deferred until ALL U2 selections are frozen.
        for method,entry in selected.items():
            _,row,state,codes,continuous=entry
            np.save(moddir/f'{method}_codes.npy',codes)
            torch.save(dict(model=state,config=vars(a),method=method,selected_u2=row,modality=mod),moddir/f'{method}_selected.pt')
        picked[mod]=dict(kmeans=km_codes,**{m:entry[3] for m,entry in selected.items()})
        writecsv(moddir/'u2_candidate_sweep.csv',[r for r in candidate_rows if r['modality']==mod])
        (moddir/'selection.json').write_text(json.dumps(dict(target_u2_entropy=target_h,u2_reference_n_items=len(u2_items),kmeans_source=source,selected={m:selected[m][1] for m in selected},unmatched=[m for m in METHODS if m not in selected]),indent=2)+'\n')
        # Evaluate on U3 even if a modality has nonmatching methods? For strict comparison require ALL methods match.
        if len(selected)!=len(METHODS):
            print('U3 EVALUATION SKIPPED for',mod,': at least one method unmatched. Preserve diagnostic selection only.',flush=True)
            continue
        for method in (*METHODS,'kmeans'):
            if method=='kmeans':
                c=km_codes;h=z;lam=0;bb=0
                alpha,u2auc,_=v2.measure_validation(c,dat['u2'],e1,256)
            else:
                _,sel,_,c,h=selected[method]
                alpha,u2auc,lam,bb=sel['alpha'],sel['u2_auc'],sel['lam'],sel['beta']
            metrics=v2.code_metrics(c,e1,e3,dat['u2'],dat['u3'],alpha,256,z,h,n,a.bootstrap,42+400+len(summary),eligible)
            P=core.code_joint(c,*e3[:3],256)
            p=P.sum(1);edge_h=float(-np.sum(p[p>0]*np.log(p[p>0])))
            metrics.update(dict(J_over_edge_H=metrics['code_mi']/edge_h if edge_h>0 else float('nan'),H_edge=edge_h,J_over_item_H=metrics['code_mi']/max(metrics['code_entropy'],1e-12)))
            summary.append(dict(domain=args.domain,modality=mod,method=method,beta=bb,lam=lam,alpha=alpha,u2_auc=u2auc,u2_entropy_target=target_h,u2_entropy=entropy(c,u2_items,256),matched_1pct=(method=='kmeans' or matches.get((mod,method),False)),**metrics))
        print('U3 EVALUATED (all methods matched) ',mod,flush=True)
    writecsv(dest/'u2_all_candidates.csv',candidate_rows)
    writecsv(dest/'summary.csv',summary)
    # Null comparisons share rewired U3 graph; avoid old item-based Δoff CI.
    if summary:
        nulls,_=v2.make_null(e3,a.nulls,a.swaps_per_edge,42)
        for row in summary:
            c=picked[row['modality']][row['method']]
            vals=np.array([core.measures(core.code_joint(c,aa,bb,e3[2],256))['i_off'] for aa,bb in nulls])
            row['delta_off']=row['i_off']-float(vals.mean())
            row['offdiag_p_null']=(1+int(np.sum(vals>=row['i_off'])))/(1+len(vals))
        writecsv(dest/'summary.csv',summary)
        # Paired user-bootstrap, rebuild W_B for *every* resample, same users for all models.
        dataset,seqname=core.DOMAINS[args.domain]
        seq=v2.resolve_input('sequence',dataset,seqname,args.sequence_path)
        ids=core.load_mapping(v2.resolve_input('mapping',dataset,seqname,args.mapping_path))
        _,seqs=core.read_users_train(seq,len(ids))
        groups,_=core.split_stratified(seqs,42)
        x=core.csr_users([seqs[int(i)] for i in groups[2]],len(ids)).tocsr()[:,eligible].astype(np.float64)
        base_a,base_b,base_w=user_graph(x,eligible,np.ones(x.shape[0],dtype=float))
        k1=core.keypair(e3[0],e3[1],n);k2=core.keypair(base_a,base_b,n)
        order1=np.argsort(k1);order2=np.argsort(k2)
        if not np.array_equal(k1[order1],k2[order2]) or not np.allclose(e3[2][order1],base_w[order2],atol=1e-12,rtol=1e-9):raise RuntimeError('IDENTITY FAILED: U3 graph reconstruction mismatch')
        print('IDENTITY PASS: U3 user graph and W_B reproduced',flush=True)
        rng=np.random.default_rng(20261111)
        draws={key:np.zeros(a.bootstrap) for mod in mods if len(picked[mod])==4 for key in [(mod,l,r) for l,r in PAIRS]}
        for t in range(a.bootstrap):
            ix=rng.integers(x.shape[0],size=x.shape[0]);mult=np.bincount(ix,minlength=x.shape[0]).astype(float)
            aa,bb,ww=user_graph(x,eligible,mult)
            J={key:core.measures(core.code_joint(c,aa,bb,ww,256))['mi_real'] for mod in mods if len(picked[mod])==4 for key,c in [( (mod,m),codes) for m,codes in picked[mod].items()]}
            for mod,l,r in draws:draws[(mod,l,r)][t]=J[(mod,l)]-J[(mod,r)]
            if (t+1)%25==0:print('USER BOOTSTRAP',t+1,'/',a.bootstrap,flush=True)
        rows=[]
        obs={(r['modality'],r['method']):r['code_mi'] for r in summary}
        for (mod,l,r),d in draws.items():
            point=obs[(mod,l)]-obs[(mod,r)];mean=d.mean();lo,hi=np.quantile(d,[.025,.975])
            rows.append(dict(domain=args.domain,modality=mod,lhs=l,rhs=r,J_diff=point,bootstrap_mean=mean,bootstrap_bias=mean-point,percentile_lo=lo,percentile_hi=hi,centered_lo=point+lo-mean,centered_hi=point+hi-mean,centered_excludes_zero=bool(point+lo-mean>0 or point+hi-mean<0)))
        writecsv(dest/'user_bootstrap_pairs.csv',rows)
        np.savez_compressed(dest/'user_bootstrap_draws.npz',**{f'{mod}_{l}_minus_{r}':v for (mod,l,r),v in draws.items()})
    manifest=dict(protocol='EXP2_BETA_MATCH_V1_EXPLORATORY',domain=args.domain,seed=42,codebook_k=256,selection_only='U2',u2_item_pool='unique endpoints of U2 cooccurrence edges, CLIP eligible',beta_grid=beta,lambda_grid=lambdas,tolerance_pct=args.tolerance_pct,all_methods_matched_by_modality={mod:all(matches.get((mod,m),False) for m in METHODS) for mod in mods},u3_evaluated_modalities=sorted({r['modality'] for r in summary}),resampling='U3 user bootstrap; rebuilt weighted W_B, fixed tokenizer',caveats=['beta sweep is exploratory follow-up not original preregistration','matches only code entropy on U2 eligible cooccurrence endpoints, not code semantic structure','K-means U1-only codes reused if available; else refitted','J is computed on U3 weighted edges; normalized edge J uses that edge marginal entropy','U3 never used to pick beta/lambda/alpha','no U3 results when matching fails; report MATCH_FAILED'],out_root=str(dest))
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (dest/'DONE.json').write_text(json.dumps(dict(status='success',sweep_complete=True,matched=manifest['all_methods_matched_by_modality'],rows=len(summary)))+'\n')
    print('SUCCESS',dest,'summary rows',len(summary),'matched',manifest['all_methods_matched_by_modality'],flush=True)

if __name__=='__main__':main()
