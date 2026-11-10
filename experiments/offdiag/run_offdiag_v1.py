#!/usr/bin/env python3
"""Off-diagonal collaborative code structure and held-out PMI-prediction experiment.
Independent of training objectives. Existing GLI loaders and split function are reused.
"""
import argparse, csv, json, math, os, sys, hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'/'motivation'))
from build_utility_candidates import split_stratified
from run_c1_pilot import read_users_train, csr_users
from run_motivation_cpu_v2 import load_mapping, load_embeddings

DOMAINS={
 'Industrial':('Industrial_and_Scientific','Industrial_5core_v2.txt'),
 'Musical':('Musical_Instruments','Musical_5core_v2.txt'),
}

def args_parse():
 p=argparse.ArgumentParser()
 p.add_argument('--domain',choices=DOMAINS,required=True)
 p.add_argument('--seed',type=int,default=42)
 p.add_argument('--k',type=int,default=256)
 p.add_argument('--nulls',type=int,default=100)
 p.add_argument('--bootstraps',type=int,default=500)
 p.add_argument('--max-positive',type=int,default=30000)
 p.add_argument('--swaps-per-edge',type=float,default=2.)
 p.add_argument('--out-root',default='outputs/offdiag_analysis_v1')
 p.add_argument('--dry-run',action='store_true')
 p.add_argument('--force',action='store_true')
 return p.parse_args()

def norm(z):
 z=np.nan_to_num(np.asarray(z,dtype=np.float32).copy(),nan=0.,posinf=0.,neginf=0.)
 z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-10)
 return z

def edges_from_users(x,eligible):
 # Binary user/item CSR from repository; no duplicate user-item purchases.
 m=(x[:,eligible].T@x[:,eligible]).tocsr()
 m.setdiag(0);m.eliminate_zeros()
 tr=sparse.triu(m,k=1,format='coo')
 a=eligible[tr.row].astype(np.int32)
 b=eligible[tr.col].astype(np.int32)
 count=tr.data.astype(np.float64)
 deg=np.asarray(x.sum(axis=0)).ravel().astype(np.float64)
 # W_B definition frozen for THIS experiment: normalized co-occurrence (cosine on user incidence).
 w=count/np.maximum(np.sqrt(deg[a]*deg[b]),1e-12)
 return a,b,w,deg

def keypair(a,b,n):
 return np.minimum(a,b).astype(np.int64)*n+np.maximum(a,b).astype(np.int64)

def code_joint(c,a,b,w,k):
 ab=np.bincount((c[a]*k+c[b]),weights=w,minlength=k*k).reshape(k,k)
 ab+=np.bincount((c[b]*k+c[a]),weights=w,minlength=k*k).reshape(k,k)
 ab=ab/np.maximum(ab.sum(),1e-30)
 return ab

def measures(P):
 m=P.sum(1)
 prod=np.outer(m,m)
 mask=P>0
 terms=np.zeros_like(P)
 terms[mask]=P[mask]*np.log(P[mask]/np.maximum(prod[mask],1e-30))
 diag=float(np.trace(terms));total=float(terms.sum());off=total-diag
 mass=float(np.trace(P))
 # Signed contributions, not guaranteed non-negative individually.
 return dict(diagonal_mass=mass,offdiag_mass=1-mass,mi_real=total,i_diag=diag,i_off=off)

def swap_graph(a0,b0,rng,n_swaps):
 # Valid simple-graph double edge swaps: undirected degree sequence exactly preserved.
 a=a0.copy(); b=b0.copy(); E=len(a)
 keys=set(keypair(a,b,int(max(a.max(initial=0),b.max(initial=0))+1)).tolist())
 n=int(max(a.max(initial=0),b.max(initial=0))+1)
 attempts=0;done=0;limit=max(n_swaps*30,300)
 while done<n_swaps and attempts<limit:
  attempts+=1
  i,j=rng.integers(0,E,size=2)
  if i==j:continue
  u,v=int(a[i]),int(b[i]); x,y=int(a[j]),int(b[j])
  if len({u,v,x,y})<4:continue
  if rng.random()<.5: p,q=u,y; r,s=x,v
  else:p,q=u,x;r,s=v,y
  p,q=sorted((p,q));r,s=sorted((r,s))
  old1=u*n+v;old2=x*n+y
  new1=p*n+q;new2=r*n+s
  if new1==new2 or new1 in keys or new2 in keys:continue
  keys.remove(old1);keys.remove(old2);keys.add(new1);keys.add(new2)
  a[i],b[i]=p,q;a[j],b[j]=r,s
  done+=1
 return a,b,done,attempts

def sample_negatives(pa,pb,excluded,degree,bins,n_items,eligible,seed,ratio=4):
 rng=np.random.default_rng(seed)
 # Each endpoint is sampled in its original endpoint's U1 degree-stratum.
 strata=np.searchsorted(bins,degree,side='right')
 pools={int(v):eligible[strata[eligible]==v] for v in np.unique(strata[eligible])}
 aa=[];bb=[];keep=[]
 for idx,(u,v) in enumerate(zip(pa,pb)):
  out=[]; seen=set()
  for _ in range(150):
   if len(out)>=ratio:break
   x=int(rng.choice(pools[int(strata[u])]))
   y=int(rng.choice(pools[int(strata[v])]))
   if x==y:continue
   key=int(min(x,y))*n_items+int(max(x,y))
   if key in excluded or key in seen:continue
   seen.add(key);out.append((x,y))
  if len(out)<ratio:continue
  keep.append(idx)
  for x,y in out:aa.append(x);bb.append(y)
 return np.asarray(keep,dtype=np.int32),np.asarray(aa,dtype=np.int32),np.asarray(bb,dtype=np.int32)

def score_pmi(c,a,b,joint):
 k=joint.shape[0];m=joint.sum(1)
 pmi=np.log(joint/np.maximum(np.outer(m,m),1e-30))
 score=pmi[c[a],c[b]]
 diag=np.where(c[a]==c[b],score,0.)
 same=(c[a]==c[b]).astype(float)
 return same,diag,score

def auc(y,s):return float(roc_auc_score(y,s))
def auc_ci_item(y,s1,s2,a,b,n_items,seed,reps):
 rng=np.random.default_rng(seed)
 res=[]
 for _ in range(reps):
  # Poisson item cluster bootstrap: endpoint multiplicities induce pair weights.
  w=rng.poisson(1,size=n_items)
  weights=w[a]*w[b]
  active=weights>0
  if np.unique(y[active]).size!=2:continue
  res.append(float(roc_auc_score(y[active],s1[active],sample_weight=weights[active])-roc_auc_score(y[active],s2[active],sample_weight=weights[active])))
 return [float(x) for x in np.quantile(res,[.025,.975])] if len(res)>10 else [None,None]

def run():
 a=args_parse();root=ROOT
 dataset,seqfile=DOMAINS[a.domain]
 mapping=root/'shared_data'/dataset/'5core_v2'/'item_mapping.tsv'
 embedding=root/'artifacts'/'embeddings'/dataset/'full'/'clip_vit_b32.npz'
 sequences=Path.home()/'datasets'/'baseline_sequences'/seqfile
 out=root/a.out_root/f'{a.domain}_CLIP_K{a.k}_seed{a.seed}'
 required=[mapping,embedding,sequences]
 print('PROJECT:',root,flush=True)
 for f in required:print('INPUT',f,'exists=',f.is_file(),flush=True)
 if not all(x.is_file() for x in required):raise FileNotFoundError('One or more data files missing')
 if a.dry_run:return
 if a.nulls<2 or a.bootstraps<20:raise ValueError('Use >=2 nulls and >=20 bootstraps')
 if out.exists() and (out/'metrics.json').exists() and not a.force:
  raise FileExistsError(f'Existing completed result: {out}; refusing overwrite')
 if out.exists() and not a.force:raise FileExistsError(f'Output directory already exists: {out}')
 out.mkdir(parents=True,exist_ok=True)
 print('Loading data',flush=True)
 ids=load_mapping(mapping);n=len(ids)
 _,seqs=read_users_train(sequences,n)
 groups,strata=split_stratified(seqs,a.seed)
 xs=[csr_users([seqs[int(i)] for i in grp],n) for grp in groups]
 emb=load_embeddings(embedding,ids)
 # For exact Utility v2 consistency, both CLIP modalities must be present.
 valid=emb['text'][1]&emb['image'][1]
 eligible=np.flatnonzero(valid).astype(np.int32)
 if eligible.size<a.k:raise ValueError('Not enough embeddings for K')
 edge=[]
 for i in range(3):
  print('Building U'+str(i+1)+' co-occurrence edges',flush=True)
  edge.append(edges_from_users(xs[i],eligible))
 e1,e2,e3=edge
 if any(len(e[0])==0 for e in edge):raise ValueError('Empty edge set')
 codes={};vec={}
 train_eligible=eligible[np.asarray(xs[0][:,eligible].sum(axis=0)).ravel()>0]
 for j,mod in enumerate(('text','image')):
  z=norm(emb[mod][0]);vec[mod]=z
  model=MiniBatchKMeans(n_clusters=a.k,random_state=a.seed,batch_size=2048,n_init=5,max_iter=200)
  model.fit(z[train_eligible])
  codes[mod]=model.predict(z).astype(np.int32)
  np.save(out/f'{mod}_codes.npy',codes[mod]);np.save(out/f'{mod}_centers.npy',model.cluster_centers_)
  print(mod,'Kmeans fitted on',len(train_eligible),'U1 items; all codes assigned',flush=True)
 # Rewiring: item degree preserved; W_B edge weights carried with the edge slot (node strength not preserved).
 rng=np.random.default_rng(a.seed+10000)
 null_edges=[];swap_rates=[]
 basea,baseb,basew,_=e3
 nsw=max(1,int(a.swaps_per_edge*len(basea)))
 # Incremental MCMC chain; a short burn-in to reduce similarity to original.
 xa,xb=basea.copy(),baseb.copy()
 for i in range(a.nulls):
  xa,xb,done,attempts=swap_graph(xa,xb,rng,nsw)
  null_edges.append((xa.copy(),xb.copy()))
  swap_rates.append(done/max(nsw,1))
  if (i+1)%10==0:print('null',i+1,'/',a.nulls,'swap acceptance completion',round(swap_rates[-1],4),flush=True)
 # Prediction pair pool: U3 positive absent from both U1 and U2, U2 positive absent from U1.
 keys1=set(keypair(e1[0],e1[1],n).tolist());keys2=set(keypair(e2[0],e2[1],n).tolist());keys3=set(keypair(e3[0],e3[1],n).tolist())
 allkeys=keys1|keys2|keys3
 # Degree strata fixed from U1 item user incidence; zero-degree has dedicated bin.
 d1=e1[3]
 posdeg=d1[d1>0]
 cuts=np.unique(np.quantile(posdeg,[.2,.4,.6,.8]))
 bins=np.r_[0,cuts]
 def cohort(e,forbidden,label,offset):
  ka=keypair(e[0],e[1],n)
  sel=np.flatnonzero(~np.isin(ka,np.fromiter(forbidden,dtype=np.int64),assume_unique=False))
  rngs=np.random.default_rng(a.seed+offset)
  if len(sel)>a.max_positive:sel=rngs.choice(sel,size=a.max_positive,replace=False)
  pp=e[0][sel];qq=e[1][sel]
  keep,na,nb=sample_negatives(pp,qq,allkeys,d1,bins,n,eligible,a.seed+offset+100)
  pp=pp[keep];qq=qq[keep]
  if len(pp)<100:raise ValueError(f'{label}: only {len(pp)} positive matches, cannot evaluate')
  aa=np.r_[pp,na];bb=np.r_[qq,nb]; y=np.r_[np.ones(len(pp),dtype=np.uint8),np.zeros(len(na),dtype=np.uint8)]
  print(label,'positive=',len(pp),'negative=',len(na),flush=True)
  return aa,bb,y
 u2=cohort(e2,keys1,'U2',200)
 u3=cohort(e3,keys1|keys2,'U3',300)
 rows=[];rngboot=np.random.default_rng(a.seed+20000)
 for mod in ('text','image'):
  c=codes[mod];print('Analyzing',mod,flush=True)
  P=code_joint(c,*e3[:3],a.k);mm=measures(P)
  P1=code_joint(c,*e1[:3],a.k)
  null_met=[]
  for na,nb in null_edges:
   pm=code_joint(c,na,nb,basew,a.k)
   null_met.append(measures(pm))
  nm_off=np.array([x['i_off'] for x in null_met]);nm_diag=np.array([x['i_diag'] for x in null_met]);nm_mi=np.array([x['mi_real'] for x in null_met])
  delta_off=mm['i_off']-float(nm_off.mean());delta_diag=mm['i_diag']-float(nm_diag.mean())
  den=delta_off+delta_diag
  share=float(delta_off/den) if abs(den)>1e-8 else None
  # One-sided Monte Carlo p-value for real off statistic vs rewiring distribution.
  p_null=(1+int(np.sum(nm_off>=mm['i_off'])))/(1+a.nulls)
  boot_off=[]
  for j in range(a.bootstraps):
   iw=rngboot.poisson(1,size=n)
   rw=basew*iw[basea]*iw[baseb]
   if rw.sum()==0:continue
   rm=measures(code_joint(c,basea,baseb,rw,a.k))
   na,nb=null_edges[int(rngboot.integers(a.nulls))]
   nw=basew*iw[na]*iw[nb]
   if nw.sum()==0:continue
   nullm=measures(code_joint(c,na,nb,nw,a.k))
   boot_off.append(rm['i_off']-nullm['i_off'])
  ci_off=[float(v) for v in np.quantile(boot_off,[.025,.975])] if len(boot_off)>10 else [None,None]
  def predict(dataset,alpha):
   ea,eb,y=dataset
   # Add-alpha is applied to weighted co-occurrence MASS in units of edge weight.
   matrix=code_joint(c,*e1[:3],a.k)*float(np.sum(e1[2]))
   matrix=matrix+alpha
   matrix/=matrix.sum()
   ss,dd,ff=score_pmi(c,ea,eb,matrix)
   cos=np.einsum('ij,ij->i',vec[mod][ea],vec[mod][eb],optimize=True)
   return dict(same=ss,diagonal=dd,full=ff,cosine=cos)
  # Frozen alpha menu chosen ONLY by U2 full-PMI AUC.
  val={}
  for alpha in (.1,1.,10.):
   val[str(alpha)]=auc(u2[2],predict(u2,alpha)['full'])
  alpha=max((.1,1.,10.),key=lambda x:(val[str(x)],-x))
  pred=predict(u3,alpha);y=u3[2]
  aucs={k:auc(y,v) for k,v in pred.items()}
  aps={k:float(average_precision_score(y,v)) for k,v in pred.items()}
  gain=aucs['full']-aucs['diagonal']
  gain_ci=auc_ci_item(y,pred['full'],pred['diagonal'],u3[0],u3[1],n,a.seed+400+(['text','image'].index(mod)),a.bootstraps)
  item=dict(domain=a.domain,modality=mod,k=a.k,seed=a.seed,n_items=n,n_u3_edges=len(basea),
   n_u2_eval_pos=int(u2[2].sum()),n_u3_eval_pos=int(y.sum()),n_u3_eval_neg=int((1-y).sum()),
   **mm,mi_null=float(nm_mi.mean()),i_off_null_mean=float(nm_off.mean()),i_diag_null_mean=float(nm_diag.mean()),
   delta_off=delta_off,delta_diag=delta_diag,offdiag_delta_share=share,offdiag_p_null=p_null,delta_off_ci_low=ci_off[0],delta_off_ci_high=ci_off[1],
   alpha_selected=alpha,u2_alpha_auc=val,
   same_code_auc=aucs['same'],diagonal_pmi_auc=aucs['diagonal'],transition_auc=aucs['full'],cosine_auc=aucs['cosine'],
   delta_auc=gain,delta_auc_ci_low=gain_ci[0],delta_auc_ci_high=gain_ci[1],
   same_code_ap=aps['same'],diagonal_pmi_ap=aps['diagonal'],transition_ap=aps['full'],cosine_ap=aps['cosine'],
   go=bool(p_null<.05 and ci_off[0] is not None and ci_off[0]>0 and gain_ci[0] is not None and gain_ci[0]>0))
  (out/f'{mod}_metrics.json').write_text(json.dumps(item,indent=2,ensure_ascii=False)+'\n')
  np.savez_compressed(out/f'{mod}_code_pair_matrix.npz',p_real=P,p_null_mean=np.mean([code_joint(c,na,nb,basew,a.k) for na,nb in null_edges],axis=0))
  rows.append(item)
 manifest=dict(protocol='offdiag_v1',domain=a.domain,dataset=dataset,split_seed=a.seed,code_seed=a.seed,
  k=a.k,nulls=a.nulls,bootstraps=a.bootstraps,max_positive=a.max_positive,swaps_per_edge=a.swaps_per_edge,
  u1_u2_u3_users=[len(g) for g in groups],user_activity_strata=strata,
  code_training='U1 active items, common valid Text/Image CLIP',
  behavior_weight='W_B = cooccurrence_count / sqrt(item_user_degree_i * item_user_degree_j)',
  graph='U1 U2 U3 all nonzero unordered user co-occurrence pairs, train prefix only',
  null='simple undirected binary degree-preserving edge swaps; original edge-slot weights carried over (weighted node strength not preserved)',
  null_caveat='for exact weighted-strength preserving null a separate null analysis is required',
  bootstrap='Poisson item cluster bootstrap with independent random null draw per replicate',
  new_positive='U2 absent in U1; U3 absent in U1 and U2',
  negative='4 per positive, endpoint degree strata from U1; reject all U1/U2/U3 observed edges',
  selection='alpha in [0.1,1,10] by U2 full-PMI AUC only',
  baseline='same-code; diagonal-PMI (zero off-diagonal); full-PMI; CLIP cosine',
  p_null='one-sided Monte Carlo rank p=(1+null>=real)/(1+n_null)',
  note='Offdiag MI contributions are SIGNED. delta_off/(delta_off+delta_diag) is not necessarily a probability or [0,1] share.',
  mean_swap_completion=float(np.mean(swap_rates)),
  outputs=[f'{mod}_metrics.json' for mod in ('text','image')])
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
 with (out/'summary.csv').open('w',newline='') as f:
  cols=[x for x in rows[0] if x!='u2_alpha_auc'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows([{k:v for k,v in row.items() if k in cols} for row in rows])
 print('SUCCESS',out,flush=True)
 for row in rows:print(row['modality'],'diag=',row['diagonal_mass'],'delta_off=',row['delta_off'],'p=',row['offdiag_p_null'],'dAUC=',row['delta_auc'],'AUC_CI=',(row['delta_auc_ci_low'],row['delta_auc_ci_high']),'GO=',row['go'],flush=True)
if __name__=='__main__':run()
