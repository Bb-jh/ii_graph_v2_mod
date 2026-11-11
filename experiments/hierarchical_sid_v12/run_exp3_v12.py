#!/usr/bin/env python3
"""Exp3 V1.2: U1 graph hierarchy + controlled residual discrete tokenizers.

This is a level-1..3 tokenizer ablation, NOT a complete generative recommender.
A = hierarchical supervised contrastive, B = hierarchical hard-code label CE (F1),
C = hard-code conditional variational predictor at each depth (F2), D = continuous InfoNCE,
E = U1-content KMeans residual codes. All learned arms share architecture, U1 training,
U2 tuning with frozen entropy tolerance, and U3-only final evaluation.
"""
import argparse, copy, csv, hashlib, json, os, random, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.naive_bayes import MultinomialNB
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'/'discrete_mi_portable_v2'))
import run_level1_v2 as v2
import offdiag_core as core

ARMS=('rq','contrastive','label_mi','pair_code_mi','infonce','kmeans')

def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--domain',choices=['Industrial','Musical'],required=False)
 p.add_argument('--mode',choices=['check','smoke','diagnose','run','self-test'],default='check')
 p.add_argument('--modality',choices=['text','image','both'],default='both')
 p.add_argument('--seed',type=int,default=42)
 p.add_argument('--mapping-path');p.add_argument('--embedding-path');p.add_argument('--sequence-path')
 p.add_argument('--out-root',default='outputs/hierarchical_sid_exp3_v12')
 p.add_argument('--device',default='cuda')
 p.add_argument('--depth',type=int,default=3)
 p.add_argument('--k',type=int,default=256)
 p.add_argument('--dim',type=int,default=64)
 p.add_argument('--steps',type=int,default=1200)
 p.add_argument('--batch',type=int,default=256)
 p.add_argument('--lr',type=float,default=0.001)
 p.add_argument('--temperature',type=float,default=.2)
 p.add_argument('--soft-temperature',type=float,default=.1)
 p.add_argument('--lambda-grid',default='.1,1.0')
 p.add_argument('--beta-grid',default='1,3,10',help='Fixed target entropy penalty candidates; selected on U2 only')
 p.add_argument('--entropy-warmup',type=int,default=50,help='Steps to estimate hard-soft gap before targeting')
 p.add_argument('--entropy-ema',type=float,default=0.9)
 p.add_argument('--evaluate-unmatched',action='store_true',help='Exploratory U3 metrics for unmatched best-U2 candidate; opens U3, NOT confirmatory')
 p.add_argument('--save-checkpoints',action=argparse.BooleanOptionalAction,default=True)
 p.add_argument('--tolerance-pct',type=float,default=1.)
 p.add_argument('--eval-positives',type=int,default=30000)
 p.add_argument('--bootstrap',type=int,default=500)
 p.add_argument('--hierarchy-branches',default='16,4,4',help='Level 1 groups; children per parent at level 2 and 3')
 p.add_argument('--force-rebuild-labels',action='store_true')
 p.add_argument('--max-candidates',type=int,default=None,help='Development-only cap per method; never for confirmatory run')
 return p.parse_args()

def entropy(c,items,k):
 h=np.bincount(c[items],minlength=k).astype(float)
 prob=h/max(h.sum(),1.)
 return float(-np.sum(prob[prob>0]*np.log(prob[prob>0])))

def save_csv(path,rows):
 if not rows:return
 fields=list(dict.fromkeys(key for row in rows for key in row.keys()))
 with open(path,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def make_hierarchy(e1,train_items,n,branches,dim,seed):
 """U1 ONLY sparse collaborative graph -> diffusion sketch -> nested clusters.
 Labels -1 for any item not observed in U1. Parent nesting is exact.
 """
 src,dst,wt,_=e1
 if len(src)==0:raise ValueError('No U1 graph edges')
 ix=np.full(n,-1,dtype=np.int32);ix[train_items]=np.arange(len(train_items))
 a,b=ix[src],ix[dst]
 good=(a>=0)&(b>=0)
 a=a[good];b=b[good];w=wt[good].astype(np.float32)
 nnodes=len(train_items)
 mat=sparse.coo_matrix((np.r_[w,w],(np.r_[a,b],np.r_[b,a])),shape=(nnodes,nnodes)).tocsr()
 mat.setdiag(0);mat.eliminate_zeros()
 deg=np.asarray(mat.sum(1)).ravel();inv=np.maximum(deg,1e-7)**-.5
 mat=sparse.diags(inv)@mat@sparse.diags(inv)
 rng=np.random.default_rng(seed+311)
 features=rng.normal(size=(nnodes,dim)).astype(np.float32)
 # Random sketch of 1-hop and 2-hop graph structure; scalable, deterministic.
 features=mat@features
 features=features+mat@features
 features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-8)
 groups=np.zeros((nnodes,len(branches)),dtype=np.int32)
 parent_groups={0:np.arange(nnodes)}
 sizes=[]
 for level,branch in enumerate(branches):
  next_groups={};next_id=0
  for par,idx in sorted(parent_groups.items()):
   if len(idx)<2:labels=np.zeros(len(idx),dtype=np.int32);subcount=1
   else:
    subcount=min(branch,len(idx))
    km=MiniBatchKMeans(n_clusters=subcount,random_state=seed+level*1000+par,n_init=3,batch_size=min(1024,len(idx)),max_iter=100)
    labels=km.fit_predict(features[idx]).astype(np.int32)
   for lab in range(subcount):
    members=idx[labels==lab]
    if len(members)==0:continue
    groups[members,level]=next_id
    next_groups[next_id]=members
    next_id+=1
  parent_groups=next_groups;sizes.append(next_id)
 labels_all=np.full((n,len(branches)),-1,dtype=np.int32)
 labels_all[train_items]=groups
 for level in range(1,len(branches)):
  vals=np.unique(groups[:,level]); parent_n=[]
  for x in vals:
   parents=np.unique(groups[groups[:,level]==x,level-1]);parent_n.append(len(parents))
  assert all(x==1 for x in parent_n),'Hierarchy nesting broken'
 return labels_all,dict(level_group_counts=sizes,train_labeled_items=nnodes,total_items=n,edge_count=int(len(src)),parent_nesting_verified=True,source='U1 weighted co-occurrence ONLY; sparse graph diffusion sketch + recursive MiniBatchKMeans')

class ResidualVQ(nn.Module):
 def __init__(self,input_dim,dim,k,depth,numlabels):
  super().__init__()
  self.encoder=nn.Linear(input_dim,dim,bias=False)
  self.codebooks=nn.ParameterList([nn.Parameter(torch.randn(k,dim)*.03) for _ in range(depth)])
  self.decoder=nn.Linear(dim,input_dim,bias=False)
  # F1 classifier consumes HARD straight-through code embeddings only; levelwise cumulative code representation.
  self.label_heads=nn.ModuleList([nn.Linear(dim,numlabels[l]) for l in range(depth)])
  # F2: q_phi(c_v,l | hard ST code(c_u,1..l), hard prefix(c_v,1..l-1)).
  # Each prefix is a sum of quantized vectors, without soft code vectors.
  self.pair_predictors=nn.ModuleList([nn.Sequential(nn.Linear(2*dim,dim),nn.ReLU(),nn.Linear(dim,k)) for _ in range(depth)])
  self.depth=depth;self.k=k
 def forward(self,x,temp):
  h=F.normalize(self.encoder(x),dim=-1)
  residual=h;cum=torch.zeros_like(h);codes=[];probs=[];recons=[]
  for cb_param in self.codebooks:
   cb=F.normalize(cb_param,dim=-1)
   logits=(F.normalize(residual,dim=-1)@cb.T)/temp
   q=logits.softmax(-1);ind=logits.argmax(-1)
   onehot=F.one_hot(ind,self.k).to(q.dtype)
   hard_st=onehot.detach()+q-q.detach()
   quant=hard_st@cb
   cum=cum+quant
   residual=h-cum
   codes.append(ind);probs.append(q);recons.append(cum)
  rec=self.decoder(cum)
  return h,codes,probs,recons,rec

def init_model(a,z,train,level_sizes,dev):
 v2.seed_all(a.seed+101)
 model=ResidualVQ(z.shape[1],a.dim,a.k,a.depth,level_sizes).to(dev)
 with torch.no_grad():
  projected=F.normalize(model.encoder(torch.as_tensor(z[train],device=dev,dtype=torch.float32)),dim=-1).cpu().numpy()
  resid=projected.copy()
  for lv in range(a.depth):
   km=MiniBatchKMeans(n_clusters=a.k,random_state=a.seed+lv*99,n_init=3,batch_size=2048,max_iter=100)
   km.fit(resid);center=km.cluster_centers_.astype(np.float32)
   center/=np.maximum(np.linalg.norm(center,axis=1,keepdims=True),1e-8)
   model.codebooks[lv].copy_(torch.as_tensor(center,device=dev))
   # mimic residual quantization of normalized features with codebook vectors
   pred=km.predict(resid);resid=resid-center[pred]
 return {key:val.detach().cpu().clone() for key,val in model.state_dict().items()}

def supervised_contrastive(h,labels,parent,temperature,rng,item_ids=None):
 """Parent-local label contrast. Pairs are distinct items; no self positive."""
 # choose one positive item per anchor within minibatch; skip singleton labels
 n=len(labels); indices=np.arange(n)
 pairs=[]
 for i in range(n):
  if labels[i]<0:continue
  same=indices[(labels==labels[i]) & (indices!=i) & ((item_ids!=item_ids[i]) if item_ids is not None else True)]
  if len(same):pairs.append((i,int(rng.choice(same))))
 if not pairs:return h.sum()*0
 anchors=torch.tensor([i for i,j in pairs],device=h.device)
 pos=torch.tensor([j for i,j in pairs],device=h.device)
 sim=(F.normalize(h,dim=-1)@F.normalize(h,dim=-1).T)/temperature
 # parent-local negatives at levels>1; level1 includes all distinct pseudo groups
 mask=torch.tensor((parent[:,None]==parent[None,:]) & (labels[:,None]!=labels[None,:]),device=h.device)
 allowed=mask[anchors]
 # positive guaranteed separate from negatives; add positive individually to partition.
 denom=torch.logsumexp(torch.cat([sim[anchors,pos,None],sim[anchors].masked_fill(~allowed,-1e5)],dim=1),dim=1)
 return torch.mean(denom-sim[anchors,pos])

def estimate_hard_entropy(codes,k):
 """Detached hard-code marginal entropy in a training minibatch (not soft perplexity)."""
 with torch.no_grad():
  return np.array([float(-(lambda p:(p[p>0]*p[p>0].log()).sum())(
   torch.bincount(c.detach(),minlength=k).float()/len(c))) for c in codes],dtype=float)

def train_model_v12(a,z,train_items,e1,labels,method,lam,initial,dev,targets_u1=None,probe_holdout=None,beta=1.0):
 """One controller only: fixed beta, hard-soft EMA correction, signed entropy loss.
 Uniform U1 item samples are independent of degree-weighted task edge samples.
 RQ reference measured on U1 for gradients; U2 is selection-only.
 """
 v2.seed_all(a.seed+101)
 size=[int(labels[:,lv].max())+1 for lv in range(a.depth)]
 model=ResidualVQ(z.shape[1],a.dim,a.k,a.depth,size).to(dev)
 model.load_state_dict(initial)
 optimizer=torch.optim.Adam(model.parameters(),lr=a.lr)
 xx=torch.as_tensor(z,device=dev,dtype=torch.float32)
 e_a,e_b,e_w,_=e1
 rng=np.random.default_rng(a.seed+987)
 selection=rng.choice(len(e_a),size=a.steps*a.batch,replace=True,p=e_w/e_w.sum())
 hard_hist_ema=np.zeros((a.depth,a.k),dtype=np.float64)
 soft_hist_ema=np.zeros((a.depth,a.k),dtype=np.float64)
 logs=[]
 for step in range(a.steps):
  edges=selection[step*a.batch:(step+1)*a.batch]
  left,right=e_a[edges],e_b[edges]
  chosen=np.r_[left,right]
  h,codes,probs,cums,rec=model(xx[chosen],a.soft_temperature)
  recon=F.mse_loss(rec,xx[chosen])*z.shape[1]
  # Controller operates on uniformly sampled U1 ITEMS (not degree-biased U1 edges).
  # Thus its hard entropy and the RQ-only U1 entropy reference use the same item distribution.
  reg_idx=rng.choice(train_items,size=a.batch*2,replace=True)
  _,reg_codes,reg_probs,_,_=model(xx[reg_idx],a.soft_temperature)
  # Soft entropy is differentiable; hard histogram provides detached target correction.
  # The two EMA marginal distributions are computed from the SAME uniform U1 item minibatch.
  # No adaptive beta / dual loop is combined with this controller.
  soft_h_batch=torch.stack([-(q.mean(0)*q.mean(0).clamp_min(1e-10).log()).sum() for q in reg_probs])
  with torch.no_grad():
   obs_hard=[];obs_soft=[]
   for lv,(hard_c,soft_q) in enumerate(zip(reg_codes,reg_probs)):
    hard_p=torch.bincount(hard_c.detach(),minlength=a.k).float().cpu().numpy()
    hard_p/=max(hard_p.sum(),1.)
    soft_p=soft_q.detach().mean(0).cpu().numpy()
    soft_p/=max(soft_p.sum(),1.e-12)
    momentum=a.entropy_ema if step>0 else 0.
    hard_hist_ema[lv]=momentum*hard_hist_ema[lv]+(1.-momentum)*hard_p
    soft_hist_ema[lv]=momentum*soft_hist_ema[lv]+(1.-momentum)*soft_p
    hp=hard_hist_ema[lv];sp=soft_hist_ema[lv]
    obs_hard.append(float(-(hp[hp>0]*np.log(hp[hp>0])).sum()))
    obs_soft.append(float(-(sp[sp>0]*np.log(sp[sp>0])).sum()))
  # H(soft)-H(hard) correction; stopgrad because it is a selection target, not a loss path.
  if targets_u1 is None or step+1<=a.entropy_warmup or method=='rq':
   entropy_penalty=soft_h_batch.sum()*0.
   corrected_targets=[float('nan')]*a.depth
  else:
   corrected_targets=np.asarray(targets_u1)+np.asarray(obs_soft)-np.asarray(obs_hard)
   target_t=torch.as_tensor(corrected_targets,dtype=soft_h_batch.dtype,device=dev)
   entropy_penalty=((soft_h_batch-target_t)**2).mean()
  task=h.sum()*0
  if method=='rq':pass
  elif method=='contrastive':
   for lv in range(a.depth):
    lab=labels[chosen,lv].copy()
    if probe_holdout is not None:lab[probe_holdout[chosen]]=-1
    parent=(labels[chosen,lv-1].copy() if lv else np.zeros(len(chosen),dtype=np.int32))
    if probe_holdout is not None:parent[probe_holdout[chosen]]=-1
    task=task+supervised_contrastive(cums[lv],lab,parent,a.temperature,rng,chosen)/a.depth
  elif method=='label_mi':
   for lv in range(a.depth):
    valid=np.flatnonzero((labels[chosen,lv]>=0) & (~probe_holdout[chosen] if probe_holdout is not None else True))
    if len(valid):
     ids=torch.tensor(valid,device=dev)
     targets=torch.tensor(labels[chosen[valid],lv].astype(np.int64),device=dev)
     task=task+F.cross_entropy(model.label_heads[lv](cums[lv][ids]),targets)/a.depth
  elif method=='pair_code_mi':
   for lv in range(a.depth):
    # Hard-code ST cumulative quantized embeddings retain gradients to the encoder.
    # Target codes are discrete labels; target prefix is hard ST code embeddings.
    source=cums[lv][:a.batch]
    target_prefix=cums[lv-1][a.batch:] if lv else torch.zeros_like(source)
    logits=model.pair_predictors[lv](torch.cat([source,target_prefix],dim=1))
    targets=codes[lv][a.batch:].detach()
    forward_ce=F.cross_entropy(logits,targets)
    rev_source=cums[lv][a.batch:]
    rev_prefix=cums[lv-1][:a.batch] if lv else torch.zeros_like(source)
    reverse_logits=model.pair_predictors[lv](torch.cat([rev_source,rev_prefix],dim=1))
    backward_ce=F.cross_entropy(reverse_logits,codes[lv][:a.batch].detach())
    task=task+(forward_ce+backward_ce)/(2*a.depth)
  elif method=='infonce':
   u=h[:a.batch];v=h[a.batch:]
   logit=u@v.T/a.temperature;tgt=torch.arange(a.batch,device=dev)
   task=(F.cross_entropy(logit,tgt)+F.cross_entropy(logit.T,tgt))/2
  else:raise ValueError(method)
  loss=recon+beta*entropy_penalty+lam*task
  optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
  hard_h=obs_hard;soft_h=obs_soft
  if step==0 or (step+1)%200==0 or step+1==a.steps:
   r=dict(step=step+1,loss=float(loss.detach()),recon=float(recon.detach()),task=float(task.detach()),level1_codes=int(codes[0].unique().numel()),beta_levels=[float(beta)]*a.depth,hard_entropy_ema=hard_h,soft_entropy_ema=soft_h,soft_entropy_batch=soft_h_batch.detach().cpu().tolist(),entropy_target_corrected=np.asarray(corrected_targets).tolist(),entropy_penalty=float(entropy_penalty.detach()))
   logs.append(r);print('TRAIN',method,'lambda',lam,r,flush=True)
 return model,logs,[float(beta)]*a.depth

@torch.no_grad()
def encode(model,z,device,temp):
 xx=torch.as_tensor(z,dtype=torch.float32)
 out=[]
 for st in range(0,len(z),2048):
  _,codes,_,_,_=model(xx[st:st+2048].to(device),temp)
  out.append(torch.stack(codes,dim=1).cpu().numpy().astype(np.int32))
 return np.concatenate(out,axis=0)

def kmeans_residual(a,z,train):
 residual=z.copy();allcodes=[]
 for lv in range(a.depth):
  km=MiniBatchKMeans(n_clusters=a.k,random_state=a.seed+lv*99,n_init=5,batch_size=2048,max_iter=200)
  km.fit(residual[train]);c=km.predict(residual).astype(np.int32)
  allcodes.append(c)
  residual=residual-km.cluster_centers_[c]
 return np.stack(allcodes,axis=1)

def eval_auc(c,dat,k,depth):
 """U1 -> U2 smoothing selection; no U3 scores until final arm selection."""
 out=[]
 for lv in range(depth):
  alpha,auc,_=v2.measure_validation(c[:,lv],dat['u2'],dat['edges'][0],k)
  out.append((alpha,auc))
 return out

def u3_metrics(c,dat,k,alphas,depth):
 e1,e2,e3=dat['edges']; u3=dat['u3']
 alllevel=[]
 score_levels=[]
 for lv in range(depth):
  codes=c[:,lv];p=core.code_joint(codes,*e3[:3],k)
  mi=core.measures(p)
  marg=p.sum(1);hh=float(-(marg[marg>0]*np.log(marg[marg>0])).sum())
  weighted=core.code_joint(codes,*e1[:3],k)*np.sum(e1[2]);weighted+=alphas[lv];weighted/=weighted.sum()
  score=core.score_pmi(codes,u3[0],u3[1],weighted)[2]
  score_levels.append(score)
  alllevel.append(dict(level=lv+1,J=float(mi['mi_real']),H_edge=hh,J_over_H_edge=float(mi['mi_real']/hh) if hh else float('nan'),H_item=entropy(codes,dat['eligible'],k),used_codes=int(np.unique(codes[dat['eligible']]).size),AUC=roc_auc_score(u3[2],score)))
 # AUC main = average per-level PMI score, with fixed equal weighting, no U3 tuning
 stacked=np.mean(score_levels,axis=0)
 return alllevel, float(roc_auc_score(u3[2],stacked)),stacked

def bootstrap_paired(scores,labs,items_a,items_b,n_items,nboot,seed):
 """Paired fixed-candidate item-Poisson bootstrap (NOT user bootstrap)."""
 rng=np.random.default_rng(seed);records=[]
 for _ in range(nboot):
  counts=rng.poisson(1,size=n_items)
  weights=counts[items_a].astype(float)*counts[items_b]
  if weights.sum()<=1 or weights[labs==1].sum()==0 or weights[labs==0].sum()==0:continue
  # weighted AUC via sorted score ranks, tie-safe group accumulation
  vals=[]
  for s in scores:
   order=np.argsort(s,kind='mergesort');ss=s[order];yy=labs[order];ww=weights[order]
   neg_total=(ww*(1-yy)).sum();pos_total=(ww*yy).sum()
   # Vectorized tie-aware weighted Mann-Whitney statistic.
   begin=np.r_[0, np.flatnonzero(np.diff(ss)!=0)+1]
   wneg=np.add.reduceat(ww*(1-yy),begin)
   wpos=np.add.reduceat(ww*yy,begin)
   neg_below=np.cumsum(wneg)-wneg
   wins=float(np.dot(wpos,neg_below+.5*wneg))
   vals.append(wins/(pos_total*neg_total))
  records.append(vals[0]-vals[1])
 return np.asarray(records)

def self_test():
 v2.seed_all(42)
 n=1200;k=16;depth=3;d=32
 z=np.random.default_rng(42).normal(size=(n,d)).astype(np.float32)
 train_idx=np.arange(900,dtype=np.int32)
 rng=np.random.default_rng(22)
 ea=rng.integers(0,900,size=3000,dtype=np.int32);eb=rng.integers(0,900,size=3000,dtype=np.int32)
 w=np.ones(len(ea),dtype=float)
 e1=(ea,eb,w,np.zeros(n))
 labels,meta=make_hierarchy(e1,train_idx,n,(4,2,2),16,42)
 assert np.all(labels[900:]==-1) and meta['parent_nesting_verified']
 a=argparse.Namespace(seed=42,dim=16,k=k,depth=depth,steps=2,batch=16,lr=.001,soft_temperature=.1,temperature=.2)
 init=init_model(a,z,train_idx,meta['level_group_counts'],torch.device('cpu'))
 for method in ARMS[:-1]:
  a.entropy_ema=.9;a.entropy_warmup=0
  model,logs,_=train_model_v12(a,z,train_idx,e1,labels,method,.1,init,torch.device('cpu'))
  c=encode(model,z,torch.device('cpu'),.1)
  assert c.shape==(n,depth) and c.min()>=0 and c.max()<k
 c=kmeans_residual(a,z,train_idx);assert c.shape==(n,depth)
 print('SELF-TEST PASS: nested U1-only hierarchy, all 4 losses, hard residual codes, kmeans',flush=True)

def main():
 a=parser()
 if a.mode=='self-test':self_test();return
 if not a.domain:raise SystemExit('--domain required')
 if a.depth<2 or a.k<2 or a.batch<2 or a.tolerance_pct<=0:raise ValueError('invalid basic settings')
 branch=tuple(int(s) for s in a.hierarchy_branches.split(','))
 if len(branch)!=a.depth or min(branch)<2:raise ValueError('hierarchy-branches must have one >=2 integer per depth')
 lambdas=tuple(float(s) for s in a.lambda_grid.split(','))
 betas=tuple(float(x) for x in a.beta_grid.split(','))
 if min(lambdas)<0 or min(betas)<=0 or not 0<=a.entropy_ema<1 or a.entropy_warmup<0:raise ValueError('invalid regularizer/controller')
 if a.mode=='smoke':a.steps=2;a.batch=16;a.eval_positives=100;a.bootstrap=10;lambdas=(.1,);betas=(1.,);a.entropy_warmup=0
 if a.mode=='diagnose':lambdas=(0.,)
 old=argparse.Namespace(domain=a.domain,mapping_path=a.mapping_path,embedding_path=a.embedding_path,sequence_path=a.sequence_path,seed=a.seed,k=a.k,dim=a.dim,steps=a.steps,batch=a.batch,lr=a.lr,temperature=a.temperature,soft_temperature=a.soft_temperature,balance_weight=.1,lambda_grid=','.join(map(str,lambdas)),eval_every=200,eval_positives=a.eval_positives,bootstrap=a.bootstrap,nulls=0,swaps_per_edge=0,modality=a.modality,out_root=a.out_root,device=a.device,dry_run=a.mode=='check')
 dat=v2.make_data(old)
 if a.mode=='check':print('CHECK PASS: input files found; no training performed');return
 if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable. Set CUDA_VISIBLE_DEVICES=1 or use --device cpu')
 device=torch.device(a.device)
 dest=(ROOT/a.out_root/('smoke' if a.mode=='smoke' else ('diagnostic' if a.mode=='diagnose' else 'full'))/f'{a.domain}_K{a.k}_seed{a.seed}').resolve()
 if dest.exists():raise FileExistsError('Refusing to overwrite output '+str(dest))
 dest.mkdir(parents=True)
 (dest/'config.json').write_text(json.dumps(dict(vars(a),lambda_grid=list(lambdas)),indent=2))
 # U1 graph labels, withheld from all U2/U3 scoring and hyperparameter selection
 labels,meta=make_hierarchy(dat['edges'][0],dat['train_items'],dat['n'],branch,a.dim,a.seed)
 probe_holdout=make_probe_holdout(dat['train_items'],dat['n'],a.seed)
 meta['probe_holdout_items']=int(probe_holdout.sum())
 np.save(dest/'u1_hierarchy_labels.npy',labels)
 np.save(dest/'probe_holdout_mask.npy',probe_holdout)
 (dest/'hierarchy_manifest.json').write_text(json.dumps(meta,indent=2))
 u2items=np.unique(np.r_[dat['edges'][1][0],dat['edges'][1][1]]).astype(np.int32)
 mods=('text','image') if a.modality=='both' else (a.modality,)
 candidates=[];summary=[];levelrows=[];pairrows=[];proberows=[]
 for mod in mods:
  z=core.norm(dat['emb'][mod][0]);train=dat['train_items']
  # Same fixed initialized projection for all arms and Euclidean residual KMeans baseline.
  initial=init_model(a,z,train,meta['level_group_counts'],device)
  init_weight=initial['encoder.weight'].numpy()
  projected=z@init_weight.T
  projected/=np.maximum(np.linalg.norm(projected,axis=1,keepdims=True),1e-8)
  baseline=kmeans_residual(a,projected,train)
  kmeans_u2=[entropy(baseline[:,lv],u2items,a.k) for lv in range(a.depth)]
  mdir=dest/mod;mdir.mkdir()
  np.save(mdir/'kmeans_codes.npy',baseline)
  # First compute architecture-matched RQ-only reference, NOT the Euclidean KMeans target.
  rq_model,rq_log,rq_dual=train_model_v12(a,z,train,dat['edges'][0],labels,'rq',0.,initial,device,probe_holdout=probe_holdout)
  rq_codes=encode(rq_model,z,device,a.soft_temperature)
  target_u1=[entropy(rq_codes[:,lv],train,a.k) for lv in range(a.depth)]
  target_u2=[entropy(rq_codes[:,lv],u2items,a.k) for lv in range(a.depth)]
  np.save(mdir/'rq_codes.npy',rq_codes)
  torch.save(dict(model={k:v.detach().cpu() for k,v in rq_model.state_dict().items()},logs=rq_log,model_kind='ResidualVQ_V12'),mdir/'rq_selected.pt')
  (mdir/'reference_entropy.json').write_text(json.dumps(dict(target='RQ-only learned architecture',rq_u1_h=target_u1,rq_u2_h=target_u2,kmeans_u2_h=kmeans_u2,rq_vs_kmeans_u2_gap_pct=[100*abs(x-y)/max(y,1e-12) for x,y in zip(target_u2,kmeans_u2)]),indent=2))
  print('REFERENCE',mod,'RQ-only U2 entropy',target_u2,'KMeans U2 entropy',kmeans_u2,flush=True)
  winners={'kmeans':dict(c=baseline,alpha=[v[0] for v in eval_auc(baseline,dat,a.k,a.depth)],beta_levels=[0.]*a.depth,lam=0.),
           'rq':dict(c=rq_codes,alpha=[v[0] for v in eval_auc(rq_codes,dat,a.k,a.depth)],beta_levels=rq_dual,lam=0.)}
  candidates.append(dict(domain=a.domain,seed=a.seed,modality=mod,method='rq',lam=0.,u2_entropy=json.dumps(target_u2),u2_entropy_target=json.dumps(target_u2),u2_entropy_gaps_pct=json.dumps([0.]*a.depth),matched_all_levels=True))
  if a.mode!='diagnose':
   for method in ('contrastive','label_mi','pair_code_mi','infonce'):
    eligible=[];all_candidates=[]
    for j,(lam,beta) in enumerate([(x,y) for x in lambdas for y in betas]):
     if a.max_candidates is not None and j>=a.max_candidates:break
     model,log,dual=train_model_v12(a,z,train,dat['edges'][0],labels,method,lam,initial,device,target_u1,probe_holdout,beta=beta)
     codes=encode(model,z,device,a.soft_temperature)
     u1h=[entropy(codes[:,lv],train,a.k) for lv in range(a.depth)]
     us=[entropy(codes[:,lv],u2items,a.k) for lv in range(a.depth)]
     gaps=[100*abs(x-y)/max(abs(y),1e-12) for x,y in zip(us,target_u2)]
     val=eval_auc(codes,dat,a.k,a.depth)
     auc=float(np.mean([x[1] for x in val]));matched=bool(all(g<=a.tolerance_pct for g in gaps))
     row=dict(domain=a.domain,seed=a.seed,modality=mod,method=method,lam=lam,beta=beta,u2_auc_mean=auc,u2_entropy_gaps_pct=json.dumps(gaps),u2_entropy=json.dumps(us),u2_entropy_target=json.dumps(target_u2),u1_entropy=json.dumps(u1h),beta_levels=json.dumps(dual),matched_all_levels=matched)
     candidates.append(row)
     # Persist EVERY candidate regardless of U2 entropy matching outcome.
     candidate_dir=mdir/'candidates'/f'{method}_lambda_{str(lam).replace(".","p")}_beta_{str(beta).replace(".","p")}'
     candidate_dir.mkdir(parents=True,exist_ok=False)
     np.save(candidate_dir/'codes.npy',codes)
     state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
     if a.save_checkpoints:torch.save(dict(model=state,selection=row,logs=log,model_kind='ResidualVQ_V12'),candidate_dir/'model.pt')
     (candidate_dir/'metrics.json').write_text(json.dumps(dict(row,logs=log),indent=2))
     print('U2',mod,method,'lambda',lam,'mean_auc',round(auc,5),'gaps',np.round(gaps,2),'MATCH',matched,flush=True)
     rec=(auc,-max(gaps),-lam,codes.copy(),val,row,state)
     all_candidates.append(rec)
     if matched:eligible.append(rec)
     del model
    if eligible:
     win=max(eligible,key=lambda x:x[:3]);_,_,_,codes,vals,row,state=win
     winners[method]=dict(c=codes,alpha=[x[0] for x in vals],beta_levels=json.loads(row['beta_levels']),lam=row['lam'])
     np.save(mdir/f'{method}_codes.npy',codes)
     if a.save_checkpoints:torch.save(dict(model=state,selection=row,model_kind='ResidualVQ_V12'),mdir/f'{method}_selected.pt')
    else:
     win=max(all_candidates,key=lambda x:x[:3]);_,_,_,codes,vals,row,state=win
     # Unmatched candidates are selected ONLY on U2 and retained for exploratory audit.
     # By default, do NOT reveal U3 to avoid making repeated protocol revision adaptive to U3.
     np.save(mdir/f'{method}_unmatched_codes.npy',codes)
     (mdir/f'{method}_unmatched_selection.json').write_text(json.dumps(row,indent=2))
     if a.evaluate_unmatched:winners[method]=dict(c=codes,alpha=[x[0] for x in vals],beta_levels=json.loads(row['beta_levels']),lam=row['lam'],exploratory=True)
     print('MATCH_FAILED',mod,method,'No U2 matched candidate; best-U2 unmatched retained. U3 not opened unless --evaluate-unmatched.',flush=True)
  (mdir/'u2_selected.json').write_text(json.dumps({m:{k:v for k,v in entry.items() if k!='c'} for m,entry in winners.items()},indent=2))
  if a.mode=='diagnose':continue
  if len(winners)!=len(ARMS):
   print('PARTIAL',mod,'matched',sorted(winners),'expected',ARMS,'Evaluating available arms without aborting.',flush=True)
  u3scores={}
  for method,entry in winners.items():
   levels,auc,scores=u3_metrics(entry['c'],dat,a.k,entry['alpha'],a.depth)
   u3scores[method]=scores
   summary.append(dict(domain=a.domain,seed=a.seed,modality=mod,method=method,match_status=('exploratory_unmatched' if entry.get('exploratory') else 'matched'),beta_levels=json.dumps(entry['beta_levels']),lam=entry['lam'],u3_auc_mean_score=auc,u3_J_level1=levels[0]['J'],u3_J_H_level1=levels[0]['J_over_H_edge'],u3_H_edge_level1=levels[0]['H_edge'],u3_J_mean=np.mean([x['J'] for x in levels]),u3_J_H_mean=np.mean([x['J_over_H_edge'] for x in levels]),u3_all_levels_entropy=json.dumps([x['H_item'] for x in levels]),u3_all_levels_J=json.dumps([x['J'] for x in levels])))
   for lvl in levels:levelrows.append(dict(domain=a.domain,seed=a.seed,modality=mod,method=method,**lvl))
   for pr in cold_probe(entry['c'],labels,train,probe_holdout,a.k,a.depth):proberows.append(dict(domain=a.domain,seed=a.seed,modality=mod,method=method,**pr))
  u3=dat['u3'];pairs=[('label_mi','contrastive'),('label_mi','pair_code_mi'),('label_mi','infonce'),('pair_code_mi','infonce'),('label_mi','kmeans')]
  for j,(lhs,rhs) in enumerate(pairs):
   if lhs not in u3scores or rhs not in u3scores:continue
   dist=bootstrap_paired([u3scores[lhs],u3scores[rhs]],u3[2],u3[0],u3[1],dat['n'],a.bootstrap,a.seed+1901+j)
   point=roc_auc_score(u3[2],u3scores[lhs])-roc_auc_score(u3[2],u3scores[rhs])
   if len(dist)>5:lo,hi=np.quantile(dist,[.025,.975]);bias=dist.mean()-point
   else:lo=hi=bias=float('nan')
   pairrows.append(dict(domain=a.domain,seed=a.seed,modality=mod,lhs=lhs,rhs=rhs,AUC_diff=point,bootstrap_bias=bias,CI_percentile_lo=lo,CI_percentile_hi=hi,CI_centered_lo=lo-bias,CI_centered_hi=hi-bias,bootstrap_unit='U3 fixed-candidate item Poisson; NOT user-level'))
 save_csv(dest/'u2_candidates.csv',candidates);save_csv(dest/'summary.csv',summary);save_csv(dest/'level_metrics.csv',levelrows);save_csv(dest/'paired_auc_ci.csv',pairrows);save_csv(dest/'cold_label_probe.csv',proberows)
 (dest/'manifest.json').write_text(json.dumps(dict(protocol='EXP3_HIERARCHICAL_SID_TOKENIZER_V1.2',status='exploratory',data_split='U1/U2/U3 from Exp2 v2 (seed-configurable)',labels='U1 graph only; unlabeled external items -1',model='3-depth residual straight-through quantizer (hard code in F1 CE heads)',entropy_match='U2 levelwise H(C) matching to learned RQ-only reference; uniform U1 items for detached hard/soft EMA target correction and fixed beta signed entropy loss; all levels within 1 percent',evaluation='U3 weighted pair-code MI per depth and PMI transition AUC; scores equal mean across 3 levels',limitations=['New exploratory v1.1 protocol established after v1 U2 matching failure; no U3 was inspected in v1','RQ-only target uses U1-only model training but U2 reference code histogram for selection','Variational CE is a predictive surrogate, not proven unbiased conditional mutual information estimator','Fixed beta target controller uses uniform U1 item minibatches; no guarantee of 1% matching','Not T5 recommender or multi-branch tokenizer','Cold probe labels inferred from full U1 graph, so probe is item-disjoint supervised training but transductive in graph construction','Cold MultinomialNB hard-code prefix probe evaluated on disjoint U1 item holdout','Item cluster paired AUC bootstrap not user-level','Not a conditional hierarchical MI estimator; levels evaluated separately','Parent-local supervised contrastive uses pseudo labels sampled within minibatches; sparse groups may yield few positives','No multiple seed confirmed yet']),indent=2))
 (dest/'DONE.json').write_text(json.dumps(dict(status=('diagnostic_complete' if a.mode=='diagnose' else ('success' if len(summary)==len(mods)*len(ARMS) and not any(r.get('match_status')=='exploratory_unmatched' for r in summary) else 'partial_matching')),rows=len(summary),candidate_rows=len(candidates),modalities_completed=sorted({r['modality'] for r in summary})),indent=2))
 print(('DIAG SUCCESS' if a.mode=='diagnose' else ('SUCCESS' if len(summary)==len(mods)*len(ARMS) else ('PARTIAL_SUCCESS' if summary else 'MATCH_FAILED'))) ,dest,'summary rows',len(summary),flush=True)

def train_model_wrapper(*args,**kwargs):return train_model_v12(*args,**kwargs)

def make_probe_holdout(train_items,n,seed,frac=.2):
 rng=np.random.default_rng(seed+9000)
 selected=rng.choice(train_items,size=max(1,int(len(train_items)*frac)),replace=False)
 flag=np.zeros(n,dtype=bool);flag[selected]=True
 return flag

def onehot_prefix(c,upto,k):
 n=len(c);cols=np.arange(upto)[None,:]*k+c[:,:upto]
 rr=np.repeat(np.arange(n),upto)
 return sparse.csr_matrix((np.ones(n*upto,dtype=np.float32),(rr,cols.ravel())),shape=(n,upto*k))

def cold_probe(c,labels,train_items,holdout,k,depth):
 """Held-out item pseudo-label predictive information from hard code prefix.
 Naive Bayes is fitted ONLY on non-heldout U1 items, labels originally inferred
 from U1 graph. H(Y)-heldout CE is a predictive bound, not full mutual information.
 """
 tr=train_items[~holdout[train_items]];te=train_items[holdout[train_items]]
 rows=[]
 for lv in range(depth):
  valid_tr=tr[labels[tr,lv]>=0];valid_te=te[labels[te,lv]>=0]
  if len(valid_tr)<10 or len(valid_te)<10:continue
  ytrain=labels[valid_tr,lv];ytest=labels[valid_te,lv]
  fitted=MultinomialNB(alpha=1.)
  fitted.fit(onehot_prefix(c[valid_tr],lv+1,k),ytrain)
  x=onehot_prefix(c[valid_te],lv+1,k)
  pred=fitted.predict(x); proba=fitted.predict_proba(x)
  ce=float(log_loss(ytest,proba,labels=fitted.classes_)) if set(np.unique(ytest)).issubset(set(fitted.classes_)) else float('nan')
  counts=np.bincount(ytrain,minlength=int(np.max(labels[train_items,lv]))+1).astype(float)
  prior=counts/counts.sum();base_ce=-float(np.mean(np.log(prior[ytest].clip(1e-12))))
  rows.append(dict(level=lv+1,probe_train_items=len(valid_tr),probe_test_items=len(valid_te),probe_accuracy=float(accuracy_score(ytest,pred)),probe_label_ce=ce,probe_prior_ce=base_ce,predictive_info_gain_nats=base_ce-ce))
 return rows

if __name__=='__main__':main()
