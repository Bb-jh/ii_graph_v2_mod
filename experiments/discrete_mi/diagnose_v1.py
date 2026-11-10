#!/usr/bin/env python3
"""Read-only diagnosis of trained EXP2_LEVEL1_V1 checkpoints.
Does not load U2/U3 labels or alter any training outputs.
"""
import argparse, csv, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'/'offdiag'))
import run_offdiag_v1 as old
from run_level1 import VQ

p=argparse.ArgumentParser()
p.add_argument('--domain',choices=['Industrial','Musical'],required=True)
p.add_argument('--modality',choices=['text','image','both'],default='both')
p.add_argument('--out-root',default='outputs/discrete_mi_level1_v1')
p.add_argument('--batch',type=int,default=2048)
a=p.parse_args()
dataset,_=old.DOMAINS[a.domain]
folder=ROOT/a.out_root/f'{a.domain}_K256_seed42'
if not folder.exists():raise FileNotFoundError(folder)
mapping=ROOT/'shared_data'/dataset/'5core_v2'/'item_mapping.tsv'
embedding=ROOT/'artifacts'/'embeddings'/dataset/'full'/'clip_vit_b32.npz'
ids=old.load_mapping(mapping)
data=old.load_embeddings(embedding,ids)
rows=[]
for mod in (['text','image'] if a.modality=='both' else [a.modality]):
 x=old.norm(data[mod][0]);eligible=data[mod][1]
 subset=np.flatnonzero(eligible)
 print(f'=== {a.domain}/{mod}: n_items={len(subset)} ===',flush=True)
 for method in ['rq','infonce','code_mi']:
  fp=folder/mod/f'{method}_selected.pt'
  if not fp.exists():
   print('NOT FOUND:',fp);continue
  checkpoint=torch.load(fp,map_location='cpu',weights_only=True)
  cfg=checkpoint['config'];k=int(cfg['k']);d=int(cfg['dim']);temperature=float(cfg['soft_temperature'])
  model=VQ(x.shape[1],d,k).cpu()
  model.load_state_dict(checkpoint['model']);model.eval()
  max_probs=[];q_entropy=[];hard_counts=np.zeros(k,dtype=np.int64);qtotal=np.zeros(k,dtype=np.float64)
  with torch.no_grad():
   cb=F.normalize(model.codebook,dim=1)
   for ix in range(0,len(subset),a.batch):
    batch_ix=subset[ix:ix+a.batch]
    z=torch.from_numpy(np.asarray(x[batch_ix],dtype=np.float32))
    h=F.normalize(model.encoder(z),dim=1)
    logits=h@cb.T/temperature
    q=logits.softmax(-1)
    codes=logits.argmax(-1)
    hard_counts+=np.bincount(codes.numpy(),minlength=k)
    max_probs.append(q.max(-1).values.numpy())
    q_entropy.append((-(q*q.clamp_min(1e-12).log()).sum(-1)).numpy())
    qtotal+=q.sum(0).numpy()
  mx=np.concatenate(max_probs);he=np.concatenate(q_entropy)
  counts=hard_counts/hard_counts.sum();soft=qtotal/qtotal.sum()
  hard_used=int((hard_counts>0).sum())
  hard_entropy=float(-(counts[counts>0]*np.log(counts[counts>0])).sum())
  soft_entropy=float(-(soft[soft>0]*np.log(soft[soft>0])).sum())
  row=dict(domain=a.domain,modality=mod,method=method,k=k,temp=temperature,hard_used_codes=hard_used,hard_max_code_share=float(counts.max()),hard_perplexity=float(np.exp(hard_entropy)),mean_soft_max_probability=float(mx.mean()),mean_per_item_soft_entropy=float(he.mean()),mean_per_item_soft_perplexity=float(np.exp(he.mean())),aggregate_soft_perplexity=float(np.exp(soft_entropy)),uniform_soft_probability=1/k)
  rows.append(row)
  print(json.dumps(row,ensure_ascii=False),flush=True)
 dest=ROOT/'outputs'/'discrete_mi_diagnostics_v1'/f'{a.domain}_K256_seed42'
 dest.mkdir(parents=True,exist_ok=True)
 with (dest/'assignment_diagnostics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print('Saved:',dest/'assignment_diagnostics.csv')
