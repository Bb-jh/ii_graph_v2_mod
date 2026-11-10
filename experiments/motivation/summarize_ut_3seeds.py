#!/usr/bin/env python3
"""Summarize 3 user-resplit experiments. SD is across correlated resplits."""
from pathlib import Path
import argparse,csv
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);a=p.parse_args()
 rows=[]
 for s in a.seeds:
  fn=a.root/f'industrial_split{s}'/'routing_deltas_item_boot.csv'
  if not fn.is_file():raise FileNotFoundError(fn)
  with fn.open() as f:rows.extend(list(csv.DictReader(f)))
 out=[]
 for m in ('T','V'):
  for q in (.1,.2,.3):
   for comparator in (f'U_{m}_minus_Binmatched_RawCosine_{m}',f'U_{m}_minus_Raw_wB_gs_{m}'):
    z=[x for x in rows if x['modality']==m and abs(float(x['q'])-q)<1e-8 and x['comparison']==comparator]
    if len(z)!=len(a.seeds):raise ValueError(f'missing seeds: {m} {q} {comparator}')
    v=np.array([float(x['delta_rec_pp']) for x in z]);w=np.array([float(x['delta_strength']) for x in z]);
    out.append(dict(modality=m,q=q,comparison=comparator,n=len(v),mean_delta_rec_pp=v.mean(),
        sample_sd_delta_rec_pp=v.std(ddof=1) if len(v)>1 else 0,positive_splits=int((v>0).sum()),
        mean_delta_strength=w.mean(),sample_sd_delta_strength=w.std(ddof=1) if len(w)>1 else 0))
 dest=a.root/'multiseed_summary.csv'
 with dest.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
 print('Wrote',dest)
 for r in out:
  if r['q']==.2:print(r)
if __name__=='__main__':main()
