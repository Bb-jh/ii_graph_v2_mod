#!/usr/bin/env python3
"""Summarize per-seed neighbor reliability contrasts (descriptive mean/sample SD)."""
import argparse,csv
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--root',type=Path,required=True)
 p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
 p.add_argument('--dataset',default='Industrial_and_Scientific')
 a=p.parse_args()
 rows=[]
 for seed in a.seeds:
  path=a.root/f'{a.dataset}_split{seed}'/'paired_anchor_bootstrap_ci.csv'
  if not path.exists():raise FileNotFoundError(f'Missing seed {seed}: {path}')
  with path.open(newline='') as f:
   for r in csv.DictReader(f):
    r['seed']=seed;rows.append(r)
 groups={}
 for r in rows:
  key=(int(r['k']),r['degree_group'],r['metric'])
  groups.setdefault(key,[]).append((int(r['seed']),float(r['pbase_minus_wb'])))
 out=[]
 for (k,group,metric),values in sorted(groups.items()):
  vals=np.array([v for _,v in values]);seeds=','.join(str(s) for s,_ in values)
  out.append(dict(dataset=a.dataset,k=k,degree_group=group,metric=metric,n_seeds=len(values),seeds=seeds,
    mean_delta=float(vals.mean()),sample_sd=float(vals.std(ddof=1)) if len(vals)>1 else float('nan'),
    positive_seeds=int((vals>0).sum()),per_seed_values=';'.join(f'{v:+.7f}' for v in vals)))
 dest=a.root/'multiseed_neighbor_summary.csv'
 with dest.open('w',newline='') as f:
  w=csv.DictWriter(f,out[0].keys());w.writeheader();w.writerows(out)
 print('SUMMARY:',dest)
 for r in out:
  if r['metric']=='recurrence':print(f"k={r['k']:2d} {r['degree_group']:<4} Pbase-WB: {r['mean_delta']:+.3f} +/- {r['sample_sd']:.3f} pp, positive {r['positive_seeds']}/{r['n_seeds']}")
if __name__=='__main__':main()
