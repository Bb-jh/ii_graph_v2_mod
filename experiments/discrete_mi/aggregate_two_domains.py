#!/usr/bin/env python3
"""Combine two local complete summary.csv files supplied as paths; no remote transfers."""
import argparse,csv
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('industrial',type=Path)
p.add_argument('musical',type=Path)
p.add_argument('--out',type=Path,default=Path('outputs/discrete_mi_level1_v1/combined_summary.csv'))
a=p.parse_args()
rows=[]
for file in (a.industrial,a.musical):
 with file.open(newline='') as f:rows.extend(csv.DictReader(f))
if not rows:raise SystemExit('No rows')
a.out.parent.mkdir(parents=True,exist_ok=True)
with a.out.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('WROTE',a.out,len(rows),'rows')
