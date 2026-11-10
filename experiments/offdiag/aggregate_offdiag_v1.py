#!/usr/bin/env python3
import csv,glob,json
from pathlib import Path
root=Path('outputs/offdiag_analysis_v1')
rows=[]
for path in sorted(root.glob('*_CLIP_K*_seed*/summary.csv')):
 with path.open() as f: rows.extend(csv.DictReader(f))
if not rows:raise SystemExit('No summary.csv files found')
out=root/'summary_all.csv'
if out.exists():raise SystemExit(f'{out} exists; refusing overwrite')
with out.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print('Wrote',out,'rows=',len(rows))
