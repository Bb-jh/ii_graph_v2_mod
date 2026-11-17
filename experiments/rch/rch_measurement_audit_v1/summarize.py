#!/usr/bin/env python3
"""Read-only summarizer for measurement audit results."""
import argparse
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser()
p.add_argument('--domain',default='Industrial')
p.add_argument('--seed',type=int,default=42)
p.add_argument('--mode',default='run')
p.add_argument('--out-root',default='outputs/rch_measurement_audit_v1')
a=p.parse_args()
root=Path(a.out_root)/a.mode/f'{a.domain}_seed{a.seed}'
for name in ('root_learning_curve.csv','node_matched_depth1.csv','node_matched_depth2.csv','node_matched_depth3.csv','pooled_depth1_summary.csv','pooled_depth2_summary.csv','pooled_depth3_summary.csv','pooled_depth2_parents.csv','pooled_depth3_parents.csv'):
    path=root/name
    if path.exists():
        print('\n===== '+name+' =====')
        df=pd.read_csv(path)
        cols=[x for x in ['depth','parent','requested_train_n','status','n_train','n_test','R','R_lo','R_hi','null_control_R','null_control_lo','null_control_gain','pass_T'] if x in df]
        print(df[cols].to_string(index=False))
print('\nInterpretation: CI is conditional on fitted probe; this is a transductive U1-label, item-heldout diagnostic, NOT strict cold-item generalization.')
