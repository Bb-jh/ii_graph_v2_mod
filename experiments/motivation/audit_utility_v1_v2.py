#!/usr/bin/env python3
"""Audit that the patched evaluator preserved old metrics while repairing U3 fold transfer."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True);a=p.parse_args()
old=pd.read_csv(Path(a.old)/'utility_metrics.csv').set_index('split')
new=pd.read_csv(Path(a.new)/'utility_metrics.csv').set_index('split')
cols=[c for c in old.columns if c in new.columns and pd.api.types.is_numeric_dtype(old[c])]
for split in ('U2_OOF','U3_TRANSFER'):
    if split not in old.index or split not in new.index:raise SystemExit(f'Missing split {split}')
    delta=[(c,float(old.loc[split,c]),float(new.loc[split,c])) for c in cols if not np.isclose(old.loc[split,c],new.loc[split,c],atol=1e-10,rtol=1e-9,equal_nan=True)]
    print(split,'MATCH' if not delta else 'DIFFERENCES',len(delta))
    for d in delta[:20]:print(' ',d)
    if delta:raise SystemExit('Audit failed: old metrics changed')
if 'U3_TRANSFER_ITEM_FOLD' not in new.index:raise SystemExit('Missing new item-fold U3 transfer')
print('v1 U3 full and OOF subset identical?', ('U3_TRANSFER_OOF_SUBSET' in old.index and np.isclose(old.loc['U3_TRANSFER','u_T'],old.loc['U3_TRANSFER_OOF_SUBSET','u_T'])))
print('Corrected fold transfer u_T/u_V:',new.loc['U3_TRANSFER_ITEM_FOLD',['u_T','u_V']].to_dict())
print('Full transfer u_T/u_V:',new.loc['U3_TRANSFER',['u_T','u_V']].to_dict())
print('AUDIT PASS')
