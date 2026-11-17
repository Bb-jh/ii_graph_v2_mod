#!/usr/bin/env python3
"""Locked-root RCH consistency audit; no model re-training, no U3 access.

Compares the ORIGINAL P1 point estimate (old null-corrected formula converted from
saved CE) and spectral recursive P2 root. Optional --rerun reproduces both probe
R values on identical root labels and source configuration.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, hashlib, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments' / 'rch_probe'))
sys.path.insert(0, str(ROOT / 'experiments' / 'rch_next_audit_v1'))
import run_rch_probe as core
import run_next_audit as nxt


def readrows(p):
    if not p.is_file(): raise FileNotFoundError(p)
    with p.open(newline='') as f: return list(csv.DictReader(f))


def root_row(path, pred):
    rows = [r for r in readrows(path) if pred(r)]
    if len(rows) != 1: raise ValueError(f'expected one row at {path}, got {len(rows)}')
    return rows[0]


def corrected_r(r, mod='T'):
    den=float(r[f'den_{mod}']); ce=float(r[f'ce_model_{mod}']); return (den-ce)/den


def score_row(tag, r):
    return dict(source=tag, Q_B_U2=float(r['Q_B_U2']),n_items=int(r['n_items']),
                n_items_R=int(r['n_items_R']),C_T=float(r['C_T']),
                ce_model_T=float(r['ce_model_T']),den_T=float(r['den_T']),
                R_T_corrected=corrected_r(r),R_V_corrected=corrected_r(r,'V'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--domain',choices=['Industrial','Musical'],default='Industrial')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--rerun',action='store_true',help='Re-evaluate exact frozen root with original P1 and P2 seeds; CPU intensive')
    ap.add_argument('--out-root',default='outputs/rch_root_consistency_v1')
    ap.add_argument('--p1-root',default='outputs/rch_precheck_reviewed_v2/run')
    ap.add_argument('--next-root',default='outputs/rch_next_audit_v1/run')
    a=ap.parse_args()
    os.chdir(ROOT)
    name=f'{a.domain}_seed{a.seed}'
    p1=ROOT/a.p1_root/name
    nxtp=ROOT/a.next_root/name
    dest=ROOT/a.out_root/name
    dest.mkdir(parents=True,exist_ok=True)
    p1r=root_row(p1/'p1_candidates.csv',lambda r:r.get('name')=='behavior_spectral_k4_s0')
    p2r=root_row(nxtp/'spectral_p2_splits.csv',lambda r:r.get('depth')=='1' and r.get('parent')=='0')
    summary=[score_row('P1_saved',p1r),score_row('P2_saved',p2r)]
    rootcache=nxtp/'root_codes.npz'
    if not rootcache.exists():raise FileNotFoundError(f'Frozen root assignments required: {rootcache}')
    with np.load(rootcache) as z:
        y=np.array(z['s0'],dtype=np.int32)
    diag={'domain':a.domain,'seed':a.seed,'root':'behavior_spectral_k4_s0',
          'frozen_root_sha256':hashlib.sha256(y.tobytes()).hexdigest(),
          'P1_seed_formula':"stable_seed(seed, 'p1', 'behavior_spectral_k4_s0')",
          'P2_seed_formula':"stable_seed(seed, 'p2', 1, 0)",
          'P1_probe_seed':int(core.stable_seed(a.seed,'p1','behavior_spectral_k4_s0')),
          'P2_probe_seed':int(core.stable_seed(a.seed,'p2',1,0)),
          'P1_and_P2_Q_diff':summary[1]['Q_B_U2']-summary[0]['Q_B_U2'],
          'P2_minus_P1_R_T':summary[1]['R_T_corrected']-summary[0]['R_T_corrected'],
          'same_n_items_R':summary[0]['n_items_R']==summary[1]['n_items_R'],
          'same_C_T':summary[0]['C_T']==summary[1]['C_T'],
          'same_den_T':abs(summary[0]['den_T']-summary[1]['den_T'])<1e-10,
          'interpretation':'Probe seeds differ by design; matching Q alone does not prove assignments identical. Optional --rerun checks both seeds on the same frozen assignments.',
          'do_not_use_u3_for_selection':True}
    print('=== EXISTING RESULT CHECK ===')
    for r in summary:print(r)
    print('seed P1:',diag['P1_probe_seed'],'seed P2:',diag['P2_probe_seed'])
    if a.rerun:
        print('Loading dataset and verifying root edges...',flush=True)
        ns=core.parse_args(['--domain',a.domain,'--seed',str(a.seed),'--mode','run'])
        dat=core.load_server_data(ns)
        items=np.asarray(dat['train_items'])
        if len(y)!=len(items):raise ValueError('Cached root length != current U1 train_items length')
        aa,bb,ww=core.local_edges(dat['edges'][1],items,dat['n'])
        q=core.modularity(y,aa,bb,ww,4)['Q']
        diag['Q_B_U2_recomputed_from_frozen_root']=q
        diag['Q_matches_P1']=bool(np.isclose(q,summary[0]['Q_B_U2'],rtol=0,atol=1e-8))
        diag['Q_matches_P2']=bool(np.isclose(q,summary[1]['Q_B_U2'],rtol=0,atol=1e-8))
        if not diag['Q_matches_P1'] or not diag['Q_matches_P2']:
            raise RuntimeError('Frozen root Q differs from earlier P1/P2; inspect root cache and data first')
        cfg=dict(folds=5, permutations=5,bootstrap=500,c_grid=[.01,.1,1.],max_iter=300,
                 max_items=12000,min_items=60,min_child=5,eps=.05,delta_grid=[.01,.02,.05],
                 delta_rel=.02,delta_abs=.005)
        ff={'T':core.l2norm(dat['emb']['text'][0]), 'V':core.l2norm(dat['emb']['image'][0])}
        # use same evaluation function, same root, only changing evaluation random seed
        for tag,probe_seed in [('P1_rerun',diag['P1_probe_seed']),('P2_rerun',diag['P2_probe_seed'])]:
            print('Evaluating',tag,'(this can take minutes)...',flush=True)
            t=time.time()
            r=core.evaluate_split({'name':tag},items,y,dat['edges'],dat['n'],ff,cfg,probe_seed)
            if r.get('status')!='ok':raise RuntimeError(f'{tag}: {r.get("status")}')
            sr=score_row(tag,r)
            sr['seconds']=round(time.time()-t,1)
            summary.append(sr)
            print(sr,flush=True)
        diff1=abs(summary[2]['R_T_corrected']-summary[0]['R_T_corrected'])
        diff2=abs(summary[3]['R_T_corrected']-summary[1]['R_T_corrected'])
        diag['P1_rerun_vs_saved_R_delta']=diff1
        diag['P2_rerun_vs_saved_R_delta']=diff2
        diag['probe_reproduction_within_1e_minus_5']=bool(diff1<1e-5 and diff2<1e-5)
        diag['frozen_root_same_under_both_evaluations']=True
        print('P1 reproduction error:',diff1,'P2 reproduction error:',diff2)
        print('Reproduced?',diag['probe_reproduction_within_1e_minus_5'])
    keys=list(dict.fromkeys(k for r in summary for k in r))
    with (dest/'root_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader();writer.writerows(summary)
    (dest/'root_check.json').write_text(json.dumps(diag,ensure_ascii=False,indent=2))
    print('ROOT CHECK COMPLETE:',dest,flush=True)

if __name__=='__main__':main()
