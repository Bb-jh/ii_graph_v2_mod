#!/usr/bin/env python3
"""One-command Experiments 3 V1.2 + 3-B final reproducibility runner.

Never opens U3 for unmatched arms. Existing completed runs are reused only
when manifest/input checks pass; interrupted partial directories are not resumed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=Path(__file__).resolve().parent
RUN=ROOT/'experiments/hierarchical_sid_v12/run_exp3_v12.py'
PROBE=ROOT/'experiments/hierarchical_sid_probe_final/run_probe_final.py'


def cmdrun(args,log_path,env):
    log_path.parent.mkdir(parents=True,exist_ok=True)
    print('\nRUN:', ' '.join(map(str,args)),flush=True)
    with log_path.open('w',encoding='utf8') as log:
        child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in child.stdout:
                sys.stdout.write(line); sys.stdout.flush();log.write(line);log.flush()
            code=child.wait()
        except KeyboardInterrupt:
            child.terminate();child.wait()
            raise
    if code:raise RuntimeError(f'Command failed with exit {code}. Log: {log_path}')


def assert_sources():
    expected=json.loads((SCRIPTS/'SOURCE_SHA256.json').read_text())
    incorrect=[]
    for name,digest in expected.items():
        p=ROOT/name
        now=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else '<missing>'
        if now!=digest:incorrect.append((name,now))
    if incorrect:
        text='\n'.join('  '+name+' (actual SHA256 '+actual+')' for name,actual in incorrect)
        raise RuntimeError('Reproducibility check failed. Existing dependency files are missing or differ from bundle:\n'+text+'\nUse a clean Git worktree or compare local edits before replacing files. Do not blindly overwrite personal changes.')
    for name in ('experiments/motivation/run_c1_pilot.py','experiments/motivation/run_motivation_cpu_v2.py'):
        p=ROOT/name
        if not p.is_file(): raise RuntimeError('Missing REAL project dependency: '+name+'; push actual source from glis2, not synthetic stubs')
        if 'TEST STUB' in p.read_text(errors='replace')[:500]: raise RuntimeError('Detected synthetic test stub in '+name+'; restore the original source from the real Git repository')
    print('SOURCE CHECK PASS — all '+str(len(expected))+' bundled Python sources match and real motivation modules are present',flush=True)


def assert_complete(results,domain,seed,k):
    donefile=results/'DONE.json'
    if not donefile.is_file():raise RuntimeError('Existing result directory is incomplete: '+str(results)+'\nDo not delete it; move it aside after inspection or restart in a clean worktree.')
    done=json.loads(donefile.read_text())
    if done.get('status') not in ('success','partial_matching'):
        raise RuntimeError('Unexpected completed status: '+str(done))
    if not (results/'u2_candidates.csv').is_file() or not (results/'summary.csv').is_file() or not (results/'u1_hierarchy_labels.npy').is_file():
        raise RuntimeError('Results incomplete despite DONE.json: '+str(results))
    config=json.loads((results/'config.json').read_text())
    fixed={'domain':domain,'seed':seed,'k':k,'depth':3,'steps':1200,'batch':256,'tolerance_pct':1.0,'beta_grid':'1,3,10','lambda_grid':[0.1,1.0]}
    for key,value in fixed.items():
        actual=config.get(key)
        if key=='lambda_grid' and isinstance(actual,list): actual=[float(x) for x in actual]
        if actual!=value:raise RuntimeError(f'Existing run has a different setting {key}: {actual!r} vs {value!r}. Refusing to silently reuse.')
    if config.get('evaluate_unmatched') is True or config.get('modality')!='both':
        raise RuntimeError('Existing results are not a clean both-modality matched-only run')
    return done


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain',choices=['Industrial','Musical'],required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--check-only',action='store_true',help='Resolve input paths and check fixed sources only')
    p.add_argument('--skip-probe',action='store_true',help='Only when running a preflight, not for final shared results')
    p.add_argument('--probe-permutations',type=int,default=5)
    p.add_argument('--probe-bootstrap',type=int,default=500)
    a=p.parse_args()
    if a.seed !=42:raise SystemExit('Final single-run protocol is frozen to seed 42. For additional seeds create a separately versioned protocol.')
    env=os.environ.copy()
    env.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    env.setdefault('OMP_NUM_THREADS','4')
    env.setdefault('OPENBLAS_NUM_THREADS','4')
    env.setdefault('MKL_NUM_THREADS','4')
    assert_sources()
    dataset=f'{a.domain}_K256_seed{a.seed}'
    logs=ROOT/'outputs/hierarchical_sid_exp3_final_logs'/dataset
    print('PROJECT ROOT',ROOT)
    print('DOMAIN',a.domain,'SEED',a.seed,'DEVICE',a.device,'VISIBLE GPUS',env.get('CUDA_VISIBLE_DEVICES','<not set>'))
    check=[sys.executable,'-u',str(RUN),'--domain',a.domain,'--seed',str(a.seed),'--mode','check','--device',a.device]
    cmdrun(check,logs/'check.log',env)
    if a.check_only:
        print('FINAL PREFLIGHT PASS');return
    result=ROOT/'outputs/hierarchical_sid_exp3_v12/full'/dataset
    if result.exists():
        status=assert_complete(result,a.domain,a.seed,256)
        print('REUSING COMPLETE V1.2 RESULTS',status,flush=True)
    else:
        print('BEGIN EXP3 V1.2 FINAL FULL RUN — source unchanged, U2 matching then U3',flush=True)
        cmdrun([sys.executable,'-u',str(RUN),'--domain',a.domain,'--seed',str(a.seed),'--mode','run','--device',a.device],logs/'exp3_full.log',env)
        status=assert_complete(result,a.domain,a.seed,256)
    if a.skip_probe:
        print('SKIPPED PROBE BY REQUEST');return
    probe=ROOT/'outputs/hierarchical_sid_exp3b_probe_final_v1'/dataset
    if probe.exists():
        required=['probe_final_summary.csv','f1_vs_contrastive_ci.csv','manifest.json','input_audit.csv']
        if any(not (probe/name).is_file() for name in required):
            raise RuntimeError('Probe output exists but incomplete: '+str(probe))
        manifest=json.loads((probe/'manifest.json').read_text())
        if manifest.get('permutations')!=a.probe_permutations or manifest.get('bootstrap')!=a.probe_bootstrap:
            raise RuntimeError('Probe output used different bootstrap/null settings; refusing silent reuse')
        print('REUSING COMPLETE EXP3-B PROBE',probe,flush=True)
    else:
        cmdrun([sys.executable,'-u',str(PROBE),'--domain',a.domain,'--seed',str(a.seed),'--k','256','--project-root',str(ROOT),'--permutations',str(a.probe_permutations),'--bootstrap',str(a.probe_bootstrap)],logs/'exp3b_probe.log',env)
    import csv
    with (result/'summary.csv').open(newline='') as f: results=list(csv.DictReader(f))
    with (probe/'probe_final_summary.csv').open(newline='') as f: probes=list(csv.DictReader(f))
    selection={m:{r['method'] for r in results if r['modality']==m and r.get('match_status')=='matched'} for m in ('text','image')}
    critical={'rq','contrastive','label_mi','pair_code_mi','kmeans'}
    missing={m:sorted(critical-selection[m]) for m in selection}
    print('\n=== FINAL REPORT ===')
    print('Exp3 DONE status:',status.get('status'))
    print('Matched by modality:', {m:sorted(selection[m]) for m in selection})
    print('Missing core matched arms:',missing)
    print('InfoNCE matching:',{m:'infonce' in selection[m] for m in selection})
    print('Exp3 method rows:',len(results),'Exp3B probe rows:',len(probes))
    print('Result:',result)
    print('Probe:',probe)
    print('Logs:',logs)
    print('PROTOCOL COMPLETE. Matching failure is an observed result, not a reason to retune after opening U3.')
    report={'bundle':'EXP3_V12_EXP3B_FINAL_FROZEN_V1','domain':a.domain,'seed':a.seed,'date_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':json.loads((SCRIPTS/'SOURCE_SHA256.json').read_text()),'exp3_status':status,'matched_methods':{m:sorted(selection[m]) for m in selection},'missing_core_matched':missing,'exp3_results':str(result),'exp3b_results':str(probe),'warning':'Exp3 exploratory; probe transductive; Exp4 Conditional MI is a separate, not-included experiment.'}
    logs.mkdir(parents=True,exist_ok=True)
    (logs/'FINAL_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('FINAL RUN FAILED:',repr(e),file=sys.stderr,flush=True)
        raise
