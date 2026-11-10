#!/usr/bin/env python3
"""Read-only audit of saved Exp2 V2 hard assignments and entropy confounding.
No U3 model selection, retraining, or mutation of the source directory.
"""
import argparse, csv, json, hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT=Path(__file__).resolve().parents[2]
METHODS=('rq','infonce','code_mi','kmeans')
MODALITIES=('text','image')

def ent_of_codes(a,k):
    count=np.bincount(a.astype(np.int64),minlength=k).astype(float)
    p=count/count.sum()
    h=float(-(p[p>0]*np.log(p[p>0])).sum())
    return h,int((count>0).sum()),float(np.exp(h)),float(p.max())

def load_summary(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def checksum(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--domain',choices=('Industrial','Musical'),required=True)
    p.add_argument('--source-root',default='outputs/discrete_mi_level1_v2_portable')
    p.add_argument('--out-root',default='outputs/discrete_mi_entropy_audit_v1')
    p.add_argument('--entropy-tolerance-pct',type=float,default=1.0)
    args=p.parse_args()
    src=Path(args.source_root).expanduser()
    if not src.is_absolute():src=ROOT/src
    folder=src/f'{args.domain}_K256_seed42'
    if not (folder/'DONE.json').exists():raise FileNotFoundError('Completed V2 run missing: '+str(folder))
    summary=load_summary(folder/'summary.csv')
    conf=json.loads((folder/'config.json').read_text())
    k=int(conf['k'])
    rows=[];pairs=[];errors=[]
    for mod in MODALITIES:
        filtered={r['method']:r for r in summary if r['modality']==mod}
        if set(filtered)!=set(METHODS):raise RuntimeError(f'{mod}: required methods missing: {set(METHODS)-set(filtered)}')
        arrays={}
        for method in METHODS:
            path=folder/mod/f'{method}_codes.npy'
            if not path.is_file():raise FileNotFoundError(path)
            a=np.load(path,allow_pickle=False)
            if a.ndim!=1 or not np.issubdtype(a.dtype,np.integer) or a.min()<0 or a.max()>=k:raise ValueError('Invalid codes: '+str(path))
            arrays[method]=a
        if len(set(len(a) for a in arrays.values()))!=1:raise ValueError(f'{mod} code array lengths differ')
        # V2 calculates code entropy over items eligible in BOTH modalities, not all mapping ids.
        # Reconstruct the exact eligible mask by importing the same core data loader as V2.
        # Torch is deliberately avoided; numpy/sklearn only.
        import sys
        sys.path.insert(0,str(ROOT/'experiments'/'discrete_mi_portable_v2'))
        import offdiag_core as core
        from argparse import Namespace
        # Mirrors original path resolver without importing torch-heavy run_level1_v2.py
        dataset,seqname=core.DOMAINS[args.domain]
        embedding_candidates=[
            ROOT/'artifacts'/'embeddings'/dataset/'full'/'clip_vit_b32.npz',
            ROOT/'shared_data'/dataset/'clip_vit_b32.npz',
            ROOT.parent/'artifacts'/'embeddings'/dataset/'full'/'clip_vit_b32.npz',
        ]
        cfg_emb=conf.get('embedding_path')
        if cfg_emb:embedding_candidates.insert(0,Path(cfg_emb).expanduser())
        mapping_candidates=[ROOT/'shared_data'/dataset/'5core_v2'/'item_mapping.tsv',ROOT.parent/'shared_data'/dataset/'5core_v2'/'item_mapping.tsv']
        if conf.get('mapping_path'):mapping_candidates.insert(0,Path(conf['mapping_path']).expanduser())
        mp=next((f for f in mapping_candidates if f.is_file()),None)
        ep=next((f for f in embedding_candidates if f.is_file()),None)
        if mp is None or ep is None:raise FileNotFoundError(f'Mapping or embedding missing: mapping={mp}, embedding={ep}. Use actual original data paths.')
        ids=core.load_mapping(mp)
        em=core.load_embeddings(ep,ids)
        eligible=np.flatnonzero(em['text'][1] & em['image'][1]).astype(np.int32)
        if len(ids)!=len(arrays['rq']):raise AssertionError('Mapping item count differs from stored code arrays')
        if mod=='text':print('ELIGIBLE',len(eligible),'of',len(ids),'mapping=',mp,'embedding=',ep,flush=True)
        for method in METHODS:
            a=arrays[method][eligible]
            h,nused,ppl,maxshare=ent_of_codes(a,k)
            r=filtered[method]
            j=float(r['code_mi'])
            old_h=float(r['code_entropy'])
            if abs(old_h-h)>1e-6:errors.append(f'{mod}/{method} entropy disagreement: audit {h:.10f} summary {old_h:.10f}')
            old_usage=int(float(r['n_used_codes']))
            if nused!=old_usage:errors.append(f'{mod}/{method} usage disagreement {nused} vs {old_usage}')
            rows.append(dict(domain=args.domain,modality=mod,method=method,k=k,n_eligible=len(eligible),used_codes=nused,entropy_h=h,entropy_h_over_logk=h/np.log(k),perplexity=ppl,max_code_share=maxshare,J=j,J_over_H=(j/h if h>0 else float('nan')),J_over_logK=j/np.log(k),delta_off=float(r['delta_off']),transition_auc=float(r['transition_auc']),code_sha256=checksum(arrays[method])))
        for lhs,rhs in [('code_mi','kmeans'),('code_mi','infonce'),('infonce','kmeans'),('rq','kmeans')]:
            a=arrays[lhs][eligible];b=arrays[rhs][eligible]
            equal=float(np.mean(a==b))
            ari=float(adjusted_rand_score(a,b))
            nmi=float(normalized_mutual_info_score(a,b,average_method='arithmetic'))
            ra=filtered[lhs];rb=filtered[rhs]
            ja=float(ra['code_mi']);jb=float(rb['code_mi']);oa=float(ra['delta_off']);ob=float(rb['delta_off'])
            ha=ent_of_codes(a,k)[0];hb=ent_of_codes(b,k)[0]
            pairs.append(dict(domain=args.domain,modality=mod,lhs=lhs,rhs=rhs,exact_label_agreement=equal,ARI=ari,NMI=nmi,J_diff=ja-jb,J_over_H_diff=ja/ha-jb/hb,delta_off_diff=oa-ob,delta_off_values_equal_4dp=(round(oa,4)==round(ob,4)),entropy_diff_pct_logk=100*(ha-hb)/np.log(k),entropy_diff_pct_relative=100*(ha-hb)/hb if hb>0 else np.nan))
    if errors:
        print('AUDIT FAILED: source summary is inconsistent with codes',flush=True)
        for e in errors:print(' -',e,flush=True)
        raise RuntimeError('Do not interpret these results; reconcile data paths and masks.')
    out=Path(args.out_root).expanduser()
    if not out.is_absolute():out=ROOT/out
    dest=out/f'{args.domain}_K256_seed42'
    if dest.exists():raise FileExistsError('Refusing to overwrite previous audit: '+str(dest))
    dest.mkdir(parents=True)
    for name,data in [('entropy_summary.csv',rows),('assignment_comparisons.csv',pairs)]:
        with (dest/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    km={r['modality']:r for r in rows if r['method']=='kmeans'}
    status=[]
    for r in rows:
        if r['method']=='kmeans':continue
        diff=abs(r['entropy_h']-km[r['modality']]['entropy_h'])/km[r['modality']]['entropy_h']*100
        status.append(dict(modality=r['modality'],method=r['method'],entropy_diff_from_kmeans_pct=diff,within_predefined_1pct=(diff<=args.entropy_tolerance_pct)))
    with (dest/'entropy_match_diagnostic.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(status[0]));w.writeheader();w.writerows(status)
    (dest/'AUDIT_INFO.json').write_text(json.dumps(dict(source=str(folder),k=k,seed=conf['seed'],eligibility='V2 intersection mask (text & image)',scope='post-hoc diagnostic; not entropy matched retraining',entropy_tolerance_pct=args.entropy_tolerance_pct,source_untouched=True),indent=2)+'\n')
    print('AUDIT SUCCESS',dest,flush=True)
    for r in rows:print('ENTROPY',r['modality'],r['method'],'H',round(r['entropy_h'],5),'J/H',round(r['J_over_H'],5),'used',r['used_codes'],'delta_off',round(r['delta_off'],7),flush=True)
    for r in pairs:
        if r['lhs']=='code_mi' and r['rhs']=='kmeans':print('CODE AGREEMENT',r['modality'],'exact',round(r['exact_label_agreement'],5),'ARI',round(r['ARI'],5),'NMI',round(r['NMI'],5),'delta_off_diff',round(r['delta_off_diff'],9),flush=True)

if __name__=='__main__':main()
