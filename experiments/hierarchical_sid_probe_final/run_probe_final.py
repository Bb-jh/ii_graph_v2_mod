#!/usr/bin/env python3
"""Exp3-B: train-only CV + repeated shuffled-label null for hard SID-prefix probes.

This is a transductive, item-heldout diagnostic, NOT a genuine cold-item test,
MI estimator, or downstream generative recommendation evaluation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


def csv_save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('EMPTY\n')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def features(codes, depth, k):
    n = len(codes)
    cols = (np.arange(depth)[None, :] * k + codes[:, :depth]).ravel()
    rows = np.repeat(np.arange(n), depth)
    return sparse.csr_matrix((np.ones(n * depth, np.float32), (rows, cols)), shape=(n, depth*k))


def make_model(kind, alpha, seed, max_iter):
    if kind == 'nb':
        return MultinomialNB(alpha=alpha)
    if kind == 'sgd':
        return SGDClassifier(loss='log_loss', alpha=alpha, max_iter=max_iter,
                             tol=1e-3, random_state=seed, n_jobs=1)
    raise ValueError(kind)


def target_losses(model, X, y):
    """Cross-entropy per item; unseen test labels get a documented floor of 1e-12."""
    classes = model.classes_
    logits = model.predict_proba(X)
    pos = np.searchsorted(classes, y)
    valid = pos < len(classes)
    valid[valid] &= classes[pos[valid]] == y[valid]
    prob = np.full(len(y), 1e-12, dtype=np.float64)
    ix = np.flatnonzero(valid)
    prob[ix] = np.maximum(logits[ix, pos[ix]], 1e-12)
    return -np.log(prob)


def prior_losses(y_train, y_test, smoothing=1.0):
    classes, counts = np.unique(y_train, return_counts=True)
    pos = np.searchsorted(classes, y_test)
    seen = pos < len(classes)
    seen[seen] &= classes[pos[seen]] == y_test[seen]
    # Reserve one unseen class bucket so this evaluates consistently when
    # held-out classes are absent from the probe train partition.
    probs = (counts + smoothing) / (len(y_train) + smoothing*(len(classes)+1))
    out = np.full(len(y_test), 1.0/(len(y_train)+smoothing*(len(classes)+1)))
    idx = np.flatnonzero(seen)
    out[idx] = probs[pos[idx]]
    return -np.log(np.maximum(out, 1e-12))


def choose_alpha(kind, X, y, seed, max_iter):
    grid = [0.1, 1., 10., 100.] if kind == 'nb' else [1e-3, 1e-2, 1e-1]
    ix = np.arange(len(y))
    _, counts = np.unique(y, return_counts=True)
    strat = y if len(counts) > 1 and counts.min() >= 2 and len(ix)//5 >= len(counts) else None
    tr, va = train_test_split(ix, test_size=.2, random_state=seed, stratify=strat)
    rows = []
    for alpha in grid:
        mdl = make_model(kind, alpha, seed, max_iter)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mdl.fit(X[tr], y[tr])
            loss = float(target_losses(mdl, X[va], y[va]).mean())
        except Exception as e:
            print('CV WARNING', kind, alpha, str(e), flush=True)
            loss = float('inf')
        rows.append((alpha, loss))
    valid = [(a, ce) for a, ce in rows if np.isfinite(ce)]
    if not valid:
        raise RuntimeError(f'All CV choices failed for {kind}')
    return min(valid, key=lambda a: (a[1], a[0]))[0], rows


def run(a):
    root = Path(a.project_root).resolve()
    src = root / a.results_dir / f'{a.domain}_K{a.k}_seed{a.seed}'
    dest = root / a.output_dir / f'{a.domain}_K{a.k}_seed{a.seed}'
    if not src.exists():
        raise FileNotFoundError(src)
    if dest.exists():
        raise FileExistsError(f'Output already exists, refusing overwrite: {dest}')
    dest.mkdir(parents=True)
    labels = np.load(src / 'u1_hierarchy_labels.npy')
    held = np.load(src / 'probe_holdout_mask.npy').astype(bool)
    if labels.ndim != 2 or held.ndim != 1 or len(labels) != len(held):
        raise ValueError('Label/mask shapes differ')
    methods = [x.strip() for x in a.methods.split(',') if x.strip()]
    probes = [x.strip() for x in a.probes.split(',') if x.strip()]
    rows, cvs, pairs, alignments = [], [], [], []
    per_item = {}
    for mod in ['text', 'image']:
        for method in methods:
            path = src / mod / f'{method}_codes.npy'
            if not path.is_file():
                print('SKIP (codes missing)', path, flush=True)
                continue
            codes = np.load(path, mmap_mode='r')
            if codes.shape != labels.shape or np.any(codes < 0) or np.any(codes >= a.k):
                raise ValueError(f'Code shape/range mismatch: {path}')
            alignments.append(dict(modality=mod, method=method, file=str(path),
                sha256=hashlib.sha256(np.asarray(codes).tobytes()).hexdigest(),
                n_items=len(codes), depth=labels.shape[1]))
            for level in range(1, labels.shape[1]+1):
                tr = np.flatnonzero((labels[:,level-1] >= 0) & ~held)
                te = np.flatnonzero((labels[:,level-1] >= 0) & held)
                if len(tr) < 10 or len(te) < 10:
                    print('SKIP small probe', mod, method, level, flush=True)
                    continue
                ytr = labels[tr,level-1].astype(np.int32)
                yte = labels[te,level-1].astype(np.int32)
                if len(np.unique(ytr)) < 2:
                    print('SKIP single train label', mod, method, level, flush=True)
                    continue
                Xtr = features(np.asarray(codes[tr]), level, a.k)
                Xte = features(np.asarray(codes[te]), level, a.k)
                p_loss = prior_losses(ytr, yte)
                # Shared shuffle index across methods within each (modality,level,probe).
                # Train partition and item order are also shared across all methods.
                for probe in probes:
                    alpha, search = choose_alpha(probe, Xtr, ytr,
                                                  a.seed + 100*level, a.sgd_max_iter)
                    for trial_alpha, cv_ce in search:
                        cvs.append(dict(modality=mod,method=method,level=level,probe=probe,
                                        alpha=trial_alpha,cv_ce=cv_ce,selected=(trial_alpha==alpha)))
                    model = make_model(probe, alpha, a.seed, a.sgd_max_iter)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        model.fit(Xtr,ytr)
                    real = target_losses(model, Xte, yte)
                    null_losses = []
                    for p in range(a.permutations):
                        rng = np.random.default_rng(a.seed + 180011 + p + 1000*level + (mod=='image')*10000)
                        shuffled_y = rng.permutation(ytr)
                        nullmdl = make_model(probe, alpha, a.seed + p, a.sgd_max_iter)
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            nullmdl.fit(Xtr, shuffled_y)
                        null_losses.append(target_losses(nullmdl,Xte,yte))
                    null_avg = np.stack(null_losses).mean(axis=0)
                    # Real gain vs prior: prior CE - real CE.
                    # Null correction: real gain - null gain = null CE - real CE.
                    adjusted = null_avg - real
                    per_item[(mod,level,method,probe)] = (te,adjusted)
                    rows.append(dict(domain=a.domain,seed=a.seed,modality=mod,method=method,
                        level=level,probe=probe,alpha=alpha,n_train=len(tr),n_test=len(te),
                        n_train_labels=len(np.unique(ytr)),n_unseen_test_labels=len(np.setdiff1d(np.unique(yte),np.unique(ytr))),
                        prior_ce=float(p_loss.mean()),real_ce=float(real.mean()),
                        shuffled_ce_mean=float(null_avg.mean()),
                        real_gain=float((p_loss-real).mean()),
                        shuffled_gain=float((p_loss-null_avg).mean()),
                        null_corrected_gain=float(adjusted.mean()),
                        null_ce_sd_across_permutations=float(np.std([z.mean() for z in null_losses],ddof=1)) if a.permutations>1 else 0.,
                        permutations=a.permutations, note='permutation-adjusted probe score, NOT MI'))
                    print('RESULT', mod, method, 'L'+str(level), probe,
                          'alpha',alpha,'real_gain',round(float((p_loss-real).mean()),5),
                          'null_corrected',round(float(adjusted.mean()),5),flush=True)
    for mod in ['text','image']:
        for level in range(1,labels.shape[1]+1):
            for probe in probes:
                l = per_item.get((mod,level,'label_mi',probe))
                r = per_item.get((mod,level,'contrastive',probe))
                if l is None or r is None:continue
                il, gl = l; ir, gr = r
                if not np.array_equal(il,ir):raise AssertionError('F1/Contrastive heldout item order mismatch')
                d = gl-gr
                rng = np.random.default_rng(a.seed+19909+level+(mod=='image')*50)
                boot = np.empty(a.bootstrap)
                for i in range(a.bootstrap):
                    ix = rng.integers(0,len(d),size=len(d))
                    boot[i]=d[ix].mean()
                lo,hi=np.quantile(boot,[.025,.975])
                pairs.append(dict(modality=mod,level=level,probe=probe,lhs='label_mi',rhs='contrastive',
                    observed_null_corrected_gain_diff=float(d.mean()),ci_percentile_lo=float(lo),
                    ci_percentile_hi=float(hi),n_heldout_items=len(d),
                    ci_unit='paired heldout item; fitted models and finite null draws fixed'))
    csv_save(dest/'probe_final_summary.csv', rows)
    csv_save(dest/'probe_cv.csv', cvs)
    csv_save(dest/'f1_vs_contrastive_ci.csv', pairs)
    csv_save(dest/'input_audit.csv', alignments)
    (dest/'manifest.json').write_text(json.dumps({
        'source':str(src),'protocol':'Exp3-B terminal exploratory probe',
        'probe_prefix':'L1:C1; L2:C1,C2; L3:C1,C2,C3',
        'regularization_selection':'80/20 split INSIDE U1 probe train only',
        'permutations':a.permutations,'bootstrap':a.bootstrap,
        'caveats':['Transductive pseudo labels generated from U1 graph including heldout probe items.',
        'Fitted models selected without accessing probe heldout data, but protocol developed after earlier heldout observations; exploratory only.',
        'Permutation correction is not unbiased mutual information and does not guarantee overfit bias cancellation.',
        'Bootstrap holds model fits and permutation random draws fixed; uncertainty across training seeds omitted.',
        'Code order is checked by shape/range/hash, NOT independently verified against an external item-ID mapping.',
        'Cold-start HR/NDCG must be established by separate inductive split and downstream recommender.']},indent=2))
    print('AUDIT SUCCESS', dest, 'rows',len(rows),'paired',len(pairs),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain',choices=['Industrial','Musical'],required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--k',type=int,default=256)
    p.add_argument('--project-root',default='.')
    p.add_argument('--results-dir',default='outputs/hierarchical_sid_exp3_v12/full')
    p.add_argument('--output-dir',default='outputs/hierarchical_sid_exp3b_probe_final_v1')
    p.add_argument('--methods',default='label_mi,contrastive,rq')
    p.add_argument('--probes',default='nb,sgd')
    p.add_argument('--sgd-max-iter',type=int,default=80)
    p.add_argument('--permutations',type=int,default=5)
    p.add_argument('--bootstrap',type=int,default=500)
    args=p.parse_args()
    if args.permutations < 2 or args.bootstrap < 100:raise ValueError('Use at least 2 permutations and 100 bootstraps')
    run(args)
