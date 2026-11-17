#!/usr/bin/env python3
"""RCH pre-check P1/P2: does a collaborative split exist that is both
behavior-meaningful (Q_B) and content-realizable on held-out items (R)?

This is a teacher-construction diagnostic. It trains NO tokenizer and NO recommender.

Pre-registered definitions (frozen before any run; see manifest.json)
----------------------------------------------------------------------
Item universe   U1 train items from Exp2/Exp3 make_data (same seed, same split).
Split S         A parent item set P partitioned into K>=2 children (local labels 0..K-1).
Q_B(S)          Weighted Newman modularity of S on HELD-OUT U2 co-purchase edges whose two
                endpoints are both in P (edge weight = W_B normalized co-occurrence).
                Q_B on U1 edges is reported only as an overfitting check.
R_m(S)          Inductive content realizability for modality m in {T, V}.
                Item-level K-fold cross-fitting inside P. Linear multinomial logistic
                probe on standardized CLIP embeddings; C chosen on an inner split of the
                first training fold only, then frozen for all folds and nulls of S.
                ce_model_i : probe cross-entropy of held-out item i
                ce_null_i  : same probe trained on PERMUTED training labels (child sizes
                             preserved = same tree shape, item-child correspondence broken),
                             averaged over permutations
                ce_prior_i : Laplace-smoothed training-fold child frequencies
                R  = mean(ce_null - ce_model) / mean(ce_prior)   (normalized, null-corrected)
                dCE = mean(ce_null - ce_model)                  (absolute, nats)
R_{V|T}(S)      Conditional realizability of image given text: probe on [T,V] vs the same
                probe with image rows permuted among training items.
                R  = mean(ce_TVnull - ce_TV) / mean(ce_T)
Pass(S, m)      denominator >= eps (else 'undefined'), bootstrap 95% CI lower bound of R
                > delta_rel (primary 0.02), and dCE > delta_abs (0.005 nats).
P1 Go           >=1 behavior/mixed candidate with Pass(T) and Q_B_U2 >= 0.5 * max Q_B_U2
                of behavior candidates.
P2 Go           root split passes (T) AND >= 30% of items have >= 2 consecutive passing
                splits along their path (T).
Both Go rules are advisory; the CSVs hold everything needed to apply other thresholds.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = 'RCH_PRECHECK_P1P2_V2_REVIEWED'
EXP3_DIR = Path('outputs') / 'hierarchical_sid_exp3_v12' / 'full'


# ----------------------------------------------------------------------------- args
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--domain', choices=['Industrial', 'Musical'])
    p.add_argument('--mode', choices=['check', 'smoke', 'run', 'self-test'], default='check')
    p.add_argument('--parts', default='p2,p1', help='Comma list of p1,p2')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-root', default='outputs/rch_precheck_v1')
    p.add_argument('--hierarchy-path', default=None, help='Override path to u1_hierarchy_labels.npy')
    p.add_argument('--hierarchy-branches', default='16,4,4', help='Used only when rebuilding (Exp3 default)')
    p.add_argument('--sketch-dim', type=int, default=64, help='Behavior sketch dim (Exp3 default)')
    p.add_argument('--eval-positives', type=int, default=30000, help='Passed to make_data; keeps data dict identical to Exp3')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--permutations', type=int, default=5)
    p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--c-grid', default='0.01,0.1,1.0')
    p.add_argument('--max-iter', type=int, default=300)
    p.add_argument('--max-items', type=int, default=12000, help='Per-split item cap for R (Q_B always uses all items)')
    p.add_argument('--min-items', type=int, default=60)
    p.add_argument('--min-child', type=int, default=5)
    p.add_argument('--eps-denominator', type=float, default=0.05, help='Min denominator (nats) for normalized R')
    p.add_argument('--delta-rel-grid', default='0.01,0.02,0.05')
    p.add_argument('--delta-rel', type=float, default=0.02, help='Primary threshold')
    p.add_argument('--delta-abs', type=float, default=0.005)
    p.add_argument('--p1-ks', default='4,8,16,32')
    p.add_argument('--p1-seeds', type=int, default=3)
    p.add_argument('--spectral-dim', type=int, default=40, help='Number of eigenvectors for true U1 graph spectral candidates')
    p.add_argument('--no-spectral', action='store_true', help='Disable exact spectral candidates for large-graph pilot; recorded in manifest')
    p.add_argument('--p1-alphas', default='0.25,0.5,0.75', help='Behavior weight in mixed affinity')
    p.add_argument('--p1-mixed-ks', default='8,16')
    p.add_argument('--jobs', type=int, default=4)
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- utils
def l2norm(z):
    z = np.nan_to_num(np.asarray(z, dtype=np.float32), nan=0., posinf=0., neginf=0.)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-10)


def stable_seed(*parts):
    h = hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def save_csv(path, rows):
    import csv
    if not rows:
        Path(path).write_text('EMPTY\n')
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ----------------------------------------------------------------------------- behavior side
def behavior_sketch(e1, items, n, dim, seed):
    """Identical construction to Exp3 make_hierarchy: U1 weighted graph, sym-normalized,
    random features propagated over 1-hop + 2-hop, L2-normalized. Rows follow `items`."""
    src, dst, wt = e1[0], e1[1], e1[2]
    ix = np.full(n, -1, dtype=np.int64)
    ix[items] = np.arange(len(items))
    a, b = ix[src], ix[dst]
    good = (a >= 0) & (b >= 0)
    a, b, w = a[good], b[good], wt[good].astype(np.float32)
    m = len(items)
    mat = sparse.coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(m, m)).tocsr()
    mat.setdiag(0)
    mat.eliminate_zeros()
    deg = np.asarray(mat.sum(1)).ravel()
    inv = np.maximum(deg, 1e-7) ** -.5
    mat = sparse.diags(inv) @ mat @ sparse.diags(inv)
    rng = np.random.default_rng(seed + 311)
    f = rng.normal(size=(m, dim)).astype(np.float32)
    f = mat @ f
    f = f + mat @ f
    return l2norm(f)



def behavior_spectral(e1, items, n, dim):
    """True spectral embedding of the U1 normalized weighted adjacency (NOT diffusion-sketch K-means).

    Exactly one global eigenbasis per run; all K/seeds share it. Uses symmetric eigsh.
    This is an exploratory candidate generator, not equivalent to a CID-tree spectral teacher.
    """
    src, dst, wt = e1[0], e1[1], e1[2]
    ix = np.full(n, -1, dtype=np.int64)
    ix[items] = np.arange(len(items))
    a, b = ix[src], ix[dst]
    good = (a >= 0) & (b >= 0)
    a, b = a[good], b[good]
    w = np.asarray(wt, dtype=np.float64)[good]
    m = len(items)
    mat = sparse.coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(m, m)).tocsr()
    mat.setdiag(0)
    mat.eliminate_zeros()
    deg = np.asarray(mat.sum(axis=1)).ravel()
    dinv = 1. / np.sqrt(np.maximum(deg, 1e-12))
    mat = (sparse.diags(dinv) @ mat @ sparse.diags(dinv)).tocsr()
    nd = min(dim, m - 2)
    if nd < 2:
        raise ValueError('Not enough items for spectral embedding')
    evals, vecs = eigsh(mat, k=nd, which='LA', tol=1e-3, maxiter=max(1000, m // 5))
    idx = np.argsort(evals)[::-1]
    z = np.asarray(vecs[:, idx], dtype=np.float32)
    return l2norm(z)


def local_edges(edge, node_items, n):
    """Edges with both endpoints in node_items, re-indexed to local positions."""
    ix = np.full(n, -1, dtype=np.int64)
    ix[node_items] = np.arange(len(node_items))
    a, b = ix[edge[0]], ix[edge[1]]
    keep = (a >= 0) & (b >= 0)
    return a[keep], b[keep], np.asarray(edge[2], dtype=np.float64)[keep]


def modularity(y, a, b, w, k):
    """Weighted Newman modularity of partition y on undirected edges listed once."""
    W = float(w.sum())
    if W <= 0 or len(a) == 0:
        return dict(Q=float('nan'), within_frac=float('nan'), n_edges=int(len(a)), weight=W)
    same = y[a] == y[b]
    e_in = float(w[same].sum())
    deg = np.bincount(y[a], weights=w, minlength=k) + np.bincount(y[b], weights=w, minlength=k)
    q = e_in / W - float(((deg / (2 * W)) ** 2).sum())
    return dict(Q=q, within_frac=e_in / W, n_edges=int(len(a)), weight=W)


# ----------------------------------------------------------------------------- content probe
def make_folds(y, folds, seed):
    _, counts = np.unique(y, return_counts=True)
    if counts.min() >= folds:
        it = StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    else:
        it = KFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(y)))
    return [(tr, te) for tr, te in it]


def prior_ce(ytr, yte, k):
    p = (np.bincount(ytr, minlength=k) + 1.) / (len(ytr) + k)
    return -np.log(p[yte])


def fit_ce(Xtr, ytr, Xte, yte, k, C, max_iter, seed):
    """Per-item held-out CE of a multinomial logistic probe; classes mapped to all k."""
    if len(np.unique(ytr)) < 2:
        return prior_ce(ytr, yte, k)
    sc = StandardScaler().fit(Xtr)
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.simplefilter('ignore', UserWarning)
        model.fit(sc.transform(Xtr), ytr)
    proba = np.zeros((len(yte), k))
    proba[:, model.classes_] = model.predict_proba(sc.transform(Xte))
    proba += 1e-6
    proba /= proba.sum(1, keepdims=True)
    return -np.log(proba[np.arange(len(yte)), yte])


def select_c(X, y, tr, k, grid, max_iter, seed):
    """Inner 80/20 split of the FIRST training fold only; smaller C wins ties."""
    yy = y[tr]
    _, counts = np.unique(yy, return_counts=True)
    strat = yy if counts.min() >= 2 else None
    try:
        a, b = train_test_split(tr, test_size=.2, random_state=seed, stratify=strat)
    except ValueError:
        a, b = train_test_split(tr, test_size=.2, random_state=seed)
    scores = [(fit_ce(X[a], y[a], X[b], y[b], k, c, max_iter, seed).mean(), c) for c in grid]
    return min(scores, key=lambda s: (round(s[0], 6), s[1]))[1]


def boot_ratio(num, den, reps, seed):
    rng = np.random.default_rng(seed)
    n = len(num)
    vals = np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, n, size=n)
        vals[r] = num[i].mean() / max(den[i].mean(), 1e-12)
    return [float(x) for x in np.quantile(vals, [.025, .975])]


def realizability(feats, y, k, cfg, seed):
    """Returns R_T, R_V, R_V|T with CIs. `feats` = {'T': X, 'V': X} aligned with y."""
    folds = make_folds(y, cfg['folds'], seed)
    n = len(y)
    grid = cfg['c_grid']
    ce_prior = np.empty(n)
    for tr, te in folds:
        ce_prior[te] = prior_ce(y[tr], y[te], k)
    ce = {}
    out = {}
    for name in ('T', 'V'):
        X = feats[name]
        c = select_c(X, y, folds[0][0], k, grid, cfg['max_iter'], seed)
        model = np.empty(n)
        null = np.zeros(n)
        for f, (tr, te) in enumerate(folds):
            model[te] = fit_ce(X[tr], y[tr], X[te], y[te], k, c, cfg['max_iter'], seed)
            for p in range(cfg['permutations']):
                rng = np.random.default_rng(stable_seed(seed, name, f, p))
                null[te] += fit_ce(X[tr], rng.permutation(y[tr]), X[te], y[te], k, c, cfg['max_iter'], seed)
            null[te] /= cfg['permutations']
        ce[name] = model
        num = null - model
        den = ce_prior
        out[name] = dict(C=c, ce_prior=float(den.mean()), ce_model=float(model.mean()), ce_null=float(null.mean()),
                         dCE=float(num.mean()), denominator=float(den.mean()))
        out[name]['R'] = float(num.mean() / den.mean()) if den.mean() >= cfg['eps'] else float('nan')
        out[name]['R_ci'] = boot_ratio(num, den, cfg['bootstrap'], stable_seed(seed, name, 'boot')) \
            if den.mean() >= cfg['eps'] else [float('nan')] * 2
    # Incremental conditional predictive value: text-only CE vs joint text+image CE.
    # Shuffled-image null is a negative control, NOT a conditional MI estimator.
    X = np.concatenate([feats['T'], feats['V']], axis=1)
    dt = feats['T'].shape[1]
    c = select_c(X, y, folds[0][0], k, grid, cfg['max_iter'], seed)
    model = np.empty(n)
    null = np.zeros(n)
    for f, (tr, te) in enumerate(folds):
        model[te] = fit_ce(X[tr], y[tr], X[te], y[te], k, c, cfg['max_iter'], seed)
        for p in range(cfg['permutations']):
            rng = np.random.default_rng(stable_seed(seed, 'VT', f, p))
            xn = X[tr].copy()
            xn[:, dt:] = xn[rng.permutation(len(tr)), dt:]
            xte_null = X[te].copy()
            xte_null[:, dt:] = xte_null[rng.permutation(len(te)), dt:]
            null[te] += fit_ce(xn, y[tr], xte_null, y[te], k, c, cfg['max_iter'], seed)
        null[te] /= cfg['permutations']
    num = ce['T'] - model  # direct addition of image on held-out targets
    den = ce['T']
    out['V|T'] = dict(C=c, ce_prior=float(ce_prior.mean()), ce_model=float(model.mean()), ce_null=float(null.mean()),
                      dCE=float(num.mean()), deltaCE_vs_text=float(num.mean()),
                      shuffled_image_deltaCE=float((ce['T'] - null).mean()),
                      shuffled_image_control_gap=float((null - model).mean()), denominator=float(den.mean()))
    ok = den.mean() >= cfg['eps']
    out['V|T']['R'] = float(num.mean() / den.mean()) if ok else float('nan')
    out['V|T']['R_ci'] = boot_ratio(num, den, cfg['bootstrap'], stable_seed(seed, 'VT', 'boot')) if ok else [float('nan')] * 2
    return out


def passes(res, delta_rel, delta_abs):
    lo = res['R_ci'][0]
    return bool(np.isfinite(lo) and lo > delta_rel and res['dCE'] > delta_abs)


def evaluate_split(tag, node_items, y_global, edges, n, feats_all, cfg, seed):
    """One split: Q_B on U2 (and U1 check) over all node items, R on <= max_items items."""
    _, y = np.unique(y_global, return_inverse=True)
    k = int(y.max()) + 1
    sizes = np.bincount(y, minlength=k)
    row = dict(tag)
    row.update(n_items=int(len(node_items)), n_children=k, min_child=int(sizes.min()), max_child=int(sizes.max()))
    if len(node_items) < cfg['min_items'] or k < 2 or sizes.min() < cfg['min_child']:
        row['status'] = 'skipped_small'
        return row
    for name, e in (('U2', edges[1]), ('U1', edges[0])):
        a, b, w = local_edges(e, node_items, n)
        m = modularity(y, a, b, w, k)
        row.update({f'Q_B_{name}': m['Q'], f'within_frac_{name}': m['within_frac'], f'n_edges_{name}': m['n_edges']})
    # Stratified sampling: random sampling can eliminate small children and silently
    # change the actual split being evaluated. Preserve every child and >=folds items.
    sel = np.arange(len(node_items))
    if len(sel) > cfg['max_items']:
        rng = np.random.default_rng(stable_seed(seed, 'sub'))
        min_each = max(cfg['folds'], cfg['min_child'])
        if k * min_each > cfg['max_items']:
            row['status'] = 'skipped_sample_budget'
            return row
        counts = np.bincount(y, minlength=k)
        allocations = np.full(k, min_each, dtype=int)
        left = cfg['max_items'] - allocations.sum()
        capacity = counts - allocations
        if np.any(capacity < 0):
            row['status'] = 'skipped_child_too_small_for_cv'
            return row
        # Allocate leftover budget proportionally, then distribute rounding remainder.
        if capacity.sum():
            extra = left * capacity / capacity.sum()
            add = np.minimum(capacity, np.floor(extra).astype(int))
            allocations += add
            remaining = cfg['max_items'] - allocations.sum()
            for j in np.argsort(-(extra - np.floor(extra))):
                if remaining == 0:
                    break
                if allocations[j] < counts[j]:
                    allocations[j] += 1
                    remaining -= 1
        sel = np.sort(np.concatenate([
            rng.choice(np.flatnonzero(y == j), size=allocations[j], replace=False)
            for j in range(k)]))
    ys = y[sel]
    ks = k
    assert len(np.unique(ys)) == k, 'stratified sample lost a child'
    feats = {m: feats_all[m][node_items[sel]] for m in ('T', 'V')}
    res = realizability(feats, ys, ks, cfg, seed)
    row['n_items_R'] = int(len(sel))
    for m, key in (('T', 'T'), ('V', 'V'), ('V|T', 'VgT')):
        r = res[m]
        row.update({f'R_{key}': r['R'], f'R_{key}_lo': r['R_ci'][0], f'R_{key}_hi': r['R_ci'][1],
                    f'dCE_{key}': r['dCE'], f'den_{key}': r['denominator'], f'C_{key}': r['C'],
                    f'ce_model_{key}': r['ce_model'], f'ce_null_{key}': r['ce_null'],
                    f'control_gap_{key}': r.get('shuffled_image_control_gap', float('nan'))})
        for d in cfg['delta_grid']:
            row[f'pass_{key}_d{d}'] = passes(r, d, cfg['delta_abs'])
        row[f'pass_{key}'] = passes(r, cfg['delta_rel'], cfg['delta_abs'])
    row['status'] = 'ok'
    return row


# ----------------------------------------------------------------------------- P2
def run_p2(labels, items, edges, n, feats_all, cfg, seed, jobs, smoke=False):
    depth = labels.shape[1]
    jobs_list = []
    for d in range(depth):
        if d == 0:
            parents = {0: items}
        else:
            par = labels[items, d - 1]
            parents = {int(p): items[par == p] for p in np.unique(par)}
        for pid, node_items in sorted(parents.items()):
            tag = dict(depth=d + 1, parent=pid)
            jobs_list.append((tag, node_items, labels[node_items, d]))
    if smoke:
        jobs_list = jobs_list[:3]
    log(f'P2: {len(jobs_list)} splits (root + parents at each depth)')
    rows = Parallel(n_jobs=jobs, verbose=5)(
        delayed(evaluate_split)(tag, ni, yg, edges, n, feats_all, cfg, stable_seed(seed, 'p2', tag['depth'], tag['parent']))
        for tag, ni, yg in jobs_list)
    return rows


def p2_paths(rows, labels, items, key):
    """Per item: number of consecutive passing splits from the root along its path."""
    depth = labels.shape[1]
    ok = {(r['depth'], r['parent']): (r.get('status') == 'ok' and bool(r.get(key))) for r in rows}
    run = np.zeros(len(items), dtype=int)
    alive = np.ones(len(items), dtype=bool)
    for d in range(depth):
        par = np.zeros(len(items), dtype=int) if d == 0 else labels[items, d - 1]
        step = np.array([ok.get((d + 1, int(p)), False) for p in par])
        alive &= step
        run += alive
    return run


def summarize_p2(rows, labels, items, cfg):
    depth = labels.shape[1]
    summary = []
    for d in range(1, depth + 1):
        rs = [r for r in rows if r['depth'] == d]
        okr = [r for r in rs if r.get('status') == 'ok']
        tot = sum(r['n_items'] for r in rs)
        s = dict(depth=d, splits=len(rs), evaluated=len(okr),
                 items_covered_by_evaluated=float(sum(r['n_items'] for r in okr) / max(tot, 1)))
        for key in ('T', 'V', 'VgT'):
            pass_rows = [r for r in okr if r.get(f'pass_{key}')]
            s[f'frac_splits_pass_{key}'] = len(pass_rows) / max(len(okr), 1)
            s[f'frac_items_under_pass_{key}'] = sum(r['n_items'] for r in pass_rows) / max(tot, 1)
            vals = [r[f'R_{key}'] for r in okr if np.isfinite(r.get(f'R_{key}', np.nan))]
            s[f'median_R_{key}'] = float(np.median(vals)) if vals else float('nan')
        qs = [r['Q_B_U2'] for r in okr if np.isfinite(r.get('Q_B_U2', np.nan))]
        s['median_Q_B_U2'] = float(np.median(qs)) if qs else float('nan')
        summary.append(s)
    paths = {}
    for key in ('pass_T',):
        run = p2_paths(rows, labels, items, key)
        paths[key] = {f'frac_items_depth_ge_{j}': float((run >= j).mean()) for j in range(1, depth + 1)}
    for r in rows:
        r['pass_T_or_VgT'] = bool(r.get('pass_T')) or bool(r.get('pass_VgT'))
    run = p2_paths(rows, labels, items, 'pass_T_or_VgT')
    paths['pass_T_or_VgT'] = {f'frac_items_depth_ge_{j}': float((run >= j).mean()) for j in range(1, depth + 1)}
    root = [r for r in rows if r['depth'] == 1]
    root_pass = bool(root and root[0].get('status') == 'ok' and root[0].get('pass_T'))
    go = root_pass and paths['pass_T']['frac_items_depth_ge_2'] >= 0.30 if depth >= 2 else root_pass
    decision = dict(rule='root split passes (T) AND >=30% items have >=2 consecutive passing splits (T)',
                    root_pass_T=root_pass, frac_items_depth_ge_2_T=paths['pass_T'].get('frac_items_depth_ge_2'),
                    P2_GO=bool(go))
    return summary, paths, decision


# ----------------------------------------------------------------------------- P1
def p1_candidates(items, sketch, spectral, feats_all, ref_labels, cfg, seed, smoke=False):
    ks = cfg['p1_ks'][:2] if smoke else cfg['p1_ks']
    seeds = 1 if smoke else cfg['p1_seeds']
    cands = []
    km = lambda X, k, s: MiniBatchKMeans(n_clusters=k, random_state=s, n_init=3,
                                         batch_size=min(2048, len(X)), max_iter=200).fit_predict(X)
    for k in ks:
        for s in range(seeds):
            cands.append(dict(family='behavior_sketch', name=f'behavior_sketch_k{k}_s{s}', k=k,
                              y=km(sketch, k, stable_seed(seed, 'beh', k, s))))
            if spectral is not None:
                cands.append(dict(family='behavior_spectral', name=f'behavior_spectral_k{k}_s{s}', k=k,
                                  y=km(spectral, k, stable_seed(seed, 'spectral', k, s))))
    if ref_labels is not None:
        cands.append(dict(family='behavior_reference', name='exp3_tree_L1', k=int(len(np.unique(ref_labels))), y=ref_labels))
    for mod, fam in (('T', 'content_text'), ('V', 'content_image')):
        X = feats_all[mod][items]
        for k in ks:
            cands.append(dict(family=fam, name=f'{fam}_k{k}', k=k, y=km(X, k, stable_seed(seed, fam, k))))
    alphas = cfg['p1_alphas'][:1] if smoke else cfg['p1_alphas']
    mks = cfg['p1_mixed_ks'][:1] if smoke else cfg['p1_mixed_ks']
    text = feats_all['T'][items]
    for al in alphas:
        X = np.concatenate([np.sqrt(al) * sketch, np.sqrt(1 - al) * text], axis=1)
        for k in mks:
            cands.append(dict(family='mixed', name=f'mixed_a{al}_k{k}', k=k, alpha=al,
                              y=km(X, k, stable_seed(seed, 'mix', al, k))))
    return cands


def run_p1(items, edges, n, feats_all, sketch, spectral, ref_labels, cfg, seed, jobs, smoke=False):
    cands = p1_candidates(items, sketch, spectral, feats_all, ref_labels, cfg, seed, smoke)
    log(f'P1: {len(cands)} root candidate splits')
    rows = Parallel(n_jobs=jobs, verbose=5)(
        delayed(evaluate_split)(dict(family=c['family'], name=c['name'], k_requested=c['k'], alpha=c.get('alpha', '')),
                                items, c['y'], edges, n, feats_all, cfg, stable_seed(seed, 'p1', c['name']))
        for c in cands)
    beh = [r for r in rows if r['family'].startswith('behavior_') and r.get('status') == 'ok']
    qmax = max([r['Q_B_U2'] for r in beh if np.isfinite(r.get('Q_B_U2', np.nan))], default=float('nan'))
    for r in rows:
        r['Q_B_U2_rel_to_best_behavior'] = (r.get('Q_B_U2', np.nan) / qmax
                                              if np.isfinite(qmax) and qmax > 0 else float('nan'))
        r['upper_right'] = bool(r.get('status') == 'ok' and r.get('pass_T') and r['Q_B_U2_rel_to_best_behavior'] >= 0.5)
    good = [r['name'] for r in rows if r['upper_right'] and (r['family'].startswith('behavior_') or r['family'] == 'mixed')]
    decision = dict(rule='>=1 behavior/mixed candidate with pass_T and Q_B_U2 >= 0.5 * best behavior Q_B_U2',
                    best_behavior_Q_B_U2=qmax, upper_right_behavior_or_mixed=good,
                    upper_right_by_family={f: int(sum(r['upper_right'] for r in rows if r['family'] == f))
                                           for f in sorted({r['family'] for r in rows})},
                    P1_GO=bool(good))
    return rows, decision


def plot_p1(rows, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        log('plot skipped:', e)
        return
    colors = dict(behavior_sketch='tab:blue', behavior_spectral='tab:cyan', behavior_reference='tab:grey', mixed='tab:purple', content_text='tab:orange', content_image='tab:green')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, key, title in ((axes[0], 'T', 'Text realizability R_T'), (axes[1], 'V', 'Image realizability R_V')):
        for fam, col in colors.items():
            rs = [r for r in rows if r['family'] == fam and r.get('status') == 'ok']
            if not rs:
                continue
            x = np.array([r[f'R_{key}'] for r in rs])
            lo = np.array([r[f'R_{key}_lo'] for r in rs])
            hi = np.array([r[f'R_{key}_hi'] for r in rs])
            y = np.array([r['Q_B_U2'] for r in rs])
            ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt='o', color=col, alpha=.8, label=fam, ms=5, capsize=2)
            for r in rs:
                if r['name'] == 'exp3_tree_L1':
                    ax.annotate('Exp3 tree L1', (r[f'R_{key}'], r['Q_B_U2']), fontsize=8)
        ax.set_title(title)
        ax.set_xlabel('normalized null-corrected dCE (held-out items)')
        ax.axvline(0, color='grey', lw=.6)
    axes[0].set_ylabel('Q_B: modularity on held-out U2 edges')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ----------------------------------------------------------------------------- data
def load_server_data(a):
    sys.path.insert(0, str(ROOT / 'experiments' / 'discrete_mi_portable_v2'))
    import run_level1_v2 as v2  # noqa: E402  (imports offdiag_core -> experiments/motivation)
    ns = argparse.Namespace(domain=a.domain, mapping_path=None, embedding_path=None, sequence_path=None,
                            seed=a.seed, k=256, eval_positives=a.eval_positives, dry_run=(a.mode == 'check'))
    return v2.make_data(ns)


def load_or_build_hierarchy(a, dat):
    path = Path(a.hierarchy_path) if a.hierarchy_path else ROOT / EXP3_DIR / f'{a.domain}_K256_seed{a.seed}' / 'u1_hierarchy_labels.npy'
    branches = tuple(int(x) for x in a.hierarchy_branches.split(','))
    meta = dict(path=str(path))
    if path.is_file():
        labels = np.load(path)
        meta['source'] = 'loaded Exp3 v1.2 U1 hierarchy'
    else:
        sys.path.insert(0, str(ROOT / 'experiments' / 'hierarchical_sid_v12'))
        from run_exp3_v12 import make_hierarchy  # noqa: E402
        labels, m = make_hierarchy(dat['edges'][0], dat['train_items'], dat['n'], branches, a.sketch_dim, a.seed)
        meta.update(source='rebuilt with Exp3 make_hierarchy (same settings)', **m)
    validate_hierarchy(labels, dat['train_items'], dat['n'])
    meta.update(shape=list(labels.shape), groups_per_level=[int(len(np.unique(labels[dat['train_items'], d])))
                                                            for d in range(labels.shape[1])],
                sha256=hashlib.sha256(labels.tobytes()).hexdigest())
    return labels, meta


def validate_hierarchy(labels, items, n):
    if labels.ndim != 2 or labels.shape[0] != n:
        raise ValueError(f'hierarchy shape {labels.shape} != ({n}, depth)')
    if np.any(labels[items] < 0):
        raise ValueError('some U1 train items lack hierarchy labels: split/seed mismatch with Exp3')
    mask = np.ones(n, dtype=bool)
    mask[items] = False
    if np.any(labels[mask] >= 0):
        raise ValueError('labels exist outside U1 train items: split/seed mismatch with Exp3')
    for d in range(1, labels.shape[1]):
        child, parent = labels[items, d], labels[items, d - 1]
        pairs = np.unique(np.stack([child, parent], 1), axis=0)
        if len(np.unique(pairs[:, 0])) != len(pairs):
            raise ValueError(f'hierarchy not nested at depth {d + 1}')


# ----------------------------------------------------------------------------- main
def make_cfg(a):
    cfg = dict(folds=a.folds, permutations=a.permutations, bootstrap=a.bootstrap,
               c_grid=[float(x) for x in a.c_grid.split(',')], max_iter=a.max_iter, max_items=a.max_items,
               min_items=a.min_items, min_child=a.min_child, eps=a.eps_denominator,
               delta_grid=[float(x) for x in a.delta_rel_grid.split(',')], delta_rel=a.delta_rel, delta_abs=a.delta_abs,
               p1_ks=[int(x) for x in a.p1_ks.split(',')], p1_seeds=a.p1_seeds,
               p1_alphas=[float(x) for x in a.p1_alphas.split(',')], p1_mixed_ks=[int(x) for x in a.p1_mixed_ks.split(',')])
    if a.mode in ('smoke', 'self-test'):
        cfg.update(folds=3, permutations=2, bootstrap=50, max_items=min(a.max_items, 3000), c_grid=cfg['c_grid'][1:2])
    return cfg


def run_all(a, dat, labels, hier_meta, dest):
    cfg = make_cfg(a)
    parts = {x.strip() for x in a.parts.split(',') if x.strip()}
    if not parts or not parts.issubset({'p1','p2'}):
        raise ValueError(f'--parts must consist only of p1,p2; got {sorted(parts)}')
    smoke = a.mode == 'smoke'
    items = np.asarray(dat['train_items'])
    n = dat['n']
    feats_all = {'T': l2norm(dat['emb']['text'][0]), 'V': l2norm(dat['emb']['image'][0])}
    for m in ('T', 'V'):
        if feats_all[m].shape[0] != n:
            raise ValueError(f'embedding rows {feats_all[m].shape[0]} != n {n}')
    edges = dat['edges']
    manifest = dict(protocol=PROTOCOL, mode=a.mode, domain=a.domain, seed=a.seed, args=vars(a), cfg=cfg,
                    hierarchy=hier_meta, n_items=int(n), n_u1_items=int(len(items)),
                    n_edges=[int(len(e[0])) for e in edges[:2]], definitions=__doc__,
                    caveats=['Child labels of held-out items come from the U1 graph that includes their own edges '
                             '(transductive labels); the content probe itself is item-inductive.',
                             'P2 evaluates the behavior-only Exp3 tree (CID-like), not an RCH tree.',
                             'Linear probe is model-dependent; not a mathematically certified lower bound on realizability.',
                             'Item-heldout probe labels use a U1 graph containing those items: not fully inductive.',
                             'P1 content-only and mixed candidates were clustered with all U1 item embeddings; transductive candidate discovery.',
                             'P1 candidates are selected using U2 Q_B; a separate untouched split is needed for confirmatory Go.',
                             'Image conditional score is held-out CE_T minus CE_TV, NOT conditional MI.',
                             'R is a null-corrected predictive score, not mutual information.',
                             'Q_B uses only U2 edges whose endpoints are U1 train items.',
                             'Modularity depends on K; compare candidates with similar K when possible.'])
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    decisions = {}
    if 'p2' in parts:
        t0 = time.time()
        rows = run_p2(labels, items, edges, n, feats_all, cfg, a.seed, a.jobs, smoke)
        summary, paths, dec = summarize_p2(rows, labels, items, cfg)
        save_csv(dest / 'p2_splits.csv', rows)
        save_csv(dest / 'p2_depth_summary.csv', summary)
        (dest / 'p2_paths.json').write_text(json.dumps(paths, indent=2))
        decisions['P2'] = dec
        log(f'P2 done in {time.time() - t0:.0f}s', json.dumps(dec))
    if 'p1' in parts:
        t0 = time.time()
        sketch = behavior_sketch(edges[0], items, n, a.sketch_dim, a.seed)
        spectral = None
        if not a.no_spectral:
            log('P1: computing true U1 spectral embedding (can be expensive)')
            spectral = behavior_spectral(edges[0], items, n, min(a.spectral_dim, max(cfg['p1_ks']) + 8))
        rows, dec = run_p1(items, edges, n, feats_all, sketch, spectral, labels[items, 0], cfg, a.seed, a.jobs, smoke)
        save_csv(dest / 'p1_candidates.csv', rows)
        plot_p1(rows, dest / 'p1_scatter.png')
        decisions['P1'] = dec
        log(f'P1 done in {time.time() - t0:.0f}s', json.dumps(dec))
    (dest / 'DECISION.json').write_text(json.dumps(decisions, indent=2, default=str))
    (dest / 'DONE.json').write_text(json.dumps(dict(status='success', parts=sorted(parts)), indent=2))
    log('SUCCESS', dest)
    return decisions


def self_test():
    """Synthetic sanity check: content encodes the coarse behavior split only.
    Expected: root split realizable (T), deeper splits mostly not; P1 behavior candidate is upper-right."""
    rng = np.random.default_rng(0)
    n, coarse, fine, d = 2400, 4, 4, 32
    g1 = rng.integers(0, coarse, n)
    g2 = g1 * fine + rng.integers(0, fine, n)
    centers = rng.normal(size=(coarse, d)) * 2.5
    text = centers[g1] + rng.normal(size=(n, d))          # text knows coarse group only
    image = rng.normal(size=(n, d))                       # image knows nothing
    image[:, :4] += np.eye(4)[g1] * .5                    # weak coarse signal
    def edges_for(m):
        a, b = [], []
        for _ in range(m):
            i = rng.integers(0, n)
            same = np.flatnonzero(g2 == g2[i]) if rng.random() < .7 else np.flatnonzero(g1 == g1[i])
            j = rng.choice(same)
            if i != j:
                a.append(min(i, j)); b.append(max(i, j))
        a, b = np.array(a), np.array(b)
        return a, b, np.ones(len(a)), np.bincount(np.r_[a, b], minlength=n).astype(float)
    edges = [edges_for(20000), edges_for(8000)]
    items = np.arange(n)
    # Planted nested behavior tree (tests the probe itself, independent of tree-building noise)
    labels = np.stack([g1, g2], 1).astype(np.int32)
    validate_hierarchy(labels, items, n)
    # Negative control: a non-nested tree must be rejected
    bad = labels.copy()
    bad[:5, 1] = (bad[:5, 1] + fine) % (coarse * fine)
    try:
        validate_hierarchy(bad, items, n)
        raise AssertionError('non-nested hierarchy was not rejected')
    except ValueError:
        pass
    dat = dict(n=n, train_items=items, edges=edges, emb={'text': (text, None), 'image': (image, None)})
    import tempfile
    dest = Path(tempfile.mkdtemp(prefix='rch_selftest_'))
    a = parse_args(['--mode', 'self-test', '--jobs', '1', '--p1-ks', '4,16', '--p1-seeds', '1',
                    '--p1-alphas', '0.5', '--p1-mixed-ks', '4'])
    a.domain = 'synthetic'
    dec = run_all(a, dat, labels, dict(source='synthetic'), dest)
    import csv
    rows = list(csv.DictReader(open(dest / 'p2_splits.csv')))
    root = [r for r in rows if r['depth'] == '1'][0]
    deep = [r for r in rows if r['depth'] == '2' and r['status'] == 'ok']
    deep_pass = np.mean([r['pass_T'] == 'True' for r in deep])
    print('SELF-TEST root R_T', root['R_T'], 'pass', root['pass_T'], '| depth-2 pass rate', deep_pass)
    assert root['pass_T'] == 'True', 'root coarse split should be text-realizable'
    assert deep_pass <= 0.25, 'fine splits carry no content signal and should mostly fail'
    assert root['pass_VgT'] in ('False', 'True')
    assert dec['P1']['P1_GO'], 'behavior candidate should be upper-right'
    print('SELF-TEST PASS', dest)


def main():
    a = parse_args()
    if a.mode == 'self-test':
        self_test()
        return
    if not a.domain:
        raise SystemExit('--domain required')
    if a.folds < 2 or a.permutations < 1 or a.bootstrap < 10:
        raise ValueError('Require folds>=2, permutations>=1, bootstrap>=10')
    dat = load_server_data(a)
    if a.mode == 'check':
        hp = Path(a.hierarchy_path) if a.hierarchy_path else ROOT / EXP3_DIR / f'{a.domain}_K256_seed{a.seed}' / 'u1_hierarchy_labels.npy'
        print('CHECK: inputs found by make_data;', 'hierarchy', hp, 'exists' if hp.is_file() else 'MISSING -> will rebuild with Exp3 make_hierarchy')
        print('CHECK PASS (make_data completed; no P1/P2 computation performed)')
        return
    dest = (ROOT / a.out_root / a.mode / f'{a.domain}_seed{a.seed}').resolve()
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite {dest}')
    labels, meta = load_or_build_hierarchy(a, dat)
    dest.mkdir(parents=True)
    log('hierarchy', json.dumps({k: v for k, v in meta.items() if k != 'sha256'}))
    run_all(a, dat, labels, meta, dest)


if __name__ == '__main__':
    main()
