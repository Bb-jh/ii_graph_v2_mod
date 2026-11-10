#!/usr/bin/env python3
"""Independent user-resampling audit for fixed U3 code-MI comparison.

Read-only for EXP2 V2 files. User-level resampling re-derives the U3
weighted co-occurrence graph W_B in each draw. No fitting or U3 selection.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
MOD = ROOT / 'experiments' / 'discrete_mi_portable_v2'
sys.path.insert(0, str(MOD))
import offdiag_core as core
from run_level1_v2 import resolve_input

METHODS = ('rq', 'infonce', 'code_mi')
PAIRS = (('code_mi', 'infonce'), ('code_mi', 'rq'), ('infonce', 'rq'))


def weighted_u3_edges(x_el, eligible, multiplicity):
    """Compute W_B from user-repeat-count weighted incidence, exactly as original.

    Weighted coocc(i,j) = sum_u multiplicity[u] * X[u,i]*X[u,j]
    Weighted deg(i) = sum_u multiplicity[u] * X[u,i]
    """
    count = np.asarray(multiplicity, dtype=np.float64)
    if count.ndim != 1 or count.size != x_el.shape[0]:
        raise ValueError('User multiplicity mismatch')
    if (count < 0).any() or count.sum() <= 0:
        raise ValueError('Invalid multiplicities')
    degree = np.asarray(x_el.T @ count).ravel()
    weighted_x = x_el.multiply(count[:, None])
    cooc = (x_el.T @ weighted_x).tocsr()
    cooc.setdiag(0)
    cooc.eliminate_zeros()
    upper = sparse.triu(cooc, k=1, format='coo')
    ia, ib = upper.row, upper.col
    weights = upper.data.astype(np.float64) / np.maximum(np.sqrt(degree[ia] * degree[ib]), 1e-12)
    return eligible[ia].astype(np.int32), eligible[ib].astype(np.int32), weights


def joint_mi(codes, a, b, w, k):
    return core.measures(core.code_joint(codes, a, b, w, k))['mi_real']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', choices=('Industrial', 'Musical'), required=True)
    p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--source-root', default='outputs/discrete_mi_level1_v2_portable')
    p.add_argument('--out-root', default='outputs/discrete_mi_user_bootstrap_audit_v2')
    p.add_argument('--progress-every', type=int, default=25)
    args = p.parse_args()
    if args.bootstrap < 20:
        raise ValueError('At least 20 bootstrap replicates required')
    src = (ROOT / args.source_root).resolve()
    dataset, seqname = core.DOMAINS[args.domain]
    # source directory defaults to frozen K256 seed42 but metadata is authoritative
    folder = src / f'{args.domain}_K256_seed42'
    if not (folder / 'DONE.json').exists():
        raise FileNotFoundError(f'Completed experiment missing: {folder}')
    config = json.loads((folder / 'config.json').read_text())
    k = int(config['k'])
    split_seed = int(config['seed'])
    from argparse import Namespace
    overrides = Namespace(mapping=config.get('mapping_path'), embedding=config.get('embedding_path'), sequence=config.get('sequence_path'))
    mapping = resolve_input('mapping', dataset, seqname, overrides.mapping)
    embfile = resolve_input('embedding', dataset, seqname, overrides.embedding)
    sequence = resolve_input('sequence', dataset, seqname, overrides.sequence)
    ids = core.load_mapping(mapping)
    n = len(ids)
    _, seqs = core.read_users_train(sequence, n)
    groups, _ = core.split_stratified(seqs, split_seed)
    # Match v2's eligible set; read only the embedding validity masks.
    embedding = core.load_embeddings(embfile, ids)
    eligible = np.flatnonzero(embedding['text'][1] & embedding['image'][1]).astype(np.int32)
    del embedding
    u3_users = [seqs[int(i)] for i in groups[2]]
    x = core.csr_users(u3_users, n).tocsr()
    x_el = x[:, eligible].tocsr()
    x_el = x_el.astype(np.float64)
    if x_el.nnz and not np.all(x_el.data == 1):
        raise ValueError('U3 incidence is not binary; cannot reproduce frozen edge definition')
    n_users = x_el.shape[0]
    codes = {}
    for mod in ('text', 'image'):
        for method in METHODS:
            filename = folder / mod / f'{method}_codes.npy'
            c = np.load(filename, allow_pickle=False).astype(np.int64, copy=False)
            if len(c) != n or np.any((c[eligible] < 0) | (c[eligible] >= k)):
                raise ValueError(f'Invalid codes: {filename}')
            codes[(mod, method)] = c
    print(f'U3 users={n_users} eligible_items={len(eligible)} nonzeros={x_el.nnz}', flush=True)
    # Identity: rebuilding graph at one of every user must match frozen algorithm.
    base = core.edges_from_users(x, eligible)
    aa, bb, ww = weighted_u3_edges(x_el, eligible, np.ones(n_users, dtype=np.float64))
    key1 = core.keypair(base[0], base[1], n)
    key2 = core.keypair(aa, bb, n)
    o1 = np.argsort(key1)
    o2 = np.argsort(key2)
    if len(key1) != len(key2) or not np.array_equal(key1[o1], key2[o2]):
        raise AssertionError('IDENTITY FAIL: edge keys differ')
    if not np.allclose(base[2][o1], ww[o2], rtol=1e-9, atol=1e-12):
        raise AssertionError('IDENTITY FAIL: W_B differs from original')
    observed = {(mod, method): joint_mi(c, aa, bb, ww, k) for (mod, method), c in codes.items()}
    with (folder / 'summary.csv').open(newline='') as f:
        saved = {(r['modality'], r['method']): float(r['code_mi']) for r in csv.DictReader(f)}
    for key in observed:
        if not np.isclose(observed[key], saved[key], atol=1e-8, rtol=1e-8):
            raise AssertionError(f'IDENTITY FAIL: {key} reconstructed J {observed[key]} vs saved {saved[key]}')
    print('IDENTITY PASS: edges, W_B and all 6 J values reproduce saved experiment', flush=True)
    out = (ROOT / args.out_root / f'{args.domain}_K{k}_seed{split_seed}').resolve()
    if out.exists():
        raise FileExistsError(f'Output already exists; will not overwrite: {out}')
    out.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    draws = {(mod, lhs, rhs): np.zeros(args.bootstrap) for mod in ('text', 'image') for lhs, rhs in PAIRS}
    # Every method and modality shares the exact same sampled U3 user IDs per replicate.
    for t in range(args.bootstrap):
        selected_users = rng.integers(0, n_users, size=n_users)
        multiplicity = np.bincount(selected_users, minlength=n_users).astype(np.float64)
        a, b, w = weighted_u3_edges(x_el, eligible, multiplicity)
        if not len(a) or np.sum(w) <= 0:
            raise RuntimeError(f'Empty graph in bootstrap draw {t}')
        jm = {key: joint_mi(c, a, b, w, k) for key, c in codes.items()}
        for mod in ('text', 'image'):
            for lhs, rhs in PAIRS:
                draws[(mod, lhs, rhs)][t] = jm[(mod, lhs)] - jm[(mod, rhs)]
        if (t + 1) % args.progress_every == 0 or t + 1 == args.bootstrap:
            print('USER BOOTSTRAP', t+1, '/', args.bootstrap, flush=True)
    rows = []
    for (mod, lhs, rhs), d in draws.items():
        point = observed[(mod, lhs)] - observed[(mod, rhs)]
        mean = float(np.mean(d))
        quant = np.quantile(d, [.025, .975])
        # Note: percentile/basic are not guaranteed coverage-correct for nonlinear MI.
        rows.append(dict(domain=args.domain, modality=mod, lhs=lhs, rhs=rhs,
                         observed_J_diff=point, boot_mean=mean, boot_sd=float(np.std(d, ddof=1)),
                         boot_bias=mean-point, percentile_lo=float(quant[0]), percentile_hi=float(quant[1]),
                         centered_lo=float(point+quant[0]-mean), centered_hi=float(point+quant[1]-mean),
                         basic_lo=float(2*point-quant[1]), basic_hi=float(2*point-quant[0]),
                         percentile_excludes_zero=bool(quant[0]>0 or quant[1]<0),
                         centered_excludes_zero=bool(point+quant[0]-mean>0 or point+quant[1]-mean<0)))
    with (out / 'paired_J_user_bootstrap.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out / 'paired_J_user_bootstrap_draws.npz', **{f'{mod}_{lhs}_minus_{rhs}': value for (mod, lhs, rhs), value in draws.items()})
    (out / 'manifest.json').write_text(json.dumps(dict(domain=args.domain, seed=args.seed, split_seed=split_seed,
        k=k, bootstrap=args.bootstrap, resampling_unit='U3 user', method='multinomial resampling users with replacement; rebuild weighted U3 W_B graph per draw',
        n_u3_users=n_users, n_items=n, n_eligible=len(eligible), n_u3_edges_original=len(aa),
        identity_pass=True, no_training=True, no_U3_tuning=True, source=str(folder),
        caveat='User bootstrap assesses user-sample variability conditional on fixed learned codebooks. It does not include tokenizer-training or U1/U2 sampling uncertainty; nonlinear MI finite-sample bias may remain.'), indent=2)+'\n')
    print('AUDIT SUCCESS:', out, flush=True)
    for r in rows:
        print(f"{r['modality']} {r['lhs']}-{r['rhs']}: ΔJ={r['observed_J_diff']:+.6f} user-bootstrap percentile=[{r['percentile_lo']:+.6f}, {r['percentile_hi']:+.6f}] centered=[{r['centered_lo']:+.6f},{r['centered_hi']:+.6f}]", flush=True)

if __name__ == '__main__':
    main()
