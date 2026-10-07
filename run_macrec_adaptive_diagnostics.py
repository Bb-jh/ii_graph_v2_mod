"""CPU diagnostics on real MACRec embeddings; Student=Original proxy, not RQ training."""
import argparse
import csv
import hashlib
import json
from itertools import islice
from pathlib import Path

import numpy as np
from scipy import sparse
import torch

from hierarchy.confidence import graph_teacher_confidence
from hierarchy.adaptive_distillation import _js
from run_adaptive_diagnostics import summarize, weight_variants, write_csv


def normalized_rows(z, rows):
    values = np.asarray(z[rows], dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 0):
        raise ValueError('Nonfinite or zero embedding encountered')
    return values/norms


def cosine_rows(z, anchors, candidates, chunk=16):
    values = np.zeros(candidates.shape, dtype=np.float32)
    for start in range(0, len(anchors), chunk):
        stop = start+chunk
        a = normalized_rows(z, anchors[start:stop])
        b = normalized_rows(z, np.maximum(candidates[start:stop], 0))
        values[start:stop] = np.einsum('bd,bkd->bk', a, b)
    return torch.from_numpy(values)


def distribution(scores, mask, temperature):
    logits = (scores/temperature).masked_fill(~mask, -torch.inf)
    logits = torch.where(mask.any(-1, keepdim=True), logits, torch.zeros_like(logits))
    return logits.softmax(-1).masked_fill(~mask, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--hierarchy', type=Path, default=Path('outputs/Toys_multimodal_hierarchy_smoke'))
    parser.add_argument('--graph-dir', type=Path, default=Path('outputs/Toys_multimodal_mutual'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-size', type=int, default=1024)
    parser.add_argument('--candidates', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[.05, .1, .2])
    args = parser.parse_args()
    if args.sample_size < 1 or args.candidates < 2 or any(not np.isfinite(t) or t <= 0 for t in args.temperatures):
        parser.error('Require positive sample size/temperatures and >=2 candidates')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('Use a new or empty output directory')
    torch.set_num_threads(4)
    with np.load(args.hierarchy/'pseudo_labels.npz', allow_pickle=False) as data:
        labels, nodes, ids = (data[k].copy() for k in ('labels', 'lcc_nodes', 'item_ids'))
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError('This MACRec NPY adapter requires numeric row-index item IDs')
    embeddings = {name: np.load(args.data/filename, mmap_mode='r', allow_pickle=False)
                  for name, filename in [('Text', 'Toys.emb-llama-td.npy'), ('Image', 'Toys.emb-ViT-L-14.npy')]}
    for z in embeddings.values():
        if z.ndim != 2 or len(z) != len(ids):
            raise ValueError('Embedding rows do not match graph item universe')
    # Do not infer alignment solely from shape: reproduce stored graph edge similarities.
    with (args.graph_dir/'edge_evidence.csv').open() as stream:
        evidence = list(islice(csv.DictReader(stream), 512))
    if len(evidence) < 32:
        raise ValueError('Insufficient edge evidence to check row alignment')
    src = np.array([int(r['item_i']) for r in evidence])
    dst = np.array([int(r['item_j']) for r in evidence])
    alignment = {}
    for name, z in embeddings.items():
        actual = (normalized_rows(z, src)*normalized_rows(z, dst)).sum(-1)
        expected = np.array([float(r[name.lower()+'_similarity']) for r in evidence])
        error = np.abs(actual-expected)
        if not np.isfinite(error).all() or error.max() > 1e-3:
            raise ValueError(f'{name}: embeddings do not reproduce stored edge similarities')
        alignment[name] = dict(edges=len(evidence), max_cosine_error=float(error.max()))
    graph_path = args.graph_dir/'multimodal_behavior_graph.npz'
    manifest = json.loads((args.hierarchy/'hierarchy_run.json').read_text())
    if hashlib.sha256(graph_path.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise ValueError('Graph hash differs from hierarchy training source')
    graph = sparse.load_npz(graph_path)
    lcc_z = np.load(args.hierarchy/'structural_embeddings.npy', allow_pickle=False)
    graph_z = np.zeros((len(ids), lcc_z.shape[1]), dtype=np.float32)
    graph_z[nodes] = lcc_z
    print('Computing full-graph confidence (centroids reconstructed from saved embeddings)...', flush=True)
    confidence = graph_teacher_confidence(graph, graph_z, labels)
    rng = np.random.default_rng(args.seed)
    anchors = np.sort(rng.choice(nodes, min(args.sample_size, len(nodes)), replace=False))
    levels = labels.shape[1]
    candidates = np.full((len(anchors), levels, args.candidates), -1, dtype=np.int64)
    for level in range(levels):
        groups = {int(p): nodes[labels[nodes, level-1] == p] for p in np.unique(labels[nodes, level-1])} if level else {0: nodes}
        for index, item in enumerate(anchors):
            pool = groups[int(labels[item, level-1]) if level else 0]
            pool = pool[pool != item]
            chosen = rng.choice(pool, min(len(pool), args.candidates), replace=False)
            candidates[index, level, :len(chosen)] = chosen
    usable = torch.from_numpy((candidates >= 0).sum(-1) >= 2)
    c = torch.from_numpy(confidence['confidence'][anchors])
    summaries, item_rows = [], []
    scores = {name: [] for name in ('Graph', *embeddings)}
    for level in range(levels):
        print(f'Computing level {level} sampled relation scores...', flush=True)
        for name, z in dict(Graph=graph_z, **embeddings).items():
            scores[name].append(cosine_rows(z, anchors, candidates[:, level]))
    for modality in embeddings:
        for temperature in args.temperatures:
            need = torch.zeros_like(c)
            for level in range(levels):
                mask = torch.from_numpy(candidates[:, level] >= 0)
                teacher = distribution(scores['Graph'][level], mask, temperature)
                original = distribution(scores[modality][level], mask, temperature)
                need[:, level] = _js(teacher, original)
            need.masked_fill_(~usable, 0)
            compatibility = (1-need).masked_fill(~usable, 0)
            result = dict(need=need, compatibility=compatibility, usable=usable)
            prefix = dict(modality=modality, temperature=temperature, student_mode='original_proxy')
            summaries.extend(dict(**prefix, **row) for row in summarize(result, c))
            variants = weight_variants(c, need, compatibility)
            for index, item in enumerate(anchors):
                for level in range(levels):
                    item_rows.append(dict(**prefix, item_id=int(ids[item]), level=level,
                        usable=bool(usable[index, level]), candidates=int((candidates[index, level] >= 0).sum()),
                        C=float(c[index, level]), N_proxy=float(need[index, level]), M=float(compatibility[index, level]),
                        **{k: float(v[index, level]) for k, v in variants.items()}))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'confidence.npz', item_ids=ids, **confidence)
    np.savez_compressed(args.output/'candidates.npz', anchors=anchors, candidates=candidates)
    write_csv(args.output/'summary.csv', summaries)
    write_csv(args.output/'items.csv', item_rows)
    c_summary = []
    for level in range(levels):
        row = dict(level=level, labeled_items=len(nodes))
        for key in ('assignment', 'topology', 'support', 'confidence'):
            values = confidence[key][nodes, level]
            row.update({key+'_mean': float(values.mean()), key+'_zero_fraction': float((values == 0).mean())})
        c_summary.append(row)
    write_csv(args.output/'confidence_summary.csv', c_summary)
    metadata = dict(mode='real_embeddings_student_original_proxy', parameters={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        embedding_shapes={k:list(v.shape) for k,v in embeddings.items()}, alignment=alignment,
        graph_sha256=manifest['source_sha256'], teacher_run_kind=manifest['run_kind'],
        teacher_epochs=manifest['parameters']['epochs'], labeled_items=len(nodes), sample_items=len(anchors),
        candidate_policy='Uniform without replacement within parent; same candidates across modalities and temperatures; excludes self.',
        assignment_assumption='Euclidean distance to means of final labels in saved normalized structural embedding; upstream clustering code unavailable.',
        limitations=['No trained RQ student: N is Graph-Original discrepancy proxy.', 'One-epoch smoke teacher, not converged teacher.',
                     'Graph itself used these modalities for edge reweighting; M is not independent validation.', 'Weight ratios are not gradient or downstream performance ratios.'])
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print('REAL embeddings; Student=Original proxy; one-epoch teacher')
    for row in summaries:
        if row['band'] == 'all':
            print(f"{row['modality']:5s} tau={row['temperature']:.2f} L={row['level']} C={row['C_mean']:.4f} N_proxy={row['N_mean']:.4f} M={row['M_mean']:.4f} alpha={row['CNM_mean']:.4f} retained={row['CNM_retained_vs_CN']}")
    print(f'Results: {args.output.resolve()}')


if __name__ == '__main__':
    main()
