"""Stage 2: hierarchical pseudo-labels (fixed after this stage).

Encoder (separate from the tokenizer's Graph branch): LightGCN on the LCC of W,
trained self-supervised by BPR link prediction with positives sampled ∝ w_ij.
h^G = L2-normalized layer-mean embedding. Recursive k-means:
Global -> Local inside each Global -> Fine inside each Local.
Labels are composite IDs, globally unique per level; items outside the LCC get -1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
import torch
from torch.nn import functional as F


def normalized_adjacency(graph, device, self_loops=False):
    graph = graph.tocsr()
    if self_loops:
        graph = graph + sparse.identity(graph.shape[0], dtype=graph.dtype, format="csr")
    degree = np.asarray(graph.sum(1)).ravel()
    inv = np.where(degree > 0, degree ** -0.5, 0)
    norm = sparse.diags(inv) @ graph @ sparse.diags(inv)
    coo = norm.tocoo()
    index = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(index, torch.as_tensor(coo.data, dtype=torch.float32),
                                   norm.shape, device=device).coalesce()


def train_encoder(graph, cfg, seed, device):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n = graph.shape[0]
    adjacency = normalized_adjacency(graph, device)
    embedding = torch.nn.Parameter(0.1*torch.randn(n, cfg.dim, device=device))
    optimizer = torch.optim.Adam([embedding], lr=cfg.lr)
    coo = graph.tocoo()
    probability = coo.data / coo.data.sum()

    def propagate():
        layers, h = [embedding], embedding
        for _ in range(cfg.layers):
            h = torch.sparse.mm(adjacency, h)
            layers.append(h)
        return torch.stack(layers).mean(0)

    for epoch in range(cfg.epochs):
        picks = rng.choice(len(coo.data), len(coo.data), p=probability)
        total = 0.0
        for start in range(0, len(picks), cfg.batch_size):
            batch = picks[start:start+cfg.batch_size]
            u = torch.as_tensor(coo.row[batch], device=device)
            p = torch.as_tensor(coo.col[batch], device=device)
            q = torch.as_tensor(rng.integers(0, n, len(batch)), device=device)
            h = propagate()
            score = (h[u] * (h[p] - h[q])).sum(-1)
            reg = embedding[u].square().sum() + embedding[p].square().sum() + embedding[q].square().sum()
            loss = -F.logsigmoid(score).mean() + cfg.reg * reg / len(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        if epoch == 0 or (epoch+1) % 20 == 0 or epoch+1 == cfg.epochs:
            print(f"[stage2] encoder epoch {epoch+1}/{cfg.epochs} bpr={total/len(picks):.4f}", flush=True)
    with torch.no_grad():
        return F.normalize(propagate(), dim=-1).cpu().numpy()


def recursive_kmeans(h, ks, min_cluster_size, seed, n_init):
    """Parent-relative path labels [n, len(ks)]."""
    paths = np.zeros((len(h), len(ks)), dtype=np.int64)
    groups = [np.arange(len(h))]
    for level, k in enumerate(ks):
        children = []
        for rows in groups:
            k_eff = max(1, min(k, len(rows) // max(min_cluster_size, 1)))
            if k_eff == 1:
                assign = np.zeros(len(rows), dtype=np.int64)
            else:
                assign = KMeans(k_eff, n_init=n_init, random_state=seed).fit_predict(h[rows])
            paths[rows, level] = assign
            children.extend(rows[assign == c] for c in np.unique(assign))
        groups = children
    return paths


def composite_labels(paths):
    labels = np.zeros_like(paths)
    for depth in range(1, paths.shape[1]+1):
        labels[:, depth-1] = np.unique(paths[:, :depth], axis=0, return_inverse=True)[1].ravel()
    return labels


def run(graph, cfg, seed, out: Path, device):
    out.mkdir(parents=True, exist_ok=True)
    count, components = connected_components(graph, directed=False)
    lcc = np.flatnonzero(components == np.bincount(components).argmax())
    sub = graph[lcc][:, lcc]
    h_lcc = train_encoder(sub, cfg, seed, device)
    paths = recursive_kmeans(h_lcc, cfg.ks, cfg.min_cluster_size, seed, cfg.kmeans_init)
    n = graph.shape[0]
    labels = np.full((n, len(cfg.ks)), -1, dtype=np.int64)
    path_labels = labels.copy()
    labels[lcc] = composite_labels(paths)
    path_labels[lcc] = paths
    h = np.zeros((n, h_lcc.shape[1]), dtype=np.float32)
    h[lcc] = h_lcc
    np.savez_compressed(out / "pseudo_labels.npz", labels=labels, path_labels=path_labels, h=h,
                        valid_mask=labels[:, 0] >= 0, lcc_nodes=lcc)
    stats = dict(labeled=len(lcc), labeled_fraction=len(lcc)/n,
                 clusters=[int(labels[lcc, l].max()+1) for l in range(len(cfg.ks))],
                 median_size=[float(np.median(np.bincount(labels[lcc, l]))) for l in range(len(cfg.ks))])
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[stage2] {json.dumps(stats)}", flush=True)
    return stats
