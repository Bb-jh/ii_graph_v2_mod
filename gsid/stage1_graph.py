"""Stage 1: shared relation graph W (fixed after this stage).

B = UltraGCN item-item weights on train co-occurrence. With G_ii removed,
w_ij = G_ij/g_i * sqrt(g_i/g_j) = G_ij / sqrt(g_i g_j) (symmetric).
M_ij = f(clip01 cos_T, clip01 cos_V) on edges of B only.
W_ij = B_ij [(1-λ) + λ M_ij]; Top-K per item by W, union-symmetrized.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import torch
from torch.nn import functional as F


def edge_cosines(x, rows, cols, device, chunk=65536):
    z = F.normalize(torch.as_tensor(x, device=device), dim=-1)
    out = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), chunk):
        r = torch.as_tensor(rows[start:start+chunk], device=device)
        c = torch.as_tensor(cols[start:start+chunk], device=device)
        out[start:start+chunk] = (z[r] * z[c]).sum(-1).cpu().numpy()
    return out


def top_k_rows(w, k):
    w = w.tocsr()
    keep = np.zeros(w.nnz, dtype=bool)
    for i in range(w.shape[0]):
        start, end = w.indptr[i], w.indptr[i+1]
        if end - start <= k:
            keep[start:end] = True
        else:
            keep[start + np.argpartition(-w.data[start:end], k)[:k]] = True
    coo = w.tocoo()
    return sparse.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=w.shape)


def build_relation_graph(data, cfg, device):
    co = (data.interactions.T @ data.interactions).tocsr().astype(np.float64)
    co.setdiag(0)
    co.eliminate_zeros()
    if cfg.min_cooc > 1:
        co.data[co.data < cfg.min_cooc] = 0
        co.eliminate_zeros()
    g = np.asarray(co.sum(1)).ravel()
    coo = co.tocoo()
    b = coo.data / np.sqrt(g[coo.row] * g[coo.col])
    cos_t = np.clip(edge_cosines(data.text, coo.row, coo.col, device), 0, 1)
    cos_v = np.clip(edge_cosines(data.image, coo.row, coo.col, device), 0, 1)
    m = dict(mean=(cos_t+cos_v)/2, product=cos_t*cos_v, min=np.minimum(cos_t, cos_v))[cfg.reliability]
    w = b * ((1-cfg.lambda_mm) + cfg.lambda_mm*m)
    full = sparse.csr_matrix((w, (coo.row, coo.col)), shape=co.shape)
    top = top_k_rows(full, cfg.top_k)
    graph = top.maximum(top.T).tocsr().astype(np.float32)
    graph.eliminate_zeros()
    return graph, dict(candidate_edges=int(co.nnz//2), M_mean=float(m.mean()) if len(m) else None)


def run(data, cfg, out: Path, device):
    out.mkdir(parents=True, exist_ok=True)
    graph, info = build_relation_graph(data, cfg.graph, device)
    count, components = connected_components(graph, directed=False)
    sizes = np.bincount(components)
    degree = np.diff(graph.indptr)
    stats = dict(**info, nodes=graph.shape[0], edges=int(graph.nnz//2), components=int(count),
                 largest_component=int(sizes.max()), lcc_fraction=float(sizes.max()/graph.shape[0]),
                 isolated=int((degree == 0).sum()), mean_degree=float(degree.mean()))
    sparse.save_npz(out / "graph.npz", graph)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[stage1] {json.dumps(stats)}", flush=True)
    return stats
