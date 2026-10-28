"""Cluster-aware batches (so V_l is non-empty) and EdgeDrop for the Graph branch input."""
import math

import numpy as np
import torch


def cluster_aware_batches(labels, batch_size, *, per_fine=4, fines_per_local=2, locals_per_global=2, rng=None):
    """Yield unique labeled row indices per batch, ~ceil(labeled/batch_size) batches.

    Each draw picks a size-weighted Global, then up to `locals_per_global` Locals,
    `fines_per_local` Fines per Local and `per_fine` items per Fine, so every level
    gets positives (same child) and in-parent negatives (sibling child).
    labels: [items, 3] composite IDs (pseudo_labels.npz "labels"), -1 = unlabeled.
    Unlabeled items are never yielded; feed them to the RQ loss separately.
    """
    rng = np.random.default_rng(rng)
    labels = np.asarray(labels)
    rows = np.flatnonzero((labels >= 0).all(1))
    if not len(rows) or batch_size < 1:
        raise ValueError("Need labeled rows and a positive batch size")
    tree = {}
    for row in rows:
        g, l, f = labels[row]
        tree.setdefault(g, {}).setdefault(l, {}).setdefault(f, []).append(row)
    globals_ = np.array(list(tree))
    sizes = np.array([sum(len(v) for local in tree[g].values() for v in local.values()) for g in globals_], dtype=float)
    target = min(batch_size, len(rows))
    for _ in range(math.ceil(len(rows) / batch_size)):
        batch, seen = [], set()
        for _ in range(100 * target):
            if len(batch) >= target:
                break
            local = tree[rng.choice(globals_, p=sizes/sizes.sum())]
            for l in rng.choice(list(local), min(locals_per_global, len(local)), replace=False):
                fine = local[l]
                for f in rng.choice(list(fine), min(fines_per_local, len(fine)), replace=False):
                    for row in rng.choice(fine[f], min(per_fine, len(fine[f])), replace=False):
                        if row not in seen and len(batch) < target:
                            seen.add(row)
                            batch.append(row)
        yield np.array(batch, dtype=np.int64)


def edge_drop(edge_index, edge_weight, num_nodes, *, edge_p=0.1, node_p=0.1, generator=None):
    """EdgeDrop(Ã): drop each undirected edge with edge_p and ALL edges of a node with node_p.

    edge_index [2,E] lists every undirected edge once (as edges.npz, src < dst);
    symmetrize after dropping. Isolated nodes rehearse cold-start items.
    """
    if not (0 <= edge_p < 1 and 0 <= node_p < 1):
        raise ValueError("Drop probabilities must be in [0,1)")
    device = edge_index.device
    keep = torch.rand(edge_index.shape[1], generator=generator, device=device) >= edge_p
    isolated = torch.rand(num_nodes, generator=generator, device=device) < node_p
    keep &= ~isolated[edge_index[0]] & ~isolated[edge_index[1]]
    return edge_index[:, keep], (None if edge_weight is None else edge_weight[keep]), isolated
