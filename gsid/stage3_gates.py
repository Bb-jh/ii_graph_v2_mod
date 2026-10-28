"""Stage 3: Confidence C [n,L] and Compatibility M [n,L] per modality (fixed after this stage).

M is computed on the FULL parent sets S_l(i) over labeled items (not per batch),
from L2-normalized frozen original features, with the Stage 4 contrastive ℓ.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from hierarchy.adaptive_distillation import contrastive_from_scores, normalize_loss
from hierarchy.confidence import graph_teacher_confidence


@torch.no_grad()
def full_compatibility(x, labels, temperature, chunk_size, device):
    """M [n,L] and validity [n,L] (|P|>=1, |N|>=1 within the full parent set)."""
    n, levels = labels.shape
    rows = np.flatnonzero(labels[:, 0] >= 0)
    z = F.normalize(torch.as_tensor(x[rows], device=device), dim=-1)
    y = torch.as_tensor(labels[rows], device=device)
    compat = np.zeros((n, levels), dtype=np.float32)
    valid = np.zeros((n, levels), dtype=bool)
    for start in range(0, len(rows), chunk_size):
        index = torch.arange(start, min(start+chunk_size, len(rows)), device=device)
        scores = z[index] @ z.T
        not_self = torch.ones_like(scores, dtype=torch.bool)
        not_self[torch.arange(len(index)), index] = False
        for level in range(levels):
            parent = not_self if level == 0 else not_self & (y[index, level-1, None] == y[None, :, level-1])
            positive = parent & (y[index, level, None] == y[None, :, level])
            ok = positive.any(-1) & (parent & ~positive).any(-1)
            loss = contrastive_from_scores(scores, parent, positive, temperature)
            m = (1 - normalize_loss(loss, parent, positive, ok)).clamp(0, 1).masked_fill(~ok, 0)
            compat[rows[start:start+len(index)], level] = m.cpu().numpy()
            valid[rows[start:start+len(index)], level] = ok.cpu().numpy()
    return compat, valid


def run(data, graph, pseudo, cfg, out: Path, device):
    out.mkdir(parents=True, exist_ok=True)
    labels = pseudo["labels"]
    conf = graph_teacher_confidence(graph, pseudo["h"], labels, tau_s=cfg.tau_s, eps=cfg.eps)
    m_t, valid = full_compatibility(data.text, labels, cfg.compat_temperature, cfg.chunk_size, device)
    m_v, _ = full_compatibility(data.image, labels, cfg.compat_temperature, cfg.chunk_size, device)
    np.savez_compressed(out / "gates.npz", C=conf["confidence"], A=conf["assignment"], T=conf["topology"],
                        S=conf["support"], neighbor_count=conf["neighbor_count"], M_T=m_t, M_V=m_v, valid=valid)
    labeled = labels[:, 0] >= 0
    stats = []
    for level in range(labels.shape[1]):
        v = valid[:, level]
        row = dict(level=level+1, full_set_valid=int(v.sum()))
        for key, value in (("A", conf["assignment"]), ("T", conf["topology"]), ("S", conf["support"]),
                           ("C", conf["confidence"])):
            row[key+"_mean"] = float(value[labeled, level].mean())
        row["C_zero_fraction"] = float((conf["confidence"][labeled, level] == 0).mean())
        for key, value in (("M_T", m_t), ("M_V", m_v)):
            row[key+"_mean"] = float(value[v, level].mean()) if v.any() else None
            row[key+"_zero_fraction"] = float((value[v, level] == 0).mean()) if v.any() else None
        stats.append(row)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    for row in stats:
        print(f"[stage3] {json.dumps(row)}", flush=True)
    return stats
