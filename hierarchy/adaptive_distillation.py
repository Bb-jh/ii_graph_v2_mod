"""Level-conditional hierarchy contrastive loss with adaptive weights.

No dependency on a particular RQ model. Level index 0..L-1 corresponds to
Global/Local/Fine (l = 1..3); the final collision level is never passed here.
Branches: "T"/"V" (Text/Image, weight C*M*R*N) and "G" (Graph, weight C*R).
"""
import math

import torch
from torch.nn import functional as F


def hierarchy_masks(labels):
    """[B,L] composite labels -> candidate S [B,L,B], positive P [B,L,B], valid V [B,L].

    S_l(i): labeled j != i sharing the level l-1 label (all labeled j at level 0).
    P_l(i): j in S_l(i) sharing the level l label. V: |P| >= 1 and |N| >= 1.
    Labels must be globally unique per level (pseudo_labels.npz "labels").
    Batch rows must be unique items.
    """
    if labels.ndim != 2 or labels.shape[1] < 1:
        raise ValueError("Expected labels [batch, levels]")
    b, levels = labels.shape
    labeled = (labels >= 0).all(-1)
    base = labeled[:, None] & labeled[None, :] & ~torch.eye(b, device=labels.device, dtype=torch.bool)
    same = labels[:, None, :] == labels[None, :, :]  # [B,B,L]
    candidate = torch.stack([base] + [base & same[..., level-1] for level in range(1, levels)], dim=1)
    positive = candidate & same.permute(0, 2, 1)
    valid = positive.any(-1) & (candidate & ~positive).any(-1)
    return candidate, positive, valid


def st_softmax_blocks(z, codebooks, *, tau_q=1.0, codes=None):
    """Residual ST-softmax blocks ē [B,L,D] and hard codes [B,L].

    r_{l-1} = z - sum_{k<l} sg[e_{k,c_k}];  q = softmax(-||r - e||^2 / tau_q);
    ē = normalize(e_ĉ + sum q e - sg[sum q e]). Forward uses the hard code,
    backward also flows through q. Pass the RQ's own codes [B,>=L] when it does
    not select by nearest codeword (e.g. balanced assignment).
    """
    if z.ndim != 2 or not codebooks or any(e.ndim != 2 or e.shape[1] != z.shape[1] for e in codebooks):
        raise ValueError("Expected z [B,D] and codebooks [K_l,D]")
    if not (math.isfinite(tau_q) and tau_q > 0):
        raise ValueError("tau_q must be finite and positive")
    residual, blocks, chosen = z, [], []
    for level, codebook in enumerate(codebooks):
        distance = (residual[:, None, :] - codebook[None, :, :]).square().sum(-1)
        q = (-distance / tau_q).softmax(-1)
        c = distance.argmin(-1) if codes is None else codes[:, level]
        hard = codebook[c]
        soft = q @ codebook
        blocks.append(F.normalize(hard + soft - soft.detach(), dim=-1))
        chosen.append(c)
        residual = residual - hard.detach()
    return torch.stack(blocks, 1), torch.stack(chosen, 1)


def contrastive_from_scores(scores, candidate, positive, temperature):
    """Unreduced level-conditional SupCon over the last axis; 0 where |P| or |S| is empty.

    ℓ = -1/|P| sum_{p in P} log( exp(s_ip/τ) / sum_{n in S} exp(s_in/τ) ).
    """
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError("temperature must be finite and positive")
    positive = positive & candidate
    logits = (scores.float() / temperature).masked_fill(~candidate, -torch.inf)
    has = candidate.any(-1, keepdim=True)
    lse = torch.where(has, logits, torch.zeros_like(logits)).logsumexp(-1, keepdim=True)
    log_prob = torch.where(positive, logits - lse, torch.zeros_like(logits))
    count = positive.sum(-1)
    return -(log_prob.sum(-1) / count.clamp_min(1)).masked_fill(count == 0, 0)


def level_contrastive_loss(blocks, candidate, positive, *, temperature=0.1):
    """ē [B,L,D] (already normalized) -> unreduced ℓ [B,L]."""
    scores = torch.einsum("ild,jld->ilj", blocks, blocks)
    return contrastive_from_scores(scores, candidate, positive, temperature)


def normalize_loss(loss, candidate, positive, valid):
    """ℓ̂ = (ℓ - log|P|) / (log|S| - log|P|): 0 = perfect separation, 1 = random. 0 outside V."""
    p = positive.sum(-1).clamp_min(1).float().log()
    s = candidate.sum(-1).clamp_min(1).float().log()
    span = (s - p).clamp_min(1e-12)
    return ((loss.detach() - p) / span).masked_fill(~valid, 0)


@torch.no_grad()
def compatibility(original_z, candidate, positive, valid, *, temperature=0.1):
    """M = clip(1 - ℓ̂^orig, 0, 1) from frozen original modality features [B,D] -> [B,L]."""
    z = F.normalize(original_z.float(), dim=-1)
    scores = (z @ z.T)[:, None, :].expand(-1, candidate.shape[1], -1)
    loss = contrastive_from_scores(scores, candidate, positive, temperature)
    return (1 - normalize_loss(loss, candidate, positive, valid)).clamp(0, 1).masked_fill(~valid, 0)


class LossEMA:
    """Per-item EMA of sg[ℓ] (for Need) and sg[ℓ̂] (for Readiness), per branch and level."""

    def __init__(self, num_items, branches=("T", "V", "G"), levels=3, *, momentum=0.9, device=None):
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0,1)")
        self.branches = {name: index for index, name in enumerate(branches)}
        self.momentum = momentum
        shape = (num_items, len(branches), levels)
        self.loss = torch.zeros(shape, device=device)
        self.normalized = torch.zeros(shape, device=device)
        self.seen = torch.zeros(shape, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, branch, item_ids, loss, normalized, valid):
        """Update only entries in V; the first observation initializes the EMA."""
        if len(torch.unique(item_ids)) != len(item_ids):
            raise ValueError("item_ids must be unique within a batch")
        m = self.branches[branch]
        for state, value in ((self.loss, loss), (self.normalized, normalized)):
            old = state[item_ids, m]
            mixed = torch.where(self.seen[item_ids, m], self.momentum*old + (1-self.momentum)*value.detach(), value.detach())
            state[item_ids, m] = torch.where(valid, mixed, old)
        self.seen[item_ids, m] |= valid

    @torch.no_grad()
    def need(self, branch, item_ids, valid):
        """N̂ = EMA_i(ℓ) / mean_{V_l} EMA(ℓ), per level over the current batch; 0 outside V."""
        ema = self.loss[item_ids, self.branches[branch]].masked_fill(~valid, 0)
        mean = ema.sum(0) / valid.sum(0).clamp_min(1)
        return (ema / mean.clamp_min(1e-12)).masked_fill(~valid, 0)

    @torch.no_grad()
    def readiness(self, branch, item_ids):
        """R_l = clip(1 - ℓ̂_EMA(l-1), 0, 1), R_1 = 1. A parent level never observed gives 0."""
        m = self.branches[branch]
        parent = (1 - self.normalized[item_ids, m, :-1]).clamp(0, 1).masked_fill(~self.seen[item_ids, m, :-1], 0)
        return torch.cat([torch.ones_like(parent[:, :1]), parent], dim=1)


@torch.no_grad()
def adaptive_weights(tracker, item_ids, losses, normalized, confidence, compat, valid,
                     *, components=("C", "M", "R", "N")):
    """Update the EMA and return detached weights {branch: [B,L]}.

    losses/normalized: {"T","V","G"} -> [B,L]; confidence C [B,L] in [0,1];
    compat: {"T","V"} -> M [B,L]. Text/Image: C*M*R*N̂. Graph: C*R.
    A component left out of `components` is replaced by 1 (ablation).
    """
    if ((confidence < 0) | (confidence > 1)).any() or not torch.isfinite(confidence).all():
        raise ValueError("Confidence must be finite and in [0,1]")
    for branch, loss in losses.items():
        tracker.update(branch, item_ids, loss, normalized[branch], valid)
    weights = {}
    for branch in losses:
        w = torch.ones_like(confidence)
        if "C" in components:
            w = w * confidence
        if "R" in components:
            w = w * tracker.readiness(branch, item_ids)
        if branch != "G":
            if "M" in components:
                w = w * compat[branch]
            if "N" in components:
                w = w * tracker.need(branch, item_ids, valid)
        weights[branch] = w.masked_fill(~valid, 0).detach()
    return weights


def fixed_weights(losses, valid, levels):
    """Fixed schedule baseline: w = 1 on the given 1-based levels (e.g. (2, 3)), else 0."""
    on = torch.zeros(valid.shape[1], dtype=torch.bool, device=valid.device)
    on[[level-1 for level in levels]] = True
    return {branch: (valid & on).float() for branch in losses}


def weighted_guidance_loss(losses, weights, valid):
    """sum_l 1/|V_l| sum_{i in V_l} sum_m w_{i,m,l} ℓ_{i,m,l}.

    Divide by |V_l|, NOT sum(w): low C/M/R weakens guidance in absolute terms.
    """
    total = 0
    count = valid.sum(0).clamp_min(1)
    for branch, loss in losses.items():
        if loss.shape != valid.shape or weights[branch].shape != valid.shape:
            raise ValueError("Expected matching [B,L] losses, weights and validity")
        total = total + ((loss * weights[branch].detach()).masked_fill(~valid, 0).sum(0) / count).sum()
    return total


def guidance_lambda(epoch, *, start, ramp, maximum=1.0):
    """λ(e): 0 before `start`, linear warm-up over `ramp` epochs, then `maximum`."""
    if ramp <= 0:
        return maximum if epoch >= start else 0.0
    return maximum * min(max((epoch - start) / ramp, 0.0), 1.0)


def hierarchy_guidance(tracker, item_ids, labels, blocks, confidence, compat, *, temperature=0.1,
                       components=("C", "M", "R", "N"), fixed_levels=None):
    """One training step of the hierarchy term (before λ(e)).

    blocks: {"T","V","G"} -> ST-softmax ē [B,L,D] (see st_softmax_blocks).
    compat: {"T","V"} -> M [B,L], precomputed on the full parent sets (Stage 3).
    fixed_levels replaces adaptive weights by the fixed schedule baseline.
    Returns (loss, info) where info holds per-branch ℓ, ℓ̂, w and V.
    """
    candidate, positive, valid = hierarchy_masks(labels)
    losses = {m: level_contrastive_loss(e, candidate, positive, temperature=temperature) for m, e in blocks.items()}
    normalized = {m: normalize_loss(loss, candidate, positive, valid) for m, loss in losses.items()}
    if fixed_levels is None:
        weights = adaptive_weights(tracker, item_ids, losses, normalized, confidence, compat, valid,
                                   components=components)
    else:
        weights = fixed_weights(losses, valid, fixed_levels)
    loss = weighted_guidance_loss(losses, weights, valid)
    return loss, dict(loss=losses, normalized=normalized, weights=weights, valid=valid)
