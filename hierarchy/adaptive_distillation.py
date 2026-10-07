"""Mini-batch relation weights; no dependency on a particular RQ model."""
import math

import torch
from torch.nn import functional as F


def hierarchy_candidate_mask(labels):
    """[B,L] globally unique labels -> [B,L,B] parent-restricted mask.

    Excludes self and unlabeled items. Batch must contain unique item rows.
    Do not restrict candidates to the SAME target cluster: siblings are needed.
    """
    if labels.ndim != 2 or labels.shape[1] < 1:
        raise ValueError("Expected labels [batch, levels]")
    b, levels = labels.shape
    valid = (labels >= 0).all(-1)
    base = valid[:, None] & valid[None, :] & ~torch.eye(b, device=labels.device, dtype=torch.bool)
    masks = [base]
    for level in range(1, levels):
        masks.append(base & (labels[:, level-1, None] == labels[None, :, level-1]))
    return torch.stack(masks, dim=1)


def _relation(z, mask, temperature):
    z = F.normalize(z.float(), dim=-1)
    logits = torch.einsum("ild,jld->ilj", z, z) / temperature
    logits = logits.masked_fill(~mask, -torch.inf)
    # Empty candidate sets must not yield NaN softmax values.
    logits = torch.where(mask.any(-1, keepdim=True), logits, torch.zeros_like(logits))
    return logits.softmax(-1).masked_fill(~mask, 0)


def _js(p, q):
    middle = (p+q)*0.5
    tiny = torch.finfo(p.dtype).tiny
    log_middle = middle.clamp_min(tiny).log()
    divergence = 0.5 * (p * (p.clamp_min(tiny).log()-log_middle)
                        + q * (q.clamp_min(tiny).log()-log_middle)).sum(-1)
    return (divergence/math.log(2)).clamp(0, 1)


@torch.no_grad()
def adaptive_weights(graph_z, student_z, original_z, confidence, candidate_mask,
                     *, temperature=0.1, modality_temperature=0.1):
    """Return detached alpha, need, compatibility, usable; all [B,L].

    graph_z [B,L,Dg], student_z [B,L,Ds], original_z [B,Dm], confidence [B,L].
    All branches use the same [B,L,B] boolean mask and identical item ordering.
    Call separately for Text and Image. Teacher/frozen features never receive
    gradients. Fewer than two candidates provides no relational comparison.
    Memory scales as O(B^2 L); use moderate batch sizes.
    """
    if graph_z.ndim != 3 or student_z.ndim != 3 or original_z.ndim != 2:
        raise ValueError("Expected graph/student [B,L,D] and original [B,D]")
    b, levels, _ = graph_z.shape
    if (student_z.shape[:2] != (b, levels) or original_z.shape[0] != b
            or confidence.shape != (b, levels) or candidate_mask.shape != (b, levels, b)
            or candidate_mask.dtype != torch.bool or b < 1 or levels < 1):
        raise ValueError("Inconsistent batch, level, confidence or candidate-mask dimensions")
    if not all(math.isfinite(t) and t > 0 for t in (temperature, modality_temperature)):
        raise ValueError("Temperatures must be finite and positive")
    for z in (graph_z, student_z, original_z, confidence):
        if not torch.isfinite(z).all():
            raise ValueError("Inputs must be finite")
    if ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Confidence must be in [0,1]")
    mask = candidate_mask & ~torch.eye(b, device=graph_z.device, dtype=torch.bool)[:, None, :]
    teacher = _relation(graph_z, mask, temperature)
    student = _relation(student_z, mask, temperature)
    original = _relation(original_z[:, None, :].expand(-1, levels, -1), mask, modality_temperature)
    usable = mask.sum(-1) >= 2
    need = _js(teacher, student).masked_fill(~usable, 0)
    compatibility = (1-_js(teacher, original)).masked_fill(~usable, 0)
    return dict(alpha=(confidence*need*compatibility).detach(), need=need,
                compatibility=compatibility, usable=usable)


def weighted_guidance_loss(per_item_level_loss, alpha, eligible):
    """Reduce an unreduced [B,L] guidance loss, retaining absolute gate strength.

    eligible is a bool [B,L] mask for valid guidance targets/positive sets.
    Divide by eligible count, NOT sum(alpha), so low confidence weakens guidance.
    The caller must compute finite losses, including on ineligible entries.
    """
    if per_item_level_loss.shape != alpha.shape or eligible.shape != alpha.shape or eligible.dtype != torch.bool:
        raise ValueError("Expected matching [B,L] loss, alpha and boolean eligibility")
    weighted = per_item_level_loss * alpha.detach()
    return weighted.masked_fill(~eligible, 0).sum()/eligible.sum().clamp_min(1)
