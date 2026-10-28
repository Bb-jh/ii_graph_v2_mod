"""Stage 5: item SIDs = [T 3 | G 3 | V 3 | dedup] (TIGER-style collision index).

Vocabulary is separated per branch AND level (e.g. <T1_17> != <T2_17> != <G1_17>).
Cold items have no train edges in W, so the full-graph GNN already sees them isolated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .stage4_tokenizer import build_model, encode_codes, input_dims, prepare_inputs

PAD, EOS, TASK_REC, TASK_T2V, TASK_V2T = 0, 1, 2, 3, 4
SPECIAL = 5


def assign_dedup(codes):
    """Index within each identical code tuple, in item-ID order."""
    _, inverse = np.unique(codes, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    dedup = np.zeros(len(codes), dtype=np.int64)
    seen = {}
    for item, group in enumerate(inverse):
        dedup[item] = seen.get(group, 0)
        seen[group] = dedup[item] + 1
    return dedup


def build_tokens(codes, order, codebook_size, dedup):
    """Map codes to token IDs. Returns tokens [n, 3*len(order)+1] and vocab metadata."""
    offsets, offset, parts = {}, SPECIAL, []
    for m in order:
        for level in range(codes[m].shape[1]):
            offsets[f"{m}{level+1}"] = offset
            parts.append(codes[m][:, level] + offset)
            offset += codebook_size
    offsets["D"] = offset
    parts.append(dedup + offset)
    vocab_size = offset + int(dedup.max()) + 1
    return np.stack(parts, 1).astype(np.int64), dict(offsets=offsets, vocab_size=vocab_size,
                                                     codebook_size=codebook_size, order=list(order))


def run(data, graph, cfg, tokenizer_dir: Path, out: Path, device):
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(tokenizer_dir / "tokenizer.pt", map_location=device)
    x, _, _, full_adj = prepare_inputs(data, graph, device)
    model = build_model(input_dims(x), cfg.tokenizer, checkpoint["branches"]).to(device)
    model.load_state_dict(checkpoint["state"])
    codes = encode_codes(model, x, full_adj, data.n_items)
    order = tuple(cfg.sid.order)
    joint = np.concatenate([codes[m] for m in order], 1)
    dedup = assign_dedup(joint)
    tokens, vocab = build_tokens(codes, order, cfg.tokenizer.codebook_size, dedup)
    if len(np.unique(tokens, axis=0)) != len(tokens):
        raise AssertionError("SIDs are not unique after dedup")
    np.savez_compressed(out / "sid.npz", tokens=tokens, dedup=dedup, **{f"codes_{m}": codes[m] for m in codes})
    stats = dict(items=len(tokens), tokens_per_item=tokens.shape[1], vocab_size=vocab["vocab_size"],
                 collided_items=int((dedup > 0).sum()), collision_rate=float((dedup > 0).mean()),
                 max_dedup=int(dedup.max()), unique_before_dedup=int(len(np.unique(joint, axis=0))),
                 cold_items=int((~data.train_items).sum()))
    (out / "vocab.json").write_text(json.dumps(vocab, indent=2))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[stage5] {json.dumps(stats)}", flush=True)
    return stats
