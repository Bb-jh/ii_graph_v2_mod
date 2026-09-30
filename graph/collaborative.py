"""Interaction -> behavior evidence -> sparse Collaborative I-I graph."""
from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import Optional

import networkx as nx

from config import CollaborativeGraphConfig
from data_loader import Interaction
from evidence import EdgeEvidenceStore, canonical_edge


def compute_behavior_evidence(
    interactions: list[Interaction], cfg: CollaborativeGraphConfig, evidence: EdgeEvidenceStore
) -> dict[tuple[str, str], float]:
    user_items: dict[str, set[str]] = defaultdict(set)
    item_users: dict[str, set[str]] = defaultdict(set)
    for x in interactions:
        user_items[x.user_id].add(x.item_id)
        item_users[x.item_id].add(x.user_id)

    cooccurrence: dict[tuple[str, str], int] = defaultdict(int)
    for items in user_items.values():
        for a, b in combinations(sorted(items), 2):
            cooccurrence[(a, b)] += 1

    n_users = max(len(user_items), 1)
    scores: dict[tuple[str, str], float] = {}
    for (a, b), count in cooccurrence.items():
        if count < cfg.min_cooccurrence:
            continue
        freq_a = len(item_users[a])
        freq_b = len(item_users[b])
        score = _behavior_score(count, freq_a, freq_b, n_users, cfg.weighting)
        key = canonical_edge(a, b)
        scores[key] = score
        evidence.update(
            a, b,
            cooccurrence_count=count,
            item_user_count_i=freq_a if key[0] == a else freq_b,
            item_user_count_j=freq_b if key[1] == b else freq_a,
            behavior_score=score,
        )
    return scores


def build_collaborative_graph(
    item_ids: list[str], scores: dict[tuple[str, str], float], cfg: CollaborativeGraphConfig
) -> nx.Graph:
    selected = _select_topk(scores, cfg.top_k, cfg.symmetrize)
    g = nx.Graph()
    g.add_nodes_from(item_ids)
    for (u, v), score in selected.items():
        g.add_edge(u, v, weight=float(score), behavior_score=float(score))
    return g


def _behavior_score(count: int, freq_a: int, freq_b: int, n_users: int, weighting: str) -> float:
    if weighting == "count":
        return float(count)
    if weighting == "degree_aware":
        return float(count / math.sqrt(max(freq_a * freq_b, 1)))
    if weighting == "pmi":
        pmi = math.log((count * n_users) / max(freq_a * freq_b, 1))
        return float(max(pmi, 0.0))
    raise ValueError(f"Unknown behavior weighting: {weighting}")


def _select_topk(
    scores: dict[tuple[str, str], float], top_k: Optional[int], symmetrize: str
) -> dict[tuple[str, str], float]:
    if top_k is None:
        return scores
    per_node: dict[str, list[tuple[float, tuple[str, str]]]] = defaultdict(list)
    for edge, score in scores.items():
        u, v = edge
        per_node[u].append((score, edge))
        per_node[v].append((score, edge))
    chosen_by: dict[tuple[str, str], int] = defaultdict(int)
    for node, entries in per_node.items():
        del node
        for _, edge in sorted(entries, key=lambda x: x[0], reverse=True)[:top_k]:
            chosen_by[edge] += 1
    if symmetrize == "union":
        keep = {e for e, c in chosen_by.items() if c >= 1}
    elif symmetrize == "mutual":
        keep = {e for e, c in chosen_by.items() if c >= 2}
    else:
        raise ValueError(f"Unknown symmetrize: {symmetrize}")
    return {e: scores[e] for e in keep}
