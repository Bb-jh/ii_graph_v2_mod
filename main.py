"""Run the modular I-I graph experimentation pipeline."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np

from analysis.case_study import run_case_studies
from analysis.statistics import save_analysis
from config import PipelineConfig
from data_loader import build_item_universe, load_category, load_item_metadata, restrict_interactions
from embeddings import get_embedder
from embeddings.precomputed import load_precomputed
from evidence import EdgeEvidenceStore
from graph.collaborative import build_collaborative_graph, compute_behavior_evidence
from graph.denoise import apply_cross_modal_consistency
from graph.fusion import fuse_graphs
from graph.multimodal_reweight import build_multimodal_behavior_graph
from graph.semantic import build_semantic_graph, score_candidate_edges
from hierarchy.community import detect_communities


def _save_graph(g: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gpickle(g, path) if hasattr(nx, "write_gpickle") else pickle.dump(g, path.open("wb"))


def _load_or_build_embeddings(cfg: PipelineConfig, category: str, item_ids: list[str], meta: dict):
    cache_dir = cfg.data.cache_dir / category
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"embeddings_{cfg.embedding.backend}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        if payload.get("item_ids") == item_ids:
            return payload["text"], payload["image"]

    if cfg.embedding.backend == "precomputed":
        if cfg.embedding.precomputed_path is None:
            raise ValueError("EmbeddingConfig.precomputed_path is required for backend='precomputed'")
        text, image = load_precomputed(cfg.embedding.precomputed_path, item_ids)
    else:
        embedder = get_embedder(cfg.embedding)
        texts = [f"{meta[i]['title']} {meta[i]['description']}".strip() for i in item_ids]
        image_urls = [meta[i]["image_url"] for i in item_ids]
        text = embedder.embed_texts(texts)
        image = embedder.embed_images(image_urls)

    with cache_path.open("wb") as f:
        pickle.dump({"item_ids": item_ids, "text": text, "image": image}, f)
    return text, image


def run(cfg: PipelineConfig, category: str) -> dict:
    out_dir = cfg.data.output_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)

    interactions = load_category(category, cfg.data)
    meta = load_item_metadata(category, cfg.data)
    item_ids = build_item_universe(interactions, meta)
    interactions = restrict_interactions(interactions, set(item_ids))
    print(f"[{category}] universe={len(item_ids):,}, interactions={len(interactions):,}")

    evidence = EdgeEvidenceStore()

    # 1) Behavior graph: co-occurrence -> scoring -> pruning/top-k
    behavior_scores = compute_behavior_evidence(interactions, cfg.collaborative, evidence)
    collaborative = build_collaborative_graph(item_ids, behavior_scores, cfg.collaborative)
    print(f"[{category}] collaborative: {collaborative.number_of_edges():,} edges")

    # 2) Embeddings (backend can be swapped independently)
    text_emb, image_emb = _load_or_build_embeddings(cfg, category, item_ids, meta)

    # 3) Score Text/Image only on the already-pruned Behavior I-I edges.
    # This is the recommended DA-MRS-inspired path: Behavior chooses the edge
    # topology; multimodal evidence only changes edge strength.
    behavior_candidates = list(collaborative.edges())
    score_candidate_edges(
        behavior_candidates, item_ids, text_emb.values, text_emb.valid_mask, "text", evidence
    )
    score_candidate_edges(
        behavior_candidates, item_ids, image_emb.values, image_emb.valid_mask, "image", evidence
    )

    if cfg.fusion.strategy == "multimodal_behavior":
        multimodal_behavior = build_multimodal_behavior_graph(
            item_ids,
            collaborative,
            evidence,
            lambda_multimodal=cfg.fusion.lambda_multimodal,
            reliability_mode=cfg.fusion.reliability_mode,
            cosine_transform=cfg.fusion.cosine_transform,
            behavior_normalization=cfg.fusion.behavior_normalization,
            missing_modality=cfg.fusion.missing_modality,
        )
        graphs = {
            "collaborative": collaborative,
            "multimodal_behavior": multimodal_behavior,
        }
        print(
            f"[{category}] multimodal behavior: "
            f"{multimodal_behavior.number_of_edges():,} edges "
            "(topology frozen)"
        )
    else:
        # Legacy comparison path: construct full Text/Image KNN graphs.
        # This is intentionally skipped in multimodal_behavior mode, avoiding
        # unnecessary O(N^2) semantic KNN work.
        text_graph = build_semantic_graph(
            item_ids, text_emb.values, text_emb.valid_mask, cfg.semantic, "text", evidence
        )
        image_graph = build_semantic_graph(
            item_ids, image_emb.values, image_emb.valid_mask, cfg.semantic, "image", evidence
        )
        semantic = apply_cross_modal_consistency(
            {"text": text_graph, "image": image_graph}, cfg.denoise
        )

        fused = fuse_graphs(item_ids, collaborative, semantic, evidence, cfg.fusion)
        if isinstance(fused, dict):
            graphs = fused
        else:
            graphs = {"collaborative": collaborative, **semantic, "fused": fused}

    # 5) Case study / graph diagnostics: this is a first-class output for the current stage.
    if cfg.analysis.enabled:
        save_analysis(graphs, out_dir / "graph_analysis.json")
        run_case_studies(
            graphs, item_ids, cfg.analysis.case_study_items,
            cfg.analysis.top_n_neighbors, out_dir / "case_study.json"
        )
    if cfg.analysis.export_edge_evidence:
        evidence.export_csv(out_dir / "edge_evidence.csv")

    # 6) Community discovery is experimental and OFF by default.
    communities = None
    if cfg.hierarchy.enabled:
        target = graphs.get("multimodal_behavior", graphs.get("fused", graphs.get("collaborative")))
        communities = detect_communities(target, cfg.hierarchy)
        (out_dir / "communities.json").write_text(json.dumps(communities, indent=2), encoding="utf-8")

    for name, g in graphs.items():
        _save_graph(g, out_dir / f"{name}.gpickle")

    return {
        "graphs": graphs,
        "communities": communities,
        "edge_evidence": evidence,
    }


if __name__ == "__main__":
    cfg = PipelineConfig()
    for category in cfg.data.categories:
        run(cfg, category)
